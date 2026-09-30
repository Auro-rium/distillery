"""Student serving path 1: generate on CPU inside a Nebius Sandbox (ConTree).

UNVERIFIED against the real platform (spike ``spikes/s4_student_cpu.py`` has not been run):
* whether the sandbox has network egress for ``pip install`` and the Hugging Face download,
* whether the CPU/RAM limits fit Qwen3-1.7B in float32 (~7 GB) and how many seconds per sample,
* the adapter file names (we upload whatever files training downloaded, by basename),
* whether ``files`` keys without a leading slash land at ``/<key>`` as the docs say.

One instance serves one model: ``adapter_files=()`` serves the plain base model (this is the
``base_factory`` path), otherwise the LoRA adapter is applied with ``peft``. The image is built
once, lazily, by a non-disposable ``branch`` (pip install + base weights + adapter + script),
then batches run as disposable jobs with bounded concurrency; the batch travels on stdin.
"""

from __future__ import annotations

import json
import shlex
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from distillery.finetune import TrainedArtifact
from distillery.sandbox import Job, Sandbox
from distillery.sandbox_executor import AsyncBridge
from distillery.student import ChatMessages, StudentServingError

OUT_MARKER = "DISTILLERY_OUT:"
BASE_DIR = "/models/base"
ADAPTER_DIR = "/work/adapter"
SCRIPT_PATH = "/work/gen.py"

GEN_SCRIPT = """\
import argparse
import json
import sys
import time

ap = argparse.ArgumentParser()
ap.add_argument("--base", required=True)
ap.add_argument("--adapter")
ap.add_argument("--max-new-tokens", type=int, default=256)
args = ap.parse_args()
batch = json.load(sys.stdin)["messages_batch"]

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

t0 = time.time()
tok = AutoTokenizer.from_pretrained(args.base)
tok.padding_side = "left"
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
model = AutoModelForCausalLM.from_pretrained(args.base, torch_dtype=torch.float32)
if args.adapter:
    from peft import PeftModel

    model = PeftModel.from_pretrained(model, args.adapter)
model.eval()
load_s = time.time() - t0

t1 = time.time()
prompts = [
    tok.apply_chat_template(m, tokenize=False, add_generation_prompt=True) for m in batch
]
enc = tok(prompts, return_tensors="pt", padding=True)
with torch.no_grad():
    out = model.generate(**enc, max_new_tokens=args.max_new_tokens, do_sample=False)
texts = tok.batch_decode(out[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
gen_s = time.time() - t1
res = {"texts": texts, "load_s": load_s, "gen_s": gen_s, "n": len(batch)}
print("DISTILLERY_OUT:" + json.dumps(res))
"""


def setup_shell(base_model: str) -> str:
    """Shell that prepares the image. CPU-only torch wheels; base weights baked into the image."""
    dl = (
        "from huggingface_hub import snapshot_download; "
        f"snapshot_download({json.dumps(base_model)}, local_dir={json.dumps(BASE_DIR)})"
    )
    return (
        "pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu"
        " && pip install --no-cache-dir transformers peft accelerate huggingface_hub"
        f" && python -c {shlex.quote(dl)}"
    )


def parse_output(stdout: str) -> dict[str, object]:
    for line in reversed(stdout.splitlines()):
        if line.startswith(OUT_MARKER):
            parsed = json.loads(line[len(OUT_MARKER) :])
            if isinstance(parsed, dict):
                return parsed
    raise StudentServingError(f"no {OUT_MARKER} line in generation output")


class SandboxCpuStudent:
    """``StudentServer`` that runs ``transformers`` (+ ``peft``) on sandbox CPU."""

    def __init__(
        self,
        sandbox: Sandbox,
        base_image: str,
        bridge: AsyncBridge,
        *,
        base_model: str,
        adapter_files: Sequence[Path] = (),
        batch_size: int = 8,
        concurrency: int = 4,
        max_new_tokens: int = 256,
        timeout_s: float = 1800.0,
    ) -> None:
        if batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        self._sandbox = sandbox
        self._base_image = base_image
        self._bridge = bridge
        self._base_model = base_model
        self._adapter_files = tuple(Path(p) for p in adapter_files)
        self._batch_size = batch_size
        self._concurrency = concurrency
        self._max_new_tokens = max_new_tokens
        self._timeout_s = timeout_s
        self._image: str | None = None
        self._closed = False
        self.timings: list[dict[str, object]] = []  # per batch: load_s, gen_s, n (measured)

    @classmethod
    def for_artifact(
        cls,
        sandbox: Sandbox,
        base_image: str,
        bridge: AsyncBridge,
        trained: TrainedArtifact,
        *,
        base_model: str,
        **kw: Any,
    ) -> SandboxCpuStudent:
        return cls(
            sandbox, base_image, bridge, base_model=base_model,
            adapter_files=[f.path for f in trained.files], **kw,
        )  # fmt: skip

    def _ensure_image(self) -> str:
        if self._image is None:
            files: dict[str, bytes | Path] = {SCRIPT_PATH.lstrip("/"): GEN_SCRIPT.encode()}
            for p in self._adapter_files:
                files[f"{ADAPTER_DIR.lstrip('/')}/{p.name}"] = p
            self._image = self._bridge.run(
                self._sandbox.branch(
                    self._base_image, setup_shell(self._base_model), files=files,
                    timeout=self._timeout_s,
                )
            )  # fmt: skip
        return self._image

    def _command(self) -> str:
        cmd = f"python {SCRIPT_PATH} --base {BASE_DIR} --max-new-tokens {self._max_new_tokens}"
        return cmd + (f" --adapter {ADAPTER_DIR}" if self._adapter_files else "")

    def generate(self, messages_batch: Sequence[ChatMessages]) -> list[str]:
        if self._closed:
            raise RuntimeError("SandboxCpuStudent is closed")
        image = self._ensure_image()
        chunks = [
            list(messages_batch[i : i + self._batch_size])
            for i in range(0, len(messages_batch), self._batch_size)
        ]
        jobs = [
            Job(
                shell=self._command(),
                stdin=json.dumps({"messages_batch": chunk}),
                timeout=self._timeout_s,
            )
            for chunk in chunks
        ]
        results = self._bridge.run(
            self._sandbox.run_batch(image, jobs, concurrency=self._concurrency)
        )
        outputs: list[str] = []
        for chunk, res in zip(chunks, results, strict=True):
            if not res.ok:
                raise StudentServingError(
                    f"generation batch failed (exit={res.exit_code}, timed_out={res.timed_out}, "
                    f"error={res.error!r}): {res.stderr[-500:]!r}"
                )
            parsed = parse_output(res.stdout)
            texts = parsed.get("texts")
            if not isinstance(texts, list) or len(texts) != len(chunk):
                raise StudentServingError("generation batch returned the wrong number of outputs")
            outputs.extend(str(t) for t in texts)
            self.timings.append({k: parsed.get(k) for k in ("load_s", "gen_s", "n")})
        return outputs

    def close(self) -> None:
        self._closed = True
