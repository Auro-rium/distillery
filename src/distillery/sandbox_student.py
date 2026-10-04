"""Student serving path 1: generate on CPU inside a Nebius Sandbox (ConTree).

Measured live (2026-09-30, see DECISIONS.md) for Qwen3-0.6B in bf16: ``pip install`` of the
CPU torch wheels + transformers/peft takes ~63 s, the weights download inside the sandbox ~20 s,
load ~1.3 s, peak RSS ~2.1 GB (a sandbox has 4 CPUs and ~4 GB), ~7.8 s per sample at ~900 prompt
tokens and ``max_new_tokens=160``. Qwen3-1.7B does NOT fit reliably; do not use it here.

Token Factory LoRA checkpoints (verified live 2026-09-30): ``adapter_config.json`` (peft 0.19.1
format, ``base_model_name_or_path`` = the hub id, ``init_lora_weights: false``),
``adapter_model.safetensors`` (bf16; tensor keys WITHOUT peft's ``base_model.model.`` prefix), plus
tokenizer/chat-template files. ``PeftModel.from_pretrained`` silently loads nothing from such keys
(``strict=False``) and leaves a random adapter, so the generation script loads it explicitly and
strictly (adds the prefix, fails on any unexpected/missing LoRA key).

Still UNVERIFIED live: many sandboxes running this heavy job at the same time.

One ``SandboxCpuStudent`` serves one model: no adapter = the plain base model (the ``base_factory``
path), otherwise the LoRA adapter is applied with ``peft``. The heavy image (pip + weights +
script) is built ONCE per run by a ``ServingImages`` shared by every server of that run, as a
non-disposable ``branch``; an adapter is a small extra layer on top of it. Generation batches then
run as disposable jobs with bounded concurrency; the batch travels on stdin.
"""

from __future__ import annotations

import hashlib
import json
import shlex
import threading
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from distillery.finetune import TrainedArtifact
from distillery.sandbox import Job, RunResult, Sandbox
from distillery.sandbox_executor import AsyncBridge
from distillery.student import ChatMessages, StudentServingError

OUT_MARKER = "DISTILLERY_OUT:"
BASE_DIR = "/models/base"
ADAPTER_DIR = "/work/adapter"
SCRIPT_PATH = "/work/gen.py"
DEFAULT_MAX_NEW_TOKENS = 160  # what the S4 spike measured
GENERATION_JOB_RETRIES = 2  # re-submissions of a generation job that hit a sandbox-level failure
ADAPTER_CONFIG = "adapter_config.json"
ADAPTER_WEIGHTS = ("adapter_model.safetensors", "adapter_model.bin")  # the names peft loads

GEN_SCRIPT = """\
import argparse
import json
import os
import resource
import sys
import time

ap = argparse.ArgumentParser()
ap.add_argument("--base", required=True)
ap.add_argument("--adapter")
ap.add_argument("--max-new-tokens", type=int, default=160)
args = ap.parse_args()
batch = json.load(sys.stdin)["messages_batch"]

import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

torch.set_num_threads(os.cpu_count() or 4)
t0 = time.time()
tok = AutoTokenizer.from_pretrained(args.base)
model = AutoModelForCausalLM.from_pretrained(args.base, torch_dtype=torch.bfloat16)
versions = {"torch": getattr(torch, "__version__", None),
            "transformers": getattr(transformers, "__version__", None)}
adapter_info = None


def load_adapter(model, adapter_dir):
    # Apply a LoRA adapter STRICTLY. PeftModel.from_pretrained loads with strict=False and only
    # warns when no key matches: a Token Factory checkpoint has tensor keys WITHOUT the
    # 'base_model.model.' prefix and 'init_lora_weights': false, so the plain call left a randomly
    # initialised adapter (garbage text; verified live). Add the prefix, then fail on any mismatch.
    from peft import PeftConfig, get_peft_model, set_peft_model_state_dict

    pm = get_peft_model(model, PeftConfig.from_pretrained(adapter_dir))
    safe = os.path.join(adapter_dir, "adapter_model.safetensors")
    if os.path.exists(safe):
        from safetensors.torch import load_file

        state = load_file(safe)
    else:
        state = torch.load(os.path.join(adapter_dir, "adapter_model.bin"), map_location="cpu")
    prefix = "base_model.model."
    state = {(k if k.startswith(prefix) else prefix + k): v for k, v in state.items()}
    res = set_peft_model_state_dict(pm, state)
    unexpected = list(res.unexpected_keys)
    lora_missing = [k for k in res.missing_keys if "lora_" in k]
    if not state or unexpected or lora_missing:
        raise RuntimeError(
            "adapter weights did not load cleanly: %d tensors, %d unexpected keys, %d missing "
            "lora keys; e.g. %s" % (len(state), len(unexpected), len(lora_missing),
                                    (unexpected + lora_missing)[:3])
        )
    # Serve the MERGED model: an unmerged LoRA ran about 2x slower than the base and, with the
    # peft wrapper, kept the sandbox at the edge of its 4 GB (measured live, Qwen3-1.7B). Merging
    # happens only AFTER the strict load above has proven every adapter tensor was applied.
    merged = pm.merge_and_unload()
    return merged, {"tensors": len(state), "merged": True}


if args.adapter:
    import peft

    model, adapter_info = load_adapter(model, args.adapter)
    versions["peft"] = getattr(peft, "__version__", None)
model.eval()
load_s = time.time() - t0

results = []
t1 = time.time()
for messages in batch:
    t = time.time()
    try:
        text = tok.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
        ids = tok(text, return_tensors="pt")
        with torch.no_grad():
            out = model.generate(**ids, max_new_tokens=args.max_new_tokens, do_sample=False)
        new = out[0][ids["input_ids"].shape[1]:]
        results.append({
            "text": tok.decode(new, skip_special_tokens=True), "error": None,
            "prompt_tokens": int(ids["input_ids"].shape[1]), "new_tokens": int(len(new)),
            "seconds": round(time.time() - t, 2),
        })
    except Exception as exc:
        results.append({
            "text": "", "error": type(exc).__name__ + ": " + str(exc)[:300],
            "seconds": round(time.time() - t, 2),
        })
peak_mb = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024)
res = {"results": results, "load_s": round(load_s, 2), "gen_s": round(time.time() - t1, 2),
       "n": len(batch), "peak_rss_mb": peak_mb, "versions": versions,
       "adapter_tensors": adapter_info["tensors"] if adapter_info else None}
print("DISTILLERY_OUT:" + json.dumps(res))
"""


DUMMY_ADAPTER_SCRIPT_PATH = "/work/make_dummy_adapter.py"
DEFAULT_TARGET_MODULES = "all-linear"

# Builds an UNTRAINED LoRA inside the sandbox, shaped like a Token Factory checkpoint (tensor keys
# WITHOUT the 'base_model.model.' prefix, bf16, init_lora_weights false), so the generation script's
# strict adapter loader and peak-RSS measurement run on a realistic adapter. The base model is built
# on the meta device (no weights are loaded: only module shapes are needed), so this step is cheap
# in memory even for a model that barely fits at inference time. Values are irrelevant to memory.
DUMMY_ADAPTER_SCRIPT = """\
import argparse
import json
import os

ap = argparse.ArgumentParser()
ap.add_argument("--base", required=True)
ap.add_argument("--base-id", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--r", type=int, default=16)
ap.add_argument("--alpha", type=int, default=16)
ap.add_argument("--target-modules", default="all-linear")
args = ap.parse_args()

import torch
from accelerate import init_empty_weights
from peft import LoraConfig, get_peft_model
from safetensors.torch import save_file
from transformers import AutoConfig, AutoModelForCausalLM

tm = args.target_modules
target = tm if tm == "all-linear" else [m for m in tm.split(",") if m]
lcfg = LoraConfig(r=args.r, lora_alpha=args.alpha, target_modules=target, lora_dropout=0.0,
                  task_type="CAUSAL_LM")
cfg = AutoConfig.from_pretrained(args.base)
with init_empty_weights():
    model = AutoModelForCausalLM.from_config(cfg, torch_dtype=torch.bfloat16)
    pm = get_peft_model(model, lcfg)
gen = torch.Generator().manual_seed(0)
prefix = "base_model.model."
state = {}
for name, p in pm.named_parameters():
    if "lora_" not in name:
        continue
    key = name.replace(".default.", ".", 1)
    key = key[len(prefix):] if key.startswith(prefix) else key
    if "lora_A" in key:
        t = torch.randn(tuple(p.shape), generator=gen) * 0.01
    else:
        t = torch.zeros(tuple(p.shape))
    state[key] = t.to(torch.bfloat16).contiguous()
os.makedirs(args.out, exist_ok=True)
save_file(state, os.path.join(args.out, "adapter_model.safetensors"), metadata={"format": "pt"})
c = pm.peft_config["default"]
c.base_model_name_or_path = args.base_id
c.init_lora_weights = False
c.save_pretrained(args.out)
tmods = c.target_modules
size = os.path.getsize(os.path.join(args.out, "adapter_model.safetensors"))
print("DISTILLERY_DUMMY_ADAPTER:" + json.dumps({
    "tensors": len(state), "bytes": size, "r": args.r, "alpha": args.alpha,
    "target_modules": sorted(tmods) if isinstance(tmods, (set, list, tuple)) else tmods,
}))
"""


def dummy_adapter_shell(
    base_model: str, *, r: int = 16, alpha: int = 16, target_modules: str = DEFAULT_TARGET_MODULES
) -> str:
    """Shell that runs ``DUMMY_ADAPTER_SCRIPT`` and leaves the adapter in ``ADAPTER_DIR``."""
    return (
        f"python {DUMMY_ADAPTER_SCRIPT_PATH} --base {BASE_DIR} --base-id {shlex.quote(base_model)}"
        f" --out {ADAPTER_DIR} --r {int(r)} --alpha {int(alpha)}"
        f" --target-modules {shlex.quote(target_modules)}"
    )


def setup_shell(base_model: str) -> str:
    """Shell that prepares the heavy image: the measured recipe (CPU-only torch wheels, then the
    HF stack; base weights baked into the image, downloaded inside the sandbox)."""
    dl = (
        "from huggingface_hub import snapshot_download; "
        f"snapshot_download({json.dumps(base_model)}, local_dir={json.dumps(BASE_DIR)})"
    )
    return (
        "pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu"
        " && pip install --no-cache-dir transformers accelerate peft safetensors huggingface_hub"
        f" && HF_HUB_DISABLE_TELEMETRY=1 python -c {shlex.quote(dl)}"
    )


def parse_output(stdout: str) -> dict[str, Any]:
    for line in reversed(stdout.splitlines()):
        if line.startswith(OUT_MARKER):
            parsed = json.loads(line[len(OUT_MARKER) :])
            if isinstance(parsed, dict):
                return parsed
    raise StudentServingError(f"no {OUT_MARKER} line in generation output")


# ---------------------------------------------------------------- adapter files


@dataclass(frozen=True)
class PreparedAdapter:
    """The files of a checkpoint we will upload, under the names ``peft`` loads."""

    files: dict[str, Path]  # destination file name (inside ADAPTER_DIR) -> local path
    sha256: str
    base_model_name_or_path: str


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _pick_weights(paths: Sequence[Path], listing: str) -> tuple[str, Path]:
    by_name = {p.name: p for p in paths}
    for exact in ADAPTER_WEIGHTS:  # exact peft names win over any guess
        if exact in by_name:
            return exact, by_name[exact]
    safes = [p for p in paths if p.suffix == ".safetensors"]
    bins = [p for p in paths if p.suffix == ".bin"]
    if len(safes) == 1:
        return ADAPTER_WEIGHTS[0], safes[0]
    if not safes and len(bins) == 1:
        return ADAPTER_WEIGHTS[1], bins[0]
    if len(safes) > 1 or len(bins) > 1:
        raise StudentServingError(
            f"ambiguous adapter weights (several candidate files); downloaded files: {listing}"
        )
    raise StudentServingError(
        f"no adapter weights among the downloaded files (expected one of {ADAPTER_WEIGHTS} or a "
        f"single .safetensors/.bin file); downloaded files: {listing}"
    )


def prepare_adapter(files: Sequence[Path], base_model: str) -> PreparedAdapter:
    """Validate a downloaded checkpoint before anything is uploaded or billed.

    Requires ``adapter_config.json`` whose ``base_model_name_or_path`` equals the configured
    student, plus one weights file (safetensors or bin), which is renamed to what peft loads.
    Only these two files are shipped (checkpoints may carry optimizer state or tokenizers).
    """
    paths = [Path(p) for p in files]
    listing = ", ".join(sorted(p.name for p in paths)) or "(none)"
    cfg_path = next((p for p in paths if p.name == ADAPTER_CONFIG), None)
    if cfg_path is None:
        raise StudentServingError(f"no {ADAPTER_CONFIG} among the downloaded files: {listing}")
    try:
        cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise StudentServingError(
            f"{ADAPTER_CONFIG} is not valid JSON ({exc}); downloaded files: {listing}"
        ) from exc
    declared = cfg.get("base_model_name_or_path") if isinstance(cfg, dict) else None
    if declared is None:
        raise StudentServingError(
            f"{ADAPTER_CONFIG} has no base_model_name_or_path, cannot confirm the adapter was "
            f"trained for {base_model!r}; downloaded files: {listing}"
        )
    if declared != base_model:
        raise StudentServingError(
            f"adapter base_model_name_or_path is {declared!r} but the configured student is "
            f"{base_model!r}; downloaded files: {listing}"
        )
    weights_name, weights_path = _pick_weights([p for p in paths if p is not cfg_path], listing)
    chosen = {ADAPTER_CONFIG: cfg_path, weights_name: weights_path}
    h = hashlib.sha256()
    for name in sorted(chosen):
        h.update(f"{name}:{_sha256_file(chosen[name])}\n".encode())
    return PreparedAdapter(chosen, h.hexdigest(), str(declared))


# ---------------------------------------------------------------- images


class ServingImages:
    """Builds and caches the sandbox images shared by every ``SandboxCpuStudent`` of one run.

    ``deps_image`` = pip installs + base weights + the generation script (built once, as a
    non-disposable ``branch``); ``adapter_image`` = that plus one adapter (one small layer per
    distinct adapter). ``on_image(kind, uuid)`` is told about every image actually built, so the
    caller can record it. Thread-safe; a failed build is not cached.
    """

    def __init__(
        self,
        sandbox: Sandbox,
        base_image: str,
        bridge: AsyncBridge,
        *,
        base_model: str,
        setup_timeout_s: float = 1800.0,
        on_image: Callable[[str, str], None] | None = None,
    ) -> None:
        self._sandbox = sandbox
        self._base_image = base_image
        self._bridge = bridge
        self._base_model = base_model
        self._timeout_s = setup_timeout_s
        self._on_image = on_image
        self._lock = threading.Lock()
        self._deps: str | None = None
        self._adapters: dict[str, str] = {}

    def deps_image(self) -> str:
        with self._lock:
            if self._deps is None:
                self._deps = self._bridge.run(
                    self._sandbox.branch(
                        self._base_image, setup_shell(self._base_model),
                        files={SCRIPT_PATH: GEN_SCRIPT.encode()}, timeout=self._timeout_s,
                    )
                )  # fmt: skip
                if self._on_image is not None:
                    self._on_image("deps", self._deps)
            return self._deps

    def dummy_adapter_image(
        self, *, r: int = 16, alpha: int = 16, target_modules: str = DEFAULT_TARGET_MODULES
    ) -> str:
        """Deps image plus an UNTRAINED adapter built inside the sandbox (memory pre-flight)."""
        deps = self.deps_image()
        key = f"dummy:{r}:{alpha}:{target_modules}"
        with self._lock:
            uuid = self._adapters.get(key)
            if uuid is None:
                uuid = self._bridge.run(
                    self._sandbox.branch(
                        deps,
                        dummy_adapter_shell(
                            self._base_model, r=r, alpha=alpha, target_modules=target_modules
                        ),
                        files={DUMMY_ADAPTER_SCRIPT_PATH: DUMMY_ADAPTER_SCRIPT.encode()},
                        timeout=self._timeout_s,
                    )
                )
                self._adapters[key] = uuid
                if self._on_image is not None:
                    self._on_image("dummy_adapter", uuid)
            return uuid

    def adapter_image(self, adapter: PreparedAdapter) -> str:
        deps = self.deps_image()
        with self._lock:
            uuid = self._adapters.get(adapter.sha256)
            if uuid is None:
                uuid = self._bridge.run(
                    self._sandbox.branch(
                        deps, "true",
                        files={f"{ADAPTER_DIR}/{name}": p for name, p in adapter.files.items()},
                        timeout=self._timeout_s,
                    )
                )  # fmt: skip
                self._adapters[adapter.sha256] = uuid
                if self._on_image is not None:
                    self._on_image("adapter", uuid)
            return uuid


# ---------------------------------------------------------------- the server


def _transient_job_failure(res: RunResult) -> bool:
    """A job the sandbox itself lost (timed out, or killed with exit -1): safe to run again."""
    return not res.ok and (res.timed_out or res.exit_code == -1)


class SandboxCpuStudent:
    """``StudentServer`` that runs ``transformers`` (+ ``peft``) on sandbox CPU.

    Per-sample failures inside a generation job (the script catches them) yield ``""`` for that
    sample, are counted in ``generation_errors`` and described in ``error_samples``; more than
    ``max_sample_failure_fraction`` of a call's samples failing raises. A failed JOB (non-zero exit,
    timeout, sandbox error) fails all of its samples and raises: infrastructure trouble is loud.
    """

    def __init__(
        self,
        sandbox: Sandbox,
        base_image: str,
        bridge: AsyncBridge,
        *,
        base_model: str,
        adapter_files: Sequence[Path] = (),
        batch_size: int = 2,
        concurrency: int = 10,
        max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
        timeout_s: float = 1800.0,
        job_base_timeout_s: float = 120.0,
        job_per_sample_timeout_s: float = 90.0,  # 1.7B on sandbox CPU: one batch of 2 took 229 s
        max_sample_failure_fraction: float = 0.25,
        images: ServingImages | None = None,
        prebuilt_adapter_image: str | None = None,
    ) -> None:
        if batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        self._sandbox = sandbox
        self._bridge = bridge
        self._base_model = base_model
        # validated here, before any sandbox call: a wrong adapter must not cost anything
        self._adapter_files = tuple(Path(p) for p in adapter_files)
        self._adapter = prepare_adapter(adapter_files, base_model) if adapter_files else None
        # an image that already holds an adapter at ADAPTER_DIR (the dummy-adapter pre-flight)
        self._prebuilt = prebuilt_adapter_image
        if prebuilt_adapter_image is not None and adapter_files:
            raise ValueError("give adapter_files or prebuilt_adapter_image, not both")
        self._batch_size = batch_size
        self._concurrency = concurrency
        self._max_new_tokens = max_new_tokens
        self._timeout_s = timeout_s
        self._job_base_s = job_base_timeout_s
        self._job_per_sample_s = job_per_sample_timeout_s
        self._max_fail = max_sample_failure_fraction
        self._images = images or ServingImages(
            sandbox, base_image, bridge, base_model=base_model, setup_timeout_s=timeout_s
        )
        self._image: str | None = None
        self._closed = False
        self.generation_errors = 0
        self.cap_hits = 0  # samples that ran to max_new_tokens (truncated or runaway)
        self.job_retries = 0  # generation jobs re-submitted after a sandbox-level failure
        self.error_samples: list[str] = []
        self.timings: list[dict[str, object]] = []  # per batch: load_s, gen_s, n, failed, ...

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
        if self._image is None and self._prebuilt is not None:
            self._image = self._prebuilt
        if self._image is None:
            self._image = (
                self._images.adapter_image(self._adapter)
                if self._adapter is not None
                else self._images.deps_image()
            )
        return self._image

    def _command(self) -> str:
        cmd = f"python {SCRIPT_PATH} --base {BASE_DIR} --max-new-tokens {self._max_new_tokens}"
        has_adapter = self._adapter is not None or self._prebuilt is not None
        return cmd + (f" --adapter {ADAPTER_DIR}" if has_adapter else "")

    def _job_timeout(self, n: int) -> float:
        return min(self._timeout_s, self._job_base_s + self._job_per_sample_s * n)

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
                timeout=self._job_timeout(len(chunk)),
            )
            for chunk in chunks
        ]
        results = self._bridge.run(
            self._sandbox.run_batch(image, jobs, concurrency=self._concurrency)
        )
        # A generation job is idempotent and disposable, so a sandbox-level failure (timeout, or
        # exit -1 with no script error: seen live, an operation that sat idle until its timeout)
        # is re-submitted a bounded number of times instead of killing a multi-hour run. A job
        # that really ran and failed (script error, OOM exit code) is never retried here.
        for _ in range(GENERATION_JOB_RETRIES):
            redo = [i for i, r in enumerate(results) if _transient_job_failure(r)]
            if not redo:
                break
            self.job_retries += len(redo)
            # Greedy decoding makes a slow batch slow every time (P1.11: the same 2 batches timed
            # out 3x, then ran fine with more time), so a timed-out job retries with 2x its timeout.
            for i in redo:
                t = jobs[i].timeout
                if results[i].timed_out and t is not None:
                    jobs[i] = replace(jobs[i], timeout=min(self._timeout_s, t * 2))
            again = self._bridge.run(
                self._sandbox.run_batch(
                    image, [jobs[i] for i in redo], concurrency=self._concurrency
                )
            )
            for i, r in zip(redo, again, strict=True):
                results[i] = r
        outputs: list[str] = []
        failed = 0
        for chunk, res in zip(chunks, results, strict=True):
            if not res.ok:
                raise StudentServingError(
                    f"generation batch failed (exit={res.exit_code}, timed_out={res.timed_out}, "
                    f"error={res.error!r}): {res.stderr[-500:]!r}"
                )
            parsed = parse_output(res.stdout)
            items = parsed.get("results")
            if not isinstance(items, list) or len(items) != len(chunk):
                raise StudentServingError("generation batch returned the wrong number of outputs")
            n_failed = n_cap = 0
            for item in items:
                if not isinstance(item, dict):
                    raise StudentServingError("generation batch returned a malformed result")
                if item.get("error"):
                    n_failed += 1
                    self.error_samples.append(str(item["error"])[:300])
                    outputs.append("")
                else:
                    outputs.append(str(item.get("text", "")))
                    n_new = item.get("new_tokens")
                    n_cap += isinstance(n_new, int) and n_new >= self._max_new_tokens
            failed += n_failed
            self.generation_errors += n_failed
            self.cap_hits += n_cap
            self.timings.append(
                {
                    **{
                        k: parsed.get(k)
                        for k in (
                            "load_s",
                            "gen_s",
                            "n",
                            "peak_rss_mb",
                            "versions",
                            "adapter_tensors",
                        )
                    },
                    "failed": n_failed,
                    "cap_hits": n_cap,
                }
            )
        if failed > self._max_fail * len(outputs):
            raise StudentServingError(
                f"{failed} of {len(outputs)} samples failed inside the sandbox "
                f"(cap {self._max_fail:.0%}); first error: {self.error_samples[-failed:][0]!r}"
            )
        return outputs

    def close(self) -> None:
        self._closed = True
