# ruff: noqa: S101, S603
"""SandboxCpuStudent offline: FakeSandbox whose handler runs the REAL generation script locally
against stub torch/transformers/peft packages. Says nothing about real CPU serving (UNVERIFIED)."""

from __future__ import annotations

import subprocess
import sys
from collections.abc import Iterator
from pathlib import Path

import pytest

from distillery.sandbox import FakeExecution, FakeSandbox, Job, SandboxError
from distillery.sandbox_executor import AsyncBridge
from distillery.sandbox_student import SandboxCpuStudent, setup_shell
from distillery.student import StudentServer, StudentServingError

STUBS = {
    "torch.py": """
import contextlib
float32 = "f32"
@contextlib.contextmanager
def no_grad():
    yield
""",
    "peft.py": """
class PeftModel:
    @classmethod
    def from_pretrained(cls, model, path):
        model.tag = "adapter:" + open(path + "/adapter_config.json").read().strip()
        return model
""",
    "transformers.py": """
class T:
    def __init__(self, rows): self.rows = rows
    @property
    def shape(self): return (len(self.rows), len(self.rows[0]))
    def __getitem__(self, key):
        _, cols = key
        return T([r[cols] for r in self.rows])
PROMPTS = []
class Tok:
    padding_side = "right"; pad_token = None; eos_token = "<eos>"
    @classmethod
    def from_pretrained(cls, base): return cls()
    def apply_chat_template(self, m, tokenize, add_generation_prompt):
        return m[-1]["content"]
    def __call__(self, prompts, return_tensors, padding):
        assert self.padding_side == "left"
        ids = []
        for p in prompts:
            PROMPTS.append(p); ids.append([len(PROMPTS) - 1])
        return {"input_ids": T(ids)}
    def batch_decode(self, out, skip_special_tokens):
        return [f"{tag}|{PROMPTS[r[0]]}" for r in out.rows for tag in [MODEL.tag]]
MODEL = None
class Model:
    tag = "base"
    def eval(self): pass
    def generate(self, input_ids, max_new_tokens, do_sample):
        assert do_sample is False
        return T([r + [r[0]] for r in input_ids.rows])
class AutoTokenizer(Tok): pass
class AutoModelForCausalLM:
    @classmethod
    def from_pretrained(cls, base, torch_dtype):
        global MODEL
        MODEL = Model(); return MODEL
""",
}


@pytest.fixture
def stubs(tmp_path: Path) -> Path:
    d = tmp_path / "stubs"
    d.mkdir()
    for name, body in STUBS.items():
        (d / name).write_text(body)
    return d


def make_sandbox(stubs: Path, tmp_path: Path, log: list[str]) -> FakeSandbox:
    def handler(job: Job, fs: dict[str, bytes]) -> FakeExecution:
        shell = job.shell or ""
        if "gen.py" not in shell:
            log.append("setup:" + shell)
            return FakeExecution()
        work = tmp_path / f"w{len(log)}"
        for name, data in fs.items():
            target = work / name.lstrip("/")
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
        cmd = shell.replace("/models/base", "BASE").replace("/work/", f"{work}/work/")
        log.append("run")
        proc = subprocess.run(  # noqa: S603
            [sys.executable, *cmd.split()[1:]],
            input=job.stdin if isinstance(job.stdin, str) else "",
            capture_output=True, text=True, env={"PYTHONPATH": str(stubs)}, check=False,
        )  # fmt: skip
        return FakeExecution(proc.stdout, proc.stderr, proc.returncode, duration_s=0.01)

    return FakeSandbox(handler)


@pytest.fixture
def bridge() -> Iterator[AsyncBridge]:
    with AsyncBridge() as b:
        yield b


def msgs(n: int) -> list[list[dict[str, str]]]:
    return [[{"role": "user", "content": f"q{i}"}] for i in range(n)]


def test_base_model_batches_in_order_and_builds_image_once(
    stubs: Path, tmp_path: Path, bridge: AsyncBridge
) -> None:
    log: list[str] = []
    sb = make_sandbox(stubs, tmp_path, log)
    s: StudentServer = SandboxCpuStudent(
        sb, "python:3.12-slim", bridge, base_model="Qwen/Qwen3-1.7B", batch_size=3, concurrency=2
    )
    out = s.generate(msgs(7))
    assert out == [f"base|q{i}" for i in range(7)]
    assert log.count("run") == 3 and len([x for x in log if x.startswith("setup:")]) == 1
    assert "pip install" in log[0] and "Qwen/Qwen3-1.7B" in log[0] and "cpu" in log[0]
    assert sb.peak_inflight <= 2
    s.generate(msgs(1))
    assert len([x for x in log if x.startswith("setup:")]) == 1  # image reused
    lin = sb.lineage()
    assert lin[-1][1] == "python:3.12-slim"


def test_adapter_files_are_uploaded_and_applied(
    stubs: Path, tmp_path: Path, bridge: AsyncBridge
) -> None:
    cfg = tmp_path / "adapter_config.json"
    cfg.write_text('{"r": 8}\n')
    w = tmp_path / "adapter_model.safetensors"
    w.write_bytes(b"W")
    log: list[str] = []
    s = SandboxCpuStudent(
        make_sandbox(stubs, tmp_path, log), "img", bridge,
        base_model="Qwen/Qwen3-1.7B", adapter_files=[cfg, w],
    )  # fmt: skip
    assert s.generate(msgs(2)) == ['adapter:{"r": 8}|q0', 'adapter:{"r": 8}|q1']
    assert s.timings and s.timings[0]["n"] == 2


def test_failed_batch_is_loud(stubs: Path, tmp_path: Path, bridge: AsyncBridge) -> None:
    def handler(job: Job, fs: dict[str, bytes]) -> FakeExecution:
        if "gen.py" in (job.shell or ""):
            return FakeExecution("", "OSError: no network", 1)
        return FakeExecution()

    s = SandboxCpuStudent(FakeSandbox(handler), "img", bridge, base_model="m")
    with pytest.raises(StudentServingError, match="no network"):
        s.generate(msgs(2))


def test_setup_failure_is_loud_and_closed_student_refuses(
    tmp_path: Path, bridge: AsyncBridge
) -> None:
    sb = FakeSandbox(lambda job, fs: FakeExecution("", "pip: no egress", 1))
    s = SandboxCpuStudent(sb, "img", bridge, base_model="m")
    with pytest.raises(SandboxError, match="branch setup exited 1"):
        s.generate(msgs(1))
    s.close()
    with pytest.raises(RuntimeError, match="closed"):
        s.generate(msgs(1))


def test_setup_shell_quotes_base_model() -> None:
    assert 'snapshot_download("a/b"' in setup_shell("a/b")


def test_script_marker_matches_parser() -> None:
    from distillery.sandbox_student import GEN_SCRIPT, OUT_MARKER

    assert f'print("{OUT_MARKER}"' in GEN_SCRIPT
