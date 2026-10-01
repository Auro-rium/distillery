# ruff: noqa: S101, S603
"""SandboxCpuStudent offline: FakeSandbox whose handler runs the REAL generation script locally
against stub torch/transformers/peft packages. Says nothing about real CPU serving speed; the real
recipe (bf16, enable_thinking=False, ~7.8 s/sample) was measured live (2026-09-30, DECISIONS.md)."""

from __future__ import annotations

import json
import subprocess
import sys
from collections.abc import Iterator
from pathlib import Path

import pytest

from distillery.finetune import DownloadedFile, TrainedArtifact
from distillery.sandbox import FakeExecution, FakeSandbox, Job, SandboxError
from distillery.sandbox_executor import AsyncBridge
from distillery.sandbox_student import (
    ADAPTER_DIR,
    SandboxCpuStudent,
    ServingImages,
    prepare_adapter,
    setup_shell,
)
from distillery.student import StudentServer, StudentServingError

MODEL = "Qwen/Qwen3-0.6B"

STUBS = {
    "torch.py": """
import contextlib
bfloat16 = "bf16"
def set_num_threads(n):
    pass
@contextlib.contextmanager
def no_grad():
    yield
def load(path, map_location):
    import json
    return json.load(open(path))
""",
    # Mirrors what the real peft does with a Token Factory checkpoint (verified live): tensor keys
    # WITHOUT the "base_model.model." prefix are silently ignored by PeftModel.from_pretrained
    # (strict=False) and, with init_lora_weights=False in the config, the adapter stays random.
    "peft.py": """
import json
EXPECTED = ["base_model.model.model.layers.0.self_attn.q_proj.lora_A.default.weight",
            "base_model.model.model.layers.0.self_attn.q_proj.lora_B.default.weight"]
class PeftConfig:
    @classmethod
    def from_pretrained(cls, path):
        c = cls()
        c.path = path
        return c
class Result:
    def __init__(self, missing, unexpected):
        self.missing_keys, self.unexpected_keys = missing, unexpected
def get_peft_model(model, cfg):
    model.cfg_path = cfg.path
    model.tag = "RANDOM-INIT-ADAPTER"
    return model
def set_peft_model_state_dict(model, state):
    got = [k.replace(".weight", ".default.weight") for k in state]
    unexpected = [k for k in got if k not in EXPECTED]
    missing = ["model.embed_tokens.weight"] + [k for k in EXPECTED if k not in got]
    if not unexpected and len(missing) == 1:
        model.tag = "adapter:" + open(model.cfg_path + "/adapter_config.json").read().strip()
    return Result(missing, unexpected)
""",
    "safetensors/__init__.py": "",
    "safetensors/torch.py": """
import json
def load_file(path):
    return json.load(open(path))
""",
    "transformers.py": """
class T(list):
    @property
    def shape(self):
        return (len(self), len(self[0]))
PROMPTS = []
class AutoTokenizer:
    @classmethod
    def from_pretrained(cls, base):
        return cls()
    def apply_chat_template(self, m, tokenize, add_generation_prompt, enable_thinking):
        assert tokenize is False and add_generation_prompt is True
        assert enable_thinking is False  # thinking is disabled exactly like the spike
        return m[-1]["content"]
    def __call__(self, text, return_tensors):
        PROMPTS.append(text)
        return {"input_ids": T([[len(PROMPTS) - 1]]), "attention_mask": T([[1]])}
    def decode(self, ids, skip_special_tokens):
        assert skip_special_tokens is True
        return MODEL.tag + "|" + PROMPTS[ids[0]]
MODEL = None
class Model:
    tag = "base"
    def eval(self):
        pass
    def generate(self, input_ids, attention_mask, max_new_tokens, do_sample):
        assert do_sample is False
        text = PROMPTS[input_ids[0][0]]
        if "boom" in text:
            raise RuntimeError("boom in " + text)
        return T([[input_ids[0][0], input_ids[0][0]]])
class AutoModelForCausalLM:
    @classmethod
    def from_pretrained(cls, base, torch_dtype):
        global MODEL
        assert torch_dtype == "bf16"  # the only dtype measured live
        MODEL = Model()
        return MODEL
""",
}


@pytest.fixture
def stubs(tmp_path: Path) -> Path:
    d = tmp_path / "stubs"
    d.mkdir()
    for name, body in STUBS.items():
        (d / name).parent.mkdir(parents=True, exist_ok=True)
        (d / name).write_text(body)
    return d


class Log:
    def __init__(self) -> None:
        self.entries: list[str] = []
        self.jobs: list[Job] = []

    def count(self, prefix: str) -> int:
        return len([e for e in self.entries if e.startswith(prefix)])


def make_sandbox(stubs: Path, tmp_path: Path, log: Log) -> FakeSandbox:
    def handler(job: Job, fs: dict[str, bytes]) -> FakeExecution:
        shell = job.shell or ""
        if "gen.py" not in shell or "pip install" in shell:
            log.entries.append("setup:" + shell)
            return FakeExecution()
        log.jobs.append(job)
        work = tmp_path / f"w{len(log.entries)}"
        for name, data in fs.items():
            target = work / name.lstrip("/")
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
        cmd = shell.replace("/models/base", "BASE").replace("/work/", f"{work}/work/")
        log.entries.append("run:" + cmd.split("gen.py", 1)[1])
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


TF_KEYS = [  # what Token Factory wrote (verified live): no "base_model.model." prefix
    "model.layers.0.self_attn.q_proj.lora_A.weight",
    "model.layers.0.self_attn.q_proj.lora_B.weight",
]
PEFT_KEYS = ["base_model.model." + k for k in TF_KEYS]


def adapter_files(
    tmp_path: Path,
    *,
    base: str = MODEL,
    weights: str = "adapter_model.safetensors",
    keys: list[str] | None = None,
) -> list[Path]:
    cfg = tmp_path / "adapter_config.json"
    cfg.write_text(json.dumps({"r": 8, "base_model_name_or_path": base}))
    w = tmp_path / weights
    w.write_text(json.dumps(dict.fromkeys(keys or TF_KEYS, 1)))
    return [cfg, w]


# ---- generation --------------------------------------------------------------------------------


def test_base_model_batches_in_order_and_builds_image_once(
    stubs: Path, tmp_path: Path, bridge: AsyncBridge
) -> None:
    log = Log()
    sb = make_sandbox(stubs, tmp_path, log)
    s: StudentServer = SandboxCpuStudent(
        sb, "python:3.12-slim", bridge, base_model=MODEL, batch_size=3, concurrency=2
    )
    out = s.generate(msgs(7))
    assert out == [f"base|q{i}" for i in range(7)]
    assert log.count("run:") == 3 and log.count("setup:") == 1
    assert "pip install" in log.entries[0] and MODEL in log.entries[0] and "cpu" in log.entries[0]
    assert all("--max-new-tokens 160" in e for e in log.entries if e.startswith("run:"))
    assert all("--adapter" not in e for e in log.entries if e.startswith("run:"))
    assert sb.peak_inflight <= 2
    s.generate(msgs(1))
    assert log.count("setup:") == 1  # image reused
    assert sb.lineage()[-1][1] == "python:3.12-slim"


def test_generation_jobs_have_bounded_timeouts(
    stubs: Path, tmp_path: Path, bridge: AsyncBridge
) -> None:
    log = Log()
    s = SandboxCpuStudent(
        make_sandbox(stubs, tmp_path, log), "img", bridge, base_model=MODEL, batch_size=2
    )
    s.generate(msgs(4))
    assert log.jobs and all(j.timeout is not None and j.timeout <= 400 for j in log.jobs)


def test_adapter_files_are_uploaded_as_peft_names_and_applied(
    stubs: Path, tmp_path: Path, bridge: AsyncBridge
) -> None:
    files = adapter_files(tmp_path)
    (tmp_path / "README.md").write_text("not needed")
    files.append(tmp_path / "README.md")
    log = Log()
    sb = make_sandbox(stubs, tmp_path, log)
    s = SandboxCpuStudent(sb, "img", bridge, base_model=MODEL, adapter_files=files)
    cfg_text = files[0].read_text().strip()
    assert s.generate(msgs(2)) == [f"adapter:{cfg_text}|q0", f"adapter:{cfg_text}|q1"]
    assert s.timings and s.timings[0]["n"] == 2 and s.timings[0]["failed"] == 0
    runs = [e for e in log.entries if e.startswith("run:")]  # the stub maps /work/ to a temp dir
    assert runs and all("--adapter" in e and e.endswith(ADAPTER_DIR) for e in runs)
    layer = [c for c in sb.lineage() if c[2] == "true"]  # the adapter layer only ships two files
    assert len(layer) == 1


def test_token_factory_key_names_and_peft_key_names_both_load_strictly(
    stubs: Path, tmp_path: Path, bridge: AsyncBridge
) -> None:
    for i, keys in enumerate((TF_KEYS, PEFT_KEYS)):
        d = tmp_path / f"k{i}"
        d.mkdir()
        files = adapter_files(d, keys=keys)
        s = SandboxCpuStudent(
            make_sandbox(stubs, d, Log()), "img", bridge, base_model=MODEL, adapter_files=files
        )
        out = s.generate(msgs(1))[0]
        assert out.startswith("adapter:") and "RANDOM" not in out
        assert s.timings[0]["adapter_tensors"] == 2  # type: ignore[index]


def test_bin_weights_load_through_torch_load(
    stubs: Path, tmp_path: Path, bridge: AsyncBridge
) -> None:
    files = adapter_files(tmp_path, weights="adapter_model.bin")
    s = SandboxCpuStudent(
        make_sandbox(stubs, tmp_path, Log()), "img", bridge, base_model=MODEL, adapter_files=files
    )
    assert s.generate(msgs(1))[0].startswith("adapter:")


def test_an_adapter_that_does_not_load_cleanly_fails_loudly_not_as_random_noise(
    stubs: Path, tmp_path: Path, bridge: AsyncBridge
) -> None:
    files = adapter_files(tmp_path, keys=["some.other.layer.lora_A.weight"])
    s = SandboxCpuStudent(
        make_sandbox(stubs, tmp_path, Log()), "img", bridge, base_model=MODEL, adapter_files=files
    )
    with pytest.raises(StudentServingError, match="did not load cleanly"):
        s.generate(msgs(1))


def test_adapter_weights_are_renamed_when_the_download_uses_another_name(
    tmp_path: Path,
) -> None:
    ad = prepare_adapter(adapter_files(tmp_path, weights="model.safetensors"), MODEL)
    assert sorted(ad.files) == ["adapter_config.json", "adapter_model.safetensors"]
    assert ad.base_model_name_or_path == MODEL and len(ad.sha256) == 64


def test_bin_weights_are_supported_and_training_args_ignored(tmp_path: Path) -> None:
    files = adapter_files(tmp_path, weights="pytorch_model.bin")
    extra = tmp_path / "training_args.bin"
    extra.write_bytes(b"x")
    with pytest.raises(StudentServingError, match="ambiguous"):
        prepare_adapter([*files, extra], MODEL)
    ad = prepare_adapter(files, MODEL)
    assert sorted(ad.files) == ["adapter_config.json", "adapter_model.bin"]
    exact = tmp_path / "adapter_model.safetensors"
    exact.write_bytes(b"S")
    ad2 = prepare_adapter([*files, extra, exact], MODEL)  # exact names win over guesses
    assert sorted(ad2.files) == ["adapter_config.json", "adapter_model.safetensors"]


def test_adapter_base_model_mismatch_is_refused_with_the_evidence(tmp_path: Path) -> None:
    files = adapter_files(tmp_path, base="/data/some/other-model")
    with pytest.raises(StudentServingError) as ei:
        prepare_adapter(files, MODEL)
    msg = str(ei.value)
    assert "/data/some/other-model" in msg and MODEL in msg
    assert "adapter_config.json" in msg and "adapter_model.safetensors" in msg  # file names


def test_adapter_missing_pieces_are_refused_naming_the_files(tmp_path: Path) -> None:
    w = tmp_path / "adapter_model.safetensors"
    w.write_bytes(b"W")
    with pytest.raises(StudentServingError, match="adapter_config.json"):
        prepare_adapter([w], MODEL)
    cfg = tmp_path / "adapter_config.json"
    cfg.write_text(json.dumps({"base_model_name_or_path": MODEL}))
    with pytest.raises(StudentServingError, match="weights"):
        prepare_adapter([cfg, tmp_path / "notes.txt"], MODEL)
    cfg.write_text(json.dumps({"r": 8}))
    with pytest.raises(StudentServingError, match="base_model_name_or_path"):
        prepare_adapter([cfg, w], MODEL)
    cfg.write_text("not json")
    with pytest.raises(StudentServingError, match="not valid JSON"):
        prepare_adapter([cfg, w], MODEL)


def test_mismatching_adapter_fails_before_any_sandbox_call(
    stubs: Path, tmp_path: Path, bridge: AsyncBridge
) -> None:
    log = Log()
    sb = make_sandbox(stubs, tmp_path, log)
    with pytest.raises(StudentServingError, match="base_model_name_or_path"):
        SandboxCpuStudent(
            sb, "img", bridge, base_model=MODEL, adapter_files=adapter_files(tmp_path, base="x")
        )
    assert sb.run_count == 0


def test_for_artifact_takes_the_downloaded_files(
    stubs: Path, tmp_path: Path, bridge: AsyncBridge
) -> None:
    files = adapter_files(tmp_path)
    trained = TrainedArtifact(
        "job", "ckpt", MODEL, None, tuple(DownloadedFile(f.name, f, "0" * 64) for f in files)
    )
    s = SandboxCpuStudent.for_artifact(
        make_sandbox(stubs, tmp_path, Log()), "img", bridge, trained, base_model=MODEL
    )
    assert s.generate(msgs(1))[0].startswith("adapter:")


# ---- failures are counted, never dropped -------------------------------------------------------


def test_per_sample_failure_is_counted_and_reported_not_dropped(
    stubs: Path, tmp_path: Path, bridge: AsyncBridge
) -> None:
    s = SandboxCpuStudent(
        make_sandbox(stubs, tmp_path, Log()), "img", bridge, base_model=MODEL, batch_size=4
    )
    batch = msgs(7)
    batch[3] = [{"role": "user", "content": "boom"}]
    out = s.generate(batch)
    assert len(out) == 7 and out[3] == "" and out[0] == "base|q0" and out[6] == "base|q6"
    assert s.generation_errors == 1
    assert "boom" in s.error_samples[0]
    assert sum(t["failed"] for t in s.timings) == 1  # type: ignore[misc]


def test_too_many_sample_failures_raise_but_stay_counted(
    stubs: Path, tmp_path: Path, bridge: AsyncBridge
) -> None:
    s = SandboxCpuStudent(
        make_sandbox(stubs, tmp_path, Log()), "img", bridge, base_model=MODEL, batch_size=4
    )
    bad = [[{"role": "user", "content": "boom"}] for _ in range(3)]
    with pytest.raises(StudentServingError, match="3 of 4"):
        s.generate([*bad, *msgs(1)])
    assert s.generation_errors == 3


def test_failed_batch_is_loud(stubs: Path, tmp_path: Path, bridge: AsyncBridge) -> None:
    def handler(job: Job, fs: dict[str, bytes]) -> FakeExecution:
        if "gen.py" in (job.shell or "") and "pip install" not in (job.shell or ""):
            return FakeExecution("", "OSError: no network", 1)
        return FakeExecution()

    s = SandboxCpuStudent(FakeSandbox(handler), "img", bridge, base_model=MODEL)
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


# ---- shared images -----------------------------------------------------------------------------


def test_base_and_student_share_one_deps_image(
    stubs: Path, tmp_path: Path, bridge: AsyncBridge
) -> None:
    log = Log()
    sb = make_sandbox(stubs, tmp_path, log)
    seen: list[tuple[str, str]] = []
    images = ServingImages(
        sb, "img", bridge, base_model=MODEL, on_image=lambda kind, uuid: seen.append((kind, uuid))
    )
    files = adapter_files(tmp_path)
    for _ in range(2):
        SandboxCpuStudent(sb, "img", bridge, base_model=MODEL, images=images).generate(msgs(1))
        SandboxCpuStudent(
            sb, "img", bridge, base_model=MODEL, adapter_files=files, images=images
        ).generate(msgs(1))
    assert len([e for e in log.entries if "pip install" in e]) == 1  # one build, four servers
    kinds = [k for k, _ in seen]
    assert kinds == ["deps", "adapter"]  # each built once, recorded once
    assert all(u for _, u in seen)


def test_setup_shell_quotes_base_model_and_uses_the_measured_recipe() -> None:
    sh = setup_shell("a/b")
    assert 'snapshot_download("a/b"' in sh
    assert "download.pytorch.org/whl/cpu" in sh and "peft" in sh and "accelerate" in sh


def test_script_marker_matches_parser() -> None:
    from distillery.sandbox_student import GEN_SCRIPT, OUT_MARKER

    assert f'print("{OUT_MARKER}"' in GEN_SCRIPT
    assert "enable_thinking=False" in GEN_SCRIPT and "bfloat16" in GEN_SCRIPT
    assert "set_peft_model_state_dict" in GEN_SCRIPT  # explicit strict adapter load
