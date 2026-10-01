# ruff: noqa: S101, S604
from __future__ import annotations

import hashlib
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import httpx2
import openai
import pytest
from pydantic import ValidationError

from distillery.config import ConfigError
from distillery.finetune import (
    PINNED_HYPERPARAMETERS,
    FineTuneClient,
    FineTuneError,
    HyperParameters,
    JobFailedError,
    PollTimeoutError,
    paid_job,
    planned_steps,
    require_explicit_hyperparameters,
)

_REQ = httpx2.Request("POST", "https://example.invalid/v1/x")


def err(code: int) -> openai.APIStatusError:
    return openai.APIStatusError("e", response=httpx2.Response(code, request=_REQ), body=None)


def job(status: str, **kw: Any) -> Any:
    return SimpleNamespace(
        id="job-1",
        status=status,
        model="base/m",
        error=kw.get("error"),
        result_files=[],
        trained_steps=1,
        total_steps=2,
        trained_tokens=kw.get("trained_tokens", 5000),
        hyperparameters=kw.get("hyperparameters"),
    )


class Content:
    def __init__(self, data: bytes) -> None:
        self.data = data

    def write_to_file(self, path: str) -> None:
        Path(path).write_bytes(self.data)


class FakeOpenAI:
    def __init__(self) -> None:
        self.job_script: list[Any] = []
        self.create_script: list[Any] = []
        self.cancelled: list[str] = []
        self.cancel_exc: BaseException | None = None
        self.created: list[dict[str, Any]] = []
        self.file_store = {
            "f1": ("ck1/adapter_config.json", b"{}"),
            "f2": ("ck1/adapter_model.safetensors", b"weights"),
        }
        self.files = SimpleNamespace(
            create=self._fcreate,
            retrieve=self._fret,
            content=lambda fid: Content(self.file_store[fid][1]),
        )
        jobs = SimpleNamespace(
            create=self._create,
            retrieve=self._retrieve,
            cancel=self._cancel,
            list_events=self._events,
            checkpoints=SimpleNamespace(list=self._ckpts),
        )
        self.fine_tuning = SimpleNamespace(jobs=jobs)

    def _fcreate(self, file: Any, purpose: str) -> Any:
        assert purpose == "fine-tune"
        if self.create_script:
            item = self.create_script.pop(0)
            if isinstance(item, BaseException):
                raise item
        return SimpleNamespace(id="file-9")

    def _fret(self, fid: str) -> Any:
        return SimpleNamespace(filename=self.file_store[fid][0])

    def _create(self, **kw: Any) -> Any:
        self.created.append(kw)
        return SimpleNamespace(id="job-1")

    def _retrieve(self, jid: str) -> Any:
        item = self.job_script.pop(0)
        if isinstance(item, BaseException):
            raise item
        return item

    def _cancel(self, jid: str) -> Any:
        if self.cancel_exc:
            raise self.cancel_exc
        self.cancelled.append(jid)
        return job("cancelled")

    def _events(self, jid: str, **kw: Any) -> Any:
        if kw.get("after") is None:
            return SimpleNamespace(
                data=[SimpleNamespace(id="e1", created_at=1, level="info", message="a")],
                has_more=True,
            )
        return SimpleNamespace(
            data=[SimpleNamespace(id="e2", created_at=2, level="info", message="b")], has_more=False
        )

    def _ckpts(self, jid: str) -> Any:
        return SimpleNamespace(
            data=[
                SimpleNamespace(
                    id="ck1",
                    step_number=3,
                    fine_tuned_model_checkpoint="ft:x",
                    result_files=["f1", "f2"],
                    metrics={"train_loss": 1.0},
                )
            ]
        )


def make(
    fake: FakeOpenAI | None = None,
) -> tuple[FineTuneClient, FakeOpenAI, list[float], list[float]]:
    fake = fake or FakeOpenAI()
    sleeps: list[float] = []
    now = [0.0]

    def sleep(s: float) -> None:
        sleeps.append(s)
        now[0] += s

    return (
        FineTuneClient(fake, sleep=sleep, clock=lambda: now[0], jitter=lambda: 1.0),
        fake,
        sleeps,
        now,
    )


# ---- hyperparameters
@pytest.mark.parametrize(
    "kw",
    [
        {"lora_r": 7},
        {"lora_r": 129},
        {"lora_alpha": 7},
        {"n_epochs": 0},
        {"n_epochs": 21},
        {"lora_dropout": 1.1},
        {"warmup_ratio": -0.1},
        {"max_grad_norm": 0},
        {"batch_size": 65},
        {"learning_rate": -1e-5},
        {"context_length": 4096},
        {"bogus": 1},
    ],
)
def test_hyperparameter_bounds_rejected(kw: dict[str, Any]) -> None:
    with pytest.raises(ValidationError):
        HyperParameters(**kw)


def test_context_length_must_be_supported_value() -> None:
    with pytest.raises(ValueError):
        HyperParameters(context_length=10000).to_request()
    assert HyperParameters(context_length=16384, lora=True, lora_r=16).to_request() == {
        "context_length": 16384,
        "lora": True,
        "lora_r": 16,
    }


def test_create_job_request_shape() -> None:
    c, fake, *_ = make()
    jid = c.create_job(
        "Qwen/Qwen3-1.7B",
        "f-train",
        "f-val",
        PINNED_HYPERPARAMETERS,
        suffix="s",
        seed=42,
    )
    assert jid == "job-1"
    assert fake.created[0] == {
        "model": "Qwen/Qwen3-1.7B",
        "training_file": "f-train",
        "validation_file": "f-val",
        "hyperparameters": PINNED_HYPERPARAMETERS.to_request(),
        "suffix": "s",
        "seed": 42,
    }
    with pytest.raises(ValueError):
        c.create_job("m", "f", suffix="x" * 65)


def test_create_job_is_not_retried() -> None:
    c, fake, *_ = make()
    fake.fine_tuning.jobs.create = lambda **kw: (_ for _ in ()).throw(err(503))
    with pytest.raises(openai.APIStatusError):
        c.create_job("m", "f", hyperparameters=PINNED_HYPERPARAMETERS)


# ---- upload / retry
def test_upload_retries_5xx_then_ok(tmp_path: Path) -> None:
    f = tmp_path / "t.jsonl"
    f.write_text("{}\n")
    fake = FakeOpenAI()
    fake.create_script = [err(502), err(503)]
    c, _, sleeps, _ = make(fake)
    assert c.upload(f) == "file-9"
    assert sleeps == [1.0, 2.0]


def test_retry_is_capped_and_4xx_terminal(tmp_path: Path) -> None:
    f = tmp_path / "t.jsonl"
    f.write_text("{}\n")
    fake = FakeOpenAI()
    fake.create_script = [err(500)] * 10
    c, _, sleeps, _ = make(fake)
    with pytest.raises(openai.APIStatusError):
        c.upload(f)
    assert len(sleeps) == 3  # 4 attempts total
    fake2 = FakeOpenAI()
    fake2.create_script = [err(400), err(400)]
    c2, _, sleeps2, _ = make(fake2)
    with pytest.raises(openai.APIStatusError):
        c2.upload(f)
    assert sleeps2 == []


# ---- polling
def test_poll_until_terminal_with_interval() -> None:
    c, fake, sleeps, _ = make()
    fake.job_script = [job("validating_files"), job("queued"), job("running"), job("succeeded")]
    info = c.poll("job-1")
    assert info.status == "succeeded" and sleeps == [15.0, 15.0, 15.0]


@pytest.mark.parametrize("status", ["failed", "cancelled"])
def test_poll_stops_on_other_terminal(status: str) -> None:
    c, fake, sleeps, _ = make()
    fake.job_script = [job(status, error=SimpleNamespace(code="c", message="m"))]
    info = c.poll("job-1")
    assert info.terminal and sleeps == []
    with pytest.raises(JobFailedError):
        c.require_success(info)


def test_poll_interval_floor_and_timeout() -> None:
    c, fake, *_ = make()
    with pytest.raises(ValueError):
        c.poll("job-1", interval_s=5)
    fake.job_script = [job("running")] * 50
    with pytest.raises(PollTimeoutError):
        c.poll("job-1", timeout_s=40)


def test_poll_survives_transient_5xx() -> None:
    c, fake, *_ = make()
    fake.job_script = [err(503), job("succeeded")]
    assert c.poll("job-1").status == "succeeded"


# ---- events / checkpoints / download
def test_events_paginate() -> None:
    c, *_ = make()
    assert [e.message for e in c.events("job-1")] == ["a", "b"]


def test_download_checkpoint_hashes_and_artifact(tmp_path: Path) -> None:
    c, *_ = make()
    (ck,) = c.checkpoints("job-1")
    assert ck.result_files == ("f1", "f2") and ck.metrics == {"train_loss": 1.0}
    files = c.download_checkpoint(ck, tmp_path)
    by_name = {f.path.name: f.sha256 for f in files}
    assert by_name["adapter_model.safetensors"] == hashlib.sha256(b"weights").hexdigest()
    assert files[0].path.parent == tmp_path / "ck1"
    art = c.trained_artifact(job("succeeded") and c._job_info(job("succeeded")), ck, tmp_path)
    assert art.job_id == "job-1" and art.checkpoint_id == "ck1"
    assert art.file_sha256 == by_name
    assert len(art.adapter_sha256) == 64


def test_download_rejects_path_traversal(tmp_path: Path) -> None:
    c, fake, *_ = make()
    fake.file_store["f1"] = ("../../etc/passwd", b"x")
    (ck,) = c.checkpoints("job-1")
    c.download_checkpoint(ck, tmp_path)
    assert not (tmp_path.parent / "etc").exists()
    assert (tmp_path / "ck1" / "passwd").exists()
    fake.file_store["f1"] = ("dir/", b"x")
    with pytest.raises(FineTuneError):
        c.download_checkpoint(ck, tmp_path)


# ---- paid_job
def test_paid_job_cancels_on_exception() -> None:
    c, fake, *_ = make()
    with pytest.raises(RuntimeError, match="boom"):
        with paid_job(
            lambda: c.create_job("m", "f", hyperparameters=PINNED_HYPERPARAMETERS), c.cancel
        ) as h:
            assert h.job_id == "job-1"
            raise RuntimeError("boom")
    assert fake.cancelled == ["job-1"]


def test_paid_job_cancels_if_not_marked_succeeded() -> None:
    c, fake, *_ = make()
    with paid_job(lambda: c.create_job("m", "f", hyperparameters=PINNED_HYPERPARAMETERS), c.cancel):
        pass
    assert fake.cancelled == ["job-1"]


def test_paid_job_no_cancel_on_success() -> None:
    c, fake, *_ = make()
    with paid_job(
        lambda: c.create_job("m", "f", hyperparameters=PINNED_HYPERPARAMETERS), c.cancel
    ) as h:
        h.mark_succeeded()
    assert fake.cancelled == []


def test_paid_job_cancel_failure_does_not_mask_original() -> None:
    c, fake, *_ = make()
    fake.cancel_exc = err(409)
    with pytest.raises(RuntimeError) as ei:
        with paid_job(
            lambda: c.create_job("m", "f", hyperparameters=PINNED_HYPERPARAMETERS), c.cancel
        ):
            raise RuntimeError("boom")
    assert any("cancelling job job-1 failed" in n for n in ei.value.__notes__)


def test_paid_job_forced_failure_from_poll_cancels() -> None:
    c, fake, *_ = make()
    fake.job_script = [job("running"), job("failed", error=SimpleNamespace(code="x", message="y"))]
    with pytest.raises(JobFailedError):
        with paid_job(
            lambda: c.create_job("m", "f", hyperparameters=PINNED_HYPERPARAMETERS), c.cancel
        ) as h:
            c.require_success(c.poll(h.job_id))
            h.mark_succeeded()
    assert fake.cancelled == ["job-1"]


def test_paid_job_create_failure_means_nothing_to_cancel() -> None:
    c, fake, *_ = make()

    def bad() -> str:
        raise ValueError("no job")

    with pytest.raises(ValueError):
        with paid_job(bad, c.cancel):
            pass
    assert fake.cancelled == []


# ---- a network outage while polling must not kill (and cancel) a job that is running fine
def conn_err() -> openai.APIConnectionError:
    return openai.APIConnectionError(request=_REQ)


def test_poll_keeps_polling_through_an_outage_shorter_than_the_grace() -> None:
    c, fake, sleeps, _ = make()
    # each get() burns 4 attempts (client-level retries) before the outage reaches poll()
    fake.job_script = [job("running"), *[conn_err()] * 12, job("running"), job("succeeded")]
    seen: list[BaseException] = []
    info = c.poll("job-1", on_error=seen.append)
    assert info.status == "succeeded"
    assert len(seen) == 3 and all(isinstance(e, openai.APIConnectionError) for e in seen)


def test_poll_gives_up_when_the_outage_outlasts_the_grace() -> None:
    c, fake, *_ = make()
    fake.job_script = [job("running"), *[conn_err()] * 400]
    with pytest.raises(openai.APIConnectionError):
        c.poll("job-1", transient_error_grace_s=60.0)


def test_poll_does_not_tolerate_terminal_http_errors() -> None:
    c, fake, sleeps, _ = make()
    fake.job_script = [err(401)]
    with pytest.raises(openai.APIStatusError):
        c.poll("job-1")
    assert sleeps == []


# ---- explicit hyperparameters guard
def test_default_style_hyperparameters_refused_before_any_api_call() -> None:
    c, fake, *_ = make()
    with pytest.raises(ConfigError, match="batch_size"):
        c.create_job("m", "f", hyperparameters=HyperParameters(lora=True, n_epochs=3))
    with pytest.raises(ConfigError):
        c.create_job("m", "f")  # none at all
    assert fake.created == []


def test_guard_requires_lora_true_and_each_field() -> None:
    with pytest.raises(ConfigError, match="lora"):
        require_explicit_hyperparameters(PINNED_HYPERPARAMETERS.model_copy(update={"lora": False}))
    for k in ("batch_size", "learning_rate", "n_epochs", "lora_r", "lora_alpha", "packing"):
        with pytest.raises(ConfigError, match=k):
            require_explicit_hyperparameters(PINNED_HYPERPARAMETERS.model_copy(update={k: None}))
    assert require_explicit_hyperparameters(PINNED_HYPERPARAMETERS) is PINNED_HYPERPARAMETERS


def test_pinned_values() -> None:
    assert PINNED_HYPERPARAMETERS.to_request() == {
        "lora": True, "lora_r": 16, "lora_alpha": 16, "learning_rate": 1e-4, "n_epochs": 3,
        "batch_size": 16, "packing": False, "warmup_ratio": 0.0, "weight_decay": 0.0,
        "max_grad_norm": 1.0, "lora_dropout": 0.0, "context_length": 8192,
    }  # fmt: skip


def test_planned_steps() -> None:
    assert planned_steps(117, PINNED_HYPERPARAMETERS) == 8 * 3
    assert planned_steps(40, PINNED_HYPERPARAMETERS.model_copy(update={"packing": True})) is None


# ---- resolved job fields / loss curve
def test_job_info_carries_resolved_hyperparameters_and_tokens() -> None:
    c, fake, *_ = make()
    hp = {"batch_size": 8, "learning_rate": 1e-5, "packing": True}
    fake.job_script = [job("succeeded", hyperparameters=SimpleNamespace(**hp))]
    info = c.get("job-1")
    assert info.hyperparameters == hp
    assert info.trained_tokens == 5000 and info.trained_steps == 1 and info.total_steps == 2
    fake.job_script = [job("succeeded", hyperparameters=hp)]
    assert c.get("job-1").hyperparameters == hp


def test_loss_curve_from_checkpoints() -> None:
    c, *_ = make()
    assert c.loss_curve("job-1") == [{"step": 3, "train_loss": 1.0, "valid_loss": None}]
