# ruff: noqa: S101
"""A2: error classification, eval re-queue, ambiguous create_job, latency column, exit codes."""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from dataclasses import replace
from pathlib import Path
from typing import Any

import httpx2
import openai
import pytest

from distillery import cli
from distillery.budget import BudgetExceeded, UnknownPriceError
from distillery.errors import (
    EXIT_RETRYABLE,
    InfraError,
    RetryableRunError,
    RunSuspended,
    classify,
    is_known,
)
from distillery.evaluator import ArtifactMismatchError
from distillery.finetune import JobFailedError, PollTimeoutError
from distillery.llm import RetriesExhaustedError, SchemaValidationError, TerminalLLMError
from distillery.orchestrator import (
    ConfigRefusal,
    Pipeline,
    Scale,
    ShortfallError,
    TaskTooEasyError,
    VerifierSelfTestError,
)
from distillery.pipeline_fakes import build_dry_run
from distillery.sandbox import SandboxError, SandboxTimeoutError
from distillery.sandbox_executor import AsyncBridge
from distillery.store import HeldoutIntegrityError, Store
from distillery.student import StudentServingError

NANO = Scale(name="nano", train=16, dev=8, heldout=12, stress=6)
_REQ = httpx2.Request("POST", "https://example.invalid/v1/x")


def status_error(code: int) -> openai.APIStatusError:
    return openai.APIStatusError(
        f"http {code}", response=httpx2.Response(code, request=_REQ), body=None
    )


def conn_error() -> openai.APIConnectionError:
    return openai.APIConnectionError(request=_REQ)


def wrapped(outer: Exception, inner: BaseException) -> Exception:
    try:
        raise outer from inner
    except Exception as e:  # noqa: BLE001
        return e


@pytest.fixture
def bridge() -> Iterator[AsyncBridge]:
    with AsyncBridge() as b:
        yield b


# ---- classify -------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "exc",
    [
        RetryableRunError("x"),
        InfraError("x"),
        RetriesExhaustedError("x"),
        SandboxTimeoutError("x"),
        PollTimeoutError("x", unreachable=True),
        StudentServingError("lost", transient=True),
        status_error(503),
        status_error(429),
        conn_error(),
        wrapped(SandboxError("transport"), status_error(502)),  # the cause decides
    ],
)
def test_retryable(exc: BaseException) -> None:
    assert classify(exc) == "retryable"


@pytest.mark.parametrize(
    "exc",
    [
        BudgetExceeded("cap"),
        UnknownPriceError("no price"),
        ConfigRefusal("no"),
        VerifierSelfTestError("bad"),
        TaskTooEasyError("easy"),
        ShortfallError("short"),
        JobFailedError("j", "failed", None, None),
        TerminalLLMError("refused"),
        ArtifactMismatchError("sha"),
        HeldoutIntegrityError("seal"),
        PollTimeoutError("still running"),
        StudentServingError("bad adapter"),
        status_error(400),
        SchemaValidationError("never valid"),  # unknown to the table: terminal
        ValueError("a bug"),
        wrapped(TerminalLLMError("400"), conn_error()),  # an explicit terminal outer type wins
        wrapped(BudgetExceeded("cap"), RetriesExhaustedError("x")),
    ],
)
def test_terminal(exc: BaseException) -> None:
    assert classify(exc) == "terminal"


def test_interrupts_are_never_retryable() -> None:
    assert classify(KeyboardInterrupt()) == "terminal"
    assert classify(RunSuspended()) == "terminal"  # the supervisor handles suspension itself


def test_is_known_flags_bugs() -> None:
    assert is_known(BudgetExceeded("x")) and is_known(wrapped(SandboxError("x"), conn_error()))
    assert not is_known(ValueError("bug"))


# ---- eval re-queue (gap 3b) ------------------------------------------------------------------


def _failing_transport(dr: Any, failures: dict[str, int], armed: dict[str, bool]) -> None:
    real = dr.transport.chat.completions.create

    async def create(**kw: Any) -> Any:
        if armed["on"] and failures["left"] > 0:
            failures["left"] -= 1
            raise status_error(503)
        return await real(**kw)

    dr.transport.chat.completions.create = create


async def _no_sleep(_s: float) -> None:
    return None


def _rig(tmp_path: Path, bridge: AsyncBridge, fail_n: int) -> tuple[Pipeline, Any, dict[str, int]]:
    armed = {"on": False}
    dr = build_dry_run(
        NANO, bridge, on_stage=lambda s: armed.update(on=s == "final_eval"),
        error_rates={},  # every fake model answers gold: re-queue must not change any verdict
    )  # fmt: skip
    failures = {"left": fail_n}
    _failing_transport(dr, failures, armed)
    deps = replace(dr.deps, llm_sleep=_no_sleep)
    store = Store(tmp_path / "store")
    pipe = Pipeline(dr.pipeline_cfg, dr.config, deps, store, "dry-t", say=lambda _s: None)
    return pipe, dr, failures


def test_eval_items_lost_to_5xx_are_requeued_not_scored_wrong(
    tmp_path: Path, bridge: AsyncBridge
) -> None:
    # 5 attempts per call exhaust once for the first item, then everything succeeds
    pipe, _dr, failures = _rig(tmp_path, bridge, fail_n=5)
    rep = pipe.run()
    assert failures["left"] == 0
    assert rep["evaluation"]["accuracy"]["teacher"] == 1.0
    assert rep["counters"]["final_eval"].get("unparseable_teacher", 0) == 0
    assert pipe.runner.errors["eval_teacher:requeued"] == 1


def test_eval_items_still_failing_abort_retryably(tmp_path: Path, bridge: AsyncBridge) -> None:
    pipe, _dr, _f = _rig(tmp_path, bridge, fail_n=10_000)
    with pytest.raises(RetryableRunError, match="not scoring them as wrong"):
        pipe.run()


# ---- ambiguous create_job (gap 4) ------------------------------------------------------------


def test_ambiguous_create_adopts_the_job_it_did_create(tmp_path: Path, bridge: AsyncBridge) -> None:
    dr = build_dry_run(NANO, bridge)
    real = dr.finetune.create_job

    def create_then_502(*a: Any, **k: Any) -> str:
        real(*a, **k)  # the provider created it...
        raise status_error(502)  # ...but the response was lost

    dr.finetune.create_job = create_then_502  # type: ignore[method-assign]
    store = Store(tmp_path / "s")
    Pipeline(dr.pipeline_cfg, dr.config, dr.deps, store, "dry-t", say=lambda _s: None).run()
    r1 = [j for j in dr.finetune.created if dr.finetune.suffixes[j].endswith("-r1")]
    assert len(r1) == 1 and dr.finetune.cancelled == []
    names = [n for n, _ in store.list_experiments("dry-t")]
    assert "finetune_job_adopted" in names and "finetune_job_started" in names


def test_ambiguous_create_with_no_job_is_retryable(tmp_path: Path, bridge: AsyncBridge) -> None:
    dr = build_dry_run(NANO, bridge)

    def lost(*_a: Any, **_k: Any) -> str:
        raise conn_error()

    dr.finetune.create_job = lost  # type: ignore[method-assign]
    pipe = Pipeline(
        dr.pipeline_cfg, dr.config, dr.deps, Store(tmp_path / "s"), "dry-t", say=lambda _s: None
    )
    with pytest.raises(RetryableRunError, match="ambiguously"):
        pipe.run()


def test_unrecorded_live_job_is_found_before_creating_another(
    tmp_path: Path, bridge: AsyncBridge
) -> None:
    """The provider list may lag the create: the next attempt must look before it creates."""
    dr = build_dry_run(NANO, bridge)
    ghost = dr.finetune.create_job(
        "m", "file-x", None, dr.pipeline_cfg.hyperparameters, suffix="distillery-dry-t-r1"
    )
    dr.finetune.job_status[ghost] = "running"
    store = Store(tmp_path / "s")
    Pipeline(dr.pipeline_cfg, dr.config, dr.deps, store, "dry-t", say=lambda _s: None).run()
    r1 = [j for j in dr.finetune.created if dr.finetune.suffixes[j].endswith("-r1")]
    assert r1 == [ghost]


def test_run_id_too_long_for_the_suffix_is_refused(tmp_path: Path, bridge: AsyncBridge) -> None:
    dr = build_dry_run(NANO, bridge)
    rid = "dry-" + "x" * 60
    pipe = Pipeline(dr.pipeline_cfg, dr.config, dr.deps, Store(tmp_path / "s"), rid)
    with pytest.raises(ConfigRefusal, match="too long"):
        pipe.run()
    assert dr.finetune.created == []


# ---- latency column (gap 7) ------------------------------------------------------------------


def test_latency_is_stored_and_old_stores_are_migrated(tmp_path: Path, bridge: AsyncBridge) -> None:
    old = tmp_path / "old"
    old.mkdir()
    con = sqlite3.connect(old / "index.sqlite")
    con.execute(
        "CREATE TABLE llm_calls (id INTEGER PRIMARY KEY AUTOINCREMENT, run_id TEXT NOT NULL, "
        "model TEXT NOT NULL, input_tokens INTEGER NOT NULL, output_tokens INTEGER NOT NULL, "
        "usd REAL NOT NULL, created_at TEXT NOT NULL)"
    )
    con.execute("INSERT INTO llm_calls VALUES (1,'r','m',1,1,0.1,'t')")
    con.commit()
    con.close()
    store = Store(old)
    store.record_llm_call("r", "m", 2, 2, 0.2, latency_s=1.5)
    rows = store._query("SELECT latency_s FROM llm_calls ORDER BY id")
    assert rows == [(None,), (1.5,)]  # old rows stay NULL, nothing is invented
    store.close()

    pipe, _dr, _f = _rig(tmp_path, bridge, fail_n=0)
    pipe.run()
    lat = pipe.store._query("SELECT COUNT(*), COUNT(latency_s) FROM llm_calls")
    assert lat[0][0] > 0 and lat[0][0] == lat[0][1]


# ---- CLI exit codes --------------------------------------------------------------------------


def test_cli_maps_retryable_failure_to_75(tmp_path: Path, bridge: AsyncBridge) -> None:
    lines: list[str] = []
    code = cli.main(
        ["--root", str(tmp_path), "run", "--dry-run", "--run-id", "dry-x", "--scale", "tiny"],
        env={}, out=lines.append,
    )  # fmt: skip
    assert code == 0  # control: the plain dry run passes
    import distillery.cli as c

    real = c.build_dry_run

    def patched(*a: Any, **k: Any) -> Any:
        dr = real(*a, **k)
        dr.finetune.create_job = lambda *_a, **_k: (_ for _ in ()).throw(conn_error())
        return dr

    c.build_dry_run = patched  # type: ignore[assignment]
    try:
        lines.clear()
        code = cli.main(
            ["--root", str(tmp_path), "run", "--dry-run", "--run-id", "dry-y", "--scale", "tiny"],
            env={}, out=lines.append,
        )  # fmt: skip
    finally:
        c.build_dry_run = real  # type: ignore[assignment]
    assert code == EXIT_RETRYABLE
    assert any(ln.startswith("FAILED (retryable): RetryableRunError") for ln in lines)
    assert not any("Traceback" in ln for ln in lines)
