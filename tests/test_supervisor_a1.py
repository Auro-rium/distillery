# ruff: noqa: S101
"""A1: supervisor restarts, suspend vs cancel, startup reconcile, audit log."""

from __future__ import annotations

import json
import os
import signal
import sqlite3
import threading
import time
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from distillery.cli import _suspend_on_sigterm
from distillery.errors import RunSuspended
from distillery.orchestrator import Pipeline, Scale
from distillery.pipeline_fakes import build_dry_run
from distillery.sandbox_executor import AsyncBridge
from distillery.server import autonomy, create_app
from distillery.server.worker import Job, Worker
from distillery.store import Store
from tests.server_helpers import ADMIN, make_settings

NANO = Scale(name="nano", train=16, dev=8, heldout=12, stress=6)


@pytest.fixture
def bridge() -> Iterator[AsyncBridge]:
    with AsyncBridge() as b:
        yield b


def _wait(pred: Callable[[], bool], timeout: float = 10.0) -> None:
    end = time.monotonic() + timeout
    while not pred():
        assert time.monotonic() < end, "timed out"
        time.sleep(0.01)


class Script:
    """Executor fake: returns the scripted exit codes in order, one per attempt."""

    def __init__(self, codes: list[int]) -> None:
        self.codes = list(codes)
        self.calls = 0

    def __call__(self, job: Job, log: Callable[[str], None]) -> int:
        self.calls += 1
        return self.codes.pop(0)


def _worker(codes: list[int], **kw: Any) -> tuple[Worker, Script, list[Any], list[Any]]:
    audits: list[Any] = []
    finals: list[Any] = []
    ex = Script(codes)
    w = Worker(
        ex, restart_backoff_s=0.0, on_audit=lambda *a: audits.append(a),
        on_final=lambda j, c: finals.append((j.run_id, c)), **kw,
    )  # fmt: skip
    return w, ex, audits, finals


def _run(w: Worker, rid: str = "r1") -> Job:
    job = Job(rid, "tiny", True)
    w.submit(job)
    _wait(lambda: job.state == "finished")
    return job


# ---- supervisor rules -------------------------------------------------------------------------


def test_killed_child_is_restarted_once_and_then_completes() -> None:
    w, ex, audits, finals = _worker([-9, 0])  # SIGKILL / OOM, then the resume succeeds
    job = _run(w)
    assert ex.calls == 2 and job.restarts == 1 and job.error is None
    assert [(a[0], a[1]) for a in audits] == [("supervisor", "restart")]
    assert audits[0][3]["exit_code"] == -9 and finals == [("r1", 0)]
    w.stop()


def test_retryable_exit_is_restarted() -> None:
    w, ex, audits, _f = _worker([75, 75, 0])
    job = _run(w)
    assert ex.calls == 3 and job.restarts == 2 and job.error is None
    w.stop()


@pytest.mark.parametrize("code", [1, 2])
def test_final_exit_codes_are_never_restarted(code: int) -> None:
    w, ex, audits, finals = _worker([code])
    job = _run(w)
    assert ex.calls == 1 and job.restarts == 0 and audits == []
    assert finals == [("r1", code)] and job.error and job.error.startswith(f"exit code {code}")
    w.stop()


def test_restart_budget_is_bounded() -> None:
    w, ex, audits, finals = _worker([75] * 10, max_restarts=3)
    job = _run(w)
    assert ex.calls == 4 and job.restarts == 3
    assert [a[1] for a in audits] == ["restart"] * 3 + ["restarts_exhausted"]
    assert finals == [("r1", 75)]
    w.stop()


def test_cancel_during_backoff_is_not_restarted() -> None:
    started = threading.Event()

    def ex(job: Job, log: Callable[[str], None]) -> int:
        started.set()
        return 75

    w = Worker(ex, restart_backoff_s=60.0)
    job = Job("r1", "tiny", True)
    w.submit(job)
    _wait(lambda: job.restarts == 1)
    assert w.cancel("r1") == "cancelling"
    _wait(lambda: job.state == "finished", timeout=5)
    assert job.restarts == 1 and job.error == "cancelled"
    w.stop()


def test_server_stop_suspends_with_sigterm_not_sigint() -> None:
    sent: list[str] = []
    release = threading.Event()

    def ex(job: Job, log: Callable[[str], None]) -> int:
        job.interrupt = lambda: sent.append("SIGINT")
        job.suspend = lambda: (sent.append("SIGTERM"), release.set())  # type: ignore[func-returns-value]
        release.wait(5)
        return 75  # what the CLI returns when suspended

    finals: list[Any] = []
    w = Worker(ex, restart_backoff_s=0.0, on_final=lambda j, c: finals.append(c))
    job = Job("r1", "tiny", False)
    w.submit(job)
    _wait(lambda: job.suspend is not None)
    w.stop()
    assert sent == ["SIGTERM"] and job.restarts == 0  # a shutdown is never restarted here
    assert finals == []  # still resumable: autonomy.json stays active for the next start
    assert job.error and job.error.startswith("suspended")


# ---- suspend vs cancel inside the pipeline ----------------------------------------------------


def test_sigterm_handler_raises_run_suspended_in_main_thread() -> None:
    with pytest.raises(RunSuspended), _suspend_on_sigterm():
        os.kill(os.getpid(), signal.SIGTERM)
        time.sleep(1)
    assert signal.getsignal(signal.SIGTERM) is signal.SIG_DFL or callable(
        signal.getsignal(signal.SIGTERM)
    )


def test_suspend_during_finetune_keeps_the_job_and_the_resume_adopts_it(
    tmp_path: Path, bridge: AsyncBridge
) -> None:
    dr = build_dry_run(NANO, bridge)

    def suspended(*_a: Any, **_k: Any) -> Any:
        raise RunSuspended("SIGTERM")

    dr.finetune.checkpoints = suspended  # type: ignore[method-assign]
    store = Store(tmp_path / "s")
    with pytest.raises(RunSuspended):
        Pipeline(dr.pipeline_cfg, dr.config, dr.deps, store, "dry-t", say=lambda _s: None).run()
    (jid,) = dr.finetune.created
    assert dr.finetune.cancelled == []  # NOT cancelled (a SIGINT cancel would)
    assert "finetune_job_closed" not in [n for n, _ in store.list_experiments("dry-t")]

    dr2 = build_dry_run(NANO, bridge)
    dr2.finetune.job_status[jid] = "running"
    Pipeline(dr2.pipeline_cfg, dr2.config, dr2.deps, store, "dry-t", say=lambda _s: None).run()
    assert not [s for s in dr2.finetune.suffixes.values() if s.endswith("-r1")]  # no new r1 job
    adopted = [
        d["job_id"] for n, d in store.list_experiments("dry-t") if n == "finetune_job_adopted"
    ]
    assert adopted == [jid] and jid not in dr2.finetune.cancelled


# ---- reconcile + audit through the app --------------------------------------------------------


def _scripted(seen: list[str], code: int = 0) -> Callable[[Job, Callable[[str], None]], int]:
    def ex(job: Job, log: Callable[[str], None]) -> int:
        seen.append(job.run_id)
        return code

    return ex


def _leave_unfinished(root: Path, run_id: str, *, dry: bool, approved: bool = True) -> Path:
    base = root / "dry-runs" if dry else root
    run_dir = base / "runs" / run_id
    run_dir.mkdir(parents=True)
    autonomy.write(run_dir, Job(run_id, "tiny", dry, 3.0, None))
    if approved and not dry:
        (run_dir / "spend_approved.json").write_text(json.dumps({"run_id": run_id}))
    return run_dir


def test_startup_reconcile_resubmits_unfinished_runs_only(tmp_path: Path) -> None:
    s = make_settings(tmp_path)
    _leave_unfinished(s.root, "sql-tiny-live", dry=False)
    _leave_unfinished(s.root, "sql-tiny-noapproval", dry=False, approved=False)
    done = _leave_unfinished(s.root, "dry-done", dry=True)
    (done / "report.json").write_text("{}")
    final = _leave_unfinished(s.root, "dry-final", dry=True)
    autonomy.mark_final(final, 1, "exit code 1")
    seen: list[str] = []
    s.executor = _scripted(seen)
    app = create_app(s)
    with TestClient(app) as c:
        _wait(lambda: seen == ["sql-tiny-live"])
        audit = c.get("/api/runs/sql-tiny-live/audit").json()["audit"]
    assert [(a["actor"], a["action"]) for a in audit] == [("supervisor", "resubmit_on_start")]
    assert autonomy.read(s.root / "runs" / "sql-tiny-live")["state"] == "final"  # type: ignore[index]


def test_admin_start_is_audited_once_and_final_state_is_recorded(tmp_path: Path) -> None:
    s = make_settings(tmp_path)
    seen: list[str] = []
    s.executor = _scripted(seen)
    with TestClient(create_app(s)) as c:
        rid = c.post(
            "/api/runs",
            json={"scale": "tiny", "dry_run": False, "approve_spend": True},
            headers={"X-Admin-Token": ADMIN},
        ).json()["run_id"]
        _wait(lambda: seen == [rid])
        _wait(lambda: (autonomy.read(s.root / "runs" / rid) or {}).get("state") == "final")
        audit = c.get(f"/api/runs/{rid}/audit").json()["audit"]
        anon = c.post("/api/runs", json={"scale": "tiny", "dry_run": True}).json()["run_id"]
        _wait(lambda: anon in seen)
        assert c.get(f"/api/runs/{anon}/audit").json()["audit"] == []  # no admin action
    assert [(a["actor"], a["action"]) for a in audit] == [("admin", "start")]
    con = sqlite3.connect(s.root / "index.sqlite")
    assert con.execute("SELECT COUNT(*) FROM audit WHERE actor='admin'").fetchone()[0] == 1
    con.close()
