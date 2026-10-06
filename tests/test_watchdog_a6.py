# ruff: noqa: S101
"""A6: heartbeat, watchdog detectors and guard rails, health endpoint, metrics, alert isolation, and
the local dry rehearsal (a real child, a real SIGKILL, the real supervisor restart)."""

from __future__ import annotations

import importlib.util
import json
import os
import threading
import time
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient
from server_helpers import make_settings, scripted_executor

from distillery import chaos, heartbeat
from distillery.server import create_app
from distillery.server.settings import parse_watchdog
from distillery.server.watchdog import (
    AlertSink,
    Watchdog,
    WatchdogConfig,
    detect,
    prometheus_lines,
    read_heartbeat,
)
from distillery.server.worker import ChildInfo, Job
from distillery.store import Store

PID = 4242
T0 = 1_000_000.0
CFG = WatchdogConfig(
    hung_after_s=30, stall_after_s=100, grace_s=60, max_kills_per_stage=2, poll_s=0.05
)


def hb(
    *, at: float, progress_at: float, progress: int = 5, stage: str = "teacher_data",
    pid: int = PID, rss: float = 50.0,
) -> dict[str, Any]:  # fmt: skip
    return {
        "pid": pid, "started_at": T0, "at": at, "stage": stage, "progress": progress,
        "last_progress_at": progress_at, "rss_mb": rss, "attempt": 1,
    }  # fmt: skip


# ------------------------------------------------------------------ heartbeat


def test_bump_is_a_noop_without_an_active_heartbeat() -> None:
    before = heartbeat.snapshot()["progress"]
    heartbeat.bump("x")
    heartbeat.set_stage("y")
    assert heartbeat.snapshot()["progress"] == before


def test_heartbeat_file_counts_real_progress_only(tmp_path: Path) -> None:
    with heartbeat.running(tmp_path, {heartbeat.ENV_INTERVAL: "0.05", heartbeat.ENV_ATTEMPT: "3"}):
        first = json.loads((tmp_path / heartbeat.FILE).read_text())
        assert first["pid"] == os.getpid() and first["attempt"] == 3 and first["progress"] == 0
        time.sleep(0.3)  # several beats, no bumps: the thread must not count as progress
        assert json.loads((tmp_path / heartbeat.FILE).read_text())["progress"] == 0
        heartbeat.set_stage("teacher_data")
        heartbeat.bump("llm_call")
        time.sleep(0.2)
        later = json.loads((tmp_path / heartbeat.FILE).read_text())
        assert later["stage"] == "teacher_data" and later["progress"] == 2
        assert later["at"] > first["at"] and later["last_progress_at"] >= first["at"]
        assert set(later) >= {"pid", "started_at", "at", "stage", "progress", "rss_mb"}
    heartbeat.bump("after")  # inactive again
    assert json.loads((tmp_path / heartbeat.FILE).read_text())["progress"] == 2


def test_heartbeat_failure_never_fails_the_run(tmp_path: Path) -> None:
    blocker = tmp_path / "file"
    blocker.write_text("not a dir")
    with heartbeat.running(blocker / "sub", {heartbeat.ENV_INTERVAL: "0.05"}):
        heartbeat.bump("still fine")  # run_dir cannot be created: logged, the block still runs
    with heartbeat.running(tmp_path, {heartbeat.ENV_INTERVAL: "0"}) as off:
        assert off is None


# ------------------------------------------------------------------ detectors


def test_detect_each_kind() -> None:
    now = T0 + 1000
    ok = hb(at=now - 1, progress_at=now - 5)
    assert detect(ok, now=now, child_age_s=900, rss_mb=50, cfg=CFG) is None
    hung = detect(
        hb(at=now - 40, progress_at=now - 40), now=now, child_age_s=900, rss_mb=50, cfg=CFG
    )
    assert hung is not None and hung.kind == "hung" and hung.heartbeat_age_s == 40
    st = detect(hb(at=now - 1, progress_at=now - 150), now=now, child_age_s=900, rss_mb=50, cfg=CFG)
    assert st is not None and st.kind == "stalled" and st.progress_age_s == 150
    assert st.stage == "teacher_data" and st.rss_mb == 50
    nohb = detect(None, now=now, child_age_s=31, rss_mb=None, cfg=CFG)
    assert nohb is not None and nohb.kind == "no_heartbeat"
    assert detect(None, now=now, child_age_s=29, rss_mb=None, cfg=CFG) is None


def test_memory_pressure_needs_a_limit_and_ignores_grace() -> None:
    cfg = WatchdogConfig(memory_limit_mb=512, memory_kill_pct=92, grace_s=999)
    now = T0
    fresh = hb(at=now, progress_at=now)
    assert detect(fresh, now=now, child_age_s=1, rss_mb=470, cfg=cfg) is None  # 91.8 %
    d = detect(fresh, now=now, child_age_s=1, rss_mb=472, cfg=cfg)  # 92.2 %
    assert d is not None and d.kind == "memory_pressure" and d.rss_mb == 472
    assert detect(fresh, now=now, child_age_s=1, rss_mb=5000, cfg=WatchdogConfig()) is None  # off


def test_grace_period_covers_hung_and_stalled_only() -> None:
    now = T0 + 1000
    stale = hb(at=now - 500, progress_at=now - 500)
    assert detect(stale, now=now, child_age_s=59, rss_mb=1, cfg=CFG) is None
    assert detect(stale, now=now, child_age_s=61, rss_mb=1, cfg=CFG) is not None


def test_a_heartbeat_left_by_an_earlier_child_is_not_this_childs(tmp_path: Path) -> None:
    (tmp_path / heartbeat.FILE).write_text(json.dumps(hb(at=T0, progress_at=T0, pid=1)))
    assert read_heartbeat(tmp_path, pid=2) is None
    assert read_heartbeat(tmp_path, pid=1) is not None
    (tmp_path / heartbeat.FILE).write_text("{not json")
    assert read_heartbeat(tmp_path) is None


# ------------------------------------------------------------------ watchdog actions


class Rig:
    def __init__(self, tmp_path: Path, cfg: WatchdogConfig = CFG, **kw: Any) -> None:
        self.run_dir = tmp_path / "run"
        self.run_dir.mkdir()
        self.now = T0 + 1000.0
        self.rows: list[tuple[str, str, dict[str, Any]]] = []
        self.events: list[str] = []
        self.kills = 0
        self.job = Job("dry-chaos-w", "tiny", True)
        self.pid = PID
        self.cfg = cfg
        self.kw = kw
        self.make()

    def make(self) -> None:
        self.dog = Watchdog(
            self.cfg, lambda rid: self.run_dir, self._audit, clock=lambda: self.now,
            rss_of=lambda pid: 50.0, **self.kw,
        )  # fmt: skip

    def _audit(self, actor: str, action: str, rid: str, detail: dict[str, Any]) -> None:
        assert actor == "watchdog"
        self.rows.append((action, rid, detail))
        self.events.append(action)

    def info(self, age_s: float = 900.0) -> ChildInfo:
        def kill() -> None:
            self.kills += 1
            self.events.append("SIGKILL")

        return ChildInfo(self.pid, self.now - age_s, lambda: True, kill)

    def beat(self, progress_age: float, hb_age: float = 1.0, **kw: Any) -> None:
        (self.run_dir / heartbeat.FILE).write_text(
            json.dumps(
                hb(at=self.now - hb_age, progress_at=self.now - progress_age, pid=self.pid, **kw)
            )  # fmt: skip
        )

    def tick(self, age_s: float = 900.0) -> None:
        self.dog.tick(self.job, self.info(age_s), self.run_dir)

    def actions(self) -> list[str]:
        return [a for a, _, _ in self.rows]


def test_stalled_child_is_audited_then_sigkilled(tmp_path: Path) -> None:
    r = Rig(tmp_path)
    r.beat(progress_age=150)
    r.tick()
    assert r.actions() == ["incident_detected", "kill_child"]
    assert r.events == ["incident_detected", "kill_child", "SIGKILL"]  # audit BEFORE the kill
    det = r.rows[0][2]
    assert det["kind"] == "stalled" and det["stage"] == "teacher_data"
    assert det["progress_age_s"] == 150 and "heartbeat_age_s" in det and det["rss_mb"] == 50
    assert r.rows[1][2] == {"signal": "SIGKILL", "kind": "stalled", "stage": "teacher_data"}
    r.tick()  # the same child is not killed twice
    assert r.kills == 1 and r.dog.stats() == {"kills_total": 1, "incidents_total": {"stalled": 1}}


def test_incident_resolves_when_the_restarted_child_makes_progress(tmp_path: Path) -> None:
    r = Rig(tmp_path)
    r.beat(progress_age=150)
    r.tick()
    r.now += 10
    r.pid = PID + 1  # the supervisor restarted the child
    r.beat(progress_age=1, progress=1, stage="")  # before it reaches the stage: still open
    r.tick(age_s=5)
    assert "incident_resolved" not in r.actions()
    r.beat(progress_age=0, progress=3, stage="teacher_data")  # reached the stage
    r.tick(age_s=6)
    r.now += 4
    r.beat(progress_age=0, progress=9, stage="teacher_data")  # and moved on in it
    r.tick(age_s=10)
    resolved = [d for a, _, d in r.rows if a == "incident_resolved"]
    assert len(resolved) == 1 and resolved[0]["kind"] == "stalled"
    assert resolved[0]["recovery_s"] == pytest.approx(14.0)
    assert json.loads((r.run_dir / "watchdog_state.json").read_text())["open"] == []


def test_kill_cap_per_stage_then_escalation_only(tmp_path: Path) -> None:
    r = Rig(tmp_path)
    for attempt in range(2):  # two kills of the same stage across restarts
        r.pid = PID + attempt
        r.beat(progress_age=150)
        r.tick()
    assert r.kills == 2
    r.make()  # even a new watchdog (server restart) remembers: the count is on disk
    r.pid = PID + 2
    r.beat(progress_age=150)
    r.tick()
    r.tick()
    assert r.kills == 2  # no third kill
    assert r.actions().count("incident_escalated") == 1  # reported once, not on every tick
    esc = [d for a, _, d in r.rows if a == "incident_escalated"][0]
    assert esc["kills_for_stage"] == 2 and esc["cap"] == 2
    r.pid = PID + 3  # another stage is still protected
    r.beat(progress_age=150, stage="finetune_r1")
    r.tick()
    assert r.kills == 3


@pytest.mark.parametrize("flag", ["cancel_requested", "suspend_requested"])
def test_cancel_and_suspend_are_never_killed(tmp_path: Path, flag: str) -> None:
    r = Rig(tmp_path)
    setattr(r.job, flag, True)
    r.beat(progress_age=5000, hb_age=5000)
    r.tick()
    assert r.kills == 0 and r.rows == []


def test_worker_stopping_is_never_killed(tmp_path: Path) -> None:
    r = Rig(tmp_path, stopping=lambda: True)
    r.beat(progress_age=5000)
    r.tick()
    assert r.kills == 0 and r.rows == []


def test_a_cancel_between_detection_and_kill_wins(tmp_path: Path) -> None:
    r = Rig(tmp_path)
    r.beat(progress_age=150)
    real_audit = r._audit

    def audit_then_cancel(actor: str, action: str, rid: str, detail: dict[str, Any]) -> None:
        real_audit(actor, action, rid, detail)
        r.job.cancel_requested = True  # the user cancels right after the detection row

    r.dog._audit = audit_then_cancel
    r.tick()
    assert r.kills == 0 and r.actions() == ["incident_detected"]


def test_hung_and_no_heartbeat_and_memory_are_killed_too(tmp_path: Path) -> None:
    r = Rig(tmp_path)
    r.beat(progress_age=500, hb_age=500)
    r.tick()
    assert r.rows[0][2]["kind"] == "hung" and r.kills == 1
    (tmp_path / "b").mkdir()
    r2 = Rig(tmp_path / "b")
    r2.tick(age_s=31)  # no heartbeat file at all
    assert r2.rows[0][2]["kind"] == "no_heartbeat" and r2.kills == 1
    (tmp_path / "c").mkdir()
    r3 = Rig(tmp_path / "c", WatchdogConfig(memory_limit_mb=100, memory_kill_pct=50))
    r3.beat(progress_age=1)
    r3.tick(age_s=1)  # rss_of says 50 MB = 50 % of 100
    assert r3.rows[0][2]["kind"] == "memory_pressure" and r3.kills == 1


def test_disabled_watchdog_watches_nothing(tmp_path: Path) -> None:
    dog = Watchdog(WatchdogConfig(enabled=False), lambda rid: tmp_path, lambda *a: None)
    info = ChildInfo(1, T0, lambda: True, lambda: pytest.fail("killed"))
    stop = dog(Job("dry-x", "tiny", True), info)
    stop()
    assert parse_watchdog({"DISTILLERY_WATCHDOG": "0"}).enabled is False
    assert parse_watchdog({}).enabled is True


def test_the_watcher_thread_kills_a_real_stalled_child_via_run_hook(tmp_path: Path) -> None:
    r = Rig(tmp_path)
    r.beat(progress_age=150)
    stop = r.dog(r.job, r.info())
    deadline = time.time() + 5
    while r.kills == 0 and time.time() < deadline:
        time.sleep(0.02)
    stop()
    assert r.kills == 1


# ------------------------------------------------------------------ alerts


def test_webhook_failure_is_isolated_and_payload_has_no_secrets(tmp_path: Path) -> None:
    sent: list[dict[str, Any]] = []
    done = threading.Event()

    def post(url: str, body: bytes, timeout_s: float) -> None:
        sent.append({"url": url, "timeout": timeout_s, **json.loads(body)})
        done.set()
        raise OSError("webhook down")

    r = Rig(tmp_path, alerts=AlertSink("https://hooks.example/x", post))
    r.beat(progress_age=150)
    r.tick()  # must not raise even though every POST fails
    assert done.wait(2) and r.kills == 1
    deadline = time.time() + 2
    while len(sent) < 2 and time.time() < deadline:
        time.sleep(0.02)
    assert {s["event"] for s in sent} >= {"detected", "killed"}
    assert all(s["timeout"] == 5.0 and s["run_id"] == "dry-chaos-w" for s in sent)
    assert set(sent[0]) == {"url", "timeout", "event", "run_id", "kind", "stage", "at", "detail"}


def test_alert_sink_is_off_without_a_valid_url() -> None:
    for url in (None, "", "file:///etc/passwd"):
        s = AlertSink(url, lambda *a: pytest.fail("posted"))
        s.send("detected", "r", "k", "s", {})
        assert s.sent == 0
    with pytest.raises(ValueError, match="http"):
        parse_watchdog({"DISTILLERY_ALERT_WEBHOOK_URL": "ftp://x"})


def test_settings_parse_thresholds_and_refuse_typos() -> None:
    c = parse_watchdog(
        {
            "DISTILLERY_MEMORY_LIMIT_MB": "512",
            "DISTILLERY_WATCHDOG_STALL_AFTER_S": "99",
            "DISTILLERY_WATCHDOG_MAX_KILLS_PER_STAGE": "1",
        }
    )
    assert c.memory_limit_mb == 512 and c.stall_after_s == 99 and c.max_kills_per_stage == 1
    d = parse_watchdog({})
    assert (d.hung_after_s, d.stall_after_s, d.grace_s, d.max_kills_per_stage) == (
        180,
        2400,
        300,
        2,
    )
    assert d.memory_limit_mb is None and d.memory_kill_pct == 92
    with pytest.raises(ValueError, match="number"):
        parse_watchdog({"DISTILLERY_WATCHDOG_STALL_AFTER_S": "soon"})


# ------------------------------------------------------------------ chaos hang


def test_hang_plan_key_and_marker_make_it_once_per_run(tmp_path: Path) -> None:
    plan = chaos.ChaosPlan.parse('{"hang": {"stage": "teacher_data", "seconds": 0.3}}')
    assert plan.hang_stage == "teacher_data" and plan.hang_seconds == 0.3
    log: list[tuple[str, dict[str, Any]]] = []
    slept: list[float] = []
    audit = lambda a, d: log.append((a, d))  # noqa: E731
    chaos._hang(plan, tmp_path, "analysis", audit, slept.append)  # another stage: nothing
    assert log == [] and slept == []
    chaos._hang(plan, tmp_path, "teacher_data", audit, lambda s: slept.append(s))
    assert log[0][0] == "inject_hang" and log[0][1]["seconds"] == 0.3 and sum(slept) > 0
    n = len(slept)
    chaos._hang(plan, tmp_path, "teacher_data", audit, slept.append)  # the resume: marker is set
    assert len(log) == 1 and len(slept) == n
    assert chaos.plan_for("sql-gated-1", {chaos.ENV: '{"hang": {}}'}) is None  # not a chaos id


# ------------------------------------------------------------------ health + metrics


def _running_app(tmp_path: Path) -> tuple[TestClient, str, Path, threading.Event]:
    gate = threading.Event()
    settings = make_settings(
        tmp_path,
        executor=scripted_executor(tmp_path / "data", gate),
        reconcile_on_start=False,
        watchdog=CFG,
    )
    app = create_app(settings)
    c = TestClient(app)
    rid = "dry-sql-tiny-h1"
    assert (
        c.post("/api/runs", json={"scale": "tiny", "dry_run": True, "run_id": rid}).status_code
        == 202
    )
    run_dir = app.state.reader.store_for(rid).run_dir(rid)
    return c, rid, run_dir, gate


def test_health_endpoint_states_and_metrics(tmp_path: Path) -> None:
    c, rid, run_dir, gate = _running_app(tmp_path)
    with c:
        assert c.get("/api/runs/no-such-run-1/health").status_code == 404
        h = c.get(f"/api/runs/{rid}/health").json()
        assert h["state"] == "unknown" and h["restarts"] == 0 and h["open_incidents"] == []
        now = time.time()
        (run_dir / heartbeat.FILE).write_text(
            json.dumps(hb(at=now - 1, progress_at=now - 2, pid=os.getpid(), progress=7))
        )
        h = c.get(f"/api/runs/{rid}/health").json()
        assert h["state"] == "healthy" and h["stage"] == "teacher_data" and h["progress"] == 7
        assert h["heartbeat_age_s"] == pytest.approx(1, abs=2) and h["attempt"] == 1
        assert set(h) == {
            "run_id", "state", "stage", "heartbeat_age_s", "progress_age_s", "progress", "rss_mb",
            "attempt", "restarts", "watchdog_kills", "open_incidents", "recent_incidents",
        }  # fmt: skip
        (run_dir / heartbeat.FILE).write_text(
            json.dumps(hb(at=now - 1, progress_at=now - 500, pid=os.getpid()))
        )
        assert c.get(f"/api/runs/{rid}/health").json()["state"] == "stalled"
        (run_dir / heartbeat.FILE).write_text(
            json.dumps(hb(at=now - 500, progress_at=now - 500, pid=os.getpid()))
        )
        assert c.get(f"/api/runs/{rid}/health").json()["state"] == "hung"
        (run_dir / heartbeat.FILE).write_text(
            json.dumps(hb(at=now - 1, progress_at=now - 1, pid=os.getpid()))
        )
        store = Store(run_dir.parents[1])  # an audit row of the watchdog shows up as an incident
        store.add_audit("watchdog", "incident_detected", rid, {"kind": "stalled", "stage": "s"})
        store.close()
        h = c.get(f"/api/runs/{rid}/health").json()
        assert h["recent_incidents"][-1]["action"] == "incident_detected"
        assert "secret" not in json.dumps(h).lower()

        text = c.get("/api/metrics").text
        for name in (
            "distillery_run_heartbeat_age_seconds", "distillery_run_progress_age_seconds",
            "distillery_run_progress_total", "distillery_run_rss_mb",
            "distillery_run_restarts_total", "distillery_watchdog_kills_total",
            "distillery_incidents_total", "distillery_server_rss_mb",
            "distillery_http_requests_total",  # the existing ones are kept
        ):  # fmt: skip
            assert name in text, name
        assert f'distillery_run_progress_total{{run_id="{rid}"}} 5' in text
        tele = c.get("/api/telemetry").json()
        assert tele["run_health"][0]["run_id"] == rid and "kills_total" in tele["watchdog"]
        gate.set()
        deadline = time.time() + 10
        while (
            time.time() < deadline
            and c.get(f"/api/runs/{rid}/health").json()["state"] != "finished"
        ):
            time.sleep(0.05)
        assert c.get(f"/api/runs/{rid}/health").json()["state"] == "finished"


def test_prometheus_lines_without_runs_still_has_the_totals() -> None:
    text = "\n".join(prometheus_lines([], None, None))
    assert "distillery_watchdog_kills_total 0" in text
    assert 'distillery_incidents_total{kind="stalled"} 0' in text


# ------------------------------------------------------------------ the rehearsal


def test_local_rehearsal_hang_is_detected_killed_resumed_and_resolved(tmp_path: Path) -> None:
    spec = importlib.util.spec_from_file_location(
        "a6_watchdog_rehearsal",
        Path(__file__).parents[1] / "docs" / "proofs" / "a6_watchdog_rehearsal.py",
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    result = mod.rehearse(tmp_path, timeout_s=240.0)
    assert result["checks"] == dict.fromkeys(result["checks"], True), result["checks"]
    assert result["supervisor_restarts"] == 1


# ------------------------------------------------------------------ the progress hooks


def test_progress_hooks_bump_on_llm_calls_and_sandbox_jobs(tmp_path: Path) -> None:
    import asyncio

    from distillery.budget import Ledger
    from distillery.llm import CallRecord
    from distillery.orchestrator import LedgerSink
    from distillery.sandbox import FakeSandbox
    from distillery.sandbox import Job as SbJob

    with heartbeat.running(tmp_path, {heartbeat.ENV_INTERVAL: "60"}):
        p0 = heartbeat.snapshot()["progress"]
        sink = LedgerSink(Ledger("dry-x", 1.0, 10.0, 1.0))
        rec = CallRecord("teacher", "m", "p", "s", "h", 1, 1, 0.1, 0.0, 1, False, "boom")
        sink.record_call(rec)  # a failed attempt is a call that came back: progress too
        assert heartbeat.snapshot()["progress"] == p0 + 1
        sb = FakeSandbox()

        async def go() -> None:
            img = await sb.ensure_image("img")
            await sb.run_batch(img, [SbJob(shell="echo hi") for _ in range(3)])

        asyncio.run(go())
        snap = heartbeat.snapshot()
        assert snap["progress"] >= p0 + 4 and snap["last_kind"] in ("sandbox_job", "sandbox_branch")
