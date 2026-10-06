"""Run watchdog: detects a wedged pipeline child and resolves it, without threatening autonomy (A6).

One watcher thread per running child (plugged into ``run_child(watch=...)``) reads the child's
``heartbeat.json`` (``distillery.heartbeat``) and its liveness, and recognises four failures:

* ``hung``: the child is alive but the heartbeat file has not been rewritten for ``hung_after_s``;
* ``stalled``: the heartbeat is fresh but ``progress`` has not moved for ``stall_after_s``;
* ``memory_pressure``: the child's RSS passed ``memory_kill_pct`` of ``memory_limit_mb`` (killed
  before the kernel OOM-kills the whole instance, server included);
* ``no_heartbeat``: the child has been alive ``hung_after_s`` and never wrote a heartbeat.

Resolution is brutal but safe: SIGKILL of the child's process group, and nothing else. Never SIGINT
(a user cancel: paid fine-tune jobs would be cancelled) and never SIGTERM (a suspend: the supervisor
would NOT restart). The supervisor sees a negative exit code and restarts the same argv, which is a
resume from the stage cache with paid-job adoption (proven, plan A1/A5). Kills therefore also count
toward ``MAX_RESTARTS``: a run that wedges every time ends as ``restarts_exhausted``, never loops.

Guard rails, so the watchdog cannot itself break autonomy:

* no ``hung``/``stalled`` verdict inside ``grace_s`` of a child's start (imports, dep image builds);
* at most ``max_kills_per_stage`` kills for one stage across all restarts of a run (persisted in
  ``watchdog_state.json``); after that it only escalates (an audited incident + alert), never kills;
* it does nothing while ``job.cancel_requested`` / ``job.suspend_requested`` or the worker is
  stopping, and re-checks that right before the kill;
* the audit row is written BEFORE the kill; an audit or alert failure never stops the watcher.

``STALL_AFTER_S`` default (2400 s) must exceed the longest LEGITIMATE silence, which is one sandbox
operation that bumps progress only when it returns: ``SandboxCpuStudent`` / ``DepsImage`` setup
timeouts are 1800 s (+ 5 s backstop grace + upload), and a generation job is capped at the same
1800 s. A fine-tune poll bumps every ``poll_interval_s`` (15 s), an LLM call is bounded by the
client timeout, a SQL batch by 2 s x statements + 30 s. 2400 s = 1800 s + a 10 minute margin; the
price of that margin is that a real stall costs up to 40 minutes before it is cut.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
import urllib.request
from collections import Counter
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from distillery import heartbeat
from distillery.server.worker import ChildInfo, Job
from distillery.store import atomic_write_bytes

log = logging.getLogger("distillery.watchdog")
STATE_FILE = "watchdog_state.json"
ALERT_TIMEOUT_S = 5.0

AuditFn = Callable[[str, str, str, dict[str, Any]], None]  # actor, action, run_id, detail
PostFn = Callable[[str, bytes, float], None]  # url, body, timeout_s


@dataclass(frozen=True)
class WatchdogConfig:
    enabled: bool = True
    hung_after_s: float = 180.0  # 18 missed beats at the default 10 s interval
    stall_after_s: float = 2400.0  # see the module docstring
    grace_s: float = 300.0  # after a child's start: no hung/stalled verdicts
    max_kills_per_stage: int = 2  # then escalate only
    memory_limit_mb: float | None = None  # None: the memory check is off (Render free = 512)
    memory_kill_pct: float = 92.0
    poll_s: float = 5.0
    alert_webhook_url: str | None = None


def setup_logging() -> None:
    """Incident lines to stderr as bare JSON (idempotent; leaves other loggers alone)."""
    if log.handlers:
        return
    h = logging.StreamHandler()
    h.setFormatter(logging.Formatter("%(message)s"))
    log.addHandler(h)
    log.setLevel(logging.INFO)
    log.propagate = False


def _post_json(url: str, body: bytes, timeout_s: float) -> None:
    req = urllib.request.Request(  # noqa: S310 - scheme checked by AlertSink
        url, data=body, headers={"Content-Type": "application/json"}, method="POST"
    )
    with urllib.request.urlopen(req, timeout=timeout_s):  # noqa: S310  # nosec B310
        pass


class AlertSink:
    """Optional webhook: one small JSON POST per event, fire-and-forget on a thread. A failure is
    logged and swallowed; the body is built from the fields below only (no env, no secrets)."""

    def __init__(
        self, url: str | None, post: PostFn = _post_json,
        now: Callable[[], datetime] = lambda: datetime.now(UTC),
    ) -> None:  # fmt: skip
        ok = bool(url) and str(url).startswith(("http://", "https://"))
        self._url, self._post, self._now = (url if ok else None), post, now
        self.sent = 0

    def send(
        self, event: str, run_id: str, kind: str, stage: str, detail: Mapping[str, Any]
    ) -> None:
        if self._url is None:
            return
        body = json.dumps(
            {"event": event, "run_id": run_id, "kind": kind, "stage": stage,
             "at": self._now().isoformat(), "detail": dict(detail)},
            sort_keys=True, default=str,
        ).encode()  # fmt: skip
        self.sent += 1
        threading.Thread(
            target=self._deliver, args=(self._url, body), name="distillery-alert", daemon=True
        ).start()

    def _deliver(self, url: str, body: bytes) -> None:
        try:
            self._post(url, body, ALERT_TIMEOUT_S)
        except Exception as exc:  # noqa: BLE001 - an alert must never break the watchdog
            log.warning(json.dumps({"evt": "alert_failed", "error": type(exc).__name__}))


@dataclass(frozen=True)
class Detection:
    kind: str  # hung | stalled | memory_pressure | no_heartbeat
    stage: str
    heartbeat_age_s: float | None
    progress_age_s: float | None
    rss_mb: float | None

    def detail(self) -> dict[str, Any]:
        return {
            "kind": self.kind, "stage": self.stage, "heartbeat_age_s": self.heartbeat_age_s,
            "progress_age_s": self.progress_age_s, "rss_mb": self.rss_mb,
        }  # fmt: skip


def read_heartbeat(run_dir: Path, pid: int | None = None) -> dict[str, Any] | None:
    """The heartbeat of this child (``pid``), or None: missing, unreadable, malformed, or left by
    an earlier attempt (a SIGKILLed child's last file must not be taken for the new child's)."""
    try:
        data = json.loads((run_dir / heartbeat.FILE).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(data, dict) or not isinstance(data.get("at"), (int, float)):
        return None
    if pid is not None and data.get("pid") != pid:
        return None
    return data


def detect(
    hb: Mapping[str, Any] | None, *, now: float, child_age_s: float, rss_mb: float | None,
    cfg: WatchdogConfig,
) -> Detection | None:  # fmt: skip
    """Pure verdict from one observation. Memory first (no grace: the OOM killer does not wait)."""
    stage = str(hb.get("stage", "")) if hb else ""
    hb_age = None if hb is None else max(0.0, now - float(hb["at"]))
    pr_age = None if hb is None else max(0.0, now - float(hb.get("last_progress_at") or hb["at"]))
    if (
        cfg.memory_limit_mb
        and rss_mb is not None
        and rss_mb >= cfg.memory_limit_mb * cfg.memory_kill_pct / 100.0
    ):
        return Detection("memory_pressure", stage, hb_age, pr_age, rss_mb)
    rss = rss_mb if rss_mb is not None else (hb or {}).get("rss_mb")
    if hb is None:
        if child_age_s > cfg.hung_after_s:
            return Detection("no_heartbeat", "", None, None, rss)
        return None
    if child_age_s < cfg.grace_s:
        return None
    if hb_age is None or pr_age is None:  # unreachable (hb is set here); narrows for mypy
        return None
    if hb_age > cfg.hung_after_s:
        return Detection("hung", stage, hb_age, pr_age, rss)
    if pr_age > cfg.stall_after_s:
        return Detection("stalled", stage, hb_age, pr_age, rss)
    return None


class _StateFile:
    """``watchdog_state.json``: kill counts that survive restarts of the child (and the server)."""

    def __init__(self, run_dir: Path) -> None:
        self._path = run_dir / STATE_FILE

    def read(self) -> dict[str, Any]:
        try:
            data = json.loads(self._path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            data = {}
        data = data if isinstance(data, dict) else {}
        data.setdefault("kills", {})
        data.setdefault("kills_total", 0)
        data.setdefault("open", [])
        return data

    def write(self, data: dict[str, Any]) -> None:
        atomic_write_bytes(self._path, json.dumps(data, indent=2, sort_keys=True).encode())


class Watchdog:
    """Factory of per-child watchers; ``__call__(job, info)`` is the ``WatchHook``."""

    def __init__(
        self,
        cfg: WatchdogConfig,
        run_dir: Callable[[str], Path],
        audit: AuditFn,
        *,
        stopping: Callable[[], bool] = lambda: False,
        alerts: AlertSink | None = None,
        clock: Callable[[], float] = time.time,
        rss_of: Callable[[int], float | None] = heartbeat.rss_mb,
    ) -> None:
        self.cfg, self._run_dir, self._audit, self._stopping = cfg, run_dir, audit, stopping
        self.alerts = alerts or AlertSink(cfg.alert_webhook_url)
        self._clock, self._rss_of = clock, rss_of
        self._lock = threading.Lock()
        self.incidents_total: Counter[str] = Counter()
        self.kills_total = 0
        setup_logging()

    # ---- emit: audit + structured log + alert, each failure-isolated ----------------------
    def _audit_safe(self, action: str, run_id: str, detail: dict[str, Any]) -> None:
        try:
            self._audit("watchdog", action, run_id, detail)
        except Exception as exc:  # noqa: BLE001 - the audit log must never stop the watchdog
            log.warning(json.dumps({"evt": "audit_failed", "action": action,
                                    "error": type(exc).__name__}))  # fmt: skip

    def emit(
        self, event: str, run_id: str, kind: str, stage: str, detail: Mapping[str, Any]
    ) -> None:
        """Structured JSON log line for every incident event, then the optional webhook."""
        try:
            log.info(json.dumps(
                {"evt": f"watchdog_{event}", "run_id": run_id, "kind": kind, "stage": stage,
                 **dict(detail)}, sort_keys=True, default=str,
            ))  # fmt: skip
            self.alerts.send(event, run_id, kind, stage, detail)
        except Exception:  # noqa: BLE001, S110 - observability must not break a run
            pass

    # ---- one watcher per child -------------------------------------------------------------
    def __call__(self, job: Job, info: ChildInfo) -> Callable[[], None]:
        if not self.cfg.enabled:
            return lambda: None
        stop = threading.Event()
        run_dir = self._run_dir(job.run_id)

        def loop() -> None:
            while not stop.wait(self.cfg.poll_s):
                try:
                    self.tick(job, info, run_dir)
                except Exception as exc:  # noqa: BLE001 - the watcher itself must not die
                    log.warning(json.dumps({"evt": "watchdog_tick_failed", "run_id": job.run_id,
                                            "error": type(exc).__name__}))  # fmt: skip

        t = threading.Thread(target=loop, name=f"watchdog-{job.run_id}", daemon=True)
        t.start()

        def stop_watch() -> None:
            stop.set()
            try:  # the child ended: a clean finish settles whatever incident was still open
                if (run_dir / "report.json").exists():
                    self._resolve_all(job.run_id, run_dir, "run_finished")
            except Exception:  # noqa: BLE001, S110
                pass

        return stop_watch

    def _blocked(self, job: Job) -> bool:
        return job.cancel_requested or job.suspend_requested or self._stopping()

    def tick(self, job: Job, info: ChildInfo, run_dir: Path) -> None:
        """One observation of one child; public so tests can drive it with a fake clock."""
        if self._blocked(job) or not info.alive():
            return
        now = self._clock()
        hb = read_heartbeat(run_dir, info.pid)
        with self._lock:
            state_file = _StateFile(run_dir)
            state = state_file.read()
            self._maybe_resolve(job.run_id, state, hb, info.pid, now)
            state_file.write(state)
        det = detect(
            hb,
            now=now,
            child_age_s=now - info.started_at,
            rss_mb=self._rss_of(info.pid),
            cfg=self.cfg,
        )
        if det is None:
            return
        with self._lock:
            state = state_file.read()
            if any(i.get("pid") == info.pid for i in state["open"]):
                return  # this child's incident is already handled (killed, or escalated)
            self._handle(job, info, state_file, state, det, hb, now)

    def _handle(
        self, job: Job, info: ChildInfo, state_file: _StateFile, state: dict[str, Any],
        det: Detection, hb: Mapping[str, Any] | None, now: float,
    ) -> None:  # fmt: skip
        rid, stage_key = job.run_id, det.stage or "-"
        incident = {
            "kind": det.kind, "stage": det.stage, "pid": info.pid, "detected_at": now,
            "progress": int(hb.get("progress", 0)) if hb else 0, "reached": False,
            "killed": False, "escalated": False,
        }  # fmt: skip
        state["open"].append(incident)
        self.incidents_total[det.kind] += 1
        self._audit_safe("incident_detected", rid, det.detail())
        self.emit("detected", rid, det.kind, det.stage, det.detail())
        kills = int(state["kills"].get(stage_key, 0))
        if kills >= self.cfg.max_kills_per_stage:
            incident["escalated"] = True
            d = {**det.detail(), "kills_for_stage": kills, "cap": self.cfg.max_kills_per_stage}
            self._audit_safe("incident_escalated", rid, d)
            self.emit("escalated", rid, det.kind, det.stage, d)
            state_file.write(state)
            return
        if self._blocked(job):  # a cancel or shutdown arrived since the observation
            state["open"].remove(incident)
            state_file.write(state)
            return
        state["kills"][stage_key] = kills + 1
        state["kills_total"] = int(state["kills_total"]) + 1
        incident["killed"] = True
        state_file.write(state)  # count first: a crash between here and the kill never double-kills
        kd = {"signal": "SIGKILL", "kind": det.kind, "stage": det.stage}
        self._audit_safe("kill_child", rid, kd)  # audit row BEFORE the kill
        self.emit("killed", rid, det.kind, det.stage, kd)
        self.kills_total += 1
        info.kill()

    def _maybe_resolve(
        self, run_id: str, state: dict[str, Any], hb: Mapping[str, Any] | None, pid: int,
        now: float,
    ) -> None:  # fmt: skip
        """An incident is resolved when progress advances: for a killed child, once the restarted
        child has reached the stage it died in and moved past its entry progress; for an escalated
        one (never killed), when the same child makes progress again."""
        if hb is None:
            return
        keep: list[dict[str, Any]] = []
        for inc in state["open"]:
            done = False
            if inc["pid"] == pid:
                done = int(hb.get("progress", 0)) > int(inc["progress"]) and not inc["killed"]
            else:  # a restarted child
                if not inc["reached"] and (not inc["stage"] or hb.get("stage") == inc["stage"]):
                    inc["reached"], inc["progress"] = True, int(hb.get("progress", 0))
                elif inc["reached"]:
                    done = (
                        int(hb.get("progress", 0)) > int(inc["progress"])
                        or hb.get("stage") != inc["stage"]
                    )
            if done:
                d = {
                    "kind": inc["kind"], "stage": inc["stage"],
                    "recovery_s": round(now - float(inc["detected_at"]), 1),
                }  # fmt: skip
                self._audit_safe("incident_resolved", run_id, d)
                self.emit("resolved", run_id, inc["kind"], inc["stage"], d)
            else:
                keep.append(inc)
        state["open"] = keep

    def _resolve_all(self, run_id: str, run_dir: Path, how: str) -> None:
        with self._lock:
            sf = _StateFile(run_dir)
            state = sf.read()
            for inc in state["open"]:
                d = {
                    "kind": inc["kind"], "stage": inc["stage"], "how": how,
                    "recovery_s": round(self._clock() - float(inc["detected_at"]), 1),
                }  # fmt: skip
                self._audit_safe("incident_resolved", run_id, d)
                self.emit("resolved", run_id, inc["kind"], inc["stage"], d)
            if state["open"]:
                state["open"] = []
                sf.write(state)

    # ---- read side: health and metrics -----------------------------------------------------
    def stats(self) -> dict[str, Any]:
        return {"kills_total": self.kills_total, "incidents_total": dict(self.incidents_total)}


def _pid_alive(pid: Any) -> bool:
    if not isinstance(pid, int) or pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def run_health(
    run_id: str, run_dir: Path, job: Job | None, cfg: WatchdogConfig, *, now: float | None = None
) -> dict[str, Any]:
    """Everything the health endpoint and the metrics say about one run, from files + the job."""
    now = time.time() if now is None else now
    hb = read_heartbeat(run_dir)
    alive = hb is not None and _pid_alive(hb.get("pid"))
    hb_age = None if hb is None else round(max(0.0, now - float(hb["at"])), 1)
    pr_age = (
        None if hb is None
        else round(max(0.0, now - float(hb.get("last_progress_at") or hb["at"])), 1)
    )  # fmt: skip
    rss = heartbeat.rss_mb(int(hb["pid"])) if alive and hb is not None else None
    if rss is None and hb is not None:
        rss = hb.get("rss_mb")
    st = _StateFile(run_dir).read()
    mem_pct = (
        100.0 * float(rss) / cfg.memory_limit_mb if cfg.memory_limit_mb and rss is not None else 0.0
    )
    open_inc = [
        {k: i[k] for k in ("kind", "stage", "detected_at", "killed", "escalated")}
        for i in st["open"]
    ]
    if (run_dir / "report.json").exists():
        state = "finished"
    elif job is not None and job.state == "finished":
        state = "dead" if job.error else "finished"
    elif job is not None and job.state == "queued":
        state = "unknown"
    elif hb is None:
        state = "unknown"
    elif not alive:
        state = "degraded" if job is not None else "dead"  # a running job restarts its child
    elif hb_age is not None and hb_age > cfg.hung_after_s:
        state = "hung"
    elif pr_age is not None and pr_age > cfg.stall_after_s:
        state = "stalled"
    elif open_inc or mem_pct >= 80.0 or (pr_age or 0.0) > cfg.stall_after_s / 2:
        state = "degraded"
    else:
        state = "healthy"
    return {
        "run_id": run_id, "state": state, "stage": hb.get("stage") if hb else None,
        "heartbeat_age_s": hb_age, "progress_age_s": pr_age,
        "progress": hb.get("progress") if hb else None, "rss_mb": rss,
        "attempt": hb.get("attempt") if hb else None,
        "restarts": job.restarts if job is not None else 0,
        "watchdog_kills": int(st["kills_total"]), "open_incidents": open_inc,
    }  # fmt: skip


def recent_incidents(audit_rows: list[dict[str, Any]], n: int = 20) -> list[dict[str, Any]]:
    rows = [r for r in audit_rows if r.get("actor") == "watchdog"]
    return [
        {"at": r["at"], "action": r["action"], **r["detail"]} for r in rows[-n:]
    ]  # fmt: skip


def prometheus_lines(
    runs: list[dict[str, Any]], dog: Watchdog | None, server_rss_mb: float | None
) -> list[str]:
    """Metric lines for the runs the worker knows (a bounded label set), then the totals."""

    def esc(v: str) -> str:
        return v.replace("\\", "\\\\").replace('"', '\\"')

    spec = (
        ("distillery_run_heartbeat_age_seconds", "heartbeat_age_s", "gauge"),
        ("distillery_run_progress_age_seconds", "progress_age_s", "gauge"),
        ("distillery_run_progress_total", "progress", "counter"),
        ("distillery_run_rss_mb", "rss_mb", "gauge"),
        ("distillery_run_restarts_total", "restarts", "counter"),
    )
    out: list[str] = []
    for name, key, typ in spec:
        out.append(f"# TYPE {name} {typ}")
        for r in runs:
            if r.get(key) is not None:
                out.append(f'{name}{{run_id="{esc(str(r["run_id"]))}"}} {r[key]}')
    stats: dict[str, Any] = (
        dog.stats() if dog is not None else {"kills_total": 0, "incidents_total": {}}
    )
    out += [
        "# TYPE distillery_watchdog_kills_total counter",
        f"distillery_watchdog_kills_total {stats['kills_total']}",
        "# TYPE distillery_incidents_total counter",
    ]
    for kind in ("hung", "stalled", "memory_pressure", "no_heartbeat"):
        out.append(
            f'distillery_incidents_total{{kind="{kind}"}} {stats["incidents_total"].get(kind, 0)}'
        )
    out.append("# TYPE distillery_server_rss_mb gauge")
    if server_rss_mb is not None:
        out.append(f"distillery_server_rss_mb {server_rss_mb}")
    return out


__all__ = [
    "AlertSink", "Detection", "Watchdog", "WatchdogConfig", "detect", "prometheus_lines",
    "read_heartbeat", "recent_incidents", "run_health",
]  # fmt: skip
