"""A6 local dry rehearsal of the run watchdog (zero spend, fakes only).

    PYTHONPATH=src python docs/proofs/a6_watchdog_rehearsal.py \
        [--out docs/proofs/evidence/a6_watchdog_rehearsal.json]

Same shape as ``a5_rehearsal.py``: the real server (``create_app``, real worker, real CLI executor =
``python -m distillery run`` child processes) on a dry ``dry-chaos-...`` run. The chaos plan makes
the child go silent inside one stage (``hang``: no progress, heartbeat thread still beating, so the
verdict must be ``stalled``, not ``hung``). The watchdog runs with rehearsal-sized thresholds
(STALL_AFTER_S=4, GRACE_S=0; the production defaults are 2400 / 300 and are only legitimate for a
live run). The one and only intervention is the start. The checks below are the A6 acceptance
criteria: detect, audit, SIGKILL, supervisor restart, verdict, ``incident_resolved``, one admin
action. FAKE models and fake fine-tune: a rehearsal of the mechanism, not a result.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

from fastapi.testclient import TestClient
from pydantic import SecretStr

from distillery.config import Config
from distillery.server import create_app
from distillery.server.settings import ServerSettings
from distillery.server.watchdog import WatchdogConfig

TOKEN = "rehearsal-admin-token"  # noqa: S105 - a throwaway token for a local fake run
RUN_ID = "dry-chaos-watchdog"
HANG_STAGE = "teacher_data"
PLAN: dict[str, Any] = {"hang": {"stage": HANG_STAGE, "seconds": 120.0}}
WATCHDOG = WatchdogConfig(
    hung_after_s=30.0, stall_after_s=4.0, grace_s=0.0, max_kills_per_stage=2, poll_s=0.25
)
ENV = {"DISTILLERY_HEARTBEAT_INTERVAL_S": "0.5"}


def rehearse(
    root: Path, plan: dict[str, Any] = PLAN, *, timeout_s: float = 300.0
) -> dict[str, Any]:
    settings = ServerSettings(
        root=root, config=Config(admin_token=SecretStr(TOKEN)), replay_dir=root / "replay",
        restart_backoff_s=0.2, shutdown_grace_s=5.0, reconcile_on_start=False, watchdog=WATCHDOG,
    )  # fmt: skip
    old = {k: os.environ.get(k) for k in ("DISTILLERY_CHAOS", *ENV)}
    os.environ["DISTILLERY_CHAOS"] = json.dumps(plan)
    os.environ.update(ENV)
    health: dict[str, Any] = {}
    try:
        app = create_app(settings)
        with TestClient(app) as client:
            r = client.post(
                "/api/runs",
                json={"run_id": RUN_ID, "scale": "tiny", "dry_run": True},
                headers={"x-admin-token": TOKEN},
            )
            r.raise_for_status()
            run_dir = root / "dry-runs" / "runs" / RUN_ID
            end = time.monotonic() + timeout_s
            while not (run_dir / "report.json").exists():
                job = app.state.worker.get(RUN_ID)
                if time.monotonic() > end or (job is not None and job.state == "finished"):
                    break
                time.sleep(0.5)
            time.sleep(1.5)  # the watcher's stop hook settles the incident when the child ends
            audit = client.get(f"/api/runs/{RUN_ID}/audit").json()["audit"]
            health = client.get(f"/api/runs/{RUN_ID}/health").json()
            metrics = client.get("/api/metrics").text
    finally:
        for k, v in old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
    return evaluate(run_dir, audit, plan, health, metrics)


def evaluate(
    run_dir: Path, audit: list[dict[str, Any]], plan: dict[str, Any], health: dict[str, Any],
    metrics: str,
) -> dict[str, Any]:  # fmt: skip
    rep_path = run_dir / "report.json"
    report = json.loads(rep_path.read_text()) if rep_path.exists() else None
    by = lambda actor: [a for a in audit if a["actor"] == actor]  # noqa: E731
    dog = [(a["action"], a["detail"]) for a in by("watchdog")]
    acts = [a for a, _ in dog]
    kills = [d for a, d in dog if a == "kill_child"]
    detected = [d for a, d in dog if a == "incident_detected"]
    sup = [a["action"] for a in by("supervisor")]
    chaos_acts = [a["action"] for a in by("chaos")]
    order = [a["action"] for a in audit if a["actor"] in ("chaos", "watchdog", "supervisor")]
    checks = {
        "hang_injected_and_audited_first": chaos_acts == ["inject_hang"],
        "detected_as_stalled_not_hung": bool(detected)
        and detected[0]["kind"] == "stalled"
        and detected[0]["stage"] == HANG_STAGE,
        "exactly_one_kill_sigkill": len(kills) == 1 and kills[0]["signal"] == "SIGKILL",
        "kill_audited_after_detection_before_restart": order.index("incident_detected")
        < order.index("kill_child")
        < order.index("restart")
        if {"incident_detected", "kill_child", "restart"} <= set(order) else False,
        "supervisor_restarted_once": sup.count("restart") == 1,
        "verdict_reached": report is not None and report.get("decision") in ("PROMOTE", "REJECT"),
        "incident_resolved_audited": acts.count("incident_resolved") == len(detected) >= 1,
        "no_escalation": "incident_escalated" not in acts,
        "exactly_one_admin_action_the_start": [a["action"] for a in by("admin")] == ["start"],
        "health_endpoint_finished": health.get("state") == "finished"
        and health.get("watchdog_kills") == 1 and health.get("open_incidents") == [],
        "metrics_exposed": "distillery_watchdog_kills_total 1" in metrics
        and 'distillery_incidents_total{kind="stalled"} 1' in metrics,
    }
    return {
        "dry_run": True, "label": "FAKE models and fake fine-tune: a rehearsal, not a result",
        "run_id": RUN_ID, "plan": plan,
        "watchdog_thresholds": {
            "stall_after_s": WATCHDOG.stall_after_s, "grace_s": WATCHDOG.grace_s,
            "hung_after_s": WATCHDOG.hung_after_s, "heartbeat_interval_s": 0.5,
            "note": "rehearsal-sized; production defaults are stall 2400 s, grace 300 s",
        },
        "decision": (report or {}).get("decision"), "supervisor_restarts": sup.count("restart"),
        "audit": audit, "health": health, "checks": checks, "pass": all(checks.values()),
    }  # fmt: skip


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="docs/proofs/evidence/a6_watchdog_rehearsal.json")
    ap.add_argument("--timeout", type=float, default=300.0)
    args = ap.parse_args()
    with tempfile.TemporaryDirectory() as tmp:
        result = rehearse(Path(tmp), timeout_s=args.timeout)
    Path(args.out).write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"pass": result["pass"], "checks": result["checks"]}, indent=2))
    return 0 if result["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
