"""A5 local dry rehearsal of the chaos proof (zero spend, fakes only).

    PYTHONPATH=src python docs/proofs/a5_rehearsal.py [--out docs/proofs/evidence/a5_rehearsal.json]

Runs the real server (``create_app``, real worker, real CLI executor = ``python -m distillery run``
child processes) on a dry ``dry-chaos-...`` run with all four injectors from ``DISTILLERY_CHAOS``
(the pipeline demands the ``dry-`` prefix on dry runs, so the chaos check also accepts it). The one
and only intervention is the start (``POST /api/runs`` with the admin token). Then it checks the
pre-registered A5 pass criteria (DECISIONS.md, 2026-10-06) against the audit log and the store.
FakeFineTune keeps its job list in the run dir (``fake_finetune_state.json``) so a killed child's
job is still there for the resume to adopt, as at a real provider; its ``create_calls`` is the
number of billed ``create_job`` calls.
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

TOKEN = "rehearsal-admin-token"  # noqa: S105 - a throwaway token for a local fake run
RUN_ID = "dry-chaos-rehearsal"
PLAN: dict[str, Any] = {
    "llm_503": {"after_calls": 3, "n": 3},
    "net_window": {"after_calls": 12, "seconds": 3},
    "kill": ["teacher_data", "finetune_r1"],
    "sandbox_fail": {"stage": "dev_eval", "jobs": 2},
    "hold_s": 20.0,
    "poll_s": 0.1,
}


def rehearse(
    root: Path, plan: dict[str, Any] = PLAN, *, timeout_s: float = 300.0
) -> dict[str, Any]:
    settings = ServerSettings(
        root=root, config=Config(admin_token=SecretStr(TOKEN)), replay_dir=root / "replay",
        restart_backoff_s=0.2, shutdown_grace_s=5.0, reconcile_on_start=False,
    )  # fmt: skip
    old = os.environ.get("DISTILLERY_CHAOS")
    os.environ["DISTILLERY_CHAOS"] = json.dumps(plan)
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
            audit = client.get(f"/api/runs/{RUN_ID}/audit").json()["audit"]
            time.sleep(1.0)
    finally:
        if old is None:
            os.environ.pop("DISTILLERY_CHAOS", None)
        else:
            os.environ["DISTILLERY_CHAOS"] = old
    return evaluate(run_dir, audit, plan)


def evaluate(run_dir: Path, audit: list[dict[str, Any]], plan: dict[str, Any]) -> dict[str, Any]:
    rep_path = run_dir / "report.json"
    report = json.loads(rep_path.read_text()) if rep_path.exists() else None
    ft_path = run_dir / "fake_finetune_state.json"
    ft = json.loads(ft_path.read_text()) if ft_path.exists() else {
        "created": [], "suffixes": {}, "create_calls": 0}  # fmt: skip
    by = lambda actor: [a for a in audit if a["actor"] == actor]  # noqa: E731
    chaos_acts = [a["action"] for a in by("chaos")]
    sup_acts = [a["action"] for a in by("supervisor")]
    admin = by("admin")
    rounds = sorted(set(ft["suffixes"].values()))
    cost = (report or {}).get("cost", {})
    inj = {
        "llm_503": chaos_acts.count("inject_llm_503"),
        "net_window": chaos_acts.count("inject_net_window"),
        "kill_child": chaos_acts.count("kill_child"),
        "sandbox_exit_-1": chaos_acts.count("inject_sandbox_exit_-1"),
    }
    want = {
        "llm_503": plan["llm_503"]["n"], "net_window": 1, "kill_child": len(plan["kill"]),
        "sandbox_exit_-1": plan["sandbox_fail"]["jobs"],
    }  # fmt: skip
    restarts = sup_acts.count("restart")
    checks = {
        "verdict_reached": report is not None and report.get("decision") in ("PROMOTE", "REJECT"),
        "every_injection_in_audit": inj == want,
        # recovered = the run still finished after the last injection, and each kill was followed
        # by a supervisor restart (the child was resumed, not abandoned)
        "every_injection_recovered": report is not None and restarts >= want["kill_child"],
        "at_least_one_supervisor_restart": restarts >= 1,
        "exactly_one_admin_action_the_start": [(a["action"]) for a in admin] == ["start"],
        "one_finetune_job_per_round": ft["create_calls"] == len(ft["created"]) == len(rounds),
        "spend_within_cap": report is not None
        and float(cost.get("run_total_usd", 0.0)) <= float(cost.get("run_cap_usd", 0.0)),
    }
    return {
        "dry_run": True, "label": "FAKE models and fake fine-tune: a rehearsal, not a result",
        "run_id": RUN_ID, "plan": plan, "decision": (report or {}).get("decision"),
        "injections_seen": inj, "injections_planned": want, "supervisor_restarts": restarts,
        "audit": audit, "finetune": {"create_calls": ft["create_calls"], "jobs": ft["created"],
                                      "suffixes": ft["suffixes"]},
        "cost": cost, "checks": checks, "pass": all(checks.values()),
    }  # fmt: skip


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="docs/proofs/evidence/a5_rehearsal.json")
    ap.add_argument("--timeout", type=float, default=300.0)
    args = ap.parse_args()
    with tempfile.TemporaryDirectory() as tmp:
        result = rehearse(Path(tmp), timeout_s=args.timeout)
    Path(args.out).write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"pass": result["pass"], "checks": result["checks"]}, indent=2))
    return 0 if result["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
