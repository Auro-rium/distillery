"""Read side: everything is DERIVED from the Store tables, the run dir and the worker's state."""

from __future__ import annotations

import json
import re
import sqlite3
import threading
from pathlib import Path
from typing import Any

from distillery.driver import driver_alive
from distillery.humanset import RUN_PREFIX as HUMANSET_RUN_PREFIX
from distillery.orchestrator import DRY_PREFIX, cost_by_model, stage_rows
from distillery.pipeline_fakes import DRY_RUN_CAP_USD
from distillery.server.settings import ServerSettings
from distillery.server.views import detail_from_report
from distillery.server.worker import Job, Worker
from distillery.store import Store

RUN_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")
RESERVED_IDS = frozenset({"playground"})
_STAGE_STATUS = {"complete": "done", "running": "running", "failed": "failed"}


def valid_run_id(run_id: str) -> bool:
    return (
        bool(RUN_ID_RE.fullmatch(run_id))
        and run_id not in RESERVED_IDS
        and not run_id.startswith(HUMANSET_RUN_PREFIX)  # a drafting ledger is not a run
    )


class Events:
    """Append-only event log of one run, derived by diffing successive snapshots."""

    def __init__(self) -> None:
        self.items: list[tuple[int, str, dict[str, Any]]] = []
        self.stages: dict[str, str] = {}
        self.spend_key: tuple[Any, ...] | None = None
        self.log_cursor = 0
        self.done = False

    def add(self, event: str, data: dict[str, Any]) -> None:
        self.items.append((len(self.items) + 1, event, data))


class RunReader:
    def __init__(self, settings: ServerSettings, worker: Worker | None = None) -> None:
        self.s = settings
        self.worker = worker
        self.real = Store(settings.root)
        self.dry = Store(settings.root / "dry-runs")
        self._events: dict[str, Events] = {}
        self._lock = threading.Lock()

    def _job(self, run_id: str) -> Job | None:
        return self.worker.get(run_id) if self.worker else None

    def reset_events(self, run_id: str) -> None:
        with self._lock:
            self._events.pop(run_id, None)

    def close(self) -> None:
        self.real.close()
        self.dry.close()

    def store_for(self, run_id: str) -> Store:
        return self.dry if run_id.startswith(DRY_PREFIX) else self.real

    # ---- files / tables ---------------------------------------------------
    def report_path(self, run_id: str) -> Path:
        return self.store_for(run_id).run_dir(run_id) / "report.json"

    def read_report(self, run_id: str) -> dict[str, Any] | None:
        path = self.report_path(run_id)
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        return data if isinstance(data, dict) else None

    def _table_runs(self, store: Store) -> list[tuple[str, str]]:
        con = sqlite3.connect(f"file:{store.root / 'index.sqlite'}?mode=ro", uri=True)
        try:
            rows = con.execute("SELECT run_id, created_at FROM runs").fetchall()
        finally:
            con.close()
        return [(str(r[0]), str(r[1])) for r in rows]

    def _finetune_estimate(self, store: Store, run_id: str) -> dict[str, float | None]:
        """Fine-tune spend: the legacy total plus billed-basis and ceiling rows kept apart."""
        con = sqlite3.connect(f"file:{store.root / 'index.sqlite'}?mode=ro", uri=True)
        try:
            rows = con.execute(
                "SELECT kind, COUNT(*), COALESCE(SUM(usd),0) FROM spend "
                "WHERE run_id=? AND kind IN ('finetune','finetune_ceiling') GROUP BY kind",
                (run_id,),
            ).fetchall()
        finally:
            con.close()
        by = {str(r[0]): float(r[2]) for r in rows if r[1]}
        return {
            "total": sum(by.values()) if by else None,
            "billed": by.get("finetune"),
            "ceiling": by.get("finetune_ceiling"),
        }

    def stage_output(self, run_id: str, stage: str) -> Any | None:
        store = self.store_for(run_id)
        for r in reversed(stage_rows(store, run_id)):
            if r["stage"] == stage and r["status"] == "complete" and r["output_sha256"]:
                try:
                    return json.loads(store.get_artifact(run_id, r["output_sha256"]))
                except (OSError, ValueError):
                    return None
        return None

    # ---- runs -------------------------------------------------------------
    def exists(self, run_id: str) -> bool:
        if self._job(run_id) is not None:
            return True
        return any(rid == run_id for rid, _ in self._table_runs(self.store_for(run_id)))

    def list_runs(self) -> list[dict[str, Any]]:
        seen: dict[str, str | None] = {}
        for store in (self.real, self.dry):
            for rid, created in self._table_runs(store):
                if valid_run_id(rid):
                    seen[rid] = created
        for job in self.worker.jobs() if self.worker else []:
            seen.setdefault(job.run_id, None)
        items = []
        for rid, created_at in seen.items():
            d = self.detail(rid)
            rep = self.read_report(rid)
            items.append(
                {
                    "run_id": rid,
                    "dry_run": d["dry_run"],
                    "recorded": False,
                    "recorded_at": None,
                    "status": d["status"],
                    "decision": rep.get("decision") if rep else None,
                    "created_at": created_at,
                }
            )
        items.sort(key=lambda i: i["created_at"] or "", reverse=True)
        return items

    def detail(self, run_id: str) -> dict[str, Any]:
        store = self.store_for(run_id)
        job = self._job(run_id)
        report = self.read_report(run_id)
        rows = stage_rows(store, run_id)
        latest: dict[str, dict[str, Any]] = {}
        for r in rows:  # oldest first; a resumed stage moves to the end
            latest.pop(r["stage"], None)
            latest[r["stage"]] = r
        stages = []
        for r in latest.values():
            status = _STAGE_STATUS.get(r["status"], "pending")
            stages.append(
                {
                    "name": r["stage"],
                    "status": status,
                    # The store keeps only the last update time; the start time is not recorded.
                    "started_at": r["updated_at"] if status == "running" else None,
                    "ended_at": r["updated_at"] if status in ("done", "failed") else None,
                }
            )
        error: str | None = None
        if job is not None and job.state != "finished":
            status = "pending" if job.state == "queued" else "running"
        elif report is None and driver_alive(store.run_dir(run_id)):
            status = "running"  # driven by the CLI in another process; it holds driver.pid
        elif report is not None:
            status = "complete"
        else:
            status = "failed"
            failed = next((r["error"] for r in latest.values() if r["status"] == "failed"), None)
            error = (
                (job.error if job else None)
                or failed
                or "run is not active and produced no report (interrupted?)"
            )
        dry = run_id.startswith(DRY_PREFIX)
        # a live dry run is capped by the fake pipeline, not by this server's real-run cap, so show
        # that one: the number must not change when the run completes and its report appears
        live_cap = DRY_RUN_CAP_USD if dry else self.s.config.run_cap_usd
        cap = ((report or {}).get("cost") or {}).get("run_cap_usd", live_cap)
        ops_done = [
            r for r in rows if r["stage"] == "verifier_selftest" and r["status"] == "complete"
        ]
        selftest = None
        if ops_done:
            out = self.stage_output(run_id, "verifier_selftest")
            c = (out or {}).get("counters") if isinstance(out, dict) else None
            if isinstance(c, dict):
                # A completed self-test stage has zero failures (any failure raises).
                selftest = {
                    "accepted_gold": c.get("tasks_sampled"),
                    "rejected_corruptions": c.get("corruptions_tested"),
                    "failures": 0,
                }
        ft = self._finetune_estimate(store, run_id)
        return {
            "run_id": run_id,
            "dry_run": dry,
            "recorded": False,
            "recorded_at": None,
            "status": status,
            "error": error,
            "stages": stages,
            "spend": {
                "total_usd": store.total_spend(run_id),
                "cap_usd": cap,
                "by_model": cost_by_model(store, run_id),
                "finetune_usd_estimate": ft["total"],
                "finetune_billed_usd": ft["billed"],
                "finetune_ceiling_usd": ft["ceiling"],
            },
            "sandbox": {"operations": None, "concurrency_peak": None},
            "verifier": {"language": "sql", "code": None, "selftest": selftest},
        }

    # ---- events -----------------------------------------------------------
    def events_after(
        self, run_id: str, last_id: int, detail: dict[str, Any], decision: str | None,
        job: Job | None,
    ) -> tuple[list[tuple[int, str, dict[str, Any]]], bool]:  # fmt: skip
        """Refresh the run's event log from ``detail`` and return (new events, finished)."""
        with self._lock:
            ev = self._events.setdefault(run_id, Events())
            # NOT event time: the moment this server first observed the change while polling.
            seen = self.s.now().isoformat()
            for st in detail["stages"]:
                if ev.stages.get(st["name"]) != st["status"]:
                    ev.stages[st["name"]] = st["status"]
                    # ``at`` is the stored store timestamp of that transition (start time while
                    # running, end time when done/failed); null if the store has none.
                    at = st["started_at"] or st["ended_at"]
                    ev.add(
                        "stage",
                        {"observed_at": seen, "at": at, "name": st["name"], "status": st["status"]},
                    )
            sp = detail["spend"]
            key = (sp["total_usd"], sum(int(m["calls"]) for m in sp["by_model"].values()))
            if key != ev.spend_key:
                ev.spend_key = key
                ev.add("spend", {"observed_at": seen, **sp})
            if job is not None:
                for line in job.logs[ev.log_cursor :]:
                    ev.add("log", {"observed_at": seen, "level": "info", "message": line})
                ev.log_cursor = len(job.logs)
            if detail["status"] in ("complete", "failed") and not ev.done:
                ev.done = True
                ev.add(
                    "done", {"observed_at": seen, "status": detail["status"], "decision": decision}
                )
            return [e for e in ev.items if e[0] > last_id], ev.done

    def replay_detail(
        self, run_id: str, report: dict[str, Any], recorded: bool, recorded_at: str | None
    ) -> dict[str, Any]:
        return detail_from_report(run_id, report, recorded=recorded, recorded_at=recorded_at)
