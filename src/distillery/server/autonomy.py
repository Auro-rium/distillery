"""``autonomy.json``: what a run was submitted with, so a later server process can resume it.

Written when the API accepts a run (scale, budget, ceiling). ``state`` is ``active`` until the
worker gives the run up for good (exit 0/1/2, user cancel, restarts exhausted); a server shutdown
leaves it ``active`` on purpose, so the next server start re-submits the run (``resumable``) and
the resume adopts any paid job the suspended attempt left running.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from distillery.driver import driver_alive
from distillery.server.worker import Job
from distillery.store import Store, atomic_write_bytes

FILE = "autonomy.json"
SPEND_MARKER = "spend_approved.json"


def write(run_dir: Path, job: Job) -> None:
    data = {
        "run_id": job.run_id,
        "scale": job.scale,
        "pack": job.pack,
        "dry_run": job.dry_run,
        "budget_usd": job.budget_usd,
        "finetune_estimate_usd": job.finetune_estimate_usd,
        "state": "active",
        "submitted_at": datetime.now(UTC).isoformat(),
    }
    atomic_write_bytes(run_dir / FILE, json.dumps(data, indent=2, sort_keys=True).encode())


def read(run_dir: Path) -> dict[str, Any] | None:
    try:
        data = json.loads((run_dir / FILE).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def restore(run_dir: Path, previous: dict[str, Any] | None) -> None:
    """Undo ``write`` for a submission the worker refused: put back what was there, if anything."""
    if previous is None:
        (run_dir / FILE).unlink(missing_ok=True)
    else:
        atomic_write_bytes(run_dir / FILE, json.dumps(previous, indent=2, sort_keys=True).encode())


def mark_final(run_dir: Path, exit_code: int | None, error: str | None) -> None:
    data = read(run_dir)
    if data is None:
        return
    data.update(state="final", exit_code=exit_code, error=error,
                final_at=datetime.now(UTC).isoformat())  # fmt: skip
    atomic_write_bytes(run_dir / FILE, json.dumps(data, indent=2, sort_keys=True).encode())


def resumable(*stores: Store) -> Iterator[Job]:
    """Runs a previous server process left unfinished: ``autonomy.json`` still active, no
    report, nobody driving them (``driver.pid`` dead), and, for a spending run, the first-start
    approval on disk (the CLI refuses to resume without it anyway)."""
    for store in stores:
        runs = store.root / "runs"
        if not runs.is_dir():
            continue
        for run_dir in sorted(p for p in runs.iterdir() if p.is_dir()):
            data = read(run_dir)
            if data is None or data.get("state") != "active":
                continue
            if (run_dir / "report.json").exists() or driver_alive(run_dir):
                continue
            dry = bool(data.get("dry_run"))
            if not dry and not (run_dir / SPEND_MARKER).exists():
                continue
            yield Job(
                str(data["run_id"]), str(data["scale"]), dry,
                data.get("budget_usd"), data.get("finetune_estimate_usd"),
                str(data.get("pack", "sql")),
            )  # fmt: skip
