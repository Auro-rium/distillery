"""Replay bundles: ``replay/<run_id>/{report,run,events,tree}.json`` + ``manifest.json``.

Until a real bundle exists the default replay item is the labelled dry-run sample fixture.
"""

from __future__ import annotations

import json
import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from distillery.orchestrator import DRY_PREFIX
from distillery.server.evidence import store_experiments
from distillery.server.reader import RunReader, valid_run_id
from distillery.server.views import tree_from_report
from distillery.store import atomic_write_bytes

SAMPLE_ID = "sample-dry-run"


class ExportError(RuntimeError):
    pass


@dataclass
class Bundle:
    run_id: str
    report: dict[str, Any]
    recorded: bool
    recorded_at: str | None
    run: dict[str, Any] | None = None
    events: list[dict[str, Any]] | None = None
    tree: dict[str, Any] | None = None

    def item(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "dry_run": bool(self.report.get("dry_run")),
            "recorded": self.recorded,
            "recorded_at": self.recorded_at,
            "status": "complete",
            "decision": self.report.get("decision"),
            "created_at": self.recorded_at,
        }


def _load(path: Path) -> Any | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def load_bundles(replay_dir: Path, sample_report: Path | None) -> dict[str, Bundle]:
    out: dict[str, Bundle] = {}
    if replay_dir.is_dir():
        for d in sorted(replay_dir.iterdir()):
            report = _load(d / "report.json")
            if not d.is_dir() or not valid_run_id(d.name) or not isinstance(report, dict):
                continue
            manifest = _load(d / "manifest.json")
            recorded_at = manifest.get("recorded_at") if isinstance(manifest, dict) else None
            run, events, tree = (_load(d / f) for f in ("run.json", "events.json", "tree.json"))
            out[d.name] = Bundle(
                d.name,
                report,
                recorded=recorded_at is not None,
                recorded_at=recorded_at,
                run=run if isinstance(run, dict) else None,
                events=events if isinstance(events, list) else None,
                tree=tree if isinstance(tree, dict) else None,
            )
    has_real = any(not b.report.get("dry_run") for b in out.values())
    if not has_real and sample_report is not None:
        report = _load(sample_report)
        if isinstance(report, dict):
            report["dry_run"] = True  # the fixture is a dry run whatever the file says
            out.setdefault(SAMPLE_ID, Bundle(SAMPLE_ID, report, recorded=False, recorded_at=None))
    return out


_ABS_PATH = re.compile(
    r"(?:^|[\s\"'=(:])(?:/(?:home|tmp|Users|var|root|mnt|opt|usr|etc|srv)/|[A-Za-z]:\\)"
)


def _absolute_paths(obj: Any) -> list[str]:
    if isinstance(obj, str):
        return [obj[:120]] if _ABS_PATH.search(obj) else []
    if isinstance(obj, dict):
        return [s for v in obj.values() for s in _absolute_paths(v)]
    if isinstance(obj, list):
        return [s for v in obj for s in _absolute_paths(v)]
    return []


def _experiments(reader: RunReader, run_id: str) -> list[dict[str, Any]]:
    """Job/image/artifact rows, same filter and shape as GET /api/runs/{id}/experiments."""
    index = reader.store_for(run_id).root / "index.sqlite"
    try:
        return store_experiments(index, run_id)
    except sqlite3.Error:
        return []


def export_bundle(
    reader: RunReader, run_id: str, replay_dir: Path, *, allow_dry_run: bool, recorded_at: str
) -> Path:
    if not valid_run_id(run_id):
        raise ExportError(f"invalid run id {run_id!r}")
    if run_id.startswith(DRY_PREFIX) and not allow_dry_run:
        raise ExportError(
            f"{run_id} is a dry run (fake models); refusing to export it as a recording "
            "(pass --allow-dry-run to export it labelled as a dry run)"
        )
    report = reader.read_report(run_id)
    if report is None:
        raise ExportError(f"run {run_id} has no report.json (not finished?)")
    if bool(report.get("dry_run")) != run_id.startswith(DRY_PREFIX):
        raise ExportError(f"run {run_id}: report dry_run flag disagrees with the run id")
    data = report.get("data")
    if isinstance(data, dict) and isinstance(data.get("spot_check_file"), str):
        # older runs wrote the absolute path of the machine that ran them
        report = {**report, "data": {**data, "spot_check_file": Path(data["spot_check_file"]).name}}
    detail = reader.detail(run_id)
    detail["recorded"], detail["recorded_at"] = True, recorded_at
    events, _ = reader.events_after(run_id, 0, detail, report.get("decision"), None)
    manifest = _load(reader.store_for(run_id).run_dir(run_id) / "manifest.json") or {}
    manifest = {**manifest, "recorded_at": recorded_at, "source_run_id": run_id}
    dest = replay_dir / run_id
    files: dict[str, Any] = {
        "report.json": report,
        "manifest.json": manifest,
        "run.json": detail,
        "events.json": [{"id": i, "event": e, "data": d} for i, e, d in events],
        "tree.json": tree_from_report(report),
        "experiments.json": _experiments(reader, run_id),
    }
    leaks = [p for obj in files.values() for p in _absolute_paths(obj)]
    if leaks:
        raise ExportError(
            f"refusing to export {run_id}: the bundle contains an absolute path "
            f"(e.g. {leaks[0]!r}); bundles must carry relative paths only"
        )
    for name, obj in files.items():
        blob = json.dumps(obj, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
        atomic_write_bytes(dest / name, blob.encode("utf-8"))
    return dest
