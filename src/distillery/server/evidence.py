"""Read-only helpers for /api/evidence and /api/runs/{id}/experiments."""

from __future__ import annotations

import json
import os
import sqlite3
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from distillery.server.settings import _REPO_ROOT

EVIDENCE_ENV = "DISTILLERY_EVIDENCE_FILE"
# Only these experiment rows are exposed: job/image/artifact bookkeeping, no sealed text.
EXPERIMENT_PREFIXES = ("finetune_job_", "serving_image", "expected_artifact")


def evidence_path(env: Mapping[str, str] | None = None) -> Path:
    """DISTILLERY_EVIDENCE_FILE, else deploy/evidence.json (cwd first, as in the image, then the
    repo checkout)."""
    e = os.environ if env is None else env
    explicit = e.get(EVIDENCE_ENV)
    if explicit:
        return Path(explicit)
    cwd = Path("deploy") / "evidence.json"
    return cwd if cwd.is_file() else _REPO_ROOT / "deploy" / "evidence.json"


def read_evidence(path: Path) -> dict[str, Any] | None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def store_experiments(index: Path, run_id: str) -> list[dict[str, Any]]:
    con = sqlite3.connect(f"file:{index}?mode=ro", uri=True)
    try:
        rows = con.execute(
            "SELECT name, data_json, created_at FROM experiments WHERE run_id=? ORDER BY id",
            (run_id,),
        ).fetchall()
    finally:
        con.close()
    out: list[dict[str, Any]] = []
    for name, raw, created in rows:
        if not str(name).startswith(EXPERIMENT_PREFIXES):
            continue
        try:
            data = json.loads(raw)
        except ValueError:
            continue
        out.append({"name": str(name), "data": data, "created_at": created})
    return out


def bundle_experiments(bundle_dir: Path) -> list[dict[str, Any]] | None:
    """Rows from an exported replay bundle's experiments.json, or None if it has none."""
    try:
        raw = json.loads((bundle_dir / "experiments.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(raw, list):
        return None
    out: list[dict[str, Any]] = []
    for r in raw:
        if isinstance(r, dict) and str(r.get("name", "")).startswith(EXPERIMENT_PREFIXES):
            row: dict[str, Any] = {"name": r["name"], "data": r.get("data")}
            if r.get("created_at") is not None:
                row["created_at"] = r["created_at"]
            out.append(row)
    return out
