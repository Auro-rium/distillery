"""A report must never carry an absolute path of the machine that wrote it."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from distillery.orchestrator import Pipeline, Scale
from distillery.pipeline_fakes import build_dry_run
from distillery.sandbox_executor import AsyncBridge
from distillery.store import Store

NANO = Scale(name="nano", train=16, dev=8, heldout=12)
ROOT = Path(__file__).resolve().parents[1]


def _strings(obj: Any) -> list[str]:
    if isinstance(obj, str):
        return [obj]
    if isinstance(obj, dict):
        return [s for v in obj.values() for s in _strings(v)]
    if isinstance(obj, list):
        return [s for v in obj for s in _strings(v)]
    return []


def test_report_has_no_absolute_paths(tmp_path: Path) -> None:
    with AsyncBridge() as bridge:
        dr = build_dry_run(NANO, bridge)
        store = Store(tmp_path / "store")
        pipe = Pipeline(dr.pipeline_cfg, dr.config, dr.deps, store, "dry-rel", say=lambda _s: None)
        report = pipe.run()
        store.close()
    assert report["data"]["spot_check_file"] == "spot_check.json"
    bad = [s for s in _strings(report) if s.startswith("/") or str(tmp_path) in s]
    assert not bad, bad


def test_committed_fixtures_use_the_relative_name() -> None:
    for p in (
        ROOT / "docs/fixtures/sample-dry-run-report.json",
        ROOT / "frontend/src/testutil/contract/report.json",
    ):
        assert json.loads(p.read_text())["data"]["spot_check_file"] == "spot_check.json", p
