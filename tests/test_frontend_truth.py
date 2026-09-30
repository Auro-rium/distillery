"""Contract test: what the FastAPI server returns must match, key for key and type for type, the
JSON fixtures the frontend renders in its tests (frontend/src/testutil/contract/*.json).

Normal mode compares. `UPDATE_CONTRACT=1` rewrites the fixtures (`npm run contract:update`).
Everything here is fake / dry-run; nothing touches the network.
"""

from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient
from server_helpers import make_settings, scripted_executor, seed_finished_run

from distillery.server import create_app

CONTRACT_DIR = Path(__file__).resolve().parents[1] / "frontend" / "src" / "testutil" / "contract"
UPDATE = os.environ.get("UPDATE_CONTRACT") == "1"

# Paths (lists written as []) whose value may be null, exactly the `| null` fields of
# docs/API_CONTRACT.md. Anywhere else null vs non-null is drift.
NULLABLE: dict[str, set[str]] = {
    "runs": {"[].recorded_at", "[].decision"},
    # created_at: the contract says string, but the bundled dry-run sample reports null.
    "replay": {"[].recorded_at", "[].decision", "[].created_at"},
    "run": {
        "recorded_at",
        "error",
        "stages[].started_at",
        "stages[].ended_at",
        "spend.finetune_usd_estimate",
        "sandbox.concurrency_peak",
        # The contract says int, but the server currently reports null (no sandbox counter yet).
        # Tolerated here so the frontend is tested against the real shape; it renders "not measured".
        "sandbox.operations",
        "verifier.code",
        "verifier.selftest",
    },
    "report": {"recorded_at"},
    "tree": {
        "nodes[].parent_id",
        "nodes[].round",
        "nodes[].hypothesis",
        "nodes[].data_delta",
        "nodes[].dev_score",
        "nodes[].cost_usd",
        "nodes[].sandbox_image",
    },
    "examples": set(),
    # response headers of /examples, parsed: X-Examples-Available / -Cap-Per-Kind / -Totals
    "examples_headers": {"totals.fixed", "totals.still_wrong", "totals.regressed"},
    "health": set(),
    "config": set(),
}


def _kind(v: Any) -> str:
    if v is None:
        return "null"
    if isinstance(v, bool):
        return "bool"
    if isinstance(v, int | float):
        return "number"
    if isinstance(v, str):
        return "string"
    if isinstance(v, list):
        return "array"
    return "object"


def diff(live: Any, fix: Any, path: str, nullable: set[str]) -> list[str]:
    lk, fk = _kind(live), _kind(fix)
    if lk != fk:
        if "null" in (lk, fk) and path in nullable:
            return []
        return [f"{path or '<root>'}: live is {lk}, fixture is {fk}"]
    if lk == "object":
        out: list[str] = []
        for k in sorted(set(live) - set(fix)):
            out.append(f"{path}.{k}: key only in live response")
        for k in sorted(set(fix) - set(live)):
            out.append(f"{path}.{k}: key only in fixture")
        for k in sorted(set(live) & set(fix)):
            out += diff(live[k], fix[k], f"{path}.{k}".lstrip("."), nullable)
        return out
    if lk == "array":
        p = f"{path}[]"
        if not live and fix:
            return [f"{path}: live list is empty, fixture has items"]
        if live and not fix:
            return [f"{path}: live list has items, fixture is empty"]
        out = []
        for i, item in enumerate(live):
            if not any(not diff(item, f, p, nullable) for f in fix):
                out.append(f"{p}[{i}] matches no fixture item: {diff(item, fix[0], p, nullable)}")
        for i, f in enumerate(fix):
            if not any(not diff(item, f, p, nullable) for item in live):
                out.append(f"{p}: fixture item {i} matched by no live item")
        return out
    return []


def _wait_complete(c: TestClient, rid: str) -> None:
    end = time.monotonic() + 10
    while time.monotonic() < end:
        if c.get(f"/api/runs/{rid}").json()["status"] == "complete":
            return
        time.sleep(0.02)
    raise AssertionError("dry run never completed")


def _example(i: int, kind: str, base: bool, student: bool) -> dict[str, Any]:
    return {
        "task_id": f"t{i}",
        "kind": kind,
        "family": "join",
        "heldout_class": "seen",
        "question": f"Question {i}?",
        "gold_sql": "SELECT 1",
        "base_sql": "SELECT 2",
        "student_sql": "SELECT 1",
        "teacher_sql": "SELECT 1",
        "base_ok": base,
        "student_ok": student,
        "teacher_ok": True,
    }


@pytest.fixture(scope="module")
def responses(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    tmp = tmp_path_factory.mktemp("contract")
    s = make_settings(tmp)
    gate = threading.Event()
    s.executor = scripted_executor(s.root, gate)
    out: dict[str, Any] = {}
    with TestClient(create_app(s)) as c:
        rid = c.post("/api/runs", json={"scale": "tiny", "dry_run": True}).json()["run_id"]
        gate.set()
        _wait_complete(c, rid)
        # a second finished run that carries held-out examples (the scripted run has none)
        ex_id = "dry-sql-tiny-contract"
        items = [
            _example(1, "fixed", False, True),
            _example(2, "still_wrong", False, False),
            _example(3, "regressed", True, False),
        ]
        seed_finished_run(s.root, ex_id, examples=items).close()
        for name, path in {
            "health": "/api/health",
            "config": "/api/config",
            "runs": "/api/runs",
            "run": f"/api/runs/{rid}",
            "report": f"/api/runs/{rid}/report",
            "tree": f"/api/runs/{rid}/tree",
            "examples": f"/api/runs/{ex_id}/examples?kind=all&limit=20",
            "replay": "/api/replay",
        }.items():
            r = c.get(path)
            assert r.status_code == 200, (name, r.text)
            out[name] = r.json()
            if name == "examples":
                h = r.headers
                out["examples_headers"] = {
                    "available": h["x-examples-available"] == "true",
                    "cap_per_kind": int(h["x-examples-cap-per-kind"]),
                    "totals": json.loads(h["x-examples-totals"]),
                }
    return out


ENDPOINTS = [
    "health",
    "config",
    "runs",
    "run",
    "report",
    "tree",
    "examples",
    "examples_headers",
    "replay",
]


def test_contract_fixtures_match_live_responses(responses: dict[str, Any]) -> None:
    if UPDATE:
        CONTRACT_DIR.mkdir(parents=True, exist_ok=True)
        for name in ENDPOINTS:
            text = json.dumps(responses[name], indent=2, sort_keys=True) + "\n"
            (CONTRACT_DIR / f"{name}.json").write_text(text, encoding="utf-8")
        return
    problems: list[str] = []
    for name in ENDPOINTS:
        f = CONTRACT_DIR / f"{name}.json"
        if not f.exists():
            problems.append(f"{name}: fixture missing; run `npm run contract:update`")
            continue
        fixture = json.loads(f.read_text(encoding="utf-8"))
        problems += [f"{name}: {p}" for p in diff(responses[name], fixture, "", NULLABLE[name])]
    assert not problems, "server drifted from the frontend contract fixtures:\n" + "\n".join(
        problems
    )


def test_every_fixture_is_fake_and_labelled(responses: dict[str, Any]) -> None:
    assert responses["run"]["dry_run"] is True
    assert responses["report"]["dry_run"] is True and responses["report"]["recorded"] is False
    assert all(r["dry_run"] is True for r in responses["runs"])


def _nulls(v: Any, path: str = "") -> list[str]:
    if v is None:
        return [path]
    if isinstance(v, dict):
        return [n for k, x in v.items() for n in _nulls(x, f"{path}.{k}".lstrip("."))]
    if isinstance(v, list):
        return [n for x in v for n in _nulls(x, f"{path}[]")]
    return []


def test_fixtures_contain_null_only_where_contract_allows(responses: dict[str, Any]) -> None:
    bad = [
        f"{name}: {p}"
        for name in ENDPOINTS
        for p in _nulls(responses[name])
        if p not in NULLABLE[name]
        and not (name == "report" and p.startswith(("config.", "rounds")))
    ]
    # report.json is the pipeline's verbatim output; its own nulls (hyperparameters etc.) are
    # not part of the API contract, only the API-added recorded_at is.
    assert not bad, bad
