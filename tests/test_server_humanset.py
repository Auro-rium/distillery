"""Admin-gated human-set endpoints: GET /api/humanset/drafts, POST /api/humanset/decide."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from fastapi.testclient import TestClient
from server_helpers import ADMIN, make_config, make_settings, scripted_executor

from distillery import humanset as hs
from distillery.server import create_app
from distillery.store import Store

SID = "ab12cd34"
H = {"X-Admin-Token": ADMIN}
GOLD = "SELECT COUNT(*) FROM accounts"


def _drafts() -> dict[str, Any]:
    kept = [
        {
            "task_id": f"h-{i}-0000000{i}", "question": f"Question {i}?", "gold_sql": GOLD,
            "requires_order": False, "candidates": [GOLD] * 3,
            "preview": {"columns": ["n"], "rows": [[3]], "row_count": 1},
        }
        for i in range(3)
    ]  # fmt: skip
    return {
        "format": hs.DRAFTS_FORMAT, "question_file_sha256": SID + "0" * 56, "db_sha256": "d" * 64,
        "teacher_model": "fake-teacher", "temperature": 0.8, "k": 3, "n_questions": 5,
        "duplicates_dropped": 0, "kept": kept,
        "discarded": [{"task_id": "h-3-x", "question": "SECRET DISCARDED?", "reason": "empty",
                       "candidates": []}],
        "discarded_by_reason": {**dict.fromkeys(hs.DISCARD_REASONS, 0), "empty": 1},
    }  # fmt: skip


def _client(tmp_path: Path, **kw: Any) -> TestClient:
    s = make_settings(tmp_path, **kw)
    s.executor = scripted_executor(s.root)
    hs.write_drafts(s.root, _drafts())
    return TestClient(create_app(s))


def test_both_endpoints_are_admin_gated(tmp_path: Path) -> None:
    with _client(tmp_path) as c:
        body = {"set": SID, "task_id": "h-0-00000000", "decision": "confirm"}
        assert c.get("/api/humanset/drafts").status_code == 401
        assert c.get("/api/humanset/drafts", headers={"X-Admin-Token": "wrong"}).status_code == 403
        assert c.post("/api/humanset/decide", json=body).status_code == 401
        assert (
            c.post("/api/humanset/decide", json=body, headers={"X-Admin-Token": "x"}).status_code
            == 403
        )
        assert not (tmp_path / "data" / "humanset" / SID / "decisions.jsonl").exists()


def test_locked_when_no_admin_token_is_configured(tmp_path: Path) -> None:
    with _client(tmp_path, config=make_config(admin_token=None)) as c:
        assert (
            c.get("/api/humanset/drafts", headers={"X-Admin-Token": "anything"}).status_code == 403
        )
        assert c.get("/api/humanset/drafts").status_code == 401


def test_drafts_listing_and_decide_round_trip(tmp_path: Path) -> None:
    with _client(tmp_path) as c:
        r = c.get("/api/humanset/drafts", headers=H)
        assert r.status_code == 200
        d = r.json()
        assert d["sets"] == [SID] and d["set"] == SID
        assert [i["question"] for i in d["items"]] == ["Question 0?", "Question 1?", "Question 2?"]
        assert d["items"][0]["gold_sql"] == GOLD and d["items"][0]["preview"]["rows"] == [[3]]
        assert all(i["decision"] is None for i in d["items"])
        assert d["discarded_by_reason"]["empty"] == 1 and d["teacher_model"] == "fake-teacher"
        assert "SECRET DISCARDED?" not in r.text  # discarded questions are not served
        r = c.post("/api/humanset/decide", headers=H,
                   json={"set": SID, "task_id": "h-1-00000001", "decision": "reject"})  # fmt: skip
        assert r.status_code == 200
        assert r.json()["tally"] == {"confirmed": 0, "rejected": 1, "skipped": 0, "undecided": 2}
        c.post("/api/humanset/decide", headers=H,
               json={"set": SID, "task_id": "h-1-00000001", "decision": "confirm"})  # fmt: skip
        items = c.get(f"/api/humanset/drafts?set={SID}", headers=H).json()["items"]
        assert [i["decision"] for i in items] == [None, "confirm", None]
        # the same decisions.jsonl the CLI reads
        assert hs.read_decisions(tmp_path / "data", SID) == {"h-1-00000001": "confirm"}


def test_decide_validation(tmp_path: Path) -> None:
    with _client(tmp_path) as c:

        def post(body: Any) -> Any:
            return c.post("/api/humanset/decide", headers=H, json=body)

        ok = {"set": SID, "task_id": "h-0-00000000", "decision": "confirm"}
        assert post({**ok, "decision": "maybe"}).status_code == 422
        assert post({**ok, "task_id": "h-9-nope"}).status_code == 422
        assert post({**ok, "set": "../../etc"}).status_code == 422
        assert post({**ok, "set": "00000000"}).status_code == 404
        assert post({**ok, "extra": 1}).status_code == 422
        assert post("nope").status_code == 422
        assert c.get("/api/humanset/drafts?set=../x", headers=H).status_code == 422
        assert c.get("/api/humanset/drafts?set=00000000", headers=H).status_code == 404


def test_no_sets_is_a_404(tmp_path: Path) -> None:
    s = make_settings(tmp_path)
    s.executor = scripted_executor(s.root)
    with TestClient(create_app(s)) as c:
        r = c.get("/api/humanset/drafts", headers=H)
        assert r.status_code == 404 and r.json()["error"] == "not_found"


def test_drafting_spend_run_is_hidden_and_humanset_is_never_replayed(tmp_path: Path) -> None:
    with _client(tmp_path) as c:
        store = Store(tmp_path / "data")
        store.record_spend(f"{hs.RUN_PREFIX}{SID}", "llm", "fake-teacher", 0.01)
        store.close()
        assert c.get("/api/runs").json() == []  # a drafting ledger is not a run
        assert [i["run_id"] for i in c.get("/api/replay").json()] == ["sample-dry-run"]
        assert "humanset" not in json.dumps(c.get("/api/replay").json())
