from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient
from server_helpers import (
    GOLD,
    MODELS,
    QUESTION,
    fake_llm,
    make_config,
    make_settings,
    seed_finished_run,
)

from distillery.server import create_app


def _client(tmp_path: Path, sql: str = GOLD, **kw: object) -> tuple[TestClient, object]:
    llm, transport = fake_llm(sql)
    s = make_settings(tmp_path, playground_llm=llm, **kw)
    return TestClient(create_app(s)), transport


def test_unavailable_models_are_honest_without_llm(tmp_path: Path) -> None:
    with TestClient(create_app(make_settings(tmp_path))) as c:
        r = c.post("/api/playground", json={"question": QUESTION})
        assert r.status_code == 200
        body = r.json()
        assert set(body["results"]) == {"teacher", "base", "student"}
        for m in body["results"].values():
            assert m["available"] is False and m["reason"] and m["sql"] is None
            assert m["verified"] is None and m["rows_preview"] is None
        assert body["cost_usd"] == 0.0


def test_teacher_answer_verified_by_execution_and_unknown_gold_is_null(tmp_path: Path) -> None:
    c, transport = _client(tmp_path)
    with c:
        assert c.get("/api/config").json()["playground"]["enabled"] is True
        r = c.post("/api/playground", json={"question": QUESTION}).json()
        t = r["results"]["teacher"]
        assert t["available"] is True and t["sql"] == GOLD and t["error"] is None
        assert t["verified"] is None  # no completed run knows this question yet
        assert t["rows_preview"]["columns"] and t["rows_preview"]["rows"][0][0] > 0
        assert r["results"]["base"]["available"] is False
        assert r["results"]["student"]["available"] is False
        assert "switched off" in r["results"]["base"]["reason"]
        assert r["cost_usd"] > 0 and transport.calls == 1  # type: ignore[attr-defined]
        assert c.get("/api/config").json()["playground"]["spent_today_usd"] == r["cost_usd"]


def test_verified_true_and_false_against_known_task(tmp_path: Path) -> None:
    seed_finished_run(tmp_path / "data", "dry-sql-tiny-g").close()
    c, _ = _client(tmp_path)
    with c:
        good = c.post("/api/playground", json={"question": QUESTION}).json()
        assert good["results"]["teacher"]["verified"] is True
    c2, _ = _client(tmp_path / "x", sql="SELECT 1")
    with c2:
        pass
    seed_finished_run(tmp_path / "x" / "data", "dry-sql-tiny-g").close()
    c3, _ = _client(tmp_path / "x", sql="SELECT 1")
    with c3:
        bad = c3.post("/api/playground", json={"question": QUESTION}).json()
        assert bad["results"]["teacher"]["verified"] is False


def test_write_sql_is_refused_read_only(tmp_path: Path) -> None:
    c, _ = _client(tmp_path, sql="DELETE FROM accounts")
    with c:
        r = c.post("/api/playground", json={"question": "delete stuff"}).json()
        t = r["results"]["teacher"]
        # extract_sql only accepts SELECT/WITH; anything else is reported, never executed
        assert t["sql"] is None and t["error"] and t["verified"] is None


def test_question_validation(tmp_path: Path) -> None:
    c, _ = _client(tmp_path)
    with c:
        assert c.post("/api/playground", json={"question": "x" * 501}).status_code == 422
        assert c.post("/api/playground", json={"question": ""}).status_code == 422
        assert c.post("/api/playground", json={}).status_code == 422
        assert c.post("/api/playground", json={"question": "x" * 500}).status_code == 200


def test_per_ip_rate_limit(tmp_path: Path) -> None:
    now = [0.0]
    c, transport = _client(tmp_path, playground_per_ip_per_hour=2, clock=lambda: now[0])
    with c:
        for _ in range(2):
            assert c.post("/api/playground", json={"question": QUESTION}).status_code == 200
        r = c.post("/api/playground", json={"question": QUESTION})
        assert r.status_code == 429 and int(r.headers["retry-after"]) >= 1
        assert transport.calls == 2  # type: ignore[attr-defined]
        now[0] += 3601
        assert c.post("/api/playground", json={"question": QUESTION}).status_code == 200


def test_daily_cap_returns_503_and_stops_calls(tmp_path: Path) -> None:
    # cap is a few calls' worth at the fake price; the loop must hit it well within 40 calls
    cfg = make_config(playground_daily_cap_usd=0.01)
    c, transport = _client(tmp_path, config=cfg, playground_per_ip_per_hour=100)
    with c:
        codes = [
            c.post("/api/playground", json={"question": QUESTION}).status_code for _ in range(40)
        ]
        assert codes[0] == 200 and 503 in codes
        r = c.post("/api/playground", json={"question": QUESTION})
        assert r.status_code == 503 and r.json()["error"] == "demo_budget_exhausted"
        calls = transport.calls  # type: ignore[attr-defined]
        c.post("/api/playground", json={"question": QUESTION})
        assert transport.calls == calls  # type: ignore[attr-defined]


def test_unpriced_teacher_is_unavailable_not_free(tmp_path: Path) -> None:
    cfg = make_config(
        prices={m: p for m, p in make_config().prices.items() if m != MODELS["teacher"]}
    )
    c, transport = _client(tmp_path, config=cfg)
    with c:
        t = c.post("/api/playground", json={"question": QUESTION}).json()["results"]["teacher"]
        assert t["available"] is False and "price" in t["reason"]
        assert transport.calls == 0  # type: ignore[attr-defined]
