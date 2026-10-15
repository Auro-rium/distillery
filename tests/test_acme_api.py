# ruff: noqa: S101, S105, S106, E501
"""``POST /api/acme/ask``: the Acme flow run server-side through the gateway, in-process."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi.testclient import TestClient
from server_helpers import ADMIN, GOLD, make_config, make_settings
from test_gateway import NANO, TEACHER, Env, FakeUpstream, student_for, write_router

from distillery.acme import SUMMARY_MAX_TOKENS, summary_messages
from distillery.prompts import SYSTEM_PROMPT
from distillery.server import create_app
from distillery.server.gateway_config import GatewaySettings
from distillery.taskpacks.sql.runner import run_select

Q = "How many accounts are there?"


def ask(env: Env, question: str = Q, **kw: Any) -> Any:
    return env.client.post("/api/acme/ask", json={"question": question}, **kw)


def certified(tmp_path: Path, **kw: Any) -> Env:
    student, nn = student_for()
    return Env(tmp_path, router=write_router(tmp_path / "router"), student=student, nn_sim=nn, **kw)


def test_student_served_answer_is_the_gateways_own_and_logged(tmp_path: Path) -> None:
    env = certified(tmp_path)
    r = ask(env)
    assert r.status_code == 200
    d = r.json()
    assert d["question"] == Q
    assert d["answer"]["sql"] == GOLD
    assert d["answer"]["verified"] is True and d["answer"]["error"] is None
    assert d["answer"]["columns"] and d["answer"]["row_count"] >= 1
    assert d["summary"] == "There are some accounts."
    assert d["summary_error"] is None
    assert d["distillery"]["served_by"] == "student"
    assert d["distillery"]["skill"] == "acme-sql"
    # the teacher was never called for the SQL; the summary used the Nano passthrough
    assert [c[0] for c in env.upstream.calls] == [NANO]
    # both calls are real gateway calls, in the call log, and listed by request id
    rows = env.rows()
    assert [row["request_id"] for row in rows] == d["calls"]
    assert [row["served_by"] for row in rows] == ["student", "passthrough"]
    assert d["calls"][0] == d["distillery"]["request_id"]
    assert d["summary_cost_usd"] is not None


def test_teacher_deferral_surfaces_reason(tmp_path: Path) -> None:
    env = Env(tmp_path)  # nothing certified
    d = ask(env).json()
    assert d["distillery"]["served_by"] == "teacher"
    assert d["distillery"]["reason"] == "not certified: all traffic goes to the teacher"
    assert [c[0] for c in env.upstream.calls] == [TEACHER, NANO]
    assert env.upstream.calls[0][1][0] == {"role": "system", "content": SYSTEM_PROMPT}
    assert env.upstream.calls[1][2]["max_tokens"] == SUMMARY_MAX_TOKENS
    assert d["answer"]["verified"] is True


def test_bad_sql_is_reported_and_summary_skipped(tmp_path: Path) -> None:
    env = Env(tmp_path, upstream=FakeUpstream({TEACHER: "```sql\nSELECT * FROM nope\n```"}))
    d = ask(env).json()
    assert d["answer"]["verified"] is False
    assert d["answer"]["error"].startswith("query failed")
    assert d["summary"] is None and d["summary_error"] is None
    assert len(d["calls"]) == 1


def test_many_rows_are_capped_in_the_answer_but_the_summary_sees_the_real_count(
    tmp_path: Path,
) -> None:
    env = Env(
        tmp_path, upstream=FakeUpstream({TEACHER: "```sql\nSELECT account_id FROM accounts\n```"})
    )
    d = ask(env).json()
    assert d["answer"]["row_count"] > 20 and len(d["answer"]["rows"]) == 20
    total = d["answer"]["row_count"]
    user = env.upstream.calls[1][1][1]["content"]
    assert f"of {total}):" in user
    full = run_select(env.app.state.playground.demo_db(), "SELECT account_id FROM accounts")
    assert env.upstream.calls[1][1] == summary_messages(Q, full.columns, full.rows)


def test_no_sql_in_the_answer(tmp_path: Path) -> None:
    env = Env(tmp_path, upstream=FakeUpstream({TEACHER: "I cannot help with that."}))
    d = ask(env).json()
    assert d["answer"]["sql"] is None and d["answer"]["error"] == "the model returned no SQL"


def test_per_ip_limit_is_the_gateways_and_surfaces_as_429(tmp_path: Path) -> None:
    env = Env(tmp_path, gateway={"per_ip_per_hour": 3})
    assert ask(env).status_code == 200  # 2 gateway calls
    r = ask(env)  # the 3rd gateway call (SQL) is allowed; the summary hits the limit
    d = r.json()
    assert r.status_code == 200
    assert d["summary"] is None and d["summary_error"]["code"] == "rate_limited"
    assert d["summary_error"]["status"] == 429 and d["summary_error"]["retry_after"]
    r = ask(env)
    assert r.status_code == 429
    j = r.json()
    assert j["error"] == "rate_limited" and j["request_id"].startswith("req_")
    assert int(r.headers["retry-after"]) >= 1
    assert [row["status"] for row in env.rows()].count("refused") == 2


def test_limit_counts_the_visitor_not_the_server(tmp_path: Path) -> None:
    env = Env(tmp_path, gateway={"per_ip_per_hour": 2})
    assert ask(env).status_code == 200
    assert ask(env).status_code == 429
    # a direct gateway request from the same client address shares the same bucket
    r = env.post({"model": NANO, "messages": [{"role": "user", "content": "hi"}]})
    assert r.status_code == 429


def test_daily_cap_surfaces_as_429(tmp_path: Path) -> None:
    env = Env(tmp_path, config=make_config(playground_daily_cap_usd=0.0000001))
    r = ask(env)
    assert r.status_code == 429
    assert r.json()["error"] == "daily_budget_exhausted"


def test_upstream_unavailable_is_503(tmp_path: Path) -> None:
    cfg = make_config(nebius_api_key=None)
    app = create_app(make_settings(tmp_path, config=cfg, gateway=GatewaySettings(ip_salt="s")))
    r = TestClient(app).post("/api/acme/ask", json={"question": Q})
    assert r.status_code == 503 and r.json()["error"] == "upstream_unavailable"


def test_admin_token_is_not_forwarded(tmp_path: Path) -> None:
    env = Env(tmp_path, gateway={"per_ip_per_hour": 2})
    assert ask(env, headers={"X-Admin-Token": ADMIN}).status_code == 200
    assert ask(env, headers={"X-Admin-Token": ADMIN}).status_code == 429


def test_validation(tmp_path: Path) -> None:
    env = Env(tmp_path)
    c = env.client
    assert c.post("/api/acme/ask", json={"question": ""}).status_code == 422
    assert c.post("/api/acme/ask", json={"question": "x" * 501}).status_code == 422
    assert c.post("/api/acme/ask", json={"question": "   "}).status_code == 422
    assert c.post("/api/acme/ask", json={"question": Q, "extra": 1}).status_code == 422
    assert c.post("/api/acme/ask", json=[1]).status_code == 422
    assert c.post("/api/acme/ask", content=b"{nope").status_code == 400
    assert c.post("/api/acme/ask", content=b"x" * 9000).status_code == 413
    assert env.rows() == []  # nothing reached the gateway


def test_status_and_savings_see_the_calls(tmp_path: Path) -> None:
    env = certified(tmp_path)
    ask(env)
    assert env.client.get("/api/gateway/savings").json()["requests"] == 1
    assert env.client.get("/api/gateway/status").json()["today"]["served_by"]["student"] == 1


def test_the_drift_audit_runs_after_the_visitors_answer_not_inside_it(tmp_path: Path) -> None:
    # httpx.ASGITransport awaits an app's background tasks, so a gateway BackgroundTask would make
    # the visitor wait for the audit's teacher call. The work is handed to /api/acme/ask, which
    # runs it after its own response: by then the summary call is in the call log too.
    from test_gateway_drift import drift

    env = certified(tmp_path, gateway={"drift": drift(audit_rate=1.0)})
    gw = env.app.state.gateway
    seen: list[int] = []
    inner = gw._drift_after

    async def spy(work: Any) -> None:
        seen.append(len(env.rows()))
        await inner(work)

    gw._drift_after = spy
    d = ask(env).json()
    assert d["distillery"]["served_by"] == "student" and len(d["calls"]) == 2
    assert seen == [2]  # skill + summary were both answered before the audit started
    assert gw.drift_status()["audits"]["audits"] == 1
    assert not gw._handoff  # nothing left behind


def test_a_forged_internal_header_changes_nothing(tmp_path: Path) -> None:
    from test_gateway import skill_body
    from test_gateway_drift import drift

    env = certified(tmp_path, gateway={"drift": drift(audit_rate=1.0)})
    gw = env.app.state.gateway
    r = env.post(skill_body(), **{"x-distillery-internal": "guess"})
    assert r.status_code == 200 and not gw._handoff
    assert gw.drift_status()["audits"]["audits"] == 1  # ran as the response's own background
