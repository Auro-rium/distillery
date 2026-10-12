# ruff: noqa: S101, S105, S106, E501
"""The OpenAI-compatible gateway: skill detection, certified routing, passthrough, limits, call log,
savings, route ordering. Everything is fake: no network, no paid call."""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient
from server_helpers import ADMIN, GOLD, MODELS, make_config, make_settings

from distillery.prompts import SYSTEM_PROMPT, build_messages, parse_user_content
from distillery.router import RouterModel, decide
from distillery.server import create_app
from distillery.server.gateway import (
    NOT_CERTIFIED_REASON,
    ChatRequest,
    GatewayError,
    detect_skill,
    load_index,
    load_router,
    parse_chat_request,
)
from distillery.server.gateway_config import EmbedResult, GatewaySettings, UpstreamResult
from distillery.student import FakeStudent
from distillery.taskpacks.sql import schema as sql_schema

QUESTION = "How many accounts are there?"
TEACHER, NANO = MODELS["teacher"], MODELS["triage"]
DDL = sql_schema.schema_ddl(0)

# score = sigmoid(0.5 + 1.5 z_exec + 0.5 z_logprob + 1.0 z_nn) with the means/stds below.
MODEL = RouterModel(
    feature_names=("exec_ok", "mean_logprob", "nn_sim"),
    means=(0.5, -0.5, 0.5),
    stds=(0.5, 0.5, 0.25),
    weights=(1.5, 0.5, 1.0),
    bias=0.5,
    l2=0.01,
    n_fit=10,
    fit_sha256="f" * 64,
)
LAMBDA = 0.8


class FakeUpstream:
    """Stands in for Token Factory. ``answers`` maps a model id to the text it returns."""

    def __init__(self, answers: Mapping[str, str] | None = None, fail: bool = False) -> None:
        self.answers = {TEACHER: f"```sql\n{GOLD}\n```", NANO: "There are some accounts."}
        self.answers.update(answers or {})
        self.fail = fail
        self.embed_vector = [0.6, 0.8, 0.0]
        self.embed_fail = False
        self.embed_calls: list[tuple[str, list[str]]] = []
        self.calls: list[tuple[str, list[dict[str, Any]], dict[str, Any]]] = []

    async def embed(self, model: str, texts: Sequence[str]) -> EmbedResult:
        self.embed_calls.append((model, list(texts)))
        if self.embed_fail:
            raise RuntimeError("embed boom")
        return EmbedResult([list(self.embed_vector) for _ in texts], 20)

    async def complete(
        self, model: str, messages: Sequence[Mapping[str, Any]], **params: Any
    ) -> UpstreamResult:
        self.calls.append((model, [dict(m) for m in messages], params))
        if self.fail:
            raise RuntimeError("boom")
        return UpstreamResult(self.answers.get(model, "ok"), 1200, 30)


def write_router(
    d: Path, *, lam: float | None = LAMBDA, pin: str | None = None, prereg: str | None = "abc123"
) -> Path:
    d.mkdir(parents=True, exist_ok=True)
    (d / "model.json").write_text(
        json.dumps({"model": MODEL.to_json(), "model_sha256": MODEL.sha256()}), encoding="utf-8"
    )
    cert = {
        "format": "distillery-router-cert-v1",
        "alpha": 0.1,
        "delta": 0.05,
        "lambda_hat": lam,
        "model_sha256": pin or MODEL.sha256(),
        "cert": {"coverage": 0.62, "selective_risk": 0.07, "alpha": 0.1, "delta": 0.05},
        "prereg": {"commit": prereg} if prereg else None,
    }
    (d / "cert.json").write_text(json.dumps(cert), encoding="utf-8")
    return d


def skill_body(question: str = QUESTION, **extra: Any) -> dict[str, Any]:
    return {
        "model": TEACHER,
        "messages": build_messages(question, DDL, role="eval_teacher"),
        **extra,
    }


def student_for(
    sql: str = GOLD, logprob: float = -0.1
) -> tuple[FakeStudent, Callable[[str], float]]:
    return FakeStudent(lambda m: sql, lambda m, t: logprob), lambda q: 0.9


class Env:
    def __init__(self, tmp_path: Path, **kw: Any) -> None:
        self.upstream: FakeUpstream = kw.pop("upstream", None) or FakeUpstream()
        self.student: FakeStudent | None = kw.pop("student", None)
        router = kw.pop("router", None)
        nn = kw.pop("nn_sim", None)
        gw: dict[str, Any] = kw.pop("gateway", {})
        cfg = kw.pop("config", None) or make_config()
        seed_llm = kw.pop("seed_llm", True)
        self.settings = make_settings(
            tmp_path,
            config=cfg,
            gateway=GatewaySettings(
                router_dir=router, upstream=self.upstream, nn_sim=nn, ip_salt="s", **gw
            ),
            **kw,
        )
        self.app = create_app(self.settings)
        if self.student is not None:
            student = self.student
            self.app.state.playground.local_status = lambda role: ("ok", None, student)
        self.client = TestClient(self.app)
        if seed_llm:  # a measured teacher output length from "pipeline runs" (llm_calls)
            self.app.state.reader.real.record_llm_call("sql-run", TEACHER, 1000, 40, 0.1)

    def post(self, body: dict[str, Any], **headers: str) -> Any:
        return self.client.post("/v1/chat/completions", json=body, headers=headers)

    def rows(self) -> list[dict[str, Any]]:
        return self.app.state.reader.real.gateway_calls()  # type: ignore[no-any-return]


@pytest.fixture
def certified(tmp_path: Path) -> Env:
    student, nn = student_for()
    return Env(tmp_path, router=write_router(tmp_path / "router"), student=student, nn_sim=nn)


# ---- parse_user_content / skill detection -----------------------------------------------------


@pytest.mark.parametrize(
    "q",
    [
        QUESTION,
        "Multi\nline question?",
        "What about\n\nQuestion: nested marker?",
        "  padded  ",
        "unicode ✓ → ≥?",
    ],
)
def test_parse_user_content_round_trip(q: str) -> None:
    content = build_messages(q, DDL, role="train")[1]["content"]
    assert parse_user_content(content) == (q.strip(), DDL.strip())


@pytest.mark.parametrize(
    "content",
    [
        "",
        "hello",
        "Database schema:\nCREATE TABLE t(a)\n\nQuestion: q?",  # no marker
        "Database schema:\nx\n\nQuestion: \n\n/no_think",  # empty question
        "Database schema:\n\n\nQuestion: q\n\n/no_think",  # empty schema
        "Database schema:\nx\n\nQuestion: q\n\n/no_think\n",  # trailing junk
        "Please summarize. Database schema:\nx\n\nQuestion: q\n\n/no_think",
    ],
)
def test_parse_user_content_rejects_lookalikes(content: str) -> None:
    assert parse_user_content(content) is None


def _req(**kw: Any) -> ChatRequest:
    body = skill_body()
    body.update(kw)
    return parse_chat_request(body)


def test_detect_skill_exact_match() -> None:
    sk = detect_skill(_req(), TEACHER)
    assert sk is not None and sk.question == QUESTION and sk.ddl == DDL.strip()


def test_detect_skill_rejections() -> None:
    assert detect_skill(_req(model=NANO), TEACHER) is None  # wrong model
    assert detect_skill(_req(), None) is None  # no teacher configured
    msgs = build_messages(QUESTION, DDL, role="eval_teacher")
    tampered = [{"role": "system", "content": SYSTEM_PROMPT + " "}, msgs[1]]
    assert detect_skill(_req(messages=tampered), TEACHER) is None
    assert (
        detect_skill(_req(messages=[{**msgs[0], "content": "You are helpful."}, msgs[1]]), TEACHER)
        is None
    )
    extra = [*msgs, {"role": "assistant", "content": "SELECT 1"}]
    assert detect_skill(_req(messages=extra), TEACHER) is None  # extra message
    assert detect_skill(_req(messages=[*msgs, msgs[1]]), TEACHER) is None  # two user messages
    assert detect_skill(_req(messages=[msgs[1]]), TEACHER) is None  # no system message
    assert detect_skill(_req(messages=[msgs[1], msgs[0]]), TEACHER) is None  # wrong order
    custom = [msgs[0], {"role": "user", "content": "Write me a poem"}]
    assert detect_skill(_req(messages=custom), TEACHER) is None  # free text user message


def test_parse_chat_request_validation() -> None:
    with pytest.raises(GatewayError) as e:
        parse_chat_request(
            {"model": "m", "messages": [{"role": "user", "content": "x"}], "stream": True}
        )
    assert e.value.status == 400 and e.value.code == "stream_not_supported"
    for bad in (
        {},
        {"model": "m"},
        {"model": "m", "messages": []},
        {"model": "", "messages": [{}]},
    ):
        with pytest.raises(GatewayError):
            parse_chat_request(bad)
    ok = parse_chat_request(
        {"model": "m", "messages": [{"role": "user", "content": [{"type": "text", "text": "hi"}]}],
         "max_completion_tokens": 7, "temperature": 0, "frequency_penalty": 1}
    )  # fmt: skip
    assert (
        ok.messages[0]["content"] == "hi" and ok.max_tokens == 7 and ok.params == {"temperature": 0}
    )


# ---- certified path ---------------------------------------------------------------------------


def test_certified_student_serves_with_standard_schema_and_headers(certified: Env) -> None:
    r = certified.post(skill_body())
    assert r.status_code == 200
    b = r.json()
    assert b["object"] == "chat.completion" and b["id"].startswith("chatcmpl-")
    assert b["model"] == TEACHER  # the model the client asked for; the truth is in `distillery`
    assert b["choices"][0]["message"] == {"role": "assistant", "content": GOLD}
    assert b["choices"][0]["finish_reason"] == "stop"
    u = b["usage"]
    assert u["prompt_tokens"] > 0 and u["completion_tokens"] > 0
    assert u["total_tokens"] == u["prompt_tokens"] + u["completion_tokens"]
    d = b["distillery"]
    assert d["served_by"] == "student" and d["skill"] == "acme-sql"
    assert d["lambda"] == LAMBDA and d["score"] >= LAMBDA and d["reason"].startswith("certified")
    assert d["sql"] == GOLD and d["verified"] is True
    assert d["columns"] and d["rows"][0][0] > 0 and d["row_count"] == 1
    assert d["cost_usd"] == 0.0 and d["teacher_cost_usd_counterfactual"] > 0
    assert d["latency_ms"] >= 0 and d["request_id"].startswith("req_")
    h = r.headers
    assert h["x-distillery-served-by"] == "student" and h["x-distillery-verified"] == "true"
    assert h["x-distillery-request-id"] == d["request_id"]
    assert float(h["x-distillery-score"]) == pytest.approx(d["score"], abs=1e-6)
    assert float(h["x-distillery-lambda"]) == LAMBDA and h["x-distillery-cost-usd"]
    assert certified.upstream.calls == []  # the teacher was never called
    assert certified.student is not None and certified.student.calls == 1


def test_rows_are_capped_at_20(tmp_path: Path) -> None:
    student, nn = student_for("SELECT account_id FROM accounts")
    env = Env(tmp_path, router=write_router(tmp_path / "r"), student=student, nn_sim=nn)
    d = env.post(skill_body("List all account ids")).json()["distillery"]
    assert len(d["rows"]) == 20 and d["row_count"] > 20


def test_short_circuit_never_calls_the_student(tmp_path: Path) -> None:
    student = FakeStudent(lambda m: GOLD)
    env = Env(tmp_path, router=write_router(tmp_path / "r"), student=student, nn_sim=lambda q: 0.0)
    r = env.post(skill_body())
    d = r.json()["distillery"]
    assert d["served_by"] == "teacher" and d["short_circuit"] is True and d["score"] is None
    assert d["reason"] == "unfamiliar question pattern → teacher"
    assert "->" in r.headers["x-distillery-reason"]  # headers are ASCII
    assert student.calls == 0 and len(env.upstream.calls) == 1
    assert env.rows()[0]["student_called"] == 0


def test_exec_failure_defers_to_teacher(tmp_path: Path) -> None:
    student, nn = student_for("SELECT nope FROM missing_table")
    env = Env(tmp_path, router=write_router(tmp_path / "r"), student=student, nn_sim=nn)
    r = env.post(skill_body())
    d = r.json()["distillery"]
    assert d["served_by"] == "teacher" and d["reason"] == "student SQL failed to run → teacher"
    assert d["score"] is not None and d["score"] < LAMBDA
    assert r.json()["choices"][0]["message"]["content"] == GOLD.join(("```sql\n", "\n```"))
    assert (
        d["sql"] == GOLD and d["verified"] is True
    )  # the served (teacher) SQL is what is verified
    assert d["cost_usd"] == pytest.approx((1200 * 1.0 + 30 * 2.0) / 1e6)
    assert d["teacher_cost_usd_counterfactual"] == d["cost_usd"]
    assert env.rows()[0]["student_called"] == 1


def test_low_confidence_defers(tmp_path: Path) -> None:
    student, nn = student_for(logprob=-3.0)
    env = Env(tmp_path, router=write_router(tmp_path / "r"), student=student, nn_sim=nn)
    d = env.post(skill_body()).json()["distillery"]
    assert d["served_by"] == "teacher" and d["reason"] == "low student confidence → teacher"


def test_unknown_schema_means_exec_ok_zero(tmp_path: Path) -> None:
    student, nn = student_for()
    env = Env(tmp_path, router=write_router(tmp_path / "r"), student=student, nn_sim=nn)
    msgs = build_messages(QUESTION, "CREATE TABLE other(x INTEGER)", role="eval_teacher")
    d = env.post({"model": TEACHER, "messages": msgs}).json()["distillery"]
    assert d["served_by"] == "teacher" and d["reason"] == "student SQL failed to run → teacher"
    assert d["verified"] is False and d["rows"] == []


def test_student_failure_and_unmeasured_confidence_defer(tmp_path: Path) -> None:
    class Boom(FakeStudent):
        def generate_detailed(self, batch: Any) -> Any:
            raise RuntimeError("sandbox lost")

    env = Env(tmp_path, router=write_router(tmp_path / "r"), student=Boom(), nn_sim=lambda q: 0.9)
    d = env.post(skill_body()).json()["distillery"]
    assert d["served_by"] == "teacher" and d["reason"] == "student failed (RuntimeError) → teacher"


def test_student_unavailable_defers(tmp_path: Path) -> None:
    env = Env(tmp_path, router=write_router(tmp_path / "r"), nn_sim=lambda q: 0.9)
    d = env.post(skill_body()).json()["distillery"]
    assert d["served_by"] == "teacher" and d["reason"].startswith("student unavailable")


def test_missing_similarity_source_goes_to_teacher(tmp_path: Path) -> None:
    student, _ = student_for()
    env = Env(tmp_path, router=write_router(tmp_path / "r"), student=student)
    d = env.post(skill_body()).json()["distillery"]
    assert d["served_by"] == "teacher" and "similarity index unavailable" in d["reason"]
    assert student.calls == 0


def test_uncertified_everything_goes_to_the_teacher(tmp_path: Path) -> None:
    student = FakeStudent(lambda m: GOLD)
    env = Env(tmp_path, student=student)
    r = env.post(skill_body())
    d = r.json()["distillery"]
    assert d["served_by"] == "teacher" and d["reason"] == NOT_CERTIFIED_REASON
    assert d["score"] is None and d["lambda"] is None and student.calls == 0
    assert NOT_CERTIFIED_REASON == decide(MODEL, None, {}).reason
    st = env.client.get("/api/gateway/status").json()
    assert st["certified"] is False and "no router artifacts" in st["reason"]


def test_cert_with_no_threshold_is_not_certified(tmp_path: Path) -> None:
    env = Env(tmp_path, router=write_router(tmp_path / "r", lam=None), student=FakeStudent())
    d = env.post(skill_body()).json()["distillery"]
    assert d["served_by"] == "teacher" and d["reason"] == NOT_CERTIFIED_REASON


def test_cert_pinned_to_another_model_is_refused(tmp_path: Path) -> None:
    st = load_router(write_router(tmp_path / "r", pin="0" * 64))
    assert st.certified is False and st.lambda_hat is None
    assert st.error is not None and "different model" in st.error
    assert load_router(tmp_path / "nowhere").error is not None
    assert load_router(None).certified is False


def test_status_reports_the_certificate(certified: Env) -> None:
    certified.post(skill_body())
    st = certified.client.get("/api/gateway/status").json()
    assert st["certified"] is True and st["lambda"] == LAMBDA
    assert st["coverage"] == 0.62 and st["alpha"] == 0.1 and st["delta"] == 0.05
    assert st["model_sha256"] == MODEL.sha256() and st["prereg_commit"] == "abc123"
    assert st["today"]["served_by"] == {"student": 1} and st["today"]["student_calls"] == 1
    assert st["today"]["student_daily_max"] == 200


# ---- passthrough ------------------------------------------------------------------------------


def _chat(model: str, **extra: Any) -> dict[str, Any]:
    return {"model": model, "messages": [{"role": "user", "content": "Say hi"}], **extra}


def test_passthrough_allowed_model_clamps_max_tokens(tmp_path: Path) -> None:
    env = Env(tmp_path)
    r = env.post(_chat(NANO, max_tokens=100_000, temperature=0.2, tools=None))
    assert r.status_code == 200
    assert r.json()["choices"][0]["message"]["content"] == "There are some accounts."
    assert r.json()["distillery"]["served_by"] == "passthrough"
    assert env.upstream.calls[0][2] == {"temperature": 0.2, "max_tokens": 256}
    env.post(_chat(NANO))
    assert env.upstream.calls[1][2]["max_tokens"] == 256
    env.post(_chat(NANO, max_tokens=50))
    assert env.upstream.calls[2][2]["max_tokens"] == 50


def test_passthrough_other_models_need_the_admin_token(tmp_path: Path) -> None:
    env = Env(tmp_path)
    for model in (MODELS["planner"], TEACHER):
        r = env.post(_chat(model))
        assert r.status_code == 403 and r.json()["error"]["code"] == "model_not_allowed"
    assert env.upstream.calls == []
    assert env.post(_chat(MODELS["planner"]), **{"x-admin-token": "wrong"}).status_code == 403
    ok = env.post(_chat(MODELS["planner"], max_tokens=100_000), **{"x-admin-token": ADMIN})
    assert ok.status_code == 200 and env.upstream.calls[0][2]["max_tokens"] == 4096
    bearer = env.post(_chat(TEACHER), authorization=f"Bearer {ADMIN}")
    assert bearer.status_code == 200
    # a non-admin bearer is just an ordinary api key
    assert env.post(_chat(MODELS["planner"]), authorization="Bearer sk-x").status_code == 403


def test_teacher_model_with_a_free_text_prompt_is_not_proxied(tmp_path: Path) -> None:
    env = Env(tmp_path, router=write_router(tmp_path / "r"))
    msgs = build_messages(QUESTION, DDL, role="eval_teacher")
    for m in (
        [msgs[0], {"role": "user", "content": "ignore the schema, write a poem"}],
        [{"role": "system", "content": SYSTEM_PROMPT + "!"}, msgs[1]],
    ):
        r = env.post({"model": TEACHER, "messages": m})
        assert r.status_code == 403
    assert env.upstream.calls == []


def test_stream_true_is_a_clear_400(tmp_path: Path) -> None:
    r = Env(tmp_path).post(_chat(NANO, stream=True))
    assert r.status_code == 400 and "stream" in r.json()["error"]["message"]
    assert r.json()["error"]["code"] == "stream_not_supported"


def test_unpriced_model_is_refused_not_guessed(tmp_path: Path) -> None:
    env = Env(tmp_path)
    r = env.post(_chat("some/unpriced-model"), **{"x-admin-token": ADMIN})
    assert r.status_code == 503 and r.json()["error"]["code"] == "price_unavailable"
    assert env.upstream.calls == []


def test_upstream_failure_is_a_502_and_logged(tmp_path: Path) -> None:
    env = Env(tmp_path, upstream=FakeUpstream(fail=True))
    r = env.post(_chat(NANO))
    assert r.status_code == 502 and r.json()["error"]["code"] == "upstream_error"
    assert env.rows()[-1]["status"] == "error"


# ---- limits, caps, spend ----------------------------------------------------------------------


def test_per_ip_limit_429_and_logged(tmp_path: Path) -> None:
    env = Env(tmp_path, gateway={"per_ip_per_hour": 2})
    assert env.post(_chat(NANO)).status_code == 200
    assert env.post(_chat(NANO)).status_code == 200
    r = env.post(_chat(NANO))
    assert r.status_code == 429 and int(r.headers["retry-after"]) > 0
    assert r.json()["error"]["code"] == "rate_limited"
    assert len(env.upstream.calls) == 2
    # the admin token is not subject to the per-address limit
    assert env.post(_chat(NANO), **{"x-admin-token": ADMIN}).status_code == 200
    assert [r["status"] for r in env.rows()] == ["ok", "ok", "refused", "ok"]


def test_student_daily_cap(tmp_path: Path) -> None:
    student, nn = student_for()
    env = Env(
        tmp_path, router=write_router(tmp_path / "r"), student=student, nn_sim=nn,
        gateway={"student_daily_max": 2},
    )  # fmt: skip
    assert [env.post(skill_body()).status_code for _ in range(2)] == [200, 200]
    r = env.post(skill_body())
    assert r.status_code == 429 and r.json()["error"]["code"] == "student_daily_cap"
    assert "retry-after" in r.headers and student.calls == 2
    assert env.rows()[-1]["status"] == "refused"
    # the cap does not stop calls that never touch the student sandbox
    assert env.post(_chat(NANO)).status_code == 200


def test_daily_dollar_cap_shared_with_the_playground(tmp_path: Path) -> None:
    # reservation = input estimate + the output cap the call may reach (2048 tokens), at 1/2 $ per M
    in_est = sum(len(m["content"]) for m in skill_body()["messages"]) // 3
    reserve = (in_est * 1.0 + 2048 * 2.0) / 1e6
    env = Env(tmp_path, config=make_config(playground_daily_cap_usd=reserve * 1.3))
    assert env.post(skill_body()).status_code == 200  # teacher, ~$0.00126 settled
    spent = env.client.get("/api/config").json()["playground"]["spent_today_usd"]
    assert spent == pytest.approx(0.00126)
    for _ in range(5):
        r = env.post(skill_body())
        if r.status_code == 429:
            break
    assert r.status_code == 429 and r.json()["error"]["code"] == "daily_budget_exhausted"
    assert "retry-after" in r.headers
    assert env.rows()[-1]["status"] == "refused" and env.rows()[-1]["cost_usd"] == 0


def test_unconfigured_upstream_is_503(tmp_path: Path) -> None:
    env = Env(tmp_path)
    env.app.state.gateway.upstream = None
    r = env.post(_chat(NANO))
    assert r.status_code == 503 and r.json()["error"]["code"] == "upstream_unavailable"


# ---- call log ---------------------------------------------------------------------------------


def test_every_call_is_logged_including_refusals(certified: Env) -> None:
    certified.post(skill_body())  # student
    certified.post(_chat(NANO))  # passthrough
    certified.post(_chat(MODELS["planner"]))  # refused 403
    certified.client.post("/v1/chat/completions", content=b"{nope")  # invalid json
    certified.post(_chat(NANO, stream=True))  # 400
    rows = certified.rows()
    assert [r["status"] for r in rows] == ["ok", "ok", "refused", "refused", "refused"]
    s, p = rows[0], rows[1]
    assert (s["served_by"], s["skill"], s["student_called"]) == ("student", 1, 1)
    assert s["lambda"] == LAMBDA and s["score"] >= LAMBDA and s["reason"].startswith("certified")
    assert s["input_tokens"] > 0 and s["output_tokens"] > 0 and s["cost_usd"] == 0.0
    assert s["teacher_cost_usd_counterfactual"] > 0 and "estimated" in s["counterfactual_basis"]
    assert (p["served_by"], p["skill"], p["model"]) == ("passthrough", 0, NANO)
    assert p["input_tokens"] == 1200 and p["cost_usd"] > 0
    for r in rows:
        assert r["request_id"].startswith("req_")
        assert len(r["prompt_sha256"]) == 64 or r["error"] == "invalid_json"
        assert len(r["ip_hash"]) == 16 and "testclient" not in r["ip_hash"]
        assert r["latency_ms"] >= 0 and r["created_at"] and r["day"]
    assert rows[2]["error"] == "model_not_allowed" and rows[3]["error"] == "invalid_json"
    assert len({r["request_id"] for r in rows}) == 5


def test_counterfactual_uses_logged_teacher_output_mean(tmp_path: Path) -> None:
    student, nn = student_for()
    env = Env(tmp_path, router=write_router(tmp_path / "r"), student=student, nn_sim=nn)
    env.app.state.playground.local_status = lambda role: ("unavailable", "x", None)
    env.post(skill_body())  # teacher-served: 30 output tokens logged
    env.app.state.playground.local_status = lambda role: ("ok", None, student)
    env.post(skill_body())
    first, second = env.rows()
    assert first["served_by"] == "teacher" and second["served_by"] == "student"
    assert "30.0 mean output tokens over 1 logged" in second["counterfactual_basis"]


# ---- savings ----------------------------------------------------------------------------------


def test_savings_pending_h1_basis(certified: Env) -> None:
    for _ in range(3):
        certified.post(skill_body())
    certified.app.state.playground.local_status = lambda role: ("unavailable", "x", None)
    certified.post(skill_body())  # one teacher-served
    certified.post(_chat(NANO))  # passthrough: reported apart, not in the savings
    certified.post(_chat(MODELS["planner"]))  # refused
    s = certified.client.get("/api/gateway/savings").json()
    assert s["requests"] == 4 and s["served_by"] == {"student": 3, "teacher": 1}
    assert s["student_calls"] == 3 and s["refused"] == 1
    assert s["student_cost_basis"] == "pending H1" and s["student_usd_per_1k"] is None
    teacher_usd = (1200 * 1.0 + 30 * 2.0) / 1e6
    assert s["actual_usd"] == pytest.approx(teacher_usd)
    rows = certified.rows()
    cf = sum(r["teacher_cost_usd_counterfactual"] for r in rows if r["skill"])
    assert s["teacher_only_usd"] == pytest.approx(cf)
    assert s["saved_usd"] == pytest.approx(cf - teacher_usd)
    assert s["saved_pct"] == pytest.approx((cf - teacher_usd) / cf * 100, abs=0.01)
    assert s["saved_display"].startswith("$") and s["saved_pct_display"].endswith("%")
    assert "pending H1" in s["basis"]["actual"] and "upper bound" in s["basis"]["saved"]
    assert s["passthrough_requests"] == 1 and s["passthrough_usd"] > 0


def test_savings_with_a_measured_price_file(tmp_path: Path) -> None:
    price = tmp_path / "serving_price.json"
    price.write_text(
        json.dumps({"student_usd_per_1k": 0.5, "source": "H1 measured run", "as_of": "2026-10-20"})
    )
    student, nn = student_for()
    env = Env(
        tmp_path, router=write_router(tmp_path / "r"), student=student, nn_sim=nn,
        gateway={"serving_price_file": price},
    )  # fmt: skip
    for _ in range(2):
        env.post(skill_body())
    s = env.client.get("/api/gateway/savings").json()
    assert s["student_usd_per_1k"] == 0.5 and s["student_cost_basis"].startswith("measured: H1")
    assert s["student_cost_usd"] == pytest.approx(2 / 1000 * 0.5)
    assert s["actual_usd"] == pytest.approx(0.001)  # student invocations x price, no teacher spend
    assert "pending" not in s["basis"]["saved"]


def test_savings_empty_and_malformed_price_file(tmp_path: Path) -> None:
    s = Env(tmp_path).client.get("/api/gateway/savings").json()
    assert s["requests"] == 0 and s["saved_pct"] is None and s["saved_usd"] == 0
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"student_usd_per_1k": "cheap"}))
    with pytest.raises(ValueError):
        Env(tmp_path / "x", gateway={"serving_price_file": bad})


# ---- routing / SPA ----------------------------------------------------------------------------


def test_gateway_routes_beat_the_spa_and_api_catch_all(tmp_path: Path) -> None:
    dist = tmp_path / "dist"
    dist.mkdir()
    (dist / "index.html").write_text("<html>SPA</html>")
    env = Env(tmp_path, frontend_dist=dist)
    assert env.post(_chat(NANO)).json()["object"] == "chat.completion"
    st = env.client.get("/api/gateway/status")
    assert st.headers["content-type"].startswith("application/json") and "certified" in st.json()
    assert env.client.get("/api/gateway/savings").json()["requests"] == 0
    assert "SPA" in env.client.get("/").text
    assert "SPA" in env.client.get("/some/page").text
    assert env.client.get("/api/nothing").status_code == 404


def test_oversize_body_is_413(tmp_path: Path) -> None:
    env = Env(tmp_path, gateway={"max_body_bytes": 200})
    r = env.post(_chat(NANO, pad="x" * 500))
    assert r.status_code == 413 and env.rows()[0]["status"] == "refused"


# ---- the production similarity path: embed through the upstream, compare with the index ------

EMBED_PRICED = make_config(
    prices={**make_config().prices, "fake-embed": make_config().prices[TEACHER]}
)


def indexed_router(d: Path) -> Path:
    from distillery.embed_index import EmbeddingIndex

    write_router(d)
    EmbeddingIndex.from_vectors(
        ["a", "b"], [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], model="fake-embed", dims=3
    ).save(d / "embed_index_abc.bin")
    return d


def test_index_file_is_found_and_loaded(tmp_path: Path) -> None:
    st = load_router(indexed_router(tmp_path / "r"))
    assert st.index_file is not None and st.certified
    idx = load_index(st)
    assert idx is not None and idx.model == "fake-embed" and idx.dims == 3
    assert load_index(load_router(write_router(tmp_path / "r2"))) is None
    both = indexed_router(tmp_path / "r3")
    (both / "embed_index_zzz.bin").write_bytes((both / "embed_index_abc.bin").read_bytes())
    assert load_router(both).index_file is None  # ambiguous: no guessing


def test_embedding_is_a_priced_recorded_part_of_the_request(tmp_path: Path) -> None:
    student, _ = student_for()
    env = Env(tmp_path, router=indexed_router(tmp_path / "r"), student=student, config=EMBED_PRICED)
    r = env.post(skill_body())
    d = r.json()["distillery"]
    assert r.status_code == 200 and d["served_by"] == "student"
    assert env.upstream.embed_calls == [("fake-embed", [QUESTION])]
    embed_usd = 20 * 1.0 / 1e6  # 20 prompt tokens at $1/M
    assert d["embed_cost_usd"] == pytest.approx(embed_usd)
    assert d["cost_usd"] == pytest.approx(embed_usd) and d["cost_usd"] > 0
    assert float(r.headers["x-distillery-cost-usd"]) == pytest.approx(embed_usd)
    row = env.rows()[0]
    assert row["cost_usd"] == pytest.approx(embed_usd) and row["served_by"] == "student"
    spent = env.client.get("/api/config").json()["playground"]["spent_today_usd"]
    assert spent == pytest.approx(embed_usd)  # recorded in the ledger / daily cap
    assert env.client.get("/api/gateway/status").json()["similarity_index"] is True


def test_teacher_served_request_pays_for_the_embedding_too(tmp_path: Path) -> None:
    student, _ = student_for("SELECT nope FROM missing_table")
    env = Env(tmp_path, router=indexed_router(tmp_path / "r"), student=student, config=EMBED_PRICED)
    d = env.post(skill_body()).json()["distillery"]
    teacher_usd = (1200 * 1.0 + 30 * 2.0) / 1e6
    assert d["served_by"] == "teacher"
    assert d["cost_usd"] == pytest.approx(teacher_usd + 20 * 1.0 / 1e6)
    assert d["teacher_cost_usd_counterfactual"] == pytest.approx(teacher_usd)
    s = env.client.get("/api/gateway/savings").json()
    assert s["saved_usd"] < 0  # the routing overhead made this request dearer than teacher-only


def test_embedding_refused_when_the_daily_cap_is_hit(tmp_path: Path) -> None:
    student, _ = student_for()
    cfg = make_config(prices=EMBED_PRICED.prices, playground_daily_cap_usd=1e-9)
    env = Env(tmp_path, router=indexed_router(tmp_path / "r"), student=student, config=cfg)
    r = env.post(skill_body())
    assert r.status_code == 429 and r.json()["error"]["code"] == "daily_budget_exhausted"
    assert env.upstream.embed_calls == [] and student.calls == 0 and env.upstream.calls == []
    assert env.rows()[0]["status"] == "refused"


def test_unpriced_embedding_model_routes_to_the_teacher(tmp_path: Path) -> None:
    student, _ = student_for()
    env = Env(tmp_path, router=indexed_router(tmp_path / "r"), student=student)  # no embed price
    d = env.post(skill_body()).json()["distillery"]
    assert d["served_by"] == "teacher" and d["reason"].startswith("embedding model is unpriced")
    assert env.upstream.embed_calls == [] and student.calls == 0


def test_embedding_failure_routes_to_the_teacher(tmp_path: Path) -> None:
    student, _ = student_for()
    up = FakeUpstream()
    up.embed_fail = True
    env = Env(
        tmp_path, router=indexed_router(tmp_path / "r"), student=student, config=EMBED_PRICED,
        upstream=up,
    )  # fmt: skip
    d = env.post(skill_body()).json()["distillery"]
    assert d["served_by"] == "teacher" and d["reason"] == "similarity lookup failed → teacher"
    assert d["embed_cost_usd"] == 0 and student.calls == 0


def test_dissimilar_question_short_circuits_via_the_real_index(tmp_path: Path) -> None:
    student, _ = student_for()
    up = FakeUpstream()
    up.embed_vector = [0.0, 0.0, 1.0]  # orthogonal to both indexed questions: nn_sim = 0
    env = Env(
        tmp_path, router=indexed_router(tmp_path / "r"), student=student, config=EMBED_PRICED,
        upstream=up,
    )  # fmt: skip
    d = env.post(skill_body()).json()["distillery"]
    assert d["short_circuit"] is True and d["served_by"] == "teacher" and student.calls == 0


# ---- the counterfactual never rests on an assumed output length -------------------------------


def test_counterfactual_tiers_and_exclusion(tmp_path: Path) -> None:
    student, nn = student_for()
    env = Env(
        tmp_path, router=write_router(tmp_path / "r"), student=student, nn_sim=nn, seed_llm=False
    )  # fmt: skip
    env.post(skill_body())  # nothing measured anywhere yet
    first = env.rows()[0]
    assert first["teacher_cost_usd_counterfactual"] is None
    assert first["counterfactual_basis"].startswith("unavailable")
    s = env.client.get("/api/gateway/savings").json()
    assert s["requests"] == 1 and s["requests_in_savings"] == 0 and s["requests_excluded"] == 1
    assert s["teacher_only_usd"] == 0 and s["saved_pct"] is None
    assert s["total_spent_usd"] == 0 and "EXCLUDED" in s["basis"]["teacher_only"]

    real = env.app.state.reader.real
    real.record_llm_call("sql-run", TEACHER, 1000, 60, 0.1)
    real.record_llm_call("playground", TEACHER, 1000, 9999, 0.1)  # not a pipeline call
    env.post(skill_body())
    second = env.rows()[1]
    assert (
        "60.0 mean output tokens over 1 measured teacher calls in pipeline runs"
        in (second["counterfactual_basis"])
    )
    assert second["teacher_cost_usd_counterfactual"] is not None

    env.app.state.playground.local_status = lambda role: ("unavailable", "x", None)
    env.post(skill_body())  # teacher-served: 30 output tokens logged by the gateway
    env.app.state.playground.local_status = lambda role: ("ok", None, student)
    env.post(skill_body())
    fourth = env.rows()[3]
    assert (
        "30.0 mean output tokens over 1 logged gateway teacher calls"
        in (fourth["counterfactual_basis"])
    )
    s = env.client.get("/api/gateway/savings").json()
    assert s["requests"] == 4 and s["requests_in_savings"] == 3 and s["requests_excluded"] == 1


# ---- bounded free-text slots, executor parity -------------------------------------------------


def test_oversized_schema_is_not_the_skill(tmp_path: Path) -> None:
    env = Env(tmp_path, router=write_router(tmp_path / "r"))
    big = DDL + "\n" + "-- padding\n" * 3000
    msgs = build_messages(QUESTION, big, role="eval_teacher")
    r = env.post({"model": TEACHER, "messages": msgs})
    assert r.status_code == 403 and env.upstream.calls == []


def test_sql_runs_through_the_playground_executor_when_configured(tmp_path: Path) -> None:
    from distillery.server.local_models import LocalModels
    from distillery.taskpacks.sql.runner import ExecOutcome

    class Exec:
        def __init__(self) -> None:
            self.calls: list[tuple[str, list[str]]] = []

        def run_batch(self, db_ref: str, sqls: Sequence[str]) -> list[ExecOutcome]:
            self.calls.append((db_ref, list(sqls)))
            return [ExecOutcome(True, ("n",), ((7,),))]

    ex = Exec()
    student, nn = student_for()
    env = Env(
        tmp_path, router=write_router(tmp_path / "r"), student=student, nn_sim=nn,
        playground_local=LocalModels(executor=ex),  # type: ignore[arg-type]
    )  # fmt: skip
    d = env.post(skill_body()).json()["distillery"]
    assert d["served_by"] == "student" and d["rows"] == [[7]]
    assert ex.calls and ex.calls[0][0] == "playground-demo"
