# ruff: noqa: S101, S105, S106, E501
"""W-G: the Acme analyst agent, unchanged, against (a) a fake Token Factory and (b) the gateway app,
over the plain ``openai`` SDK. Nothing touches the network."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any

import httpx
import pytest
from openai import OpenAI
from server_helpers import GOLD, MODELS
from test_gateway import LAMBDA, NANO, TEACHER, Env, FakeUpstream, student_for, write_router

from distillery.acme import AcmeAnalyst, AcmeAnswer
from distillery.prompts import SYSTEM_PROMPT

QUESTIONS = ("How many accounts are there?", "How many accounts exist in total?")
SUMMARY = "There are some accounts."


def fake_token_factory(bodies: list[dict[str, Any]]) -> httpx.Client:
    """A stand-in for Token Factory's /v1/chat/completions."""

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/chat/completions"
        body = json.loads(request.content)
        bodies.append(body)
        text = f"```sql\n{GOLD}\n```" if body["model"] == TEACHER else SUMMARY
        return httpx.Response(
            200,
            json={
                "id": "chatcmpl-tf", "object": "chat.completion", "created": 1,
                "model": body["model"],
                "choices": [{"index": 0, "finish_reason": "stop",
                             "message": {"role": "assistant", "content": text}}],
                "usage": {"prompt_tokens": 1200, "completion_tokens": 30, "total_tokens": 1230},
            },
        )  # fmt: skip

    return httpx.Client(transport=httpx.MockTransport(handler), base_url="https://tf.invalid")


def run_agent(client: httpx.Client, base_url: str) -> list[AcmeAnswer]:
    """THE agent code. The only input that differs between the two targets is the HTTP transport
    and the base URL."""
    sdk = OpenAI(base_url=base_url, api_key="sk-anything", http_client=client, max_retries=0)
    agent = AcmeAnalyst(sdk, sql_model=TEACHER, summary_model=NANO)
    return [agent.ask(q) for q in QUESTIONS]


def test_same_agent_code_against_token_factory_and_the_gateway(tmp_path: Path) -> None:
    tf_bodies: list[dict[str, Any]] = []
    tf_answers = run_agent(fake_token_factory(tf_bodies), "https://tf.invalid/v1")

    student, nn = student_for()
    env = Env(
        tmp_path, router=write_router(tmp_path / "router"), student=student, nn_sim=nn,
        upstream=FakeUpstream({NANO: SUMMARY}),
    )  # fmt: skip
    gw_bodies: list[dict[str, Any]] = []
    env.client.event_hooks["request"].append(
        lambda r: gw_bodies.append(json.loads(r.content)) if r.method == "POST" else None
    )
    gw_answers = run_agent(env.client, "http://testserver/v1")

    # identical code path: the very same requests, byte for byte, in the same order
    assert len(tf_bodies) == len(gw_bodies) == 2 * len(QUESTIONS)
    assert tf_bodies == gw_bodies
    assert [b["model"] for b in gw_bodies] == [TEACHER, NANO] * len(QUESTIONS)
    assert gw_bodies[0]["messages"][0] == {"role": "system", "content": SYSTEM_PROMPT}
    assert gw_bodies[1]["max_tokens"] <= 256  # the summary step fits the gateway's clamp

    # identical answers for the user
    for tf, gw in zip(tf_answers, gw_answers, strict=True):
        assert (tf.sql, tf.columns, tf.rows, tf.summary, tf.error) == (
            gw.sql, gw.columns, gw.rows, gw.summary, gw.error,
        )  # fmt: skip
        assert gw.error is None and gw.rows[0][0] > 0 and gw.summary == SUMMARY
    assert all(a.routing is None for a in tf_answers)
    assert all(a.routing and a.routing["served_by"] == "student" for a in gw_answers)
    assert all(a.routing and a.routing["lambda"] == LAMBDA for a in gw_answers)

    # 100% of the gateway's calls are logged, with usage, and what the client saw is what is logged
    rows = env.rows()
    assert len(rows) == len(gw_bodies) == 4
    assert [r["status"] for r in rows] == ["ok"] * 4
    assert [r["served_by"] for r in rows] == ["student", "passthrough"] * 2
    for r in rows:
        assert r["input_tokens"] > 0 and r["output_tokens"] > 0 and r["prompt_sha256"]
    for r, a in zip(rows[::2], gw_answers, strict=True):
        assert (r["input_tokens"], r["output_tokens"]) == a.sql_usage
    for r, a in zip(rows[1::2], gw_answers, strict=True):  # real provider usage + real spend
        assert (r["input_tokens"], r["output_tokens"]) == a.summary_usage == (1200, 30)
        assert r["cost_usd"] == pytest.approx((1200 * 1.0 + 30 * 2.0) / 1e6)
    assert env.client.get("/api/gateway/savings").json()["requests"] == 2


def test_agent_reports_bad_sql_without_a_summary_call(tmp_path: Path) -> None:
    bodies: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        bodies.append(json.loads(request.content))
        msg = {"role": "assistant", "content": "SELECT nope FROM missing"}
        return httpx.Response(
            200, json={"id": "x", "object": "chat.completion", "created": 1, "model": TEACHER,
                       "choices": [{"index": 0, "finish_reason": "stop", "message": msg}],
                       "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}},
        )  # fmt: skip

    http = httpx.Client(transport=httpx.MockTransport(handler), base_url="https://tf.invalid")
    sdk = OpenAI(base_url="https://tf.invalid/v1", api_key="k", http_client=http, max_retries=0)
    ans = AcmeAnalyst(sdk, sql_model=TEACHER, summary_model=MODELS["triage"]).ask("q?")
    assert ans.error and "query failed" in ans.error and ans.summary is None
    assert len(bodies) == 1  # no summary call for a failed query


def test_cli_script_runs_the_same_agent(monkeypatch: pytest.MonkeyPatch, capsys: Any) -> None:
    spec = importlib.util.spec_from_file_location(
        "acme_agent_script", Path(__file__).resolve().parents[1] / "scripts" / "acme_agent.py"
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    tf_bodies: list[dict[str, Any]] = []
    http = fake_token_factory(tf_bodies)
    seen: dict[str, Any] = {}

    def fake_openai(base_url: str, api_key: str) -> OpenAI:
        seen.update(base_url=base_url, api_key=api_key)
        return OpenAI(base_url=base_url, api_key=api_key, http_client=http, max_retries=0)

    monkeypatch.setattr(mod, "OpenAI", fake_openai)
    code = mod.main(
        ["--base-url", "https://x.invalid/v1", "--api-key", "k", "--sql-model", TEACHER,
         "--summary-model", NANO, "--question", QUESTIONS[0]]
    )  # fmt: skip
    assert code == 0 and seen == {"base_url": "https://x.invalid/v1", "api_key": "k"}
    assert [b["model"] for b in tf_bodies] == [TEACHER, NANO]
    assert SUMMARY in capsys.readouterr().out
    with pytest.raises(SystemExit):
        mod.main(["--api-key", "k"])  # missing --base-url etc.
