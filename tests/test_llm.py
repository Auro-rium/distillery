# ruff: noqa: S101, S604
from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any

import httpx2
import openai
import pytest
from pydantic import BaseModel

from distillery.llm import (
    CallRecord,
    LLMClient,
    RetriesExhaustedError,
    RetryPolicy,
    SchemaValidationError,
    TerminalLLMError,
    backoff_delay,
    prompt_hash,
    split_reasoning,
)

_REQ = httpx2.Request("POST", "https://example.invalid/v1/chat/completions")


def status_error(code: int, headers: dict[str, str] | None = None) -> openai.APIStatusError:
    resp = httpx2.Response(code, request=_REQ, headers=headers or {})
    return openai.APIStatusError(f"http {code}", response=resp, body=None)


def completion(content: str, *, reasoning: str | None = None, pt: int = 10, ct: int = 5) -> Any:
    msg = SimpleNamespace(content=content, refusal=None, reasoning_content=reasoning)
    return SimpleNamespace(
        choices=[SimpleNamespace(message=msg, finish_reason="stop")],
        usage=SimpleNamespace(prompt_tokens=pt, completion_tokens=ct),
    )


class FakeTransport:
    def __init__(self, script: list[Any]) -> None:
        self.script = list(script)
        self.calls: list[dict[str, Any]] = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    async def _create(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        item = self.script.pop(0)
        if isinstance(item, BaseException):
            raise item
        return item


class Sink:
    def __init__(self) -> None:
        self.records: list[CallRecord] = []

    def record_call(self, record: CallRecord) -> None:
        self.records.append(record)


class Answer(BaseModel):
    value: int


MODELS = {"planner": "m-plan", "teacher": "m-teach", "triage": "m-tri", "student": "m-stu"}


def make(script: list[Any], **kw: Any) -> tuple[LLMClient, FakeTransport, Sink, list[float]]:
    sleeps: list[float] = []

    async def sleep(s: float) -> None:
        sleeps.append(s)

    tr = FakeTransport(script)
    sink = Sink()
    client = LLMClient(
        tr,
        MODELS,
        sink=sink,
        pricing=lambda m, i, o: (i + o) * 0.001,
        sleep=sleep,
        jitter=lambda: 1.0,
        **kw,
    )
    return client, tr, sink, sleeps


def run(coro: Any) -> Any:
    return asyncio.run(coro)


def test_role_routing_and_logging() -> None:
    client, tr, sink, _ = make([completion("hi")])
    res = run(client.chat("teacher", [{"role": "user", "content": "x"}], purpose="p", stage="s1"))
    assert tr.calls[0]["model"] == "m-teach"
    assert res.text == "hi" and res.parsed is None
    rec = sink.records[0]
    assert (rec.model, rec.purpose, rec.stage) == ("m-teach", "p", "s1")
    assert (rec.input_tokens, rec.output_tokens) == (10, 5)
    assert rec.cost_usd == pytest.approx(0.015)
    assert rec.prompt_sha256 == prompt_hash([{"role": "user", "content": "x"}])


def test_unknown_role_is_terminal() -> None:
    client, *_ = make([])
    with pytest.raises(TerminalLLMError):
        run(client.chat("nope", [], purpose="p"))  # type: ignore[call-overload]


def test_reasoning_separated_field_and_think_tags() -> None:
    client, *_ = make([completion("answer", reasoning="deep thoughts")])
    res = run(client.chat("planner", [], purpose="p"))
    assert res.text == "answer" and res.reasoning == "deep thoughts"
    text, reasoning = split_reasoning(SimpleNamespace(content="<think>a</think>final"))
    assert (text, reasoning) == ("final", "a")
    text, reasoning = split_reasoning(SimpleNamespace(content="ok<think>never closed"))
    assert text == "ok" and reasoning == "never closed"


def test_json_schema_parsed_and_response_format() -> None:
    client, tr, *_ = make([completion('{"value": 3}')])
    res = run(
        client.chat("triage", [{"role": "user", "content": "q"}], purpose="p", json_schema=Answer)
    )
    assert res.parsed == Answer(value=3)
    rf = tr.calls[0]["response_format"]
    assert rf["type"] == "json_schema" and rf["json_schema"]["schema"]["title"] == "Answer"


def test_schema_retry_then_success_and_cap() -> None:
    client, tr, sink, _ = make([completion("nope"), completion('{"value": 1}')])
    res = run(client.chat("triage", [], purpose="p", json_schema=Answer))
    assert res.parsed == Answer(value=1) and res.attempts == 2
    assert len(sink.records) == 2
    # the retry carries the validation error back to the model
    assert "did not validate" in tr.calls[1]["messages"][-1]["content"]

    client, tr, *_ = make([completion("bad")] * 10)
    with pytest.raises(SchemaValidationError):
        run(client.chat("triage", [], purpose="p", json_schema=Answer, max_retries=99))
    assert len(tr.calls) == 3  # 1 + capped 2 retries


def test_transient_5xx_retried_with_backoff_then_ok() -> None:
    client, tr, sink, sleeps = make([status_error(503), status_error(500), completion("ok")])
    res = run(client.chat("planner", [], purpose="p"))
    assert res.text == "ok" and len(tr.calls) == 3
    assert sleeps == [1.0, 2.0]
    assert [r.ok for r in sink.records] == [False, False, True]


def test_429_honours_retry_after() -> None:
    client, _, _, sleeps = make([status_error(429, {"retry-after": "7"}), completion("ok")])
    run(client.chat("planner", [], purpose="p"))
    assert sleeps == [7.0]


def test_4xx_is_terminal_no_retry() -> None:
    client, tr, _, sleeps = make([status_error(400), completion("never")])
    with pytest.raises(TerminalLLMError):
        run(client.chat("planner", [], purpose="p"))
    assert len(tr.calls) == 1 and sleeps == []


def test_retries_are_capped() -> None:
    client, tr, _, _ = make(
        [status_error(502)] * 10, retry=RetryPolicy(max_attempts=3, base_delay_s=0.1)
    )
    with pytest.raises(RetriesExhaustedError):
        run(client.chat("planner", [], purpose="p"))
    assert len(tr.calls) == 3


def test_timeout_is_retryable() -> None:
    client, tr, *_ = make([openai.APITimeoutError(request=_REQ), completion("ok")])
    assert run(client.chat("planner", [], purpose="p")).text == "ok"
    assert len(tr.calls) == 2


def test_backoff_is_bounded() -> None:
    p = RetryPolicy(max_attempts=9, base_delay_s=1.0, max_delay_s=10.0)
    assert backoff_delay(20, p, 1.0) == 10.0
    assert 0.5 <= backoff_delay(1, p, 0.0) <= 1.0
