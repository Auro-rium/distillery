# ruff: noqa: S101
"""LLMClient.embed: batching, order, retry reuse, usage recording, price semantics."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import httpx2
import openai
import pytest

from distillery.budget import Ledger, UnknownPriceError
from distillery.config import Price
from distillery.llm import CallRecord, LLMClient, RetriesExhaustedError, TerminalLLMError

_REQ = httpx2.Request("POST", "https://example.invalid/v1/embeddings")
MODEL = "Qwen/Qwen3-Embedding-8B"


def status_error(code: int) -> openai.APIStatusError:
    return openai.APIStatusError(
        f"http {code}", response=httpx2.Response(code, request=_REQ), body=None
    )


class FakeEmbeddings:
    """Vector of text i is [len(text), 1.0]; ``shuffle`` returns rows in reverse order."""

    def __init__(self, script: list[Any] | None = None, shuffle: bool = False) -> None:
        self.script = list(script or [])
        self.calls: list[dict[str, Any]] = []
        self.shuffle = shuffle
        self.embeddings = SimpleNamespace(create=self._create)

    async def _create(self, **kw: Any) -> Any:
        self.calls.append(kw)
        if self.script:
            item = self.script.pop(0)
            if isinstance(item, BaseException):
                raise item
        data = [
            SimpleNamespace(index=i, embedding=[float(len(t)), 1.0])
            for i, t in enumerate(kw["input"])
        ]
        if self.shuffle:
            data.reverse()
        n = sum(len(t) for t in kw["input"])
        return SimpleNamespace(data=data, usage=SimpleNamespace(prompt_tokens=n, total_tokens=n))


class Sink:
    def __init__(self) -> None:
        self.records: list[CallRecord] = []

    def record_call(self, record: CallRecord) -> None:
        self.records.append(record)


def make(tr: FakeEmbeddings, **kw: Any) -> tuple[LLMClient, Sink, list[float]]:
    sleeps: list[float] = []

    async def sleep(s: float) -> None:
        sleeps.append(s)

    sink = Sink()
    c = LLMClient(
        tr, {}, sink=sink, pricing=lambda m, i, o: i * 1e-8 + o * 5.0, sleep=sleep,
        jitter=lambda: 1.0, **kw,
    )  # fmt: skip
    return c, sink, sleeps


def test_batches_of_64_order_preserved_and_usage_recorded() -> None:
    tr = FakeEmbeddings(shuffle=True)
    c, sink, _ = make(tr)
    texts = ["x" * (i + 1) for i in range(130)]
    vecs = asyncio.run(c.embed(texts, model=MODEL, purpose="router_embed"))
    assert [len(x["input"]) for x in tr.calls] == [64, 64, 2]
    assert all(x["model"] == MODEL for x in tr.calls)
    assert [v[0] for v in vecs] == [float(i + 1) for i in range(130)]  # index order, not wire order
    assert len(sink.records) == 3
    r = sink.records[0]
    assert (r.role, r.model, r.purpose, r.ok, r.output_tokens) == (
        "embedding", MODEL, "router_embed", True, 0,
    )  # fmt: skip
    assert r.input_tokens == sum(range(1, 65))
    assert r.cost_usd == pytest.approx(r.input_tokens * 1e-8)  # prompt tokens only, no output price
    assert sum(x.input_tokens for x in sink.records) == sum(range(1, 131))


def test_empty_and_bad_batch_size() -> None:
    tr = FakeEmbeddings()
    c, sink, _ = make(tr)
    assert asyncio.run(c.embed([], model=MODEL, purpose="p")) == []
    assert tr.calls == [] and sink.records == []
    with pytest.raises(ValueError):
        asyncio.run(c.embed(["a"], model=MODEL, purpose="p", batch_size=0))


def test_retry_policy_shared_with_chat() -> None:
    tr = FakeEmbeddings(script=[status_error(503), status_error(429)])
    c, sink, sleeps = make(tr)
    vecs = asyncio.run(c.embed(["abc"], model=MODEL, purpose="p"))
    assert vecs == [[3.0, 1.0]] and len(tr.calls) == 3
    assert [r.ok for r in sink.records] == [False, False, True]
    assert len(sleeps) == 2
    tr = FakeEmbeddings(script=[status_error(400)])
    c, _, _ = make(tr)
    with pytest.raises(TerminalLLMError):
        asyncio.run(c.embed(["abc"], model=MODEL, purpose="p"))
    assert len(tr.calls) == 1
    tr = FakeEmbeddings(script=[status_error(500)] * 9)
    c, _, _ = make(tr)
    with pytest.raises(RetriesExhaustedError):
        asyncio.run(c.embed(["abc"], model=MODEL, purpose="p"))


def test_wrong_vector_count_is_terminal() -> None:
    tr = FakeEmbeddings()

    async def short(**kw: Any) -> Any:
        return SimpleNamespace(data=[], usage=SimpleNamespace(prompt_tokens=1, total_tokens=1))

    tr.embeddings.create = short
    c, _, _ = make(tr)
    with pytest.raises(TerminalLLMError):
        asyncio.run(c.embed(["a"], model=MODEL, purpose="p"))


def test_unknown_price_refuses_via_ledger_pricing(tmp_path: Path) -> None:
    ledger = Ledger("r", 1.0, 1.0, 1.0, prices={})
    tr = FakeEmbeddings()
    c = LLMClient(tr, {}, sink=Sink(), pricing=ledger.estimate_llm_cost)
    with pytest.raises(UnknownPriceError):
        asyncio.run(c.embed(["a"], model=MODEL, purpose="p"))


def test_prices_file_has_embedding_model() -> None:
    raw = json.loads((Path(__file__).parents[1] / "deploy" / "prices.json").read_text())
    p = Price.model_validate(raw[MODEL])
    assert (p.input_per_mtok, p.output_per_mtok) == (0.01, 0.0)
    ledger = Ledger("r", 1.0, 1.0, 1.0, prices={MODEL: p})
    assert ledger.estimate_llm_cost(MODEL, 1_000_000, 0) == pytest.approx(0.01)
