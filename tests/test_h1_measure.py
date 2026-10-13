# ruff: noqa: S101, S105, S106
from __future__ import annotations

import json
import sqlite3
import threading
from collections.abc import Sequence
from pathlib import Path

import pytest

from distillery import h1_measure as hm
from distillery.prompts import SYSTEM_PROMPT, build_messages
from distillery.store import Store
from distillery.taskpacks.sql.executor import LocalExecutor

DDL = "CREATE TABLE t (id INTEGER, v TEXT);"


def make_db() -> bytes:
    con = sqlite3.connect(":memory:")
    con.executescript(DDL + "INSERT INTO t VALUES (1,'a'),(2,'b'),(3,'c');")
    data = con.serialize()
    con.close()
    return bytes(data)


def items(n: int = 4) -> list[dict[str, object]]:
    golds = [
        "SELECT v FROM t WHERE id = 1",
        "SELECT v FROM t WHERE id = 2",
        "SELECT COUNT(*) FROM t",
        "SELECT v FROM t ORDER BY id",
    ]
    return [
        {"question": f"q{i}", "schema_ddl": DDL, "gold_sql": golds[i], "requires_order": i == 3}
        for i in range(n)
    ]


class FakeChat:
    """Deterministic: latency depends only on the prompt index parsed from the question."""

    def __init__(self, fail_on: Sequence[int] = (), tokens: int = 10) -> None:
        self.fail_on, self.tokens = set(fail_on), tokens
        self.lock = threading.Lock()
        self.calls = 0

    def idx(self, messages: hm.ChatMessages) -> int:
        return int(messages[1]["content"].split("Question: q")[1].split("\n")[0])

    def chat(self, messages: hm.ChatMessages, *, max_tokens: int) -> hm.ChatResponse:
        i = self.idx(messages)
        with self.lock:
            self.calls += 1
        if i in self.fail_on:
            raise hm.ChatError("HTTP 500")
        return hm.ChatResponse(f"SELECT {i}", self.tokens, 0.1 * (i + 1))

    def lists_model(self) -> bool:
        return True


class SteppedClock:
    """First call 0.0, then +``step`` per call: with one pool per level, wall = step."""

    def __init__(self, step: float) -> None:
        self.t, self.step, self.first = 0.0, step, True

    def __call__(self) -> float:
        if self.first:
            self.first = False
            return self.t
        self.t += self.step
        return self.t


# ---------------------------------------------------------------- prompts


def test_prompts_use_the_canonical_eval_messages() -> None:
    p = hm.build_prompts(items(2))
    assert p[0] == build_messages("q0", DDL, role="eval_student")
    assert p[0][0]["content"] == SYSTEM_PROMPT and p[0][1]["content"].endswith("/no_think")
    own = [{"role": "user", "content": "x"}]
    assert hm.build_prompts([{"messages": own}]) == [own]
    with pytest.raises(ValueError, match="needs 'question'"):
        hm.build_prompts([{"question": "only"}])


def test_load_items_json_and_jsonl(tmp_path: Path) -> None:
    rows = items(2)
    a, b = tmp_path / "a.json", tmp_path / "b.jsonl"
    a.write_text(json.dumps(rows))
    b.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    assert hm.load_items(a) == rows == hm.load_items(b)
    c = tmp_path / "o.json"
    c.write_text(json.dumps(["x", {"output": "y"}, {"text": "z"}]))
    assert hm.load_outputs(c) == ["x", "y", "z"]


# ---------------------------------------------------------------- throughput


def test_throughput_stats_are_exact_on_a_deterministic_client() -> None:
    prompts = hm.build_prompts(items(4))
    res = hm.measure_throughput(FakeChat(), prompts, (1, 4), 160, monotonic=SteppedClock(2.0))
    assert [lv.concurrency for lv in res.levels] == [1, 4] and not res.stopped_early
    lv = res.levels[0]
    assert (lv.requests, lv.ok, lv.errors, lv.skipped) == (4, 4, 0, 0)
    assert lv.wall_s == pytest.approx(2.0) and lv.req_per_s == pytest.approx(2.0)
    # latencies 0.1, 0.2, 0.3, 0.4: nearest-rank p50 = 0.2, p95 = 0.4
    assert lv.latency_p50_s == pytest.approx(0.2) and lv.latency_p95_s == pytest.approx(0.4)
    assert lv.output_tokens == 40 and lv.output_tokens_per_s == pytest.approx(20.0)
    assert len(res.samples) == 8 and {s["level"] for s in res.samples} == {1, 4}
    assert res.outputs_at(4, 4) == ["SELECT 0", "SELECT 1", "SELECT 2", "SELECT 3"]
    js = res.to_json()
    assert js["levels"][1]["concurrency"] == 4 and js["n_samples"] == 8


def test_throughput_counts_errors_and_excludes_them_from_latency() -> None:
    res = hm.measure_throughput(
        FakeChat(fail_on=[3]), hm.build_prompts(items(4)), (2,), monotonic=SteppedClock(1.0)
    )
    lv = res.levels[0]
    assert (lv.requests, lv.ok, lv.errors) == (4, 3, 1) and lv.first_error == "HTTP 500"
    assert lv.latency_p95_s == pytest.approx(0.3) and lv.req_per_s == pytest.approx(3.0)
    assert res.outputs_at(2, 4) is None  # a failed request: no complete output list
    assert hm.best_req_per_hour(res) is None  # a level with errors never sets the headline


def test_throughput_stops_when_the_watchdog_fires() -> None:
    chat = FakeChat()
    stop = {"now": False}

    class StoppingChat(FakeChat):
        def chat(self, messages: hm.ChatMessages, *, max_tokens: int) -> hm.ChatResponse:
            r = chat.chat(messages, max_tokens=max_tokens)
            if chat.calls >= 2:
                stop["now"] = True  # the time box expires after the second request
            return r

    res = hm.measure_throughput(
        StoppingChat(), hm.build_prompts(items(4)), (1, 8, 32),
        monotonic=SteppedClock(1.0), should_stop=lambda: stop["now"],
    )  # fmt: skip
    assert res.stopped_early
    assert len(res.levels) == 1  # no further level after the stop
    lv = res.levels[0]
    assert lv.requests == 2 and lv.skipped == 2 and chat.calls == 2


def test_throughput_does_not_start_when_already_expired() -> None:
    chat = FakeChat()
    res = hm.measure_throughput(chat, hm.build_prompts(items(2)), (1,), should_stop=lambda: True)
    assert res.stopped_early and res.levels == [] and chat.calls == 0
    with pytest.raises(ValueError):
        hm.measure_throughput(chat, [], (1,))
    with pytest.raises(ValueError):
        hm.measure_throughput(chat, hm.build_prompts(items(1)), (0,))


# ---------------------------------------------------------------- real client (fake transport)


def test_http_chat_client_request_shape_and_secrecy() -> None:
    sent: list[tuple[str, str, dict[str, str], bytes | None]] = []

    def send(method: str, url: str, headers, body, timeout):  # type: ignore[no-untyped-def]
        sent.append((method, url, dict(headers), body))
        if url.endswith("/models"):
            return 200, json.dumps({"data": [{"id": "Qwen/Qwen3-1.7B"}, {"id": "n1"}]}).encode()
        if body and b"boom" in body:
            return 500, b"upstream says TOKEN-XYZ"
        reply = {
            "choices": [{"message": {"content": "SELECT 1"}}],
            "usage": {"completion_tokens": 7},
        }
        return 200, json.dumps(reply).encode()

    ticks = iter([10.0, 10.25, 11.0, 11.5])
    c = hm.HttpChatClient("https://h/v1/", "TOKEN-XYZ", send=send, monotonic=lambda: next(ticks))
    r = c.chat([{"role": "user", "content": "hi"}], max_tokens=160)
    assert r == hm.ChatResponse("SELECT 1", 7, 0.25)
    method, url, headers, body = sent[0]
    assert (method, url) == ("POST", "https://h/v1/chat/completions")
    assert headers["Authorization"] == "Bearer TOKEN-XYZ"
    payload = json.loads(body or b"")
    assert payload["model"] == "n1" and payload["temperature"] == 0 and payload["max_tokens"] == 160
    assert "TOKEN-XYZ" not in repr(c) and c.lists_model()
    with pytest.raises(hm.ChatError, match="HTTP 500") as ei:
        c.chat([{"role": "user", "content": "boom"}], max_tokens=1)
    assert "TOKEN-XYZ" not in str(ei.value)


def test_lists_model_is_false_until_the_adapter_is_listed() -> None:
    def send(method, url, headers, body, timeout):  # type: ignore[no-untyped-def]
        return 200, json.dumps({"data": [{"id": "Qwen/Qwen3-1.7B"}]}).encode()

    assert not hm.HttpChatClient("https://h/v1", "t", send=send).lists_model()

    def down(method, url, headers, body, timeout):  # type: ignore[no-untyped-def]
        raise hm.ChatError("connection refused")

    assert not hm.HttpChatClient("https://h/v1", "t", send=down).lists_model()


# ---------------------------------------------------------------- parity


def test_parity_and_accuracy_math() -> None:
    its = items(4)
    ex = LocalExecutor({"db": make_db()})
    endpoint = [
        "SELECT v FROM t WHERE id = 1",  # right
        "```sql\nSELECT v   FROM t\nWHERE id = 2;\n```",  # right, equal to sandbox after normalising
        "SELECT 99",  # wrong
        "no sql here",  # unparseable
    ]
    sandbox = [
        "SELECT v FROM t WHERE id = 1",  # identical to endpoint, right
        "SELECT v FROM t WHERE id = 2",  # identical after whitespace normalisation, right
        "SELECT COUNT(*) FROM t",  # right (endpoint wrong)
        "SELECT v FROM t ORDER BY id",  # right (endpoint unparseable)
    ]
    p = hm.parity(endpoint, sandbox, its, executor=ex, db_ref="db")
    assert p["n"] == 4 and p["identical_sql_rate"] == pytest.approx(0.5)
    assert p["identical_output_rate"] == pytest.approx(0.25)  # raw: only item 0 byte-identical
    assert p["endpoint_accuracy"] == pytest.approx(0.5) and p["sandbox_accuracy"] == 1.0
    assert p["accuracy_delta"] == pytest.approx(-0.5)
    assert p["agreement"] == pytest.approx(0.5)  # items 0,1 agree (both right); 2,3 disagree
    assert (p["both_correct"], p["only_endpoint_correct"], p["only_sandbox_correct"]) == (2, 0, 2)
    assert p["endpoint_unparseable"] == 1 and p["sandbox_unparseable"] == 0


def test_parity_honours_row_order_requirement_and_checks_lengths() -> None:
    its = items(4)
    ex = LocalExecutor({"db": make_db()})
    outs = [str(i["gold_sql"]) for i in its]
    outs[3] = "SELECT v FROM t ORDER BY id DESC"  # same rows, wrong order, order required
    acc = hm.accuracy(outs, its, executor=ex, db_ref="db")
    assert acc["correct"] == [True, True, True, False] and acc["accuracy"] == 0.75
    with pytest.raises(ValueError):
        hm.parity(outs, outs[:2], its, executor=ex, db_ref="db")
    assert hm.normalise_sql(" SELECT  a\n FROM t ") == "SELECT a FROM t"
    assert hm.normalise_sql(None) is None


# ---------------------------------------------------------------- cost / break-even


def test_cost_per_1k() -> None:
    c = hm.cost_per_1k(2.0, 4000.0)  # $2/h at 4000 req/h
    assert c["usd_per_1k"] == pytest.approx(0.5) and c["projected"] is True
    assert hm.cost_per_1k(2.0, 4000.0, 0.5)["usd_per_1k"] == pytest.approx(1.0)
    with pytest.raises(ValueError):
        hm.cost_per_1k(2.0, 0.0)
    with pytest.raises(ValueError):
        hm.cost_per_1k(2.0, 10.0, 1.5)


def test_best_req_per_hour_uses_the_highest_clean_level() -> None:
    lv = lambda c, rps, errs: hm.LevelStats(  # noqa: E731
        c, 10, 10 - errs, errs, 0, 1.0, rps, 0.1, 0.2, 100, 100.0
    )
    res = hm.MeasureResult([lv(1, 2.0, 0), lv(8, 9.0, 0), lv(32, 20.0, 1)], [], False)
    assert hm.best_req_per_hour(res) == pytest.approx(9.0 * 3600)


def test_blended_per_1k_and_breakeven() -> None:
    # 80% coverage, student $0.10/1k, teacher $2.00/1k, 10% of student traffic audited
    b = hm.blended_per_1k(0.8, 0.10, 2.0, 0.10)
    assert b == pytest.approx(0.8 * 0.10 + 0.2 * 2.0 + 0.8 * 0.1 * 2.0)  # 0.64
    # saving per 1k = 2.0 - 0.64 = 1.36; an always-on $48/day host breaks even at 48/1.36*1000
    assert hm.breakeven_volume(0.8, 0.0, 2.0, 0.10, 48.0) == pytest.approx(
        48.0 * 1000 / (2.0 - 0.56)
    )
    # a student that costs more than the (non-audited) teacher saves nothing: never breaks even
    assert hm.breakeven_volume(0.8, 3.0, 2.0, 0.0, 10.0) is None
    with pytest.raises(ValueError):
        hm.blended_per_1k(1.2, 0.1, 2.0, 0.1)


def test_blended_curve_points_are_projected_and_capacity_flagged() -> None:
    curve = hm.blended_curve(
        0.5, 0.0, 2.0, 0.0, [1_000, 100_000],
        student_fixed_daily_usd=24.0, student_capacity_per_day=10_000.0,
    )  # fmt: skip
    assert curve["projected"] is True and curve["breakeven_never"] is False
    assert curve["blended_per_1k"] == pytest.approx(1.0)
    assert curve["breakeven_volume_per_day"] == pytest.approx(24_000.0)  # 24 / (2-1) per 1k
    lo, hi = curve["points"]
    assert lo["teacher_only_usd"] == pytest.approx(2.0) and lo["blended_usd"] == pytest.approx(25.0)
    assert lo["savings_usd"] < 0 and lo["student_capacity_ok"] is True
    assert hi["savings_usd"] == pytest.approx(200.0 - 124.0) and hi["student_capacity_ok"] is False
    never = hm.blended_curve(0.5, 5.0, 2.0, 0.0, [10])
    assert never["breakeven_never"] is True and never["breakeven_volume_per_day"] is None


def test_teacher_cost_per_1k_reads_the_ledger_rows(tmp_path: Path) -> None:
    st = Store(tmp_path)
    for i in range(1, 11):
        st.record_llm_call("r", "teacher-m", 100, 10, 0.002, i / 10)
    st.close()
    t = hm.teacher_cost_per_1k(tmp_path, "teacher-m")
    assert t["usd_per_1k"] == pytest.approx(2.0) and t["n_calls"] == 10 and t["projected"] is False
    assert t["latency_p50_s"] == pytest.approx(0.5)
