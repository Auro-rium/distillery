# ruff: noqa: S101, S105, S106
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import pytest
from aicloud_helpers import TOKEN, FakeClock, FakeNebius
from test_h1_measure import FakeChat, items, make_db

from distillery.aicloud_endpoint import EndpointHandle
from distillery.budget import Ledger
from distillery.h1_measure import ChatResponse
from distillery.store import Store

SPEC = importlib.util.spec_from_file_location(
    "h1_endpoint", Path(__file__).resolve().parents[1] / "scripts" / "h1_endpoint.py"
)
assert SPEC and SPEC.loader
h1 = importlib.util.module_from_spec(SPEC)
sys.modules["h1_endpoint"] = h1  # dataclasses resolve their module by name
SPEC.loader.exec_module(h1)

PLAN_ARGS = [
    "--vllm-tag", "v0.0.test", "--hourly-usd", "2.0", "--adapter-s3", "s3://bkt/n1",
    "--max-lora-rank", "32", "--name", "t-ep",
]  # fmt: skip


class ExplodingRunner:
    def run(self, argv):  # type: ignore[no-untyped-def]
        raise AssertionError(f"the CLI must not run here: {argv}")


def run_main(argv: list[str], **kw: Any) -> tuple[int, list[str]]:
    lines: list[str] = []
    rc = h1.main(argv, out=lines.append, **kw)
    return rc, lines


def test_plan_only_prints_commands_and_estimate_without_running_anything() -> None:
    rc, lines = run_main([*PLAN_ARGS, "--plan-only"], runner=ExplodingRunner())
    text = "\n".join(lines)
    assert rc == 0
    assert "plan-only: nothing was run, no network, no spend" in text
    assert "preflight estimate $1.5000 (= $2.0000/h x 45/60)" in text
    assert (
        "create  nebius ai endpoint create --image vllm/vllm-openai:v0.0.test --name t-ep" in text
    )
    assert "--volume s3://bkt/n1:/adapters/n1:ro" in text and "--max-lora-rank 32" in text
    assert "--token '***'" in text and "--auth token" in text
    assert "get     nebius ai endpoint get --id '<ID>' --format json" in text
    assert "delete  nebius ai endpoint delete --id '<ID>'" in text
    assert "(operator-supplied)" in text


def test_plan_only_never_leaks_a_token_even_with_an_explicit_one() -> None:
    rc, lines = run_main([*PLAN_ARGS, "--plan-only"])
    plan = h1.plan_from_args(h1.build_parser().parse_args([*PLAN_ARGS]))
    assert rc == 0 and plan.auth_token not in "\n".join(lines)
    assert TOKEN not in h1.plan_only_text(h1.ae.EndpointPlan(
        vllm_tag="v1", hourly_usd=1.0, adapter_s3_source="s3://b", auth_token=TOKEN))  # fmt: skip


def test_plan_only_with_a_ledger_reports_headroom_and_refusal(tmp_path: Path) -> None:
    ok = Ledger("h1", 10.0, 19.5, 1.0, store=Store(tmp_path))
    rc, lines = run_main([*PLAN_ARGS, "--plan-only"], ledger=ok)
    assert rc == 0 and "ledger headroom    $10.0000" in "\n".join(lines)
    assert "preflight          would PASS" in lines[-1]
    tight = Ledger("h1", 1.0, 19.5, 1.0)
    rc, lines = run_main([*PLAN_ARGS, "--plan-only"], ledger=tight)
    assert rc == 0 and "would be REFUSED" in lines[-1] and "run cap" in lines[-1]


def test_invalid_plan_is_refused() -> None:
    rc, lines = run_main(["--vllm-tag", "latest", "--hourly-usd", "1", "--adapter-s3", "s3://b",
                          "--plan-only"])  # fmt: skip
    assert rc == 3 and "refused" in lines[0]


def test_live_path_requires_explicit_spend_approval() -> None:
    rc, lines = run_main(PLAN_ARGS, runner=ExplodingRunner(), ledger=Ledger("h1", 10, 19.5, 1.0))
    assert rc == 3 and "--i-approve-spend" in lines[0]
    rc, lines = run_main([*PLAN_ARGS, "--i-approve-spend"], runner=ExplodingRunner())
    assert rc == 3 and "needs --root" in lines[0]


# ---------------------------------------------------------------- live runs on fakes


def write_inputs(tmp: Path, n: int = 4) -> list[str]:
    its = items(n)
    (tmp / "prompts.jsonl").write_text("\n".join(json.dumps(i) for i in its) + "\n")
    (tmp / "db.sqlite").write_bytes(make_db())
    # the sandbox student: right everywhere except item 1
    sandbox = [str(i["gold_sql"]) for i in its]
    sandbox[1] = "SELECT 'zzz'"
    (tmp / "sandbox.json").write_text(json.dumps(sandbox))
    return [
        "--prompts", str(tmp / "prompts.jsonl"), "--sandbox-outputs", str(tmp / "sandbox.json"),
        "--db", str(tmp / "db.sqlite"), "--out-dir", str(tmp / "out"),
        "--concurrency", "1,2", "--root", str(tmp / "store"),
    ]  # fmt: skip


class GoldChat(FakeChat):
    """Answers each prompt with its gold SQL so accuracy is knowable."""

    def __init__(self, golds: list[str], clock: FakeClock | None = None, step: float = 0.0) -> None:
        super().__init__()
        self.golds, self.clock, self.step = golds, clock, step

    def chat(self, messages, *, max_tokens):  # type: ignore[no-untyped-def]
        i = self.idx(messages)
        with self.lock:
            self.calls += 1
        if self.clock is not None:
            self.clock.t += self.step
        return ChatResponse(self.golds[i], 10, 0.1 * (i + 1))


def live(tmp: Path, cli: FakeNebius, fc: FakeClock, extra: list[str] | None = None, chat=None):  # type: ignore[no-untyped-def]
    store = Store(tmp / "store")
    lg = Ledger("h1", 10.0, 19.5, 1.0, store=store)
    golds = [str(i["gold_sql"]) for i in items(4)]
    chat = chat or GoldChat(golds)
    argv = [*PLAN_ARGS, *write_inputs(tmp), "--i-approve-spend", *(extra or [])]
    rc, lines = run_main(
        argv, runner=cli, ledger=lg, clock=fc.clock(), client_factory=lambda h: chat
    )
    return rc, lines, lg, json.loads((tmp / "out" / "h1_endpoint.json").read_text())


def test_live_happy_path_writes_the_evidence(tmp_path: Path) -> None:
    fc = FakeClock()
    cli = FakeNebius()
    # a teacher in the store, for the cost comparison
    st = Store(tmp_path / "store")
    for i in range(1, 11):
        st.record_llm_call("r", "teacher-m", 100, 10, 0.002, i / 10)
    st.close()
    golds = [str(i["gold_sql"]) for i in items(4)]
    rc, lines, lg, ev = live(
        tmp_path, cli, fc,
        ["--teacher-model", "teacher-m", "--coverage", "0.5", "--audit-rate", "0.1",
         "--volumes", "1000,100000"],
        chat=GoldChat(golds, clock=fc, step=1.0),
    )  # fmt: skip
    assert rc == 0 and ev["proof"] == "H1" and ev["error"] is None
    assert ev["plan"]["auth_token"] == "***" and ev["plan"]["hourly_usd"] == 2.0
    life = ev["lifecycle"]
    assert life["endpoint_id"] == "ep-1" and life["deleted_confirmed"] is True
    assert life["created_at"] and life["ready_at"] and life["deleted_at"]
    assert cli.endpoints == {} and cli.verbs()[-2:] == ["delete", "list"]
    assert [lv["concurrency"] for lv in ev["measurements"]["levels"]] == [1, 2]
    assert ev["endpoint_accuracy"]["accuracy"] == 1.0
    assert ev["parity"]["sandbox_accuracy"] == 0.75 and ev["parity"]["agreement"] == 0.75
    assert ev["parity"]["identical_sql_rate"] == 0.75
    assert ev["ledger"]["delta_usd"] == pytest.approx(lg.spent()) and lg.spent() > 0
    assert ev["ledger"]["delta_usd"] == pytest.approx(life["spend_usd"])
    c = ev["costs"]
    assert c["projected"] is True and c["student"]["projected"] is True
    assert c["teacher"]["usd_per_1k"] == pytest.approx(2.0)
    assert c["blended_busy"]["projected"] is True and len(c["blended_always_on"]["points"]) == 2
    assert c["blended_always_on"]["student_fixed_daily_usd"] == pytest.approx(48.0)
    samples = (tmp_path / "out" / "h1_endpoint_samples.jsonl").read_text().splitlines()
    assert len(samples) == 8 and json.loads(samples[0])["text"]
    assert TOKEN not in json.dumps(ev) and TOKEN not in "\n".join(lines)
    raw = (tmp_path / "out" / "h1_endpoint.json").read_text()
    assert '"auth_token": "***"' in raw


def test_live_delete_failure_exits_2_and_records_the_id(tmp_path: Path) -> None:
    fc = FakeClock()
    cli = FakeNebius(delete_rc=1)
    rc, lines, lg, ev = live(tmp_path, cli, fc)
    assert rc == 2 and any("DELETE NOT CONFIRMED" in ln and "ep-1" in ln for ln in lines)
    assert ev["lifecycle"]["deleted_confirmed"] is False and "may still be billing" in ev["error"]
    assert lg.spent() > 0 and ev["parity"] is None  # no analysis on a doubtful state


def test_live_start_failure_deletes_and_records_spend(tmp_path: Path) -> None:
    cli = FakeNebius(states=("PENDING", "ERROR"))
    rc, lines, lg, ev = live(tmp_path, cli, FakeClock())
    assert rc == 1 and "EndpointStartError" in ev["error"]
    assert ev["lifecycle"]["deleted_confirmed"] is True and ev["lifecycle"]["endpoint_id"] == "ep-1"
    assert cli.endpoints == {} and lg.spent() > 0 and ev["measurements"] is None


def test_live_time_box_expiry_stops_measuring_and_deletes(tmp_path: Path) -> None:
    fc = FakeClock()
    cli = FakeNebius()
    golds = [str(i["gold_sql"]) for i in items(4)]
    chat = GoldChat(golds, clock=fc, step=400.0)  # each request "takes" 400 s of the clock
    rc, _, lg, ev = live(tmp_path, cli, fc, ["--time-box-minutes", "10"], chat=chat)
    assert rc == 0
    assert ev["measurements"]["stopped_early"] is True
    assert ev["lifecycle"]["timed_out"] is True and ev["lifecycle"]["deleted_confirmed"] is True
    assert cli.endpoints == {} and chat.calls == 1
    assert ev["parity"] is None and "missing" in ev["endpoint_accuracy"]["note"]
    assert lg.spent() == pytest.approx(ev["lifecycle"]["spend_usd"])


def test_live_preflight_refusal_creates_nothing(tmp_path: Path) -> None:
    cli = FakeNebius()
    store = Store(tmp_path / "store")
    lg = Ledger("h1", 1.0, 19.5, 1.0, store=store)  # run cap below the $1.50 estimate
    argv = [*PLAN_ARGS, *write_inputs(tmp_path), "--i-approve-spend"]
    rc, _ = run_main(argv, runner=cli, ledger=lg, clock=FakeClock().clock(),
                     client_factory=lambda h: FakeChat())  # fmt: skip
    ev = json.loads((tmp_path / "out" / "h1_endpoint.json").read_text())
    assert rc == 1 and "BudgetExceeded" in ev["error"] and cli.calls == []
    assert lg.spent() == 0.0


def test_default_client_factory_builds_an_http_client() -> None:
    h = EndpointHandle(
        endpoint_id="e", name="n", base_url="https://h/v1", token="t", deadline=1.0,
        clock=FakeClock().clock(),
    )  # fmt: skip
    seen: dict[str, Any] = {}

    def fake_run_live(*a: Any, **kw: Any) -> int:
        seen["client"] = kw["client_factory"](h)
        return 0

    orig = h1.run_live
    h1.run_live = fake_run_live
    try:
        rc, _ = run_main([*PLAN_ARGS, "--i-approve-spend", "--prompts", "x"],
                         ledger=Ledger("h1", 10, 19.5, 1.0))  # fmt: skip
    finally:
        h1.run_live = orig
    assert rc == 0 and "https://h/v1" in repr(seen["client"])


# ---------------------------------------------------------------- refusals before any spend


def refused(tmp: Path, extra: list[str], mutate=None) -> tuple[int, list[str], FakeNebius]:  # type: ignore[no-untyped-def]
    cli = FakeNebius()
    argv = [*PLAN_ARGS, *write_inputs(tmp), "--i-approve-spend", *extra]
    if mutate:
        mutate(tmp)
    lg = Ledger("h1", 10.0, 19.5, 1.0, store=Store(tmp / "store"))
    rc, lines = run_main(
        argv, runner=cli, ledger=lg, clock=FakeClock().clock(), client_factory=lambda h: FakeChat()
    )
    assert lg.spent() == 0.0
    return rc, lines, cli


def test_sandbox_outputs_of_the_wrong_length_are_refused_before_creating(tmp_path: Path) -> None:
    def shorten(tmp: Path) -> None:
        (tmp / "sandbox.json").write_text(json.dumps(["SELECT 1"]))

    rc, lines, cli = refused(tmp_path, [], shorten)
    assert rc == 3 and "1 outputs for 4 items" in lines[0] and cli.calls == []
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize(
    ("extra", "needle"),
    [
        (["--coverage", "1.5", "--teacher-model", "t"], "--coverage must be in [0, 1]"),
        (["--audit-rate", "2"], "--audit-rate must be in [0, 1]"),
        (["--coverage", "0.5"], "needs --teacher-model"),
        (["--db", "/nonexistent/db.sqlite"], "is not a file"),
        (["--concurrency", "0"], "must be >= 1"),
        (["--volumes", "0"], "must be positive"),
        (
            ["--teacher-model", "absent-model", "--teacher-root", "STORE"],
            "teacher cost unavailable",
        ),
    ],
)
def test_bad_inputs_are_refused_before_any_cli_call(
    tmp_path: Path, extra: list[str], needle: str
) -> None:
    extra = [str(tmp_path / "store") if x == "STORE" else x for x in extra]
    Store(tmp_path / "store").close()
    rc, lines, cli = refused(tmp_path, extra)
    assert rc == 3 and needle in lines[0] and cli.calls == []


def test_items_without_gold_are_refused_when_a_db_is_given(tmp_path: Path) -> None:
    def drop_gold(tmp: Path) -> None:
        row = {"question": "q0", "schema_ddl": "CREATE TABLE t (id INTEGER);"}
        (tmp / "prompts.jsonl").write_text(json.dumps(row) + "\n")
        (tmp / "sandbox.json").write_text(json.dumps(["SELECT 1"]))

    rc, lines, cli = refused(tmp_path, [], drop_gold)
    assert rc == 3 and "lack gold_sql" in lines[0] and cli.calls == []


def test_schema_ddl_fallback_for_items_without_their_own(tmp_path: Path) -> None:
    its = [{k: v for k, v in i.items() if k != "schema_ddl"} for i in items(4)]
    (tmp_path / "prompts.jsonl").write_text("\n".join(json.dumps(i) for i in its) + "\n")
    (tmp_path / "ddl.sql").write_text("CREATE TABLE t (id INTEGER, v TEXT);")
    prompts = str(tmp_path / "prompts.jsonl")
    a = h1.build_parser().parse_args(
        [*PLAN_ARGS, "--prompts", prompts, "--schema-ddl", str(tmp_path / "ddl.sql")]
    )
    prep = h1.prepare(a)
    assert "CREATE TABLE t (id INTEGER, v TEXT);" in prep.prompts[0][1]["content"]
    b = h1.build_parser().parse_args([*PLAN_ARGS, "--prompts", prompts])
    with pytest.raises(h1.Refused, match="schema_ddl"):
        h1.prepare(b)


def test_analysis_failure_after_the_spend_still_writes_the_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def boom(*a: Any, **kw: Any) -> Any:
        raise ValueError("analysis exploded")

    monkeypatch.setattr(h1, "accuracy_and_parity", boom)
    fc = FakeClock()
    golds = [str(i["gold_sql"]) for i in items(4)]
    rc, lines, lg, ev = live(tmp_path, FakeNebius(), fc, chat=GoldChat(golds, clock=fc, step=1.0))
    assert rc == 0 and ev["measurements"]["levels"] and ev["lifecycle"]["deleted_confirmed"] is True
    assert ev["analysis_errors"] == ["accuracy/parity: ValueError: analysis exploded"]
    assert (tmp_path / "out" / "h1_endpoint_samples.jsonl").read_text().strip()
    assert lg.spent() > 0


def test_plan_only_shows_the_delete_reserve() -> None:
    rc, lines = run_main([*PLAN_ARGS, "--plan-only", "--delete-reserve-minutes", "7"])
    text = "\n".join(lines)
    assert "measuring stops    at 38 min (last 7 min reserved for delete + confirm)" in text
