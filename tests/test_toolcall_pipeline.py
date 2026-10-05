# ruff: noqa: S101
"""C3: the tool-call pack runs the whole pipeline in a DRY run under the same integrity rules."""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient
from server_helpers import make_settings, scripted_executor
from test_orchestrator import NANO, make

from distillery.orchestrator import DRY_RUN_LABEL, VerifierSelfTestError
from distillery.pipeline_fakes import question_of
from distillery.sandbox_executor import AsyncBridge
from distillery.server import create_app
from distillery.taskpacks.base import get_pack
from distillery.taskpacks.toolcall import env as E
from distillery.taskpacks.toolcall import questions as Q
from distillery.taskpacks.toolcall.executor import LocalExecutor


@pytest.fixture(scope="module")
def tc_run(tmp_path_factory: pytest.TempPathFactory) -> Iterator[dict[str, Any]]:
    tmp = tmp_path_factory.mktemp("tc")
    with AsyncBridge() as bridge:
        pipe, dr, store = make(tmp, bridge, run_id="dry-tc", pack="toolcall")
        report = pipe.run()
        yield {"pipe": pipe, "dr": dr, "store": store, "report": report}
        store.close()


def test_dry_run_reaches_a_report_with_integrity_fields(tc_run: dict[str, Any]) -> None:
    rep, store = tc_run["report"], tc_run["store"]
    assert rep["pack"] == "toolcall" and rep["label"] == DRY_RUN_LABEL and rep["dry_run"] is True
    assert rep["decision"] in {"PROMOTE", "REJECT"} and rep["decision_reasons"]
    d = rep["data"]
    assert d["heldout_sealed_sha256"] and d["stress_sealed_sha256"]
    reserved = ["dependent_chain", "single_call"]  # seed 777, drawn once (DECISIONS.md)
    assert list(get_pack("toolcall").stress_families()) == reserved
    assert d["stress_families"] and set(d["stress_families"]) <= set(reserved)
    assert rep["headroom"]["base_dev_acc"] <= rep["headroom"]["max_allowed"]
    assert d["teacher_verified_rows_round1"] > 0
    ev = rep["evaluation"]
    assert set(ev["accuracy"]) == {"base", "student", "teacher"}
    held = store.load_heldout("dry-tc")
    assert len(held) == NANO.heldout
    assert all(it["family"] not in reserved for it in held)
    assert "gold_answer" in held[0] and held[0]["gold_answer"].startswith("[")


def test_teacher_rows_are_execution_verified(tc_run: dict[str, Any]) -> None:
    """Every kept training row's assistant answer replays to its task's gold final state."""
    pipe = tc_run["pipe"]
    pk = get_pack("toolcall")
    ex = LocalExecutor()
    n = 0
    for rnd in sorted(pipe.run_dir.glob("round*")):
        for f in rnd.glob("*.jsonl"):
            for line in f.read_text().splitlines():
                row = json.loads(line)
                if "messages" not in row:
                    continue
                msgs = row["messages"]
                goal = question_of(msgs[:-1])
                ans = pk.extract_answer(msgs[-1]["content"])
                assert goal is not None and ans is not None
                gold = tc_run["dr"].oracle.gold(goal)
                o_a, o_g = ex.run_batch(pipe.db_ref, [ans, gold])
                assert pk.compare(o_a, o_g, False).ok, goal
                n += 1
    assert n > 0


def test_selftest_refuses_when_a_corruption_is_accepted(tmp_path: Path) -> None:
    class Liar(LocalExecutor):
        lie = False

        def run_batch(self, env_ref: str, answers: Any) -> Any:
            out = super().run_batch(env_ref, answers)
            return [out[0]] * len(out) if self.lie and out else out

    liar = Liar()
    with AsyncBridge() as bridge:
        pipe, dr, _s = make(
            tmp_path, bridge, pack="toolcall",
            on_stage=lambda s: setattr(liar, "lie", s == "verifier_selftest"),
        )  # fmt: skip
        dr.deps.executor = liar
        with pytest.raises(VerifierSelfTestError, match="ACCEPTED"):
            pipe.run()


def test_selftest_refuses_a_verifier_that_accepts_everything(tmp_path: Path) -> None:
    pk = get_pack("toolcall")
    env = pk.build_env(0, tmp_path / "e.json")
    tasks = pk.generate_tasks(env.ref, 20, 1).tasks
    assert pk.selftest(tasks, env.ref, seed=1)["corruptions_tested"] > 0

    class Wrong(LocalExecutor):  # a "verifier" fed a corrupted gold outcome: all rows collide
        def run_batch(self, env_ref: str, answers: Any) -> Any:
            out = super().run_batch(env_ref, answers)
            return [out[0]] * len(out)

    pk.executor = Wrong()
    try:
        with pytest.raises(VerifierSelfTestError, match="ACCEPTED"):
            pk.selftest(tasks, env.ref, seed=1)
    finally:
        pk.executor = LocalExecutor()


def test_fake_question_of_and_wrong_answer_for_toolcall(tmp_path: Path) -> None:
    pk = get_pack("toolcall")
    env = pk.build_env(0, tmp_path / "e.json")
    t = pk.generate_tasks(env.ref, 5, 1).tasks[0]
    for msgs in (
        pk.build_messages(t.question, env.context_text, role="train"),
        pk.paraphrase_messages(t.question, 3),
    ):
        assert question_of(msgs) == t.question
    ex = LocalExecutor()
    wrong, gold = ex.run_batch(env.ref, [pk.dry_wrong_answer, t.gold_answer])
    assert not wrong.ok and not pk.compare(wrong, gold, False).ok
    assert pk.stress_families() == Q.stress_families() == ("dependent_chain", "single_call")
    assert E.parse_env_ref(env.ref) == 0


def test_paraphrase_instruction_is_pack_provided() -> None:
    sql = get_pack("sql").paraphrase_messages("How many users?", 2)
    assert "do not mention SQL or tables" in sql[0]["content"]
    tc = get_pack("toolcall").paraphrase_messages("Close ticket 3", 2)
    assert "SQL" not in tc[0]["content"] and tc[1]["content"].endswith("Goal: Close ticket 3")


def test_api_validates_and_starts_a_toolcall_run(tmp_path: Path) -> None:
    s = make_settings(tmp_path)
    s.executor = s.executor or scripted_executor(s.root)
    with TestClient(create_app(s)) as c:
        packs = {p["name"]: p for p in c.get("/api/config").json()["packs"]}
        assert packs["toolcall"] == {
            "name": "toolcall", "language": "json", "answer_label": "tool calls"
        }  # fmt: skip
        r = c.post("/api/runs", json={"pack": "toolcall", "scale": "tiny", "dry_run": True})
        assert r.status_code == 202 and r.json()["run_id"].startswith("dry-toolcall-tiny-")
        bad = c.post("/api/runs", json={"pack": "nope", "scale": "tiny", "dry_run": True})
        assert bad.status_code == 422
