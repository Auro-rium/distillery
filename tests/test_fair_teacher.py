# ruff: noqa: S101
"""Workstream B (B2/B3): per-item vectors, cached re-score, claim rule, P4 reproduction check,
read-only source, the dry run end to end, and the pre-flight estimate. All offline, all fake."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import pytest

from distillery import fairteacher as ft
from distillery.config import GateThresholds
from distillery.evaluator import Setup, score_open_set, score_setups
from distillery.prompts import build_messages
from distillery.store import Store
from distillery.taskpacks.sql import questions as q
from distillery.taskpacks.sql import schema as s
from distillery.taskpacks.sql.executor import LocalExecutor
from tests.test_evaluator import Lookup
from tests.test_score_human import confirmed_file

CFG = GateThresholds()


@pytest.fixture
def env(tmp_path: Path):  # type: ignore[no-untyped-def]
    db = tmp_path / "db.sqlite"
    s.write_database(0, str(db))
    tasks = q.generate_tasks(str(db), 30, 0).tasks[:30]
    items = [
        {"task_id": t.task_id, "family": t.family, "question": t.question,
         "gold_sql": t.gold_sql, "requires_order": t.requires_order, "heldout_class": "seen"}
        for t in tasks
    ]  # fmt: skip
    store = Store(tmp_path / "root")
    store.seal_heldout("r1", items)
    answers = {it["question"]: it["gold_sql"] for it in items}
    return store, str(db), s.schema_ddl(0), items, answers


def _setups(answers: dict[str, str], ddl: str) -> list[Setup]:
    def zero(it: Any) -> Any:
        return build_messages(str(it["question"]), ddl, role="eval_teacher")

    return [
        Setup("base", Lookup(answers, wrong_every=2), zero),
        Setup("student", Lookup(answers, wrong_every=5), zero),
        Setup("S0", Lookup(answers, wrong_every=10), zero),
    ]


def test_score_setups_saves_vectors_and_marks_human_pending(env) -> None:  # type: ignore[no-untyped-def]
    store, db, ddl, items, answers = env
    p = score_setups(store, "r1", _setups(answers, ddl), LocalExecutor(), db_ref=db, schema_ddl=ddl)
    json.dumps(p)  # JSON-able as is
    h = p["sets"]["heldout"]
    assert h["status"] == "scored" and h["n"] == len(items)
    assert h["task_ids"] == [it["task_id"] for it in items]
    for name in ("base", "student", "S0"):
        b = h["setups"][name]
        assert len(b["correct"]) == len(b["reasons"]) == len(b["outputs"]) == len(items)
        assert b["accuracy"] == sum(b["correct"]) / len(items)
    assert (
        p["sets"]["human"]["status"] == "pending" and "no human set" in p["sets"]["human"]["reason"]
    )
    assert all(set(m) >= {"task_id", "category", "rule", "diff"} for m in h["misses"])
    assert "question" not in json.dumps(h["misses"]) and "gold_sql" not in json.dumps(h["misses"])
    assert len(h["misses"]) == sum(len(items) - sum(b["correct"]) for b in h["setups"].values())


def test_rescore_from_cache_calls_no_model_and_is_identical(env) -> None:  # type: ignore[no-untyped-def]
    store, db, ddl, _items, answers = env
    setups = _setups(answers, ddl)
    before = score_setups(store, "r1", setups, LocalExecutor(), db_ref=db, schema_ddl=ddl)
    calls = [st.generator.calls for st in setups]  # type: ignore[union-attr]
    nogen = [Setup(st.name, None, st.prompt) for st in setups]
    after = score_setups(
        store, "r1", nogen, LocalExecutor(), db_ref=db, schema_ddl=ddl, cached=before
    )
    assert [st.generator.calls for st in setups] == calls  # type: ignore[union-attr]
    assert after == before
    ba = ft.before_after(before, after, before["setups"])
    assert all(v["before"] == v["after"] for v in ba["heldout"].values())
    stale = json.loads(json.dumps(before))
    stale["sets"]["heldout"]["task_ids"].reverse()
    with pytest.raises(ValueError, match="task ids"):
        score_setups(store, "r1", nogen, LocalExecutor(), db_ref=db, schema_ddl=ddl, cached=stale)


def test_open_set_scoring_and_strongest_pick(env) -> None:  # type: ignore[no-untyped-def]
    _store, db, ddl, items, answers = env
    block = score_open_set(items, _setups(answers, ddl), LocalExecutor(), db_ref=db, schema_ddl=ddl)
    assert block["n"] == len(items) and set(block["setups"]) == {"base", "student", "S0"}
    dev = {"setups": {"S0": {"accuracy": 0.8}, "S1": {"accuracy": 0.9}, "S2": {"accuracy": 0.9}}}
    pick = ft.pick_strongest(dev)
    assert pick["setup"] == "S2" and pick["selected_on"] == "dev"  # tie goes to the later setup
    dev["setups"]["S1"]["accuracy"] = 0.95
    assert ft.pick_strongest(dev)["setup"] == "S1"


def test_claim_rule_is_strict_and_uses_gate_bootstrap() -> None:
    teacher = [True] * 90 + [False] * 10
    equal = ft.claim_rule(teacher, teacher, CFG)
    assert (
        equal["claim"] is False and equal["ratio_point"] == 1.0 and equal["ratio_lo"] == 1.0
    )  # a bound of exactly 1.0 is not a win
    strong = ft.claim_rule([True] * 100, teacher, CFG)  # same 90 plus 10 more: lower bound > 1
    assert strong["claim"] is True and strong["ratio_lo"] > 1.0
    assert ft.claim_rule([True] * 5, [False] * 5, CFG)["claim"] is False  # teacher 0: undefined
    # bound exactly 1.0 does not count: threshold is strict
    assert ft.CLAIM_RATIO_MIN == 1.0


def _vectors(n: int, k: int, seed: int) -> list[bool]:
    idx = set(random.Random(seed).sample(range(n), k))
    return [i in idx for i in range(n)]


def test_gate_recomputed_from_vectors_reproduces_p4_counts() -> None:
    """The code path of the run-time validation: 277/300 student, 270/300 S0, 77/300 base."""
    payload = {
        "sets": {
            "heldout": {
                "status": "scored",
                "n": 300,
                "setups": {
                    "student": {"correct": _vectors(300, 277, 1)},
                    "S0": {"correct": _vectors(300, 270, 2)},
                    "base": {"correct": _vectors(300, 77, 3)},
                },
            }
        }
    }
    r = ft.check_p4_reproduction(payload, CFG)
    assert r["reproduced"] is True and r["differences"] == {"student": 0, "teacher": 0, "base": 0}
    assert r["accuracy"]["student"] == pytest.approx(0.9233, abs=1e-4)
    assert r["accuracy"]["teacher"] == 0.9 and r["accuracy"]["base"] == pytest.approx(
        0.2567, abs=1e-4
    )
    assert r["gate"]["n"] == 300 and r["gate_decision_recomputed"] in {"PROMOTE", "REJECT"}
    payload["sets"]["heldout"]["setups"]["S0"]["correct"] = _vectors(300, 266, 2)
    off = ft.check_p4_reproduction(payload, CFG)
    assert off["reproduced"] is False and off["differences"]["teacher"] == -4


def test_dry_run_end_to_end_without_human_set(tmp_path: Path) -> None:
    ev = ft.dry_run(tmp_path, say=lambda _m: None)
    assert ev["dry_run"] is True and "NOT results" in ev["label"]
    assert ev["source_tree_unchanged"] is True and len(ev["fewshot"]["task_ids"]) == 8
    work = tmp_path / "work"
    for name in ("vectors.json", "dev_vectors.json", "b3_misses.md", "b_fair_teacher.json",
                 "fewshot_examples.json", "vectors_rescored.json", "rescore_summary.json"):  # fmt: skip
        assert (work / name).is_file(), name
    vec = json.loads((work / "vectors.json").read_text())
    assert set(vec["setups"]) == {"base", "student", "S0", "S1", "S2"}
    assert vec["sets"]["human"]["status"] == "pending"
    assert ev["analysis"]["claims"]["human"]["status"] == "pending"
    assert (
        ev["analysis"]["claims"]["heldout"]["vs_strongest_teacher"]["teacher"]
        == (ev["analysis"]["strongest_teacher"]["setup"])
    )
    # few-shot examples come from train only
    ex = json.loads((work / "fewshot_examples.json").read_text())
    assert ex["from"] == "train split only" and len(ex["examples"]) == 8
    # S1/S2 prompts really are few-shot (about 9x the input tokens of S0)
    use = ev["usage_by_purpose"]
    assert use["fair_s1"]["input_tokens"] > 5 * use["fair_s0"]["input_tokens"]
    # the re-score used no model and changed nothing
    ba = ev["rescore"]["before_after_correct"]["heldout"]
    assert ev["rescore"]["model_calls"] == 0 and all(v["before"] == v["after"] for v in ba.values())


def test_a_deferred_setup_is_absent_everywhere_and_never_picked(tmp_path: Path) -> None:
    """DECISIONS.md 2026-10-06: S2 deferred on budget. It is not scored, not picked on dev, not a
    claim opponent, and the evidence names it as deferred; a later run with all setups in the
    same --out reuses the cached S0/S1/student outputs."""
    ev = ft.dry_run(tmp_path, say=lambda _m: None, teacher_names=("S0", "S1"))
    work = tmp_path / "work"
    vec = json.loads((work / "vectors.json").read_text())
    dev = json.loads((work / "dev_vectors.json").read_text())
    assert set(vec["setups"]) == {"base", "student", "S0", "S1"}
    assert set(dev["setups"]) == {"S0", "S1"}
    st = ev["analysis"]["strongest_teacher"]
    assert st["setup"] in {"S0", "S1"} and st["deferred"] == ["S2"]
    assert set(ev["analysis"]["claims"]["heldout"]["vs_each_teacher_for_information"]) == {
        "S0",
        "S1",
    }
    assert ev["teacher_setups_scored"] == ["S0", "S1"] and ev["teacher_setups_deferred"] == ["S2"]
    assert ev["usage_by_purpose"]["fair_s2"] == {}


def test_dry_run_scores_a_sealed_human_set_when_present(tmp_path: Path) -> None:
    f = confirmed_file(tmp_path / "human.json")
    ev = ft.dry_run(tmp_path / "o", human_set=f, say=lambda _m: None)
    human = ev["analysis"]["claims"]["human"]
    assert human["status"] == "scored" and human["n"] == 6
    vec = json.loads((tmp_path / "o" / "work" / "vectors.json").read_text())
    assert vec["sets"]["human"]["status"] == "scored"
    assert len(vec["sets"]["human"]["setups"]["student"]["correct"]) == 6


def test_workspace_refuses_overlap_and_never_writes_source(tmp_path: Path) -> None:
    ft.dry_run(tmp_path / "o", say=lambda _m: None)
    src = tmp_path / "o" / "dry_source"
    with pytest.raises(ft.FairTeacherRefusal):
        ft.prepare_workspace(src, src / "inside", "dry-fair-teacher")
    with pytest.raises(ft.FairTeacherRefusal):
        ft.prepare_workspace(src, tmp_path, "dry-fair-teacher")  # out contains root
    assert ft.find_run_id(src) == "dry-fair-teacher"
    before = ft.tree_fingerprint(src)
    ft.prepare_workspace(src, tmp_path / "elsewhere", "dry-fair-teacher")
    assert ft.tree_fingerprint(src) == before


def test_estimate_table_uses_real_prompt_tokens_and_prices(tmp_path: Path) -> None:
    from distillery.config import Price

    ft.dry_run(tmp_path, say=lambda _m: None)
    src = tmp_path / "dry_source"
    from distillery.config import Config
    from distillery.pipeline_fakes import FAKE_MODELS, FAKE_PRICE

    cfg = Config(model_ids=dict(FAKE_MODELS), prices={m: FAKE_PRICE for m in FAKE_MODELS.values()})
    est = ft.estimate_table(src, "dry-fair-teacher", cfg, human_n=10)
    assert [r["calls"] for r in est["rows"]][:3] == [est["rows"][0]["calls"]] * 3
    zero, few = est["rows"][0], est["rows"][1]
    assert few["input_tokens"] > 5 * zero["input_tokens"]
    price: Price = FAKE_PRICE
    expect = (
        zero["input_tokens"] * price.input_per_mtok + zero["output_tokens"] * price.output_per_mtok
    ) / 1e6
    assert zero["usd_budget_basis"] == pytest.approx(expect)
    assert "TOTAL" in ft.format_estimate(est)
    unpriced = ft.estimate_table(src, "dry-fair-teacher", Config(model_ids=dict(FAKE_MODELS)))
    assert unpriced["total_usd_budget_basis"] is None  # never priced at a guess


def test_cached_generator_reuses_by_prompt_digest(tmp_path: Path) -> None:
    class Counting:
        n = 0

        def generate(self, batch: Any) -> list[str]:
            Counting.n += 1
            return [f"out{len(b)}" for b in batch]

    g = ft.CachedGenerator(Counting(), tmp_path, "m")
    b1 = [[{"role": "user", "content": "a"}]]
    assert g.generate(b1) == ["out1"] and g.generate(b1) == ["out1"] and Counting.n == 1
    assert g.generate([[{"role": "user", "content": "b"}]]) == ["out1"] and Counting.n == 2
