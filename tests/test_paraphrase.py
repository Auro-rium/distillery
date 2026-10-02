# ruff: noqa: S101
from __future__ import annotations

import ast
import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from distillery import paraphrase as para
from distillery.orchestrator import Pipeline, Scale
from distillery.pipeline_fakes import DryRun, build_dry_run
from distillery.sandbox_executor import AsyncBridge
from distillery.store import Store
from distillery.taskpacks.sql.questions import SqlTask
from distillery.taskpacks.sql.verifier import compare_outcomes

NANO = Scale(name="nano", train=16, dev=8, heldout=12, stress=6)
SRC = Path(__file__).parents[1] / "src/distillery/paraphrase.py"


def _task(i: int = 1) -> SqlTask:
    return SqlTask(
        task_id=f"t{i}",
        family="agg",
        question=f"How many orders were placed by customer {i}?",
        gold_sql="SELECT 1",
        requires_order=True,
        difficulty="easy",
        tables=("orders",),
        template="x",
    )


# ---- pure helpers ---------------------------------------------------------------------------


def test_normalise_ignores_case_punctuation_and_spacing() -> None:
    assert para.normalise("  How MANY   orders?! ") == para.normalise("how many orders")


def test_clean_candidates_dedupes_against_original_and_each_other_and_caps() -> None:
    q = "How many orders?"
    items = ["how many orders", "Count the orders.", "count the  orders", "", "  ", "Orders total?",
             "Number of orders?", "Extra one?"]  # fmt: skip
    kept, discards = para.clean_candidates(q, items, limit=3)
    assert kept == ["Count the orders.", "Orders total?", "Number of orders?"]
    assert discards == {
        "empty": 2, "duplicate_of_original": 1, "duplicate_of_paraphrase": 1, "over_limit": 1,
    }  # fmt: skip


def test_pseudo_task_keeps_template_gold_and_ordering_with_new_wording() -> None:
    t = _task()
    p = para.pseudo_task(t, 2, "Orders by customer 1, counted?")
    assert p.task_id == "t1:p2" and p.question == "Orders by customer 1, counted?"
    assert (p.gold_sql, p.requires_order, p.family) == (t.gold_sql, t.requires_order, t.family)


def test_messages_contain_only_the_question() -> None:
    msgs = para.paraphrase_messages("Q-TEXT-123", 3)
    text = "\n".join(m["content"] for m in msgs)
    assert "Q-TEXT-123" in text and "literal" in text.lower()


def test_select_rows_is_deterministic_keeps_originals_first_and_caps() -> None:
    orig = [(f"o{i}", {"i": i}) for i in range(5)]
    extra = [(f"o{i}:p{k}", {"i": i, "k": k}) for i in range(5) for k in (1, 2)]
    a = para.select_rows(orig, extra, cap=8, seed=7)
    b = para.select_rows(orig, extra, cap=8, seed=7)
    c = para.select_rows(orig, extra, cap=8, seed=8)
    assert a == b and len(a) == 8 and a != c
    assert [i for i, _ in a[:5]] == [f"o{i}" for i in range(5)]
    assert len(para.select_rows(orig, extra, cap=100, seed=7)) == 15
    only = para.select_rows(orig, extra, cap=3, seed=7)
    assert len(only) == 3 and all(i.startswith("o") and ":" not in i for i, _ in only)


# ---- isolation (AST) ------------------------------------------------------------------------


def test_paraphrase_module_imports_nothing_sealed_and_names_no_loader() -> None:
    tree = ast.parse(SRC.read_text())
    mods: list[str] = []
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            mods += [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom):
            mods.append(node.module or "")
            names |= {a.name for a in node.names}
        elif isinstance(node, ast.Name):
            names.add(node.id)
        elif isinstance(node, ast.Attribute):
            names.add(node.attr)
    assert not [m for m in mods if "evaluator" in m or m.endswith("store") or "orchestrator" in m]
    assert not {"load_heldout", "load_stress", "load_human"} & names
    assert "evaluator" not in names and "Store" not in names


# ---- pipeline integration (offline fakes) ----------------------------------------------------


@pytest.fixture(scope="module")
def run(tmp_path_factory: pytest.TempPathFactory) -> Iterator[dict[str, Any]]:
    tmp = tmp_path_factory.mktemp("para")
    with AsyncBridge() as bridge:
        dr: DryRun = build_dry_run(NANO, bridge)
        store = Store(tmp / "store")
        pipe = Pipeline(dr.pipeline_cfg, dr.config, dr.deps, store, "dry-p", say=lambda _s: None)
        report = pipe.run()
        yield {"pipe": pipe, "dr": dr, "store": store, "report": report}
        store.close()


def test_paraphrase_stage_runs_after_teacher_data_and_feeds_rounds(run: dict[str, Any]) -> None:
    pipe, rep = run["pipe"], run["report"]
    assert pipe.ran.index("paraphrase") == pipe.ran.index("teacher_data") + 1
    res = pipe._results["paraphrase"]
    assert (
        rep["rounds"][0]["train_rows"] == len(res["rows"]) == rep["paraphrase"]["final_train_rows"]
    )


def test_report_paraphrase_section(run: dict[str, Any]) -> None:
    p = run["report"]["paraphrase"]
    for key in ("questions_paraphrased", "generated", "verified", "yield", "yield_by_family",
                "discards", "rows_added", "final_train_rows", "originals", "row_cap",
                "per_task", "seed"):  # fmt: skip
        assert key in p, key
    assert p["questions_paraphrased"] > 0 and p["generated"] > 0 and p["verified"] > 0
    assert p["yield"] == pytest.approx(p["verified"] / p["generated"])
    assert p["final_train_rows"] == p["originals"] + p["rows_added"] <= p["row_cap"]
    assert p["discards"]["duplicate_of_original"] >= 0
    fam = p["yield_by_family"]
    assert sum(f["verified"] for f in fam.values()) == p["verified"]
    assert sum(f["generated"] for f in fam.values()) == p["generated"]
    counters = run["report"]["counters"]["paraphrase"]
    assert all(isinstance(v, int) and v >= 0 for v in counters.values())


def test_paraphrase_rows_are_real_rows_with_template_gold_verified(run: dict[str, Any]) -> None:
    pipe = run["pipe"]
    res = pipe._results["paraphrase"]
    ids = res["task_ids"]
    assert len(ids) == len(res["rows"]) and len(set(ids)) == len(ids)
    train = {d["task_id"]: d for d in pipe._results["split"]["train"]}
    added = [i for i in ids if ":p" in i]
    assert added
    for i in added:
        base = i.split(":p")[0]
        assert base in train
        row = res["rows"][ids.index(i)]
        assert row["messages"][-1]["role"] == "assistant"
        wording = row["messages"][-2]["content"].split("Question: ")[-1].split("\n\n/no_think")[0]
        assert wording.strip() != train[base]["question"]
        sql_out, gold_out = pipe.deps.executor.run_batch(
            pipe.db_ref, [row["messages"][-1]["content"], train[base]["gold_sql"]]
        )
        assert compare_outcomes(sql_out, gold_out, train[base]["requires_order"]).ok


def test_paraphrase_ids_never_carry_sealed_ids(run: dict[str, Any]) -> None:
    pipe = run["pipe"]
    split = pipe._results["split"]
    sealed = set(split["heldout_task_ids"]) | set(split["stress_task_ids"])
    train_ids = {d["task_id"] for d in split["train"]}
    for i in pipe._results["paraphrase"]["task_ids"]:
        assert i.split(":p")[0] in train_ids
        assert i.split(":p")[0] not in sealed and i not in sealed


def test_paraphraser_requests_hold_only_train_questions(run: dict[str, Any]) -> None:
    dr, store, pipe = run["dr"], run["store"], run["pipe"]
    sealed = [*store.load_heldout("dry-p"), *(store.load_stress("dry-p") or [])]
    assert sealed
    calls = [c for c in dr.transport.calls if c["schema"] == "Paraphrases"]
    assert calls and all(c["model"] == "fake-triage" for c in calls)
    train_q = {d["question"] for d in pipe._results["split"]["train"]}
    asked: set[str] = set()
    for c in calls:
        text = "\n".join(m["content"] for m in c["messages"])
        for it in sealed:
            assert it["question"] not in text and it["gold_sql"] not in text
        hits = [q for q in train_q if q in text]
        assert hits, "each request carries a train question"
        asked.update(hits)
    dev_q = {d["question"] for d in pipe._results["split"]["dev"]}
    assert not any(
        q in "\n".join(m["content"] for c in calls for m in c["messages"]) for q in dev_q
    )
    assert asked


def test_row_cap_and_paraphrase_count_are_honoured(tmp_path: Path) -> None:
    with AsyncBridge() as bridge:
        dr = build_dry_run(NANO, bridge, train_row_cap=20, paraphrases_per_task=2)
        store = Store(tmp_path / "s")
        pipe = Pipeline(dr.pipeline_cfg, dr.config, dr.deps, store, "dry-c", say=lambda _s: None)
        rep = pipe.run()
        p = rep["paraphrase"]
        assert p["final_train_rows"] <= 20 and p["per_task"] == 2
        assert p["after_dedup"] <= 2 * p["questions_paraphrased"]  # the fake over-delivers
        assert p["discards"]["over_limit"] > 0
        store.close()


def test_zero_paraphrases_disables_calls_and_keeps_originals(tmp_path: Path) -> None:
    with AsyncBridge() as bridge:
        dr = build_dry_run(NANO, bridge, paraphrases_per_task=0)
        store = Store(tmp_path / "s")
        pipe = Pipeline(dr.pipeline_cfg, dr.config, dr.deps, store, "dry-z", say=lambda _s: None)
        rep = pipe.run()
        assert not [c for c in dr.transport.calls if c["schema"] == "Paraphrases"]
        assert rep["paraphrase"]["rows_added"] == 0
        assert rep["paraphrase"]["final_train_rows"] == rep["data"]["teacher_verified_rows_round1"]
        store.close()


def test_report_is_json_serialisable(run: dict[str, Any]) -> None:
    json.dumps(run["report"]["paraphrase"])
