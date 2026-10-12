# ruff: noqa: S101, S311
from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

import pytest
from router_helpers import DB, DB_SHA, DDL, PACK, RUN, confirmed_file, make_run

from distillery import evaluator as ev
from distillery.humanset import HumanSetError
from distillery.router_data import (
    SPLIT_SEED,
    RouterDataError,
    SplitParams,
    build_splits,
    compute_signals,
    question_of_training_row,
    read_signal_rows,
    training_questions,
    unpack_student_output,
    write_signal_rows,
)
from distillery.taskpacks.sql.executor import LocalExecutor
from distillery.taskpacks.sql.human import normalise_question

PRM = SplitParams(cal_n=40, test_n=40, oversample=3.0)
NATURAL_PAIRS = [
    (f"Natural style question {i}: how many users have role {r}?", f"SELECT COUNT(*) FROM users WHERE role = '{r}'")
    for i, r in enumerate(["owner", "admin", "member", "viewer"] * 4)
]  # fmt: skip


def fresh_draw() -> list[Any]:
    reserved = PACK.stress_families()
    return PACK.generate_tasks(
        DB, math.ceil(PRM.pool_target("templated") * PRM.oversample), SPLIT_SEED,
        exclude_families=reserved, skeleton_cap=PACK.default_skeleton_cap,
    ).tasks  # fmt: skip


def test_training_questions_from_rows_and_split_stage(tmp_path: Path) -> None:
    run = make_run(tmp_path, n_train=40, natural_rows=3)
    tq = training_questions(run.store, RUN)
    assert tq.row_files == ("round1/train.jsonl", "round1-nabcd1234/train.jsonl")
    assert len(tq.train) == 43 and tq.split_train == 40 and tq.row_count == 83 + 0
    assert "natural training question number 0?" in tq.train
    assert len(tq.dev) == 20
    only = training_questions(run.store, RUN, round_dirs=["round1"])
    assert len(only.train) == 40
    row = json.loads(
        (run.store.run_dir(RUN) / "round1" / "train.jsonl").read_text().splitlines()[0]
    )
    assert question_of_training_row(row) == normalise_question(run.train[0].question)
    with pytest.raises(RouterDataError):
        training_questions(run.store, RUN, round_dirs=["round9"])


def test_build_splits_counts_mix_exchangeable_overlap_and_seal(tmp_path: Path) -> None:
    fresh = fresh_draw()
    inject = [fresh[0], fresh[1], fresh[2]]
    nat = confirmed_file(tmp_path / "nat.json", NATURAL_PAIRS)
    gateb_pairs = [
        ("What is the number of accounts?", "SELECT COUNT(*) FROM accounts"),
        (NATURAL_PAIRS[0][0], NATURAL_PAIRS[0][1]),  # duplicates a natural-file question
    ]
    gateb = confirmed_file(tmp_path / "gateb.json", gateb_pairs, tag="g")
    # natural file also holds: a train-overlap, a sealed-human overlap, a bad-gold row
    nat_pairs = [
        *NATURAL_PAIRS,
        ("natural training question number 0?", "SELECT COUNT(*) FROM users"),
        (_human_pair := ("How many plans have unlimited seats?", "SELECT COUNT(*) FROM plans")),
        ("Broken gold please", "SELECT nonexistent FROM nowhere"),
    ]
    nat = confirmed_file(tmp_path / "nat.json", nat_pairs)
    run = make_run(
        tmp_path,
        extra_train_q=[inject[0].question],
        extra_dev_q=[inject[1].question],
        extra_gate=[inject[2]],
    )
    assert _human_pair[0] in {p[0] for p in run.human_pairs}

    rep = build_splits(
        run.store, RUN, executor=run.executor, db_ref=run.db_ref, db_sha=DB_SHA,
        natural_files=[nat], human_files=[gateb], params=PRM,
    )  # fmt: skip
    cal = ev.router_split_items(run.store, RUN, "cal")
    test = ev.router_split_items(run.store, RUN, "test")
    assert rep["cal_sha256"] == run.store.router_split_sha(RUN, "cal")
    assert rep["test_sha256"] == run.store.router_split_sha(RUN, "test")
    assert rep["n"] == {"cal": len(cal), "test": len(test)}

    # drop accounting: exact overlap with train / dev / gate / sealed human / Gate B file
    drops = rep["per_kind"]["templated"]["dropped"]
    assert drops["train"] >= 1 and drops["dev"] >= 1 and drops["gate"] >= 1
    ndrops = rep["per_kind"]["natural"]["dropped"]
    assert ndrops["train"] == 1 and ndrops["human_sealed"] == 1 and ndrops["human_file"] >= 1
    assert rep["generation"]["natural"]["gold_failed"] == 1
    # nothing that overlaps made it into either split
    qs = [normalise_question(i["question"]) for i in [*cal, *test]]
    assert len(qs) == len(set(qs))
    banned = {
        normalise_question(t.question) for t in [*run.train, *run.dev, *run.heldout, *run.stress]
    }
    banned |= {normalise_question(x.question) for x in inject}
    banned |= {normalise_question(q) for q, _ in [*run.human_pairs, *gateb_pairs]}
    banned |= {"natural training question number 0?"}
    assert not banned & set(qs)
    assert rep["overlap"]["exact_cal_vs_test"] == 0

    # mix: templated 40 (20/20), stress 20 (10/10), natural = what exists (13 valid), split evenly
    mix = rep["realised_mix"]
    assert mix["cal"]["templated"] == mix["test"]["templated"] == 20
    assert mix["cal"]["stress"] == mix["test"]["stress"] == 10
    n_nat = rep["per_kind"]["natural"]["pool"]
    assert (
        0 < n_nat < PRM.pool_target("natural")
        and abs(mix["cal"]["natural"] - mix["test"]["natural"]) <= 1
    )
    assert mix["cal"]["natural"] + mix["test"]["natural"] == n_nat
    assert rep["per_kind"]["natural"]["shortfall"] == PRM.pool_target("natural") - n_nat
    # kinds / families: stress only from the reserved families, templated never
    reserved = set(PACK.stress_families())
    for i in [*cal, *test]:
        assert (i["family"] in reserved) == (i["kind"] == "stress")
        assert i["kind"] in {"templated", "natural", "stress"}
    # sealed splits are what was written; seal is stable
    assert run.store.load_router_split(RUN, "cal") == cal
    with pytest.raises(RouterDataError, match="already sealed"):
        build_splits(
            run.store, RUN, executor=run.executor, db_ref=run.db_ref, db_sha=DB_SHA,
            natural_files=[nat], human_files=[gateb], params=PRM,
        )  # fmt: skip
    saved = json.loads((run.store.run_dir(RUN) / "router" / "splits.json").read_text())
    assert saved["cal_sha256"] == rep["cal_sha256"]
    text = json.dumps(saved)  # counts only: no question or gold text
    assert all(i["question"] not in text and i["gold_sql"] not in text for i in [*cal, *test])


def test_build_splits_is_deterministic_and_seeded(tmp_path: Path) -> None:
    nat = confirmed_file(tmp_path / "nat.json", NATURAL_PAIRS)
    a = make_run(tmp_path / "a")
    b = make_run(tmp_path / "b")
    kw: dict[str, Any] = {"natural_files": [nat], "params": PRM, "db_sha": DB_SHA}
    ra = build_splits(a.store, RUN, executor=a.executor, db_ref=a.db_ref, **kw)
    rb = build_splits(b.store, RUN, executor=b.executor, db_ref=b.db_ref, **kw)
    assert (ra["cal_sha256"], ra["test_sha256"]) == (rb["cal_sha256"], rb["test_sha256"])
    c = make_run(tmp_path / "c")
    other = SplitParams(seed=7, cal_n=40, test_n=40)
    rc = build_splits(
        c.store, RUN, executor=c.executor, db_ref=c.db_ref, natural_files=[nat], params=other
    )
    assert rc["cal_sha256"] != ra["cal_sha256"] and rc["seed"] == 7


def test_exchangeability_cal_and_test_are_a_random_halving(tmp_path: Path) -> None:
    run = make_run(tmp_path)
    rep = build_splits(run.store, RUN, executor=run.executor, db_ref=run.db_ref, params=PRM)
    cal = ev.router_split_items(run.store, RUN, "cal")
    test = ev.router_split_items(run.store, RUN, "test")
    # no natural file: the natural share is simply absent, the rest is split evenly
    assert rep["realised_mix"]["cal"]["natural"] == 0 == rep["realised_mix"]["test"]["natural"]
    for kind in ("templated", "stress"):
        c = Counter(i["family"] for i in cal if i["kind"] == kind)
        t = Counter(i["family"] for i in test if i["kind"] == kind)
        for fam in set(c) | set(t):  # families spread over both halves, not sorted into one
            assert abs(c[fam] - t[fam]) <= max(4, 0.5 * (c[fam] + t[fam]))
    assert {i["task_id"] for i in cal}.isdisjoint({i["task_id"] for i in test})


def test_fit_items_tagging_and_db_mismatch(tmp_path: Path) -> None:
    run = make_run(tmp_path)
    gb = confirmed_file(
        tmp_path / "gb.json", [("Totally new question about accounts?", "SELECT 1")]
    )
    fit = ev.router_fit_items(run.store, RUN, [gb], db_sha=DB_SHA)
    assert Counter(i["source"] for i in fit) == Counter(
        {"heldout": 30, "stress": 25, "human": 2, "human_file:gb.json": 1}
    )
    assert {i["kind"] for i in fit if i["source"] == "stress"} == {"stress"}
    bad = confirmed_file(tmp_path / "bad.json", [("x?", "SELECT 1")], db_sha="f" * 64)
    with pytest.raises(HumanSetError):
        ev.router_fit_items(run.store, RUN, [bad], db_sha=DB_SHA)
    with pytest.raises(HumanSetError):
        build_splits(
            run.store, RUN, executor=run.executor, db_ref=run.db_ref, db_sha=DB_SHA,
            natural_files=[bad], params=PRM,
        )  # fmt: skip
    assert run.store.router_split_sha(RUN, "cal") is None  # nothing sealed on refusal


def test_split_params_validation() -> None:
    with pytest.raises(RouterDataError):
        SplitParams(mix={"templated": 0.5, "natural": 0.5, "stress": 0.5})
    with pytest.raises(RouterDataError):
        SplitParams(cal_n=0)
    assert SplitParams().pool_target("templated") == 500
    assert SplitParams().pool_target("natural") == SplitParams().pool_target("stress") == 250


# ------------------------------------------------------------------ signals


class Sim:
    def __init__(self, v: float = 0.5) -> None:
        self.v = v
        self.seen: list[str] = []

    def nn_sim_many(self, questions: Any) -> list[float]:
        self.seen += list(questions)
        return [self.v + 0.01 * i for i, _ in enumerate(questions)]


class Out:
    def __init__(self, text: str, lp: float | None) -> None:
        self.text, self.mean_logprob = text, lp


class Student:
    """Right for items whose index is even; wrong (valid SQL) for odd; broken for index 4."""

    def __init__(self, gold: list[str], with_logprob: bool = True) -> None:
        self.gold, self.with_lp, self.calls = gold, with_logprob, 0

    def generate(self, batch: Any) -> list[Any]:
        self.calls += 1
        outs: list[Any] = []
        for i, _ in enumerate(batch):
            text = self.gold[i] if i % 2 == 0 else "SELECT 1 WHERE 1 = 0"
            if i == 4:
                text = "no sql here"
            outs.append(Out(text, -0.1 * (i + 1)) if self.with_lp else text)
        return outs


class Teacher:
    def __init__(self, gold: list[str]) -> None:
        self.gold = gold

    def generate(self, batch: Any) -> list[str]:
        return [f"```sql\n{g}\n```" for g in self.gold[: len(batch)]]


def test_unpack_student_output_variants() -> None:
    assert unpack_student_output("x") == ("x", None)
    assert unpack_student_output(Out("t", -0.5)) == ("t", -0.5)
    assert unpack_student_output({"text": "t", "mean_logprob": float("nan")}) == ("t", None)
    with pytest.raises(RouterDataError):
        unpack_student_output(3)


def test_compute_signals_rows() -> None:
    reserved = PACK.stress_families()
    tasks = PACK.generate_tasks(DB, 10, 5, exclude_families=reserved).tasks
    items = [{**t.model_dump(mode="json"), "kind": "templated"} for t in tasks]
    ex = LocalExecutor({"db": DB})
    st = Student([t.gold_sql for t in tasks])
    teacher = Teacher([t.gold_sql for t in tasks])
    sim = Sim()
    rows = compute_signals(
        items, st, ex, sim, "db", schema_ddl=DDL, teacher=teacher, split="test"
    )  # fmt: skip
    assert st.calls == 1 and len(rows) == 10 and len(sim.seen) == 10
    assert [r["task_id"] for r in rows] == [t.task_id for t in tasks]
    for i, r in enumerate(rows):
        assert r["kind"] == "templated" and r["split"] == "test"
        assert r["mean_logprob"] == pytest.approx(-0.1 * (i + 1))
        assert r["nn_sim"] == pytest.approx(0.5 + 0.01 * i)
        assert len(r["output_sha256"]) == 64
        assert r["teacher_correct"] is True  # the teacher fake answers every item with gold
    assert rows[0]["correct"] is True and rows[0]["exec_ok"] == 1
    assert rows[1]["correct"] is False and rows[1]["exec_ok"] == 0  # valid SQL but 0 rows
    assert rows[4]["correct"] is False and rows[4]["exec_ok"] == 0  # nothing extractable
    assert rows[2]["correct"] is True
    # plain-string student: no logprob; no teacher: teacher_correct None
    rows2 = compute_signals(
        items,
        Student([t.gold_sql for t in tasks], with_logprob=False),
        ex,
        sim,
        "db",
        schema_ddl=DDL,
    )
    assert all(r["mean_logprob"] is None and r["teacher_correct"] is None for r in rows2)
    assert rows2[0]["output_sha256"] == rows[0]["output_sha256"]  # hash of the raw text
    assert rows[1]["teacher_correct"] is True


def test_compute_signals_failures_and_roundtrip(tmp_path: Path) -> None:
    tasks = PACK.generate_tasks(DB, 3, 5, exclude_families=PACK.stress_families()).tasks
    items = [{**t.model_dump(mode="json"), "kind": "templated"} for t in tasks]
    ex = LocalExecutor({"db": DB})

    class Short:
        def generate(self, batch: Any) -> list[str]:
            return ["SELECT 1"]

    with pytest.raises(RuntimeError, match="1 outputs for 3"):
        compute_signals(items, Short(), ex, Sim(), "db", schema_ddl=DDL)
    bad = [{**items[0], "gold_sql": "SELECT nope FROM nowhere"}]
    with pytest.raises(RuntimeError, match="gold answer failed"):
        compute_signals(bad, Student([""]), ex, Sim(), "db", schema_ddl=DDL)
    assert compute_signals([], Short(), ex, Sim(), "db", schema_ddl=DDL) == []
    rows = compute_signals(
        items, Student([t.gold_sql for t in tasks]), ex, Sim(), "db", schema_ddl=DDL
    )
    p = tmp_path / "s.json"
    sha = write_signal_rows(p, split="fit", split_sha256=None, rows=rows, extra={"model": "m"})
    doc = read_signal_rows(p)
    assert doc["rows"] == rows and doc["model"] == "m" and len(sha) == 64
    p.write_text("{}")
    with pytest.raises(RouterDataError):
        read_signal_rows(p)
