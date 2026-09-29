# ruff: noqa: S101, E501, S108, S608
from __future__ import annotations

import re
from collections import Counter

import pytest

from distillery.taskpacks.sql.executor import ExecOutcome, Executor, LocalExecutor
from distillery.taskpacks.sql.questions import (
    FAMILIES,
    SEEN,
    TEMPLATES,
    UNSEEN_FAMILY,
    UNSEEN_TABLE_COMBO,
    SqlTask,
    classify_heldout,
    generate_tasks,
    split_by_family,
    unseen_family_ids,
    unseen_fraction,
)
from distillery.taskpacks.sql.runner import run_select
from distillery.taskpacks.sql.schema import TABLES, build_database
from distillery.taskpacks.sql.sqltext import has_top_level_order_by
from distillery.taskpacks.sql.verifier import execution_match


@pytest.fixture(scope="module")
def db() -> bytes:
    return build_database(0)


@pytest.fixture(scope="module")
def report(db: bytes):  # type: ignore[no-untyped-def]
    return generate_tasks(db, 500, seed=7)


def test_families_and_template_count() -> None:
    assert len(TEMPLATES) >= 40
    assert len(FAMILIES) >= 10
    for f in (
        "filter",
        "join",
        "aggregation",
        "window",
        "date_math",
        "null_handling",
        "subquery_cte",
        "anti_join",
    ):
        assert f in FAMILIES


def test_generates_requested_count_and_reports_drops(report) -> None:  # type: ignore[no-untyped-def]
    assert len(report.tasks) == 500
    assert (
        report.dropped_total
        == report.dropped_error + report.dropped_empty + report.dropped_duplicate
    )
    assert report.dropped_error == 0  # no template should produce broken SQL
    assert report.attempts >= 500


def test_every_family_and_template_represented(report) -> None:  # type: ignore[no-untyped-def]
    assert {t.family for t in report.tasks} == set(FAMILIES)
    assert {t.template for t in report.tasks} == {t.name for t in TEMPLATES}


def test_gold_executes_nonempty_and_self_matches(db: bytes, report) -> None:  # type: ignore[no-untyped-def]
    for t in report.tasks:
        out = run_select(db, t.gold_sql)
        assert out.ok and len(out.rows) >= 1, t.gold_sql
        assert execution_match(db, t.gold_sql, t.gold_sql, t.requires_order).ok


def test_requires_order_consistent_and_question_mentions_it(report) -> None:  # type: ignore[no-untyped-def]
    assert any(t.requires_order for t in report.tasks)
    for t in report.tasks:
        assert t.requires_order == has_top_level_order_by(t.gold_sql)
        if t.requires_order:
            assert re.search(
                r"order|sorted|highest first|newest|from newest|top|latest|most", t.question, re.I
            )


def test_declared_tables_appear_in_sql(report) -> None:  # type: ignore[no-untyped-def]
    for t in report.tasks:
        used = {x for x in TABLES if re.search(rf"\b{x}\b", t.gold_sql)}
        assert used == set(t.tables), (t.template, used, t.tables)


def test_ids_unique_and_deterministic(db: bytes, report) -> None:  # type: ignore[no-untyped-def]
    ids = [t.task_id for t in report.tasks]
    assert len(set(ids)) == len(ids)
    assert len({t.gold_sql for t in report.tasks}) == len(ids)
    again = generate_tasks(db, 500, seed=7)
    assert [t.model_dump() for t in again.tasks] == [t.model_dump() for t in report.tasks]
    other = generate_tasks(db, 50, seed=8)
    assert [t.gold_sql for t in other.tasks] != [t.gold_sql for t in report.tasks[:50]]


def test_paraphrase_variation(report) -> None:  # type: ignore[no-untyped-def]
    by_tpl: dict[str, set[str]] = {}
    for t in report.tasks:
        by_tpl.setdefault(t.template, set()).add(re.sub(r"[\w'.$-]*\d[\w'.$-]*", "#", t.question))
    assert sum(len(v) > 1 for v in by_tpl.values()) >= 20


def test_empty_gold_is_dropped_and_counted() -> None:
    from distillery.taskpacks.sql import questions as q

    empty_db = build_database(0)
    # A DB where everything filtered by country 'ZZ' is empty: emulate via a custom template pool.
    tpl = q.Template(
        "t_empty",
        "filter",
        ("accounts",),
        "easy",
        lambda rng: ("nothing?", "SELECT name FROM accounts WHERE country = 'ZZ'"),
    )
    bad = q.Template(
        "t_bad",
        "filter",
        ("accounts",),
        "easy",
        lambda rng: ("broken", "SELECT nope FROM accounts"),
    )
    good = q.Template(
        "t_good", "filter", ("accounts",), "easy", lambda rng: ("ok", "SELECT name FROM accounts")
    )
    old = q.TEMPLATES
    q.TEMPLATES = (tpl, bad, good)
    try:
        rep = q.generate_tasks(empty_db, 1, seed=0)
    finally:
        q.TEMPLATES = old
    assert len(rep.tasks) == 1 and rep.tasks[0].template == "t_good"
    assert rep.dropped_empty == 1 and rep.dropped_error == 1 and rep.dropped_total == 2


def test_family_filter_and_unknown_family(db: bytes) -> None:
    rep = generate_tasks(db, 30, seed=1, families=("window",))
    assert rep.tasks and {t.family for t in rep.tasks} == {"window"}
    with pytest.raises(ValueError):
        generate_tasks(db, 5, seed=1, families=("nope",))


def test_tasks_are_frozen_models() -> None:
    t = SqlTask(
        task_id="x",
        family="f",
        question="q",
        gold_sql="SELECT 1",
        requires_order=False,
        difficulty="easy",
    )
    with pytest.raises(ValueError):
        t.question = "z"


# ---- split -----------------------------------------------------------------------------------


def test_split_holds_out_whole_families(report) -> None:  # type: ignore[no-untyped-def]
    held = {"window", "set_ops"}
    res = split_by_family(report.tasks, held, seed=3)
    assert not {t.family for t in res.train} & held
    assert {t.task_id for t in res.train}.isdisjoint({t.task_id for t in res.heldout})
    assert len(res.train) + len(res.heldout) == len(report.tasks)
    assert all(t.family in held for t in res.heldout if t.family in held)
    assert {t.family for t in report.tasks if t.family in held} <= {t.family for t in res.heldout}


def test_split_unseen_share_at_least_20_percent(report) -> None:  # type: ignore[no-untyped-def]
    for held in ({"window"}, {"set_ops"}, {"window", "set_ops", "conditional"}):
        res = split_by_family(report.tasks, held, seed=1, in_dist_fraction=0.9)
        assert unseen_fraction(res.train, res.heldout) >= 0.2
        ids = unseen_family_ids(res.train, res.heldout)
        assert ids == {t.task_id for t in res.heldout if t.family in held}


def test_split_deterministic_and_seed_sensitive(report) -> None:  # type: ignore[no-untyped-def]
    a = split_by_family(report.tasks, {"window"}, seed=1)
    b = split_by_family(report.tasks, {"window"}, seed=1)
    c = split_by_family(report.tasks, {"window"}, seed=2)
    assert [t.task_id for t in a.heldout] == [t.task_id for t in b.heldout]
    assert [t.task_id for t in a.heldout] != [t.task_id for t in c.heldout]


def test_split_requires_heldout_family_tasks(report) -> None:  # type: ignore[no-untyped-def]
    with pytest.raises(ValueError):
        split_by_family(report.tasks, {"no_such_family"}, seed=0)


def test_classify_heldout_labels() -> None:
    def mk(i: str, fam: str, tables: tuple[str, ...]) -> SqlTask:
        return SqlTask(
            task_id=i,
            family=fam,
            question="q",
            gold_sql="SELECT 1",
            requires_order=False,
            difficulty="easy",
            tables=tables,
        )

    train = [mk("a", "f1", ("accounts",)), mk("b", "f1", ("users",))]
    held = [
        mk("c", "f2", ("accounts",)),
        mk("d", "f1", ("accounts", "users")),
        mk("e", "f1", ("users",)),
    ]
    labels = classify_heldout(train, held)
    assert labels == {"c": UNSEEN_FAMILY, "d": UNSEEN_TABLE_COMBO, "e": SEEN}
    assert unseen_family_ids(train, held) == {"c"}
    assert unseen_fraction(train, []) == 0.0


def test_difficulty_spread(report) -> None:  # type: ignore[no-untyped-def]
    assert set(Counter(t.difficulty for t in report.tasks)) == {"easy", "medium", "hard"}


# ---- executor --------------------------------------------------------------------------------


def test_local_executor_batch_by_registry_and_path(db: bytes, tmp_path) -> None:  # type: ignore[no-untyped-def]
    ex: Executor = LocalExecutor({"mem": db})
    outs = ex.run_batch("mem", ["SELECT COUNT(*) FROM plans", "SELECT bad", "DROP TABLE plans"])
    assert [o.ok for o in outs] == [True, False, False]
    assert outs[0].rows == ((6,),) and outs[0].columns == ("COUNT(*)",)
    assert outs[1].error_kind == "syntax" and outs[2].error_kind == "forbidden"
    p = tmp_path / "d.db"
    p.write_bytes(db)
    outs2 = LocalExecutor().run_batch(str(p), ["SELECT COUNT(*) FROM plans"])
    assert outs2[0].rows == ((6,),)
    assert isinstance(outs2[0], ExecOutcome)


def test_local_executor_register_timeout_and_empty_batch(db: bytes) -> None:
    ex = LocalExecutor(timeout_s=0.2)
    ex.register("d", db)
    assert ex.run_batch("d", []) == []
    inf = "WITH RECURSIVE r(n) AS (SELECT 1 UNION ALL SELECT n+1 FROM r) SELECT COUNT(*) FROM r"
    (o,) = ex.run_batch("d", [inf])
    assert not o.ok and o.error_kind == "timeout"
