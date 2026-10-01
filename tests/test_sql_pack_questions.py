# ruff: noqa: S101, E501, S108, S608
from __future__ import annotations

import math
import re
from collections import Counter

import pytest

from distillery.taskpacks.sql.executor import ExecOutcome, Executor, LocalExecutor
from distillery.taskpacks.sql.questions import (
    DEFAULT_SKELETON_CAP,
    FAMILIES,
    STRESS_FAMILY_COUNT,
    STRESS_SEED,
    TEMPLATES,
    SqlTask,
    generate_tasks,
    skeleton,
    stress_families,
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
    assert report.dropped_total == (
        report.dropped_error + report.dropped_empty + report.dropped_duplicate + report.dropped_cap
        + report.dropped_skeleton_cap + report.dropped_duplicate_question
    )  # fmt: skip
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
                r"order|sorted|highest first|newest|top|latest|most|largest|biggest|date order|chronolog",
                t.question,
                re.I,
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


# ---- stress families, skeletons, capacity caps ------------------------------------------------


def test_stress_families_follow_the_written_rule() -> None:
    import random

    assert STRESS_SEED == 777 and STRESS_FAMILY_COUNT == 2
    expected = tuple(sorted(random.Random(777).sample(sorted(FAMILIES), k=2)))
    assert stress_families() == expected
    assert len(expected) == 2 and set(expected) <= set(FAMILIES)
    assert stress_families() == stress_families()  # fixed, not data dependent


def test_skeleton_masks_digits_quotes_and_vocabulary() -> None:
    a = skeleton("List the names of active accounts in DE.")
    b = skeleton("List the names of  trial accounts in US.")
    assert a == b
    assert skeleton("Invoices over $500 issued after 2024-01-03?") == skeleton(
        "Invoices over $2000 issued after 2025-11-30?"
    )
    assert skeleton("Users with role 'admin'") == skeleton("Users with role 'viewer'")
    assert skeleton("How many users?") != skeleton("How many accounts?")


def test_exclude_families(db: bytes) -> None:
    stress = stress_families()
    rep = generate_tasks(db, 200, seed=3, exclude_families=stress)
    assert len(rep.tasks) == 200 and not {t.family for t in rep.tasks} & set(stress)
    with pytest.raises(ValueError):
        generate_tasks(db, 5, seed=1, families=("window",), exclude_families=("window",))


def _tpl(name: str, family: str, build):  # type: ignore[no-untyped-def]
    from distillery.taskpacks.sql import questions as q

    return q.Template(name, family, ("accounts",), "easy", build)


def _with_templates(tpls, fn):  # type: ignore[no-untyped-def]
    from distillery.taskpacks.sql import questions as q

    old = q.TEMPLATES
    q.TEMPLATES = tuple(tpls)
    try:
        return fn()
    finally:
        q.TEMPLATES = old


def test_tiny_template_is_capped_at_its_distinct_sql_capacity(db: bytes) -> None:
    from distillery.taskpacks.sql import questions as q

    tiny = _tpl(
        "t_tiny", "filter",
        lambda rng: (f"tiny {(v := rng.choice((1, 2)))}?",
                     f"SELECT name FROM accounts WHERE account_id = {v}"),
    )  # fmt: skip
    big = _tpl(
        "t_big", "join",
        lambda rng: (f"big {(v := rng.randint(1, 150))}?",
                     f"SELECT name FROM accounts WHERE account_id = {v}"),
    )  # fmt: skip
    assert q.template_capacity(tiny) == 2
    rep = _with_templates(
        (tiny, big), lambda: q.generate_tasks(db, 60, seed=0, template_share=1.0, family_share=1.0)
    )
    counts = Counter(t.template for t in rep.tasks)
    assert counts["t_tiny"] == 2 and len(rep.tasks) == 60
    assert "t_tiny" in rep.capped_templates
    # the tiny template left the pool as soon as its capacity was used: no streak of duplicates
    assert rep.dropped_by_template["t_tiny"] == 0


def test_skeleton_cap_is_enforced_and_counted(db: bytes) -> None:
    rep = generate_tasks(db, 300, seed=5, skeleton_cap=3)
    per = Counter(skeleton(t.question) for t in rep.tasks)
    assert max(per.values()) <= 3
    assert rep.dropped_skeleton_cap > 0
    assert rep.dropped_total == (
        rep.dropped_error + rep.dropped_empty + rep.dropped_duplicate + rep.dropped_cap
        + rep.dropped_skeleton_cap + rep.dropped_duplicate_question
    )  # fmt: skip


def test_duplicate_questions_with_different_sql_are_dropped(db: bytes) -> None:
    from distillery.taskpacks.sql import questions as q

    amb = _tpl(
        "t_amb", "filter",
        lambda rng: ("same words?", f"SELECT name FROM accounts WHERE account_id = {rng.randint(1, 99)}"),
    )  # fmt: skip
    other = _tpl(
        "t_other", "join",
        lambda rng: (f"other {(v := rng.randint(1, 150))}?",
                     f"SELECT name FROM accounts WHERE account_id = {v}"),
    )  # fmt: skip
    rep = _with_templates((amb, other), lambda: q.generate_tasks(db, 20, seed=0, family_share=1.0))
    assert Counter(t.template for t in rep.tasks)["t_amb"] == 1
    assert rep.dropped_duplicate_question > 0
    assert len({t.question for t in rep.tasks}) == len(rep.tasks)


def test_prereg_training_pool_fills_exactly(db: bytes) -> None:
    """The generator alone must supply the prereg pool (train+dev+gate x oversample) without the
    stress families and under the default skeleton cap; otherwise the split would fall short."""
    n = math.ceil((1800 + 150 + 300) * 1.1)
    rep = generate_tasks(
        db, n, seed=1234, exclude_families=stress_families(), skeleton_cap=DEFAULT_SKELETON_CAP
    )
    assert len(rep.tasks) == n
    assert len({t.question for t in rep.tasks}) == n == len({t.gold_sql for t in rep.tasks})
    assert max(Counter(skeleton(t.question) for t in rep.tasks).values()) <= DEFAULT_SKELETON_CAP


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


# ---- caps and difficulty mix -----------------------------------------------------------------


@pytest.fixture(scope="module")
def report_1500(db: bytes):  # type: ignore[no-untyped-def]
    return generate_tasks(db, 1500, seed=0)


def test_caps_hold_at_1500(report_1500) -> None:  # type: ignore[no-untyped-def]
    assert len(report_1500.tasks) == 1500
    assert len({t.gold_sql for t in report_1500.tasks}) == 1500
    assert report_1500.template_cap == 60 and report_1500.family_cap == 375
    by_tpl = Counter(t.template for t in report_1500.tasks)
    by_fam = Counter(t.family for t in report_1500.tasks)
    assert max(by_tpl.values()) <= 0.04 * 1500
    assert max(by_fam.values()) <= 0.25 * 1500
    assert len(by_fam) == len(FAMILIES)


def test_difficulty_mix_at_1500(report_1500) -> None:  # type: ignore[no-untyped-def]
    d = Counter(t.difficulty for t in report_1500.tasks)
    assert d["hard"] >= 0.15 * 1500
    assert d["medium"] >= 0.35 * 1500


def test_caps_are_enforced_and_counted(db: bytes) -> None:
    from distillery.taskpacks.sql import questions as q

    # one template with a huge parameter space, one tiny: the big one must be capped.
    big = q.Template(
        "t_big",
        "filter",
        ("accounts",),
        "easy",
        lambda rng: (
            f"big {(v := rng.randint(0, 100000) % 150)}",
            f"SELECT name FROM accounts WHERE account_id > {v}",
        ),
    )
    small = q.Template(
        "t_small", "join", ("accounts",), "easy", lambda rng: ("q", "SELECT name FROM accounts")
    )
    other = q.Template(
        "t_o",
        "window",
        ("accounts",),
        "easy",
        lambda rng: (
            f"other {(v := rng.randint(1, 150))}",
            f"SELECT name FROM accounts WHERE account_id = {v}",
        ),
    )
    old = q.TEMPLATES
    q.TEMPLATES = (big, small, other)
    try:
        rep = q.generate_tasks(db, 100, seed=0, template_share=0.2, family_share=0.4)
    finally:
        q.TEMPLATES = old
    counts = Counter(t.template for t in rep.tasks)
    assert rep.template_cap == 34  # ceil(n / len(pool)) lifts the cap so n stays reachable
    assert counts["t_big"] <= rep.template_cap and counts["t_small"] == 1
    assert Counter(t.family for t in rep.tasks)["filter"] <= rep.family_cap


def test_cap_drops_are_reported(db: bytes) -> None:
    rep = generate_tasks(db, 400, seed=0, template_share=0.01, family_share=0.25)
    # tiny template cap (ceil(400/len(pool)) lifts it): every template stays within the cap
    assert max(Counter(t.template for t in rep.tasks).values()) <= rep.template_cap
    assert rep.dropped_total == (
        rep.dropped_error + rep.dropped_empty + rep.dropped_duplicate + rep.dropped_cap
        + rep.dropped_skeleton_cap + rep.dropped_duplicate_question
    )  # fmt: skip
