# ruff: noqa: S101
"""B3 pre-classification: every rule, the genuine default, and agreement with the verifier."""

from __future__ import annotations

import pytest

from distillery.misses import (
    CONVENTION,
    GENUINE,
    VERIFIER_FN,
    classify_miss,
    render_table,
    summarize_misses,
)
from distillery.taskpacks.sql.runner import ExecOutcome
from distillery.taskpacks.sql.verifier import compare_outcomes


def oc(cols: tuple[str, ...], rows: list[tuple[object, ...]]) -> ExecOutcome:
    return ExecOutcome(True, cols, tuple(rows))  # type: ignore[arg-type]


GOLD = oc(("id", "name"), [(1, "a"), (2, "b")])


@pytest.mark.parametrize(
    ("cand", "gold", "order", "category", "rule"),
    [
        (
            oc(("id", "name", "x"), [(1, "a", 9), (2, "b", 9)]),
            GOLD,
            False,
            CONVENTION,
            "extra_columns",
        ),
        (oc(("name", "id"), [("a", 1), ("b", 2)]), GOLD, False, CONVENTION, "column_order"),
        (oc(("id", "name"), [(2, "b"), (1, "a")]), GOLD, True, VERIFIER_FN, "order_not_required"),
        (
            oc(("avg",), [(3.14,), (2.72,)]),
            oc(("avg",), [(3.14159,), (2.71828,)]),
            False,
            VERIFIER_FN,
            "float_rounding_2dp",
        ),
        (
            oc(("a",), [("",), ("x",)]),
            oc(("a",), [(None,), ("x",)]),
            False,
            CONVENTION,
            "null_vs_empty",
        ),
        (
            oc(("a",), [(1,), (1,), (2,)]),
            oc(("a",), [(1,), (2,)]),
            False,
            CONVENTION,
            "duplicate_rows",
        ),
        (oc(("id", "name"), [(1, "a")]), GOLD, False, GENUINE, "wrong_result"),
        (oc(("id",), [(1,), (2,)]), GOLD, False, GENUINE, "wrong_result"),  # a column is MISSING
        (
            ExecOutcome(False, error_kind="syntax", error="no such column"),
            GOLD,
            False,
            GENUINE,
            "execution_error",
        ),
        (None, GOLD, False, GENUINE, "no_sql_extracted"),
    ],
)
def test_rules(cand, gold, order, category, rule) -> None:  # type: ignore[no-untyped-def]
    got = classify_miss(cand, gold, order)
    assert (got["category"], got["rule"]) == (category, rule)
    assert got["diff"]


def test_a_classified_miss_is_a_verifier_miss() -> None:
    """Every non-genuine example above is something the verifier rejects (so it IS a miss)."""
    cand = oc(("name", "id"), [("a", 1), ("b", 2)])
    assert not compare_outcomes(cand, GOLD, False).ok


def test_summary_and_table_have_ids_and_no_question_text() -> None:
    entries = [
        {"set": "heldout", "setup": "S0", "task_id": "t1", **classify_miss(None, GOLD, False)},
        {
            "set": "heldout",
            "setup": "S1",
            "task_id": "t2",
            **classify_miss(oc(("name", "id"), [("a", 1), ("b", 2)]), GOLD, False),
        },
    ]
    s = summarize_misses(entries)
    assert s["counts"]["S0"][GENUINE] == 1 and s["counts"]["S1"][CONVENTION] == 1
    assert [r["task_id"] for r in s["review"]] == ["t2"]
    table = render_table(s["review"])
    assert "t2" in table and "column_order" in table and "question" not in table
