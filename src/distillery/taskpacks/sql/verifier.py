# ruff: noqa: S608
"""Execution-match verification of candidate SQL against gold SQL, plus deliberate corruptions.

Comparison rules: column count must match (names are ignored); rows are compared as a multiset
unless ``requires_order``; numbers compare by value (int 3 == float 3.0, floats rounded to 6 dp);
NULL only equals NULL; strings are compared exactly (no case or whitespace normalisation).
"""

from __future__ import annotations

import random
import re
from collections import Counter
from dataclasses import dataclass

from distillery.taskpacks.sql.runner import (
    DEFAULT_MAX_ROWS,
    DEFAULT_TIMEOUT_S,
    DbRef,
    ExecOutcome,
    Scalar,
    run_select,
)
from distillery.taskpacks.sql.sqltext import has_top_level_order_by, mask_sql

FLOAT_DP = 6


@dataclass(frozen=True)
class VerifyResult:
    ok: bool
    reason: str


def normalise_value(v: Scalar) -> Scalar:
    """Canonical form of one cell: floats rounded to 6 dp; everything else unchanged."""
    if isinstance(v, float):
        return round(v, FLOAT_DP)
    return v


def normalise_rows(rows: tuple[tuple[Scalar, ...], ...]) -> list[tuple[Scalar, ...]]:
    return [tuple(normalise_value(c) for c in r) for r in rows]


def compare_outcomes(
    candidate: ExecOutcome, gold: ExecOutcome, requires_order: bool
) -> VerifyResult:
    """Compare two already-executed outcomes (usable with any Executor implementation)."""
    if not gold.ok:
        return VerifyResult(False, f"gold error ({gold.error_kind}): {gold.error}")
    if not candidate.ok:
        return VerifyResult(False, f"candidate error ({candidate.error_kind}): {candidate.error}")
    if len(candidate.columns) != len(gold.columns):
        return VerifyResult(
            False,
            f"column count mismatch: got {len(candidate.columns)}, expected {len(gold.columns)}",
        )
    got, want = normalise_rows(candidate.rows), normalise_rows(gold.rows)
    if requires_order:
        if got == want:
            return VerifyResult(True, "match (ordered)")
        if Counter(got) == Counter(want):
            return VerifyResult(False, "rows match but order differs")
    elif Counter(got) == Counter(want):
        return VerifyResult(True, "match")
    if len(got) != len(want):
        return VerifyResult(False, f"row count mismatch: got {len(got)}, expected {len(want)}")
    extra = Counter(got) - Counter(want)
    missing = Counter(want) - Counter(got)
    return VerifyResult(
        False,
        f"row contents differ: {sum(extra.values())} unexpected, {sum(missing.values())} missing",
    )


def execution_match(
    db: DbRef,
    candidate_sql: str,
    gold_sql: str,
    requires_order: bool,
    *,
    timeout_s: float = DEFAULT_TIMEOUT_S,
    max_rows: int = DEFAULT_MAX_ROWS,
) -> VerifyResult:
    """Run both queries read-only and compare results."""
    gold = run_select(db, gold_sql, timeout_s=timeout_s, max_rows=max_rows)
    cand = run_select(db, candidate_sql, timeout_s=timeout_s, max_rows=max_rows)
    return compare_outcomes(cand, gold, requires_order)


# --- corruption helpers ----------------------------------------------------------------------


@dataclass(frozen=True)
class Corruption:
    kind: str
    sql: str


_TEXT_MUTATIONS: tuple[tuple[str, re.Pattern[str], str], ...] = (
    ("is_not_null_flip", re.compile(r"\bIS\s+NOT\s+NULL\b", re.I), "IS NULL"),
    ("is_null_flip", re.compile(r"\bIS\s+NULL\b", re.I), "IS NOT NULL"),
    ("inner_to_left_join", re.compile(r"(?<!LEFT )\bJOIN\b", re.I), "LEFT JOIN"),
    ("left_to_inner_join", re.compile(r"\bLEFT\s+JOIN\b", re.I), "JOIN"),
    ("gt_to_gte", re.compile(r"(?<![<>=!])>(?!=)"), ">="),
    ("lt_to_lte", re.compile(r"(?<![<>=!])<(?![=>])"), "<="),
    ("gte_to_gt", re.compile(r">="), ">"),
    ("desc_to_asc", re.compile(r"\bDESC\b", re.I), "ASC"),
    ("asc_to_desc", re.compile(r"\bASC\b", re.I), "DESC"),
    ("drop_distinct", re.compile(r"\bDISTINCT\s+", re.I), ""),
    ("count_distinct_to_count", re.compile(r"COUNT\(\s*DISTINCT\s+", re.I), "COUNT("),
    ("avg_to_sum", re.compile(r"\bAVG\(", re.I), "SUM("),
    ("sum_to_max", re.compile(r"\bSUM\(", re.I), "MAX("),
    ("and_to_or", re.compile(r"\bAND\b", re.I), "OR"),
    ("coalesce_zero_removed", re.compile(r"COALESCE\(([^(),]+),\s*0\)", re.I), r"\1"),
    ("count_star_off_by_one", re.compile(r"COUNT\(\*\)", re.I), "COUNT(1) + 1"),
    ("not_exists_flip", re.compile(r"\bNOT\s+EXISTS\b", re.I), "EXISTS"),
    ("in_flip", re.compile(r"\bNOT\s+IN\b", re.I), "IN"),
)


def _replace_nth_unmasked(sql: str, pat: re.Pattern[str], repl: str, n: int) -> str | None:
    """Replace the n-th (mod count) match of ``pat`` lying outside literals/comments."""
    hits = list(pat.finditer(mask_sql(sql)))
    if not hits:
        return None
    m = hits[n % len(hits)]
    orig = pat.match(sql, m.start())
    replacement = orig.expand(repl) if orig is not None else repl
    return sql[: m.start()] + replacement + sql[m.end() :]


_NUM_LIT = re.compile(r"(?<![\w.'])(\d+)(?![\w.'])")


def _mutate_numeric_literal(sql: str, rng: random.Random) -> str | None:
    masked = mask_sql(sql)
    hits = [m for m in _NUM_LIT.finditer(masked)]
    if not hits:
        return None
    m = rng.choice(hits)
    return sql[: m.start()] + str(int(m.group(1)) + rng.choice((1, 2, 7))) + sql[m.end() :]


def _mutate_string_literal(sql: str, rng: random.Random) -> str | None:
    masked = mask_sql(sql)
    hits = [m for m in re.finditer(r"'([^']*)'", masked)]
    hits = [m for m in hits if m.group(1).strip() == "" and sql[m.start() + 1 : m.end() - 1]]
    if not hits:
        return None
    m = rng.choice(hits)
    lit = sql[m.start() + 1 : m.end() - 1]
    return sql[: m.start()] + "'" + lit + "x'" + sql[m.end() :]


def _unique_names(cols: tuple[str, ...]) -> bool:
    return len(set(cols)) == len(cols) and all(c for c in cols)


def _q(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def corrupt_sql(
    gold_sql: str,
    db: DbRef,
    rng: random.Random,
    *,
    max_variants: int = 8,
) -> list[Corruption]:
    """Deliberately wrong but executable variants of ``gold_sql``.

    Every returned variant is executed against ``db`` and kept only if it runs successfully AND
    ``execution_match`` rejects it (using ORDER BY detection on the gold). Deterministic for a
    given ``rng`` state. Kinds include: extra_row, missing_row, extra_column, dropped_column,
    null_to_empty (wrong NULL handling), and text mutations (join type, comparison, aggregate,
    NULL predicate, literal changes, ...).
    """
    gold = run_select(db, gold_sql)
    if not gold.ok:
        return []
    requires_order = has_top_level_order_by(gold_sql)
    body = gold_sql.strip().rstrip(";").rstrip()
    cands: list[Corruption] = []

    cands.append(
        Corruption(
            "extra_row",
            f"SELECT * FROM ({body}) UNION ALL SELECT * FROM (SELECT * FROM ({body}) LIMIT 1)",
        )
    )
    cands.append(
        Corruption("missing_row", f"SELECT * FROM ({body}) LIMIT {max(len(gold.rows) - 1, 0)}")
    )
    cands.append(Corruption("extra_column", f"SELECT *, 1 AS extra_col FROM ({body})"))
    if _unique_names(gold.columns):
        if len(gold.columns) >= 2:
            keep = ", ".join(_q(c) for c in gold.columns[:-1])
            cands.append(Corruption("dropped_column", f"SELECT {keep} FROM ({body})"))
            rev = ", ".join(_q(c) for c in reversed(gold.columns))
            cands.append(Corruption("swapped_columns", f"SELECT {rev} FROM ({body})"))
        for j in range(len(gold.columns)):
            if any(r[j] is None for r in gold.rows):
                sel = ", ".join(
                    f"COALESCE({_q(c)}, '')" if i == j else _q(c)
                    for i, c in enumerate(gold.columns)
                )
                cands.append(Corruption("null_to_empty", f"SELECT {sel} FROM ({body})"))
                break
    if requires_order:
        cands.append(
            Corruption("reversed_order", f"SELECT * FROM (SELECT * FROM ({body})) ORDER BY 1 DESC")
        )
    for kind, pat, repl in _TEXT_MUTATIONS:
        mutated = _replace_nth_unmasked(gold_sql, pat, repl, rng.randrange(1000))
        if mutated is not None:
            cands.append(Corruption(kind, mutated))
    for kind, fn in (
        ("numeric_literal", _mutate_numeric_literal),
        ("string_literal", _mutate_string_literal),
    ):
        mutated = fn(gold_sql, rng)
        if mutated is not None:
            cands.append(Corruption(kind, mutated))

    rng.shuffle(cands)
    good: list[Corruption] = []
    seen: set[str] = set()
    for c in cands:
        if c.sql in seen or c.sql.strip() == gold_sql.strip():
            continue
        seen.add(c.sql)
        outcome = run_select(db, c.sql)
        if not outcome.ok:
            continue
        if compare_outcomes(outcome, gold, requires_order).ok:
            continue
        good.append(c)
        if len(good) >= max_variants:
            break
    return good
