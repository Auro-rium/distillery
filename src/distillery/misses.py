"""B3: rule-based pre-classification of a model's misses. Pure; no LLM, no I/O, no verifier change.

For a miss (the verifier said "not equal") the candidate's and the gold's already-executed result
sets are compared under ONE relaxation at a time. If a relaxation makes them equal, the miss is
a candidate for human review rather than a genuine error:

* ``convention_mismatch``: the answer is right but its shape follows another convention
  (extra columns, different column order, NULL vs '', duplicate rows from a missing DISTINCT).
* ``verifier_fn_candidate``: the verifier may be too strict (float rounding finer than the
  checker's 6 dp, ordering demanded by the gold's ORDER BY).
* ``genuine``: everything else, including execution errors and a missing SQL.

A classification is a hint for the reviewer, never a verdict: nothing here changes any score.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from itertools import islice, permutations
from typing import Any

from distillery.taskpacks.sql.runner import ExecOutcome, Scalar
from distillery.taskpacks.sql.verifier import normalise_rows

GENUINE = "genuine"
CONVENTION = "convention_mismatch"
VERIFIER_FN = "verifier_fn_candidate"
CATEGORIES = (CONVENTION, VERIFIER_FN, GENUINE)
MAX_CANDIDATE_COLUMNS = 7  # column-mapping search is P(n, m): 5040 mappings at most
MAX_MAPPINGS = 5040
FLOAT_DIGITS = (2, 1, 0)  # coarser roundings tried when the verifier's 6 dp rejects a result
DIFF_ROWS = 2
DIFF_CHARS = 100

Row = tuple[Scalar, ...]


def _clip(x: object) -> str:
    s = repr(x)
    return s if len(s) <= DIFF_CHARS else s[: DIFF_CHARS - 3] + "..."


def _round_cell(v: Scalar, digits: int) -> Scalar:
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return round(float(v), digits)
    return v


def _null_to_empty(v: Scalar) -> Scalar:
    return "" if v is None else v


def _same(a: Sequence[Row], b: Sequence[Row]) -> bool:
    return Counter(a) == Counter(b)


def _shape_diff(
    cand: ExecOutcome, gold: ExecOutcome, got: Sequence[Row], want: Sequence[Row]
) -> str:
    extra, missing = Counter(got) - Counter(want), Counter(want) - Counter(got)
    parts = [
        f"cols {list(cand.columns)} vs gold {list(gold.columns)}",
        f"rows {len(got)} vs gold {len(want)}",
    ]
    if extra:
        parts.append("unexpected " + "; ".join(_clip(r) for r in islice(extra, DIFF_ROWS)))
    if missing:
        parts.append("missing " + "; ".join(_clip(r) for r in islice(missing, DIFF_ROWS)))
    return " | ".join(parts)


def _column_rule(got: Sequence[Row], want: Sequence[Row], n_cand: int, n_gold: int) -> str | None:
    """``extra_columns`` / ``column_order`` when SOME selection of the candidate's columns, in
    some order, reproduces the gold rows exactly; else None."""
    if n_cand < n_gold or n_cand > MAX_CANDIDATE_COLUMNS or n_gold == 0:
        return None
    ident = tuple(range(n_gold))
    for perm in islice(permutations(range(n_cand), n_gold), MAX_MAPPINGS):
        if n_cand == n_gold and perm == ident:
            continue  # the unmapped comparison already failed
        if _same([tuple(r[i] for i in perm) for r in got], want):
            return "extra_columns" if n_cand > n_gold else "column_order"
    return None


def classify_miss(
    cand: ExecOutcome | None, gold: ExecOutcome, requires_order: bool
) -> dict[str, str]:
    """``{"category", "rule", "diff"}`` for one miss (a candidate the verifier rejected)."""
    if cand is None:
        return {"category": GENUINE, "rule": "no_sql_extracted", "diff": "no SQL in the output"}
    if not cand.ok:
        return {
            "category": GENUINE,
            "rule": "execution_error",
            "diff": f"{cand.error_kind}: {_clip(cand.error)}",
        }
    if not gold.ok:  # not a model error at all; never hide it behind a class
        return {"category": GENUINE, "rule": "gold_error", "diff": _clip(gold.error)}
    got, want = normalise_rows(cand.rows), normalise_rows(gold.rows)
    n_cand, n_gold = len(cand.columns), len(gold.columns)
    diff = _shape_diff(cand, gold, got, want)

    def out(category: str, rule: str) -> dict[str, str]:
        return {"category": category, "rule": rule, "diff": diff}

    if n_cand == n_gold and _same(got, want):  # same rows, same columns: only order can differ
        return out(VERIFIER_FN, "order_not_required")
    rule = _column_rule(got, want, n_cand, n_gold)
    if rule is not None:
        return out(CONVENTION, rule)
    if n_cand == n_gold:
        for digits in FLOAT_DIGITS:
            g2 = [tuple(_round_cell(c, digits) for c in r) for r in got]
            w2 = [tuple(_round_cell(c, digits) for c in r) for r in want]
            if _same(g2, w2):
                return out(VERIFIER_FN, f"float_rounding_{digits}dp")
        if _same(
            [tuple(_null_to_empty(c) for c in r) for r in got],
            [tuple(_null_to_empty(c) for c in r) for r in want],
        ):
            return out(CONVENTION, "null_vs_empty")
        if set(got) == set(want):  # same distinct rows, different multiplicities
            return out(CONVENTION, "duplicate_rows")
    return out(GENUINE, "wrong_result")


def summarize_misses(entries: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Counts per setup and category, plus the review table (every non-genuine miss, in order)."""
    counts: dict[str, dict[str, int]] = {}
    for e in entries:
        counts.setdefault(str(e["setup"]), {c: 0 for c in CATEGORIES})[str(e["category"])] += 1
    review = [dict(e) for e in entries if e["category"] != GENUINE]
    return {"counts": counts, "review": review}


def render_table(entries: Sequence[Mapping[str, Any]]) -> str:
    """Markdown review table (ids, setup, class, rule, diff); ids and result shapes only, never
    a question or SQL text."""
    lines = ["| set | setup | task_id | class | rule | diff |", "|---|---|---|---|---|---|"]
    for e in entries:
        diff = str(e["diff"]).replace("|", "\\|").replace("\n", " ")
        lines.append(
            f"| {e.get('set', '')} | {e['setup']} | {e['task_id']} | {e['category']} | "
            f"{e['rule']} | {diff} |"
        )
    return "\n".join(lines) + "\n"
