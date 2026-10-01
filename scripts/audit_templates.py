"""Offline audit of the SQL question templates (no LLM, no network, spends nothing).

For every template, draw many questions and execute the gold SQL, then flag:
  * gold that errors (an empty result is only counted: the generator drops those draws);
  * ORDER BY ties: the gold is run on the database and on a copy with every table's physical row
    order reversed; an ordered query whose result differs between the two has an ambiguous order
    (several answers are equally "correct"), so exact ordered comparison would be unfair;
  * LIMIT without ORDER BY (which rows survive is arbitrary);
  * a question text whose gold SQL variants return different results (the question does not
    determine its answer, e.g. a hidden filter the wording never mentions).
Prints a JSON report and the sha256 of the sorted template names (the "frozen list" recorded in
DECISIONS.md). Exit status 1 when anything is flagged.

Usage: python scripts/audit_templates.py [--draws 300] [--seed 4242] [--out audit.json]
"""

import argparse
import hashlib
import json
import random
import re
import sqlite3
import sys
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any

from distillery.taskpacks.sql import schema as sql_schema
from distillery.taskpacks.sql.questions import TEMPLATES, has_top_level_order_by
from distillery.taskpacks.sql.runner import run_select

_LIMIT = re.compile(r"\bLIMIT\b", re.IGNORECASE)


def reversed_copy(src: Path, dst: Path) -> None:
    """A copy of the database whose tables hold the same rows in reverse physical order."""
    with sqlite3.connect(src) as a, sqlite3.connect(dst) as b:
        for (name,) in a.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall():
            cols = [r[1] for r in a.execute(f'PRAGMA table_info("{name}")')]
            col_list = ", ".join(f'"{c}"' for c in cols)
            rows = a.execute(f'SELECT {col_list} FROM "{name}" ORDER BY rowid DESC').fetchall()
            b.execute(f'CREATE TABLE "{name}" ({col_list})')
            b.executemany(f'INSERT INTO "{name}" VALUES ({",".join("?" * len(cols))})', rows)
        b.commit()


def audit(draws: int, seed: int, db_seed: int = 0) -> dict[str, Any]:
    tmp = Path(tempfile.mkdtemp(prefix="audit_"))
    db, rev = tmp / "db.sqlite", tmp / "rev.sqlite"
    sql_schema.write_database(db_seed, db)
    reversed_copy(db, rev)
    per: dict[str, dict[str, Any]] = {}
    question_to_sql: dict[str, set[str]] = defaultdict(set)
    for tpl in TEMPLATES:
        rng = random.Random(f"{seed}|{tpl.name}")
        rec = {
            "family": tpl.family,
            "draws": 0,
            "distinct_sql": 0,
            "errors": 0,
            "empty": 0,
            "order_ties": 0,
            "limit_without_order": 0,
        }
        seen: set[str] = set()
        for _ in range(draws):
            q, sql = tpl.build(rng)
            rec["draws"] += 1
            question_to_sql[q].add(sql)
            if sql in seen:
                continue
            seen.add(sql)
            out = run_select(str(db), sql)
            if not out.ok:
                rec["errors"] += 1
                continue
            if not out.rows:
                rec["empty"] += 1
                continue
            ordered = has_top_level_order_by(sql)
            if _LIMIT.search(sql) and not ordered:
                rec["limit_without_order"] += 1
            other = run_select(str(rev), sql)
            if other.ok and (
                (ordered and other.rows != out.rows)
                or (not ordered and sorted(map(repr, other.rows)) != sorted(map(repr, out.rows)))
            ):
                rec["order_ties"] += 1
        rec["distinct_sql"] = len(seen)
        per[tpl.name] = rec
    # a question is ambiguous only if its gold SQL strings give DIFFERENT results (two equivalent
    # formulations of one answer, e.g. LEFT JOIN vs NOT EXISTS, are fine under execution matching)
    ambiguous = []
    for q, sqls in sorted(question_to_sql.items()):
        if len(sqls) < 2:
            continue
        results = set()
        for sql in sqls:
            o = run_select(str(db), sql)
            rows = tuple(o.rows if has_top_level_order_by(sql) else sorted(map(repr, o.rows)))
            results.add(repr(rows))
        if len(results) > 1:
            ambiguous.append(q)
    flagged = {
        name: r
        for name, r in per.items()
        if r["errors"] or r["order_ties"] or r["limit_without_order"]
    }
    names = sorted(per)
    return {
        "templates": len(per),
        "frozen_list_sha256": hashlib.sha256("\n".join(names).encode()).hexdigest(),
        "draws_per_template": draws,
        "seed": seed,
        "flagged_templates": flagged,
        "questions_with_multiple_gold_sql": ambiguous[:20],
        "n_questions_with_multiple_gold_sql": len(ambiguous),
        "per_template": per,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--draws", type=int, default=300)
    ap.add_argument("--seed", type=int, default=4242)
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args(argv)
    rep = audit(a.draws, a.seed)
    text = json.dumps(rep, indent=2, sort_keys=True)
    if a.out:
        a.out.write_text(text)
    summary = {k: v for k, v in rep.items() if k != "per_template"}
    print(json.dumps(summary, indent=2, sort_keys=True))  # noqa: T201
    return 1 if rep["flagged_templates"] or rep["n_questions_with_multiple_gold_sql"] else 0


if __name__ == "__main__":
    sys.exit(main())
