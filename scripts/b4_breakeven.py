#!/usr/bin/env python3
"""B4: teacher latency and cost, student hosting cost, and the break-even daily volume.

Reads a store READ-ONLY (``index.sqlite`` with ``mode=ro``); calls nothing, spends nothing.

* Teacher p50/p95 come from ``llm_calls.latency_s`` (plan A2: only calls recorded after the column
  existed have one; the P4 store has none, so run the sample first). ``--limit N`` keeps the most
  recent N such calls (the plan's 100-question sample at production concurrency).
* Cost per 1k questions = mean ``usd`` per call x 1000 from the ledger rows (one call = one question
  in the eval path; retries inside a call are one row).
* Break-even: a student host costing H USD/hour beats the teacher above V* = 24*H / c_teacher_per_q
  questions per day. Capacity check: the student must be able to serve V*, i.e. V* <= 24 x
  throughput (questions/hour, ``--student-throughput-per-hour``); otherwise the break-even is
  unreachable on one host and the output says so.
* The student host label is explicit: ``sandbox-cpu`` (default; "sandbox CPU, not Dedicated") or
  ``dedicated`` (only with a confirmed ``--hourly-usd``; see evidence/b4_dedicated_research.md).

  python scripts/b4_breakeven.py --root OUT/store --run-id RUN \
      --model nvidia/nemotron-3-super-120b-a12b --limit 100 --hourly-usd 3.85 \
      --student-label dedicated --student-throughput-per-hour 20000
"""

from __future__ import annotations

import argparse
import json
import math
import sqlite3
from collections.abc import Sequence
from pathlib import Path
from typing import Any

SANDBOX_LABEL = "sandbox CPU, not Dedicated"


def percentile(values: Sequence[float], q: float) -> float:
    """Nearest-rank percentile (q in [0, 100]) of a non-empty sample."""
    if not values:
        raise ValueError("no values")
    s = sorted(values)
    return s[max(0, math.ceil(q / 100 * len(s)) - 1)]


def teacher_stats(
    root: Path, model: str, run_id: str | None = None, limit: int | None = None
) -> dict[str, Any]:
    con = sqlite3.connect(f"file:{root / 'index.sqlite'}?mode=ro", uri=True)
    try:
        cols = {r[1] for r in con.execute("PRAGMA table_info(llm_calls)")}
        if "latency_s" not in cols:
            raise SystemExit("llm_calls has no latency_s column: this store predates plan A2")
        sql = "SELECT latency_s, usd, input_tokens, output_tokens FROM llm_calls WHERE model=?"
        args: list[Any] = [model]
        if run_id:
            sql += " AND run_id=?"
            args.append(run_id)
        sql += " AND latency_s IS NOT NULL ORDER BY id DESC"
        if limit:
            sql += " LIMIT ?"
            args.append(limit)
        rows = con.execute(sql, args).fetchall()
    finally:
        con.close()
    if not rows:
        raise SystemExit(f"no calls with a recorded latency for model {model!r}")
    lat = [float(r[0]) for r in rows]
    usd = sum(float(r[1]) for r in rows)
    n = len(rows)
    return {
        "model": model,
        "n_calls": n,
        "latency_p50_s": percentile(lat, 50),
        "latency_p95_s": percentile(lat, 95),
        "latency_mean_s": sum(lat) / n,
        "usd_per_question": usd / n,
        "usd_per_1k_questions": usd / n * 1000,
        "mean_input_tokens": sum(r[2] for r in rows) / n,
        "mean_output_tokens": sum(r[3] for r in rows) / n,
        "basis": "ledger usd (measured tokens x configured price) / calls with latency_s",
    }


def breakeven(
    teacher_usd_per_q: float,
    hourly_usd: float,
    throughput_per_hour: float | None,
    label: str = "sandbox-cpu",
) -> dict[str, Any]:
    """V* = 24*H / c_teacher_per_q (questions/day); capacity check V* <= 24 * throughput."""
    if teacher_usd_per_q <= 0 or hourly_usd < 0:
        raise ValueError("teacher cost per question must be > 0 and the hourly price >= 0")
    v_star = 24 * hourly_usd / teacher_usd_per_q
    out: dict[str, Any] = {
        "student_host": SANDBOX_LABEL if label == "sandbox-cpu" else "Dedicated endpoint",
        "hourly_usd": hourly_usd,
        "teacher_usd_per_question": teacher_usd_per_q,
        "breakeven_questions_per_day": v_star,
        "formula": "V* = 24 * H / c_teacher_per_q",
        "daily_student_cost_usd": 24 * hourly_usd,
    }
    if throughput_per_hour is None:
        out["capacity_check"] = {"status": "unknown", "reason": "no --student-throughput-per-hour"}
    else:
        cap = 24 * throughput_per_hour
        out["capacity_check"] = {
            "status": "ok" if v_star <= cap else "break-even unreachable on one host",
            "daily_capacity_questions": cap,
            "rule": "V* <= 24 * throughput_per_hour",
        }
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--root", required=True, help="store root holding index.sqlite (read-only)")
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--model", required=True, help="teacher model id as recorded in llm_calls")
    ap.add_argument("--limit", type=int, default=None, help="most recent N calls with latency")
    ap.add_argument("--hourly-usd", type=float, required=True, help="student host price per hour")
    ap.add_argument("--student-label", choices=("sandbox-cpu", "dedicated"), default="sandbox-cpu")
    ap.add_argument("--student-throughput-per-hour", type=float, default=None)
    ap.add_argument("--out", default=None, help="write the JSON here as well")
    a = ap.parse_args(argv)
    if a.student_label == "dedicated" and a.hourly_usd <= 0:
        ap.error("--student-label dedicated needs a confirmed --hourly-usd > 0")
    t = teacher_stats(Path(a.root), a.model, a.run_id, a.limit)
    res = {
        "teacher": t,
        "breakeven": breakeven(
            t["usd_per_question"], a.hourly_usd, a.student_throughput_per_hour, a.student_label
        ),
    }
    text = json.dumps(res, indent=1)
    if a.out:
        Path(a.out).write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
