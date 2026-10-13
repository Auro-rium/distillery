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
from pathlib import Path

from distillery.costing import SANDBOX_LABEL, breakeven, percentile, teacher_stats

__all__ = ["SANDBOX_LABEL", "breakeven", "main", "percentile", "teacher_stats"]


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
