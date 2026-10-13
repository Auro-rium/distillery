"""Teacher cost/latency statistics and the break-even formula (shared by B4 and H1).

Factored out of ``scripts/b4_breakeven.py`` unchanged so ``h1_measure`` can reuse it. The
statistics read a store READ-ONLY and call nothing.
"""

from __future__ import annotations

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
