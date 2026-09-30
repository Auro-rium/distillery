"""Pre-flight cost table for a run scale, from a price file and MEASURED per-call token averages.

Read-only: it opens the reference run's ``index.sqlite`` with ``mode=ro`` and reads its
``round1/train.jsonl``; it calls no API and spends nothing. A line without a price (or without a
measured token average) is marked MISSING and the grand total is withheld: only a subtotal of the
priced lines is shown, plus the list of what is missing. Nothing is ever priced at a guess.

Usage: python scripts/estimate_run_cost.py --scale mini --prices deploy/prices.json \
           --root .distillery/live [--run sql-mini-live1]
"""

from __future__ import annotations

import argparse
import math
import sqlite3
import sys
from dataclasses import dataclass
from pathlib import Path

from distillery.config import PriceFile, load_price_file
from distillery.orchestrator import SCALES, Scale

TEACHER = "nvidia/nemotron-3-super-120b-a12b"
PLANNER = "nvidia/Nemotron-3-Ultra-550b-a55b"
STUDENT = "Qwen/Qwen3-0.6B"
MISSING_PRICE = "MISSING PRICE"
REAL_PRICE = "real price"


@dataclass(frozen=True)
class Avg:
    calls: int
    input_tokens: float
    output_tokens: float


@dataclass(frozen=True)
class Line:
    name: str
    quantity: str
    usd: float | None  # None: cannot be priced
    status: str  # REAL_PRICE or "MISSING PRICE: <what>"


def measure_avgs(root: Path, run_id: str) -> dict[str, Avg]:
    """Per-model average input/output tokens per call from the run's ``llm_calls`` (read-only)."""
    con = sqlite3.connect(f"file:{root / 'index.sqlite'}?mode=ro", uri=True)
    try:
        rows = con.execute(
            "SELECT model, COUNT(*), AVG(input_tokens), AVG(output_tokens) FROM llm_calls "
            "WHERE run_id=? GROUP BY model",
            (run_id,),
        ).fetchall()
    finally:
        con.close()
    return {str(m): Avg(int(c), float(i), float(o)) for m, c, i, o in rows}


def avg_row_chars(train_jsonl: Path) -> float | None:
    """Mean characters per training row of a written ``train.jsonl`` (None if absent/empty)."""
    try:
        lines = [ln for ln in train_jsonl.read_text(encoding="utf-8").splitlines() if ln.strip()]
    except OSError:
        return None
    return sum(len(ln) for ln in lines) / len(lines) if lines else None


def _llm_line(name: str, model: str, calls: float, prices: PriceFile, avgs: dict[str, Avg]) -> Line:
    qty = f"{calls:g} calls x {model}"
    price = prices.llm.get(model)
    avg = avgs.get(model)
    if price is None:
        return Line(name, qty, None, f"{MISSING_PRICE}: LLM price for {model}")
    if avg is None:
        return Line(name, qty, None, f"MISSING MEASUREMENT: no llm_calls for {model} in the run")
    usd = calls * (
        avg.input_tokens * price.input_per_mtok + avg.output_tokens * price.output_per_mtok
    )
    return Line(
        name,
        f"{qty} (avg {avg.input_tokens:.0f} in / {avg.output_tokens:.0f} out)",
        usd / 1e6,
        REAL_PRICE,
    )


def build_estimate(
    scale: Scale,
    prices: PriceFile,
    avgs: dict[str, Avg],
    *,
    row_chars: float | None,
    chars_per_token: float = 3.0,
    epochs: int = 3,
    rounds: int = 2,
    retries_factor: float = 1.0,
    stress_n: int = 0,
    analysis_calls: float | None = None,
    sec_per_generation: float = 7.8,
    teacher: str = TEACHER,
    planner: str = PLANNER,
    student: str = STUDENT,
) -> list[Line]:
    lines = [
        _llm_line("teacher: train rows", teacher, scale.train * retries_factor, prices, avgs),
        _llm_line("teacher: held-out/stress eval", teacher, scale.heldout + stress_n, prices, avgs),
    ]
    n_analysis = analysis_calls if analysis_calls is not None else scale.dev * max(rounds - 1, 0)
    lines.append(
        _llm_line("planner: failure analysis (upper bound)", planner, n_analysis, prices, avgs)
    )

    ft_price = prices.finetune.get(student)
    if row_chars is None:
        lines.append(
            Line("fine-tune", "no train.jsonl to measure", None, "MISSING MEASUREMENT: train.jsonl")
        )
    else:
        tokens = math.ceil(scale.train * (row_chars / chars_per_token)) * epochs * rounds
        qty = (
            f"{tokens:,} trained tokens ({scale.train} rows x {row_chars:.0f} chars / "
            f"{chars_per_token:g} x {epochs} epochs x {rounds} rounds)"
        )
        if ft_price is None:
            lines.append(
                Line("fine-tune", qty, None, f"{MISSING_PRICE}: fine-tune price for {student}")
            )
        else:
            lines.append(
                Line(
                    "fine-tune",
                    qty,
                    tokens * ft_price.usd_per_mtok_trained_tokens / 1e6,
                    REAL_PRICE,
                )
            )

    n_gen = (scale.dev + scale.heldout) + (scale.dev * rounds + scale.heldout)  # base + student
    secs = n_gen * sec_per_generation
    qty = f"{n_gen} generations x {sec_per_generation:g} s = {secs:,.0f} s"
    if prices.sandbox is None:
        lines.append(Line("sandbox generation", qty, None, f"{MISSING_PRICE}: sandbox price"))
    else:
        lines.append(Line("sandbox generation", qty, secs * prices.sandbox.per_second, REAL_PRICE))
    return lines


def render(lines: list[Line]) -> str:
    out = [f"{'line':<40} {'usd':>10}  status / quantity"]
    for ln in lines:
        usd = f"{ln.usd:10.4f}" if ln.usd is not None else f"{'-':>10}"
        out.append(f"{ln.name:<40} {usd}  {ln.status}; {ln.quantity}")
    missing = [ln for ln in lines if ln.usd is None]
    subtotal = sum(ln.usd for ln in lines if ln.usd is not None)
    if missing:
        out.append(f"SUBTOTAL of priced lines only: ${subtotal:.4f} (NOT a run total)")
        out.append(
            "TOTAL withheld; missing: " + "; ".join(f"{m.name} ({m.status})" for m in missing)
        )
    else:
        out.append(f"TOTAL: ${subtotal:.4f} (all lines priced from the price file)")
    return "\n".join(out)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scale", required=True, choices=sorted(SCALES))
    ap.add_argument("--prices", required=True, type=Path)
    ap.add_argument("--root", type=Path, default=Path(".distillery/live"))
    ap.add_argument("--run", default="sql-mini-live1")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("--chars-per-token", type=float, default=3.0)
    ap.add_argument("--retries-factor", type=float, default=1.0)
    ap.add_argument("--stress-n", type=int, default=0)
    ap.add_argument("--analysis-calls", type=float, default=None)
    ap.add_argument("--sec-per-generation", type=float, default=7.8)
    ap.add_argument("--teacher-model", default=TEACHER)
    ap.add_argument("--planner-model", default=PLANNER)
    ap.add_argument("--student-model", default=STUDENT)
    a = ap.parse_args(argv)
    prices = load_price_file(a.prices)
    avgs = measure_avgs(a.root, a.run)
    row_chars = avg_row_chars(a.root / "runs" / a.run / "round1" / "train.jsonl")
    lines = build_estimate(
        SCALES[a.scale], prices, avgs, row_chars=row_chars, chars_per_token=a.chars_per_token,
        epochs=a.epochs, rounds=a.rounds, retries_factor=a.retries_factor, stress_n=a.stress_n,
        analysis_calls=a.analysis_calls, sec_per_generation=a.sec_per_generation,
        teacher=a.teacher_model, planner=a.planner_model, student=a.student_model,
    )  # fmt: skip
    print(f"scale={a.scale} run={a.run} prices={a.prices}")  # noqa: T201
    print(render(lines))  # noqa: T201
    return 0


if __name__ == "__main__":
    sys.exit(main())
