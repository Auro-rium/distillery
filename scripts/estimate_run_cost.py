"""Pre-flight cost table for a run scale, from a price file and MEASURED per-call token averages.

Read-only: it opens the reference run's ``index.sqlite`` with ``mode=ro`` and reads its
``round1/train.jsonl``; it calls no API and spends nothing. A line without a price (or without a
measured token average) is marked MISSING and the grand total is withheld: only a subtotal of the
priced lines is shown, plus the list of what is missing. Nothing is ever priced at a guess.

``--scale gated`` prices the claims-real design (STATUS.md "Claims-real plan"): teacher originals,
Nano paraphrase calls, teacher verification of paraphrases (up to 3 per original), human-set
drafting (k=3 teacher calls per question), teacher eval on gate+stress+human, fine-tune tokens from
the FINAL row count, sandbox generations and a wall-clock estimate from the measured c20 throughput
of Qwen3-1.7B. Nano token averages for the paraphrase task were never measured (the mini run's Nano
calls were triage), so that line is MISSING MEASUREMENT unless you pass --nano-in-tokens and
--nano-out-tokens (then it says USER-SUPPLIED). ``--finetune-ceiling-usd`` adds an informational
ASSUMED ceiling line when no console fine-tune price exists; it is never a price and never enters
a total.

Usage: python scripts/estimate_run_cost.py --scale mini --prices deploy/prices.json \
           --root .distillery/live [--run sql-mini-live1]
       python scripts/estimate_run_cost.py --scale gated --prices .distillery/prices.json \
           --human-n 100 [--final-rows 1800] [--finetune-ceiling-usd 8]
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
STUDENT_1_7B = "Qwen/Qwen3-1.7B"
NANO = "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B"
# Measured (docs/spikes/s5-qwen3-1_7b.json, concurrency_20 phase, 25 samples, batch 1):
# 0.2197 samples/s wall-clock; per_sample_s 25.49 + load_s 2.28 per job = ~27.8 sandbox-seconds.
MEASURED_1_7B_SAMPLES_PER_S = 0.22
MEASURED_1_7B_SEC_PER_SAMPLE = 27.8
# Measured: 117 of 120 train tasks became rows in round 1 of sql-mini-live1 (teacher == gold).
MEASURED_ORIGINAL_YIELD = 0.975
MISSING_PRICE = "MISSING PRICE"
ASSUMED_PRICE = "ASSUMED ceiling price (not a console price)"


def price_status(source: str) -> str:
    """Say where a price came from; anything marked ASSUMED is never presented as a listed price."""
    return ASSUMED_PRICE if "ASSUMED" in source.upper() else f"listed price: {source}"


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
    informational: bool = False  # an ASSUMED ceiling: shown, never part of a subtotal or total


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
        price_status(price.source),
    )


def build_estimate(
    scale: Scale,
    prices: PriceFile,
    avgs: dict[str, Avg],
    *,
    row_chars: float | None,
    chars_per_token: float = 3.0,
    tokens_per_row: float | None = None,
    epochs: int = 3,
    rounds: int = 2,
    retries_factor: float = 1.0,
    stress_n: int | None = None,
    analysis_calls: float | None = None,
    sec_per_generation: float = 7.8,
    teacher: str = TEACHER,
    planner: str = PLANNER,
    student: str = STUDENT,
) -> list[Line]:
    stress_n = scale.stress if stress_n is None else stress_n
    lines = [
        _llm_line("teacher: train rows", teacher, scale.train * retries_factor, prices, avgs),
        _llm_line("teacher: held-out/stress eval", teacher, scale.heldout + stress_n, prices, avgs),
    ]
    n_analysis = analysis_calls if analysis_calls is not None else scale.dev * max(rounds - 1, 0)
    lines.append(
        _llm_line("planner: failure analysis (upper bound)", planner, n_analysis, prices, avgs)
    )

    ft_price = prices.finetune.get(student)
    if tokens_per_row is not None:
        tokens = math.ceil(scale.train * tokens_per_row) * epochs * rounds
        qty = (
            f"{tokens:,} trained tokens ({scale.train} rows x {tokens_per_row:g} tokens/row, "
            f"MEASURED by the 3 real jobs, x {epochs} epochs x {rounds} rounds)"
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
                    price_status(ft_price.source),
                )
            )
    elif row_chars is None:
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
                    price_status(ft_price.source),
                )
            )

    # base: dev + gate + stress once; student: dev every round + gate + stress
    n_gen = (scale.dev + scale.heldout + stress_n) + (scale.dev * rounds + scale.heldout + stress_n)
    secs = n_gen * sec_per_generation
    qty = f"{n_gen} generations x {sec_per_generation:g} s = {secs:,.0f} s"
    if prices.sandbox is None:
        lines.append(Line("sandbox generation", qty, None, f"{MISSING_PRICE}: sandbox price"))
    else:
        lines.append(
            Line(
                "sandbox generation",
                qty,
                secs * prices.sandbox.per_second,
                price_status(prices.sandbox.source),
            )
        )
    return lines


def build_gated_estimate(
    scale: Scale,
    prices: PriceFile,
    avgs: dict[str, Avg],
    *,
    human_n: int,
    final_rows: int = 1800,
    tokens_per_row: float = 960.0,
    epochs: int = 3,
    rounds: int = 1,
    original_yield: float = MEASURED_ORIGINAL_YIELD,
    paraphrases_per_original: int = 3,
    drafting_k: int = 3,
    retries_factor: float = 1.0,
    nano_avg: Avg | None = None,
    sec_per_sample: float = MEASURED_1_7B_SEC_PER_SAMPLE,
    finetune_ceiling_usd: float | None = None,
    teacher: str = TEACHER,
    planner: str = PLANNER,
    nano: str = NANO,
    student: str = STUDENT_1_7B,
) -> list[Line]:
    """The claims-real design: see the module docstring. Nothing here is priced at a guess."""
    verified = scale.train * original_yield
    gate_stress_human = scale.heldout + scale.stress + human_n
    lines = [
        _llm_line("teacher: train originals", teacher, scale.train * retries_factor, prices, avgs),
    ]
    nano_calls = verified
    if nano_avg is not None:
        ln = _llm_line("nano: paraphrase calls", nano, nano_calls, prices, {nano: nano_avg})
        lines.append(
            Line(ln.name, ln.quantity, ln.usd, ln.status + "; tokens USER-SUPPLIED, not measured")
            if ln.usd is not None
            else ln
        )
    else:
        lines.append(
            Line(
                "nano: paraphrase calls",
                f"{nano_calls:g} calls x {nano}",
                None,
                "MISSING MEASUREMENT: no paraphrase-task token averages (the mini run's Nano calls "
                "were triage); pass --nano-in-tokens/--nano-out-tokens to supply them",
            )
        )
    lines += [
        _llm_line(
            "teacher: verify paraphrases (upper bound)",
            teacher,
            verified * paraphrases_per_original,
            prices,
            avgs,
        ),
        _llm_line(
            f"teacher: draft human set (k={drafting_k})",
            teacher,
            human_n * drafting_k,
            prices,
            avgs,
        ),
        _llm_line("teacher: eval gate+stress+human", teacher, gate_stress_human, prices, avgs),
        _llm_line(
            "planner: failure analysis (upper bound)",
            planner,
            scale.dev * max(rounds - 1, 0),
            prices,
            avgs,
        ),
    ]

    tokens = math.ceil(final_rows * tokens_per_row) * epochs * rounds
    qty = (
        f"{tokens:,} trained tokens ({final_rows} FINAL rows x {tokens_per_row:g} tokens/row "
        f"MEASURED by the 3 real jobs, x {epochs} epochs x {rounds} round)"
    )
    ft_price = prices.finetune.get(student)
    if ft_price is None:
        lines.append(
            Line("fine-tune", qty, None, f"{MISSING_PRICE}: fine-tune price for {student}")
        )
        if finetune_ceiling_usd is not None:
            lines.append(
                Line(
                    "fine-tune: ASSUMED ceiling",
                    f"operator-supplied ${finetune_ceiling_usd:.2f}, no console price exists",
                    finetune_ceiling_usd,
                    "ASSUMED ceiling, NOT a price and NOT in any total",
                    informational=True,
                )
            )
    else:
        lines.append(
            Line(
                "fine-tune",
                qty,
                tokens * ft_price.usd_per_mtok_trained_tokens / 1e6,
                price_status(ft_price.source),
            )
        )

    # headroom (base on dev) + student on dev + base AND student on gate+stress+human
    n_gen = scale.dev + scale.dev * rounds + 2 * gate_stress_human
    secs = n_gen * sec_per_sample
    qty = f"{n_gen} generations x {sec_per_sample:g} sandbox-s = {secs:,.0f} s"
    if prices.sandbox is None:
        lines.append(Line("sandbox generation", qty, None, f"{MISSING_PRICE}: sandbox price"))
    else:
        lines.append(
            Line(
                "sandbox generation",
                qty,
                secs * prices.sandbox.per_second,
                price_status(prices.sandbox.source),
            )
        )
    return lines


def gated_generations(scale: Scale, human_n: int, rounds: int = 1) -> int:
    return scale.dev + scale.dev * rounds + 2 * (scale.heldout + scale.stress + human_n)


def wall_clock_note(n_gen: int, samples_per_s: float) -> str:
    secs = n_gen / samples_per_s
    return (
        f"sandbox generation wall-clock: about {secs / 3600:.2f} h ({secs:,.0f} s) for {n_gen} "
        f"samples at {samples_per_s:g} samples/s (measured concurrency-20 throughput of "
        f"Qwen3-1.7B); excludes image build (~87 s), LLM call time and the fine-tune job"
    )


def render(lines: list[Line]) -> str:
    out = [f"{'line':<40} {'usd':>10}  status / quantity"]
    for ln in lines:
        usd = f"{ln.usd:10.4f}" if ln.usd is not None else f"{'-':>10}"
        out.append(f"{ln.name:<40} {usd}  {ln.status}; {ln.quantity}")
    real = [ln for ln in lines if not ln.informational]
    ceilings = [ln for ln in lines if ln.informational and ln.usd is not None]
    missing = [ln for ln in real if ln.usd is None]
    subtotal = sum(ln.usd for ln in real if ln.usd is not None)
    assumed = [ln.name for ln in real if ln.status.startswith("ASSUMED")]
    if missing:
        out.append(
            f"SUBTOTAL of priced lines only: ${subtotal:.4f} (NOT a run total)"
            + (f"; at ASSUMED ceiling prices: {', '.join(assumed)}" if assumed else "")
        )
        out.append(
            "TOTAL withheld; missing: " + "; ".join(f"{m.name} ({m.status})" for m in missing)
        )
    else:
        note = (
            f"; lines at ASSUMED ceiling prices, not console prices: {', '.join(assumed)}"
            if assumed
            else " (all lines at console prices)"
        )
        out.append(f"TOTAL: ${subtotal:.4f}{note}")
    for ln in ceilings:
        out.append(
            f"ASSUMED CEILING (not a price, not a quote): {ln.name} ${ln.usd:.2f}; priced subtotal "
            f"+ this ceiling = ${subtotal + float(ln.usd or 0):.4f} (an upper-bound guess, "
            "NOT a run total)"
        )
    return "\n".join(out)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scale", required=True, choices=sorted(SCALES))
    ap.add_argument("--prices", required=True, type=Path)
    ap.add_argument("--root", type=Path, default=Path(".distillery/live"))
    ap.add_argument("--run", default="sql-mini-live1")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--rounds", type=int, default=None, help="default: 1 for gated, else 2")
    ap.add_argument("--chars-per-token", type=float, default=3.0)
    ap.add_argument(
        "--tokens-per-row",
        type=float,
        default=960.0,
        help="measured: 963/960/959 trained tokens per row in the three real jobs; 0 = estimate "
        "from characters instead",
    )
    ap.add_argument("--retries-factor", type=float, default=1.0)
    ap.add_argument("--stress-n", type=int, default=None, help="default: the scale's stress size")
    ap.add_argument("--analysis-calls", type=float, default=None)
    ap.add_argument("--sec-per-generation", type=float, default=7.8)
    ap.add_argument("--teacher-model", default=TEACHER)
    ap.add_argument("--planner-model", default=PLANNER)
    ap.add_argument(
        "--student-model", default=None, help="default: Qwen3-1.7B for gated, else 0.6B"
    )
    ap.add_argument("--human-n", type=int, default=100, help="gated: human-set questions")
    ap.add_argument("--final-rows", type=int, default=1800, help="gated: rows actually trained on")
    ap.add_argument("--original-yield", type=float, default=MEASURED_ORIGINAL_YIELD)
    ap.add_argument("--nano-in-tokens", type=float, default=None)
    ap.add_argument("--nano-out-tokens", type=float, default=None)
    ap.add_argument(
        "--sec-per-sample", type=float, default=MEASURED_1_7B_SEC_PER_SAMPLE,
        help="gated: BILLED sandbox seconds per generated sample (default: 1.7B, c20 measured)",
    )  # fmt: skip
    ap.add_argument(
        "--samples-per-s", type=float, default=MEASURED_1_7B_SAMPLES_PER_S,
        help="gated: wall-clock throughput for the time estimate (default: 1.7B, c20 measured)",
    )  # fmt: skip
    ap.add_argument(
        "--finetune-ceiling-usd", type=float, default=None,
        help="show an ASSUMED ceiling line when no console fine-tune price exists",
    )  # fmt: skip
    a = ap.parse_args(argv)
    prices = load_price_file(a.prices)
    avgs = measure_avgs(a.root, a.run)
    if a.scale == "gated":
        nano_avg = (
            Avg(0, a.nano_in_tokens, a.nano_out_tokens)
            if a.nano_in_tokens is not None and a.nano_out_tokens is not None
            else None
        )
        glines = build_gated_estimate(
            SCALES["gated"], prices, avgs, human_n=a.human_n, final_rows=a.final_rows,
            tokens_per_row=a.tokens_per_row or 960.0, epochs=a.epochs, rounds=a.rounds or 1,
            original_yield=a.original_yield, retries_factor=a.retries_factor, nano_avg=nano_avg,
            sec_per_sample=a.sec_per_sample, finetune_ceiling_usd=a.finetune_ceiling_usd,
            teacher=a.teacher_model, planner=a.planner_model,
            student=a.student_model or STUDENT_1_7B,
        )  # fmt: skip
        print(f"scale=gated run={a.run} prices={a.prices} human_n={a.human_n}")  # noqa: T201
        print(render(glines))  # noqa: T201
        print(wall_clock_note(gated_generations(SCALES["gated"], a.human_n), a.samples_per_s))  # noqa: T201
        return 0
    row_chars = avg_row_chars(a.root / "runs" / a.run / "round1" / "train.jsonl")
    lines = build_estimate(
        SCALES[a.scale], prices, avgs, row_chars=row_chars, chars_per_token=a.chars_per_token,
        tokens_per_row=a.tokens_per_row or None,
        epochs=a.epochs, rounds=a.rounds or 2, retries_factor=a.retries_factor, stress_n=a.stress_n,
        analysis_calls=a.analysis_calls, sec_per_generation=a.sec_per_generation,
        teacher=a.teacher_model, planner=a.planner_model,
        student=a.student_model or STUDENT,
    )  # fmt: skip
    print(f"scale={a.scale} run={a.run} prices={a.prices}")  # noqa: T201
    print(render(lines))  # noqa: T201
    return 0


if __name__ == "__main__":
    sys.exit(main())
