"""Recompute a finished run's cost from its STORED MEASUREMENTS at the prices of a price file.

Measurements used (all recorded during the run, none re-measured): LLM tokens per model from
``llm_calls``, trained tokens per fine-tune job from ``report.finetune``, sandbox seconds from the
``spend`` rows. The result goes under a NEW report key ``cost_repriced``; the original ``cost``
block is never touched. A line whose price or measurement is missing is listed under ``missing``
and the total is withheld: nothing is priced at a guess.
"""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from distillery.budget import KIND_SANDBOX, KIND_SANDBOX_UNPRICED, spend_lines
from distillery.config import PriceFile
from distillery.orchestrator import cost_by_model
from distillery.store import Store, atomic_write_bytes


class RepriceError(RuntimeError):
    """The run cannot be repriced (no report, or the report is unreadable)."""


def _sandbox_secs_by_purpose(store: Store, run_id: str) -> dict[str, float]:
    out: dict[str, float] = defaultdict(float)
    for ln in spend_lines(store, run_id):
        if ln["kind"] in (KIND_SANDBOX, KIND_SANDBOX_UNPRICED):
            out[str(ln["model"])] += float(ln["units"])
    return dict(out)


def compute_repriced(
    store: Store,
    run_id: str,
    report: Mapping[str, Any],
    prices: PriceFile,
    *,
    prices_label: str,
    balance_before: float | None = None,
    balance_after: float | None = None,
    now: str | None = None,
) -> dict[str, Any]:
    lines: list[dict[str, Any]] = []
    missing: list[str] = []

    for model, c in cost_by_model(store, run_id).items():
        price = prices.llm.get(model)
        line: dict[str, Any] = {
            "kind": "llm",
            "model": model,
            "calls": c["calls"],
            "input_tokens": c["input_tokens"],
            "output_tokens": c["output_tokens"],
        }
        if price is None:
            line.update(usd=None, status=f"MISSING PRICE: LLM price for {model}")
            missing.append(f"llm {model}")
        else:
            line.update(
                usd=(
                    c["input_tokens"] * price.input_per_mtok
                    + c["output_tokens"] * price.output_per_mtok
                )
                / 1e6,
                price={"source": price.source, "as_of": price.date},
            )
        lines.append(line)

    for ft in report.get("finetune") or []:
        base = str(ft.get("base_model") or "")
        tokens = ft.get("trained_tokens")
        line = {
            "kind": "finetune",
            "round": ft.get("round"),
            "job_id": ft.get("job_id"),
            "model": base,
            "trained_tokens": tokens,
        }
        fp = prices.finetune.get(base)
        if tokens is None:
            line.update(usd=None, status="MISSING MEASUREMENT: job reported no trained_tokens")
            missing.append(f"finetune round {ft.get('round')} (no trained_tokens)")
        elif fp is None:
            line.update(usd=None, status=f"MISSING PRICE: fine-tune price for {base}")
            missing.append(f"finetune {base}")
        else:
            line.update(
                usd=int(tokens) * fp.usd_per_mtok_trained_tokens / 1e6,
                price={"source": fp.source, "as_of": fp.as_of},
            )
        lines.append(line)

    secs_by_purpose = _sandbox_secs_by_purpose(store, run_id)
    total_secs = sum(secs_by_purpose.values())
    if secs_by_purpose:
        sline: dict[str, Any] = {
            "kind": "sandbox",
            "sandbox_seconds": total_secs,
            "seconds_by_purpose": secs_by_purpose,
        }
        if prices.sandbox is None:
            sline.update(usd=None, status="MISSING PRICE: sandbox price")
            missing.append("sandbox")
        else:
            sline.update(
                usd=total_secs * prices.sandbox.per_second,
                price={"source": prices.sandbox.source, "as_of": prices.sandbox.as_of},
            )
        lines.append(sline)

    subtotal = sum(float(x["usd"]) for x in lines if x["usd"] is not None)
    total = None if missing else subtotal
    out: dict[str, Any] = {
        "basis": "RECOMPUTED from stored measurements (llm_calls tokens, report.finetune "
        "trained_tokens, sandbox seconds from spend rows) at the prices of the named price file; "
        "the original `cost` block is unchanged",
        "run_id": run_id,
        "prices_file": prices_label,
        "computed_at": now or datetime.now(UTC).isoformat(),
        "lines": lines,
        "subtotal_usd": subtotal,
        "total_usd": total,
        "missing": missing,
        "total_note": (
            "all lines priced"
            if total is not None
            else "TOTAL withheld; subtotal covers priced lines only and is NOT a run total"
        ),
    }
    if balance_before is not None and balance_after is not None:
        delta = balance_before - balance_after
        out["reconciliation"] = _reconcile(delta, total, subtotal, balance_before, balance_after)
    return out


def _reconcile(
    delta: float, total: float | None, subtotal: float, before: float, after: float
) -> dict[str, Any]:
    ref = total if total is not None else subtotal
    label = "repriced total" if total is not None else "repriced SUBTOTAL (priced lines only)"
    diff = delta - ref
    return {
        "balance_before": before,
        "balance_after": after,
        "balance_delta_usd": delta,
        "compared_with": label,
        "compared_usd": ref,
        "unexplained_usd": diff,
        "line": f"console balance fell ${delta:.4f}; {label} is ${ref:.4f}; "
        f"difference ${diff:+.4f} (console balances are coarse and other spend may be included)",
    }


def reprice_run(
    store: Store,
    run_id: str,
    prices: PriceFile,
    *,
    prices_label: str,
    balance_before: float | None = None,
    balance_after: float | None = None,
    write: bool = True,
) -> dict[str, Any]:
    """Compute ``cost_repriced`` and (by default) add it to the run's report.json."""
    path: Path = store.run_dir(run_id) / "report.json"
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as e:
        raise RepriceError(f"cannot read report for run {run_id}: {e}") from e
    if (balance_before is None) != (balance_after is None):
        raise RepriceError("give both --balance-before and --balance-after, or neither")
    repriced = compute_repriced(
        store,
        run_id,
        report,
        prices,
        prices_label=prices_label,
        balance_before=balance_before,
        balance_after=balance_after,
    )
    if write:
        report["cost_repriced"] = repriced  # `cost` is left exactly as it was
        data = json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
        atomic_write_bytes(path, data.encode("utf-8"))
    return repriced


def render(rp: Mapping[str, Any]) -> list[str]:
    out = [f"repriced cost of {rp['run_id']} at {rp['prices_file']} ({rp['basis']})"]
    for ln in rp["lines"]:
        usd = "-" if ln["usd"] is None else f"${ln['usd']:.4f}"
        what = ln.get("model") or ln["kind"]
        note = ln.get("status") or ln["price"]["source"] + f" (as of {ln['price']['as_of']})"
        out.append(f"  {ln['kind']:<9} {what}: {usd}  [{note}]")
    total = rp["total_usd"]
    out.append(
        f"TOTAL ${total:.4f}"
        if total is not None
        else f"TOTAL withheld; priced subtotal ${rp['subtotal_usd']:.4f}; "
        f"missing: {'; '.join(rp['missing'])}"
    )
    if "reconciliation" in rp:
        out.append(rp["reconciliation"]["line"])
    return out
