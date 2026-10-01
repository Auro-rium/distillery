"""Recompute past spend at the real prices and show old vs new, per run. Report-only by default.

Old = everything the ledger recorded for the run (``spend`` table, incl. earlier corrections).
New = LLM tokens from ``llm_calls`` x the price file's per-model price, + fine-tunes at
``trained_tokens x price`` where BOTH are known (else the recorded fine-tune rows stay as they
were, labelled ceiling), + recorded sandbox rows unchanged.
A run whose LLM model has no price is reported as not rebasable: nothing is guessed.

``--apply`` APPENDS one ``rebase`` row per run with the difference (new - old); it never edits or
deletes an existing row, and re-running it is a no-op once the deltas are zero.

Usage: python scripts/rebase_ledger.py --root .distillery/live --prices deploy/prices.json \
           [--trained-tokens tokens.json] [--apply]
       tokens.json: {"<run_id>": [115611, 336855], ...}  (one entry per fine-tune job)
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from dataclasses import dataclass
from pathlib import Path

from distillery.budget import (
    KIND_FINETUNE,
    KIND_FINETUNE_CEILING,
    KIND_SANDBOX,
    KIND_SANDBOX_UNPRICED,
    PLAYGROUND,
)
from distillery.config import PriceFile, load_price_file
from distillery.store import Store

REBASE_KIND = "rebase"
_FT_KINDS = (KIND_FINETUNE, KIND_FINETUNE_CEILING)
_SB_KINDS = (KIND_SANDBOX, KIND_SANDBOX_UNPRICED)


@dataclass(frozen=True)
class RunDelta:
    run_id: str
    old_usd: float
    new_usd: float | None  # None: cannot be rebased
    note: str

    @property
    def delta(self) -> float | None:
        return None if self.new_usd is None else self.new_usd - self.old_usd


def compute(
    root: Path,
    prices: PriceFile,
    trained_tokens: dict[str, list[int]] | None = None,
    finetune_model: str = "Qwen/Qwen3-0.6B",
) -> list[RunDelta]:
    tokens = trained_tokens or {}
    con = sqlite3.connect(f"file:{root / 'index.sqlite'}?mode=ro", uri=True)
    try:
        run_ids = [
            str(r[0]) for r in con.execute("SELECT run_id FROM runs ORDER BY created_at, run_id")
        ]
        out: list[RunDelta] = []
        for rid in run_ids:
            if rid == PLAYGROUND:
                continue
            old = float(
                con.execute(
                    "SELECT COALESCE(SUM(usd),0) FROM spend WHERE run_id=?", (rid,)
                ).fetchone()[0]
            )
            llm = con.execute(
                "SELECT model, SUM(input_tokens), SUM(output_tokens) FROM llm_calls "
                "WHERE run_id=? GROUP BY model",
                (rid,),
            ).fetchall()
            missing = sorted(str(m) for m, _, _ in llm if str(m) not in prices.llm)
            if missing:
                out.append(RunDelta(rid, old, None, "no price for " + ", ".join(missing)))
                continue
            new = sum(
                (
                    int(i) * prices.llm[str(m)].input_per_mtok
                    + int(o) * prices.llm[str(m)].output_per_mtok
                )
                / 1e6
                for m, i, o in llm
            )
            notes: list[str] = []
            ft_rows = con.execute(
                "SELECT COALESCE(SUM(usd),0), COUNT(*) FROM spend WHERE run_id=? AND kind IN (?,?)",
                (rid, *_FT_KINDS),
            ).fetchone()
            ft_price = prices.finetune.get(finetune_model)
            if ft_rows[1] and ft_price is not None and rid in tokens:
                new += sum(tokens[rid]) * ft_price.usd_per_mtok_trained_tokens / 1e6
                notes.append(f"fine-tune at {sum(tokens[rid]):,} trained tokens x price")
            elif ft_rows[1]:
                new += float(ft_rows[0])
                notes.append("fine-tune left as recorded (no price or no trained_tokens)")
            sb = con.execute(
                "SELECT COALESCE(SUM(usd),0) FROM spend WHERE run_id=? AND kind IN (?,?)",
                (rid, *_SB_KINDS),
            ).fetchone()[0]
            new += float(sb)
            out.append(RunDelta(rid, old, new, "; ".join(notes)))
        return out
    finally:
        con.close()


def apply(root: Path, deltas: list[RunDelta], eps: float = 1e-9) -> int:
    """Append one corrective ``rebase`` spend row per run with a non-zero delta."""
    store = Store(root)
    n = 0
    try:
        for d in deltas:
            if d.delta is None or abs(d.delta) <= eps:
                continue
            store.record_spend(d.run_id, REBASE_KIND, None, d.delta)
            n += 1
    finally:
        store.close()
    return n


def render(deltas: list[RunDelta]) -> str:
    out = [f"{'run':<28} {'old usd':>12} {'new usd':>12} {'delta':>12}  note"]
    for d in deltas:
        if d.new_usd is None:
            out.append(
                f"{d.run_id:<28} {d.old_usd:12.6f} {'-':>12} {'-':>12}  NOT REBASED: {d.note}"
            )
        else:
            delta = d.new_usd - d.old_usd
            out.append(
                f"{d.run_id:<28} {d.old_usd:12.6f} {d.new_usd:12.6f} {delta:+12.6f}  {d.note}"
            )
    return "\n".join(out)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--prices", type=Path, required=True)
    ap.add_argument("--trained-tokens", type=Path, default=None)
    ap.add_argument("--finetune-model", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--apply", action="store_true")
    a = ap.parse_args(argv)
    tokens = json.loads(a.trained_tokens.read_text()) if a.trained_tokens else None
    deltas = compute(a.root, load_price_file(a.prices), tokens, a.finetune_model)
    print(render(deltas))
    if a.apply:
        print(f"appended {apply(a.root, deltas)} corrective row(s)")
    else:
        print("report only; pass --apply to append corrective rows")
    return 0


if __name__ == "__main__":
    sys.exit(main())
