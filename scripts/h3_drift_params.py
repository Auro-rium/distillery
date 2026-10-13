#!/usr/bin/env python3
"""H3 drift-guard parameters, fixed BEFORE any calibration/test item is scored (free, no network).

Uses only the FIT split's recorded signals (``router/signals_fit.json``) and the fitted router
(``router/model.json``):

1. tau table: for every certification grid level (fit coverage 0.20..0.90), the threshold lambda
   at that level, the in-distribution deferral share, and the smallest tau (window W) whose false-
   rollback rate on in-distribution fit traffic is <= the target per 1,000 requests
   (``drift_sim.calibrate_tau``, iid resampling: the fit rows are stored grouped by source, so a
   block bootstrap would invent runs of deferrals that live traffic does not have).
2. audit tail: P(Binomial(M, alpha) >= k) for k in 5..12, the chance one in-distribution audit
   window already holds k disagreements when the served error equals alpha.
3. latency: synthetic bursts in which a share p of burst requests is deferred and every student
   answer is wrong, replayed after n_before in-distribution fit requests over many seeds; reports
   false alarms before the burst, misses within the burst, the detection latency quantiles and
   the wrong student answers served during the burst.

  uv run python scripts/h3_drift_params.py --router-dir <run>/router --out params.json

The output file is never overwritten.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

from distillery.drift import DriftParams
from distillery.drift_sim import DriftSimError, audit_tail, calibrate_tau, run_burst
from distillery.router import RouterModel
from distillery.router_eval import DEFAULT_LEVELS, features_of, write_new_json
from distillery.stats import coverage_grid


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _q(xs: list[int], f: float) -> int | None:
    return xs[min(len(xs) - 1, int(f * len(xs)))] if xs else None


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--router-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--window", type=int, default=20)
    ap.add_argument("--target-fa", type=float, default=0.1, help="false rollbacks per 1,000")
    ap.add_argument("--seed", type=int, default=20261013)
    ap.add_argument("--alpha", type=float, default=0.08)
    ap.add_argument("--audit-rate", type=float, default=0.10)
    ap.add_argument("--audit-window", type=int, default=50)
    ap.add_argument("--audit-k", type=int, default=10)
    ap.add_argument("--n-before", type=int, default=200)
    ap.add_argument("--n-burst", type=int, default=30)
    ap.add_argument("--latency-seeds", type=int, default=300)
    ap.add_argument("--latency-levels", type=float, nargs="+", default=[0.46, 0.56, 0.66])
    a = ap.parse_args(argv)
    if a.out.exists():
        print(f"error: {a.out} already exists; it is never overwritten", file=sys.stderr)
        return 2
    fit_path, model_path = a.router_dir / "signals_fit.json", a.router_dir / "model.json"
    doc = json.loads(model_path.read_text(encoding="utf-8"))
    model = RouterModel.from_json(doc["model"])
    rows = json.loads(fit_path.read_text(encoding="utf-8"))["rows"]
    rows = [{**r, "score": model.score(features_of(r))} for r in rows]
    grid = coverage_grid([r["score"] for r in rows], list(DEFAULT_LEVELS))

    table: list[dict[str, Any]] = []
    for lvl, lam in zip(DEFAULT_LEVELS, grid, strict=True):
        row: dict[str, Any] = {"fit_coverage_level": round(lvl, 2), "lambda": lam}
        try:
            c = calibrate_tau(rows, a.window, a.target_fa, seed=a.seed, lam=lam, block_len=1)
        except DriftSimError as exc:
            row.update(tau=None, note=f"no feasible tau: {exc}")
        else:
            row.update(
                tau=c["tau"],
                baseline_deferral_share=c["baseline_deferral_share"],
                false_alarm_per_1000=c["false_alarm_per_1000"],
                validation_false_alarm_per_1000=c["validation_false_alarm_per_1000"],
            )
        table.append(row)
        print(f"level {lvl:.2f} lambda {lam:.4f} tau {row['tau']}", file=sys.stderr)

    tails = {str(k): audit_tail(a.audit_window, k, a.alpha) for k in range(5, 13)}

    by_level = {r["fit_coverage_level"]: r for r in table}
    latency: list[dict[str, Any]] = []
    for lvl in a.latency_levels:
        t = by_level[round(lvl, 2)]
        if t["tau"] is None:
            continue
        params = DriftParams(
            window_w=a.window, tau=t["tau"], audit_rate=a.audit_rate,
            audit_window_m=a.audit_window, audit_k=a.audit_k, min_requests=a.window,
        )  # fmt: skip
        for p in (1.0, 0.95, 0.9, 0.8):
            burst = [
                {"task_id": f"b{i}", "score": 0.0 if (i % 100) < p * 100 else 1.0, "correct": False}
                for i in range(100)
            ]
            lats: list[int] = []
            wrong: list[int] = []
            fa = miss = 0
            for seed in range(a.latency_seeds):
                rep = run_burst(
                    rows, burst, params, n_before=a.n_before, n_burst=a.n_burst, seed=seed,
                    lam=t["lambda"],
                )  # fmt: skip
                if rep["false_alarms_before_burst"]:
                    fa += 1
                    continue
                if rep["detection_latency_requests"] is None:
                    miss += 1
                else:
                    lats.append(rep["detection_latency_requests"])
                    wrong.append(rep["wrong_student_answers_served_during_burst"])
            lats.sort()
            wrong.sort()
            latency.append(
                {
                    "fit_coverage_level": round(lvl, 2),
                    "tau": t["tau"],
                    "burst_deferral_share": p,
                    "seeds": a.latency_seeds,
                    "false_alarm_before_burst": fa,
                    "missed_within_burst": miss,
                    "latency_median": _q(lats, 0.5),
                    "latency_p95": _q(lats, 0.95),
                    "latency_max": lats[-1] if lats else None,
                    "wrong_during_burst_median": _q(wrong, 0.5),
                    "wrong_during_burst_p95": _q(wrong, 0.95),
                    "wrong_during_burst_max": wrong[-1] if wrong else None,
                }
            )
            print(f"latency {latency[-1]}", file=sys.stderr)

    out = {
        "format": "distillery-h3-drift-params-v1",
        "inputs": {
            "signals_fit": {"sha256": _sha(fit_path), "n": len(rows)},
            "model_sha256": doc["model_sha256"],
        },
        "window_w": a.window,
        "min_requests": a.window,
        "target_false_alarm_per_1000": a.target_fa,
        "seed": a.seed,
        "resampling": "iid (block_len 1)",
        "tau_table": table,
        "audit": {
            "rate": a.audit_rate,
            "window_m": a.audit_window,
            "k": a.audit_k,
            "alpha": a.alpha,
            "p_window_has_k_disagreements_at_alpha": tails,
        },
        "latency_sim": {
            "n_before": a.n_before,
            "n_burst": a.n_burst,
            "note": "synthetic bursts: a share p deferred, every student answer wrong; "
            "in-distribution requests drawn with replacement from the fit rows",
            "rows": latency,
        },
    }
    write_new_json(a.out, out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
