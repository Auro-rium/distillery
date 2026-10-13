#!/usr/bin/env python3
"""H3 sizing on the REAL fit signals, run BEFORE any calibration/test item is scored (free).

Bootstraps the 5-fold out-of-fold router scores of the fit split (``router_sizing_sim``'s pool
mode) into calibration (489) and test (490) draws, certifies each calibration draw with
Learn-then-Test and applies lambda_hat to the test draw. Reports P(certification fails), mean
coverage, and the chance the W-C1 risk + coverage clauses pass (selective error <= 0.08 and test
coverage >= 0.30) when certifying at several alphas; the W-C1 bar itself stays 0.08. The pool
holds ~290 items per half, so draws with replacement overstate the test-split variance a little.

  uv run python scripts/h3_router_sizing.py --fit <run>/router/signals_fit.json --out sizing.json
"""

from __future__ import annotations

import argparse
import hashlib
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from router_sizing_sim import (  # noqa: E402
    load_fit_items,
    make_levels,
    out_of_fold_scores,
    pool_rep_factory,
)

from distillery.router_eval import write_new_json  # noqa: E402
from distillery.stats import certify, risk_coverage_curve  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--fit", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--n-cal", type=int, default=489)
    ap.add_argument("--n-test", type=int, default=490)
    ap.add_argument("--reps", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=20261013)
    ap.add_argument("--bar", type=float, default=0.08, help="the W-C1 selective-error bar")
    ap.add_argument("--min-coverage", type=float, default=0.30)
    ap.add_argument("--cert-alphas", type=float, nargs="+", default=[0.08, 0.07, 0.06, 0.05])
    a = ap.parse_args(argv)
    if a.out.exists():
        print(f"error: {a.out} already exists; it is never overwritten", file=sys.stderr)
        return 2
    rows, labels = load_fit_items(a.fit)
    oof = out_of_fold_scores(rows, labels)
    levels = make_levels(0.20, 0.90, 0.02)
    make = pool_rep_factory(oof, labels, a.n_test, levels)
    results = []
    for a_cert in a.cert_alphas:
        fail = wc1 = over = 0
        cov_sum = risk_sum = 0.0
        n_cert = 0
        for i in range(a.reps):
            rep = make(random.Random(a.seed * 1_000_003 + i))  # noqa: S311 (simulation)
            s, c = rep.draw_cal(a.n_cal)
            res = certify(s, c, alpha=a_cert, delta=0.05, grid=rep.grid)
            if res.lambda_hat is None:
                fail += 1
                continue
            n_cert += 1
            pt = risk_coverage_curve(rep.test_scores, rep.test_correct, [res.lambda_hat])[0]
            cov = float(pt["coverage"] or 0.0)
            risk = None if pt["risk"] is None else float(pt["risk"])
            cov_sum += cov
            risk_sum += risk or 0.0
            over += risk is not None and risk > a.bar
            wc1 += risk is not None and risk <= a.bar and cov >= a.min_coverage
        results.append(
            {
                "cert_alpha": a_cert,
                "p_cert_fail": fail / a.reps,
                "mean_test_coverage_given_cert": cov_sum / n_cert if n_cert else None,
                "mean_test_risk_given_cert": risk_sum / n_cert if n_cert else None,
                "p_test_risk_over_bar_given_cert": over / n_cert if n_cert else None,
                "p_wc1_risk_and_coverage_pass": wc1 / a.reps,
            }
        )
        print(results[-1], file=sys.stderr)
    write_new_json(
        a.out,
        {
            "format": "distillery-h3-router-sizing-v1",
            "inputs": {
                "signals_fit_sha256": hashlib.sha256(a.fit.read_bytes()).hexdigest(),
                "n_fit": len(rows),
            },
            "n_cal": a.n_cal,
            "n_test": a.n_test,
            "reps": a.reps,
            "seed": a.seed,
            "delta": 0.05,
            "wc1_bar": a.bar,
            "wc1_min_coverage": a.min_coverage,
            "note": "the system-accuracy clause of W-C1 is not simulated (it needs teacher "
            "outcomes on test)",
            "results": results,
        },
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
