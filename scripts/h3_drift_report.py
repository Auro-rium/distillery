#!/usr/bin/env python3
"""H3 W-D numbers that are REPORTED, not gated (pre-registered in DECISIONS.md, 2026-10-13).

Free: replays recorded router signal rows through the drift monitor, no model calls.

1. The pre-registered replay (test rows, then the 30 burst rows) over seeds 0..N-1: latency
   quantiles, misses within the burst, false alarms before it and how often the verdict passes.
2. A burst-free stream of in-distribution test rows: false rollbacks per 1,000 requests.
3. A second scenario: the burst is the test rows of the fit family with the highest student
   error (stress ``date_math``).

  uv run python scripts/h3_drift_report.py --router-dir <run>/router --tau 0.9 --out report.json

lambda_hat comes from ``cert.json`` and rows are scored with ``model.json``. Never overwrites.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

from distillery.drift import DriftParams
from distillery.drift_sim import load_rows, run_burst
from distillery.router import RouterModel
from distillery.router_eval import features_of, write_new_json

sys.path.insert(0, str(Path(__file__).resolve().parent))
from drift_sim import verdict  # noqa: E402


def _q(xs: list[int], f: float) -> int | None:
    return xs[min(len(xs) - 1, int(f * len(xs)))] if xs else None


def _sweep(
    in_dist: list[dict[str, Any]],
    burst: list[dict[str, Any]],
    params: DriftParams,
    lam: float,
    seeds: int,
    n_before: int,
    n_burst: int,
    n: int,
    max_wrong: int,
) -> dict[str, Any]:
    lats: list[int] = []
    wrong: list[int] = []
    fa = miss = passes = 0
    for seed in range(seeds):
        rep = run_burst(
            in_dist, burst, params, n_before=n_before, n_burst=n_burst, seed=seed, lam=lam
        )
        passes += verdict(rep, n, max_wrong)["pass"]
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
    return {
        "seeds": seeds,
        "verdict_pass": passes,
        "false_alarm_before_burst": fa,
        "missed_within_burst": miss,
        "latency_median": _q(lats, 0.5),
        "latency_p95": _q(lats, 0.95),
        "latency_max": lats[-1] if lats else None,
        "wrong_during_burst_median": _q(wrong, 0.5),
        "wrong_during_burst_max": wrong[-1] if wrong else None,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--router-dir", type=Path, required=True)
    ap.add_argument("--tau", type=float, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--seeds", type=int, default=300)
    ap.add_argument("--no-burst-requests", type=int, default=1000)
    ap.add_argument("--family", default="date_math")
    a = ap.parse_args(argv)
    if a.out.exists():
        print(f"error: {a.out} already exists; it is never overwritten", file=sys.stderr)
        return 2
    d = a.router_dir
    lam = float(json.loads((d / "cert.json").read_text(encoding="utf-8"))["lambda_hat"])
    model = RouterModel.from_json(
        json.loads((d / "model.json").read_text(encoding="utf-8"))["model"]
    )

    def scored(name: str) -> list[dict[str, Any]]:
        return [{**r, "score": model.score(features_of(r))} for r in load_rows(d / name)]

    test, burst = scored("signals_test.json"), scored("signals_burst.json")
    params = DriftParams(
        window_w=20, tau=a.tau, audit_rate=0.10, audit_window_m=50, audit_k=10, min_requests=20
    )
    fam = [r for r in test if r.get("family") == a.family]
    no_burst = run_burst(  # n_burst 0: the burst rows are never drawn
        test, test, params, n_before=a.no_burst_requests, n_burst=0, seed=20261013, lam=lam
    )
    out = {
        "format": "distillery-h3-drift-report-v1",
        "inputs": {
            name: hashlib.sha256((d / name).read_bytes()).hexdigest()
            for name in ("signals_test.json", "signals_burst.json", "cert.json", "model.json")
        },
        "lambda_hat": lam,
        "params": params.to_json(),
        "preregistered_burst_over_seeds": _sweep(test, burst, params, lam, a.seeds, 200, 30, 25, 3),
        "no_burst_stream": {
            "requests": a.no_burst_requests,
            "seed": 20261013,
            "rolled_back": no_burst["rolled_back"],
            "rollback_index": no_burst["rollback_index"],
            "detector": no_burst["detector"],
        },
        "family_burst": {
            "family": a.family,
            "n_rows": len(fam),
            **_sweep(test, fam, params, lam, a.seeds, 200, 30, 25, 3),
        },
    }
    write_new_json(a.out, out)
    print(json.dumps({k: v for k, v in out.items() if k != "inputs"}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
