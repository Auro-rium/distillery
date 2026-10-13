#!/usr/bin/env python3
"""Drift-guard simulation on RECORDED rows (free: no network, no model calls).

1. Calibrate tau (the deferral-share threshold of detector (a)) on the FIT rows so that in-
   distribution traffic false-alarms at most ``--target-fa`` times per 1,000 requests.
2. Replay in-distribution traffic followed by a burst of ``--burst-rows`` through the drift monitor
   and report the detection latency, the wrong student answers served before the rollback, the
   student answers served after it (must be 0) and the false alarms before the burst.

Rows are router signal rows (``signals_*.json`` of ``distillery router signals``: a list or
``{"rows": [...]}``) with ``correct`` and either ``served_by`` or ``score``. For ``score`` rows pass
the certified threshold with ``--lambda`` or ``--router-dir`` (cert.json + model.json; rows with
neither ``served_by`` nor ``score`` are scored with the model).

  python scripts/drift_sim.py --fit-rows router/signals_fit.json --burst-rows burst.json \\
      --router-dir router --window 100 --target-fa 1 --audit-rate 0.05 --out drift_sim.json

The output file is never overwritten.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

from distillery.drift import DriftError, DriftParams
from distillery.drift_sim import (
    DEFAULT_BLOCK_LEN,
    DEFAULT_N_STREAMS,
    DriftSimError,
    audit_tail,
    calibrate_tau,
    load_rows,
    route_of,
    run_burst,
)
from distillery.router import RouterModel
from distillery.router_eval import RouterRefusal, features_of, write_new_json


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _score_rows(rows: list[dict[str, Any]], model: RouterModel | None) -> list[dict[str, Any]]:
    """Rows with neither ``served_by`` nor ``score`` get a score from the router model."""
    out = []
    for r in rows:
        if r.get("served_by") in ("student", "teacher") or r.get("score") is not None:
            out.append(r)
        elif model is not None:
            out.append({**r, "score": model.score(features_of(r))})
        else:
            raise DriftSimError(
                "a row has no served_by/score and no --router-dir model to score it"
            )
    return out


def _load_router_dir(d: Path) -> tuple[float | None, RouterModel | None]:
    cert = json.loads((d / "cert.json").read_text(encoding="utf-8"))
    lam = cert.get("lambda_hat")
    mpath = d / "model.json"
    model = None
    if mpath.is_file():
        doc = json.loads(mpath.read_text(encoding="utf-8"))
        model = RouterModel.from_json(doc.get("model", doc))
    return (None if lam is None else float(lam)), model


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--fit-rows", type=Path, required=True, help="rows tau is calibrated on")
    p.add_argument("--burst-rows", type=Path, required=True, help="rows of the shifted traffic")
    p.add_argument("--in-dist-rows", type=Path, help="in-distribution replay rows (default: fit)")
    p.add_argument("--out", type=Path, required=True, help="JSON artifact (never overwritten)")
    p.add_argument("--lambda", dest="lam", type=float, help="certified threshold for score rows")
    p.add_argument("--router-dir", type=Path, help="router dir: lambda_hat + model for scoring")
    p.add_argument("--window", type=int, required=True, help="W: the deferral window")
    p.add_argument(
        "--target-fa", type=float, required=True, help="false rollbacks per 1,000 requests"
    )
    p.add_argument("--tau", type=float, help="skip calibration and use this tau")
    p.add_argument("--min-requests", type=int, help="default: the window")
    p.add_argument("--audit-rate", type=float, default=0.05)
    p.add_argument("--audit-window", type=int, default=50, help="M: audited answers looked at")
    p.add_argument("--audit-k", type=int, default=5, help="k disagreements in M roll back")
    p.add_argument("--n-before", type=int, default=1000, help="in-distribution requests first")
    p.add_argument("--n-burst", type=int, default=500)
    p.add_argument("--n-streams", type=int, default=DEFAULT_N_STREAMS)
    p.add_argument("--block-len", type=int, default=DEFAULT_BLOCK_LEN)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--pass-within", type=int, help="verdict: pass needs a rollback within N burst requests"
    )
    p.add_argument(
        "--max-wrong-during-burst", type=int, help="verdict: at most K wrong answers in the burst"
    )
    return p


def verdict(burst: dict[str, Any], n: int, max_wrong: int) -> dict[str, Any]:
    """The pre-registered W-D rule: no false alarm before the burst, a rollback within ``n``
    burst requests (the first one counts as 1) and at most ``max_wrong`` wrong student answers
    served during the burst."""
    lat = burst["detection_latency_requests"]
    clauses = {
        "no_false_alarm": burst["false_alarms_before_burst"] == 0,
        "rolled_back_within_n": lat is not None and lat <= n,
        "wrong_during_burst_ok": burst["wrong_student_answers_served_during_burst"] <= max_wrong,
    }
    return {"n": n, "max_wrong_during_burst": max_wrong, **clauses, "pass": all(clauses.values())}


def main(argv: list[str] | None = None) -> int:
    a = build_parser().parse_args(argv)
    try:
        if a.out.exists():
            raise RouterRefusal(f"{a.out} already exists; it is never overwritten")
        lam, model = a.lam, None
        if a.router_dir is not None:
            d_lam, model = _load_router_dir(a.router_dir)
            lam = lam if lam is not None else d_lam
        fit = _score_rows(load_rows(a.fit_rows), model)
        burst = _score_rows(load_rows(a.burst_rows), model)
        in_dist = _score_rows(load_rows(a.in_dist_rows), model) if a.in_dist_rows else fit
        mreq = a.min_requests or a.window
        cal = None
        if a.tau is None:
            cal = calibrate_tau(
                fit, a.window, a.target_fa, a.n_streams, a.seed, lam=lam,
                block_len=a.block_len, min_requests=mreq,
            )  # fmt: skip
            tau = cal["tau"]
        else:
            tau = a.tau
        params = DriftParams(
            window_w=a.window, tau=tau, audit_rate=a.audit_rate, audit_window_m=a.audit_window,
            audit_k=a.audit_k, min_requests=mreq,
        )  # fmt: skip
        burst_report = run_burst(
            in_dist, burst, params, n_before=a.n_before, n_burst=a.n_burst, seed=a.seed, lam=lam
        )
        served = [r for r in fit if route_of(r, lam) == "student" and r.get("correct") is not None]
        err = (sum(1 for r in served if not r["correct"]) / len(served)) if served else None
        doc = {
            "format": "distillery-drift-sim-v1",
            "inputs": {
                "fit_rows": {"path": str(a.fit_rows), "sha256": _sha(a.fit_rows), "n": len(fit)},
                "burst_rows": {
                    "path": str(a.burst_rows),
                    "sha256": _sha(a.burst_rows),
                    "n": len(burst),
                },
                "in_dist_rows": str(a.in_dist_rows) if a.in_dist_rows else "fit rows (in-sample)",
                "lambda": lam,
            },  # fmt: skip
            "calibration": cal,
            "tau_source": "calibrated" if cal else "given on the command line",
            "params": params.to_json(),
            "burst": burst_report,
            "audit_check": {
                "served_student_error_rate_fit": err,
                "p_window_trips_in_distribution": None
                if err is None
                else audit_tail(a.audit_window, a.audit_k, err),
                "note": "P(Binomial(M, error rate) >= k): the chance one in-distribution audit "
                "window already holds k disagreements; keep it small",
            },
        }
        if (a.pass_within is None) != (a.max_wrong_during_burst is None):
            raise DriftSimError("--pass-within and --max-wrong-during-burst go together")
        if a.pass_within is not None:
            doc["verdict"] = verdict(burst_report, a.pass_within, a.max_wrong_during_burst)
        write_new_json(a.out, doc)
    except (DriftSimError, DriftError, RouterRefusal, OSError, ValueError, KeyError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    b = burst_report
    print(
        f"tau={tau} rollback_index={b['rollback_index']} detector={b['detector']} "
        f"latency={b['detection_latency_requests']} "
        f"wrong_before_rollback={b['wrong_student_answers_served_before_rollback']} "
        f"student_after_rollback={b['student_answers_served_after_rollback']} "
        f"false_alarms_before_burst={b['false_alarms_before_burst']} -> {a.out}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
