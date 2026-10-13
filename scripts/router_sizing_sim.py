#!/usr/bin/env python3
"""Router sizing: how much calibration data does Learn-then-Test need to certify a useful router?

Pure stdlib, seeded, no I/O except an optional ``--from-fit`` input and stdout. For each
calibration size it simulates ``--reps`` independent pipelines:

  1. draw a FIT split, fit the logistic router (``distillery.router.fit_logistic``) and build the
     threshold grid from FIT scores only (``coverage_grid``, fit-coverage levels 20%..90%);
  2. draw a sealed CALIBRATION split of the given size, score it, run ``certify`` (alpha, delta);
  3. draw a TEST split and apply lambda_hat to it.

and prints, per calibration size: mean calibration coverage, mean test coverage, P(certification
fails) (no lambda_hat: everything goes to the teacher), and how often the TEST point estimate of
the selective risk exceeds alpha (conditional on being certified, and unconditionally). A test
estimate above alpha is not a broken guarantee -- the guarantee is about the population risk, and a
finite test split adds binomial noise -- but it is what a reader of the final report will see.

Toy model (default): requests come from families (``--mix name:share:accuracy[:nn_mean]``). Given
correctness the three signals are drawn with a controllable separation ``--separation d``:
exec_ok runs with prob 0.98 when correct and 0.98 - 0.30 d when wrong; mean_logprob is
-exp(N(mu, 0.5)) with mu 0.5 d lower (more negative) when wrong; nn_sim is N(family mean +/-
0.03 d, 0.08), the family mean being the main signal. d = 0 means the signals carry no
information about correctness beyond the family.

``--from-fit FIT.json`` replaces the toy model: FIT.json is a list (or ``{"items": [...]}``) of
``{"exec_ok", "mean_logprob", "nn_sim", "correct"}`` objects from the fit split. The router is fit
on it and the simulation bootstraps the 5-fold OUT-OF-FOLD scores (in-sample scores would make the
router look better than it will on new traffic). Each repetition splits the pool at random into
disjoint calibration and test halves and draws calibration/test sets from them with replacement,
so power estimates are only trustworthy when the pool is much larger than the calibration size.

  python scripts/router_sizing_sim.py                       # defaults, ~20 s
  python scripts/router_sizing_sim.py --alpha 0.1 --reps 300 --separation 1.5
  python scripts/router_sizing_sim.py --from-fit fit_signals.json
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from distillery.router import FEATURES, RouterModel, fit_logistic
from distillery.stats import certify, coverage_grid, risk_coverage_curve

DEFAULT_MIX = "templated:0.5:0.92:0.85,natural:0.25:0.57:0.45,stress:0.25:0.66:0.55"
Draw = Callable[[int], tuple[list[float], list[bool]]]


@dataclass(frozen=True)
class Family:
    name: str
    share: float
    accuracy: float
    nn_mean: float


@dataclass(frozen=True)
class Rep:
    """One simulated pipeline: a fixed grid, a sampler for calibration sets, one test set."""

    grid: list[float]
    draw_cal: Draw
    test_scores: list[float]
    test_correct: list[bool]


def parse_mix(text: str) -> list[Family]:
    """``name:share:accuracy[:nn_mean]`` entries, comma separated; shares must sum to 1."""
    families: list[Family] = []
    for part in text.split(","):
        fields = part.strip().split(":")
        if len(fields) not in (3, 4):
            raise ValueError(f"bad mix entry {part!r}: want name:share:accuracy[:nn_mean]")
        nn_mean = float(fields[3]) if len(fields) == 4 else 0.6
        families.append(Family(fields[0], float(fields[1]), float(fields[2]), nn_mean))
    if not families or abs(sum(f.share for f in families) - 1.0) > 1e-6:
        raise ValueError("mix shares must sum to 1")
    if any(not 0.0 <= f.accuracy <= 1.0 or f.share < 0.0 for f in families):
        raise ValueError("mix shares must be >= 0 and accuracies in [0, 1]")
    return families


def toy_item(rng: random.Random, fam: Family, sep: float) -> tuple[dict[str, float], bool]:
    correct = rng.random() < fam.accuracy
    sign = 1.0 if correct else -1.0
    p_exec = 0.98 if correct else min(0.98, max(0.05, 0.98 - 0.30 * sep))
    mu = -1.9 + (0.0 if correct else 0.5 * sep)
    logprob = -math.exp(rng.gauss(mu, 0.5))
    sim = min(1.0, max(-1.0, rng.gauss(fam.nn_mean + 0.03 * sep * sign, 0.08)))
    row = {
        "exec_ok": 1.0 if rng.random() < p_exec else 0.0,
        "mean_logprob": logprob,
        "nn_sim": sim,
    }
    return row, correct


def draw_toy(
    rng: random.Random, families: Sequence[Family], n: int, sep: float
) -> tuple[list[dict[str, float]], list[bool]]:
    shares = [f.share for f in families]
    rows: list[dict[str, float]] = []
    labels: list[bool] = []
    for fam in rng.choices(families, weights=shares, k=n):
        row, ok = toy_item(rng, fam, sep)
        rows.append(row)
        labels.append(ok)
    return rows, labels


def score_all(model: RouterModel, rows: Sequence[dict[str, float]]) -> list[float]:
    return [model.score(r) for r in rows]


def make_levels(start: float, stop: float, step: float) -> list[float]:
    """Increasing fit-coverage levels start, start + step, ... <= stop (rounded against drift)."""
    if not 0.0 < start <= stop <= 1.0 or step <= 0.0:
        raise ValueError("need 0 < level-start <= level-stop <= 1 and level-step > 0")
    n = int(math.floor((stop - start) / step + 1e-9)) + 1
    return [round(start + i * step, 10) for i in range(n)]


def toy_rep_factory(
    families: Sequence[Family], sep: float, fit_size: int, test_size: int, levels: Sequence[float]
) -> Callable[[random.Random], Rep]:
    def make(rng: random.Random) -> Rep:
        fit_rows, fit_labels = draw_toy(rng, families, fit_size, sep)
        model = fit_logistic(fit_rows, fit_labels)
        grid = coverage_grid(score_all(model, fit_rows), levels)

        def draw_cal(n: int) -> tuple[list[float], list[bool]]:
            rows, labels = draw_toy(rng, families, n, sep)
            return score_all(model, rows), labels

        test_rows, test_labels = draw_toy(rng, families, test_size, sep)
        return Rep(grid, draw_cal, score_all(model, test_rows), test_labels)

    return make


def load_fit_items(path: Path) -> tuple[list[dict[str, float]], list[bool]]:
    data: Any = json.loads(path.read_text(encoding="utf-8"))
    # a router signals file (``distillery router signals``) keeps them under "rows"
    items = (data.get("items") or data.get("rows") or []) if isinstance(data, dict) else data
    rows = [{f: float(it[f]) for f in FEATURES} for it in items]
    labels = [bool(it["correct"]) for it in items]
    if len(rows) < 20:
        raise ValueError("need at least 20 fit items")
    return rows, labels


def out_of_fold_scores(
    rows: Sequence[dict[str, float]], labels: Sequence[bool], folds: int = 5
) -> list[float]:
    """Score every item with a router fit on the other folds (fold = index mod ``folds``)."""
    out = [0.0] * len(rows)
    for k in range(folds):
        train = [i for i in range(len(rows)) if i % folds != k]
        model = fit_logistic([rows[i] for i in train], [labels[i] for i in train])
        for i in range(k, len(rows), folds):
            out[i] = model.score(rows[i])
    return out


def pool_rep_factory(
    scores: Sequence[float], labels: Sequence[bool], test_size: int, levels: Sequence[float]
) -> Callable[[random.Random], Rep]:
    grid = coverage_grid(scores, levels)
    idx = list(range(len(scores)))

    def make(rng: random.Random) -> Rep:
        rng.shuffle(idx)
        half = len(idx) // 2
        cal_pool, test_pool = idx[:half], idx[half:]

        def draw_cal(n: int) -> tuple[list[float], list[bool]]:
            pick = rng.choices(cal_pool, k=n)
            return [scores[i] for i in pick], [labels[i] for i in pick]

        pick = rng.choices(test_pool, k=test_size)
        return Rep(grid, draw_cal, [scores[i] for i in pick], [labels[i] for i in pick])

    return make


def run(
    make_rep: Callable[[random.Random], Rep],
    *,
    alpha: float,
    delta: float,
    cal_sizes: Sequence[int],
    reps: int,
    seed: int,
) -> list[dict[str, float]]:
    """Per calibration size: the averaged outcome over ``reps`` simulated pipelines."""
    acc = {
        n: {"fail": 0, "cal_cov": 0.0, "test_cov": 0.0, "test_risk": 0.0, "over": 0}
        for n in cal_sizes
    }
    n_grid = 0
    for rep_i in range(reps):
        rng = random.Random(seed * 1_000_003 + rep_i)  # noqa: S311 - simulation, not security
        rep = make_rep(rng)
        n_grid = len(rep.grid)
        for n in cal_sizes:
            scores, correct = rep.draw_cal(n)
            res = certify(scores, correct, alpha=alpha, delta=delta, grid=rep.grid)
            a = acc[n]
            if res.lambda_hat is None:
                a["fail"] += 1
                continue
            a["cal_cov"] += res.coverage
            pt = risk_coverage_curve(rep.test_scores, rep.test_correct, [res.lambda_hat])[0]
            a["test_cov"] += float(pt["coverage"] or 0.0)
            if pt["risk"] is not None:
                a["test_risk"] += float(pt["risk"])
                a["over"] += float(pt["risk"]) > alpha
    out: list[dict[str, float]] = []
    for n in cal_sizes:
        a = acc[n]
        cert = reps - a["fail"]
        out.append(
            {
                "cal_size": n,
                "reps": reps,
                "n_grid": n_grid,
                "p_fail": a["fail"] / reps,
                "mean_cal_coverage": a["cal_cov"] / reps,  # failures count as coverage 0
                "mean_test_coverage": a["test_cov"] / reps,
                "mean_test_risk_certified": a["test_risk"] / cert if cert else float("nan"),
                "p_test_over_alpha_given_cert": a["over"] / cert if cert else float("nan"),
                "p_cert_and_test_over_alpha": a["over"] / reps,
            }
        )
    return out


def format_table(results: Sequence[dict[str, float]]) -> str:
    head = (
        f"{'cal_n':>6} {'P(fail)':>8} {'cal_cov':>8} {'test_cov':>9} {'test_risk':>10} "
        f"{'P(test>a|cert)':>15} {'P(cert&test>a)':>15}"
    )
    lines = [head, "-" * len(head)]
    for r in results:
        lines.append(
            f"{int(r['cal_size']):>6} {r['p_fail']:>8.3f} {r['mean_cal_coverage']:>8.3f} "
            f"{r['mean_test_coverage']:>9.3f} {r['mean_test_risk_certified']:>10.4f} "
            f"{r['p_test_over_alpha_given_cert']:>15.3f} {r['p_cert_and_test_over_alpha']:>15.3f}"
        )
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0] if __doc__ else None)
    ap.add_argument("--alpha", type=float, default=0.08, help="target selective risk")
    ap.add_argument("--delta", type=float, default=0.05, help="certification failure probability")
    ap.add_argument("--cal-sizes", type=int, nargs="+", default=[300, 400, 500, 600])
    ap.add_argument("--fit-size", type=int, default=1500)
    ap.add_argument("--test-size", type=int, default=1000)
    ap.add_argument("--reps", type=int, default=400, help="simulated pipelines per run")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--mix", default=DEFAULT_MIX, help="name:share:accuracy[:nn_mean],...")
    ap.add_argument("--separation", type=float, default=1.0, help="signal separation d (0 = none)")
    ap.add_argument("--level-start", type=float, default=0.20)
    ap.add_argument("--level-stop", type=float, default=0.90)
    ap.add_argument("--level-step", type=float, default=0.02)
    ap.add_argument("--from-fit", type=Path, default=None, help="fit-split signals JSON")
    ap.add_argument("--json", action="store_true", help="print results as JSON")
    args = ap.parse_args(argv)

    levels = make_levels(args.level_start, args.level_stop, args.level_step)
    if args.from_fit is not None:
        rows, labels = load_fit_items(args.from_fit)
        oof = out_of_fold_scores(rows, labels)
        make_rep = pool_rep_factory(oof, labels, args.test_size, levels)
        source = f"fit split {args.from_fit} ({len(rows)} items, 5-fold out-of-fold scores)"
    else:
        families = parse_mix(args.mix)
        make_rep = toy_rep_factory(families, args.separation, args.fit_size, args.test_size, levels)
        overall = sum(f.share * f.accuracy for f in families)
        mix_txt = " | ".join(f"{f.name} {f.share:.0%} acc {f.accuracy:.2f}" for f in families)
        source = f"toy model: {mix_txt} (student acc {overall:.3f}), separation {args.separation}"
    results = run(
        make_rep,
        alpha=args.alpha,
        delta=args.delta,
        cal_sizes=args.cal_sizes,
        reps=args.reps,
        seed=args.seed,
    )
    if args.json:
        clean = [
            {k: (None if isinstance(v, float) and math.isnan(v) else v) for k, v in r.items()}
            for r in results
        ]
        print(json.dumps(clean, indent=2, allow_nan=False))
        return 0
    print(
        f"router_sizing_sim: alpha={args.alpha} delta={args.delta} "
        f"reps={args.reps} seed={args.seed}"
    )
    print(f"source: {source}")
    print(
        f"grid: {len(levels)} fit-coverage levels {levels[0]:.0%}..{levels[-1]:.0%} "
        f"(first step must select >= ~{math.ceil(math.log(args.delta) / math.log(1 - args.alpha))} "
        f"calibration items to certify with 0 errors)"
    )
    print(f"fit={'(from file)' if args.from_fit else args.fit_size} test={args.test_size}")
    print(format_table(results))
    print(
        "\ncal_cov / test_cov: share served by the student (0 when certification fails). "
        "P(test>a|cert): the test point estimate exceeds alpha although the router was certified."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
