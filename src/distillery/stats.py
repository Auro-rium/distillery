"""Pure-Python statistics: paired bootstrap ratio CI, Wilson CI, exact McNemar, and the
Learn-then-Test selective-risk certificate (binomial tails, Clopper-Pearson, fixed-sequence)."""

from __future__ import annotations

import math
import random
from bisect import bisect_right
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class RatioCI:
    point: float
    lo: float
    hi: float
    resamples_requested: int
    resamples_used: int
    skipped_teacher_zero: int


@dataclass(frozen=True)
class McNemarResult:
    p_value: float
    a_only: int  # items only A got right
    b_only: int  # items only B got right
    n_discordant: int


def _check_pair(a: Sequence[bool], b: Sequence[bool]) -> int:
    if len(a) != len(b):
        raise ValueError(f"length mismatch: {len(a)} vs {len(b)}")
    if len(a) == 0:
        raise ValueError("need at least one item (n=0)")
    return len(a)


def paired_bootstrap_ratio_ci(
    student: Sequence[bool],
    teacher: Sequence[bool],
    resamples: int = 10000,
    seed: int = 1234,
    alpha: float = 0.05,
) -> RatioCI:
    """Percentile bootstrap CI of acc(student)/acc(teacher), resampling items jointly.

    Resamples where the teacher scores 0 are skipped (ratio undefined) and counted.
    Raises ValueError if the teacher has zero accuracy on the full sample, or if
    every resample was skipped.
    """
    n = _check_pair(student, teacher)
    if resamples < 1:
        raise ValueError("resamples must be >= 1")
    if not 0 < alpha < 1:
        raise ValueError("alpha must be in (0, 1)")
    s = [1 if x else 0 for x in student]
    t = [1 if x else 0 for x in teacher]
    t_total = sum(t)
    if t_total == 0:
        raise ValueError("teacher accuracy is 0; ratio undefined")
    point = sum(s) / t_total
    rng = random.Random(seed)  # noqa: S311 - statistical resampling, not security
    idx = range(n)
    ratios: list[float] = []
    skipped = 0
    for _ in range(resamples):
        pick = rng.choices(idx, k=n)
        tc = sum(t[i] for i in pick)
        if tc == 0:
            skipped += 1
            continue
        ratios.append(sum(s[i] for i in pick) / tc)
    if not ratios:
        raise ValueError("all bootstrap resamples had teacher accuracy 0")
    ratios.sort()
    m = len(ratios)
    lo_i = min(m - 1, max(0, math.floor(alpha / 2 * m)))
    hi_i = min(m - 1, max(0, math.ceil((1 - alpha / 2) * m) - 1))
    return RatioCI(point, ratios[lo_i], ratios[hi_i], resamples, m, skipped)


def wilson_ci(k: int, n: int, z: float = 1.959963984540054) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion (default 95%)."""
    if n <= 0:
        raise ValueError("n must be positive")
    if not 0 <= k <= n:
        raise ValueError("k must be in [0, n]")
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def mcnemar_exact(a_correct: Sequence[bool], b_correct: Sequence[bool]) -> McNemarResult:
    """Exact two-sided McNemar test on discordant pairs (Binomial(n_d, 0.5))."""
    _check_pair(a_correct, b_correct)
    a_only = sum(1 for a, b in zip(a_correct, b_correct, strict=True) if a and not b)
    b_only = sum(1 for a, b in zip(a_correct, b_correct, strict=True) if b and not a)
    nd = a_only + b_only
    if nd == 0:
        return McNemarResult(1.0, 0, 0, 0)
    k = min(a_only, b_only)
    tail = sum(math.comb(nd, i) for i in range(k + 1)) / 2**nd
    return McNemarResult(min(1.0, 2 * tail), a_only, b_only, nd)


# ---------------------------------------------------------------------------------------------
# Learn-then-Test selective-risk certificate (Angelopoulos et al.), fixed-sequence testing.
# ---------------------------------------------------------------------------------------------


def binom_cdf(k: int, n: int, p: float) -> float:
    """Exact P(X <= k) for X ~ Binomial(n, p); stable for n in the thousands.

    The pmf at an anchor term is computed in log space (log C(n, k) from the exact integer
    ``math.comb``, so no lgamma error), then the neighbouring terms follow from the ratio
    recurrence, which only ever DEcreases away from the anchor: the lower tail sums downward from
    k when k <= (n + 1) p, otherwise the upper tail sums upward from k + 1 and the cdf is its
    complement. Nothing overflows before the answer itself underflows (a cdf below ~1e-308
    returns 0.0).
    """
    if n < 0:
        raise ValueError("n must be >= 0")
    if not 0.0 <= p <= 1.0:
        raise ValueError("p must be in [0, 1]")
    if k < 0:
        return 0.0
    if k >= n or p == 0.0:
        return 1.0
    if p == 1.0:
        return 0.0
    q = 1.0 - p
    log_p = math.log(p)
    log_q = math.log1p(-p)
    if k <= (n + 1) * p:
        # pmf(i - 1) / pmf(i) = i q / ((n - i + 1) p) <= 1 for every i <= k here.
        log_anchor = math.log(math.comb(n, k)) + k * log_p + (n - k) * log_q
        term = total = 1.0
        for i in range(k, 0, -1):
            term *= (i * q) / ((n - i + 1) * p)
            total += term
            if term < total * 1e-20:
                break
        return min(1.0, math.exp(log_anchor + math.log(total)))
    j = k + 1
    # pmf(i + 1) / pmf(i) = (n - i) p / ((i + 1) q) <= 1 for every i >= j here.
    log_anchor = math.log(math.comb(n, j)) + j * log_p + (n - j) * log_q
    term = total = 1.0
    for i in range(j, n):
        term *= ((n - i) * p) / ((i + 1) * q)
        total += term
        if term < total * 1e-20:
            break
    return max(0.0, 1.0 - math.exp(log_anchor + math.log(total)))


def clopper_pearson_upper(errors: int, n: int, conf: float = 0.95) -> float:
    """One-sided Clopper-Pearson upper bound on a binomial error rate.

    The smallest p with P(Binomial(n, p) <= errors) <= 1 - conf, found by bisection on
    ``binom_cdf`` (which is decreasing in p). n = 0 carries no information, so it returns 1.0;
    so does errors == n.
    """
    if not 0.0 < conf < 1.0:
        raise ValueError("conf must be in (0, 1)")
    if n < 0 or not 0 <= errors <= n:
        raise ValueError("need 0 <= errors <= n")
    if n == 0 or errors == n:
        return 1.0
    tail = 1.0 - conf
    lo, hi = 0.0, 1.0  # invariant: cdf(lo) > tail >= cdf(hi)
    for _ in range(200):
        mid = (lo + hi) / 2
        if mid == lo or mid == hi:
            break
        if binom_cdf(errors, n, mid) > tail:
            lo = mid
        else:
            hi = mid
    return hi


def clopper_pearson_lower(successes: int, n: int, conf: float = 0.95) -> float:
    """One-sided Clopper-Pearson LOWER bound on a binomial rate.

    The largest p with P(Binomial(n, p) >= successes) <= 1 - conf, i.e. with
    P(Binomial(n, p) <= successes - 1) >= conf; found by bisection on ``binom_cdf`` (decreasing in
    p). With no successes (or n = 0) there is no information above zero: 0.0.
    """
    if not 0.0 < conf < 1.0:
        raise ValueError("conf must be in (0, 1)")
    if n < 0 or not 0 <= successes <= n:
        raise ValueError("need 0 <= successes <= n")
    if n == 0 or successes == 0:
        return 0.0
    lo, hi = 0.0, 1.0  # invariant: cdf(lo) >= conf > cdf(hi)
    for _ in range(200):
        mid = (lo + hi) / 2
        if mid == lo or mid == hi:
            break
        if binom_cdf(successes - 1, n, mid) >= conf:
            lo = mid
        else:
            hi = mid
    return lo


@dataclass(frozen=True)
class CertStep:
    threshold: float
    n_selected: int
    n_errors: int
    p_value: float  # P(Binomial(n_selected, alpha) <= n_errors); 1.0 when nothing is selected
    rejected: bool  # p_value <= delta: "selective risk > alpha" is rejected at this threshold
    cp_upper: float  # one-sided Clopper-Pearson upper bound at confidence 1 - delta

    def to_json(self) -> dict[str, Any]:
        return {
            "threshold": self.threshold,
            "n_selected": self.n_selected,
            "n_errors": self.n_errors,
            "p_value": self.p_value,
            "rejected": self.rejected,
            "cp_upper": self.cp_upper,
        }


@dataclass(frozen=True)
class CertResult:
    alpha: float
    delta: float
    lambda_hat: float | None  # None: nothing certified, all traffic goes to the teacher
    coverage: float  # calibration share served by the student at lambda_hat (0.0 if None)
    selective_risk: (
        float | None
    )  # calibration error rate among served items (None if no lambda_hat)
    steps: tuple[CertStep, ...]

    def to_json(self) -> dict[str, Any]:
        return {
            "alpha": self.alpha,
            "delta": self.delta,
            "lambda_hat": self.lambda_hat,
            "coverage": self.coverage,
            "selective_risk": self.selective_risk,
            "steps": [step.to_json() for step in self.steps],
        }


def _sorted_prefix(
    scores: Sequence[float], correct: Sequence[bool]
) -> tuple[list[float], list[int]]:
    """Negated scores in ascending order and cumulative error counts (error = not correct).

    With these, the number of items with score >= t is ``bisect_right(neg, -t)`` and their error
    count is ``cum[that]``; exact because negation is exact in floating point.
    """
    if len(scores) != len(correct):
        raise ValueError(f"length mismatch: {len(scores)} scores vs {len(correct)} labels")
    if not all(math.isfinite(s) for s in scores):
        raise ValueError("scores must be finite")
    order = sorted(range(len(scores)), key=lambda i: -scores[i])
    neg = [-scores[i] for i in order]
    cum = [0]
    for i in order:
        cum.append(cum[-1] + (0 if correct[i] else 1))
    return neg, cum


def certify(
    scores: Sequence[float],
    correct: Sequence[bool],
    *,
    alpha: float,
    delta: float,
    grid: Sequence[float],
) -> CertResult:
    """Learn-then-Test certificate for the selective risk of "serve the student iff score >= t".

    ``grid`` must be strictly DECREASING (most conservative threshold first) and fixed BEFORE
    the calibration data is looked at (build it from the fit split with ``coverage_grid``). For
    each threshold in order, among the n calibration items with score >= t and their e errors,
    p = P(Binomial(n, alpha) <= e) (1.0 if n == 0) is the exact p-value for the null "selective
    risk > alpha"; the null is rejected iff p <= delta. Testing stops at the first non-rejection
    (fixed-sequence testing: FWER <= delta with no multiplicity correction and no monotonicity
    assumption), so lambda_hat, the last rejected threshold, satisfies
    P(true selective risk at lambda_hat <= alpha) >= 1 - delta, for calibration data exchangeable
    with the traffic being served. ``rejected`` is equivalent to ``cp_upper <= alpha``.
    No rejected threshold: lambda_hat is None, coverage 0.0, everything goes to the teacher.
    An empty grid is a bug, not "nothing to test", and raises.
    """
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be in (0, 1)")
    if not 0.0 < delta < 1.0:
        raise ValueError("delta must be in (0, 1)")
    if len(grid) == 0:
        raise ValueError("grid must not be empty")
    if not all(math.isfinite(t) for t in grid):
        raise ValueError("grid thresholds must be finite")
    if any(a <= b for a, b in zip(grid, grid[1:], strict=False)):
        raise ValueError("grid must be strictly decreasing (most conservative threshold first)")
    neg, cum = _sorted_prefix(scores, correct)
    steps: list[CertStep] = []
    lambda_hat: float | None = None
    last = (0, 0)
    for t in grid:
        n = bisect_right(neg, -t)
        e = cum[n]
        p = binom_cdf(e, n, alpha) if n > 0 else 1.0
        rejected = p <= delta
        steps.append(CertStep(t, n, e, p, rejected, clopper_pearson_upper(e, n, 1.0 - delta)))
        if not rejected:
            break
        lambda_hat, last = t, (n, e)
    if lambda_hat is None:
        return CertResult(alpha, delta, None, 0.0, None, tuple(steps))
    n, e = last
    return CertResult(alpha, delta, lambda_hat, n / len(scores), e / n, tuple(steps))


def coverage_grid(fit_scores: Sequence[float], levels: Sequence[float]) -> list[float]:
    """Strictly decreasing thresholds t such that about a fraction ``level`` of FIT scores is >= t.

    ``levels`` must be strictly increasing and in (0, 1]. For m fit scores and a level, the
    threshold is the k-th largest score with k = max(1, ceil(level * m)) (a 1e-9 slack keeps
    float drift like 0.07 * 100 = 7.000000000000001 from costing an item). Tie rule: the
    realized fit coverage is >= level (every tied score is >= the threshold), and a threshold
    that is not strictly below the previous one is dropped, so the most conservative level of a
    tie is kept. The grid depends on the fit split only; it must not be tuned on calibration data.
    """
    m = len(fit_scores)
    if m == 0:
        raise ValueError("need at least one fit score")
    if not all(math.isfinite(s) for s in fit_scores):
        raise ValueError("fit scores must be finite")
    if len(levels) == 0:
        raise ValueError("levels must not be empty")
    if not all(0.0 < lv <= 1.0 for lv in levels):
        raise ValueError("levels must be in (0, 1]")
    if any(a >= b for a, b in zip(levels, levels[1:], strict=False)):
        raise ValueError("levels must be strictly increasing")
    desc = sorted(fit_scores, reverse=True)
    grid: list[float] = []
    for lv in levels:
        k = min(m, max(1, math.ceil(lv * m - 1e-9)))
        t = desc[k - 1]
        if not grid or t < grid[-1]:
            grid.append(t)
    return grid


def risk_coverage_curve(
    scores: Sequence[float], correct: Sequence[bool], thresholds: Sequence[float]
) -> list[dict[str, float | int | None]]:
    """Empirical selective-risk / coverage point at each threshold (selected = score >= t).

    ``risk`` is the error rate among selected items, None when nothing is selected (so the
    output stays valid JSON). These are point estimates, not certificates.
    """
    neg, cum = _sorted_prefix(scores, correct)
    total = len(scores)
    curve: list[dict[str, float | int | None]] = []
    for t in thresholds:
        n = bisect_right(neg, -t)
        e = cum[n]
        curve.append(
            {
                "threshold": t,
                "coverage": n / total if total else 0.0,
                "n_selected": n,
                "n_errors": e,
                "risk": e / n if n else None,
            }
        )
    return curve
