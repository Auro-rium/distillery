"""Pure-Python statistics: paired bootstrap ratio CI, Wilson CI, exact McNemar."""

from __future__ import annotations

import math
import random
from collections.abc import Sequence
from dataclasses import dataclass


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
