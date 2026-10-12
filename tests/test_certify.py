# ruff: noqa: S101
from __future__ import annotations

import json
import math
import random
from fractions import Fraction

import pytest

from distillery.stats import (
    CertResult,
    binom_cdf,
    certify,
    clopper_pearson_lower,
    clopper_pearson_upper,
    coverage_grid,
    risk_coverage_curve,
)


def exact_cdf(k: int, n: int, p: float) -> float:
    """Reference: exact rational sum of the pmf (p taken as the exact float value)."""
    pf = Fraction(p)
    total = sum(
        Fraction(math.comb(n, i)) * pf**i * (1 - pf) ** (n - i)
        for i in range(max(0, min(k, n)) + 1)
    )
    return float(total) if k >= 0 else 0.0


# ---------------------------------------------------------------------------- binom_cdf


@pytest.mark.parametrize("p", [1e-6, 0.01, 0.08, 0.3, 0.5, 0.77, 0.99, 1 - 1e-9])
def test_binom_cdf_matches_exact_comb(p: float) -> None:
    for n in (0, 1, 2, 5, 17, 36, 60):
        for k in range(-1, n + 2):
            want = exact_cdf(k, n, p)
            got = binom_cdf(k, n, p)
            assert got == pytest.approx(want, rel=1e-11, abs=1e-300), (k, n, p)


def test_binom_cdf_edges() -> None:
    assert binom_cdf(-1, 10, 0.3) == 0.0
    assert binom_cdf(10, 10, 0.3) == 1.0 and binom_cdf(99, 10, 0.3) == 1.0
    assert binom_cdf(0, 0, 0.3) == 1.0
    assert binom_cdf(0, 10, 0.0) == 1.0 and binom_cdf(5, 10, 0.0) == 1.0
    assert binom_cdf(0, 10, 1.0) == 0.0 and binom_cdf(9, 10, 1.0) == 0.0
    assert binom_cdf(10, 10, 1.0) == 1.0
    assert binom_cdf(0, 36, 0.08) == pytest.approx(0.92**36, rel=1e-12)
    with pytest.raises(ValueError):
        binom_cdf(1, 10, -0.1)
    with pytest.raises(ValueError):
        binom_cdf(1, 10, 1.1)
    with pytest.raises(ValueError):
        binom_cdf(1, -1, 0.5)


def test_binom_cdf_monotone() -> None:
    for p in (0.02, 0.08, 0.5):
        vals = [binom_cdf(k, 80, p) for k in range(-1, 82)]
        assert all(a <= b for a, b in zip(vals, vals[1:], strict=False))
        assert vals[0] == 0.0 and vals[-1] == 1.0
    for k, n in ((0, 40), (3, 40), (20, 40)):
        vals = [binom_cdf(k, n, i / 100) for i in range(101)]
        assert all(a >= b for a, b in zip(vals, vals[1:], strict=False))


def test_binom_cdf_large_n_is_stable() -> None:
    # Reflection identity P(X <= k; p) + P(Y <= n - k - 1; 1 - p) = 1 at n = 5000.
    for p in (0.002, 0.08, 0.5, 0.93):
        for k in (0, 7, 100, 400, 2500, 4900):
            a = binom_cdf(k, 5000, p)
            b = binom_cdf(5000 - k - 1, 5000, 1 - p)
            assert 0.0 <= a <= 1.0
            assert a + b == pytest.approx(1.0, abs=1e-9)
    # Closed forms: all-zero outcome, and the symmetric median of Binomial(5000, 1/2).
    assert binom_cdf(0, 5000, 0.01) == pytest.approx(0.99**5000, rel=1e-9)
    assert binom_cdf(2499, 5000, 0.5) == pytest.approx(
        (1 - math.comb(5000, 2500) / 2**5000) / 2, rel=1e-9
    )
    # Underflowing tails return 0.0 rather than raising or going negative.
    assert binom_cdf(0, 5000, 0.5) == 0.0
    assert binom_cdf(4999, 5000, 0.5) == pytest.approx(1.0)
    # A tail that is representable stays accurate (n = 3000, p = 0.3, k far below the mean 900).
    assert binom_cdf(700, 3000, 0.3) == pytest.approx(
        float(
            sum(
                Fraction(math.comb(3000, i)) * Fraction(3, 10) ** i * Fraction(7, 10) ** (3000 - i)
                for i in range(701)
            )
        ),
        rel=1e-9,
    )


# ---------------------------------------------------------------------------- Clopper-Pearson


def test_clopper_pearson_known_values() -> None:
    assert clopper_pearson_upper(0, 36, 0.95) == pytest.approx(0.0799, abs=1e-4)
    for n in (1, 5, 36, 100, 800):
        for conf in (0.8, 0.95, 0.99):
            assert clopper_pearson_upper(0, n, conf) == pytest.approx(
                1 - (1 - conf) ** (1 / n), rel=1e-9
            )
    # Rule of three: 0 errors in 100 -> about 3 / 100.
    assert clopper_pearson_upper(0, 100, 0.95) == pytest.approx(0.0295, abs=1e-4)


def test_clopper_pearson_defining_property() -> None:
    for e, n in ((1, 20), (3, 100), (10, 100), (40, 500), (0, 7)):
        for conf in (0.9, 0.95):
            u = clopper_pearson_upper(e, n, conf)
            assert binom_cdf(e, n, u) == pytest.approx(1 - conf, rel=1e-6)
            assert u > e / n
    # Larger n with the same rate tightens the bound; more errors loosens it.
    assert clopper_pearson_upper(5, 100) > clopper_pearson_upper(50, 1000)
    assert clopper_pearson_upper(6, 100) > clopper_pearson_upper(5, 100)


def test_clopper_pearson_degenerate_and_validation() -> None:
    assert clopper_pearson_upper(0, 0) == 1.0
    assert clopper_pearson_upper(4, 4) == 1.0
    for bad in (0.0, 1.0, -0.2, 1.5):
        with pytest.raises(ValueError):
            clopper_pearson_upper(1, 10, bad)
    with pytest.raises(ValueError):
        clopper_pearson_upper(11, 10)
    with pytest.raises(ValueError):
        clopper_pearson_upper(-1, 10)


# ---------------------------------------------------------------------------- certify


def test_certify_validation() -> None:
    ok = {"alpha": 0.1, "delta": 0.05, "grid": [0.9, 0.5]}
    s, c = [0.9, 0.6], [True, False]
    with pytest.raises(ValueError, match="strictly decreasing"):
        certify(s, c, alpha=0.1, delta=0.05, grid=[0.5, 0.9])
    with pytest.raises(ValueError, match="strictly decreasing"):
        certify(s, c, alpha=0.1, delta=0.05, grid=[0.9, 0.9])
    with pytest.raises(ValueError, match="empty"):
        certify(s, c, alpha=0.1, delta=0.05, grid=[])
    with pytest.raises(ValueError, match="mismatch"):
        certify([0.9], c, **ok)
    for alpha in (0.0, 1.0, -0.1, 2.0):
        with pytest.raises(ValueError, match="alpha"):
            certify(s, c, alpha=alpha, delta=0.05, grid=[0.9])
    for delta in (0.0, 1.0, -0.1):
        with pytest.raises(ValueError, match="delta"):
            certify(s, c, alpha=0.1, delta=delta, grid=[0.9])
    with pytest.raises(ValueError):
        certify([math.nan, 0.5], c, **ok)
    with pytest.raises(ValueError):
        certify(s, c, alpha=0.1, delta=0.05, grid=[math.inf, 0.5])


def test_certify_empty_selection_stops() -> None:
    # Every score is below the first threshold: n = 0, p = 1, no rejection, the loop stops.
    r = certify([0.1, 0.2, 0.3], [True, True, True], alpha=0.1, delta=0.05, grid=[0.9, 0.5, 0.1])
    assert r.lambda_hat is None and r.coverage == 0.0 and r.selective_risk is None
    assert len(r.steps) == 1
    step = r.steps[0]
    assert (step.n_selected, step.n_errors, step.p_value, step.rejected, step.cp_upper) == (
        0,
        0,
        1.0,
        False,
        1.0,
    )
    # No calibration data at all is also "nothing certified", not a crash.
    r = certify([], [], alpha=0.1, delta=0.05, grid=[0.5])
    assert r.lambda_hat is None and r.coverage == 0.0


def test_certify_all_wrong_is_not_certified() -> None:
    scores = [i / 200 for i in range(200)]
    r = certify(scores, [False] * 200, alpha=0.1, delta=0.05, grid=[0.9, 0.5, 0.0])
    assert r.lambda_hat is None and r.coverage == 0.0 and r.selective_risk is None
    assert len(r.steps) == 1 and not r.steps[0].rejected
    assert r.steps[0].n_errors == r.steps[0].n_selected > 0
    assert r.steps[0].p_value == pytest.approx(
        binom_cdf(r.steps[0].n_selected, r.steps[0].n_selected, 0.1)
    )


def test_certify_hand_built_steps() -> None:
    alpha, delta = 0.08, 0.05
    # 40 items at 0.9 (all right), 40 at 0.7 (1 wrong), 20 at 0.5 (10 wrong).
    scores = [0.9] * 40 + [0.7] * 40 + [0.5] * 20
    correct = [True] * 40 + [False] * 1 + [True] * 39 + [False] * 10 + [True] * 10
    r = certify(scores, correct, alpha=alpha, delta=delta, grid=[0.95, 0.9, 0.7, 0.5, 0.3])
    # 0.95: nothing selected; the very first step stops everything.
    assert [s.threshold for s in r.steps] == [0.95]
    r = certify(scores, correct, alpha=alpha, delta=delta, grid=[0.9, 0.7, 0.5, 0.3])
    assert [s.threshold for s in r.steps] == [0.9, 0.7, 0.5]  # 0.3 is never tested
    s1, s2, s3 = r.steps
    assert (s1.n_selected, s1.n_errors) == (40, 0)
    assert s1.p_value == pytest.approx(0.92**40, rel=1e-12) and s1.rejected
    assert (s2.n_selected, s2.n_errors) == (80, 1)
    want = sum(math.comb(80, i) * 0.08**i * 0.92 ** (80 - i) for i in range(2))
    assert s2.p_value == pytest.approx(want, rel=1e-12) and s2.rejected  # about 0.010 <= 0.05
    assert (s3.n_selected, s3.n_errors) == (100, 11)
    assert s3.p_value > delta and not s3.rejected
    assert r.lambda_hat == 0.7
    assert r.coverage == pytest.approx(0.8)
    assert r.selective_risk == pytest.approx(1 / 80)
    for step in r.steps:
        assert step.rejected == (step.cp_upper <= alpha)


def test_certify_selection_includes_ties_at_threshold() -> None:
    r = certify([0.5] * 40, [True] * 40, alpha=0.08, delta=0.05, grid=[0.5])
    assert r.steps[0].n_selected == 40 and r.lambda_hat == 0.5 and r.coverage == 1.0


def test_certify_needs_36_clean_items_at_alpha_008() -> None:
    # 0.92^35 = 0.0540 > 0.05 >= 0.92^36 = 0.0498: the smallest first step that can certify.
    for n, expect in ((35, False), (36, True)):
        r = certify([0.9] * n + [0.1] * 400, [True] * (n + 400), alpha=0.08, delta=0.05, grid=[0.9])
        assert r.steps[0].n_selected == n and r.steps[0].rejected is expect


def test_certify_is_fixed_sequence_it_does_not_skip_ahead() -> None:
    # Step 2 fails (too many errors in the 0.6 stratum) but step 3 would pass on its own
    # (the 0.3 stratum is spotless); a fixed-sequence test must stop at step 2.
    scores = [0.9] * 60 + [0.6] * 60 + [0.3] * 400
    correct = [True] * 60 + [True] * 50 + [False] * 10 + [True] * 400
    r = certify(scores, correct, alpha=0.08, delta=0.05, grid=[0.9, 0.6, 0.3])
    assert [s.rejected for s in r.steps] == [True, False]
    assert r.lambda_hat == 0.9 and r.coverage == pytest.approx(60 / 520)
    lone = certify(scores, correct, alpha=0.08, delta=0.05, grid=[0.3])
    assert lone.lambda_hat == 0.3  # on its own the lowest threshold would have been accepted


def test_rejected_iff_cp_upper_at_most_alpha() -> None:
    rng = random.Random(7)
    grid = [round(0.95 - 0.05 * i, 2) for i in range(18)]
    for _ in range(40):
        a = rng.choice([0.05, 0.08, 0.15])
        d = rng.choice([0.05, 0.1])
        scores = [rng.random() for _ in range(300)]
        correct = [rng.random() >= 0.12 * (1 - s) for s in scores]
        r = certify(scores, correct, alpha=a, delta=d, grid=grid)
        for step in r.steps:
            assert step.rejected == (step.p_value <= d)
            if abs(step.cp_upper - a) > 1e-9:
                assert step.rejected == (step.cp_upper <= a)


def test_cert_result_to_json_roundtrips_through_json() -> None:
    r = certify(
        [0.9] * 50 + [0.4] * 50, [True] * 50 + [False] * 50, alpha=0.08, delta=0.05, grid=[0.9, 0.4]
    )
    blob = json.loads(json.dumps(r.to_json(), allow_nan=False))
    assert blob["lambda_hat"] == 0.9 and blob["coverage"] == 0.5 and blob["selective_risk"] == 0.0
    assert [s["rejected"] for s in blob["steps"]] == [True, False]
    assert set(blob["steps"][0]) == {
        "threshold",
        "n_selected",
        "n_errors",
        "p_value",
        "rejected",
        "cp_upper",
    }
    none = certify([0.1], [True], alpha=0.1, delta=0.05, grid=[0.9])
    assert isinstance(none, CertResult)
    assert json.loads(json.dumps(none.to_json()))["lambda_hat"] is None


# ---------------------------------------------------------------------------- grid + curve


def test_coverage_grid_levels_and_tie_rule() -> None:
    fit = [i / 100 for i in range(1, 101)]  # 0.01 .. 1.00
    g = coverage_grid(fit, [0.2, 0.5, 0.9])
    assert g == [0.81, 0.51, 0.11]  # k-th largest with k = 20, 50, 90
    assert coverage_grid(fit, [0.07]) == [0.94]  # 0.07 * 100 = 7.000000000000001: k must stay 7
    assert coverage_grid(fit, [1.0]) == [0.01]
    # Ties: 10 copies of each of 0.1..1.0. Levels 0.2/0.25/0.3 all hit the 0.9 / 0.8 blocks.
    tied = [round(0.1 * j, 1) for j in range(1, 11) for _ in range(10)]
    g = coverage_grid(tied, [0.05, 0.1, 0.15, 0.2, 0.25, 0.3])
    assert g == [1.0, 0.9, 0.8]  # duplicates dropped, most conservative level of a tie kept
    assert all(a > b for a, b in zip(g, g[1:], strict=False))
    # Realized fit coverage is at least the requested level.
    for lv, t in zip([0.05, 0.15, 0.25], [1.0, 0.9, 0.8], strict=True):
        assert sum(1 for s in tied if s >= t) / len(tied) >= lv


def test_coverage_grid_default_style_levels_are_strictly_decreasing() -> None:
    rng = random.Random(3)
    fit = [rng.random() for _ in range(1000)]
    levels = [0.20 + 0.02 * i for i in range(36)]  # drifting floats on purpose
    g = coverage_grid(fit, levels)
    assert len(g) == 36 and all(a > b for a, b in zip(g, g[1:], strict=False))
    assert sum(1 for s in fit if s >= g[0]) == 200
    assert sum(1 for s in fit if s >= g[-1]) >= 900


def test_coverage_grid_validation() -> None:
    with pytest.raises(ValueError):
        coverage_grid([], [0.5])
    with pytest.raises(ValueError):
        coverage_grid([0.5], [])
    with pytest.raises(ValueError):
        coverage_grid([0.5, 0.6], [0.5, 0.3])  # must be increasing
    with pytest.raises(ValueError):
        coverage_grid([0.5, 0.6], [0.3, 0.3])
    with pytest.raises(ValueError):
        coverage_grid([0.5, 0.6], [0.0, 0.5])
    with pytest.raises(ValueError):
        coverage_grid([0.5, 0.6], [0.5, 1.5])
    with pytest.raises(ValueError):
        coverage_grid([math.nan, 0.6], [0.5])


def test_risk_coverage_curve() -> None:
    scores = [0.9, 0.9, 0.8, 0.5, 0.2]
    correct = [True, False, True, True, False]
    curve = risk_coverage_curve(scores, correct, [0.95, 0.9, 0.5, 0.0])
    assert curve[0] == {
        "threshold": 0.95,
        "coverage": 0.0,
        "n_selected": 0,
        "n_errors": 0,
        "risk": None,
    }
    assert curve[1] == {
        "threshold": 0.9,
        "coverage": 0.4,
        "n_selected": 2,
        "n_errors": 1,
        "risk": 0.5,
    }
    assert curve[2]["n_selected"] == 4 and curve[2]["n_errors"] == 1 and curve[2]["risk"] == 0.25
    assert curve[3]["coverage"] == 1.0 and curve[3]["risk"] == 0.4
    json.dumps(curve, allow_nan=False)
    assert risk_coverage_curve([], [], [0.5]) == [
        {"threshold": 0.5, "coverage": 0.0, "n_selected": 0, "n_errors": 0, "risk": None}
    ]
    with pytest.raises(ValueError):
        risk_coverage_curve([0.1], [True, False], [0.5])


# ---------------------------------------------------------------------------- simulations

ALPHA, DELTA = 0.08, 0.05


def _draw_profile(rng: random.Random, n: int) -> tuple[list[float], list[bool]]:
    """Score s ~ U(0,1); P(error | s) = 0.6 (1 - s)^2, so the true selective risk of 'score >= t'
    is E[0.6 (1-s)^2 | s >= t] = 0.2 (1 - t)^2 (below ALPHA from t = 1 - sqrt(0.4) = 0.368 up)."""
    s = [rng.random() for _ in range(n)]
    return s, [rng.random() >= 0.6 * (1 - x) ** 2 for x in s]


def _true_risk(t: float) -> float:
    return 0.2 * (1 - t) ** 2


def test_validity_simulation_selective_risk_guarantee() -> None:
    seeds = 300
    grid = [round(0.90 - 0.01 * i, 2) for i in range(71)]  # fixed in advance of every draw
    bound = DELTA + 3 * math.sqrt(DELTA * (1 - DELTA) / seeds)
    violations = certified = naive_violations = 0
    for seed in range(seeds):
        scores, correct = _draw_profile(random.Random(seed), 500)
        r = certify(scores, correct, alpha=ALPHA, delta=DELTA, grid=grid)
        if r.lambda_hat is not None:
            certified += 1
            violations += _true_risk(r.lambda_hat) > ALPHA
        # Negative control: the lowest threshold whose EMPIRICAL risk is <= alpha, no test, no stop.
        curve = risk_coverage_curve(scores, correct, grid)
        ok = [pt["threshold"] for pt in curve if pt["risk"] is not None and pt["risk"] <= ALPHA]  # type: ignore[operator]
        naive_violations += bool(ok) and _true_risk(float(ok[-1])) > ALPHA  # type: ignore[arg-type]
    assert violations / seeds <= bound
    # Not vacuous: the test certifies most of the time, and the naive rule breaks the guarantee.
    assert certified / seeds >= 0.7
    assert naive_violations / seeds > 3 * bound


def test_validity_simulation_flat_null_fwer() -> None:
    # Every threshold is a true null (error rate 0.081 > alpha, scores uninformative); the first
    # step selects ~35 of 500 items. Rejecting anywhere is a false certification: rate <= delta.
    seeds = 1000
    grid = [round(0.93 - 0.01 * i, 2) for i in range(60)]
    bound = DELTA + 3 * math.sqrt(DELTA * (1 - DELTA) / seeds)
    false_cert = 0
    for seed in range(seeds):
        rng = random.Random(10_000 + seed)
        scores = [rng.random() for _ in range(500)]
        correct = [rng.random() >= 0.081 for _ in scores]
        false_cert += (
            certify(scores, correct, alpha=ALPHA, delta=DELTA, grid=grid).lambda_hat is not None
        )
    assert false_cert / seeds <= bound


def test_grid_trap_two_percent_start_never_certifies() -> None:
    # alpha = 0.08 needs >= 36 clean items for the first step; 2% of 500 is ~10, so a grid that
    # starts at fit-coverage 2% stops at step 1 with coverage 0 -- while one that starts at 20% works.
    fit_scores, _ = _draw_profile(random.Random(99), 2000)
    low_start = coverage_grid(fit_scores, [0.02 + 0.02 * i for i in range(45)])
    high_start = coverage_grid(fit_scores, [0.20 + 0.02 * i for i in range(36)])
    ok_high = 0
    for seed in range(60):
        scores, correct = _draw_profile(random.Random(500 + seed), 500)
        bad = certify(scores, correct, alpha=ALPHA, delta=DELTA, grid=low_start)
        assert bad.lambda_hat is None and bad.coverage == 0.0
        assert len(bad.steps) == 1 and bad.steps[0].n_selected < 36 and not bad.steps[0].rejected
        good = certify(scores, correct, alpha=ALPHA, delta=DELTA, grid=high_start)
        ok_high += good.lambda_hat is not None and good.coverage > 0
    assert ok_high >= 57
    scores, correct = _draw_profile(random.Random(500), 500)  # the pinned scenario
    assert certify(scores, correct, alpha=ALPHA, delta=DELTA, grid=high_start).coverage > 0.2


def test_clopper_pearson_lower_known_values_and_duality() -> None:
    # all successes: p^n = 1 - conf
    assert clopper_pearson_lower(10, 10, 0.95) == pytest.approx(0.05 ** (1 / 10), abs=1e-9)
    assert clopper_pearson_lower(0, 10) == 0.0 and clopper_pearson_lower(0, 0) == 0.0
    # one-sided 95% lower bound for 8/10 (= two-sided 90% CP limit)
    assert clopper_pearson_lower(8, 10, 0.95) == pytest.approx(0.4931, abs=1e-4)
    for k, n in [(1, 5), (7, 20), (80, 100), (99, 100)]:
        lo = clopper_pearson_lower(k, n, 0.95)
        # defining property: P(X >= k | lo) = 0.05
        assert 1 - binom_cdf(k - 1, n, lo) == pytest.approx(0.05, abs=1e-9)
        # duality with the upper bound on the failure count
        assert lo == pytest.approx(1 - clopper_pearson_upper(n - k, n, 0.95), abs=1e-9)
    with pytest.raises(ValueError):
        clopper_pearson_lower(3, 2)
    with pytest.raises(ValueError):
        clopper_pearson_lower(1, 2, 1.0)
