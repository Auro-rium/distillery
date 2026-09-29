# ruff: noqa: S101, S105, S106
import pytest

from distillery.stats import mcnemar_exact, paired_bootstrap_ratio_ci, wilson_ci


def test_mcnemar_10_vs_2() -> None:
    a = [True] * 10 + [False] * 2 + [True] * 5
    b = [False] * 10 + [True] * 2 + [True] * 5
    r = mcnemar_exact(a, b)
    assert (r.a_only, r.b_only, r.n_discordant) == (10, 2, 12)
    assert r.p_value == pytest.approx(158 / 4096)  # 2*(1+12+66)/2^12


def test_mcnemar_symmetric_and_equal() -> None:
    a = [True, False, True, False]
    b = [False, True, True, False]
    r = mcnemar_exact(a, b)
    assert r.p_value == 1.0 and (r.a_only, r.b_only) == (1, 1)
    r = mcnemar_exact([True, False], [True, False])
    assert r.p_value == 1.0 and r.n_discordant == 0


def test_mcnemar_extreme() -> None:
    r = mcnemar_exact([True] * 8, [False] * 8)
    assert r.p_value == pytest.approx(2 / 256)


def test_validation() -> None:
    with pytest.raises(ValueError):
        mcnemar_exact([], [])
    with pytest.raises(ValueError):
        mcnemar_exact([True], [True, False])
    with pytest.raises(ValueError):
        paired_bootstrap_ratio_ci([], [])
    with pytest.raises(ValueError):
        paired_bootstrap_ratio_ci([True], [True, True])
    with pytest.raises(ValueError):
        wilson_ci(1, 0)
    with pytest.raises(ValueError):
        wilson_ci(5, 4)


def test_wilson_known() -> None:
    lo, hi = wilson_ci(50, 100)
    assert lo == pytest.approx(0.4038, abs=1e-3) and hi == pytest.approx(0.5962, abs=1e-3)
    lo, hi = wilson_ci(0, 10)
    assert lo == pytest.approx(0.0) and hi == pytest.approx(0.2775, abs=1e-3)
    lo, hi = wilson_ci(10, 10)
    assert hi == pytest.approx(1.0) and lo == pytest.approx(0.7225, abs=1e-3)


def test_bootstrap_identical_is_one() -> None:
    x = [True, False] * 20
    r = paired_bootstrap_ratio_ci(x, x, resamples=500, seed=1)
    assert r.point == 1.0 and r.lo == 1.0 and r.hi == 1.0


def test_bootstrap_point_and_determinism() -> None:
    s = [True] * 8 + [False] * 12
    t = [True] * 16 + [False] * 4
    r1 = paired_bootstrap_ratio_ci(s, t, resamples=2000, seed=3)
    r2 = paired_bootstrap_ratio_ci(s, t, resamples=2000, seed=3)
    assert r1 == r2
    assert r1.point == pytest.approx(0.5)
    assert r1.lo < r1.point < r1.hi
    assert r1.resamples_used + r1.skipped_teacher_zero == 2000


def test_bootstrap_teacher_zero_in_resample_counted() -> None:
    t = [True] + [False] * 4
    s = [True] + [False] * 4
    r = paired_bootstrap_ratio_ci(s, t, resamples=1000, seed=1)
    assert r.skipped_teacher_zero > 0
    assert r.resamples_used == 1000 - r.skipped_teacher_zero


def test_bootstrap_teacher_zero_overall_raises() -> None:
    with pytest.raises(ValueError, match="teacher"):
        paired_bootstrap_ratio_ci([True, False], [False, False])
