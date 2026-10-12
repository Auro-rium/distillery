# ruff: noqa: S101
from __future__ import annotations

import dataclasses
import json
import math
import random

import pytest

from distillery.router import (
    FEATURES,
    RouteDecision,
    RouterModel,
    _solve,
    decide,
    fit_logistic,
    max_score_given_sim,
    precheck,
)


def hand_model(**over: object) -> RouterModel:
    base: dict[str, object] = {
        "feature_names": FEATURES,
        "means": (0.8, -0.5, 0.5),
        "stds": (0.4, 0.3, 0.2),
        "weights": (1.5, 1.0, 2.0),
        "bias": 0.3,
        "l2": 0.01,
        "n_fit": 100,
        "fit_sha256": "0" * 64,
    }
    base.update(over)
    return RouterModel(**base)  # type: ignore[arg-type]


def sig(exec_ok: float, lp: float, sim: float) -> dict[str, float]:
    return {"exec_ok": exec_ok, "mean_logprob": lp, "nn_sim": sim}


def sigmoid(x: float) -> float:
    return 1 / (1 + math.exp(-x))


# ---------------------------------------------------------------------------- model + score


def test_score_standardizes_and_applies_logistic() -> None:
    m = hand_model()
    # z = (0, 0, 0): the score is sigmoid(bias).
    assert m.score(sig(0.8, -0.5, 0.5)) == pytest.approx(sigmoid(0.3))
    # exec_ok = 1 -> z = 0.5, lp = -0.2 -> z = 1, sim = 0.9 -> z = 2.
    want = sigmoid(0.3 + 1.5 * 0.5 + 1.0 * 1.0 + 2.0 * 2.0)
    assert m.score(sig(1.0, -0.2, 0.9)) == pytest.approx(want)
    assert m.contributions(sig(1.0, -0.2, 0.9)) == pytest.approx((0.75, 1.0, 4.0))
    assert m.score({**sig(1.0, -0.2, 0.9), "ignored_extra": 5.0}) == pytest.approx(want)
    # Extreme logits do not overflow.
    assert m.score(sig(1.0, 1e6, 1.0)) == 1.0
    assert 0.0 <= m.score(sig(0.0, -1e6, -1.0)) < 1e-300


def test_zero_std_is_treated_as_one() -> None:
    m = hand_model(stds=(0.4, 0.0, 0.2))
    z_lp = (-0.2 - -0.5) / 1.0
    assert m.score(sig(1.0, -0.2, 0.9)) == pytest.approx(sigmoid(0.3 + 0.75 + z_lp + 4.0))


def test_model_json_roundtrip_and_sha() -> None:
    m = hand_model()
    blob = json.loads(json.dumps(m.to_json(), allow_nan=False))
    back = RouterModel.from_json(blob)
    assert back == m and back.sha256() == m.sha256()
    assert len(m.sha256()) == 64
    assert dataclasses.replace(m, bias=0.31).sha256() != m.sha256()
    assert dataclasses.replace(m, fit_sha256="1" * 64).sha256() != m.sha256()
    # Key order of the input must not matter (canonical JSON).
    reordered = RouterModel.from_json(dict(reversed(list(m.to_json().items()))))
    assert reordered.sha256() == m.sha256()


def test_model_validation() -> None:
    with pytest.raises(ValueError):
        hand_model(means=(0.8, -0.5))
    with pytest.raises(ValueError):
        hand_model(weights=(1.0, math.nan, 2.0))
    with pytest.raises(ValueError):
        hand_model(feature_names=("a", "a", "b"))


# ---------------------------------------------------------------------------- linear solve


def test_solve_with_partial_pivoting() -> None:
    a = [[0.0, 2.0, 1.0], [1.0, 1.0, 1.0], [2.0, 1.0, 0.0]]  # zero leading pivot
    x_true = [1.0, -2.0, 3.0]
    b = [sum(r * x for r, x in zip(row, x_true, strict=True)) for row in a]
    assert _solve(a, b) == pytest.approx(x_true)
    assert a[0][0] == 0.0  # inputs untouched
    tiny = [[1e-12, 1.0], [1.0, 1.0]]
    assert _solve(tiny, [1.0, 2.0]) == pytest.approx([1.0, 1.0], rel=1e-9)
    with pytest.raises(ValueError, match="singular"):
        _solve([[1.0, 2.0], [2.0, 4.0]], [1.0, 2.0])


# ---------------------------------------------------------------------------- fit_logistic


def simulate(
    n: int, beta: tuple[float, float, float], intercept: float, seed: int
) -> tuple[list[dict[str, float]], list[bool]]:
    rng = random.Random(seed)
    rows: list[dict[str, float]] = []
    labels: list[bool] = []
    for _ in range(n):
        e = 1.0 if rng.random() < 0.85 else 0.0
        lp = -math.exp(rng.gauss(-1.5, 0.6))
        sim = min(1.0, max(-1.0, rng.gauss(0.6, 0.15)))
        eta = intercept + beta[0] * e + beta[1] * lp + beta[2] * sim
        rows.append(sig(e, lp, sim))
        labels.append(rng.random() < sigmoid(eta))
    return rows, labels


def test_fit_recovers_sign_and_ordering_of_true_coefficients() -> None:
    beta = (2.0, 3.0, 6.0)  # standardized effects (beta * sd) ~ 0.72, 0.5, 0.9: lp < exec < sim
    rows, labels = simulate(5000, beta, -4.5, seed=11)
    m = fit_logistic(rows, labels)
    assert m.feature_names == FEATURES and m.n_fit == 5000
    # Compare in standardized units: true effect of a one-sd move is beta_j * sd_j.
    true_std_effect = [b * s for b, s in zip(beta, m.stds, strict=True)]
    assert all(w > 0 for w in m.weights)
    order = lambda v: sorted(range(3), key=lambda j: v[j])  # noqa: E731
    assert order(m.weights) == order(true_std_effect)
    for w, t in zip(m.weights, true_std_effect, strict=True):
        assert w == pytest.approx(t, rel=0.3)
    # Calibrated overall: the mean score tracks the base rate.
    mean_score = sum(m.score(r) for r in rows) / len(rows)
    assert mean_score == pytest.approx(sum(labels) / len(labels), abs=0.01)


def test_fit_recovers_negative_and_null_coefficients() -> None:
    rows, labels = simulate(6000, (1.5, 0.0, -5.0), 2.5, seed=5)
    m = fit_logistic(rows, labels)
    assert m.weights[0] > 0 and m.weights[2] < 0
    assert abs(m.weights[1]) < 0.25 * abs(m.weights[2])


def test_fit_separable_data_stays_finite() -> None:
    rows = [sig(float(i % 2), -0.1 - 0.001 * (i % 7), 0.5) for i in range(200)]
    labels = [bool(i % 2) for i in range(200)]  # exec_ok predicts the label perfectly
    m = fit_logistic(rows, labels)
    assert all(math.isfinite(v) for v in (*m.weights, m.bias))
    assert max(abs(w) for w in m.weights) < 1e3
    assert min(m.score(r) for r, y in zip(rows, labels, strict=True) if y) > 0.95
    assert max(m.score(r) for r, y in zip(rows, labels, strict=True) if not y) < 0.05
    # Separable on a continuous feature, many rows: still converges inside max_iter.
    rng = random.Random(2)
    rows2 = [sig(1.0, -rng.random(), rng.random()) for _ in range(1500)]
    labels2 = [r["nn_sim"] > 0.5 for r in rows2]
    m2 = fit_logistic(rows2, labels2)
    assert all(math.isfinite(v) for v in (*m2.weights, m2.bias))
    assert m2.weights[2] > 0 and max(abs(w) for w in m2.weights) < 1e3
    assert m2.score(sig(1.0, -0.5, 0.95)) > 0.99 and m2.score(sig(1.0, -0.5, 0.05)) < 0.01


def test_fit_constant_feature_is_harmless() -> None:
    rows, labels = simulate(1500, (2.0, 1.0, 6.0), -4.5, seed=3)
    for r in rows:
        r["nn_sim"] = 0.1  # constant
    m = fit_logistic(rows, labels)
    assert m.stds[2] == 0.0 and m.weights[2] == 0.0
    assert m.score(sig(1.0, -0.2, 0.1)) == m.score(sig(1.0, -0.2, 0.9))
    assert all(math.isfinite(w) for w in m.weights) and m.weights[0] > 0 and m.weights[1] > 0
    # All three constant except the label: weights all zero, bias = base-rate logit.
    flat = [sig(1.0, -0.2, 0.5)] * 40
    m2 = fit_logistic(flat, [i < 10 for i in range(40)])
    assert m2.weights == (0.0, 0.0, 0.0)
    assert m2.bias == pytest.approx(math.log(0.25 / 0.75), abs=1e-9)


def test_fit_is_deterministic_and_data_bound() -> None:
    rows, labels = simulate(800, (2.0, 1.0, 6.0), -4.5, seed=4)
    a = fit_logistic(rows, labels)
    b = fit_logistic([dict(r) for r in rows], list(labels))
    assert a == b and a.sha256() == b.sha256() and a.fit_sha256 == b.fit_sha256
    assert len(a.fit_sha256) == 64
    c = fit_logistic(rows[:-1], labels[:-1])
    assert c.fit_sha256 != a.fit_sha256
    flipped = [*labels[:-1], not labels[-1]]
    assert fit_logistic(rows, flipped).fit_sha256 != a.fit_sha256
    # Integer/bool inputs are accepted and hash like their float values.
    ints = [{k: (int(v) if k == "exec_ok" else v) for k, v in r.items()} for r in rows]
    assert fit_logistic(ints, labels).fit_sha256 == a.fit_sha256
    # l2 shrinks weights.
    strong = fit_logistic(rows, labels, l2=500.0)
    assert sum(w * w for w in strong.weights) < sum(w * w for w in a.weights)


def test_fit_is_a_penalized_optimum() -> None:
    # At the optimum the penalized gradient is ~0: sum (p - y) z + l2 w = 0 and sum (p - y) = 0.
    rows, labels = simulate(1200, (1.0, 0.5, 4.0), -3.0, seed=8)
    m = fit_logistic(rows, labels, l2=0.5)
    g0 = 0.0
    g = [0.0, 0.0, 0.0]
    for r, y in zip(rows, labels, strict=True):
        resid = m.score(r) - (1.0 if y else 0.0)
        g0 += resid
        for j, name in enumerate(FEATURES):
            g[j] += resid * (r[name] - m.means[j]) / m.stds[j]
    assert abs(g0) < 1e-6
    for j in range(3):
        assert g[j] + 0.5 * m.weights[j] == pytest.approx(0.0, abs=1e-6)


def test_fit_validation() -> None:
    rows, labels = simulate(50, (1.0, 1.0, 1.0), 0.0, seed=1)
    with pytest.raises(ValueError, match="both classes"):
        fit_logistic(rows, [True] * 50)
    with pytest.raises(ValueError, match="both classes"):
        fit_logistic(rows, [False] * 50)
    with pytest.raises(ValueError, match="mismatch"):
        fit_logistic(rows, labels[:-1])
    with pytest.raises(ValueError, match="l2"):
        fit_logistic(rows, labels, l2=-1.0)
    with pytest.raises(ValueError, match="max_iter"):
        fit_logistic(rows, labels, max_iter=0)
    bad = [dict(r) for r in rows]
    bad[3]["nn_sim"] = math.nan
    with pytest.raises(ValueError, match="finite"):
        fit_logistic(bad, labels)


# ---------------------------------------------------------------------------- decide


def test_decide_not_certified_goes_to_teacher() -> None:
    d = decide(hand_model(), None, sig(1.0, -0.1, 0.9))
    assert d == RouteDecision(
        "teacher", None, None, "not certified: all traffic goes to the teacher", False
    )


def test_decide_certified_student() -> None:
    m = hand_model()
    d = decide(m, 0.9, sig(1.0, -0.1, 0.8))
    assert d.served_by == "student" and d.short_circuit is False and d.threshold == 0.9
    assert d.score == pytest.approx(m.score(sig(1.0, -0.1, 0.8)))
    assert d.reason == "certified: confidence 1.00 ≥ 0.90"
    # Exactly at the threshold counts as certified (score >= lambda).
    s = sig(1.0, -0.3, 0.6)
    assert decide(m, m.score(s), s).served_by == "student"


def test_decide_reasons_name_the_most_negative_contribution() -> None:
    m = hand_model()
    d = decide(m, 0.9, sig(0.0, -0.5, 0.5))  # exec: -3.0, others 0
    assert (d.served_by, d.reason) == ("teacher", "student SQL failed to run → teacher")
    d = decide(m, 0.9, sig(1.0, -0.5, 0.1))  # sim: -4.0, exec +0.75
    assert d.reason == "unfamiliar question pattern → teacher"
    d = decide(m, 0.9, sig(1.0, -1.4, 0.5))  # logprob: -3.0
    assert d.reason == "low student confidence → teacher"
    assert d.score is not None and d.score < 0.9 and d.threshold == 0.9 and not d.short_circuit
    # Several negative contributions: the most negative one wins.
    d = decide(m, 0.9, sig(0.0, -1.1, 0.4))  # exec -3.0, lp -2.0, sim -1.0
    assert d.reason == "student SQL failed to run → teacher"
    # Tie: FEATURES order (exec_ok before mean_logprob).
    tie = hand_model(
        means=(0.0, 0.0, 0.0), stds=(1.0, 1.0, 1.0), weights=(1.0, 1.0, 1.0), bias=-5.0
    )
    assert decide(tie, 0.9, sig(-1.0, -1.0, 0.0)).reason == "student SQL failed to run → teacher"
    # Nothing negative at all (the bias alone is low): still names the smallest contribution.
    low_bias = hand_model(bias=-10.0)
    assert decide(low_bias, 0.9, sig(1.0, -0.5, 0.5)).reason == "low student confidence → teacher"


# ---------------------------------------------------------------------------- precheck


def test_max_score_given_sim_is_the_best_case() -> None:
    m = hand_model()
    for sim in (-1.0, 0.0, 0.3, 0.5, 1.0):
        best = max_score_given_sim(m, sim)
        assert best == m.score(sig(1.0, 0.0, sim))  # exec_ok = 1, logprob = 0
        for e in (0.0, 1.0):
            for lp in (0.0, -1e-9, -0.01, -0.5, -3.0, -100.0):
                assert m.score(sig(e, lp, sim)) <= best  # type: ignore[operator]
    # exec_ok weight negative -> the best case is exec_ok = 0.
    neg_exec = hand_model(weights=(-1.5, 1.0, 2.0))
    assert max_score_given_sim(neg_exec, 0.4) == neg_exec.score(sig(0.0, 0.0, 0.4))


def test_max_score_is_none_when_logprob_cannot_be_bounded() -> None:
    assert max_score_given_sim(hand_model(weights=(1.5, -1.0, 2.0)), 0.5) is None
    assert max_score_given_sim(hand_model(weights=(1.5, 0.0, 2.0)), 0.5) is None
    # Original-unit weight is w / std: a std of 0 counts as 1, the sign is unchanged.
    assert (
        max_score_given_sim(hand_model(stds=(0.4, 0.0, 0.2), weights=(1.5, 1.0, 2.0)), 0.5)
        is not None
    )
    assert (
        max_score_given_sim(hand_model(stds=(0.4, 0.0, 0.2), weights=(1.5, -1.0, 2.0)), 0.5) is None
    )
    with pytest.raises(ValueError):
        max_score_given_sim(hand_model(), math.nan)


def test_precheck_short_circuits_unfamiliar_questions() -> None:
    m = hand_model()
    d = precheck(m, 0.9, 0.1)
    assert d is not None
    assert (d.served_by, d.score, d.threshold, d.short_circuit) == ("teacher", None, 0.9, True)
    assert d.reason == "unfamiliar question pattern → teacher"
    assert precheck(m, 0.9, 0.8) is None  # best case 0.99+ >= 0.9: the student must be called
    assert precheck(m, None, 0.1) is None  # caller handles "not certified" through decide()
    assert precheck(hand_model(weights=(1.5, -1.0, 2.0)), 0.9, 0.1) is None  # unbounded


def test_precheck_never_changes_a_decision_on_a_grid() -> None:
    m = hand_model()
    sims = [i / 20 for i in range(-20, 21)]
    lambdas = (0.05, 0.3, 0.5, 0.8, 0.9, 0.99, 0.999999)
    lps = (0.0, -1e-6, -0.01, -0.1, -0.3, -0.5, -1.0, -2.0, -5.0, -20.0, -300.0)
    fired = silent = 0
    for sim in sims:
        for lam in lambdas:
            pre = precheck(m, lam, sim)
            if pre is None:
                silent += 1
                continue
            fired += 1
            for e in (0.0, 1.0):
                for lp in lps:
                    d = decide(m, lam, sig(e, lp, sim))
                    assert d.served_by == "teacher", (sim, lam, e, lp)
    assert fired > 20 and silent > 20  # both branches exercised


def test_precheck_never_changes_a_decision_on_random_models() -> None:
    rng = random.Random(21)
    fired = 0
    for _ in range(300):
        m = hand_model(
            means=(rng.uniform(0.3, 1.0), rng.uniform(-1.0, -0.1), rng.uniform(0.2, 0.9)),
            stds=(
                rng.choice([0.0, rng.uniform(0.1, 0.5)]),
                rng.uniform(0.05, 0.6),
                rng.uniform(0.05, 0.3),
            ),
            weights=(rng.uniform(-1, 3), rng.uniform(-1, 3), rng.uniform(-1, 4)),
            bias=rng.uniform(-3, 2),
        )
        lam = rng.choice([0.2, 0.5, 0.7, 0.9, 0.97])
        for _ in range(20):
            sim = rng.uniform(-1.0, 1.0)
            pre = precheck(m, lam, sim)
            if pre is None:
                continue
            fired += 1
            for _ in range(10):
                s = sig(
                    float(rng.random() < 0.5),
                    -rng.expovariate(2.0) * rng.choice([0.01, 1, 30]),
                    sim,
                )
                assert decide(m, lam, s).served_by == "teacher"
    assert fired > 100
