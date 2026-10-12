"""Certified router: a logistic score P(student correct) from three per-request signals, and the
serve-the-student-or-defer decision against a Learn-then-Test threshold (see ``stats.certify``).

Pure functions, no I/O: fitting happens on the FIT split, the threshold is certified on the sealed
CALIBRATION split, and ``decide``/``precheck`` are the only things the serving path calls.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

# exec_ok: 0/1, student SQL ran read-only with >= 1 row. mean_logprob: mean token log-prob of the
# student's answer tokens (<= 0). nn_sim: max cosine similarity to the training questions.
FEATURES = ("exec_ok", "mean_logprob", "nn_sim")

_REASON_NOT_CERTIFIED = "not certified: all traffic goes to the teacher"
_REASON_UNFAMILIAR = "unfamiliar question pattern → teacher"
_REASONS = {
    "exec_ok": "student SQL failed to run → teacher",
    "nn_sim": _REASON_UNFAMILIAR,
    "mean_logprob": "low student confidence → teacher",
}
_REASON_FALLBACK = "low student confidence → teacher"


def _canonical(obj: Any) -> str:
    # Same shape as store.canonical_json (sorted keys, no spaces); NaN/inf are an error here.
    return json.dumps(
        obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    )


def _sigmoid(x: float) -> float:
    if x >= 0.0:
        return 1.0 / (1.0 + math.exp(-x))
    e = math.exp(x)
    return e / (1.0 + e)


def _softplus(x: float) -> float:
    return max(x, 0.0) + math.log1p(math.exp(-abs(x)))


@dataclass(frozen=True)
class RouterModel:
    """Standardize each signal with (mean, std), then a logistic: score = sigmoid(bias + w . z).

    A std of 0 (constant feature in the fit split) is treated as 1 when scoring. ``fit_sha256``
    is the hash of the fit data (provenance), ``sha256()`` the hash of this model's JSON.
    """

    feature_names: tuple[str, ...]
    means: tuple[float, ...]
    stds: tuple[float, ...]
    weights: tuple[float, ...]
    bias: float
    l2: float
    n_fit: int
    fit_sha256: str

    def __post_init__(self) -> None:
        k = len(self.feature_names)
        if not (len(self.means) == len(self.stds) == len(self.weights) == k):
            raise ValueError("feature_names, means, stds and weights must have equal length")
        if len(set(self.feature_names)) != k:
            raise ValueError("feature_names must be unique")
        vals = (*self.means, *self.stds, *self.weights, self.bias, self.l2)
        if not all(math.isfinite(v) for v in vals):
            raise ValueError("model parameters must be finite")

    def _std(self, j: int) -> float:
        return self.stds[j] if self.stds[j] > 0.0 else 1.0

    def contributions(self, signals: Mapping[str, float]) -> tuple[float, ...]:
        """Each feature's share w_j * z_j of the logit (z_j is the standardized signal)."""
        out: list[float] = []
        for j, name in enumerate(self.feature_names):
            z = (float(signals[name]) - self.means[j]) / self._std(j)
            out.append(self.weights[j] * z)
        return tuple(out)

    def logit(self, signals: Mapping[str, float]) -> float:
        total = self.bias
        for c in self.contributions(signals):
            total += c
        return total

    def score(self, signals: Mapping[str, float]) -> float:
        """P(student correct) for one request; ``signals`` needs every feature name."""
        return _sigmoid(self.logit(signals))

    def to_json(self) -> dict[str, Any]:
        return {
            "feature_names": list(self.feature_names),
            "means": list(self.means),
            "stds": list(self.stds),
            "weights": list(self.weights),
            "bias": self.bias,
            "l2": self.l2,
            "n_fit": self.n_fit,
            "fit_sha256": self.fit_sha256,
        }

    @classmethod
    def from_json(cls, data: Mapping[str, Any]) -> RouterModel:
        return cls(
            feature_names=tuple(str(x) for x in data["feature_names"]),
            means=tuple(float(x) for x in data["means"]),
            stds=tuple(float(x) for x in data["stds"]),
            weights=tuple(float(x) for x in data["weights"]),
            bias=float(data["bias"]),
            l2=float(data["l2"]),
            n_fit=int(data["n_fit"]),
            fit_sha256=str(data["fit_sha256"]),
        )

    def sha256(self) -> str:
        """SHA-256 of the canonical JSON of the whole model (what a certificate should pin)."""
        return hashlib.sha256(_canonical(self.to_json()).encode("utf-8")).hexdigest()


def _solve(a: list[list[float]], b: list[float]) -> list[float]:
    """Solve A x = b by Gaussian elimination with partial pivoting (A, b are not modified)."""
    n = len(b)
    m = [[*row, rhs] for row, rhs in zip(a, b, strict=True)]
    scale = max((abs(v) for row in a for v in row), default=0.0)
    for col in range(n):
        piv = max(range(col, n), key=lambda r: abs(m[r][col]))
        if abs(m[piv][col]) <= 1e-14 * scale or m[piv][col] == 0.0:
            raise ValueError("singular system")
        m[col], m[piv] = m[piv], m[col]
        for r in range(col + 1, n):
            f = m[r][col] / m[col][col]
            if f != 0.0:
                for c in range(col, n + 1):
                    m[r][c] -= f * m[col][c]
    x = [0.0] * n
    for r in range(n - 1, -1, -1):
        x[r] = (m[r][n] - sum(m[r][c] * x[c] for c in range(r + 1, n))) / m[r][r]
    return x


def fit_logistic(
    rows: Sequence[Mapping[str, float]],
    labels: Sequence[bool],
    *,
    l2: float = 1e-2,
    max_iter: int = 100,
    tol: float = 1e-10,
) -> RouterModel:
    """Fit the router by Newton/IRLS on standardized features with an L2 penalty.

    Minimizes sum_i logloss_i + (l2 / 2) * |w|^2 (a SUM over items, so l2 shrinks less as n
    grows); the bias is not penalized. The penalty keeps the optimum finite on perfectly
    separable data and the Hessian nonsingular for constant features (their weight stays 0).
    Full Newton steps are halved until the penalized objective decreases, so it cannot overshoot
    on separable data. Deterministic: plain float loops in input order, start at zero.
    Stops when the Newton decrement (half the squared Newton step in the Hessian norm, about the
    penalized loss still to gain) is below ``tol``, or after ``max_iter`` iterations.
    Raises ValueError if the labels hold a single class (an unpenalized bias would diverge).
    """
    n = len(rows)
    if len(labels) != n:
        raise ValueError(f"length mismatch: {n} rows vs {len(labels)} labels")
    if l2 < 0.0 or not math.isfinite(l2):
        raise ValueError("l2 must be a finite number >= 0")
    if max_iter < 1:
        raise ValueError("max_iter must be >= 1")
    y = [1.0 if v else 0.0 for v in labels]
    n_pos = sum(y)
    if n_pos == 0 or n_pos == n:
        raise ValueError("labels must contain both classes")
    k = len(FEATURES)
    raw = [[float(row[f]) for f in FEATURES] for row in rows]
    if not all(math.isfinite(v) for r in raw for v in r):
        raise ValueError("signals must be finite")

    means: list[float] = []
    stds: list[float] = []
    for j in range(k):
        col = [r[j] for r in raw]
        mean = math.fsum(col) / n
        var = math.fsum((v - mean) ** 2 for v in col) / n
        std = math.sqrt(var)
        if min(col) == max(col) or std <= 1e-12 * max(1.0, abs(mean)):
            std = 0.0  # constant feature: scored with std 1, so z is 0 and its weight stays 0
        means.append(mean)
        stds.append(std)
    z = [[(r[j] - means[j]) / (stds[j] if stds[j] > 0.0 else 1.0) for j in range(k)] for r in raw]

    def objective(th: list[float]) -> float:
        total = 0.5 * l2 * math.fsum(w * w for w in th[1:])
        for zi, yi in zip(z, y, strict=True):
            eta = th[0] + sum(th[j + 1] * zi[j] for j in range(k))
            total += _softplus(eta) - yi * eta
        return total

    theta = [0.0] * (k + 1)  # [bias, w_1 .. w_k]
    f = objective(theta)
    for _ in range(max_iter):
        grad = [0.0] * (k + 1)
        hess = [[0.0] * (k + 1) for _ in range(k + 1)]
        for zi, yi in zip(z, y, strict=True):
            eta = theta[0] + sum(theta[j + 1] * zi[j] for j in range(k))
            p = _sigmoid(eta)
            r = p - yi
            s = max(p * (1.0 - p), 1e-12)  # floor keeps the system solvable once p saturates
            grad[0] += r
            hess[0][0] += s
            for a in range(k):
                grad[a + 1] += r * zi[a]
                hess[0][a + 1] += s * zi[a]
                for b in range(a, k):
                    hess[a + 1][b + 1] += s * zi[a] * zi[b]
        for a in range(k + 1):
            for b in range(a):
                hess[a][b] = hess[b][a]
        for j in range(k):
            grad[j + 1] += l2 * theta[j + 1]
            hess[j + 1][j + 1] += l2
        step = _solve(hess, grad)
        decrease = sum(g * s for g, s in zip(grad, step, strict=True))
        if 0.5 * decrease < tol:
            # Newton decrement: the penalized loss still to gain. Below tol we take the (tiny)
            # full step and stop; a step-size test would chase the gradient's float noise.
            theta = [t - s for t, s in zip(theta, step, strict=True)]
            break
        t = 1.0
        accepted = False
        while t > 1e-12:
            cand = [th - t * s for th, s in zip(theta, step, strict=True)]
            fc = objective(cand)
            if fc <= f - 1e-4 * t * decrease:
                theta, f, accepted = cand, fc, True
                break
            t *= 0.5
        if not accepted:
            break  # at the optimum to float precision

    fit_sha = hashlib.sha256(
        _canonical({"features": list(FEATURES), "rows": raw, "labels": [int(v) for v in y]}).encode(
            "utf-8"
        )
    ).hexdigest()
    return RouterModel(
        feature_names=FEATURES,
        means=tuple(means),
        stds=tuple(stds),
        weights=tuple(theta[1:]),
        bias=theta[0],
        l2=l2,
        n_fit=n,
        fit_sha256=fit_sha,
    )


def max_score_given_sim(model: RouterModel, nn_sim: float) -> float | None:
    """Largest score reachable for this nn_sim over exec_ok in {0, 1}, mean_logprob in (-inf, 0].

    Evaluated with ``model.score`` itself on the best-case signals, so it is bit-identical to what
    ``decide`` would compute for that request. If the mean_logprob weight (original units, w / std)
    is positive the supremum is at logprob = 0. If it is <= 0 the score can grow without bound as
    logprob falls (or, at exactly 0, we do not special-case) and there is no finite bound: None
    (no short-circuit is possible). A model without a mean_logprob feature is bounded as well.
    Models with features beyond ``FEATURES`` cannot be bounded: None.
    """
    if not math.isfinite(nn_sim):
        raise ValueError("nn_sim must be finite")
    if any(name not in FEATURES for name in model.feature_names):
        return None
    if "mean_logprob" in model.feature_names:
        j = model.feature_names.index("mean_logprob")
        if model.weights[j] / model._std(j) <= 0.0:
            return None
    return max(
        model.score({"exec_ok": float(e), "mean_logprob": 0.0, "nn_sim": nn_sim}) for e in (0, 1)
    )


@dataclass(frozen=True)
class RouteDecision:
    served_by: Literal["student", "teacher"]
    score: float | None  # None when no score was computed (not certified / short-circuit)
    threshold: float | None  # the certified lambda_hat, None when not certified
    reason: str
    short_circuit: bool  # True when decided before calling the student


def decide(
    model: RouterModel, lambda_hat: float | None, signals: Mapping[str, float]
) -> RouteDecision:
    """Serve the student iff its score reaches the certified threshold, else defer.

    No certified threshold: everything goes to the teacher. A deferral is explained by the
    feature whose contribution w_j * z_j to the logit is most negative (ties: FEATURES order).
    """
    if lambda_hat is None:
        return RouteDecision("teacher", None, None, _REASON_NOT_CERTIFIED, False)
    score = model.score(signals)
    if score >= lambda_hat:
        return RouteDecision(
            "student",
            score,
            lambda_hat,
            f"certified: confidence {score:.2f} ≥ {lambda_hat:.2f}",
            False,
        )
    contrib = model.contributions(signals)
    worst = min(range(len(contrib)), key=lambda j: contrib[j])
    reason = _REASONS.get(model.feature_names[worst], _REASON_FALLBACK)
    return RouteDecision("teacher", score, lambda_hat, reason, False)


def precheck(model: RouterModel, lambda_hat: float | None, nn_sim: float) -> RouteDecision | None:
    """Skip the student when even its best case cannot reach the threshold.

    Returns a short-circuit teacher decision iff ``max_score_given_sim`` is finite and below
    lambda_hat; None means the student must be called (also when lambda_hat is None: the caller
    handles "not certified" through ``decide``). Never changes an outcome: any reachable score
    for this nn_sim is <= the bound < lambda_hat, so ``decide`` would also say teacher.
    """
    if lambda_hat is None:
        return None
    best = max_score_given_sim(model, nn_sim)
    if best is not None and best < lambda_hat:
        return RouteDecision("teacher", None, lambda_hat, _REASON_UNFAMILIAR, True)
    return None
