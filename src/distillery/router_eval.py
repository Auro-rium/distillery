"""Router fit / certify / evaluate on persisted signal rows, and the pre-registration receipt.

Pure functions over signal rows (``router_data.compute_signals`` output) plus one git helper:

* ``verify_prereg``: the integrity gate shared by ``signals --split cal|test``, ``certify`` and
  ``evaluate``. The pre-registration commit must exist, be an ancestor of HEAD, and the
  DECISIONS.md AT THAT COMMIT must contain the sealed sha256 of every split the step touches. The
  receipt (commit hash and its commit timestamp) goes into every artifact those steps write.
* ``fit_model`` (FIT rows only), ``certify_cal`` (Learn-then-Test on CAL only, grid from FIT),
  ``evaluate_test`` (ONE-SHOT on TEST: the pre-registered W-C1 and W-C2 computations).
"""

from __future__ import annotations

import json
import os
import re
import subprocess  # noqa: S404 - fixed argv, never a shell: only `git` read commands
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from distillery.router import FEATURES, RouterModel, decide, fit_logistic
from distillery.stats import (
    CertResult,
    certify,
    clopper_pearson_lower,
    clopper_pearson_upper,
    coverage_grid,
    risk_coverage_curve,
)

DEFAULT_LEVELS = tuple(round(0.05 * k, 2) for k in range(1, 20))  # 0.05 .. 0.95
MIN_COVERAGE = 0.30  # W-C1: pre-registered
SYSTEM_TOLERANCE = 0.01  # W-C1: system accuracy >= teacher-only - 0.01
DEFERRED_SHARE_MIN = 0.80  # W-C2
CONF = 0.95
_COMMIT_RE = re.compile(r"^[0-9a-fA-F]{7,64}$")


class RouterRefusal(RuntimeError):
    """A precondition failed; nothing was computed, written or spent."""


class PreregError(RouterRefusal):
    """The pre-registration commit is missing, not an ancestor of HEAD, or lacks a sealed sha."""


# ---------------------------------------------------------------- pre-registration receipt


def _git(repo: Path, *args: str) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(  # noqa: S603
            ["git", "-C", str(repo), *args],  # noqa: S607
            capture_output=True, text=True, check=False, timeout=60,
        )  # fmt: skip
    except (OSError, subprocess.SubprocessError) as exc:
        raise PreregError(f"cannot run git in {repo}: {exc}") from None


def verify_prereg(
    repo: Path | str, commit: str | None, required_shas: Mapping[str, str | None]
) -> dict[str, Any]:
    """Check the pre-registration commit and return its receipt.

    ``required_shas`` maps a label (``"cal"``, ``"test"``) to the sealed sha256 of that split in the
    store's seals table (None = not sealed: refused). Refuses unless: a commit was given, it
    resolves in ``repo``, it is an ancestor of (or equal to) HEAD, and ``DECISIONS.md`` as it was
    at that commit contains every required sha256."""
    if not commit:
        raise PreregError("a pre-registration commit is required: pass --prereg-commit <hash>")
    if not _COMMIT_RE.fullmatch(commit):
        raise PreregError(f"not a commit hash: {commit!r}")
    repo = Path(repo)
    missing = sorted(k for k, v in required_shas.items() if not v)
    if missing:
        raise PreregError(f"split(s) not sealed yet: {', '.join(missing)}")
    res = _git(repo, "rev-parse", "--verify", "--quiet", f"{commit}^{{commit}}")
    if res.returncode != 0 or not res.stdout.strip():
        raise PreregError(f"pre-registration commit {commit} does not exist in {repo}")
    full = res.stdout.strip()
    anc = _git(repo, "merge-base", "--is-ancestor", full, "HEAD")
    if anc.returncode == 1:
        raise PreregError(f"pre-registration commit {full[:12]} is not an ancestor of HEAD")
    if anc.returncode != 0:
        raise PreregError(f"cannot compare {full[:12]} with HEAD: {anc.stderr.strip()}")
    show = _git(repo, "show", f"{full}:DECISIONS.md")
    if show.returncode != 0:
        raise PreregError(f"DECISIONS.md does not exist at commit {full[:12]}")
    text = show.stdout.lower()
    absent = sorted(k for k, v in required_shas.items() if str(v).lower() not in text)
    if absent:
        raise PreregError(
            f"DECISIONS.md at {full[:12]} does not contain the sealed sha256 of: "
            f"{', '.join(absent)} (pre-register the hashes before scoring)"
        )
    when = _git(repo, "show", "-s", "--format=%cI", full)
    if when.returncode != 0:
        raise PreregError(f"cannot read the commit time of {full[:12]}")
    return {
        "commit": full,
        "committed_at": when.stdout.strip(),
        "decisions_contains": {k: str(v) for k, v in sorted(required_shas.items())},
    }


# ---------------------------------------------------------------- one-shot artifact writing


def write_new_json(path: Path, doc: Mapping[str, Any]) -> str:
    """Write ``doc`` to a file that must not exist (O_EXCL). Returns the file's sha256."""
    import hashlib

    data = (json.dumps(doc, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    except FileExistsError:
        raise RouterRefusal(f"{path.name} already exists; it is never overwritten") from None
    with os.fdopen(fd, "wb") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    return hashlib.sha256(data).hexdigest()


# ---------------------------------------------------------------- signals -> model


def features_of(row: Mapping[str, Any]) -> dict[str, float]:
    if row.get("mean_logprob") is None:
        raise RouterRefusal(
            f"item {row.get('task_id')}: no mean_logprob (the student did not supply one)"
        )
    return {f: float(row[f]) for f in FEATURES}


def fit_model(rows: Sequence[Mapping[str, Any]], *, l2: float = 1e-2) -> RouterModel:
    """Fit on FIT rows only; label = the student was correct."""
    if not rows:
        raise RouterRefusal("no fit rows")
    return fit_logistic([features_of(r) for r in rows], [bool(r["correct"]) for r in rows], l2=l2)


def scores_of(model: RouterModel, rows: Sequence[Mapping[str, Any]]) -> list[float]:
    return [model.score(features_of(r)) for r in rows]


def certify_cal(
    model: RouterModel,
    fit_rows: Sequence[Mapping[str, Any]],
    cal_rows: Sequence[Mapping[str, Any]],
    *,
    alpha: float,
    delta: float,
    levels: Sequence[float] = DEFAULT_LEVELS,
) -> dict[str, Any]:
    """Learn-then-Test on the CAL rows; the threshold grid comes from the FIT scores."""
    if not cal_rows:
        raise RouterRefusal("no calibration rows")
    grid = coverage_grid(scores_of(model, fit_rows), levels)
    cal_scores = scores_of(model, cal_rows)
    cal_correct = [bool(r["correct"]) for r in cal_rows]
    cert: CertResult = certify(cal_scores, cal_correct, alpha=alpha, delta=delta, grid=grid)
    return {
        "alpha": alpha,
        "delta": delta,
        "levels": list(levels),
        "grid": grid,
        "cert": cert.to_json(),
        "lambda_hat": cert.lambda_hat,
        "model_sha256": model.sha256(),
        "n_fit": len(fit_rows),
        "n_cal": len(cal_rows),
        "cal_curve": risk_coverage_curve(cal_scores, cal_correct, grid),
    }


# ---------------------------------------------------------------- the one-shot test evaluation


def _rate(num: int, den: int) -> float | None:
    return num / den if den else None


def evaluate_test(
    model: RouterModel,
    lambda_hat: float | None,
    test_rows: Sequence[Mapping[str, Any]],
    *,
    alpha: float,
    min_coverage: float = MIN_COVERAGE,
    tolerance: float = SYSTEM_TOLERANCE,
    deferred_share_min: float = DEFERRED_SHARE_MIN,
    conf: float = CONF,
) -> dict[str, Any]:
    """The pre-registered test computations (W-C1 and W-C2), exactly as specified.

    Route each item with ``router.decide`` (student iff score >= lambda_hat; no certified
    threshold: everything to the teacher). Every row needs ``teacher_correct``."""
    n = len(test_rows)
    if n == 0:
        raise RouterRefusal("no test rows")
    if any(r.get("teacher_correct") is None for r in test_rows):
        raise RouterRefusal("every test row needs teacher_correct (score the teacher on test)")
    items: list[dict[str, Any]] = []
    for r in test_rows:
        d = decide(model, lambda_hat, features_of(r))
        items.append(
            {
                "task_id": r["task_id"], "kind": r["kind"], "family": r["family"],
                "score": d.score, "served_by": d.served_by, "reason": d.reason,
                "output_sha256": r["output_sha256"], "correct": bool(r["correct"]),
                "teacher_correct": bool(r["teacher_correct"]),
            }
        )  # fmt: skip
    served = [i for i in items if i["served_by"] == "student"]
    n_served = len(served)
    errors = sum(1 for i in served if not i["correct"])
    coverage = n_served / n
    sel_err = _rate(errors, n_served)
    sys_correct = sum(
        i["correct"] if i["served_by"] == "student" else i["teacher_correct"] for i in items
    )
    system_acc = sys_correct / n
    teacher_acc = sum(i["teacher_correct"] for i in items) / n
    wc1_err_ok = sel_err is not None and sel_err <= alpha
    wc1_cov_ok = coverage >= min_coverage
    wc1_sys_ok = system_acc >= teacher_acc - tolerance
    w_c1 = {
        "coverage": coverage,
        "n_served": n_served,
        "n_errors_served": errors,
        "selective_error": sel_err,
        "alpha": alpha,
        "selective_error_pass": wc1_err_ok,
        "cp_upper_95": clopper_pearson_upper(errors, n_served, conf) if n_served else 1.0,
        "min_coverage": min_coverage,
        "min_coverage_pass": wc1_cov_ok,
        "system_accuracy": system_acc,
        "teacher_only_accuracy": teacher_acc,
        "tolerance": tolerance,
        "system_accuracy_pass": wc1_sys_ok,
        "pass": wc1_err_ok and wc1_cov_ok and wc1_sys_ok,
    }
    ns = [i for i in items if i["kind"] in ("natural", "stress")]
    e_items = [i for i in ns if not i["correct"]]
    c_items = [i for i in ns if i["correct"]]
    e_def = sum(1 for i in e_items if i["served_by"] == "teacher")
    c_def = sum(1 for i in c_items if i["served_by"] == "teacher")
    share = _rate(e_def, len(e_items))
    lower = clopper_pearson_lower(e_def, len(e_items), conf) if e_items else None
    rate_c = _rate(c_def, len(c_items))
    note = None
    if not e_items:
        note = "no student-wrong natural/stress items: W-C2 cannot pass"
    elif not c_items:
        note = "no student-correct natural/stress items: deferral rate undefined, W-C2 cannot pass"
    wc2_pass = (
        share is not None and lower is not None and rate_c is not None
        and share >= deferred_share_min and lower > rate_c
    )  # fmt: skip
    by_family: dict[str, dict[str, int]] = {}
    for i in ns:
        f = by_family.setdefault(
            f"{i['kind']}:{i['family']}",
            {"n": 0, "student_wrong": 0, "wrong_deferred": 0, "student_correct": 0,
             "correct_deferred": 0},
        )  # fmt: skip
        f["n"] += 1
        wrong, deferred = not i["correct"], i["served_by"] == "teacher"
        f["student_wrong" if wrong else "student_correct"] += 1
        if deferred:
            f["wrong_deferred" if wrong else "correct_deferred"] += 1
    stress = [i for i in items if i["kind"] == "stress"]
    w_c2 = {
        "n_error_items": len(e_items),
        "n_error_deferred": e_def,
        "deferred_share": share,
        "deferred_share_min": deferred_share_min,
        "deferred_share_lower_95": lower,
        "n_correct_items": len(c_items),
        "n_correct_deferred": c_def,
        "deferral_rate_on_correct": rate_c,
        "per_family": dict(sorted(by_family.items())),
        "stress_routing_share_student": _rate(
            sum(1 for i in stress if i["served_by"] == "student"), len(stress)
        ),
        "stress_routing_gated": False,
        "note": note,
        "pass": wc2_pass,
    }
    return {
        "n_test": n,
        "lambda_hat": lambda_hat,
        "model_sha256": model.sha256(),
        "counts_by_kind": dict(sorted(Counter(i["kind"] for i in items).items())),
        "W-C1": w_c1,
        "W-C2": w_c2,
        "per_item": items,
    }  # fmt: skip
