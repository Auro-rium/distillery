# ruff: noqa: S101, S603, S607
from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

import pytest

from distillery.router import FEATURES, RouterModel, fit_logistic
from distillery.router_eval import (
    PreregError,
    RouterRefusal,
    certify_cal,
    evaluate_test,
    features_of,
    fit_model,
    verify_prereg,
    write_new_json,
)
from distillery.stats import clopper_pearson_lower, clopper_pearson_upper

SHA_CAL = "a" * 64
SHA_TEST = "b" * 64


# ------------------------------------------------------------------ prereg (tmp git repo)


def git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), *args], capture_output=True, text=True, check=True
    ).stdout.strip()


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    r = tmp_path / "repo"
    r.mkdir()
    git(r, "init", "-q", "-b", "main")
    git(r, "config", "user.email", "t@example.invalid")
    git(r, "config", "user.name", "t")
    git(r, "config", "commit.gpgsign", "false")
    (r / "DECISIONS.md").write_text("# decisions\n")
    git(r, "add", "-A")
    git(r, "commit", "-q", "-m", "base")
    return r


def commit_decisions(repo: Path, text: str, msg: str) -> str:
    (repo / "DECISIONS.md").write_text(text)
    git(repo, "add", "-A")
    git(repo, "commit", "-q", "-m", msg)
    return git(repo, "rev-parse", "HEAD")


def test_prereg_ok_receipt_and_short_hash(repo: Path) -> None:
    c = commit_decisions(repo, f"cal {SHA_CAL}\ntest {SHA_TEST.upper()}\n", "prereg")
    commit_decisions(repo, "later\n", "unrelated later commit")  # DECISIONS no longer has them
    rec = verify_prereg(repo, c[:10], {"cal": SHA_CAL, "test": SHA_TEST})
    assert rec["commit"] == c
    assert rec["committed_at"] == git(repo, "show", "-s", "--format=%cI", c)
    assert rec["decisions_contains"] == {"cal": SHA_CAL, "test": SHA_TEST}


def test_prereg_refusals(repo: Path) -> None:
    good = commit_decisions(repo, f"{SHA_CAL}\n", "prereg")
    with pytest.raises(PreregError, match="required"):
        verify_prereg(repo, None, {"cal": SHA_CAL})
    with pytest.raises(PreregError, match="not a commit hash"):
        verify_prereg(repo, "HEAD; rm -rf", {"cal": SHA_CAL})
    with pytest.raises(PreregError, match="does not exist"):
        verify_prereg(repo, "deadbeefdeadbeef", {"cal": SHA_CAL})
    with pytest.raises(PreregError, match="not sealed"):
        verify_prereg(repo, good, {"cal": None})
    # commit whose DECISIONS.md lacks the split sha
    with pytest.raises(PreregError, match="does not contain.*test"):
        verify_prereg(repo, good, {"cal": SHA_CAL, "test": SHA_TEST})
    # the sha only appearing in the working tree / a later commit does not count
    commit_decisions(repo, f"{SHA_CAL}\n{SHA_TEST}\n", "later")
    with pytest.raises(PreregError, match="does not contain.*test"):
        verify_prereg(repo, good, {"cal": SHA_CAL, "test": SHA_TEST})


def test_prereg_non_ancestor_commit(repo: Path) -> None:
    git(repo, "checkout", "-q", "-b", "side")
    side = commit_decisions(repo, f"{SHA_CAL}\n", "on a side branch")
    git(repo, "checkout", "-q", "main")
    commit_decisions(repo, "main moves on\n", "main")
    with pytest.raises(PreregError, match="not an ancestor"):
        verify_prereg(repo, side, {"cal": SHA_CAL})


def test_prereg_no_decisions_file(repo: Path) -> None:
    git(repo, "rm", "-q", "DECISIONS.md")
    git(repo, "commit", "-q", "-m", "drop")
    c = git(repo, "rev-parse", "HEAD")
    with pytest.raises(PreregError, match="DECISIONS.md does not exist"):
        verify_prereg(repo, c, {"cal": SHA_CAL})
    with pytest.raises(PreregError):
        verify_prereg(repo / "nope", c, {"cal": SHA_CAL})


def test_write_new_json_is_one_shot(tmp_path: Path) -> None:
    p = tmp_path / "x" / "eval.json"
    sha = write_new_json(p, {"a": 1})
    assert len(sha) == 64 and json.loads(p.read_text()) == {"a": 1}
    with pytest.raises(RouterRefusal, match="never overwritten"):
        write_new_json(p, {"a": 2})
    assert json.loads(p.read_text()) == {"a": 1}


# ------------------------------------------------------------------ W-C1 / W-C2 on hand-built rows


def model() -> RouterModel:
    """score = sigmoid(10 * nn_sim - 5): >= 0.5 iff nn_sim >= 0.5."""
    return RouterModel(
        FEATURES, (0.0, 0.0, 0.0), (1.0, 1.0, 1.0), (0.0, 0.0, 10.0), -5.0, 0.0, 1, "x"
    )


def row(
    i: int, kind: str, fam: str, sim: float, correct: bool, teacher: bool = True
) -> dict[str, Any]:
    return {
        "task_id": f"t{i}", "kind": kind, "family": fam, "exec_ok": 1, "mean_logprob": -0.1,
        "nn_sim": sim, "correct": correct, "output_sha256": f"{i:064x}", "teacher_correct": teacher,
    }  # fmt: skip


def build_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    i = 0

    def add(n: int, kind: str, fam: str, sim: float, correct: bool, teacher: bool = True) -> None:
        nonlocal i
        for _ in range(n):
            rows.append(row(i, kind, fam, sim, correct, teacher))
            i += 1

    # served (sim .9): 30 templated correct, 1 templated wrong, 9 natural correct (4 stress correct)
    add(30, "templated", "agg", 0.9, True)
    add(1, "templated", "agg", 0.9, False)
    add(9, "natural", "human", 0.9, True)
    add(4, "stress", "date_math", 0.9, True)
    # deferred (sim .1): natural/stress WRONG 9 (teacher right), natural/stress correct-but-deferred 3
    add(5, "natural", "human", 0.1, False)
    add(4, "stress", "set_ops", 0.1, False)
    add(3, "natural", "human", 0.1, True)
    # one wrong natural item that the router still serves (the leak)
    add(1, "natural", "human", 0.9, False, teacher=True)
    # deferred templated, teacher wrong on 2 of 12
    add(10, "templated", "agg", 0.1, True)
    add(2, "templated", "agg", 0.1, False, teacher=False)
    return rows


def test_wc1_and_wc2_hand_computed() -> None:
    rows = build_rows()
    assert len(rows) == 69
    out = evaluate_test(model(), 0.5, rows, alpha=0.05)
    c1 = out["W-C1"]
    # served = 30+1+9+4+1 = 45 ; errors among served = 1 templated + 1 natural = 2
    assert (c1["n_served"], c1["n_errors_served"]) == (45, 2)
    assert c1["coverage"] == pytest.approx(45 / 69)
    assert c1["selective_error"] == pytest.approx(2 / 45)
    assert c1["selective_error_pass"] is True  # 0.0444 <= 0.05
    assert c1["cp_upper_95"] == pytest.approx(clopper_pearson_upper(2, 45, 0.95))
    assert c1["min_coverage_pass"] is True
    # system: served -> student correct (43 of 45); deferred 24 -> teacher correct (22 of 24)
    assert c1["system_accuracy"] == pytest.approx((43 + 22) / 69)
    # teacher-only: 69 - 2 teacher-wrong
    assert c1["teacher_only_accuracy"] == pytest.approx(67 / 69)
    assert c1["system_accuracy_pass"] is False  # 65/69 < 67/69 - 0.01
    assert c1["pass"] is False

    c2 = out["W-C2"]
    # natural+stress wrong: 5 + 4 deferred + 1 served = 10
    assert (c2["n_error_items"], c2["n_error_deferred"]) == (10, 9)
    assert c2["deferred_share"] == pytest.approx(0.9)
    assert c2["deferred_share_lower_95"] == pytest.approx(clopper_pearson_lower(9, 10, 0.95))
    assert c2["deferred_share_lower_95"] == pytest.approx(0.6058, abs=1e-3)
    # natural+stress correct: 9 + 4 + 3 = 16, deferred 3
    assert (c2["n_correct_items"], c2["n_correct_deferred"]) == (16, 3)
    assert c2["deferral_rate_on_correct"] == pytest.approx(3 / 16)
    assert c2["pass"] is True  # 0.9 >= 0.8 and 0.6058 > 0.1875
    assert c2["per_family"]["natural:human"] == {
        "n": 18, "student_wrong": 6, "wrong_deferred": 5, "student_correct": 12, "correct_deferred": 3,
    }  # fmt: skip
    assert c2["stress_routing_share_student"] == pytest.approx(4 / 8)
    assert c2["stress_routing_gated"] is False
    assert len(out["per_item"]) == 69 and out["per_item"][0]["served_by"] == "student"
    assert out["per_item"][0]["output_sha256"] == f"{0:064x}"
    assert out["counts_by_kind"] == {"natural": 18, "stress": 8, "templated": 43}


def test_wc1_passes_and_wc2_boundaries() -> None:
    rows = [r for r in build_rows() if r["teacher_correct"]]  # teacher perfect
    out = evaluate_test(model(), 0.5, rows, alpha=0.05)
    c1 = out["W-C1"]
    assert c1["system_accuracy"] == pytest.approx((43 + 22) / 67)
    assert c1["teacher_only_accuracy"] == 1.0
    assert c1["pass"] is False  # system 0.9701 < 1 - 0.01
    # a stricter alpha flips only the selective-error check
    strict = evaluate_test(model(), 0.5, rows, alpha=0.04)["W-C1"]
    assert strict["selective_error_pass"] is False and strict["system_accuracy_pass"] is False
    # coverage floor: no certified threshold -> everything to the teacher
    none = evaluate_test(model(), None, rows, alpha=0.05)
    assert none["W-C1"]["coverage"] == 0.0 and none["W-C1"]["selective_error"] is None
    assert none["W-C1"]["pass"] is False and none["W-C1"]["system_accuracy"] == 1.0
    assert none["W-C2"]["n_error_deferred"] == none["W-C2"]["n_error_items"] == 10
    assert none["W-C2"]["deferral_rate_on_correct"] == 1.0 and none["W-C2"]["pass"] is False


def test_wc2_fails_on_low_share_or_weak_bound_and_edge_cases() -> None:
    # 5 wrong, 4 deferred: share .8 passes the 0.80 floor but lower bound 0.4161 vs rate .9 fails
    rows = [row(i, "natural", "human", 0.1, False) for i in range(4)]
    rows += [row(10, "natural", "human", 0.9, False)]
    rows += [row(20 + i, "natural", "human", 0.1, True) for i in range(9)]
    rows += [row(40, "natural", "human", 0.9, True)]
    c2 = evaluate_test(model(), 0.5, rows, alpha=0.05)["W-C2"]
    assert c2["deferred_share"] == pytest.approx(0.8)
    assert c2["deferral_rate_on_correct"] == pytest.approx(0.9)
    assert c2["pass"] is False
    only_templated = [row(i, "templated", "agg", 0.9, True) for i in range(5)]
    nc = evaluate_test(model(), 0.5, only_templated, alpha=0.05)["W-C2"]
    assert nc["pass"] is False and "no student-wrong" in nc["note"]
    assert nc["deferred_share"] is None and nc["stress_routing_share_student"] is None
    no_correct = [row(i, "natural", "human", 0.1, False) for i in range(5)]
    assert (
        "no student-correct" in evaluate_test(model(), 0.5, no_correct, alpha=0.05)["W-C2"]["note"]
    )


def test_evaluate_refusals() -> None:
    with pytest.raises(RouterRefusal, match="no test rows"):
        evaluate_test(model(), 0.5, [], alpha=0.05)
    r = row(0, "natural", "human", 0.9, True)
    with pytest.raises(RouterRefusal, match="teacher_correct"):
        evaluate_test(model(), 0.5, [{**r, "teacher_correct": None}], alpha=0.05)
    with pytest.raises(RouterRefusal, match="mean_logprob"):
        features_of({**r, "mean_logprob": None})


# ------------------------------------------------------------------ fit / certify


def synth_rows(n: int, seed: int, offset: int = 0) -> list[dict[str, Any]]:
    import random

    rng = random.Random(seed)
    rows = []
    for i in range(n):
        sim = rng.random()
        lp = -rng.random() * 2
        ok = 1 if rng.random() < 0.8 else 0
        p = 0.1 + 0.8 * sim * (1 if ok else 0.3) * (1 + lp / 4)
        rows.append(
            {
                "task_id": f"t{offset + i}", "kind": "templated", "family": "f", "exec_ok": ok,
                "mean_logprob": lp, "nn_sim": sim, "correct": rng.random() < p,
                "output_sha256": "0" * 64, "teacher_correct": True,
            }
        )  # fmt: skip
    return rows


def test_fit_and_certify_use_fit_grid_and_cal_only() -> None:
    fit, cal = synth_rows(400, 1), synth_rows(600, 2, 1000)
    m = fit_model(fit)
    assert m == fit_logistic([features_of(r) for r in fit], [r["correct"] for r in fit])
    out = certify_cal(m, fit, cal, alpha=0.3, delta=0.1)
    assert out["model_sha256"] == m.sha256() and out["n_fit"] == 400 and out["n_cal"] == 600
    grid = out["grid"]
    assert grid == sorted(grid, reverse=True) and len(grid) >= 5
    assert out["cert"]["steps"][0]["threshold"] == grid[0]
    # CAL data changes the certificate, not the grid
    other = certify_cal(m, fit, synth_rows(600, 3, 2000), alpha=0.3, delta=0.1)
    assert other["grid"] == grid
    with pytest.raises(RouterRefusal):
        certify_cal(m, fit, [], alpha=0.3, delta=0.1)
    with pytest.raises(RouterRefusal):
        fit_model([])


def test_system_gate_is_exact_at_the_boundary() -> None:
    """system == teacher_only - 0.01 must PASS; in float 0.06 >= 0.07 - 0.01 is False."""
    assert not (6 / 100 >= 7 / 100 - 0.01)  # the float trap this test pins

    def rows(teacher_deferred_ok: int) -> list[dict[str, Any]]:
        out = [row(0, "templated", "agg", 0.9, False, teacher=True)]  # served, wrong, teacher right
        for i in range(1, 100):
            out.append(row(i, "templated", "agg", 0.1, True, teacher=i <= teacher_deferred_ok))
        return out

    at = evaluate_test(model(), 0.5, rows(6), alpha=0.05)["W-C1"]
    assert (at["system_accuracy"], at["teacher_only_accuracy"]) == (0.06, 0.07)
    assert at["system_accuracy_pass"] is True  # exactly teacher - 0.01
    below = evaluate_test(model(), 0.5, rows(5), alpha=0.05)["W-C1"]
    assert below["system_accuracy"] == 0.05 and below["teacher_only_accuracy"] == 0.06
    assert below["system_accuracy_pass"] is True  # 0.05 >= 0.06 - 0.01 again exactly on the line
    worse = rows(4)  # system 4/100, teacher 5/100 -> still exactly -0.01
    assert evaluate_test(model(), 0.5, worse, alpha=0.05)["W-C1"]["system_accuracy_pass"] is True
    bad = rows(6)
    bad[0] = row(0, "templated", "agg", 0.9, False, teacher=True)
    bad[1] = row(1, "templated", "agg", 0.9, False, teacher=False)
    bad[2] = row(2, "templated", "agg", 0.9, False, teacher=True)  # served wrong, teacher right
    assert evaluate_test(model(), 0.5, bad, alpha=0.05)["W-C1"]["system_accuracy_pass"] is False
    # coverage and selective-error gates are exact as well: 30/100 coverage passes the 0.30 floor
    cov = [row(i, "templated", "agg", 0.9 if i < 30 else 0.1, True) for i in range(100)]
    c1 = evaluate_test(model(), 0.5, cov, alpha=0.05)["W-C1"]
    assert c1["min_coverage_pass"] is True and c1["pass"] is True
    assert evaluate_test(model(), 0.5, cov[1:], alpha=0.05)["W-C1"]["coverage"] < 0.30
