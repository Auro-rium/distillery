# ruff: noqa: S101, S105, S106
import pytest

from distillery.config import GateThresholds
from distillery.gate import evaluate_gate

CFG = GateThresholds(bootstrap_resamples=2000, seed=5)


def test_promote() -> None:
    n = 200
    teacher = [i % 10 != 0 for i in range(n)]  # 90%
    student = [i % 10 != 0 for i in range(n)]  # equal to teacher
    base = [i % 2 == 0 and i % 10 != 0 for i in range(n)]  # much weaker
    r = evaluate_gate(base, student, teacher, CFG)
    assert r.decision == "PROMOTE", r.reasons
    assert r.ratio_lo is not None and r.ratio_lo >= 0.85
    assert r.mcnemar_p < 0.05 and r.thresholds == CFG and r.reasons


def test_reject_by_ratio() -> None:
    n = 200
    teacher = [True] * n
    student = [i < 120 for i in range(n)]  # 0.6 of teacher
    base = [i < 20 for i in range(n)]
    r = evaluate_gate(base, student, teacher, CFG)
    assert r.decision == "REJECT"
    assert r.mcnemar_p < 0.05  # mcnemar passes, ratio is the reason
    assert any("ratio lower bound" in x and "<" in x for x in r.reasons)


def test_reject_by_mcnemar() -> None:
    n = 100
    teacher = [True] * n
    student = [True] * n
    base = [i >= 4 for i in range(n)]  # student wins only 4 discordants: p=0.125
    r = evaluate_gate(base, student, teacher, CFG)
    assert r.decision == "REJECT"
    assert r.ratio_lo is not None and r.ratio_lo >= 0.85
    assert r.mcnemar_p == pytest.approx(0.125)
    assert any("McNemar" in x and ">=" in x for x in r.reasons)


def test_reject_student_worse_than_base() -> None:
    n = 100
    teacher = [True] * n
    student = [i < 90 for i in range(n)]
    base = [True] * n
    r = evaluate_gate(base, student, teacher, CFG)
    assert r.decision == "REJECT"
    assert r.student_only_vs_base == 0 and r.base_only_vs_student == 10
    assert any("does not beat base" in x for x in r.reasons)


def test_tiny_n_rejects() -> None:
    r = evaluate_gate([False] * 5, [True] * 5, [True] * 5, CFG)
    assert r.decision == "REJECT"  # 5 wins => p = 0.0625
    assert r.mcnemar_p == pytest.approx(0.0625)


def test_teacher_zero_rejects_without_crash() -> None:
    r = evaluate_gate([False, False], [True, False], [False, False], CFG)
    assert r.decision == "REJECT" and r.ratio_lo is None


def test_validation() -> None:
    with pytest.raises(ValueError):
        evaluate_gate([], [], [], CFG)
    with pytest.raises(ValueError):
        evaluate_gate([True], [True, True], [True], CFG)


def test_deterministic() -> None:
    a = evaluate_gate([False] * 30, [True] * 25 + [False] * 5, [True] * 30, CFG)
    b = evaluate_gate([False] * 30, [True] * 25 + [False] * 5, [True] * 30, CFG)
    assert a == b
