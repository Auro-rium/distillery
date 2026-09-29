"""Pure promotion gate: no I/O, no LLM."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

from pydantic import BaseModel

from distillery.config import GateThresholds
from distillery.stats import mcnemar_exact, paired_bootstrap_ratio_ci, wilson_ci


class GateResult(BaseModel):
    decision: Literal["PROMOTE", "REJECT"]
    n: int
    base_acc: float
    student_acc: float
    teacher_acc: float
    student_ci: tuple[float, float]
    ratio_point: float | None
    ratio_lo: float | None
    ratio_hi: float | None
    bootstrap_resamples_used: int
    bootstrap_skipped: int
    student_only_vs_base: int  # student right, base wrong
    base_only_vs_student: int  # base right, student wrong
    mcnemar_p: float
    thresholds: GateThresholds
    reasons: list[str]


def evaluate_gate(
    base: Sequence[bool],
    student: Sequence[bool],
    teacher: Sequence[bool],
    cfg: GateThresholds,
) -> GateResult:
    """PROMOTE iff ratio lower bound >= min AND student beats base (McNemar p < alpha)."""
    n = len(student)
    if len(base) != n or len(teacher) != n:
        raise ValueError("base, student, teacher must have equal length (same item order)")
    if n == 0:
        raise ValueError("held-out set is empty")

    reasons: list[str] = []
    ok = True
    base_acc = sum(base) / n
    student_acc = sum(student) / n
    teacher_acc = sum(teacher) / n

    ratio_point = ratio_lo = ratio_hi = None
    used = skipped = 0
    if teacher_acc == 0:
        ok = False
        reasons.append("teacher accuracy is 0; ratio undefined")
    else:
        ci = paired_bootstrap_ratio_ci(student, teacher, cfg.bootstrap_resamples, cfg.seed)
        ratio_point, ratio_lo, ratio_hi = ci.point, ci.lo, ci.hi
        used, skipped = ci.resamples_used, ci.skipped_teacher_zero
        if ci.lo >= cfg.ratio_lower_bound_min:
            reasons.append(
                f"ratio lower bound {ci.lo:.3f} >= {cfg.ratio_lower_bound_min} "
                f"(point {ci.point:.3f})"
            )
        else:
            ok = False
            reasons.append(
                f"ratio lower bound {ci.lo:.3f} < {cfg.ratio_lower_bound_min} "
                f"(point {ci.point:.3f})"
            )

    mc = mcnemar_exact(student, base)  # a = student, b = base
    if mc.a_only <= mc.b_only:
        ok = False
        reasons.append(
            f"student does not beat base: student-only wins {mc.a_only} "
            f"<= base-only wins {mc.b_only}"
        )
    elif mc.p_value < cfg.mcnemar_alpha:
        reasons.append(
            f"McNemar exact p={mc.p_value:.4f} < {cfg.mcnemar_alpha} "
            f"(student-only {mc.a_only}, base-only {mc.b_only})"
        )
    else:
        ok = False
        reasons.append(
            f"McNemar exact p={mc.p_value:.4f} >= {cfg.mcnemar_alpha} "
            f"(student-only {mc.a_only}, base-only {mc.b_only})"
        )

    return GateResult(
        decision="PROMOTE" if ok else "REJECT",
        n=n,
        base_acc=base_acc,
        student_acc=student_acc,
        teacher_acc=teacher_acc,
        student_ci=wilson_ci(sum(student), n),
        ratio_point=ratio_point,
        ratio_lo=ratio_lo,
        ratio_hi=ratio_hi,
        bootstrap_resamples_used=used,
        bootstrap_skipped=skipped,
        student_only_vs_base=mc.a_only,
        base_only_vs_student=mc.b_only,
        mcnemar_p=mc.p_value,
        thresholds=cfg,
        reasons=reasons,
    )
