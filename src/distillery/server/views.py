"""Pure builders: report dict -> tree / examples / detail. Nothing here invents data; anything
not present in the report is None (shown as "not measured" by the UI)."""

from __future__ import annotations

import re
from typing import Any

_ROUND_LABEL = re.compile(r"^round-(\d+)$")
EXAMPLE_KINDS = ("fixed", "still_wrong", "regressed", "all")


def tree_from_report(report: dict[str, Any]) -> dict[str, Any]:
    """Nodes: the base image (sandbox lineage root) plus one node per fine-tune round.

    A round's ``sandbox_image`` is the lineage uuid whose label is ``round-N`` (branches are made
    after round N's failure analysis); rounds without a branch have ``sandbox_image`` null.
    ``dev_score`` is that round's measured dev accuracy; ``cost_usd`` is not recorded per round.
    """
    rounds = [r for r in report.get("rounds", []) if isinstance(r, dict)]
    lineage = [x for x in report.get("sandbox_lineage", []) if isinstance(x, dict)]
    if not rounds:
        return {"nodes": []}
    by_label = {str(x.get("label")): x for x in lineage}
    base_id = str(lineage[0].get("parent")) if lineage else "base"
    nodes: list[dict[str, Any]] = [
        {
            "id": base_id, "parent_id": None, "label": "base model", "round": None,
            "hypothesis": None, "data_delta": None,
            "dev_score": (report.get("headroom") or {}).get("base_dev_acc"),
            "cost_usd": None, "sandbox_image": None, "selected": False,
        }
    ]  # fmt: skip
    prev_id = base_id
    selected = report.get("candidate_round")
    for r in rounds:
        n = r.get("round")
        branch = by_label.get(f"round-{n}")
        node_id = str(branch["uuid"]) if branch and "uuid" in branch else f"round-{n}"
        clusters = [c for c in r.get("clusters", []) if isinstance(c, dict)]
        fams = sorted({f for c in clusters for f in c.get("target_families", [])})
        added = r.get("targeted_new_rows")
        hyp = "; ".join(str(c.get("description")) for c in clusters if c.get("description"))
        nodes.append(
            {
                "id": node_id,
                "parent_id": prev_id,
                "label": f"round {n}",
                "round": n,
                "hypothesis": hyp or None,
                "data_delta": {"added_rows": added, "families": fams}
                if added is not None
                else None,
                "dev_score": r.get("dev_acc"),
                "cost_usd": None,
                "sandbox_image": str(branch["uuid"]) if branch and "uuid" in branch else None,
                "selected": n == selected,
            }
        )
        prev_id = node_id
    return {"nodes": nodes}


def filter_examples(report: dict[str, Any], kind: str, limit: int) -> tuple[list[Any], bool]:
    """Returns (items, available). ``available`` is False when the report has no examples field."""
    items = report.get("examples")
    if not isinstance(items, list):
        return [], False

    def keep(e: dict[str, Any]) -> bool:
        student, base = bool(e.get("student_ok")), bool(e.get("base_ok"))
        if kind == "fixed":
            return student and not base
        if kind == "still_wrong":
            return not student
        if kind == "regressed":
            return base and not student
        return True

    return [e for e in items if isinstance(e, dict) and keep(e)][:limit], True


def detail_from_report(
    run_id: str, report: dict[str, Any], *, recorded: bool, recorded_at: str | None
) -> dict[str, Any]:
    """Run detail for a finished run known only through its report (replay / sample)."""
    cost = report.get("cost") or {}
    stages = [
        {"name": s.get("stage"), "status": "done", "started_at": None, "ended_at": None}
        for s in report.get("stages", [])
        if isinstance(s, dict)
    ]
    return {
        "run_id": run_id,
        "dry_run": bool(report.get("dry_run")),
        "recorded": recorded,
        "recorded_at": recorded_at,
        "status": "complete",
        "error": None,
        "stages": stages,
        "spend": {
            "total_usd": cost.get("run_total_usd"),
            "cap_usd": cost.get("run_cap_usd"),
            "by_model": cost.get("llm_by_model") or {},
            "finetune_usd_estimate": None,
        },
        "sandbox": {"operations": None, "concurrency_peak": None},
        "verifier": {"language": report.get("pack", "sql"), "code": None, "selftest": None},
    }
