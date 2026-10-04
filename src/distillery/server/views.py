"""Pure builders: report dict -> tree / examples / detail. Nothing here invents data; anything
not present in the report is None (shown as "not measured" by the UI)."""

from __future__ import annotations

from typing import Any

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


def stored_examples(report: dict[str, Any]) -> list[Any] | None:
    """The core writes ``report['evaluation']['examples']``; fall back to a top-level list
    (older/hand-made reports). None means the report records no examples field at all."""
    ev = report.get("evaluation")
    if isinstance(ev, dict) and isinstance(ev.get("examples"), list):
        return list(ev["examples"])
    top = report.get("examples")
    return list(top) if isinstance(top, list) else None


def example_kind(e: dict[str, Any]) -> str | None:
    """The core's kind: the stored ``kind`` when present, else derived from base_ok/student_ok
    with the same rule (fixed / regressed / still_wrong; both-correct items are never examples)."""
    stored = e.get("kind")
    if isinstance(stored, str):
        return stored
    if "base_ok" not in e or "student_ok" not in e:
        return None
    base, student = bool(e["base_ok"]), bool(e["student_ok"])
    if student and not base:
        return "fixed"
    if base and not student:
        return "regressed"
    if not base and not student:
        return "still_wrong"
    return None


def filter_examples(report: dict[str, Any], kind: str, limit: int) -> tuple[list[Any], bool]:
    """Returns (items, available). ``available`` is False when the report has no examples field."""
    items = stored_examples(report)
    if items is None:
        return [], False
    picked = [
        e for e in items if isinstance(e, dict) and (kind == "all" or example_kind(e) == kind)
    ]
    return picked[:limit], True


def example_totals(report: dict[str, Any]) -> dict[str, int | None]:
    """Per-kind totals over the WHOLE held-out set, only where the report's gate records them.

    fixed = gate.student_only_vs_base and regressed = gate.base_only_vs_student are exact.
    still_wrong is derived from the recorded accuracies (base correct = base_acc * n) and is null
    if the report lacks the inputs or they are inconsistent. Never estimated."""
    ev = report.get("evaluation")
    gate = ev.get("gate") if isinstance(ev, dict) else None
    out: dict[str, int | None] = {"fixed": None, "still_wrong": None, "regressed": None}
    if not isinstance(gate, dict):
        return out
    fixed, regressed = gate.get("student_only_vs_base"), gate.get("base_only_vs_student")
    n, base_acc = gate.get("n"), gate.get("base_acc")
    if isinstance(fixed, int) and isinstance(regressed, int):
        out["fixed"], out["regressed"] = fixed, regressed
        if isinstance(n, int) and isinstance(base_acc, int | float):
            both_right = round(base_acc * n) - regressed
            wrong = n - both_right - fixed - regressed
            if both_right >= 0 and wrong >= 0 and abs(base_acc * n - round(base_acc * n)) < 1e-6:
                out["still_wrong"] = wrong
    return out


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
            "finetune_billed_usd": None,
            "finetune_ceiling_usd": None,
        },
        "sandbox": {"operations": None, "concurrency_peak": None},
        "verifier": {"language": report.get("pack", "sql"), "code": None, "selftest": None},
    }
