"""Gate B after the fact: score a confirmed human set against an ALREADY FINISHED run.

Adding ``--human-set`` to ``run`` re-keys the ``split`` stage and everything after it, so it cannot
be added to a finished run without repeating paid work. This module instead seals the confirmed
set into the finished run (after training and after the Gate A decision) and scores base, student
and teacher on it once, with the run's own gate thresholds and the same serving wiring as
``final_eval``. All sealed-set access stays in ``evaluator.py``; this module only hands it paths,
digests and the train/dev question text.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from distillery import evaluator as evaluator_mod
from distillery.config import Config, GateThresholds
from distillery.orchestrator import (
    Deps,
    LLMGenerator,
    Pipeline,
    PipelineConfig,
    PipelineError,
    _close_all,
    _pretty,
    artifact_from_json,
    stage_rows,
)
from distillery.store import HeldoutIntegrityError, Store, atomic_write_bytes, sha256_hex
from distillery.student import StudentServer
from distillery.taskpacks.sql import schema as sql_schema

REPORT_HUMAN_FILE = "report_human.json"
PROVENANCE_NOTE = (
    "human set sealed AFTER training and after the Gate A decision; "
    "it was not used for training, selection or tuning"
)
SCORE_STAGE = "score_human"


class HumanScoreRefusal(RuntimeError):
    """A precondition failed; nothing was sealed, scored or spent."""


def load_report(store: Store, run_id: str) -> dict[str, Any]:
    path = store.run_dir(run_id) / "report.json"
    if not path.is_file():
        raise HumanScoreRefusal(f"run {run_id} has no report.json (not finished?)")
    report: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return report


def _outputs(store: Store, run_id: str, stage: str) -> list[dict[str, Any]]:
    """Every completed output of a stage, newest first."""
    rows = [
        r for r in stage_rows(store, run_id) if r["stage"] == stage and r["status"] == "complete"
    ]
    return [json.loads(store.get_artifact(run_id, str(r["output_sha256"]))) for r in reversed(rows)]


def check_report_preconditions(store: Store, run_id: str, report: Mapping[str, Any]) -> None:
    """Refuse (before any spend) unless the run finished a real final_eval and has no Gate B yet."""
    ev = report.get("evaluation") or {}
    if ev.get("human") is not None or report.get("decision_human") is not None:
        raise HumanScoreRefusal(
            f"run {run_id} already has a Gate B result; it is never overwritten"
        )
    if (store.run_dir(run_id) / REPORT_HUMAN_FILE).exists():
        raise HumanScoreRefusal(
            f"{REPORT_HUMAN_FILE} already exists for run {run_id}; it is never overwritten"
        )
    if not _outputs(store, run_id, "final_eval"):
        raise HumanScoreRefusal(f"run {run_id} has no completed final_eval stage")
    r = report.get("candidate_round")
    if not any(
        n == "expected_artifact" and d.get("round") == r for n, d in store.list_experiments(run_id)
    ):
        raise HumanScoreRefusal(f"run {run_id} has no expected_artifact for round {r}")


def score_human_run(
    store: Store,
    run_id: str,
    human_path: Path | str,
    config: Config,
    deps: Deps,
    *,
    say: Callable[[str], None] = print,
) -> dict[str, Any]:
    """Seal and score the confirmed human set for a finished run; write ``report_human.json`` and
    add the Gate B block to ``report.json`` (only where absent). Returns the Gate B block.

    Raises ``HumanScoreRefusal`` / ``HumanSetError`` before anything is sealed or spent when a
    precondition fails. Spend goes to the run's own ledger, preflighted against the caps.
    """
    report = load_report(store, run_id)
    check_report_preconditions(store, run_id, report)
    cfg_block = report["config"]
    gate_cfg = GateThresholds.model_validate(cfg_block["gate_thresholds"])  # the run's own
    pcfg = PipelineConfig.model_validate(cfg_block["pipeline"])
    pipe = Pipeline(
        pcfg, config.model_copy(update={"gate": gate_cfg}), deps, store, run_id, say=say
    )
    run_dir = store.run_dir(run_id)

    schemas = _outputs(store, run_id, "schema")
    if not schemas:
        raise HumanScoreRefusal(f"run {run_id} has no schema stage output")
    schema = schemas[0]
    db_path = run_dir / "db.sqlite"
    if not db_path.is_file() or sha256_hex(db_path.read_bytes()) != schema["db_sha256"]:
        raise HumanScoreRefusal(f"run {run_id}: db.sqlite is missing or differs from the record")
    pipe.ddl = sql_schema.schema_ddl(int(schema["db_seed"]))

    sealed_sha = report["data"]["heldout_sealed_sha256"]
    splits = [s for s in _outputs(store, run_id, "split") if s.get("sealed_sha256") == sealed_sha]
    if not splits:
        raise HumanScoreRefusal(f"run {run_id}: no split stage output matches the sealed gate set")
    split = splits[0]

    r = int(report["candidate_round"])
    try:
        expected = pipe._expected(r)  # noqa: SLF001 - the run's own recorded identity
    except PipelineError as exc:
        raise HumanScoreRefusal(str(exc)) from None
    ft = [
        o
        for o in _outputs(store, run_id, f"finetune_r{r}")
        if o["artifact"]["adapter_sha256"] == expected.adapter_sha256
        and o["artifact"]["job_id"] == expected.job_id
    ]
    if not ft:
        raise HumanScoreRefusal(f"run {run_id}: no finetune_r{r} output matches expected_artifact")
    trained = artifact_from_json(ft[0]["artifact"], run_dir)
    try:  # cheap, before sealing or serving anything: the adapter on disk must be the trained one
        evaluator_mod.verify_artifact(trained, expected)
    except evaluator_mod.ArtifactMismatchError as exc:
        raise HumanScoreRefusal(f"adapter check failed: {exc}") from None

    pipe._resolve_serving()  # noqa: SLF001 - same live wiring as final_eval
    deps = pipe.deps
    if deps.student_factory is None or deps.base_factory is None:
        raise HumanScoreRefusal("no student/base serving path is available")

    try:
        meta = evaluator_mod.seal_human_set(
            store, run_id, human_path, str(schema["db_sha256"]),
            train_questions=[d["question"] for d in split["train"]],
            dev_questions=[d["question"] for d in split["dev"]],
        )  # fmt: skip
    except HeldoutIntegrityError as exc:
        raise HumanScoreRefusal(f"a different human set is already sealed: {exc}") from None

    spent_before = pipe.ledger.spent()
    snap = pipe.sink.snapshot()
    opened: list[StudentServer] = []
    try:
        student = deps.student_factory(trained)
        opened.append(student)
        base = deps.base_factory()
        opened.append(base)
        gens: dict[str, evaluator_mod.Generator] = {
            "base": base,
            "student": student,
            "teacher": LLMGenerator(pipe.runner, "teacher", "eval_teacher", SCORE_STAGE),
        }
        block = evaluator_mod.score_sealed_human(
            store, run_id, gens, deps.executor,
            db_ref=pipe.db_ref, schema_ddl=pipe.ddl, gate_cfg=gate_cfg, pack=pipe.pack,
            trained=trained, expected=expected,
        )  # fmt: skip
    except BaseException:
        _close_all(opened, raise_first=False)
        raise
    _close_all(opened)

    teacher_use = pipe.sink.snapshot().get("eval_teacher", {})
    before = snap.get("eval_teacher", {})
    block.update(
        {
            "file_sha256": meta["file_sha256"],
            "db_sha256": meta["db_sha256"],
            "adapter_sha256": trained.adapter_sha256,
            "candidate_round": r,
            "dropped_exact_overlap": meta["dropped_exact_overlap"],
            "skeleton_in_train": meta["skeleton_in_train"],
            "counts": meta["counts"],
            "question_file_sha256": meta["question_file_sha256"],
            "teacher_model": meta["teacher_model"],
            "scored_at": datetime.now(UTC).isoformat(),
            "sealed_after_training_note": PROVENANCE_NOTE,
            "spend_usd_this_command": pipe.ledger.spent() - spent_before,
            "teacher_usage_this_command": {k: v - before.get(k, 0) for k, v in teacher_use.items()},
            "generation_errors": {
                name: int(getattr(srv, "generation_errors", 0) or 0)
                for name, srv in (("student", student), ("base", base))
            },
        }
    )
    out = {
        "run_id": run_id,
        "gate_b": block,
        "decision_human": block["gate"]["decision"],
        "gate_a_decision_unchanged": report["decision"],
        "note": PROVENANCE_NOTE,
    }
    atomic_write_bytes(run_dir / REPORT_HUMAN_FILE, _pretty(out).encode("utf-8"))
    merged = dict(report)
    evaluation = dict(merged.get("evaluation") or {})
    if evaluation.get("human") is None:
        evaluation["human"] = block
    merged["evaluation"] = evaluation
    if merged.get("decision_human") is None:
        merged["decision_human"] = block["gate"]["decision"]
    atomic_write_bytes(run_dir / "report.json", _pretty(merged).encode("utf-8"))
    return block
