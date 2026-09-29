"""Sealed held-out evaluation. The ONLY module allowed to call ``Store.load_heldout``.

Scores base, student and teacher on the same held-out items in the same order, verifies the
evaluated adapter is the trained one, and hands the per-item outcomes to the pure gate.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

from distillery.config import GateThresholds
from distillery.finetune import TrainedArtifact, sha256_file
from distillery.gate import GateResult, evaluate_gate
from distillery.prompts import build_messages, extract_sql
from distillery.store import Store, canonical_json, sha256_hex
from distillery.taskpacks.sql.executor import Executor
from distillery.taskpacks.sql.runner import ExecOutcome
from distillery.taskpacks.sql.verifier import compare_outcomes

ChatMessages = list[dict[str, str]]
MODEL_ROLES = ("base", "student", "teacher")


class Generator(Protocol):
    def generate(self, messages_batch: Sequence[ChatMessages]) -> list[str]: ...


class ArtifactMismatchError(RuntimeError):
    """The artifact being evaluated is not the artifact that was trained."""


@dataclass(frozen=True)
class ExpectedArtifact:
    """Identity recorded at training time (job, checkpoint, adapter digest)."""

    job_id: str
    checkpoint_id: str
    adapter_sha256: str


@dataclass(frozen=True)
class ModelScores:
    correct: list[bool]
    reasons: list[str]
    unparseable: int  # outputs with no extractable SQL (counted, never skipped)

    @property
    def accuracy(self) -> float:
        return sum(self.correct) / len(self.correct) if self.correct else 0.0


@dataclass(frozen=True)
class EvalReport:
    heldout_sha256: str
    n: int
    scores: dict[str, ModelScores]
    gate: GateResult
    accuracy_by_class: dict[str, dict[str, float]]  # heldout_class -> model -> accuracy
    class_counts: dict[str, int]
    artifact: dict[str, str] = field(default_factory=dict)

    def to_json(self) -> dict[str, Any]:
        return {
            "heldout_sha256": self.heldout_sha256,
            "n": self.n,
            "accuracy": {m: s.accuracy for m, s in self.scores.items()},
            "unparseable": {m: s.unparseable for m, s in self.scores.items()},
            "accuracy_by_class": self.accuracy_by_class,
            "class_counts": self.class_counts,
            "artifact": self.artifact,
            "gate": self.gate.model_dump(mode="json"),
        }


def verify_artifact(trained: TrainedArtifact, expected: ExpectedArtifact) -> None:
    """Re-hash the files on disk and compare with what training recorded (rule 5.5)."""
    if trained.job_id != expected.job_id:
        raise ArtifactMismatchError(f"job id {trained.job_id!r} != trained {expected.job_id!r}")
    if trained.checkpoint_id != expected.checkpoint_id:
        raise ArtifactMismatchError(
            f"checkpoint id {trained.checkpoint_id!r} != trained {expected.checkpoint_id!r}"
        )
    recomputed = TrainedArtifact(
        job_id=trained.job_id,
        checkpoint_id=trained.checkpoint_id,
        base_model=trained.base_model,
        fine_tuned_model_checkpoint=trained.fine_tuned_model_checkpoint,
        files=tuple(
            type(f)(file_id=f.file_id, path=f.path, sha256=sha256_file(f.path))
            for f in trained.files
        ),
    )
    if recomputed.adapter_sha256 != expected.adapter_sha256:
        raise ArtifactMismatchError("adapter files on disk do not match the trained digest")


def score_model(
    generator: Generator,
    items: Sequence[Mapping[str, Any]],
    gold: Sequence[ExecOutcome],
    *,
    schema_ddl: str,
    db_ref: str,
    executor: Executor,
    role: str,
) -> ModelScores:
    if role not in MODEL_ROLES:
        raise ValueError(f"unknown model role {role!r}")
    batch = [
        build_messages(str(it["question"]), schema_ddl, role=f"eval_{role}")  # type: ignore[arg-type]
        for it in items
    ]
    outputs = generator.generate(batch)
    if len(outputs) != len(items):
        raise RuntimeError(f"{role}: got {len(outputs)} outputs for {len(items)} items")
    sqls = [extract_sql(o) for o in outputs]
    runnable = [i for i, s in enumerate(sqls) if s is not None]
    outcomes = executor.run_batch(db_ref, [sqls[i] or "" for i in runnable])
    by_index = dict(zip(runnable, outcomes, strict=True))
    correct: list[bool] = []
    reasons: list[str] = []
    for i, it in enumerate(items):
        if sqls[i] is None:
            correct.append(False)
            reasons.append("no SQL extracted from output")
            continue
        verdict = compare_outcomes(by_index[i], gold[i], bool(it["requires_order"]))
        correct.append(verdict.ok)
        reasons.append(verdict.reason)
    return ModelScores(correct, reasons, unparseable=sum(1 for s in sqls if s is None))


def evaluate(
    store: Store,
    run_id: str,
    generators: Mapping[str, Generator],
    executor: Executor,
    *,
    db_ref: str,
    schema_ddl: str,
    gate_cfg: GateThresholds,
    trained: TrainedArtifact | None = None,
    expected: ExpectedArtifact | None = None,
    on_scores: Callable[[str, ModelScores], None] | None = None,
) -> EvalReport:
    """Evaluate base/student/teacher on the sealed held-out set and run the gate."""
    if set(generators) != set(MODEL_ROLES):
        raise ValueError(f"generators must be exactly {MODEL_ROLES}")
    if (trained is None) != (expected is None):
        raise ValueError("pass both `trained` and `expected`, or neither")
    if trained is not None and expected is not None:
        verify_artifact(trained, expected)

    items = store.load_heldout(run_id)
    heldout_sha = sha256_hex(canonical_json(items).encode("utf-8"))
    gold = executor.run_batch(db_ref, [str(it["gold_sql"]) for it in items])
    for it, g in zip(items, gold, strict=True):
        if not g.ok:
            raise RuntimeError(f"gold SQL failed for held-out item {it.get('task_id')}: {g.error}")

    scores: dict[str, ModelScores] = {}
    for role in MODEL_ROLES:
        scores[role] = score_model(
            generators[role],
            items,
            gold,
            schema_ddl=schema_ddl,
            db_ref=db_ref,
            executor=executor,
            role=role,
        )
        if on_scores is not None:
            on_scores(role, scores[role])

    classes = [str(it.get("heldout_class", "seen")) for it in items]
    class_counts = {c: classes.count(c) for c in sorted(set(classes))}
    by_class: dict[str, dict[str, float]] = {}
    for c in class_counts:
        idx = [i for i, k in enumerate(classes) if k == c]
        by_class[c] = {m: sum(scores[m].correct[i] for i in idx) / len(idx) for m in MODEL_ROLES}

    gate = evaluate_gate(
        scores["base"].correct, scores["student"].correct, scores["teacher"].correct, gate_cfg
    )
    artifact = (
        {
            "job_id": trained.job_id,
            "checkpoint_id": trained.checkpoint_id,
            "adapter_sha256": trained.adapter_sha256,
        }
        if trained is not None
        else {}
    )
    return EvalReport(
        heldout_sha256=heldout_sha,
        n=len(items),
        scores=scores,
        gate=gate,
        accuracy_by_class=by_class,
        class_counts=class_counts,
        artifact=artifact,
    )
