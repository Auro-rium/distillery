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
    outputs: list[str] = field(default_factory=list)  # raw model output per item
    sqls: list[str | None] = field(default_factory=list)  # extracted SQL per item

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
    # Capped per-item examples; revealed only here, after scoring (see build_examples).
    examples: list[dict[str, Any]] = field(default_factory=list)

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
            "examples": self.examples,
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
    return ModelScores(
        correct,
        reasons,
        unparseable=sum(1 for s in sqls if s is None),
        outputs=list(outputs),
        sqls=sqls,
    )


EXAMPLES_PER_KIND = 20  # 3 kinds -> at most 60 examples


def build_examples(
    items: Sequence[Mapping[str, Any]], scores: Mapping[str, ModelScores]
) -> list[dict[str, Any]]:
    """Deterministic capped examples in held-out order: the first ``EXAMPLES_PER_KIND`` of each
    kind. ``fixed`` = student ok, base wrong; ``regressed`` = base ok, student wrong;
    ``still_wrong`` = both wrong. Items both models solve are omitted."""

    def shown(role: str, i: int) -> str:
        return scores[role].sqls[i] or scores[role].outputs[i]

    picked: dict[str, list[dict[str, Any]]] = {"fixed": [], "still_wrong": [], "regressed": []}
    for i, it in enumerate(items):
        base_ok, student_ok = scores["base"].correct[i], scores["student"].correct[i]
        if base_ok and student_ok:
            continue
        kind = "fixed" if student_ok else "regressed" if base_ok else "still_wrong"
        if len(picked[kind]) >= EXAMPLES_PER_KIND:
            continue
        picked[kind].append(
            {
                "kind": kind,
                "task_id": it.get("task_id"),
                "family": it.get("family"),
                "heldout_class": it.get("heldout_class", "seen"),
                "question": it["question"],
                "gold_sql": it["gold_sql"],
                "base_sql": shown("base", i),
                "student_sql": shown("student", i),
                "teacher_sql": shown("teacher", i),
                "base_ok": base_ok,
                "student_ok": student_ok,
                "teacher_ok": scores["teacher"].correct[i],
            }
        )
    return [e for kind in ("fixed", "still_wrong", "regressed") for e in picked[kind]]


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
        examples=build_examples(items, scores),
    )


# ---------------------------------------------------------------- student diagnostics


def identical_rate(a: Sequence[str | None], b: Sequence[str | None]) -> float:
    """Fraction of positions where two output lists are exactly equal (0.0 when empty)."""
    if len(a) != len(b):
        raise ValueError("output lists differ in length")
    return sum(x == y for x, y in zip(a, b, strict=True)) / len(a) if a else 0.0


def diagnose_generate(
    generators: Mapping[str, Generator],
    sets: Mapping[str, Sequence[Mapping[str, Any]]],
    executor: Executor,
    *,
    db_ref: str,
    schema_ddl: str,
    compare_to: str = "base",
) -> dict[str, Any]:
    """Generate with every model over every named item set and score it.

    Each item needs ``task_id``, ``gold_sql``, ``requires_order`` and either ``messages`` (the exact
    prompt to send) or ``question`` (prompt built with the eval prompt builder). All sets go to a
    model in ONE ``generate`` call, so each model (sandbox image + weights) is loaded once. Returns
    ``{"models": {model: {set: {n, accuracy, items[{task_id, raw, sql, correct, reason}]}}},
    "identical_to_<compare_to>": {model: {set: {raw, sql}}}}``.
    """
    names = list(sets)
    flat: list[Mapping[str, Any]] = [it for n in names for it in sets[n]]
    prompts: list[ChatMessages] = [
        list(it["messages"])
        if it.get("messages") is not None
        else build_messages(str(it["question"]), schema_ddl, role="eval_student")
        for it in flat
    ]
    gold = executor.run_batch(db_ref, [str(it["gold_sql"]) for it in flat]) if flat else []
    models: dict[str, Any] = {}
    for model, gen in generators.items():
        outputs = gen.generate(prompts) if prompts else []
        if len(outputs) != len(flat):
            raise RuntimeError(f"{model}: got {len(outputs)} outputs for {len(flat)} items")
        sqls = [extract_sql(o) for o in outputs]
        runnable = [i for i, s in enumerate(sqls) if s is not None]
        outcomes = executor.run_batch(db_ref, [sqls[i] or "" for i in runnable]) if runnable else []
        by_index = dict(zip(runnable, outcomes, strict=True))
        per_set: dict[str, Any] = {}
        pos = 0
        for n in names:
            rows: list[dict[str, Any]] = []
            for it in sets[n]:
                i = pos
                pos += 1
                if sqls[i] is None:
                    ok, why = False, "no SQL extracted from output"
                else:
                    v = compare_outcomes(by_index[i], gold[i], bool(it.get("requires_order")))
                    ok, why = v.ok, v.reason
                rows.append(
                    {"task_id": it.get("task_id"), "raw": outputs[i], "sql": sqls[i],
                     "correct": ok, "reason": why}
                )  # fmt: skip
            per_set[n] = {
                "n": len(rows),
                "accuracy": sum(r["correct"] for r in rows) / len(rows) if rows else 0.0,
                "items": rows,
            }
        models[model] = per_set
    ident: dict[str, Any] = {}
    if compare_to in models:
        for model, per_set in models.items():
            if model == compare_to:
                continue
            ident[model] = {
                n: {
                    f: identical_rate(
                        [r[f] for r in models[compare_to][n]["items"]],
                        [r[f] for r in per_set[n]["items"]],
                    )
                    for f in ("raw", "sql")
                }
                for n in names
            }
    return {"models": models, f"identical_to_{compare_to}": ident}


def diagnose_with_heldout(
    store: Store,
    run_id: str,
    generators: Mapping[str, Generator],
    sets: Mapping[str, Sequence[Mapping[str, Any]]],
    executor: Executor,
    *,
    db_ref: str,
    schema_ddl: str,
) -> dict[str, Any]:
    """``diagnose_generate`` with the sealed held-out added as the set ``"heldout"``. Only task
    ids, outputs and verdicts are returned for it (no held-out question or gold SQL)."""
    if "heldout" in sets:
        raise ValueError("'heldout' is reserved for the sealed set")
    items = store.load_heldout(run_id)
    return diagnose_generate(
        generators, {**sets, "heldout": items}, executor,
        db_ref=db_ref, schema_ddl=schema_ddl,
    )  # fmt: skip
