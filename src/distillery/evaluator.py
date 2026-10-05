"""Sealed held-out evaluation. The ONLY module allowed to call ``Store.load_heldout`` (and the
stress and human siblings).

Scores base, student and teacher on the same held-out items in the same order, verifies the
evaluated adapter is the trained one, and hands the per-item outcomes to the pure gate.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from distillery.config import GateThresholds
from distillery.finetune import TrainedArtifact, sha256_file
from distillery.gate import GateResult, evaluate_gate
from distillery.humanset import SELECTION_BIAS_NOTE, HumanSetError, read_confirmed
from distillery.misses import classify_miss
from distillery.prompts import extract_sql
from distillery.store import Store, canonical_json, sha256_hex
from distillery.taskpacks.base import Pack, get_pack
from distillery.taskpacks.sql.executor import Executor
from distillery.taskpacks.sql.human import normalise_question
from distillery.taskpacks.sql.runner import ExecOutcome
from distillery.taskpacks.sql.verifier import compare_outcomes

ChatMessages = list[dict[str, str]]
MODEL_ROLES = ("base", "student", "teacher")


def _pack(pack: Pack | None) -> Pack:
    """Every scoring entry point takes ``pack``; ``None`` means the SQL pack (the pipeline always
    passes its own, so only the SQL-only diagnostics and legacy callers rely on the default).
    ``schema_ddl`` below is the pack's env ``context_text`` under its historical name."""
    return get_pack("sql") if pack is None else pack


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
    # Stress set (reserved families, never in train/dev/gate): scored after the gate, reported
    # separately, never an input of evaluate_gate. None = no stress set was sealed for this run.
    stress: dict[str, Any] | None = None
    # Human held-out set (Gate B): scored with the SAME thresholds as the gate, reported beside it,
    # never an input of Gate A. None = no human set was sealed for this run.
    human: dict[str, Any] | None = None

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
            "stress": self.stress,
            "human": self.human,
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
    pack: Pack | None = None,
) -> ModelScores:
    if role not in MODEL_ROLES:
        raise ValueError(f"unknown model role {role!r}")
    pk = _pack(pack)
    batch = [
        pk.build_messages(str(it["question"]), schema_ddl, role=f"eval_{role}")  # type: ignore[arg-type]
        for it in items
    ]
    outputs = generator.generate(batch)
    if len(outputs) != len(items):
        raise RuntimeError(f"{role}: got {len(outputs)} outputs for {len(items)} items")
    sqls = [pk.extract_answer(o) for o in outputs]
    runnable = [i for i, s in enumerate(sqls) if s is not None]
    outcomes = executor.run_batch(db_ref, [sqls[i] or "" for i in runnable])
    by_index = dict(zip(runnable, outcomes, strict=True))
    correct: list[bool] = []
    reasons: list[str] = []
    for i, it in enumerate(items):
        if sqls[i] is None:
            correct.append(False)
            reasons.append(f"no {pk.answer_label} extracted from output")
            continue
        verdict = pk.compare(by_index[i], gold[i], bool(it["requires_order"]))
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
    items: Sequence[Mapping[str, Any]],
    scores: Mapping[str, ModelScores],
    pack: Pack | None = None,
) -> list[dict[str, Any]]:
    """Deterministic capped examples in held-out order: the first ``EXAMPLES_PER_KIND`` of each
    kind. ``fixed`` = student ok, base wrong; ``regressed`` = base ok, student wrong;
    ``still_wrong`` = both wrong. Items both models solve are omitted."""

    pk = _pack(pack)
    key = pk.answer_key

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
                f"gold_{key}": it[pk.gold_field],
                f"base_{key}": shown("base", i),
                f"student_{key}": shown("student", i),
                f"teacher_{key}": shown("teacher", i),
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
    pack: Pack | None = None,
) -> EvalReport:
    """Evaluate base/student/teacher on the sealed held-out set and run the gate."""
    pk = _pack(pack)
    if set(generators) != set(MODEL_ROLES):
        raise ValueError(f"generators must be exactly {MODEL_ROLES}")
    if (trained is None) != (expected is None):
        raise ValueError("pass both `trained` and `expected`, or neither")
    if trained is not None and expected is not None:
        verify_artifact(trained, expected)

    items = store.load_heldout(run_id)
    heldout_sha = sha256_hex(canonical_json(items).encode("utf-8"))
    gold = executor.run_batch(db_ref, [str(it[pk.gold_field]) for it in items])
    for it, g in zip(items, gold, strict=True):
        if not g.ok:
            raise RuntimeError(
                f"gold answer failed for held-out item {it.get('task_id')}: {g.error}"
            )

    scores = _score_roles(
        generators, items, gold, executor,
        db_ref=db_ref, schema_ddl=schema_ddl, on_scores=on_scores, pack=pk,
    )  # fmt: skip

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
        examples=build_examples(items, scores, pk),
        stress=_score_stress(
            store, run_id, generators, executor, db_ref=db_ref, schema_ddl=schema_ddl, pack=pk
        ),
        human=_score_human(
            store, run_id, generators, executor,
            db_ref=db_ref, schema_ddl=schema_ddl, gate_cfg=gate_cfg, pack=pk,
        ),
    )  # fmt: skip


def _score_roles(
    generators: Mapping[str, Generator],
    items: Sequence[Mapping[str, Any]],
    gold: Sequence[ExecOutcome],
    executor: Executor,
    *,
    db_ref: str,
    schema_ddl: str,
    on_scores: Callable[[str, ModelScores], None] | None,
    pack: Pack | None = None,
) -> dict[str, ModelScores]:
    """Score base, student and teacher concurrently: the models are independent (the base and the
    student each run in their own sandbox), so serving them one after another only adds wall
    time. The sandbox layer's global in-flight semaphore keeps the operation count under the cap.
    ``on_scores`` is called on the calling thread, in completion order."""
    scores: dict[str, ModelScores] = {}
    with ThreadPoolExecutor(max_workers=len(MODEL_ROLES), thread_name_prefix="score") as pool:
        futures = {
            pool.submit(
                score_model,
                generators[role],
                items,
                gold,
                schema_ddl=schema_ddl,
                db_ref=db_ref,
                executor=executor,
                role=role,
                pack=pack,
            ): role
            for role in MODEL_ROLES
        }
        for fut in as_completed(futures):
            role = futures[fut]
            scores[role] = fut.result()  # a failure in any model aborts the evaluation
            if on_scores is not None:
                on_scores(role, scores[role])
    return {role: scores[role] for role in MODEL_ROLES}  # stable order


def _score_stress(
    store: Store,
    run_id: str,
    generators: Mapping[str, Generator],
    executor: Executor,
    *,
    db_ref: str,
    schema_ddl: str,
    pack: Pack | None = None,
) -> dict[str, Any] | None:
    """Accuracy of every model on the sealed stress set, per model and per family. Runs after the
    gate decision is computed and shares nothing with it: the stress set cannot move the gate."""
    items = store.load_stress(run_id)
    if items is None:
        return None
    pk = _pack(pack)
    gold = executor.run_batch(db_ref, [str(it[pk.gold_field]) for it in items])
    for it, g in zip(items, gold, strict=True):
        if not g.ok:
            raise RuntimeError(f"gold answer failed for stress item {it.get('task_id')}: {g.error}")
    scores = _score_roles(
        generators, items, gold, executor,
        db_ref=db_ref, schema_ddl=schema_ddl, on_scores=None, pack=pk,
    )  # fmt: skip
    families = sorted({str(it["family"]) for it in items})
    by_family = {
        f: {
            m: sum(scores[m].correct[i] for i, it in enumerate(items) if it["family"] == f)
            / sum(1 for it in items if it["family"] == f)
            for m in MODEL_ROLES
        }
        for f in families
    }
    return {
        "sha256": sha256_hex(canonical_json(items).encode("utf-8")),
        "n": len(items),
        "families": families,
        "accuracy": {m: s.accuracy for m, s in scores.items()},
        "accuracy_by_family": by_family,
        "unparseable": {m: s.unparseable for m, s in scores.items()},
        "note": "reserved families, never in train/dev/gate; reported separately, not a gate input",
    }


def _score_human(
    store: Store,
    run_id: str,
    generators: Mapping[str, Generator],
    executor: Executor,
    *,
    db_ref: str,
    schema_ddl: str,
    gate_cfg: GateThresholds,
    pack: Pack | None = None,
) -> dict[str, Any] | None:
    """Gate B: every model on the sealed human set, then ``evaluate_gate`` a second time with the
    SAME thresholds object as Gate A. Shares nothing with Gate A or the stress set, so neither can
    move it. Examples follow the same capped, post-evaluation rule as the gate set's."""
    items = store.load_human(run_id)
    if items is None:
        return None
    pk = _pack(pack)
    gold = executor.run_batch(db_ref, [str(it[pk.gold_field]) for it in items])
    for it, g in zip(items, gold, strict=True):
        if not g.ok:
            raise RuntimeError(f"gold answer failed for human item {it.get('task_id')}: {g.error}")
    scores = _score_roles(
        generators, items, gold, executor,
        db_ref=db_ref, schema_ddl=schema_ddl, on_scores=None, pack=pk,
    )  # fmt: skip
    gate = evaluate_gate(
        scores["base"].correct, scores["student"].correct, scores["teacher"].correct, gate_cfg
    )
    return {
        "sha256": sha256_hex(canonical_json(items).encode("utf-8")),
        "n": len(items),
        "accuracy": {m: s.accuracy for m, s in scores.items()},
        "unparseable": {m: s.unparseable for m, s in scores.items()},
        "gate": gate.model_dump(mode="json"),
        "examples": build_examples(items, scores, pk),
        "note": SELECTION_BIAS_NOTE,
    }


def score_sealed_human(
    store: Store,
    run_id: str,
    generators: Mapping[str, Generator],
    executor: Executor,
    *,
    db_ref: str,
    schema_ddl: str,
    gate_cfg: GateThresholds,
    trained: TrainedArtifact,
    expected: ExpectedArtifact,
    pack: Pack | None = None,
) -> dict[str, Any]:
    """Gate B for an ALREADY finished run: verify the adapter on disk is still the trained one,
    then score the sealed human set once with the same gate rule as ``evaluate``. Raises if no
    human set is sealed for the run."""
    if set(generators) != set(MODEL_ROLES):
        raise ValueError(f"generators must be exactly {MODEL_ROLES}")
    verify_artifact(trained, expected)
    block = _score_human(
        store, run_id, generators, executor,
        db_ref=db_ref, schema_ddl=schema_ddl, gate_cfg=gate_cfg, pack=_pack(pack),
    )  # fmt: skip
    if block is None:
        raise HumanSetError("no human set is sealed for this run")
    return block


def human_set_fingerprint(path: Path | str) -> str:
    """sha256 of the confirmed-file bytes, for the split stage's cache key. Only a digest leaves
    this module; the orchestrator never sees the contents."""
    try:
        return sha256_hex(Path(path).read_bytes())
    except OSError as exc:
        raise HumanSetError(f"cannot read human set file: {exc}") from None


def seal_human_set(
    store: Store,
    run_id: str,
    path: Path | str,
    db_sha: str,
    *,
    train_questions: Iterable[str],
    dev_questions: Iterable[str],
    pack: Pack | None = None,
) -> dict[str, Any]:
    """Verify the confirmed human set against this run's database, drop every question that exactly
    matches (whitespace-normalised) a train, dev or gate question, and seal the rest. Returns hash
    and counts ONLY: no question or gold SQL leaves this function."""
    confirmed = read_confirmed(path)
    if confirmed.db_sha256 != db_sha:
        raise HumanSetError(
            "human set was drafted against a different database "
            f"({confirmed.db_sha256[:12]} != this run's {db_sha[:12]})"
        )
    train = {normalise_question(x) for x in train_questions}
    dev = {normalise_question(x) for x in dev_questions}
    gate = {normalise_question(str(it["question"])) for it in store.load_heldout(run_id)}
    dropped = {"train": 0, "dev": 0, "gate": 0}
    kept: list[dict[str, Any]] = []
    for it in confirmed.items:
        norm = normalise_question(str(it["question"]))
        hit = (
            "train" if norm in train else "dev" if norm in dev else "gate" if norm in gate else None
        )
        if hit is not None:
            dropped[hit] += 1
        else:
            kept.append(dict(it))
    if not kept:
        raise HumanSetError("every human question matches a train, dev or gate question")
    pk = _pack(pack)
    train_skeletons = {pk.skeleton(x) for x in train}
    digest = store.seal_human(run_id, kept)
    return {
        "sha256": digest,
        "file_sha256": confirmed.file_sha256,
        "n": len(kept),
        "dropped_exact_overlap": {**dropped, "total": sum(dropped.values())},
        "skeleton_in_train": sum(
            pk.skeleton(str(it["question"])) in train_skeletons for it in kept
        ),
        "counts": confirmed.counts,
        "db_sha256": confirmed.db_sha256,
        "question_file_sha256": confirmed.question_file_sha256,
        "teacher_model": confirmed.teacher_model,
        "note": SELECTION_BIAS_NOTE,
    }


# ---------------------------------------------------------------- B2: fair teacher comparison

VECTOR_FORMAT = "distillery-setup-vectors-v1"
PromptBuilder = Callable[[Mapping[str, Any]], ChatMessages]  # item -> the exact prompt to send


@dataclass(frozen=True)
class Setup:
    """One thing to score: a generator plus how it is prompted. ``generator`` may be ``None``
    only when re-scoring cached outputs (no model is called)."""

    name: str
    generator: Generator | None
    prompt: PromptBuilder


def _judge(
    items: Sequence[Mapping[str, Any]],
    gold: Sequence[ExecOutcome],
    outputs: Sequence[str],
    executor: Executor,
    db_ref: str,
    *,
    set_name: str,
    setup: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Execute and verify one setup's raw outputs. Returns the per-item block (vectors) and the
    pre-classified misses (``misses.classify_miss``: ids, class and result shapes only)."""
    sqls = [extract_sql(o) for o in outputs]
    runnable = [i for i, q in enumerate(sqls) if q is not None]
    outcomes = executor.run_batch(db_ref, [sqls[i] or "" for i in runnable]) if runnable else []
    by_index = dict(zip(runnable, outcomes, strict=True))
    correct: list[bool] = []
    reasons: list[str] = []
    misses: list[dict[str, Any]] = []
    for i, it in enumerate(items):
        cand = by_index.get(i)
        if cand is None:
            correct.append(False)
            reasons.append("no SQL extracted from output")
        else:
            verdict = compare_outcomes(cand, gold[i], bool(it["requires_order"]))
            correct.append(verdict.ok)
            reasons.append(verdict.reason)
        if not correct[-1]:
            misses.append(
                {
                    "set": set_name, "setup": setup, "task_id": it.get("task_id"),
                    "family": it.get("family"), "verifier_reason": reasons[-1],
                    **classify_miss(cand, gold[i], bool(it["requires_order"])),
                }
            )  # fmt: skip
    block = {
        "correct": correct,
        "reasons": reasons,
        "sqls": sqls,
        "outputs": list(outputs),
        "unparseable": sum(1 for q in sqls if q is None),
        "accuracy": sum(correct) / len(correct) if correct else 0.0,
    }
    return block, misses


def _score_set(
    items: Sequence[Mapping[str, Any]],
    setups: Sequence[Setup],
    executor: Executor,
    *,
    db_ref: str,
    schema_ddl: str,
    set_name: str,
    cached: Mapping[str, Any] | None,
    label: str,
) -> dict[str, Any]:
    """Generate (or take ``cached`` outputs) and judge every setup on one item set, in item order.
    Generation runs concurrently across setups (they are independent models/endpoints)."""
    gold = executor.run_batch(db_ref, [str(it["gold_sql"]) for it in items])
    for it, g in zip(items, gold, strict=True):
        if not g.ok:
            raise RuntimeError(f"gold SQL failed for {label} item {it.get('task_id')}: {g.error}")
    task_ids = [str(it.get("task_id")) for it in items]
    outputs: dict[str, list[str]] = {}
    if cached is not None:
        if cached.get("task_ids") != task_ids:
            raise ValueError(f"{label}: cached task ids do not match the current item set")
        for st in setups:
            outputs[st.name] = [str(o) for o in cached["setups"][st.name]["outputs"]]
    else:

        def gen(st: Setup) -> list[str]:
            if st.generator is None:
                raise ValueError(f"setup {st.name!r} has no generator")
            outs = st.generator.generate([st.prompt(it) for it in items])
            if len(outs) != len(items):
                raise RuntimeError(f"{st.name}: got {len(outs)} outputs for {len(items)} items")
            return list(outs)

        with ThreadPoolExecutor(
            max_workers=max(1, len(setups)), thread_name_prefix="setup"
        ) as pool:
            futures = {pool.submit(gen, st): st.name for st in setups}
            for fut in as_completed(futures):
                outputs[futures[fut]] = fut.result()  # any failure aborts the whole pass
    blocks: dict[str, Any] = {}
    misses: list[dict[str, Any]] = []
    for st in setups:  # stable order
        blocks[st.name], m = _judge(
            items, gold, outputs[st.name], executor, db_ref, set_name=set_name, setup=st.name
        )
        misses.extend(m)
    return {
        "status": "scored",
        "n": len(items),
        "sha256": sha256_hex(canonical_json(list(items)).encode("utf-8")),
        "task_ids": task_ids,
        "setups": blocks,
        "misses": misses,
    }


PENDING_HUMAN = {
    "status": "pending",
    "reason": "no human set is sealed for this run (Gate B never ran); nothing was scored on it",
}


def score_open_set(
    items: Sequence[Mapping[str, Any]],
    setups: Sequence[Setup],
    executor: Executor,
    *,
    db_ref: str,
    schema_ddl: str,
    set_name: str = "dev",
    cached: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """The same scoring on an UNSEALED item set (dev): per-item vectors for every setup."""
    return _score_set(
        items, setups, executor, db_ref=db_ref, schema_ddl=schema_ddl, set_name=set_name,
        cached=cached, label=set_name,
    )  # fmt: skip


def score_setups(
    store: Store,
    run_id: str,
    setups: Sequence[Setup],
    executor: Executor,
    *,
    db_ref: str,
    schema_ddl: str,
    cached: Mapping[str, Any] | None = None,
    on_set: Callable[[str, dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Score arbitrary setups (base, student, teachers with any prompt) on the sealed held-out set
    and, when one is sealed, the human set. Returns a JSON-able payload with the PER-ITEM vectors,
    outputs, extracted SQL, verifier reasons and pre-classified misses (task ids and result shapes,
    never question or gold text), so the gate can be recomputed offline and nothing is lost
    (plan gap 8). ``cached`` is a previous payload: its outputs are re-executed and re-verified
    with NO generator call (the B3 re-score path). A missing human set is the explicit marker
    ``{"status": "pending"}``. This is the only sealed-set access of the fair comparison."""
    if len({st.name for st in setups}) != len(setups):
        raise ValueError("setup names must be unique")
    payload: dict[str, Any] = {
        "format": VECTOR_FORMAT, "run_id": run_id, "setups": [st.name for st in setups],
        "sets": {},
    }  # fmt: skip
    for name, items in (
        ("heldout", store.load_heldout(run_id)),
        ("human", store.load_human(run_id)),
    ):
        if items is None:
            payload["sets"][name] = dict(PENDING_HUMAN)
            continue
        prev = ((cached or {}).get("sets") or {}).get(name)
        if cached is not None and (prev is None or prev.get("status") != "scored"):
            raise ValueError(f"cached payload has no scored {name!r} set to re-score")
        block = _score_set(
            items, setups, executor, db_ref=db_ref, schema_ddl=schema_ddl, set_name=name,
            cached=prev if cached is not None else None, label=name,
        )  # fmt: skip
        if prev is not None and cached is not None and prev.get("sha256") != block["sha256"]:
            raise ValueError(f"cached {name!r} set was scored on a different item set")
        payload["sets"][name] = block
        if on_set is not None:
            on_set(name, block)
    return payload


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
    pack: Pack | None = None,
) -> dict[str, Any]:
    """Generate with every model over every named item set and score it.

    Each item needs ``task_id``, ``gold_sql``, ``requires_order`` and either ``messages`` (the exact
    prompt to send) or ``question`` (prompt built with the eval prompt builder). All sets go to a
    model in ONE ``generate`` call, so each model (sandbox image + weights) is loaded once. Returns
    ``{"models": {model: {set: {n, accuracy, items[{task_id, raw, sql, correct, reason}]}}},
    "identical_to_<compare_to>": {model: {set: {raw, sql}}}}``.
    """
    pk = _pack(pack)
    names = list(sets)
    flat: list[Mapping[str, Any]] = [it for n in names for it in sets[n]]
    prompts: list[ChatMessages] = [
        list(it["messages"])
        if it.get("messages") is not None
        else pk.build_messages(str(it["question"]), schema_ddl, role="eval_student")
        for it in flat
    ]
    gold = executor.run_batch(db_ref, [str(it[pk.gold_field]) for it in flat]) if flat else []
    models: dict[str, Any] = {}
    for model, gen in generators.items():
        outputs = gen.generate(prompts) if prompts else []
        if len(outputs) != len(flat):
            raise RuntimeError(f"{model}: got {len(outputs)} outputs for {len(flat)} items")
        sqls = [pk.extract_answer(o) for o in outputs]
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
                    v = pk.compare(by_index[i], gold[i], bool(it.get("requires_order")))
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
