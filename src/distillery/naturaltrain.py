"""Round N1 (DECISIONS.md 2026-10-07): externally authored, teacher-verified training rows.

``scripts/natural_train.py`` is the CLI. Two steps, both outside any pipeline run:

1. ``draft_natural``: the teacher drafts SQL for every authored question exactly as the human-set
   drafter does (``humanset.draft_questions``: k=3 samples at temperature 0.8 through the shared
   prompt builder). A row is KEPT only if all k drafts execute, are non-empty, agree pairwise AND
   every draft's result equals the result of the author's reference SQL (``compare_outcomes``, the
   verifier the gate uses; ordered iff the reference has a top-level ORDER BY, as the gate's gold
   decides order). The kept row is the pipeline's training row (``to_training_row``) built from the
   teacher's first draft, so it is byte-identical to what teacher_data would have produced.
2. ``check_overlap``: a pure offline report of exact (normalised) and skeleton overlap between a
   ``natural_train.json`` and given sealed question files.

The pipeline appends the kept rows (``PipelineConfig.natural_train_path``) after dropping any
whose normalised question equals a sealed held-out/stress/human, dev or excluded question; that
filter lives in ``evaluator.natural_overlap`` because only the evaluator reads sealed sets.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from distillery import humanset
from distillery.prompts import build_messages, to_training_row
from distillery.store import atomic_write_bytes
from distillery.taskpacks.sql.executor import Executor
from distillery.taskpacks.sql.human import (
    QuestionFile,
    QuestionFileError,
    load_question_file,
    normalise_question,
)
from distillery.taskpacks.sql.sqltext import has_top_level_order_by
from distillery.taskpacks.sql.verifier import compare_outcomes

NATURAL_FORMAT = "distillery-natural-train-v1"
RUN_PREFIX = "naturaltrain-"  # ledger run id of a drafting spend; hidden from /api/runs
DRAFT_PURPOSE = "natural_draft"
DRAFT_STAGE = "draft_natural"
K_CANDIDATES = humanset.K_CANDIDATES  # 3, pre-registered
DRAFT_TEMPERATURE = humanset.DRAFT_TEMPERATURE  # 0.8, pre-registered
REF_REASONS = ("ref_error", "ref_empty", "ref_mismatch")
DISCARD_REASONS = (*humanset.DISCARD_REASONS, *REF_REASONS)
LABEL = "agent-authored"  # DECISIONS.md: never "human"


class NaturalTrainError(ValueError):
    """A natural-train, reference or question file is missing, malformed or inconsistent."""


def file_sha256(path: Path | str) -> str:
    try:
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()
    except OSError as exc:
        raise NaturalTrainError(f"cannot read {path}: {exc}") from None


# ---------------------------------------------------------------- inputs


@dataclass(frozen=True)
class Reference:
    i: Any
    question: str
    reference_sql: str


def load_references(path: Path | str) -> tuple[dict[str, Reference], str]:
    """``reference.jsonl``: one ``{"i", "question", "reference_sql"}`` object per line. Returns
    normalised question -> reference, and the file's sha256. Duplicates are refused."""
    try:
        raw = Path(path).read_bytes()
    except OSError as exc:
        raise NaturalTrainError(f"cannot read reference file: {exc}") from None
    out: dict[str, Reference] = {}
    for lineno, line in enumerate(raw.decode("utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            rec = json.loads(line)
            ref = Reference(rec.get("i"), str(rec["question"]), str(rec["reference_sql"]))
        except (ValueError, KeyError, TypeError, AttributeError):
            raise NaturalTrainError(
                f"reference line {lineno}: need a JSON object with question and reference_sql"
            ) from None
        key = normalise_question(ref.question)
        if key in out:
            raise NaturalTrainError(f"reference line {lineno}: duplicate question {key!r}")
        if not ref.reference_sql.strip():
            raise NaturalTrainError(f"reference line {lineno}: empty reference_sql")
        out[key] = ref
    if not out:
        raise NaturalTrainError("reference file has no rows")
    return out, hashlib.sha256(raw).hexdigest()


def read_question_texts(path: Path | str) -> list[str]:
    """Questions of any file we may need to exclude or check against: a ``human_confirmed.json``
    (``items[].question``), a ``natural_train.json`` (``kept_meta[].question``), a JSON list of
    strings or objects with ``question``, a JSONL of such objects, or a plain question file (one per
    line, the ``draft-heldout`` format). Returned normalised."""
    p = Path(path)
    try:
        raw = p.read_bytes()
    except OSError as exc:
        raise NaturalTrainError(f"cannot read {p}: {exc}") from None
    text = raw.decode("utf-8", errors="strict")

    def from_obj(o: Any) -> str:
        if isinstance(o, str):
            return o
        if isinstance(o, Mapping) and "question" in o:
            return str(o["question"])
        raise NaturalTrainError(f"{p}: an entry has no question")

    try:
        doc = json.loads(text)
    except ValueError:
        doc = None
    items: list[Any] | None = None
    if isinstance(doc, Mapping):
        for key in ("items", "kept_meta", "questions"):
            if isinstance(doc.get(key), list):
                items = list(doc[key])
                break
        if items is None:
            raise NaturalTrainError(f"{p}: JSON file has no items/kept_meta/questions list")
    elif isinstance(doc, list):
        items = doc
    elif text.lstrip().startswith("{"):  # JSONL
        try:
            items = [json.loads(ln) for ln in text.splitlines() if ln.strip()]
        except ValueError:
            items = None
    if items is not None:
        return [normalise_question(from_obj(o)) for o in items]
    try:
        qf = load_question_file(p)
    except QuestionFileError as exc:
        raise NaturalTrainError(f"{p}: {exc}") from None
    return [q.question for q in qf.questions]


# ---------------------------------------------------------------- drafting


def estimate_calls(questions: Sequence[str], schema_ddl: str, k: int = K_CANDIDATES) -> list[int]:
    """Characters of every prompt that drafting will send (k per question), for the preflight."""
    return [
        sum(len(m["content"]) for m in build_messages(q, schema_ddl, role="train"))
        for q in questions
        for _ in range(k)
    ]


def estimate_usd(
    questions: Sequence[str],
    schema_ddl: str,
    price: Callable[[int, int], float],
    *,
    k: int = K_CANDIDATES,
    chars_per_token: int = 3,
    out_tokens_per_call: int = 256,
) -> dict[str, Any]:
    """Budget-basis estimate, the pipeline's own convention (``LLMRunner._estimate``): prompt
    chars / chars_per_token input tokens and ``est_output_tokens`` output tokens per call."""
    chars = estimate_calls(questions, schema_ddl, k)
    in_tok, out_tok = sum(chars) // chars_per_token, out_tokens_per_call * len(chars)
    return {
        "calls": len(chars),
        "input_tokens": in_tok,
        "output_tokens": out_tok,
        "usd": price(in_tok, out_tok),
        "basis": f"prompt chars / {chars_per_token} + {out_tokens_per_call} output tokens per "
        "call x configured teacher price (ESTIMATE)",
    }


def draft_natural(
    qf: QuestionFile,
    refs: Mapping[str, Reference],
    *,
    runner: Any,
    executor: Executor,
    db_ref: str,
    schema_ddl: str,
    db_sha256: str,
    teacher_model: str,
    k: int = K_CANDIDATES,
    temperature: float = DRAFT_TEMPERATURE,
) -> dict[str, Any]:
    """Draft (``humanset.draft_questions``), then keep only rows whose every draft equals the
    reference result. Returns the body of ``natural_train.json`` (without the file shas/spend)."""
    missing = [q.question for q in qf.questions if q.question not in refs]
    if missing:
        raise NaturalTrainError(
            f"{len(missing)} question(s) have no reference SQL, e.g. {missing[0]!r}"
        )
    drafts = humanset.draft_questions(
        qf, runner=runner, executor=executor, db_ref=db_ref, schema_ddl=schema_ddl,
        db_sha256=db_sha256, teacher_model=teacher_model, k=k, temperature=temperature,
    )  # fmt: skip
    kept_drafts: list[dict[str, Any]] = list(drafts["kept"])
    # re-execute every draft next to its reference (one batch): drafts already passed agreement
    sqls: list[str] = []
    for d in kept_drafts:
        sqls += [*d["candidates"], refs[d["question"]].reference_sql]
    outs = executor.run_batch(db_ref, sqls) if sqls else []
    discards: Counter[str] = Counter({r: int(n) for r, n in drafts["discarded_by_reason"].items()})
    discarded = [
        {"task_id": d["task_id"], "question": d["question"], "reason": d["reason"]}
        for d in drafts["discarded"]
    ]
    kept: list[dict[str, Any]] = []
    meta: list[dict[str, Any]] = []
    step = k + 1
    for n, d in enumerate(kept_drafts):
        cand_out, ref_out = outs[n * step : n * step + k], outs[n * step + k]
        ref = refs[d["question"]]
        reason = keep_reason(cand_out, ref_out, has_top_level_order_by(ref.reference_sql))
        if reason is not None:
            discards[reason] += 1
            discarded.append({"task_id": d["task_id"], "question": d["question"], "reason": reason})
            continue
        sql = str(d["candidates"][0])
        kept.append(to_training_row(_Q(d["question"]), sql, schema_ddl=schema_ddl))
        meta.append(
            {
                "i": ref.i,
                "task_id": d["task_id"],
                "question": d["question"],
                "sql": sql,
                "reference_sql": ref.reference_sql,
            }
        )
    return {
        "format": NATURAL_FORMAT,
        "label": LABEL,
        "db_sha256": db_sha256,
        "teacher_model": teacher_model,
        "k": k,
        "temperature": temperature,
        "n_questions": len(qf.questions),
        "duplicates_dropped": qf.duplicates_dropped,
        "kept": kept,
        "kept_meta": meta,
        "discards": {r: int(discards.get(r, 0)) for r in DISCARD_REASONS},
        "discarded": discarded,
    }


@dataclass(frozen=True)
class _Q:
    question: str


def keep_reason(cands: Sequence[Any], ref: Any, ref_ordered: bool) -> str | None:
    """None = keep. ``cands`` are the (already agreeing, non-empty) draft outcomes."""
    if not ref.ok:
        return "ref_error"
    if not ref.rows:
        return "ref_empty"
    if not all(compare_outcomes(c, ref, ref_ordered).ok for c in cands):
        return "ref_mismatch"
    return None


def write_natural(path: Path | str, doc: Mapping[str, Any]) -> Path:
    p = Path(path)
    atomic_write_bytes(
        p, (json.dumps(doc, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode("utf-8")
    )
    return p


# ---------------------------------------------------------------- reading (pipeline side)


@dataclass(frozen=True)
class NaturalSet:
    rows: tuple[dict[str, Any], ...]
    questions: tuple[str, ...]  # normalised, parallel to rows
    sqls: tuple[str, ...]
    db_sha256: str
    file_sha256: str
    questions_sha256: str
    reference_sha256: str


def read_natural(path: Path | str) -> NaturalSet:
    """Parse and validate ``natural_train.json`` (raises ``NaturalTrainError``)."""
    try:
        raw = Path(path).read_bytes()
    except OSError as exc:
        raise NaturalTrainError(f"cannot read natural train file: {exc}") from None
    try:
        doc = json.loads(raw)
    except ValueError:
        raise NaturalTrainError("natural train file is not valid JSON") from None
    if not isinstance(doc, dict) or doc.get("format") != NATURAL_FORMAT:
        raise NaturalTrainError(f"natural train file must have format {NATURAL_FORMAT!r}")
    kept, meta = doc.get("kept"), doc.get("kept_meta")
    if not isinstance(kept, list) or not isinstance(meta, list) or len(kept) != len(meta):
        raise NaturalTrainError("natural train file needs parallel kept and kept_meta lists")
    if not kept:
        raise NaturalTrainError("natural train file has no kept rows")
    for row, m in zip(kept, meta, strict=True):
        if (
            not isinstance(row, dict)
            or not isinstance(row.get("messages"), list)
            or not isinstance(m, dict)
            or "question" not in m
            or "sql" not in m
        ):
            raise NaturalTrainError("a natural train row is malformed")
    db_sha = doc.get("db_sha256")
    if not isinstance(db_sha, str) or not db_sha:
        raise NaturalTrainError("natural train file has no db_sha256")
    return NaturalSet(
        rows=tuple(dict(r) for r in kept),
        questions=tuple(normalise_question(str(m["question"])) for m in meta),
        sqls=tuple(str(m["sql"]) for m in meta),
        db_sha256=db_sha,
        file_sha256=hashlib.sha256(raw).hexdigest(),
        questions_sha256=str(doc.get("questions_sha256", "")),
        reference_sha256=str(doc.get("reference_sha256", "")),
    )


# ---------------------------------------------------------------- overlap


def overlap_counts(
    natural: Iterable[str], sealed: Iterable[str], skeleton: Callable[[str], str]
) -> dict[str, Any]:
    """Exact (normalised) matches and the held-out ``skeleton_in_train`` measure: how many sealed
    questions have a skeleton that also occurs among the natural (training) questions."""
    nat = [normalise_question(q) for q in natural]
    sea = [normalise_question(q) for q in sealed]
    nat_set = set(nat)
    nat_sk = {skeleton(q) for q in nat}
    exact = sorted({q for q in sea if q in nat_set})
    sk_in = sum(skeleton(q) in nat_sk for q in sea)
    return {
        "sealed_n": len(sea),
        "exact_overlap": len(exact),
        "exact_overlap_questions": exact,
        "skeleton_in_train": sk_in,
        "skeleton_in_train_rate": sk_in / len(sea) if sea else 0.0,
    }


def check_overlap(
    natural_path: Path | str, sealed_paths: Sequence[Path | str], skeleton: Callable[[str], str]
) -> dict[str, Any]:
    ns = read_natural(natural_path)
    return {
        "natural_train_sha256": ns.file_sha256,
        "natural_rows": len(ns.rows),
        "sets": {
            str(p): {"sha256": file_sha256(p), **overlap_counts(ns.questions,
                                                                read_question_texts(p), skeleton)}
            for p in sealed_paths
        },
    }  # fmt: skip
