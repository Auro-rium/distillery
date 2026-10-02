"""Human held-out set: drafting gold SQL, the confirm decisions and the confirmed file.

Everything here runs OFFLINE, before any pipeline run (the pipeline cannot block on a human):

1. ``draft_questions``: the teacher drafts gold SQL for each human question (k samples, one
   temperature); a draft is kept only when every sample executes, is non-empty and the samples
   agree.
2. ``confirm_drafts`` / ``decide`` / ``finalize``: a human sees question, gold SQL and result rows
   and marks each draft correct or not; decisions are appended to ``decisions.jsonl`` (resumable,
   auditable, last decision per id wins).
3. ``human_confirmed.json`` holds only the confirmed items. The pipeline hands its PATH to the
   evaluator, which alone reads it (``evaluator.seal_human_set``).

Nothing in this module may be named like a sealed-set loader: the integrity test scans the source
for the Store's human-set reader and allows it only in ``store.py`` and ``evaluator.py``.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

from distillery.prompts import build_messages, extract_sql
from distillery.store import atomic_write_bytes
from distillery.taskpacks.sql.executor import Executor
from distillery.taskpacks.sql.human import QuestionFile
from distillery.taskpacks.sql.runner import ExecOutcome
from distillery.taskpacks.sql.sqltext import has_top_level_order_by
from distillery.taskpacks.sql.verifier import compare_outcomes

DRAFTS_FORMAT = "distillery-human-drafts-v1"
CONFIRMED_FORMAT = "distillery-human-confirmed-v1"
HUMAN_FAMILY = "human"
HUMAN_SOURCE = "human"
HUMAN_CLASS = "human"  # the ``heldout_class`` of a sealed human item
K_CANDIDATES = 3
DRAFT_TEMPERATURE = 0.8
DISCARD_REASONS = ("llm_error", "no_sql", "exec_error", "empty", "disagree")
PREVIEW_ROWS = 10
DECISIONS = ("confirm", "reject", "skip")
DRAFT_PURPOSE = "human_draft"
DRAFT_STAGE = "draft_heldout"
RUN_PREFIX = "humanset-"  # ledger run id of a drafting spend; hidden from /api/runs
SET_ID_RE = re.compile(r"[0-9a-f]{8}")
SELECTION_BIAS_NOTE = (
    "Selection bias, disclosed: gold SQL was drafted by the teacher and kept only when every "
    "drafted sample executed, was non-empty and agreed with the others, then confirmed by a human. "
    "Only teacher-solvable questions survive, so teacher accuracy here is high by construction and "
    "hard questions are under-represented. Discard and rejection counts are reported."
)


class HumanSetError(ValueError):
    """A human-set file is missing, malformed or does not belong to this database."""


class _Runner(Protocol):
    def map(
        self,
        role: str,
        batch: Sequence[Sequence[Mapping[str, Any]]],
        *,
        purpose: str,
        stage: str,
        temperature: float | None = ...,
    ) -> list[Any]: ...


# ---------------------------------------------------------------- paths and ids


def valid_set_id(set_id: str) -> bool:
    return bool(SET_ID_RE.fullmatch(set_id))


def set_dir(root: Path | str, set_id: str) -> Path:
    if not valid_set_id(set_id):
        raise HumanSetError("set id must be 8 lowercase hex characters")
    return Path(root) / "humanset" / set_id


def list_sets(root: Path | str) -> list[str]:
    base = Path(root) / "humanset"
    if not base.is_dir():
        return []
    return sorted(
        p.name for p in base.iterdir() if valid_set_id(p.name) and (p / "drafts.json").is_file()
    )


# ---------------------------------------------------------------- drafting


def _cell(v: Any) -> Any:
    return v.hex() if isinstance(v, bytes) else v


def _preview(o: ExecOutcome) -> dict[str, Any]:
    return {
        "columns": list(o.columns),
        "rows": [[_cell(c) for c in r] for r in o.rows[:PREVIEW_ROWS]],
        "row_count": len(o.rows),
    }


def draft_questions(
    qf: QuestionFile,
    *,
    runner: _Runner,
    executor: Executor,
    db_ref: str,
    schema_ddl: str,
    db_sha256: str,
    teacher_model: str,
    k: int = K_CANDIDATES,
    temperature: float = DRAFT_TEMPERATURE,
) -> dict[str, Any]:
    """Teacher drafts gold SQL: ``k`` samples per question at ``temperature`` through the shared
    prompt builder. Kept only if all samples execute OK, return rows, and agree pairwise under
    ``compare_outcomes`` (ordered only when every sample has a top-level ORDER BY). Every discard is
    recorded with its reason."""
    questions = list(qf.questions)
    batch = [
        build_messages(q.question, schema_ddl, role="train") for q in questions for _ in range(k)
    ]
    results = runner.map(
        "teacher", batch, purpose=DRAFT_PURPOSE, stage=DRAFT_STAGE, temperature=temperature
    )
    if len(results) != len(batch):
        raise RuntimeError(f"runner returned {len(results)} results for {len(batch)} prompts")
    texts = [None if r is None else str(r.text) for r in results]
    sqls = [None if t is None else extract_sql(t) for t in texts]
    runnable = [i for i, s in enumerate(sqls) if s is not None]
    outs_run = executor.run_batch(db_ref, [sqls[i] or "" for i in runnable]) if runnable else []
    outcome: dict[int, ExecOutcome] = dict(zip(runnable, outs_run, strict=True))

    kept: list[dict[str, Any]] = []
    discarded: list[dict[str, Any]] = []
    for qi, q in enumerate(questions):
        idx = range(qi * k, qi * k + k)
        cand = [sqls[i] for i in idx]
        reason: str | None = None
        if any(texts[i] is None for i in idx):
            reason = "llm_error"
        elif any(s is None for s in cand):
            reason = "no_sql"
        elif any(not outcome[i].ok for i in idx):
            reason = "exec_error"
        elif any(len(outcome[i].rows) == 0 for i in idx):
            reason = "empty"
        requires_order = reason is None and all(has_top_level_order_by(str(s)) for s in cand)
        if reason is None:
            first = outcome[idx[0]]
            if not all(
                compare_outcomes(outcome[idx[a]], outcome[idx[b]], requires_order).ok
                for a in range(k)
                for b in range(a + 1, k)
            ):
                reason = "disagree"
        if reason is not None:
            discarded.append(
                {
                    "task_id": q.task_id,
                    "question": q.question,
                    "reason": reason,
                    "candidates": [s for s in cand if s is not None],
                }
            )
            continue
        kept.append(
            {
                "task_id": q.task_id,
                "question": q.question,
                "gold_sql": str(cand[0]),
                "requires_order": requires_order,
                "candidates": [str(s) for s in cand],
                "preview": _preview(first),
            }
        )
    by_reason = Counter(d["reason"] for d in discarded)
    return {
        "format": DRAFTS_FORMAT,
        "question_file_sha256": qf.file_sha256,
        "db_sha256": db_sha256,
        "teacher_model": teacher_model,
        "temperature": temperature,
        "k": k,
        "n_questions": len(questions),
        "duplicates_dropped": qf.duplicates_dropped,
        "kept": kept,
        "discarded": discarded,
        "discarded_by_reason": {r: by_reason.get(r, 0) for r in DISCARD_REASONS},
    }


def _write_json(path: Path, doc: Mapping[str, Any]) -> None:
    atomic_write_bytes(
        path, (json.dumps(doc, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode("utf-8")
    )


def write_drafts(root: Path | str, drafts: Mapping[str, Any]) -> Path:
    """``<root>/humanset/<question-file-sha8>/drafts.json``."""
    path = set_dir(root, str(drafts["question_file_sha256"])[:8]) / "drafts.json"
    _write_json(path, drafts)
    return path


def read_drafts(root: Path | str, set_id: str) -> dict[str, Any]:
    path = set_dir(root, set_id) / "drafts.json"
    try:
        doc = json.loads(path.read_bytes())
    except (OSError, ValueError):
        raise HumanSetError(f"no drafts for set {set_id}") from None
    if not isinstance(doc, dict) or doc.get("format") != DRAFTS_FORMAT:
        raise HumanSetError(f"drafts file has the wrong format (want {DRAFTS_FORMAT!r})")
    return doc


# ---------------------------------------------------------------- decisions


def read_decisions(root: Path | str, set_id: str) -> dict[str, str]:
    """task_id -> last recorded decision (``confirm`` / ``reject`` / ``skip``)."""
    path = set_dir(root, set_id) / "decisions.jsonl"
    out: dict[str, str] = {}
    if not path.is_file():
        return out
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            rec = json.loads(line)
            tid, dec = str(rec["task_id"]), str(rec["decision"])
        except (ValueError, KeyError, TypeError):
            raise HumanSetError("decisions.jsonl has a malformed line") from None
        if dec in DECISIONS:
            out[tid] = dec
    return out


def decide(root: Path | str, set_id: str, task_id: str, decision: str) -> None:
    """Append one decision (shared by the CLI and the admin endpoint)."""
    if decision not in DECISIONS:
        raise HumanSetError(f"decision must be one of {DECISIONS}")
    drafts = read_drafts(root, set_id)
    if task_id not in {k["task_id"] for k in drafts["kept"]}:
        raise HumanSetError(f"unknown draft id {task_id!r}")
    path = set_dir(root, set_id) / "decisions.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    rec = {"task_id": task_id, "decision": decision, "at": datetime.now(UTC).isoformat()}
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(rec, sort_keys=True) + "\n")


def tally(drafts: Mapping[str, Any], decisions: Mapping[str, str]) -> dict[str, int]:
    last = [decisions.get(k["task_id"]) for k in drafts["kept"]]
    return {
        "confirmed": last.count("confirm"),
        "rejected": last.count("reject"),
        "skipped": last.count("skip"),
        "undecided": last.count(None),
    }


def format_draft(item: Mapping[str, Any], position: int, total: int) -> list[str]:
    prev = item["preview"]
    cols = [str(c) for c in prev["columns"]]
    rows = [[str(c) for c in r] for r in prev["rows"]]
    widths = [
        max(len(c), *(len(r[i]) for r in rows)) if rows else len(c) for i, c in enumerate(cols)
    ]
    lines = [
        f"--- {position}/{total}  {item['task_id']} ---",
        f"Question: {item['question']}",
        f"Gold SQL: {item['gold_sql']}",
        f"Result ({prev['row_count']} rows, first {len(rows)}):",
        "  " + " | ".join(c.ljust(w) for c, w in zip(cols, widths, strict=True)),
    ]
    lines += ["  " + " | ".join(c.ljust(w) for c, w in zip(r, widths, strict=True)) for r in rows]
    return lines


def confirm_drafts(
    root: Path | str,
    set_id: str,
    *,
    ask: Callable[[str], str],
    out: Callable[[str], None],
) -> dict[str, Any]:
    """Interactive confirm loop. Items already confirmed or rejected are not asked again (skipped
    ones are). Keys: y = correct, n = wrong, s = skip for now, q = quit (EOF also quits)."""
    drafts = read_drafts(root, set_id)
    decisions = read_decisions(root, set_id)
    kept: list[dict[str, Any]] = drafts["kept"]
    pending = [k for k in kept if decisions.get(k["task_id"]) in (None, "skip")]
    quit_ = False
    for item in pending:
        for line in format_draft(item, kept.index(item) + 1, len(kept)):
            out(line)
        while True:
            try:
                key = ask("correct? [y]es / [n]o / [s]kip / [q]uit: ").strip().lower()
            except EOFError:
                key = "q"
            if key in ("y", "n", "s", "q"):
                break
            out("please answer y, n, s or q")
        if key == "q":
            quit_ = True
            break
        decide(root, set_id, item["task_id"], {"y": "confirm", "n": "reject", "s": "skip"}[key])
    return {**tally(drafts, read_decisions(root, set_id)), "quit": quit_}


def finalize(root: Path | str, set_id: str) -> Path:
    """Write ``human_confirmed.json`` with only the items marked correct, plus counts."""
    drafts = read_drafts(root, set_id)
    decisions = read_decisions(root, set_id)
    items = [
        {
            "task_id": k["task_id"],
            "family": HUMAN_FAMILY,
            "source": HUMAN_SOURCE,
            "heldout_class": HUMAN_CLASS,
            "question": k["question"],
            "gold_sql": k["gold_sql"],
            "requires_order": bool(k["requires_order"]),
            "tables": [],
            "template": "",
        }
        for k in drafts["kept"]
        if decisions.get(k["task_id"]) == "confirm"
    ]
    if not items:
        raise HumanSetError("no confirmed items: nothing to finalize")
    counts = {
        "questions": int(drafts["n_questions"]),
        "duplicates_dropped": int(drafts.get("duplicates_dropped", 0)),
        "kept": len(drafts["kept"]),
        "discarded": len(drafts["discarded"]),
        "discarded_by_reason": dict(drafts["discarded_by_reason"]),
        **tally(drafts, decisions),
    }
    path = set_dir(root, set_id) / "human_confirmed.json"
    _write_json(
        path,
        {
            "format": CONFIRMED_FORMAT,
            "db_sha256": drafts["db_sha256"],
            "question_file_sha256": drafts["question_file_sha256"],
            "teacher_model": drafts["teacher_model"],
            "counts": counts,
            "items": items,
        },
    )
    return path


# ---------------------------------------------------------------- the confirmed file


@dataclass(frozen=True)
class ConfirmedSet:
    items: tuple[dict[str, Any], ...]
    db_sha256: str
    question_file_sha256: str
    teacher_model: str
    counts: dict[str, Any]
    file_sha256: str


def read_confirmed(path: Path | str) -> ConfirmedSet:
    """Parse and validate ``human_confirmed.json`` (raises ``HumanSetError``)."""
    try:
        raw = Path(path).read_bytes()
    except OSError as exc:
        raise HumanSetError(f"cannot read human set file: {exc}") from None
    try:
        doc = json.loads(raw)
    except ValueError:
        raise HumanSetError("human set file is not valid JSON") from None
    if not isinstance(doc, dict) or doc.get("format") != CONFIRMED_FORMAT:
        raise HumanSetError(f"human set file must have format {CONFIRMED_FORMAT!r}")
    items = doc.get("items")
    if not isinstance(items, list):
        raise HumanSetError("human set file has no items list")
    if not items:
        raise HumanSetError("human set file has no confirmed items")
    required = ("task_id", "question", "gold_sql", "requires_order", "family")
    ids: set[str] = set()
    for it in items:
        if not isinstance(it, dict) or any(k not in it for k in required):
            raise HumanSetError(f"human set item is missing one of {required}")
        if str(it["task_id"]) in ids:
            raise HumanSetError(f"duplicate human task id {it['task_id']!r}")
        ids.add(str(it["task_id"]))
    db_sha = doc.get("db_sha256")
    if not isinstance(db_sha, str) or not db_sha:
        raise HumanSetError("human set file has no db_sha256")
    return ConfirmedSet(
        items=tuple(dict(it) for it in items),
        db_sha256=db_sha,
        question_file_sha256=str(doc.get("question_file_sha256", "")),
        teacher_model=str(doc.get("teacher_model", "")),
        counts=dict(doc.get("counts") or {}),
        file_sha256=hashlib.sha256(raw).hexdigest(),
    )
