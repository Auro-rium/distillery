"""The ONLY place chat messages are built for Distillery.

Training rows, student/teacher/base eval requests and any other consumer must call
``build_messages`` so that the (system, user) prefix is byte-identical everywhere; only the
assistant completion differs. Dataset format follows spikes/out/post-training_datasets.md:
"Conversational data" = one JSON object per line with a ``messages`` array of
``{"role", "content"}`` items whose LAST message is the assistant answer. This matches the spec
given to us; there was no conflict with the doc.
"""

from __future__ import annotations

import random
import re
from collections.abc import Mapping, Sequence
from typing import Any, Literal, Protocol

from distillery.taskpacks.sql.sqltext import (
    first_keyword,
    split_statements,
    strip_leading_comments,
)

Role = Literal["train", "eval_student", "eval_teacher", "eval_base"]
ROLES: tuple[Role, ...] = ("train", "eval_student", "eval_teacher", "eval_base")

# Soft switch that disables Qwen3 "thinking" mode. It is appended to the END OF THE USER MESSAGE.
# UNVERIFIED: whether Qwen3 on Nebius Token Factory honours '/no_think' (vs. only the
# `enable_thinking` chat-template kwarg) must be confirmed in the Phase 0 spike. If it does not,
# change this single constant (and the placement in ``_user_content``) and regenerate all data.
NO_THINK_MARKER = "/no_think"

SYSTEM_PROMPT = (
    "You are an expert SQLite analyst. Given a database schema and a question, reply with a "
    "single read-only SQLite SELECT query that answers the question. Output only the SQL, with "
    "no explanation and no markdown fences."
)


class _HasQuestion(Protocol):
    @property
    def question(self) -> str: ...


def _user_content(question: str, schema_ddl: str) -> str:
    return (
        f"Database schema:\n{schema_ddl.strip()}\n\n"
        f"Question: {question.strip()}\n\n"
        f"{NO_THINK_MARKER}"
    )


def build_messages(task_question: str, schema_ddl: str, *, role: Role) -> list[dict[str, str]]:
    """Return the ``[system, user]`` prefix. Identical for every role by construction.

    ``role`` is validated (so call sites are explicit about intent) but deliberately does not
    change the content: any divergence would make eval measure a different prompt than was trained.
    """
    if role not in ROLES:
        raise ValueError(f"unknown role {role!r}; expected one of {ROLES}")
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": _user_content(task_question, schema_ddl)},
    ]


def to_training_row(
    task: _HasQuestion, sql: str, *, schema_ddl: str
) -> dict[str, list[dict[str, str]]]:
    """Chat-format JSONL row: the shared prefix plus the assistant completion (bare SQL)."""
    completion = sql.strip()
    if not completion:
        raise ValueError("empty SQL completion")
    messages = build_messages(task.question, schema_ddl, role="train")
    messages.append({"role": "assistant", "content": completion})
    return {"messages": messages}


# ---- few-shot (workstream B: the fair teacher comparison) -------------------------------------

FEWSHOT_K = 8
FEWSHOT_SEED = 1234  # pre-registered (DECISIONS.md, 2026-10-05)


def build_fewshot_messages(
    question: str, schema_ddl: str, examples: Sequence[Mapping[str, str]]
) -> list[dict[str, str]]:
    """``[system, (user, assistant) x k, user]``: each example is a complete turn pair in EXACTLY
    the zero-shot format of ``build_messages`` (same system prompt, schema, ``/no_think``), so
    the final user turn is byte-identical to the zero-shot request and ``examples=[]`` reproduces
    ``build_messages`` exactly. Each example needs ``question`` and ``sql`` (bare SQL, as in
    ``to_training_row``). Byte-stable: pure string assembly, no randomness, no timestamps.
    """
    messages = build_messages(question, schema_ddl, role="eval_teacher")
    shots: list[dict[str, str]] = []
    for ex in examples:
        sql = str(ex["sql"]).strip()
        if not sql:
            raise ValueError("empty SQL in few-shot example")
        shots.append({"role": "user", "content": _user_content(str(ex["question"]), schema_ddl)})
        shots.append({"role": "assistant", "content": sql})
    return [messages[0], *shots, messages[1]]


def select_fewshot_examples(
    train_items: Sequence[Mapping[str, Any]], k: int = FEWSHOT_K, seed: int = FEWSHOT_SEED
) -> list[dict[str, str]]:
    """The one fixed example set: ``k`` TRAIN items, stratified across families, deterministic.

    Families are sorted then shuffled with ``random.Random(seed)``; within a family the items are
    sorted by ``task_id`` then shuffled with the same generator; examples are taken round-robin
    over the shuffled families, so with more families than ``k`` no family appears twice. The
    result depends only on the train items (never on dev, held-out or human data) and is
    independent of their input order. Items need ``task_id``, ``family``, ``question`` and
    ``gold_sql``; the example completion is the template gold SQL.
    """
    if k < 1:
        raise ValueError("k must be >= 1")
    rng = random.Random(seed)  # noqa: S311 - deterministic selection, not security
    by_family: dict[str, list[Mapping[str, Any]]] = {}
    for it in train_items:
        by_family.setdefault(str(it["family"]), []).append(it)
    if not by_family:
        raise ValueError("no train items to draw examples from")
    families = sorted(by_family)
    rng.shuffle(families)
    pools = {f: sorted(by_family[f], key=lambda it: str(it["task_id"])) for f in families}
    for f in families:
        rng.shuffle(pools[f])
    picked: list[dict[str, str]] = []
    depth = 0
    while len(picked) < k and any(len(p) > depth for p in pools.values()):
        for f in families:
            if len(picked) >= k:
                break
            if len(pools[f]) > depth:
                it = pools[f][depth]
                picked.append(
                    {
                        "task_id": str(it["task_id"]),
                        "family": f,
                        "question": str(it["question"]),
                        "sql": str(it["gold_sql"]).strip(),
                    }
                )
        depth += 1
    return picked


# ---- output extraction -----------------------------------------------------------------------

_THINK_BLOCK = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_FENCE = re.compile(r"```[ \t]*([A-Za-z0-9_+-]*)[ \t]*\n?(.*?)```", re.DOTALL)
_STMT_START = re.compile(
    r"^[ \t]*(?:SELECT\b|WITH\s+(?:RECURSIVE\s+)?\w+\s+AS\s*\(|WITH\s+RECURSIVE\b)",
    re.IGNORECASE | re.MULTILINE,
)
_INLINE_START = re.compile(
    r"\bSELECT\s|\bWITH\s+(?:RECURSIVE\s+)?\w+(?:\s*\([^)]*\))?\s+AS\s*\(", re.IGNORECASE
)


def strip_thinking(text: str) -> str:
    """Remove ``<think>...</think>`` blocks; an unclosed block drops the rest of the text, and a
    stray closing tag (opening tag supplied by the chat template) drops everything before it."""
    text = _THINK_BLOCK.sub("", text)
    low = text.lower()
    close = low.rfind("</think>")
    if close != -1:
        text = text[close + len("</think>") :]
        low = text.lower()
    opened = low.find("<think>")
    if opened != -1:
        text = text[:opened]
    return text


def _last_query(text: str) -> str | None:
    for stmt in reversed(split_statements(text)):
        if first_keyword(stmt) in ("SELECT", "WITH"):
            return strip_leading_comments(stmt).strip()
    return None


def extract_sql(model_output: str) -> str | None:
    """Best-effort SQL from raw model output; ``None`` if there is no SELECT/WITH statement.

    Order: strip thinking; the last fenced block containing a query; else the last blank-line
    separated chunk that contains a bare statement start. The last statement in that region wins,
    and the trailing semicolon is removed. Prose glued directly under SQL with no blank line is
    not separated (the verifier will then reject it, which is the safe failure).
    """
    text = strip_thinking(model_output)
    for m in reversed(list(_FENCE.finditer(text))):
        found = _last_query(m.group(2))
        if found:
            return found
    # A dangling, unterminated fence: take what follows the last opening fence.
    if text.count("```") % 2 == 1:
        found = _last_query(text.rsplit("```", 1)[1].split("\n", 1)[-1])
        if found:
            return found
    unfenced = _FENCE.sub("\n\n", text)
    for chunk in reversed(re.split(r"\n\s*\n", unfenced)):
        m2 = _STMT_START.search(chunk) or _INLINE_START.search(chunk)
        if m2 is None:
            continue
        found = _last_query(chunk[m2.start() :])
        if found:
            return found
    return None
