"""The ONLY place chat messages are built for Distillery.

Training rows, student/teacher/base eval requests and any other consumer must call
``build_messages`` so that the (system, user) prefix is byte-identical everywhere; only the
assistant completion differs. Dataset format follows spikes/out/post-training_datasets.md:
"Conversational data" = one JSON object per line with a ``messages`` array of
``{"role", "content"}`` items whose LAST message is the assistant answer. This matches the spec
given to us; there was no conflict with the doc.
"""

from __future__ import annotations

import re
from typing import Literal, Protocol

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
