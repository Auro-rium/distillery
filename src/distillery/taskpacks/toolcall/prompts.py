# ruff: noqa: E501
"""Prompts, answer extraction and training rows for the tool-call pack.

The (system, user) prefix is built in ONE place (``build_messages``) and is identical for every
role, like the SQL pack's. The answer is a fenced JSON list of calls; ``extract_answer`` is
forgiving about everything around it (thinking blocks, prose, missing or unterminated fences)
and returns the canonical JSON text of the call list, or None.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from typing import Any

from distillery.prompts import NO_THINK_MARKER, ROLES, Role, strip_thinking
from distillery.taskpacks.toolcall import env as E

STUDENT_MAX_NEW_TOKENS = 384  # per-pack student generation budget (the SQL pack uses 160)
ANSWER_LABEL = "tool calls"

SYSTEM_PROMPT = (
    "You are an operations assistant for a SaaS admin console. Given the available tools, the "
    "current system state and a goal, reply with the exact list of tool calls that achieves the "
    "goal, in the order they must run. Reply with a single fenced ```json block holding a JSON "
    'list of {"tool": ..., "args": {...}} objects. Use only ids that appear in the state. Make '
    "no call that is not needed; if the goal cannot or need not be done, reply with []. "
    "No explanation."
)


def build_messages(goal: str, context_text: str, *, role: Role) -> list[dict[str, str]]:
    """The ``[system, user]`` prefix. Identical for every role by construction."""
    if role not in ROLES:
        raise ValueError(f"unknown role {role!r}; expected one of {ROLES}")
    user = f"{context_text.strip()}\n\nGoal: {goal.strip()}\n\n{NO_THINK_MARKER}"
    return [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user}]


def format_answer(answer: str) -> str:
    """The assistant completion for a canonical call list: a fenced JSON block."""
    return f"```json\n{answer.strip()}\n```"


def to_training_row(
    task: Any, answer: str, *, context_text: str
) -> dict[str, list[dict[str, str]]]:
    """Chat-format JSONL row: the shared prefix plus the fenced gold calls."""
    if not answer.strip():
        raise ValueError("empty answer completion")
    messages = build_messages(task.question, context_text, role="train")
    messages.append({"role": "assistant", "content": format_answer(answer)})
    return {"messages": messages}


# ---- extraction ------------------------------------------------------------------------------

_FENCE = re.compile(r"```[ \t]*([A-Za-z0-9_+-]*)[ \t]*\n?(.*?)```", re.DOTALL)
_DECODER = json.JSONDecoder()


def _as_calls(value: Any, *, allow_empty: bool) -> list[Any] | None:
    if isinstance(value, dict) and "tool" in value:
        value = [value]
    if not isinstance(value, list):
        return None
    if not value:
        return [] if allow_empty else None
    if all(isinstance(c, dict) and "tool" in c for c in value):
        return value
    return None


def _scan(text: str, *, allow_empty: bool) -> list[Any] | None:
    """The last JSON list/object in ``text`` that is a call list (balanced, via raw_decode)."""
    found: list[Any] | None = None
    i = 0
    while i < len(text):
        if text[i] in "[{":
            try:
                val, end = _DECODER.raw_decode(text, i)
            except ValueError:
                i += 1
                continue
            calls = _as_calls(val, allow_empty=allow_empty)
            if calls is not None:
                found = calls
            i = end
        else:
            i += 1
    return found


def _canonical(calls: list[Any]) -> str:
    ok = all(isinstance(c.get("args"), dict) and set(c) == {"tool", "args"} for c in calls)
    return E.canonical_calls(calls) if ok else json.dumps(calls, sort_keys=True)


def extract_answer(model_output: str) -> str | None:
    """Canonical JSON text of the call list in ``model_output``, or None if there is none.

    Order: strip thinking; the last fenced block that holds a call list (``[]`` counts inside a
    fence); a dangling unterminated fence; else the last call list found anywhere in the text
    (a bare ``[]`` only counts when it is the entire output). Structure is not validated beyond
    "objects with a tool"; the verifier rejects bad calls.
    """
    text = strip_thinking(model_output)
    for m in reversed(list(_FENCE.finditer(text))):
        calls = _scan(m.group(2), allow_empty=True)
        if calls is None and m.group(2).strip() == "[]":
            calls = []
        if calls is not None:
            return _canonical(calls)
    if text.count("```") % 2 == 1:
        tail = text.rsplit("```", 1)[1].split("\n", 1)[-1]
        calls = _scan(tail, allow_empty=tail.strip() == "[]")
        if calls is not None:
            return _canonical(calls)
    unfenced = _FENCE.sub("\n", text)
    calls = _scan(unfenced, allow_empty=unfenced.strip() == "[]")
    return None if calls is None else _canonical(calls)


# ---- analyst / triage prompts -----------------------------------------------------------------


def build_analysis_messages(
    failures: Sequence[Mapping[str, Any]], allowed_families: Sequence[str]
) -> list[dict[str, str]]:
    """Failure-analyst prompt (dev failures only, passed in by the caller)."""
    lines = []
    for f in failures:
        gold = f.get("gold_answer", f.get("gold_sql", ""))
        got = f.get("student_answer", f.get("student_sql", ""))
        lines.append(
            f"- [family={f['family']}] Goal: {f['question']}\n"
            f"  gold calls: {gold}\n  student calls: {got}\n  verdict: {f['reason']}"
        )
    system = (
        "You analyse failures of a small model that must emit tool-call sequences for a SaaS admin "
        "console on a development set. Group the failures into a few clusters of related "
        "mistakes (wrong ids, missing prerequisite calls, wrong order, ignored preconditions, ...). "
        "For each cluster give a short name, a description, and the task families (from the "
        "allowed list only) whose extra training examples would fix it."
    )
    user = f"Allowed families: {', '.join(allowed_families)}\n\nFailures:\n" + "\n".join(lines)
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def triage_prompt(raw: str) -> str:
    return (
        "Is the following model output ONLY a single fenced ```json block containing a JSON list "
        f"of tool calls, with no prose?\n\n{raw}"
    )
