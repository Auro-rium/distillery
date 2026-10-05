"""Local executor for tool-call answers: replay a JSON call list on the seeded state.

Same shape as the SQL executor: ``run_batch(env_ref, answers)`` returns one ``ExecOutcome`` per
answer, in order, and never raises for a bad answer. An answer is the JSON text of a call list
(what ``extract_answer`` returns). A successful outcome carries the canonical final state as its
single cell (``columns=("final_state",)``), so the SQL pack's ``ExecOutcome`` is reused as is.
Nothing but JSON is interpreted here: no model-written code is ever executed, and no sandbox is
needed for verification.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Protocol

from distillery.taskpacks.sql.runner import ExecOutcome
from distillery.taskpacks.toolcall import env as E

__all__ = ["ExecOutcome", "Executor", "LocalExecutor", "run_answer"]


def run_answer(state: E.State, answer: str) -> ExecOutcome:
    """Parse ``answer`` (a JSON list of calls) and replay it on ``state``."""
    try:
        calls = json.loads(answer)
    except (ValueError, RecursionError) as e:
        return ExecOutcome(False, error_kind="syntax", error=f"not valid JSON: {e}")
    if not isinstance(calls, list):
        return ExecOutcome(False, error_kind="syntax", error="answer must be a JSON list of calls")
    res = E.replay(state, calls)
    if not res.ok:
        return ExecOutcome(False, error_kind="runtime", error=res.error)
    return ExecOutcome(True, columns=("final_state",), rows=((E.canonical_state(res.state),),))


class Executor(Protocol):
    def run_batch(self, env_ref: str, answers: Sequence[str]) -> list[ExecOutcome]: ...


class LocalExecutor:
    """In-process executor. ``env_ref`` is a key in ``states`` or else ``toolcall:<seed>``."""

    def __init__(self, states: Mapping[str, E.State] | None = None) -> None:
        self._states: dict[str, E.State] = dict(states or {})

    def register(self, env_ref: str, state: E.State) -> None:
        self._states[env_ref] = state

    def _state(self, env_ref: str) -> E.State:
        if env_ref not in self._states:
            self._states[env_ref] = E.build_state(E.parse_env_ref(env_ref))
        return self._states[env_ref]

    def run_batch(self, env_ref: str, answers: Sequence[str]) -> list[ExecOutcome]:
        state = self._state(env_ref)
        return [run_answer(state, a) for a in answers]
