"""``ToolcallPack``: the tool-call task pack as one object.

Thin adapters over env / questions / prompts / verifier / executor, shaped after the method list in
the plan's C0 ``Pack`` protocol (``build_env``, ``generate_tasks``, ``families``,
``stress_families``, ``skeleton``, ``build_messages``, ``to_training_row``, ``extract_answer``,
``executor``, ``compare``, ``corrupt``, ``analysis_prompt``, ``triage_prompt``,
``answer_label``). Wiring into the orchestrator is deliberately not done here.
"""

from __future__ import annotations

import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from distillery.prompts import Role
from distillery.taskpacks.sql.verifier import VerifyResult
from distillery.taskpacks.toolcall import env as E
from distillery.taskpacks.toolcall import prompts as P
from distillery.taskpacks.toolcall import questions as Q
from distillery.taskpacks.toolcall import verifier as V
from distillery.taskpacks.toolcall.executor import ExecOutcome, LocalExecutor


@dataclass(frozen=True)
class ToolEnv:
    ref: str
    context_text: str


class ToolcallPack:
    name = "toolcall"
    answer_label = P.ANSWER_LABEL
    student_max_new_tokens = P.STUDENT_MAX_NEW_TOKENS

    def __init__(self, executor: LocalExecutor | None = None) -> None:
        self.executor = executor or LocalExecutor()

    # environment
    def build_env(self, seed: int) -> ToolEnv:
        state = E.build_state(seed)
        ref = E.env_ref(seed)
        self.executor.register(ref, state)
        return ToolEnv(ref, E.render_context(state))

    def state(self, env_ref: str) -> E.State:
        return E.build_state(E.parse_env_ref(env_ref))

    # tasks
    @property
    def families(self) -> tuple[str, ...]:
        return Q.FAMILIES

    def stress_families(self, families: tuple[str, ...] | None = None) -> tuple[str, ...]:
        return Q.stress_families(families)

    def generate_tasks(self, env_ref: str, n: int, seed: int, **kw: Any) -> Q.GenerationReport:
        return Q.generate_tasks(
            self.state(env_ref), n, seed, state_key=E.parse_env_ref(env_ref), **kw
        )

    def skeleton(self, question: str) -> str:
        return Q.skeleton(question)

    # prompts and rows
    def build_messages(self, goal: str, context_text: str, *, role: Role) -> list[dict[str, str]]:
        return P.build_messages(goal, context_text, role=role)

    def to_training_row(self, task: Any, answer: str, *, context_text: str) -> dict[str, Any]:
        return P.to_training_row(task, answer, context_text=context_text)

    def extract_answer(self, model_output: str) -> str | None:
        return P.extract_answer(model_output)

    def gold_answer(self, task: Q.ToolTask) -> str:
        return task.gold_answer

    # verification
    def compare(
        self, candidate: ExecOutcome, gold: ExecOutcome, requires_order: bool = False
    ) -> VerifyResult:
        return V.compare_outcomes(candidate, gold, requires_order)

    def corrupt(
        self, task: Q.ToolTask, env_ref: str, rng: random.Random, *, max_variants: int = 8
    ) -> list[V.Corruption]:
        return V.corrupt_calls(task.gold_calls, self.state(env_ref), rng, max_variants=max_variants)

    def selftest(
        self, tasks: Sequence[Q.ToolTask], env_ref: str, *, seed: int, sample: int = 50, k: int = 4
    ) -> dict[str, Any]:
        return V.selftest(
            tasks, env_ref, self.executor, self.state(env_ref), seed=seed, sample=sample, k=k
        )

    # analyst prompts
    def analysis_prompt(
        self, failures: Sequence[Mapping[str, Any]], allowed_families: Sequence[str]
    ) -> list[dict[str, str]]:
        return P.build_analysis_messages(failures, allowed_families)

    def triage_prompt(self, raw: str) -> str:
        return P.triage_prompt(raw)
