"""``ToolcallPack``: the tool-call task pack behind the ``taskpacks.base.Pack`` interface.

Thin adapters over env / questions / prompts / verifier / executor. The env is the seeded SaaS
state: ``Env.ref`` is ``toolcall:<seed>`` (the executor rebuilds the state from it), the file at
``dest`` is the canonical state JSON (the reproducibility record: sha256 and size). An answer is
the canonical JSON text of a call list; its outcome carries the canonical final state, so the
pipeline's comparison is "same final state as the gold" (call order matters only through effect).
"""

from __future__ import annotations

import json
import random
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from distillery.store import atomic_write_bytes, sha256_hex
from distillery.taskpacks.base import BatchExecutor, ChatMessages, Env, Role
from distillery.taskpacks.sql.verifier import VerifyResult
from distillery.taskpacks.toolcall import env as E
from distillery.taskpacks.toolcall import prompts as P
from distillery.taskpacks.toolcall import questions as Q
from distillery.taskpacks.toolcall import verifier as V
from distillery.taskpacks.toolcall.executor import ExecOutcome, LocalExecutor


class ToolcallPack:
    name = "toolcall"
    language = "json"
    answer_label = P.ANSWER_LABEL
    gold_field = "gold_answer"
    answer_key = "answer"
    student_max_new_tokens = P.STUDENT_MAX_NEW_TOKENS
    default_skeleton_cap: int | None = 40  # the SQL pack's per-skeleton cap (same rule)
    dry_wrong_answer = '[{"tool": "no_such_tool", "args": {}}]'  # always a replay error

    def __init__(self, executor: LocalExecutor | None = None) -> None:
        self.executor = executor or LocalExecutor()

    def local_executor(self) -> BatchExecutor:
        return LocalExecutor()

    # environment
    def build_env(self, seed: int, dest: Path) -> Env:
        state = E.build_state(seed)
        data = (E.canonical_state(state) + "\n").encode("utf-8")
        atomic_write_bytes(dest, data)
        ref = E.env_ref(seed)
        self.executor.register(ref, state)
        return Env(ref, E.render_context(state), sha256_hex(data), len(data))

    def state(self, env_ref: str) -> E.State:
        return E.build_state(E.parse_env_ref(env_ref))

    # tasks
    def families(self) -> tuple[str, ...]:
        return Q.FAMILIES

    def stress_families(self) -> tuple[str, ...]:
        return Q.stress_families()

    def generate_tasks(
        self,
        env_ref: str,
        n: int,
        seed: int,
        *,
        families: tuple[str, ...] | None = None,
        exclude_families: tuple[str, ...] = (),
        skeleton_cap: int | None = None,
        template_share: float | None = None,
        family_share: float | None = None,
    ) -> Q.GenerationReport:
        return Q.generate_tasks(
            self.state(env_ref),
            n,
            seed,
            families=families,
            exclude_families=exclude_families,
            skeleton_cap=skeleton_cap,
            template_share=Q.DEFAULT_TEMPLATE_SHARE if template_share is None else template_share,
            family_share=Q.DEFAULT_FAMILY_SHARE if family_share is None else family_share,
            state_key=E.parse_env_ref(env_ref),
        )

    def task_from(self, d: Mapping[str, Any]) -> Q.ToolTask:
        return Q.ToolTask.model_validate(dict(d))

    def skeleton(self, question: str) -> str:
        return Q.skeleton(question)

    # prompts and rows
    def build_messages(self, question: str, context_text: str, *, role: Role) -> ChatMessages:
        return P.build_messages(question, context_text, role=role)

    def to_training_row(
        self, task: Any, answer: str, *, context_text: str
    ) -> dict[str, list[dict[str, str]]]:
        return P.to_training_row(task, answer, context_text=context_text)

    def extract_answer(self, model_output: str) -> str | None:
        return P.extract_answer(model_output)

    # verification
    def compare(self, candidate: Any, gold: Any, order_matters: bool = False) -> VerifyResult:
        return V.compare_outcomes(candidate, gold, order_matters)

    def gold_problem(self, gold: ExecOutcome) -> str | None:
        return None if gold.ok else "gold_error"  # an empty call list is a legal gold (guards)

    def run_local(self, env_ref: str, answer: str) -> ExecOutcome:
        return self.local_executor().run_batch(env_ref, [answer])[0]  # type: ignore[no-any-return]

    def corrupt(
        self, gold_answer: str, env_ref: str, rng: random.Random, *, max_variants: int = 8
    ) -> list[V.Corruption]:
        return V.corrupt_calls(
            list(json.loads(gold_answer)), self.state(env_ref), rng, max_variants=max_variants
        )

    def selftest(
        self, tasks: Sequence[Q.ToolTask], env_ref: str, *, seed: int, sample: int = 50, k: int = 4
    ) -> dict[str, Any]:
        """Standalone zero-tolerance self-test; refusals surface as the orchestrator's
        ``VerifierSelfTestError`` (the pipeline stage itself uses ``corrupt``/``compare``)."""
        from distillery.orchestrator import VerifierSelfTestError

        try:
            return V.selftest(
                tasks, env_ref, self.executor, self.state(env_ref), seed=seed, sample=sample, k=k
            )
        except V.SelfTestError as e:
            raise VerifierSelfTestError(str(e)) from e

    # LLM prompts
    def analysis_prompt(
        self, failures: Sequence[Mapping[str, Any]], allowed_families: Sequence[str]
    ) -> ChatMessages:
        return P.build_analysis_messages(failures, allowed_families)

    def paraphrase_messages(self, question: str, n: int) -> ChatMessages:
        return P.paraphrase_messages(question, n)

    def triage_prompt(self, raw_output: str) -> ChatMessages:
        return [{"role": "user", "content": P.triage_prompt(raw_output)}]


PACK = ToolcallPack()
