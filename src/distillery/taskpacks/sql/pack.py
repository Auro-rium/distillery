"""The text-to-SQL pack: today's functions behind the ``Pack`` interface (no behaviour change).

Every method delegates to the module that always owned the behaviour (``questions``, ``schema``,
``verifier``, ``prompts``); nothing is re-implemented here, so the SQL dry-run report is
byte-stable across the refactor.
"""

from __future__ import annotations

import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from distillery import paraphrase, prompts
from distillery.store import atomic_write_bytes, sha256_hex
from distillery.taskpacks.base import BatchExecutor, ChatMessages, Env, GenReport, Role
from distillery.taskpacks.sql import questions as q
from distillery.taskpacks.sql import schema as sql_schema
from distillery.taskpacks.sql.executor import LocalExecutor
from distillery.taskpacks.sql.runner import ExecOutcome, run_select
from distillery.taskpacks.sql.verifier import VerifyResult, compare_outcomes, corrupt_sql


@dataclass(frozen=True)
class SqlCorruption:
    kind: str
    answer: str


class SqlPack:
    name = "sql"
    language = "sql"
    answer_label = "SQL"
    gold_field = "gold_sql"
    answer_key = "sql"
    student_max_new_tokens = 160  # S4 recipe
    default_skeleton_cap: int | None = q.DEFAULT_SKELETON_CAP
    dry_wrong_answer = "SELECT 1"

    def local_executor(self) -> BatchExecutor:
        return LocalExecutor()

    def paraphrase_messages(self, question: str, n: int) -> ChatMessages:
        return paraphrase.paraphrase_messages(question, n)

    def build_env(self, seed: int, dest: Path) -> Env:
        data = sql_schema.build_database(seed)
        atomic_write_bytes(dest, data)
        return Env(str(dest), sql_schema.schema_ddl(seed), sha256_hex(data), len(data))

    def families(self) -> tuple[str, ...]:
        return q.FAMILIES

    def stress_families(self) -> tuple[str, ...]:
        return q.stress_families()

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
    ) -> GenReport:
        return q.generate_tasks(
            env_ref,
            n,
            seed,
            families=families,
            exclude_families=exclude_families,
            skeleton_cap=skeleton_cap,
            template_share=q.DEFAULT_TEMPLATE_SHARE if template_share is None else template_share,
            family_share=q.DEFAULT_FAMILY_SHARE if family_share is None else family_share,
        )

    def task_from(self, d: Mapping[str, Any]) -> q.SqlTask:
        return q.SqlTask.model_validate(dict(d))

    def skeleton(self, question: str) -> str:
        return q.skeleton(question)

    def build_messages(self, question: str, context_text: str, *, role: Role) -> ChatMessages:
        return prompts.build_messages(question, context_text, role=role)

    def to_training_row(
        self, task: Any, answer: str, *, context_text: str
    ) -> dict[str, list[dict[str, str]]]:
        return prompts.to_training_row(task, answer, schema_ddl=context_text)

    def extract_answer(self, model_output: str) -> str | None:
        return prompts.extract_sql(model_output)

    def compare(self, candidate: Any, gold: Any, order_matters: bool) -> VerifyResult:
        return compare_outcomes(candidate, gold, order_matters)

    def gold_problem(self, gold: ExecOutcome) -> str | None:
        if not gold.ok:
            return "gold_error"
        return None if gold.rows else "gold_empty"

    def run_local(self, env_ref: str, answer: str) -> ExecOutcome:
        return run_select(env_ref, answer)

    def corrupt(
        self, gold_answer: str, env_ref: str, rng: random.Random, *, max_variants: int = 8
    ) -> list[SqlCorruption]:
        return [
            SqlCorruption(c.kind, c.sql)
            for c in corrupt_sql(gold_answer, env_ref, rng, max_variants=max_variants)
        ]

    def analysis_prompt(
        self, failures: Sequence[Mapping[str, Any]], allowed_families: Sequence[str]
    ) -> ChatMessages:
        """Prompt for the planner-role failure analyst. Takes DEV failures only (by construction:
        the caller passes the dev-eval failure list; no other data source is reachable here)."""
        lines = []
        for f in failures:
            lines.append(
                f"- [family={f['family']}] Q: {f['question']}\n"
                f"  gold: {f['gold_sql']}\n  student: {f['student_sql']}\n  verdict: {f['reason']}"
            )
        system = (
            "You analyse failures of a small text-to-SQL model on a development set. Group the "
            "failures into a few clusters of related mistakes. For each cluster give a short "
            "name, a description, and the question families (from the allowed list only) whose "
            "extra training examples would fix it."
        )
        user = f"Allowed families: {', '.join(allowed_families)}\n\nFailures:\n" + "\n".join(lines)
        return [{"role": "system", "content": system}, {"role": "user", "content": user}]

    def triage_prompt(self, raw_output: str) -> ChatMessages:
        return [
            {
                "role": "user",
                "content": "Is the following model output ONLY a single SQL query, with no "
                f"prose or markdown fences?\n\n{raw_output}",
            }
        ]


PACK = SqlPack()
