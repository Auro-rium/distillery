"""The task-pack interface: everything the pipeline needs to know about a task domain.

A *pack* owns a domain end to end: how an environment (the thing answers run against) is built,
how tasks are generated and split into families, how a prompt and a training row look, how a model
output is turned into an answer, how that answer is executed and compared with the gold answer,
and how the verifier is attacked in the self-test. The orchestrator, evaluator and dry-run fakes
talk to ``Pack`` only; they never import a pack's internals. ``taskpacks/sql/pack.py`` is the
reference implementation; ``get_pack(name)`` is the registry.

Stability contract (other packs are built against this file): additions must be optional or
defaulted; renaming or removing a member is a breaking change. Words used below:

* **env** - the deterministic, seeded world a task set is asked about (SQL: a SQLite database).
  ``Env.ref`` is the handle an executor needs; ``Env.context_text`` is shown to the model
  (SQL: the schema DDL).
* **answer** - a string the model produces and the executor runs (SQL: one SELECT; tool-call: a
  JSON list of calls).
* **outcome** - what executing an answer returns. Opaque to the pipeline except for the three
  attributes in ``Outcome``; ``compare`` is the only judge of equality.
* **task** - pydantic-like object with ``task_id``, ``family``, ``question``, ``gold_answer`` and
  ``order_matters`` (also serialisable with ``model_dump(mode="json")``; ``task_from`` inverts it).
* **item** - the persisted dict form of a task. Its gold field is named ``gold_field`` (SQL:
  ``"gold_sql"``; kept so stored runs and reports stay byte-stable) and it carries the key
  ``requires_order``.
"""

from __future__ import annotations

import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Protocol, runtime_checkable

Role = Literal["train", "eval_student", "eval_teacher", "eval_base"]
ChatMessages = list[dict[str, str]]

# held-out classes of the sealed items (orchestrator-level, pack independent)
IN_DISTRIBUTION = "in_distribution"
STRESS = "stress"


@dataclass(frozen=True)
class Env:
    """A materialised environment: ``ref`` for the executor, ``context_text`` for the prompt,
    plus the sha256 and size of what was written (the reproducibility record)."""

    ref: str
    context_text: str
    sha256: str = ""
    size: int = 0


class Outcome(Protocol):
    """What the pipeline may read from an execution result (the pack's own type has more)."""

    @property
    def ok(self) -> bool: ...
    @property
    def error(self) -> str | None: ...
    @property
    def error_kind(self) -> str | None: ...


class Verdict(Protocol):
    @property
    def ok(self) -> bool: ...
    @property
    def reason(self) -> str: ...


class Corruption(Protocol):
    """A deliberately wrong but executable variant of a gold answer."""

    @property
    def kind(self) -> str: ...
    @property
    def answer(self) -> str: ...


class PackTask(Protocol):
    @property
    def task_id(self) -> str: ...
    @property
    def family(self) -> str: ...
    @property
    def question(self) -> str: ...
    @property
    def gold_answer(self) -> str: ...
    @property
    def order_matters(self) -> bool: ...
    def model_dump(self, *, mode: str = ...) -> dict[str, Any]: ...
    def model_copy(
        self, *, update: Mapping[str, Any] | None = ..., deep: bool = ...
    ) -> PackTask: ...


class GenReport(Protocol):
    """Generated tasks plus drop counters (the orchestrator copies these into its report)."""

    @property
    def tasks(self) -> Sequence[PackTask]: ...
    @property
    def attempts(self) -> int: ...
    @property
    def dropped_error(self) -> int: ...
    @property
    def dropped_empty(self) -> int: ...
    @property
    def dropped_duplicate(self) -> int: ...
    @property
    def dropped_duplicate_question(self) -> int: ...
    @property
    def dropped_skeleton_cap(self) -> int: ...
    @property
    def dropped_by_template(self) -> Mapping[str, int]: ...


class BatchExecutor(Protocol):
    """Runs answers against an env: exactly one outcome per answer, in order, never raising for
    a bad answer (the error goes in the outcome)."""

    def run_batch(self, env_ref: str, answers: Sequence[str]) -> list[Any]: ...


@runtime_checkable
class Pack(Protocol):
    # ---- identity / presentation
    name: str  # registry key, also the value of --pack
    language: str  # code-block language for the UI ("sql", "json")
    answer_label: str  # human noun for one answer ("SQL", "tool calls")
    gold_field: str  # key of the gold answer in persisted items ("gold_sql")
    answer_key: str  # suffix of per-model example fields: gold_<key>, student_<key> ("sql")
    student_max_new_tokens: int  # default generation cap for the student on this pack
    default_skeleton_cap: int | None  # per-skeleton cap the question stage passes to generation
    dry_wrong_answer: str  # an executable answer the verifier rejects for ANY gold (dry-run fakes)

    # ---- environment
    def local_executor(self) -> BatchExecutor:
        """An in-process executor for this pack (dry runs and tests; live runs use a sandbox)."""
        ...

    def build_env(self, seed: int, dest: Path) -> Env:
        """Deterministically build the env for ``seed`` at ``dest`` (atomic write) and return it."""
        ...

    # ---- tasks
    def families(self) -> tuple[str, ...]: ...
    def stress_families(self) -> tuple[str, ...]:
        """Families reserved for the stress set (a fixed, seeded sample of ``families``)."""
        ...

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
        """Up to ``n`` distinct tasks whose gold answer runs; ``None`` shares mean pack default."""
        ...

    def task_from(self, d: Mapping[str, Any]) -> PackTask: ...
    def skeleton(self, question: str) -> str:
        """Question with every literal masked; equal skeletons are near-duplicates."""
        ...

    # ---- prompts and data (the ONLY place chat messages are built)
    def build_messages(self, question: str, context_text: str, *, role: Role) -> ChatMessages:
        """``[system, user]`` prefix, identical for every role."""
        ...

    def to_training_row(
        self, task: PackTask, answer: str, *, context_text: str
    ) -> dict[str, list[dict[str, str]]]: ...
    def extract_answer(self, model_output: str) -> str | None:
        """Best-effort answer from raw output; ``None`` when there is none."""
        ...

    # ---- execution and verification
    def compare(self, candidate: Any, gold: Any, order_matters: bool) -> Verdict: ...
    def gold_problem(self, gold: Any) -> str | None:
        """Why a gold outcome is unusable ("gold_error", "gold_empty", ...), else ``None``."""
        ...

    def run_local(self, env_ref: str, answer: str) -> Any:
        """Run one answer in-process (no sandbox); the self-test's reference executor."""
        ...

    def corrupt(
        self, gold_answer: str, env_ref: str, rng: random.Random, *, max_variants: int = 8
    ) -> list[Corruption]: ...

    # ---- LLM prompts specific to the domain
    def analysis_prompt(
        self, failures: Sequence[Mapping[str, Any]], allowed_families: Sequence[str]
    ) -> ChatMessages: ...
    def paraphrase_messages(self, question: str, n: int) -> ChatMessages:
        """Prompt asking for ``n`` paraphrases of one train question (instruction is the pack's)."""
        ...

    def triage_prompt(self, raw_output: str) -> ChatMessages:
        """Ask the triage model whether a raw output is a clean, bare answer."""
        ...


_REGISTRY: dict[str, str] = {
    "sql": "distillery.taskpacks.sql.pack",
    "toolcall": "distillery.taskpacks.toolcall.pack",
}


def pack_names() -> tuple[str, ...]:
    return tuple(sorted(_REGISTRY))


def get_pack(name: str) -> Pack:
    """The pack registered under ``name`` (imported lazily); ``ValueError`` for an unknown name."""
    import importlib

    module = _REGISTRY.get(name)
    if module is None:
        raise ValueError(f"unknown pack {name!r}; expected one of {pack_names()}")
    pack: Pack = importlib.import_module(module).PACK
    return pack


def register_pack(name: str, module: str) -> None:
    """Register ``module`` (which must expose ``PACK``) under ``name``."""
    _REGISTRY[name] = module
