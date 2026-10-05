"""Deterministic fake models for OFFLINE dry runs and tests.

Everything here is FAKE. Outputs derived from these classes are NOT results and must never be
presented as such; the orchestrator labels dry-run reports accordingly.

* ``GoldOracle``      question -> gold SQL registry (fakes need to know the right answer).
* ``FakeTransport``   duck-typed ``AsyncOpenAI``: serves planner/teacher/triage through the
                      real ``LLMClient`` code path (usage, schema validation, retries).
* ``FakeFineTune``    implements the ``FineTuner`` surface with real temp checkpoint files.
* ``FakeBaseFactory``     base-model servers (same StudentServer abstraction as the student).
* ``FakeStudentFactory``  student servers whose error rate falls with each fine-tune round.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from distillery.config import Config, GateThresholds, Price
from distillery.finetune import (
    CheckpointInfo,
    DownloadedFile,
    EventInfo,
    HyperParameters,
    JobInfo,
    TrainedArtifact,
    require_explicit_hyperparameters,
    sha256_file,
)
from distillery.humanset import read_confirmed
from distillery.orchestrator import DRY_PREFIX, Deps, PipelineConfig, Scale
from distillery.sandbox import FakeSandbox
from distillery.sandbox_executor import AsyncBridge
from distillery.student import FakeStudent, StudentServer
from distillery.taskpacks.sql.executor import LocalExecutor
from distillery.taskpacks.sql.questions import SqlTask

FAKE_MODELS = {
    "planner": "fake-planner",
    "teacher": "fake-teacher",
    "triage": "fake-triage",
    "student": "fake-student-base",
}
FAKE_PRICE = Price(
    input_per_mtok=1.0,
    output_per_mtok=2.0,
    source="FAKE dry-run price, not a real price",
    date="2026-09-29",
)
DEFAULT_ERROR_RATES: dict[str, float] = {
    "fake-planner": 0.02,
    "fake-teacher": 0.05,
    "fake-triage": 0.0,
    "fake-student-base": 0.65,
}
DEFAULT_STUDENT_ERROR_RATES: tuple[float, ...] = (0.45, 0.25, 0.10)
_WRONG_SQL = "SELECT 1"


DRY_RUN_CAP_USD = (
    10.0  # the fake-model pipeline's spend cap; the API shows it while a dry run is live
)


def _unit(*parts: object) -> float:
    """Deterministic pseudo-random number in [0, 1) from the parts."""
    h = hashlib.sha256("|".join(str(p) for p in parts).encode()).hexdigest()
    return int(h[:8], 16) / 2**32


class GoldOracle:
    def __init__(self) -> None:
        self._gold: dict[str, str] = {}

    def observe(self, tasks: Sequence[SqlTask]) -> None:
        for t in tasks:
            self._gold[t.question] = t.gold_sql

    def observe_gold(self, pairs: Mapping[str, str]) -> None:
        """Register question -> gold SQL pairs that did not come from task generation (the
        human set). FAKE models only: real models never see gold."""
        self._gold.update(pairs)

    def alias(self, new_question: str, original: str) -> None:
        """A paraphrase has the same gold as the question it rewrites (fakes need the answer)."""
        self._gold.setdefault(new_question, self.gold(original))

    def gold(self, question: str) -> str:
        try:
            return self._gold[question]
        except KeyError:
            raise KeyError(f"oracle has never seen question {question!r}") from None


def question_of(messages: Sequence[Mapping[str, Any]]) -> str | None:
    """Recover the question from a prompt built by ``prompts.build_messages``."""
    for m in messages:
        content = str(m.get("content", ""))
        if "Question: " in content:
            return content.split("Question: ", 1)[1].split("\n\n/no_think", 1)[0].strip()
    return None


class FakeTransport:
    """Duck-typed async chat transport. ``calls`` records every request for test assertions."""

    def __init__(
        self,
        oracle: GoldOracle,
        error_rates: Mapping[str, float] | None = None,
        *,
        schema_glitch_rate: float = 0.03,
        salt: str = "fake",
    ) -> None:
        self.oracle = oracle
        self.error_rates = dict(DEFAULT_ERROR_RATES if error_rates is None else error_rates)
        self.schema_glitch_rate = schema_glitch_rate
        self.salt = salt
        self.calls: list[dict[str, Any]] = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    async def _create(
        self,
        *,
        model: str,
        messages: Sequence[Mapping[str, Any]],
        response_format: Mapping[str, Any] | None = None,
        temperature: float | None = None,
        **_ignored: Any,
    ) -> Any:
        schema = (response_format or {}).get("json_schema", {}).get("name")
        self.calls.append(
            {"model": model, "messages": [dict(m) for m in messages], "schema": schema}
        )
        text = self._answer(model, messages, schema, temperature)
        in_tok = sum(len(str(m.get("content", ""))) for m in messages) // 4
        message = SimpleNamespace(content=text, refusal=None)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=message, finish_reason="stop")],
            usage=SimpleNamespace(prompt_tokens=in_tok, completion_tokens=len(text) // 4 + 1),
        )

    def _answer(
        self,
        model: str,
        messages: Sequence[Mapping[str, Any]],
        schema: str | None,
        temperature: float | None,
    ) -> str:
        if schema == "FormatVerdict":
            return json.dumps({"clean": True})
        if schema == "FailureClusters":
            return self._clusters(messages)
        question = question_of(messages)
        if question is None:
            raise AssertionError("FakeTransport: unrecognised prompt (no question found)")
        if schema == "Paraphrases":
            return self._paraphrases(question)
        retrying = any("did not validate" in str(m.get("content", "")) for m in messages)
        if (
            schema is not None
            and not retrying
            and _unit(self.salt, model, question, "glitch") < self.schema_glitch_rate
        ):
            return "sorry, here is your query"  # invalid JSON: exercises the schema-retry path
        wrong = _unit(self.salt, model, question, temperature) < self.error_rates.get(model, 0.0)
        sql = _WRONG_SQL if wrong else self.oracle.gold(question)
        return f"```sql\n{sql}\n```"

    def _paraphrases(self, question: str) -> str:
        """Deterministic fake paraphrases (a fixed set of wrappers). Some answers repeat the
        original or each other so the dedupe path runs; the oracle learns each new wording."""
        items = [
            f"Please answer this: {question}",
            f"{question} (rephrased)",
            f"Could you tell me the following? {question}",
        ]
        roll = _unit(self.salt, question, "para")
        if roll < 0.15:
            items.append(question.upper())  # same question once normalised
        elif roll < 0.30:
            items[1] = items[0]  # exact repeat
        for p in items:
            self.oracle.alias(p, question)
        return json.dumps({"items": items})

    @staticmethod
    def _clusters(messages: Sequence[Mapping[str, Any]]) -> str:
        text = "\n".join(str(m.get("content", "")) for m in messages)
        fams = Counter(re.findall(r"\[family=(\w+)\]", text))
        clusters = [
            {
                "name": f"{fam} mistakes",
                "description": f"student fails {n} dev items in family {fam} (fake analysis)",
                "target_families": [fam],
            }
            for fam, n in fams.most_common(3)
        ]
        return json.dumps({"clusters": clusters})


@dataclass
class FakeFineTune:
    """``FineTuner`` fake. ``fail_rounds``: rounds (1-based, from the job suffix) whose job ends
    ``failed``. Records uploads, created jobs and cancellations."""

    fail_rounds: frozenset[int] = frozenset()
    uploads: dict[str, Path] = field(default_factory=dict)
    created: list[str] = field(default_factory=list)
    cancelled: list[str] = field(default_factory=list)
    suffixes: dict[str, str] = field(default_factory=dict)
    job_status: dict[str, str] = field(default_factory=dict)  # test override for ``get``
    _base_models: dict[str, str] = field(default_factory=dict)
    _hps: dict[str, dict[str, Any]] = field(default_factory=dict)

    def upload(self, path: Path | str) -> str:
        p = Path(path)
        fid = f"file-{sha256_file(p)[:16]}"
        self.uploads[fid] = p
        return fid

    def create_job(
        self,
        model: str,
        training_file: str,
        validation_file: str | None = None,
        hyperparameters: HyperParameters | None = None,
        suffix: str | None = None,
        seed: int | None = None,
    ) -> str:
        jid = f"ftjob-{hashlib.sha256((training_file + str(suffix)).encode()).hexdigest()[:12]}"
        self.created.append(jid)
        self.suffixes[jid] = suffix or ""
        self._base_models[jid] = model
        self._hps[jid] = require_explicit_hyperparameters(hyperparameters).to_request()
        return jid

    def list_jobs(self, *, limit: int = 100, max_pages: int = 5) -> list[JobInfo]:
        return [replace(self.get(j), suffix=self.suffixes.get(j)) for j in reversed(self.created)]

    def _round(self, job_id: str) -> int:
        m = re.search(r"-r(\d+)$", self.suffixes.get(job_id, ""))
        return int(m.group(1)) if m else 0

    def get(self, job_id: str) -> JobInfo:
        failed = self._round(job_id) in self.fail_rounds
        status = self.job_status.get(job_id, "failed" if failed else "succeeded")
        return JobInfo(
            id=job_id,
            status=status,
            model=self._base_models.get(job_id),
            error_code=None,
            error_message=None,
            result_files=(),
            trained_steps=30,
            total_steps=30,
            trained_tokens=12345,
            hyperparameters=self._hps.get(job_id),
        )

    def poll(
        self,
        job_id: str,
        *,
        interval_s: float = 15.0,
        timeout_s: float | None = None,
        on_update: Callable[[JobInfo], None] | None = None,
        on_error: Callable[[BaseException], None] | None = None,
        transient_error_grace_s: float = 900.0,
    ) -> JobInfo:
        failed = self._round(job_id) in self.fail_rounds
        for status in ("running", "failed" if failed else "succeeded"):
            info = JobInfo(
                id=job_id,
                status=status,
                model=self._base_models.get(job_id),
                error_code="fake_failure" if status == "failed" else None,
                error_message="forced by FakeFineTune" if status == "failed" else None,
                result_files=(),
                trained_steps=30,
                total_steps=30,
                trained_tokens=12345,
                hyperparameters=self._hps.get(job_id),
            )
            if on_update is not None:
                on_update(info)
        return info

    def checkpoints(self, job_id: str) -> list[CheckpointInfo]:
        return [
            CheckpointInfo(
                id=f"ckpt-{job_id}",
                step_number=10,
                fine_tuned_model_checkpoint=f"fake-ft:{self.suffixes.get(job_id, '')}:{job_id}",
                result_files=(f"{job_id}-cfg", f"{job_id}-weights"),
            )
        ]

    def loss_curve(self, job_id: str) -> list[dict[str, Any]]:
        return [{"step": 10, "train_loss": 1.0, "valid_loss": 1.1}]

    def events(self, job_id: str, *, limit: int = 50, max_pages: int = 20) -> list[EventInfo]:
        return [EventInfo("e1", 1, "info", "fake training event")]

    def trained_artifact(
        self, job: JobInfo, checkpoint: CheckpointInfo, directory: Path | str
    ) -> TrainedArtifact:
        dest = Path(directory) / checkpoint.id
        dest.mkdir(parents=True, exist_ok=True)
        files = []
        for fid, name, body in (
            (checkpoint.result_files[0], "adapter_config.json", f'{{"fake": "{job.id}"}}'),
            (checkpoint.result_files[1], "adapter_model.safetensors", f"FAKE-WEIGHTS-{job.id}"),
        ):
            target = dest / name
            target.write_text(body, encoding="utf-8")
            files.append(DownloadedFile(fid, target, sha256_file(target)))
        return TrainedArtifact(
            job_id=job.id,
            checkpoint_id=checkpoint.id,
            base_model=job.model,
            fine_tuned_model_checkpoint=checkpoint.fine_tuned_model_checkpoint,
            files=tuple(files),
        )

    def cancel(self, job_id: str) -> object:
        self.cancelled.append(job_id)
        return None


class FakeStudentFactory:
    """Serves a fake student per trained artifact; its error rate depends on the round, parsed
    from the (fake) checkpoint name so it is stable across process restarts."""

    def __init__(
        self,
        oracle: GoldOracle,
        error_rates: Sequence[float] = DEFAULT_STUDENT_ERROR_RATES,
        *,
        garbage: bool = False,
        salt: str = "student",
    ) -> None:
        self.oracle = oracle
        self.error_rates = tuple(error_rates)
        self.garbage = garbage
        self.salt = salt
        self.served: list[str] = []
        self.servers: list[FakeStudent] = []

    def __call__(self, trained: TrainedArtifact) -> StudentServer:
        m = re.search(r"-r(\d+):", trained.fine_tuned_model_checkpoint or "")
        rnd = int(m.group(1)) if m else 1
        rate = self.error_rates[min(rnd, len(self.error_rates)) - 1]
        self.served.append(trained.adapter_sha256)

        def respond(messages: Sequence[dict[str, Any]]) -> str:
            if self.garbage:
                return "I am unable to write SQL."
            q = question_of(messages)
            if q is None:
                raise AssertionError("FakeStudent: unrecognised prompt")
            bad = _unit(self.salt, trained.adapter_sha256, q) < rate
            return f"```sql\n{_WRONG_SQL if bad else self.oracle.gold(q)}\n```"

        server = FakeStudent(respond)
        self.servers.append(server)
        return server


class FakeBaseFactory:
    """Serves a fake un-tuned base model (constant error rate, FAKE) as a ``StudentServer``."""

    def __init__(self, oracle: GoldOracle, error_rate: float, *, salt: str = "base") -> None:
        self.oracle = oracle
        self.error_rate = error_rate
        self.salt = salt
        self.servers: list[FakeStudent] = []

    def __call__(self) -> StudentServer:
        def respond(messages: Sequence[dict[str, Any]]) -> str:
            q = question_of(messages)
            if q is None:
                raise AssertionError("FakeBase: unrecognised prompt")
            bad = _unit(self.salt, q) < self.error_rate
            return f"```sql\n{_WRONG_SQL if bad else self.oracle.gold(q)}\n```"

        server = FakeStudent(respond)
        self.servers.append(server)
        return server


@dataclass
class DryRun:
    """Everything needed to run the pipeline offline. All models are FAKE."""

    config: Config
    pipeline_cfg: PipelineConfig
    deps: Deps
    oracle: GoldOracle
    transport: FakeTransport
    finetune: FakeFineTune
    students: FakeStudentFactory
    sandbox: FakeSandbox
    base: FakeBaseFactory


def dry_run_id(scale: str) -> str:
    return f"{DRY_PREFIX}sql-{scale}"


def build_dry_run(
    scale: Scale,
    bridge: AsyncBridge,
    *,
    seed: int = 1234,
    error_rates: Mapping[str, float] | None = None,
    student_error_rates: Sequence[float] = DEFAULT_STUDENT_ERROR_RATES,
    fail_rounds: frozenset[int] = frozenset(),
    run_cap_usd: float = DRY_RUN_CAP_USD,
    on_stage: Callable[[str], None] | None = None,
    garbage_student: bool = False,
    **pipeline_overrides: Any,
) -> DryRun:
    oracle = GoldOracle()
    human_path = pipeline_overrides.get("human_set_path")
    if human_path is not None:  # fakes must know the human gold, like every other task set
        confirmed = read_confirmed(human_path)
        oracle.observe_gold({str(it["question"]): str(it["gold_sql"]) for it in confirmed.items})
    transport = FakeTransport(oracle, error_rates)
    ft = FakeFineTune(fail_rounds=fail_rounds)
    base = FakeBaseFactory(
        oracle,
        (error_rates or {}).get("fake-student-base", DEFAULT_ERROR_RATES["fake-student-base"]),
    )
    students = FakeStudentFactory(oracle, student_error_rates, garbage=garbage_student)
    sandbox = FakeSandbox()
    config = Config(
        run_cap_usd=run_cap_usd,
        model_ids=dict(FAKE_MODELS),
        prices={m: FAKE_PRICE for m in FAKE_MODELS.values()},
        gate=GateThresholds(seed=seed),
    )
    pipeline_overrides.setdefault("finetune_estimate_usd", 0.0)
    pcfg = PipelineConfig(scale=scale, dry_run=True, seed=seed, **pipeline_overrides)
    deps = Deps(
        transport=transport,
        finetune=ft,
        executor=LocalExecutor(),
        bridge=bridge,
        student_factory=students,
        base_factory=base,
        sandbox=sandbox,
        sandbox_image="fake-base-image",
        on_stage=on_stage,
        task_observer=oracle.observe,
    )
    return DryRun(config, pcfg, deps, oracle, transport, ft, students, sandbox, base)


__all__ = [
    "DEFAULT_ERROR_RATES",
    "FAKE_MODELS",
    "DryRun",
    "FakeBaseFactory",
    "FakeFineTune",
    "FakeStudentFactory",
    "FakeTransport",
    "GoldOracle",
    "build_dry_run",
    "dry_run_id",
    "question_of",
]
