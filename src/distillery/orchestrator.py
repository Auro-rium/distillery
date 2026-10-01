"""Distillery pipeline orchestrator (SQL pack).

Stages (each through ``Store.get_or_run``; the stage input hash covers the config slice it
depends on plus the output hashes of its upstream stages, so a rerun skips finished work and a
changed input recomputes only what depends on it):

  schema -> questions -> gold_crosscheck -> verifier_selftest -> split (seals held-out)
  -> headroom -> teacher_data -> [finetune_rN -> dev_eval_rN -> analysis_rN -> targeted_rN
  -> sandbox_branch_rN]* -> final_eval (held-out, once) -> report.json

Deviations from the written order, on purpose (see DECISIONS.md 2026-09-29): the split happens
before teacher data so the teacher only ever sees train tasks, and the headroom check runs
after the split because it needs the dev set.

Held-out discipline: this module never reads the sealed set back. Items are sealed once from the
in-memory split; the only reader is ``evaluator.evaluate`` (final_eval), called once, for the
candidate with the best DEV accuracy. All analysis and round selection use dev only.

Nothing is silently dropped or retried: every drop/retry/skip is a named integer counter that
ends up in report.json.
"""

from __future__ import annotations

import logging
import math
import random
import sqlite3
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Literal, Protocol

import openai
from pydantic import BaseModel, ConfigDict, Field

from distillery import evaluator as evaluator_mod
from distillery.budget import (
    BASIS_CEILING,
    Ledger,
    sandbox_seconds,
    spend_lines,
)
from distillery.config import Config, ConfigError
from distillery.endpoint_student import EndpointBackend, EndpointSpec
from distillery.evaluator import ExpectedArtifact, ModelScores, score_model
from distillery.finetune import (
    ACTIVE_STATUSES,
    PINNED_HYPERPARAMETERS,
    CheckpointInfo,
    DownloadedFile,
    EventInfo,
    FineTuneClient,
    HyperParameters,
    JobInfo,
    TrainedArtifact,
    paid_job,
    planned_steps,
    require_explicit_hyperparameters,
)
from distillery.llm import CallRecord, ChatResult, LLMClient, LLMError
from distillery.prompts import build_messages, extract_sql, to_training_row
from distillery.sandbox import Sandbox
from distillery.sandbox_executor import AsyncBridge
from distillery.store import Store, atomic_write_bytes, canonical_json, sha256_hex
from distillery.student import StudentServer
from distillery.taskpacks.sql import schema as sql_schema
from distillery.taskpacks.sql.executor import Executor
from distillery.taskpacks.sql.questions import (
    FAMILIES,
    SqlTask,
    classify_heldout,
    generate_tasks,
    split_by_family,
)
from distillery.taskpacks.sql.runner import ExecOutcome, run_select
from distillery.taskpacks.sql.verifier import compare_outcomes, corrupt_sql

log = logging.getLogger(__name__)


def _close_all(servers: Sequence[StudentServer], *, raise_first: bool = True) -> None:
    """Close every server (paid endpoints!) even if an earlier close raises.

    The first error is re-raised after all closes ran (later ones are logged). With
    ``raise_first=False`` (the body already failed, so its exception must win) every close error
    is only logged.
    """
    first: BaseException | None = None
    for srv in servers:
        try:
            srv.close()
        except Exception as e:  # noqa: BLE001 - keep closing the remaining servers
            if first is None and raise_first:
                first = e
            else:
                log.error("server close failure: %s", e)
    if first is not None:
        raise first


def _check_endpoint_identity(
    spec: EndpointSpec, trained: TrainedArtifact, student_model: str
) -> None:
    """File hashes prove the adapter on disk, not what the endpoint serves; pin what we can."""
    if trained.base_model is not None and trained.base_model != student_model:
        raise ConfigRefusal(
            f"trained base_model {trained.base_model!r} != configured student {student_model!r}"
        )
    ckpt = trained.fine_tuned_model_checkpoint
    if spec.model_name and ckpt and spec.model_name != ckpt:
        raise ConfigRefusal(
            f"EndpointSpec.model_name {spec.model_name!r} != trained checkpoint {ckpt!r}: "
            "the served model would not be the trained artifact"
        )


def _generation_error_counter(name: str, server: Any) -> dict[str, int]:
    """Samples the serving backend failed to generate (scored as wrong, never dropped). Only
    backends that can fail per sample (sandbox CPU) expose ``generation_errors``."""
    errors = getattr(server, "generation_errors", None)
    return {} if errors is None else {f"{name}_generation_errors": int(errors)}


def _with_serving(
    artifact: Mapping[str, str], trained: TrainedArtifact, student: Any, base: Any
) -> dict[str, str]:
    """Record what was actually served next to the file-hash identity (which only covers disk)."""
    out = dict(artifact)
    served = getattr(student, "served_model", None)
    base_served = getattr(base, "served_model", None)
    if served is not None:
        out["served_model"] = str(served)
        if trained.fine_tuned_model_checkpoint and served != trained.fine_tuned_model_checkpoint:
            out["served_model_note"] = "served name is an explicit override, not the trained ckpt"
    if base_served is not None:
        out["served_base_model"] = str(base_served)
    if trained.base_model:
        out["trained_base_model"] = trained.base_model
    if trained.fine_tuned_model_checkpoint:
        out["trained_checkpoint_name"] = trained.fine_tuned_model_checkpoint
    return out


PIPELINE_VERSION = "1"
DRY_RUN_LABEL = "DRY RUN — fake models, numbers are NOT results"
STUDENT_COST_UNAVAILABLE = (
    "unavailable: the student is served in Nebius Sandboxes (CPU) and the sandbox price is unknown"
)
DRY_PREFIX = "dry-"
_SCHEMA_OVERHEAD_TOKENS = 120  # rough prompt overhead of the JSON-schema instruction (estimate)


def _planned_trained_tokens(train_jsonl: bytes, chars_per_token: int, n_epochs: int | None) -> int:
    """Planned trained tokens for the preflight: file characters / chars-per-token, x epochs.

    An ESTIMATE used only to size the pre-spend budget check (the recorded cost uses the job's
    measured trained_tokens). Epochs default to 3 when the request leaves them to the service.
    """
    text = train_jsonl.decode("utf-8")
    return -(-len(text) // max(chars_per_token, 1)) * (n_epochs or 3)


# ---------------------------------------------------------------- errors


class PipelineError(RuntimeError):
    pass


class TaskTooEasyError(PipelineError):
    """Base model already solves the dev set: no headroom for distillation to show."""


class VerifierSelfTestError(PipelineError):
    pass


class ShortfallError(PipelineError):
    """Could not produce the requested train/dev/held-out sizes."""


class ServingUnavailableError(PipelineError):
    """No way to serve the fine-tuned student (serving path undecided, spike S4)."""


class ConfigRefusal(PipelineError):
    """Refusing to start (before any spend) because configuration is unsafe or incomplete."""


# ---------------------------------------------------------------- config


class Scale(BaseModel):
    model_config = ConfigDict(frozen=True)

    name: str
    train: int
    dev: int
    heldout: int


SCALES: dict[str, Scale] = {
    "tiny": Scale(name="tiny", train=40, dev=10, heldout=20),
    "mini": Scale(name="mini", train=120, dev=30, heldout=60),  # CLI-only: cheaper 2-round live run
    "small": Scale(name="small", train=300, dev=50, heldout=100),
    "full": Scale(name="full", train=2000, dev=150, heldout=300),
}


class PipelineConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    pack: str = "sql"
    scale: Scale = SCALES["tiny"]
    dry_run: bool = False
    db_seed: int = 0
    seed: int = 1234
    oversample: float = 1.6  # generate this multiple of the needed tasks (drops + trimming)
    headroom_max_base_acc: float = 0.80
    dev_target_acc: float = 0.90  # stop adding rounds once the student reaches this on dev
    max_rounds: int = 3  # total fine-tune rounds, including the first
    round_extra_tasks: int | None = None  # default: max(10, train // 4)
    concurrency: int = 16
    chunk: int = 50  # LLM calls per preflighted chunk
    est_output_tokens: int = 256  # budget estimate only
    chars_per_token: int = 3  # budget estimate only
    candidates_per_task: int = 1
    use_triage: bool = True
    triage_sample: int = 50
    selftest_sample: int = 50
    selftest_corruptions: int = 4
    spot_check_n: int = 30
    poll_interval_s: float = 15.0
    poll_timeout_s: float = 6 * 3600.0
    finetune_estimate_usd: float | None = None  # docs give no fine-tune price: caller must say
    # Pinned, never provider defaults (measured: defaults gave 3-9 optimizer steps for 40-138 rows).
    hyperparameters: HyperParameters = PINNED_HYPERPARAMETERS
    # Refuse before any spend when the planned optimizer steps fall below this (dry runs exempt).
    min_planned_steps: int = Field(default=50, ge=0)
    # packing=True makes the step count unknowable; refuse unless the caller says so explicitly.
    allow_packing: bool = False
    ultra_verifier_authoring: bool = False  # documented hook, intentionally unimplemented
    # How the student and base are served. "injected" (default) = use Deps.student_factory /
    # Deps.base_factory as given. The others build both from Deps (UNVERIFIED against real APIs).
    student_serving: Literal["injected", "sandbox_cpu", "endpoint"] = "injected"
    student_sandbox_image: str | None = None  # sandbox_cpu: image with pip (default: sandbox_image)
    student_batch_size: int = Field(default=2, ge=1)  # sandbox_cpu: prompts per generation job
    # sandbox_cpu: parallel generation jobs (the beta limit is 50 operations in flight in total)
    student_concurrency: int = Field(default=10, ge=1, le=20)
    student_max_new_tokens: int = Field(default=160, ge=16)  # sandbox_cpu: per sample (S4 recipe)
    student_endpoint: EndpointSpec | None = None  # endpoint: explicit spec incl. hourly price

    def extra_tasks(self) -> int:
        return self.round_extra_tasks or max(10, self.scale.train // 4)


# ---------------------------------------------------------------- structured outputs


class SqlAnswer(BaseModel):
    sql: str


class Cluster(BaseModel):
    name: str
    description: str
    target_families: list[str]


class FailureClusters(BaseModel):
    clusters: list[Cluster]


class FormatVerdict(BaseModel):
    clean: bool


# ---------------------------------------------------------------- dependencies


class FineTuner(Protocol):
    """The subset of ``FineTuneClient`` the pipeline uses (fakes implement the same)."""

    def upload(self, path: Path | str) -> str: ...

    def create_job(
        self,
        model: str,
        training_file: str,
        validation_file: str | None = None,
        hyperparameters: HyperParameters | None = None,
        suffix: str | None = None,
        seed: int | None = None,
    ) -> str: ...

    def get(self, job_id: str) -> JobInfo: ...

    def poll(
        self,
        job_id: str,
        *,
        interval_s: float = ...,
        timeout_s: float | None = None,
        on_update: Callable[[JobInfo], None] | None = None,
        on_error: Callable[[BaseException], None] | None = None,
        transient_error_grace_s: float = ...,
    ) -> JobInfo: ...

    def checkpoints(self, job_id: str) -> list[CheckpointInfo]: ...

    def loss_curve(self, job_id: str) -> list[dict[str, Any]]: ...

    def events(self, job_id: str, *, limit: int = ..., max_pages: int = ...) -> list[EventInfo]: ...

    def trained_artifact(
        self, job: JobInfo, checkpoint: CheckpointInfo, directory: Path | str
    ) -> TrainedArtifact: ...

    def cancel(self, job_id: str) -> object: ...


StudentFactory = Callable[[TrainedArtifact], StudentServer]
# Serves the un-tuned base model through the same StudentServer abstraction as the student
# (Qwen3-1.7B is not on the serverless inference API, so it cannot go through LLMClient).
BaseFactory = Callable[[], StudentServer]


@dataclass
class Deps:
    transport: Any  # duck-typed AsyncOpenAI (chat.completions.create)
    finetune: FineTuner
    executor: Executor
    bridge: AsyncBridge
    student_factory: StudentFactory | None
    sandbox: Sandbox | None = None
    sandbox_image: str = "base-image"
    llm_sleep: Callable[[float], Any] | None = None
    on_stage: Callable[[str], None] | None = None  # called inside a stage that is really running
    # Oracle registration for FAKE models only (they must know the gold SQL). Called with every
    # task set produced (also when the stage result comes from cache). Real deps leave it None.
    task_observer: Callable[[Sequence[SqlTask]], None] | None = None
    base_factory: BaseFactory | None = None
    endpoint: EndpointBackend | None = None  # needed only for student_serving="endpoint"


# ---------------------------------------------------------------- usage / metrics


class LedgerSink:
    """LLMClient usage sink: records spend in the Ledger and aggregates per purpose.

    Costs are aggregated as integer nano-USD so sums are independent of thread ordering.
    """

    def __init__(self, ledger: Ledger) -> None:
        import threading

        self._ledger = ledger
        self._lock = threading.Lock()
        self._agg: dict[str, dict[str, int]] = {}

    def record_call(self, record: CallRecord) -> None:
        with self._lock:
            a = self._agg.setdefault(
                record.purpose,
                {
                    "calls": 0,
                    "failed_attempts": 0,
                    "schema_retries": 0,
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "usd_nano": 0,
                },
            )
            if not record.ok:
                a["failed_attempts"] += 1
                return
            a["calls"] += 1
            if record.attempt > 1:
                a["schema_retries"] += 1
            a["input_tokens"] += record.input_tokens
            a["output_tokens"] += record.output_tokens
            a["usd_nano"] += round(record.cost_usd * 1e9)
        self._ledger.record(
            "llm", record.model, record.cost_usd, record.input_tokens, record.output_tokens
        )

    def snapshot(self) -> dict[str, dict[str, int]]:
        with self._lock:
            return {k: dict(v) for k, v in self._agg.items()}


def _delta(
    before: Mapping[str, Mapping[str, int]], after: Mapping[str, Mapping[str, int]]
) -> dict[str, dict[str, int]]:
    out: dict[str, dict[str, int]] = {}
    for key, vals in after.items():
        base = before.get(key, {})
        d = {k: v - base.get(k, 0) for k, v in vals.items()}
        if any(d.values()):
            out[key] = d
    return out


class LLMRunner:
    """Sync facade over ``LLMClient``: bounded concurrency, budget preflight per chunk, and
    per-item error tallies (an item that failed comes back as ``None`` and is counted)."""

    def __init__(
        self,
        llm: LLMClient,
        ledger: Ledger,
        bridge: AsyncBridge,
        cfg: PipelineConfig,
    ) -> None:
        self.llm = llm
        self.ledger = ledger
        self.bridge = bridge
        self.cfg = cfg
        self.errors: Counter[str] = Counter()

    def _estimate(
        self,
        role: str,
        batch: Sequence[Sequence[Mapping[str, Any]]],
        schema: type[BaseModel] | None,
    ) -> float:
        chars = sum(len(str(m.get("content", ""))) for msgs in batch for m in msgs)
        in_tok = chars // self.cfg.chars_per_token + (
            _SCHEMA_OVERHEAD_TOKENS * len(batch) if schema else 0
        )
        out_tok = self.cfg.est_output_tokens * len(batch)
        return self.ledger.estimate_llm_cost(self.llm.model_for(role), in_tok, out_tok)

    async def _one(
        self,
        sem: Any,
        role: Any,
        msgs: Sequence[Mapping[str, Any]],
        purpose: str,
        stage: str,
        schema: type[BaseModel] | None,
        temperature: float | None,
    ) -> ChatResult[Any] | None:
        async with sem:
            try:
                if schema is None:
                    return await self.llm.chat(
                        role, [dict(m) for m in msgs], purpose=purpose, stage=stage,
                        temperature=temperature,
                    )  # fmt: skip
                return await self.llm.chat(
                    role, [dict(m) for m in msgs], purpose=purpose, stage=stage,
                    json_schema=schema, temperature=temperature,
                )  # fmt: skip
            except LLMError as exc:
                self.errors[f"{purpose}:{type(exc).__name__}"] += 1
                return None

    async def _gather(
        self,
        role: str,
        chunk: Sequence[Sequence[Mapping[str, Any]]],
        purpose: str,
        stage: str,
        schema: type[BaseModel] | None,
        temperature: float | None,
    ) -> list[ChatResult[Any] | None]:
        import asyncio

        sem = asyncio.Semaphore(self.cfg.concurrency)
        return list(
            await asyncio.gather(
                *(self._one(sem, role, m, purpose, stage, schema, temperature) for m in chunk)
            )
        )

    def map(
        self,
        role: str,
        batch: Sequence[Sequence[Mapping[str, Any]]],
        *,
        purpose: str,
        stage: str,
        schema: type[BaseModel] | None = None,
        temperature: float | None = 0.0,
    ) -> list[ChatResult[Any] | None]:
        out: list[ChatResult[Any] | None] = []
        for i in range(0, len(batch), self.cfg.chunk):
            chunk = batch[i : i + self.cfg.chunk]
            self.ledger.preflight(self._estimate(role, chunk, schema))  # BudgetExceeded: no call
            out.extend(
                self.bridge.run(self._gather(role, chunk, purpose, stage, schema, temperature))
            )
        return out


class LLMGenerator:
    """Adapter from ``LLMRunner`` to the evaluator's synchronous ``Generator`` protocol."""

    def __init__(self, runner: LLMRunner, role: str, purpose: str, stage: str) -> None:
        self._runner = runner
        self._role = role
        self._purpose = purpose
        self._stage = stage

    def generate(self, messages_batch: Sequence[Sequence[Mapping[str, Any]]]) -> list[str]:
        res = self._runner.map(self._role, messages_batch, purpose=self._purpose, stage=self._stage)
        # a failed call is already counted in runner.errors; "" is scored as unparseable
        return [r.text if r is not None else "" for r in res]


class RecordingGenerator:
    """Wraps a generator and remembers the last batch of raw outputs (for dev failure lists)."""

    def __init__(self, inner: StudentServer) -> None:
        self._inner = inner
        self.outputs: list[str] = []

    def generate(self, messages_batch: Sequence[Sequence[Mapping[str, Any]]]) -> list[str]:
        self.outputs = self._inner.generate([[dict(m) for m in b] for b in messages_batch])
        return self.outputs


# ---------------------------------------------------------------- pure helpers


def artifact_to_json(a: TrainedArtifact, base: Path) -> dict[str, Any]:
    """Paths are stored relative to the run dir so stage artifacts hash identically anywhere."""
    return {
        "job_id": a.job_id,
        "checkpoint_id": a.checkpoint_id,
        "base_model": a.base_model,
        "fine_tuned_model_checkpoint": a.fine_tuned_model_checkpoint,
        "files": [
            {"file_id": f.file_id, "path": str(f.path.relative_to(base)), "sha256": f.sha256}
            for f in a.files
        ],
        "adapter_sha256": a.adapter_sha256,
    }


def artifact_from_json(d: Mapping[str, Any], base: Path) -> TrainedArtifact:
    return TrainedArtifact(
        job_id=str(d["job_id"]),
        checkpoint_id=str(d["checkpoint_id"]),
        base_model=d["base_model"],
        fine_tuned_model_checkpoint=d["fine_tuned_model_checkpoint"],
        files=tuple(
            DownloadedFile(str(f["file_id"]), base / str(f["path"]), str(f["sha256"]))
            for f in d["files"]
        ),
    )


def _digest(obj: Any) -> str:
    return sha256_hex(canonical_json(obj).encode("utf-8"))


def _task_from(d: Mapping[str, Any]) -> SqlTask:
    return SqlTask.model_validate(dict(d))


def _item(t: SqlTask, heldout_class: str | None = None) -> dict[str, Any]:
    d: dict[str, Any] = t.model_dump(mode="json")
    if heldout_class is not None:
        d["heldout_class"] = heldout_class
    return d


@dataclass
class SplitPlan:
    train: list[SqlTask]
    dev: list[SqlTask]
    heldout: list[SqlTask]
    heldout_families: list[str]
    counters: dict[str, int] = field(default_factory=dict)


def plan_split(tasks: Sequence[SqlTask], scale: Scale, seed: int) -> SplitPlan:
    """Family-aware train/dev/held-out split with exact target sizes (or ShortfallError).

    Whole families are held out (>= 50% of the target held-out size comes from them, at most
    60% is kept); the rest of held-out is an in-distribution sample. In-distribution surplus goes
    back to train; surplus unseen-family tasks are discarded (counted) so the family stays unseen.
    """
    rng = random.Random(seed)  # noqa: S311 - deterministic sampling, not security
    counts = Counter(t.family for t in tasks)
    fams = sorted(counts)
    rng.shuffle(fams)
    target_u = math.ceil(0.5 * scale.heldout)
    chosen: list[str] = []
    u = 0
    for f in fams:
        if u >= target_u:
            break
        chosen.append(f)
        u += counts[f]
    chosen_set = set(chosen)
    unseen_take = min(u, math.ceil(0.6 * scale.heldout))
    n_rest = len(tasks) - u
    seen_needed = max(0, scale.heldout - unseen_take)
    frac = min(1.0, seen_needed / n_rest) if n_rest else 0.0
    sp = split_by_family(list(tasks), chosen_set, seed, in_dist_fraction=frac)
    unseen = [t for t in sp.heldout if t.family in chosen_set]
    seen_h = [t for t in sp.heldout if t.family not in chosen_set]
    rng.shuffle(unseen)
    rng.shuffle(seen_h)
    keep_unseen = unseen[:unseen_take]
    keep_seen = seen_h[: scale.heldout - len(keep_unseen)]
    back_to_train = seen_h[len(keep_seen) :]
    short = scale.heldout - len(keep_unseen) - len(keep_seen)
    if short > 0:
        extra = unseen[len(keep_unseen) : len(keep_unseen) + short]
        keep_unseen += extra
    heldout = keep_unseen + keep_seen
    discarded_unseen = len(unseen) - len(keep_unseen)
    pool = list(sp.train) + back_to_train
    rng.shuffle(pool)
    dev = pool[: scale.dev]
    train = pool[scale.dev : scale.dev + scale.train]
    counters = {
        "input_tasks": len(tasks),
        "heldout_unseen_family_tasks_discarded": discarded_unseen,
        "train_pool_trimmed": max(0, len(pool) - scale.dev - len(train)),
        "heldout_seen_returned_to_train": len(back_to_train),
    }
    if len(heldout) < scale.heldout or len(dev) < scale.dev or len(train) < scale.train:
        raise ShortfallError(
            f"cannot fill scale {scale.name!r}: got train={len(train)}/{scale.train} "
            f"dev={len(dev)}/{scale.dev} heldout={len(heldout)}/{scale.heldout} "
            f"from {len(tasks)} cross-checked tasks; raise oversample or lower the scale"
        )
    return SplitPlan(train, dev, heldout, sorted(chosen), counters)


def build_analysis_messages(
    failures: Sequence[Mapping[str, Any]], allowed_families: Sequence[str]
) -> list[dict[str, str]]:
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
        "failures into a few clusters of related mistakes. For each cluster give a short name, a "
        "description, and the question families (from the allowed list only) whose extra "
        "training examples would fix it."
    )
    user = f"Allowed families: {', '.join(allowed_families)}\n\nFailures:\n" + "\n".join(lines)
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def cost_by_model(store: Store, run_id: str) -> dict[str, dict[str, float | int]]:
    """Per-model calls/tokens/usd from the ``llm_calls`` table (read-only side connection)."""
    con = sqlite3.connect(f"file:{store.root / 'index.sqlite'}?mode=ro", uri=True)
    try:
        rows = con.execute(
            "SELECT model, COUNT(*), SUM(input_tokens), SUM(output_tokens), SUM(usd) "
            "FROM llm_calls WHERE run_id=? GROUP BY model ORDER BY model",
            (run_id,),
        ).fetchall()
    finally:
        con.close()
    return {
        str(m): {"calls": int(c), "input_tokens": int(i), "output_tokens": int(o), "usd": float(u)}
        for m, c, i, o, u in rows
    }


def stage_rows(store: Store, run_id: str) -> list[dict[str, Any]]:
    """(stage, status, error) for every stage attempt of a run, oldest first."""
    con = sqlite3.connect(f"file:{store.root / 'index.sqlite'}?mode=ro", uri=True)
    try:
        rows = con.execute(
            "SELECT stage, status, output_sha256, error, updated_at FROM stages "
            "WHERE run_id=? ORDER BY updated_at",
            (run_id,),
        ).fetchall()
    finally:
        con.close()
    return [
        {"stage": r[0], "status": r[1], "output_sha256": r[2], "error": r[3], "updated_at": r[4]}
        for r in rows
    ]


# ---------------------------------------------------------------- the pipeline


class Pipeline:
    def __init__(
        self,
        cfg: PipelineConfig,
        config: Config,
        deps: Deps,
        store: Store,
        run_id: str,
        *,
        say: Callable[[str], None] = print,
    ) -> None:
        self.cfg = cfg
        self.config = config
        self.deps = deps
        self.store = store
        self.run_id = run_id
        self.say = say
        self.ledger = Ledger.from_config(run_id, config, store)
        self.sink = LedgerSink(self.ledger)
        llm_kwargs: dict[str, Any] = {}
        if deps.llm_sleep is not None:
            llm_kwargs["sleep"] = deps.llm_sleep
        self.llm = LLMClient(
            deps.transport,
            config.model_ids,
            sink=self.sink,
            pricing=self.ledger.estimate_llm_cost,
            **llm_kwargs,
        )
        self.runner = LLMRunner(self.llm, self.ledger, deps.bridge, cfg)
        self.hashes: dict[str, str] = {}
        self.ran: list[str] = []  # stages that actually executed in this process
        self.run_dir = store.run_dir(run_id)
        self.db_path = self.run_dir / "db.sqlite"
        self.db_ref = str(self.db_path)
        self.ddl = ""
        self.tasks_by_id: dict[str, SqlTask] = {}
        self.sealed_sha: str | None = None
        self.counters: dict[str, dict[str, int]] = {}
        self.usage: dict[str, dict[str, dict[str, int]]] = {}
        self._results: dict[str, dict[str, Any]] = {}

    # ---- plumbing --------------------------------------------------------
    def _snap(self) -> tuple[dict[str, dict[str, int]], dict[str, int]]:
        return self.sink.snapshot(), dict(self.runner.errors)

    def _since(self, snap: tuple[dict[str, dict[str, int]], dict[str, int]]) -> dict[str, Any]:
        usage = _delta(snap[0], self.sink.snapshot())
        errs = {k: v - snap[1].get(k, 0) for k, v in self.runner.errors.items()}
        return {"usage": usage, "llm_errors": {k: v for k, v in errs.items() if v}}

    def _stage(
        self,
        name: str,
        cfg_inputs: dict[str, Any],
        upstream: Sequence[str],
        fn: Callable[[], dict[str, Any]],
    ) -> dict[str, Any]:
        inputs = {
            "v": PIPELINE_VERSION,
            "models": self.config.model_ids,
            "dry_run": self.cfg.dry_run,
            "cfg": cfg_inputs,
            "up": {u: self.hashes[u] for u in upstream},
        }

        def wrapped() -> dict[str, Any]:
            self.ran.append(name)
            self.say(f"[stage] {name}")
            if self.deps.on_stage is not None:
                self.deps.on_stage(name)
            snap = self._snap()
            res = fn()
            extra = self._since(snap)
            res["metrics"] = extra
            return res

        result: dict[str, Any] = self.store.get_or_run(self.run_id, name, inputs, wrapped)
        self.hashes[name] = _digest(result)
        self._results[name] = result
        self.counters[name] = {k: int(v) for k, v in result.get("counters", {}).items()}
        self.usage[name] = result.get("metrics", {}).get("usage", {})
        self._stamp_manifest()
        return result

    def _stamp_manifest(self) -> None:
        import json

        path = self.run_dir / "manifest.json"
        if not path.exists():
            return
        data = json.loads(path.read_text(encoding="utf-8"))
        data["dry_run"] = self.cfg.dry_run
        data["label"] = DRY_RUN_LABEL if self.cfg.dry_run else None
        atomic_write_bytes(path, json.dumps(data, indent=2, sort_keys=True).encode("utf-8"))

    def _check_identity(self) -> None:
        import json

        if self.cfg.dry_run != self.run_id.startswith(DRY_PREFIX):
            raise ConfigRefusal(
                f"run id {self.run_id!r}: dry runs must use the {DRY_PREFIX!r} prefix and real "
                "runs must not (dry-run and real results are never mixed)"
            )
        meta_path = self.run_dir / "run_meta.json"
        if meta_path.exists():
            prev = json.loads(meta_path.read_text(encoding="utf-8"))
            if bool(prev.get("dry_run")) != self.cfg.dry_run:
                raise ConfigRefusal(
                    f"run {self.run_id!r} was created with a different dry_run flag"
                )
        meta = {
            "run_id": self.run_id,
            "dry_run": self.cfg.dry_run,
            "label": DRY_RUN_LABEL if self.cfg.dry_run else None,
            "pipeline_version": PIPELINE_VERSION,
            "config": self.cfg.model_dump(mode="json"),
        }
        atomic_write_bytes(meta_path, json.dumps(meta, indent=2, sort_keys=True).encode("utf-8"))

    def _bill_sandbox_on_close(self, server: Any, purpose: str) -> Any:
        """Record the measured sandbox seconds (per-batch ``load_s`` + ``gen_s``) when the
        server is closed. Priced by the ledger only if a sandbox price is configured."""
        inner_close = server.close

        def close() -> None:
            try:
                inner_close()
            finally:
                timings = getattr(server, "timings", None) or []
                secs = sum(
                    float(t.get("load_s") or 0) + float(t.get("gen_s") or 0) for t in timings
                )
                n = sum(int(t.get("n") or 0) for t in timings)
                if secs > 0:
                    self.ledger.record_sandbox(secs, purpose, samples=n)
                server.timings = []  # a second close must not bill the same batches again

        server.close = close
        return server

    def _resolve_serving(self) -> None:
        """Build student/base factories from Deps when ``student_serving`` asks for it."""
        mode, deps = self.cfg.student_serving, self.deps
        if mode == "injected":
            return
        if deps.student_factory is not None or deps.base_factory is not None:
            raise ConfigRefusal(f"student_serving={mode!r} but Deps already has serving factories")
        base_model = self.config.require_model("student")
        if mode == "sandbox_cpu":
            from distillery.sandbox_student import SandboxCpuStudent, ServingImages

            sandbox, bridge = deps.sandbox, deps.bridge
            image = self.cfg.student_sandbox_image or deps.sandbox_image
            if sandbox is None:
                raise ConfigRefusal("student_serving='sandbox_cpu' needs Deps.sandbox")
            # ONE heavy image (pip + weights) for the base and every student of this run
            images = ServingImages(
                sandbox, image, bridge, base_model=base_model, on_image=self._record_serving_image
            )
            kw: dict[str, Any] = {
                "batch_size": self.cfg.student_batch_size,
                "concurrency": self.cfg.student_concurrency,
                "max_new_tokens": self.cfg.student_max_new_tokens,
                "images": images,
            }
            self.deps = replace(
                deps,
                student_factory=lambda trained: self._bill_sandbox_on_close(
                    SandboxCpuStudent.for_artifact(
                        sandbox, image, bridge, trained, base_model=base_model, **kw
                    ),
                    "student",
                ),
                base_factory=lambda: self._bill_sandbox_on_close(
                    SandboxCpuStudent(sandbox, image, bridge, base_model=base_model, **kw), "base"
                ),
            )  # fmt: skip
        else:
            from distillery import endpoint_student

            if deps.endpoint is None or self.cfg.student_endpoint is None:
                raise ConfigRefusal(
                    "student_serving='endpoint' needs Deps.endpoint and PipelineConfig."
                    "student_endpoint (with an explicit hourly price)"
                )
            spec = self.cfg.student_endpoint
            if spec.base_model_name != base_model:
                raise ConfigRefusal(
                    f"EndpointSpec.base_model_name={spec.base_model_name!r} must equal the "
                    f"configured student model {base_model!r}: the endpoint must serve the model "
                    "that was fine-tuned"
                )
            inner_student = endpoint_student.make_student_factory(deps.endpoint, spec, self.ledger)

            def endpoint_student_factory(trained: TrainedArtifact) -> StudentServer:
                _check_endpoint_identity(spec, trained, base_model)  # before anything is billed
                return inner_student(trained)

            self.deps = replace(
                deps,
                student_factory=endpoint_student_factory,
                base_factory=endpoint_student.make_base_factory(deps.endpoint, spec, self.ledger),
            )

    def _record_serving_image(self, kind: str, image: str) -> None:
        """Provenance of every sandbox image built for serving (they are what ran the models)."""
        self.store.add_experiment(
            self.run_id,
            "serving_image",
            {"kind": kind, "image": image, "student": self.config.require_model("student")},
        )

    def _preconditions(self) -> None:
        self._resolve_serving()
        self._check_identity()
        if self.cfg.ultra_verifier_authoring:
            raise NotImplementedError(
                "Ultra-authored verifiers are a documented hook, not built: the SQL pack uses the "
                "deterministic execution-match verifier (DECISIONS.md 2026-09-29)."
            )
        if self.cfg.max_rounds < 1:
            raise ConfigRefusal("max_rounds must be >= 1")
        self.config.require_model("student")  # fine-tune base id; served via base_factory, not LLM
        roles = ["planner", "teacher"] + (["triage"] if self.cfg.use_triage else [])
        for role in roles:
            model = self.config.require_model(role)
            if model not in self.config.prices:
                raise ConfigRefusal(f"no price for {role} model {model!r}; refusing to guess")
        if self.deps.student_factory is None or self.deps.base_factory is None:
            raise ServingUnavailableError(
                "no student/base serving path is available (spike S4 undecided; the base model is "
                "not on the serverless API either); refusing to start before spending anything"
            )
        student = self.config.require_model("student")
        if student not in self.config.finetune_prices and self.cfg.finetune_estimate_usd is None:
            raise ConfigRefusal(
                f"no fine-tune price for {student!r} in the price file and finetune_estimate_usd "
                "is not set: refusing to guess a fine-tune cost"
            )

    # ---- run -------------------------------------------------------------
    def run(self) -> dict[str, Any]:
        self._preconditions()
        self.store.create_run(self.run_id)
        if self.cfg.dry_run:
            self.say(DRY_RUN_LABEL)
        self._stage_schema()
        self._stage_questions()
        self._stage_crosscheck()
        self._stage_selftest()
        split = self._stage_split()
        self._stage_headroom(split)
        data = self._stage_teacher_data(split)
        rounds = self._run_rounds(split, data)
        final = self._stage_final_eval(rounds)
        report = self._build_report(split, data, rounds, final)
        atomic_write_bytes(
            self.run_dir / "report.json",
            _pretty(report).encode("utf-8"),
        )
        self._stamp_manifest()
        return report

    # ---- stage 1: schema -------------------------------------------------
    def _stage_schema(self) -> dict[str, Any]:
        seed = self.cfg.db_seed

        def fn() -> dict[str, Any]:
            data = sql_schema.build_database(seed)
            atomic_write_bytes(self.db_path, data)
            return {"db_seed": seed, "db_sha256": sha256_hex(data), "db_bytes": len(data)}

        res = self._stage("schema", {"db_seed": seed}, [], fn)
        ok = self.db_path.exists() and sha256_hex(self.db_path.read_bytes()) == res["db_sha256"]
        if not ok:  # cached result but file missing/changed: rebuild deterministically
            data = sql_schema.build_database(seed)
            if sha256_hex(data) != res["db_sha256"]:
                raise PipelineError("schema build is not deterministic: sha256 differs from record")
            atomic_write_bytes(self.db_path, data)
        self.ddl = sql_schema.schema_ddl(seed)
        return res

    # ---- stage 2: questions ---------------------------------------------
    def _stage_questions(self) -> dict[str, Any]:
        sc = self.cfg.scale
        n_total = math.ceil((sc.train + sc.dev + sc.heldout) * self.cfg.oversample)

        def fn() -> dict[str, Any]:
            rep = generate_tasks(self.db_ref, n_total, self.cfg.seed)
            return {
                "tasks": [_item(t) for t in rep.tasks],
                "counters": {
                    "requested": n_total,
                    "generated": len(rep.tasks),
                    "shortfall": max(0, n_total - len(rep.tasks)),
                    "attempts": rep.attempts,
                    "dropped_error": rep.dropped_error,
                    "dropped_empty": rep.dropped_empty,
                    "dropped_duplicate": rep.dropped_duplicate,
                },
                "dropped_by_template": dict(rep.dropped_by_template),
            }

        res = self._stage("questions", {"n_total": n_total, "seed": self.cfg.seed}, ["schema"], fn)
        self._observe([_task_from(d) for d in res["tasks"]])
        return res

    def _observe(self, tasks: Sequence[SqlTask]) -> None:
        if self.deps.task_observer is not None:
            self.deps.task_observer(tasks)

    # ---- stage 3: gold cross-check ---------------------------------------
    def _stage_crosscheck(self) -> dict[str, Any]:
        def fn() -> dict[str, Any]:
            tasks = [_task_from(d) for d in self.questions["tasks"]]
            msgs = [build_messages(t.question, self.ddl, role="train") for t in tasks]
            answers: dict[str, list[str | None]] = {}
            for role in ("planner", "teacher"):
                res = self.runner.map(
                    role, msgs, purpose=f"crosscheck_{role}", stage="gold_crosscheck",
                    schema=SqlAnswer,
                )  # fmt: skip
                answers[role] = [
                    r.parsed.sql.strip() if r is not None and r.parsed is not None else None
                    for r in res
                ]
            ex = self.deps.executor
            gold = ex.run_batch(self.db_ref, [t.gold_sql for t in tasks])
            outs: dict[str, dict[int, ExecOutcome]] = {}
            for role, sqls in answers.items():
                idx = [i for i, s in enumerate(sqls) if s]
                got = ex.run_batch(self.db_ref, [sqls[i] or "" for i in idx])
                outs[role] = dict(zip(idx, got, strict=True))
            reasons: Counter[str] = Counter()
            accepted: list[dict[str, Any]] = []
            suspect: list[str] = []
            identical_text = 0
            for i, t in enumerate(tasks):
                why = self._crosscheck_one(t, gold[i], i, answers, outs, suspect)
                if why is None:
                    accepted.append(_item(t))
                    if answers["planner"][i] == answers["teacher"][i]:
                        identical_text += 1
                else:
                    reasons[why] += 1
            return {
                "accepted": accepted,
                "suspect_gold_task_ids": suspect,
                "discarded_by_reason": dict(reasons),
                "counters": {
                    "checked": len(tasks),
                    "accepted": len(accepted),
                    "discarded": len(tasks) - len(accepted),
                    "accepted_with_identical_sql_text": identical_text,
                    "suspect_gold": len(suspect),
                    **{f"discard_{k}": v for k, v in reasons.items()},
                },
            }

        return self._stage("gold_crosscheck", {}, ["questions"], fn)

    @staticmethod
    def _crosscheck_one(
        t: SqlTask,
        gold: ExecOutcome,
        i: int,
        answers: Mapping[str, Sequence[str | None]],
        outs: Mapping[str, Mapping[int, ExecOutcome]],
        suspect: list[str],
    ) -> str | None:
        """Reason the task is discarded, or None if planner, teacher and gold all agree."""
        if not gold.ok:
            return "gold_error"
        for role in ("planner", "teacher"):
            if answers[role][i] is None:
                return f"{role}_no_answer"  # LLM error, schema failure or empty sql
        for role in ("planner", "teacher"):
            if not outs[role][i].ok:
                return f"{role}_exec_error"
        pm = compare_outcomes(outs["planner"][i], gold, t.requires_order).ok
        tm = compare_outcomes(outs["teacher"][i], gold, t.requires_order).ok
        if pm and tm:
            return None
        if pm != tm:
            return "planner_mismatch_only" if tm else "teacher_mismatch_only"
        agree = compare_outcomes(outs["planner"][i], outs["teacher"][i], t.requires_order).ok
        if agree:
            suspect.append(t.task_id)  # both models agree with each other but not the template
            return "both_mismatch_agreeing_suspect_gold"
        return "both_mismatch_disagreeing"

    @property
    def questions(self) -> dict[str, Any]:
        return self._results["questions"]

    # ---- stage 4: verifier self-test -------------------------------------
    def _stage_selftest(self) -> dict[str, Any]:
        def fn() -> dict[str, Any]:
            accepted = [_task_from(d) for d in self._results["gold_crosscheck"]["accepted"]]
            rng = random.Random(self.cfg.seed + 4)  # noqa: S311
            sample = rng.sample(accepted, min(self.cfg.selftest_sample, len(accepted)))
            ex = self.deps.executor
            failures: list[str] = []
            by_kind: Counter[str] = Counter()
            without = 0
            tested = 0
            for n, t in enumerate(sample):
                variants = corrupt_sql(
                    t.gold_sql,
                    self.db_ref,
                    random.Random(self.cfg.seed * 1000 + n),  # noqa: S311
                    max_variants=self.cfg.selftest_corruptions,
                )
                if not variants:
                    without += 1
                outs = ex.run_batch(self.db_ref, [t.gold_sql, *[v.sql for v in variants]])
                g_exec, v_outs = outs[0], outs[1:]
                g_local = run_select(self.db_ref, t.gold_sql)
                cmp = compare_outcomes(g_exec, g_local, t.requires_order)
                if not g_exec.ok or not cmp.ok:
                    failures.append(
                        f"gold rejected/inconsistent for {t.task_id}: sandbox ok={g_exec.ok} "
                        f"kind={g_exec.error_kind} error={g_exec.error!r}; vs local: {cmp.reason}"
                    )
                    continue
                for v, o in zip(variants, v_outs, strict=True):
                    tested += 1
                    by_kind[v.kind] += 1
                    if not o.ok:
                        failures.append(
                            f"{t.task_id} [{v.kind}]: executor errored on a corruption that runs "
                            f"locally: {o.error}"
                        )
                    elif compare_outcomes(o, g_exec, t.requires_order).ok:
                        failures.append(f"{t.task_id} [{v.kind}]: corruption ACCEPTED: {v.sql}")
            if tested == 0:
                failures.append("no corruptions could be generated: self-test would be vacuous")
            if failures:
                shown = "\n  ".join(failures[:10])
                raise VerifierSelfTestError(
                    f"verifier self-test failed ({len(failures)} problems):\n  {shown}"
                )
            return {
                "by_kind": dict(by_kind),
                "counters": {
                    "tasks_sampled": len(sample),
                    "corruptions_tested": tested,
                    "tasks_without_corruptions": without,
                },
            }

        return self._stage(
            "verifier_selftest",
            {"sample": self.cfg.selftest_sample, "k": self.cfg.selftest_corruptions},
            ["gold_crosscheck"],
            fn,
        )

    # ---- stage 6: split & seal -------------------------------------------
    def _stage_split(self) -> dict[str, Any]:
        def fn() -> dict[str, Any]:
            accepted = [_task_from(d) for d in self._results["gold_crosscheck"]["accepted"]]
            plan = plan_split(accepted, self.cfg.scale, self.cfg.seed)
            labels = classify_heldout(plan.train, plan.heldout)
            items = [_item(t, labels[t.task_id]) for t in plan.heldout]
            sealed = self.store.seal_heldout(self.run_id, items)
            rng = random.Random(self.cfg.seed + 6)  # noqa: S311
            spot = rng.sample(items, min(self.cfg.spot_check_n, len(items)))
            atomic_write_bytes(
                self.run_dir / "spot_check.json",
                _pretty(
                    {
                        "purpose": "manual review of random held-out items (question vs gold SQL)",
                        "items": [
                            {
                                k: it[k]
                                for k in (
                                    "task_id",
                                    "family",
                                    "heldout_class",
                                    "question",
                                    "gold_sql",
                                )
                            }
                            for it in spot
                        ],
                    }
                ).encode("utf-8"),
            )
            class_counts = Counter(labels.values())
            return {
                "train": [_item(t) for t in plan.train],
                "dev": [_item(t) for t in plan.dev],
                "heldout_task_ids": sorted(t.task_id for t in plan.heldout),
                "heldout_gold_sha256": sorted(_digest(t.gold_sql) for t in plan.heldout),
                "heldout_families": plan.heldout_families,
                "heldout_class_counts": dict(class_counts),
                "sealed_sha256": sealed,
                "counters": {
                    **plan.counters,
                    "train": len(plan.train),
                    "dev": len(plan.dev),
                    "heldout": len(plan.heldout),
                    "spot_check_items": len(spot),
                },
            }

        res = self._stage(
            "split", {"scale": self.cfg.scale.model_dump(), "n_spot": self.cfg.spot_check_n},
            ["gold_crosscheck"], fn,
        )  # fmt: skip
        self.sealed_sha = str(res["sealed_sha256"])
        self.say(f"[held-out] sealed sha256={self.sealed_sha} n={res['counters']['heldout']}")
        for d in (*res["train"], *res["dev"]):
            self.tasks_by_id[d["task_id"]] = _task_from(d)
        return res

    # ---- stage 5b: headroom ---------------------------------------------
    def _stage_headroom(self, split: Mapping[str, Any]) -> dict[str, Any]:
        def fn() -> dict[str, Any]:
            dev = [_task_from(d) for d in split["dev"]]
            assert self.deps.base_factory is not None  # noqa: S101 - checked in preconditions
            server = self.deps.base_factory()
            try:
                scores = self._score(server, dev, "base")
            finally:
                server.close()
            acc = scores.accuracy
            if acc >= self.cfg.headroom_max_base_acc:
                raise TaskTooEasyError(
                    f"task too easy: base model accuracy on dev is {acc:.3f} >= "
                    f"{self.cfg.headroom_max_base_acc} (n={len(dev)}); nothing to distil"
                )
            return {
                "base_dev_acc": acc,
                "counters": {
                    "dev_n": len(dev),
                    "base_unparseable": scores.unparseable,
                    **_generation_error_counter("base", server),
                },
            }

        return self._stage("headroom", {"max_acc": self.cfg.headroom_max_base_acc}, ["split"], fn)

    def _score(self, gen: Any, tasks: Sequence[SqlTask], role: str) -> ModelScores:
        items = [
            {"question": t.question, "gold_sql": t.gold_sql, "requires_order": t.requires_order}
            for t in tasks
        ]
        gold = self.deps.executor.run_batch(self.db_ref, [t.gold_sql for t in tasks])
        for t, g in zip(tasks, gold, strict=True):
            if not g.ok:
                raise PipelineError(f"gold SQL failed for {t.task_id}: {g.error}")
        return score_model(
            gen, items, gold, schema_ddl=self.ddl, db_ref=self.db_ref,
            executor=self.deps.executor, role=role,
        )  # fmt: skip

    # ---- stage 5: teacher data -------------------------------------------
    def _teacher_rows(self, tasks: Sequence[SqlTask], purpose: str, stage: str) -> dict[str, Any]:
        ex = self.deps.executor
        gold = ex.run_batch(self.db_ref, [t.gold_sql for t in tasks])
        pending = [i for i, g in enumerate(gold) if g.ok]
        c: Counter[str] = Counter(tasks=len(tasks), gold_error=len(tasks) - len(pending))
        rows: dict[int, dict[str, list[dict[str, str]]]] = {}
        raw: dict[int, str] = {}
        for attempt in range(1, self.cfg.candidates_per_task + 1):
            if not pending:
                break
            msgs = [build_messages(tasks[i].question, self.ddl, role="train") for i in pending]
            res = self.runner.map(
                "teacher", msgs, purpose=purpose, stage=stage,
                temperature=0.0 if attempt == 1 else 0.8,
            )  # fmt: skip
            sqls: list[tuple[int, str]] = []
            for i, r in zip(pending, res, strict=True):
                c["candidates"] += 1
                if r is None:
                    c["llm_error"] += 1
                    continue
                sql = extract_sql(r.text)
                if sql is None:
                    c["no_sql_extracted"] += 1
                    continue
                raw[i] = r.text
                sqls.append((i, sql))
            outs = ex.run_batch(self.db_ref, [s for _, s in sqls])
            still: list[int] = []
            for (i, sql), o in zip(sqls, outs, strict=True):
                v = compare_outcomes(o, gold[i], tasks[i].requires_order)
                if v.ok:
                    rows[i] = to_training_row(tasks[i], sql, schema_ddl=self.ddl)
                    c["verified"] += 1
                else:
                    c["exec_error" if not o.ok else "result_mismatch"] += 1
                    still.append(i)
            failed_llm = [i for i in pending if i not in rows and i not in {j for j, _ in sqls}]
            pending = sorted(set(still) | set(failed_llm))
        c["dropped"] = len(tasks) - c["verified"]
        triage = self._triage([raw[i] for i in sorted(rows) if i in raw])
        ordered = sorted(rows)
        return {
            "rows": [rows[i] for i in ordered],
            "task_ids": [tasks[i].task_id for i in ordered],
            "counters": {**c, **triage},
        }

    def _triage(self, raws: Sequence[str]) -> dict[str, int]:
        """Advisory format check by the triage model. Never drops rows (they are execution-
        verified and their completion is the extracted SQL); only counts what it flags."""
        if not self.cfg.use_triage or not raws:
            return {"triage_calls": 0, "triage_flagged_dirty_format": 0, "triage_errors": 0}
        sample = list(raws[: self.cfg.triage_sample])
        msgs = [
            [
                {
                    "role": "user",
                    "content": "Is the following model output ONLY a single SQL query, with no "
                    f"prose or markdown fences?\n\n{raw}",
                }
            ]
            for raw in sample
        ]
        res = self.runner.map(
            "triage", msgs, purpose="triage_format", stage="teacher_data", schema=FormatVerdict
        )
        errors = sum(1 for r in res if r is None or r.parsed is None)
        flagged = sum(1 for r in res if r is not None and r.parsed and not r.parsed.clean)
        return {
            "triage_calls": len(sample),
            "triage_flagged_dirty_format": flagged,
            "triage_errors": errors,
        }

    def _stage_teacher_data(self, split: Mapping[str, Any]) -> dict[str, Any]:
        def fn() -> dict[str, Any]:
            train = [_task_from(d) for d in split["train"]]
            return self._teacher_rows(train, "teacher_train", "teacher_data")

        return self._stage(
            "teacher_data",
            {"candidates": self.cfg.candidates_per_task, "triage": self.cfg.use_triage,
             "triage_sample": self.cfg.triage_sample},
            ["split", "headroom"],
            fn,
        )  # fmt: skip

    # ---- stage 7: fine-tune ----------------------------------------------
    def _stage_finetune(
        self, r: int, rows: Sequence[Mapping[str, Any]], dev: Sequence[SqlTask], upstream: str
    ) -> dict[str, Any]:
        name = f"finetune_r{r}"
        rows_hash = _digest(list(rows))

        def fn() -> dict[str, Any]:
            ft = self.deps.finetune
            base_model = self.config.require_model("student")
            self._finetune_preflight(len(rows))
            rdir = self.run_dir / f"round{r}"
            train_p, val_p = rdir / "train.jsonl", rdir / "dev.jsonl"
            atomic_write_bytes(
                train_p, ("\n".join(canonical_json(x) for x in rows) + "\n").encode("utf-8")
            )
            val_rows = [to_training_row(t, t.gold_sql, schema_ddl=self.ddl) for t in dev]
            atomic_write_bytes(
                val_p, ("\n".join(canonical_json(x) for x in val_rows) + "\n").encode("utf-8")
            )
            orphans = self._cancel_orphans(r)
            existing = self._adoptable_job(r)  # a paid job an earlier attempt left ambiguous
            # Refuse BEFORE any upload/job: planned tokens x price when the price exists,
            # else the operator ceiling.
            planned = _planned_trained_tokens(
                train_p.read_bytes(), self.cfg.chars_per_token, self.cfg.hyperparameters.n_epochs
            )
            pre_usd, pre_basis = self.ledger.preflight_finetune(
                base_model, planned, self.cfg.finetune_estimate_usd
            )
            self.say(f"[finetune r{r}] preflight ${pre_usd:.4f} ({pre_basis})")
            job_ref: list[str] = []

            def create() -> str:
                if existing is not None:
                    jid = existing
                    self.say(f"[finetune r{r}] adopting job {jid} left by an earlier attempt")
                    self.store.add_experiment(
                        self.run_id, "finetune_job_adopted", {"round": r, "job_id": jid}
                    )
                else:
                    train_id, val_id = ft.upload(train_p), ft.upload(val_p)
                    jid = ft.create_job(
                        base_model, train_id, val_id, self.cfg.hyperparameters,
                        suffix=f"distillery-{self.run_id}-r{r}"[:64], seed=self.cfg.seed,
                    )  # fmt: skip
                    self.store.add_experiment(
                        self.run_id, "finetune_job_started", {"round": r, "job_id": jid}
                    )
                job_ref.append(jid)
                return jid

            outcome = "aborted"
            cost_line: dict[str, Any] = {}
            try:
                with paid_job(create, ft.cancel) as handle:
                    info = FineTuneClient.require_success(
                        ft.poll(
                            handle.job_id,
                            interval_s=self.cfg.poll_interval_s,
                            timeout_s=self.cfg.poll_timeout_s,
                            on_update=lambda i: self.say(f"[finetune r{r}] {i.status}"),
                            on_error=lambda e: self.say(
                                f"[finetune r{r}] status check failed ({type(e).__name__}); "
                                "the job keeps running, still polling"
                            ),
                        )
                    )
                    cks = ft.checkpoints(handle.job_id)
                    if not cks:
                        raise PipelineError(f"job {handle.job_id} succeeded with no checkpoints")
                    art = ft.trained_artifact(info, cks[-1], rdir / "checkpoints")
                    training = self._training_record(ft, r, info, base_model)
                    # Record what was trained BEFORE anything is evaluated.
                    self.store.add_experiment(
                        self.run_id,
                        "expected_artifact",
                        {
                            "round": r,
                            "job_id": art.job_id,
                            "checkpoint_id": art.checkpoint_id,
                            "adapter_sha256": art.adapter_sha256,
                        },
                    )
                    ft_usd, ft_basis = self.ledger.record_finetune(
                        base_model, info.trained_tokens, self.cfg.finetune_estimate_usd
                    )
                    cost_line = {
                        "usd": ft_usd,
                        "basis": ft_basis,
                        "trained_tokens": info.trained_tokens,
                    }
                    self.say(f"[finetune r{r}] cost ${ft_usd:.4f} ({ft_basis})")
                    handle.mark_succeeded()
                    outcome = "succeeded"
            finally:
                if job_ref:
                    self.store.add_experiment(
                        self.run_id,
                        "finetune_job_closed",
                        {"job_id": job_ref[0], "outcome": outcome},
                    )
            return {
                "artifact": artifact_to_json(art, self.run_dir),
                "training": training,
                "train_rows": len(rows),
                "dev_validation_rows": len(val_rows),
                "cost": cost_line,
                "counters": {
                    "orphan_jobs_cancelled": orphans[0],
                    "orphan_cancel_errors": orphans[1],
                    "train_rows": len(rows),
                },
            }

        return self._stage(
            name,
            {"rows": rows_hash, "hp": self.cfg.hyperparameters.to_request(),
             "est": float(self.cfg.finetune_estimate_usd or 0.0),
             "student": self.config.model_ids.get("student"), "seed": self.cfg.seed},
            [upstream],
            fn,
        )  # fmt: skip

    def _finetune_preflight(self, train_rows: int) -> None:
        """Refuse (before any upload or spend) hyperparameters that would under-train."""
        hp = self.cfg.hyperparameters
        try:
            require_explicit_hyperparameters(hp)
        except ConfigError as exc:
            raise ConfigRefusal(str(exc)) from exc
        if self.cfg.dry_run:
            return
        if hp.packing:
            if not self.cfg.allow_packing:
                raise ConfigRefusal(
                    "packing=True makes the optimizer-step count unknowable; refusing. "
                    "Set allow_packing to accept that explicitly"
                )
            return
        steps = planned_steps(train_rows, hp)
        assert steps is not None  # noqa: S101 - packing is False and batch/epochs are set
        if steps < self.cfg.min_planned_steps:
            raise ConfigRefusal(
                f"planned optimizer steps {steps} = ceil({train_rows} rows / batch "
                f"{hp.batch_size}) x {hp.n_epochs} epochs is below min_planned_steps="
                f"{self.cfg.min_planned_steps}; the student would be barely trained. Add rows or "
                "epochs, lower batch_size, or lower min_planned_steps deliberately"
            )

    def _training_record(
        self, ft: FineTuner, r: int, info: JobInfo, base_model: str
    ) -> dict[str, Any]:
        """What the provider actually trained with (resolved values) and how the loss moved.
        Diagnostics only: a failure to read them must never fail (and so cancel) a paid job."""
        rec: dict[str, Any] = {
            "round": r,
            "job_id": info.id,
            "base_model": info.model or base_model,
            "hyperparameters": info.hyperparameters,
            "trained_tokens": info.trained_tokens,
            "trained_steps": info.trained_steps,
            "total_steps": info.total_steps,
            "loss_curve": [],
            "events": [],
        }
        try:
            rec["loss_curve"] = ft.loss_curve(info.id)
            rec["events"] = [
                {"created_at": e.created_at, "level": e.level, "message": e.message}
                for e in ft.events(info.id)
            ]
        except Exception as exc:  # noqa: BLE001
            rec["diagnostics_error"] = f"{type(exc).__name__}: {exc}"
        return rec

    def _adoptable_job(self, r: int) -> str | None:
        """A job of this round that an earlier attempt aborted but may not have managed to cancel
        (e.g. the network died mid-poll and the cancel failed too): if it is still active or
        succeeded, use it rather than paying for a second one. A failed status check raises: the
        wrong guess would create a duplicate billable job."""
        started: list[str] = []
        outcome: dict[str, str] = {}
        for name, data in self.store.list_experiments(self.run_id):
            if name == "finetune_job_started" and data.get("round") == r:
                started.append(str(data["job_id"]))
            elif name == "finetune_job_closed":
                outcome[str(data["job_id"])] = str(data.get("outcome"))
        for jid in reversed(started):
            if outcome.get(jid) != "aborted":
                continue
            status = self.deps.finetune.get(jid).status
            if status == "succeeded" or status in ACTIVE_STATUSES:
                return jid
        return None

    def _cancel_orphans(self, r: int) -> tuple[int, int]:
        """Cancel jobs of this round that were started but never closed (e.g. process killed),
        so a resume cannot leave a second billable job running."""
        started: dict[str, int] = {}
        closed: set[str] = set()
        for name, data in self.store.list_experiments(self.run_id):
            if name == "finetune_job_started" and data.get("round") == r:
                started[str(data["job_id"])] = 1
            elif name == "finetune_job_closed":
                closed.add(str(data["job_id"]))
        done = errors = 0
        for jid in started:
            if jid in closed:
                continue
            try:
                self.deps.finetune.cancel(jid)
                done += 1
            except (openai.APIStatusError, openai.APIConnectionError) as exc:
                errors += 1
                log.warning("could not cancel orphaned job %s: %s", jid, exc)
            self.store.add_experiment(
                self.run_id, "finetune_job_closed", {"job_id": jid, "outcome": "orphan_cancelled"}
            )
        return done, errors

    def _expected(self, r: int) -> ExpectedArtifact:
        found = [
            d
            for n, d in self.store.list_experiments(self.run_id)
            if n == "expected_artifact" and d.get("round") == r
        ]
        if not found:
            raise PipelineError(f"no expected_artifact recorded for round {r}")
        d = found[-1]
        return ExpectedArtifact(str(d["job_id"]), str(d["checkpoint_id"]), str(d["adapter_sha256"]))

    # ---- stage 9: dev eval / analysis / targeted / branch ------------------
    def _stage_dev_eval(
        self, r: int, ft: Mapping[str, Any], dev: Sequence[SqlTask]
    ) -> dict[str, Any]:
        def fn() -> dict[str, Any]:
            assert self.deps.student_factory is not None  # noqa: S101 - checked in preconditions
            server = self.deps.student_factory(artifact_from_json(ft["artifact"], self.run_dir))
            try:
                rec = RecordingGenerator(server)
                scores = self._score(rec, dev, "student")
                outputs = list(rec.outputs)
            finally:
                server.close()
            failures = [
                {
                    "task_id": t.task_id,
                    "family": t.family,
                    "question": t.question,
                    "gold_sql": t.gold_sql,
                    "student_sql": extract_sql(outputs[i]) or outputs[i],
                    "reason": scores.reasons[i],
                }
                for i, t in enumerate(dev)
                if not scores.correct[i]
            ]
            return {
                "dev_acc": scores.accuracy,
                "failures": failures,
                "counters": {
                    "dev_n": len(dev),
                    "dev_failures": len(failures),
                    "dev_unparseable": scores.unparseable,
                    **_generation_error_counter("student", server),
                },
            }

        return self._stage(
            f"dev_eval_r{r}", {"dev": [t.task_id for t in dev]}, [f"finetune_r{r}"], fn
        )

    def _stage_analysis(
        self, r: int, dev_res: Mapping[str, Any], train_families: Sequence[str],
        heldout_families: Sequence[str],
    ) -> dict[str, Any]:  # fmt: skip
        def fn() -> dict[str, Any]:
            failures = list(dev_res["failures"])[:40]
            (res,) = self.runner.map(
                "planner",
                [build_analysis_messages(failures, train_families)],
                purpose="failure_analysis",
                stage=f"analysis_r{r}",
                schema=FailureClusters,
            )
            if res is None or res.parsed is None:
                return {"clusters": [], "target_families": [],
                        "counters": {"analysis_failed": 1, "unknown_families_dropped": 0,
                                     "heldout_families_dropped": 0}}  # fmt: skip
            targets: list[str] = []
            unknown = heldout = 0
            for cl in res.parsed.clusters:
                for f in cl.target_families:
                    if f in heldout_families:
                        heldout += 1
                    elif f not in train_families or f not in FAMILIES:
                        unknown += 1
                    elif f not in targets:
                        targets.append(f)
            return {
                "clusters": [c.model_dump() for c in res.parsed.clusters],
                "target_families": sorted(targets),
                "counters": {"analysis_failed": 0, "unknown_families_dropped": unknown,
                             "heldout_families_dropped": heldout},
            }  # fmt: skip

        return self._stage(f"analysis_r{r}", {}, [f"dev_eval_r{r}"], fn)

    def _stage_targeted(
        self, r: int, analysis: Mapping[str, Any], split: Mapping[str, Any]
    ) -> dict[str, Any]:
        def fn() -> dict[str, Any]:
            known_gold = {_digest(t.gold_sql) for t in self.tasks_by_id.values()}
            known_gold |= set(split["heldout_gold_sha256"])
            rep = generate_tasks(
                self.db_ref,
                self.cfg.extra_tasks(),
                self.cfg.seed + 1000 * r,
                families=tuple(analysis["target_families"]),
            )
            fresh: list[SqlTask] = []
            c: Counter[str] = Counter(generated=len(rep.tasks))
            for t in rep.tasks:
                if t.task_id in split["heldout_task_ids"] or t.family in split["heldout_families"]:
                    c["dropped_heldout_overlap"] += 1
                elif _digest(t.gold_sql) in known_gold or t.task_id in self.tasks_by_id:
                    c["dropped_already_known"] += 1
                else:
                    fresh.append(t)
            self._observe(fresh)
            out = self._teacher_rows(fresh, "teacher_targeted", f"targeted_r{r}")
            out["tasks"] = [_item(t) for t in fresh]
            out["counters"] = {**c, **out["counters"], "new_tasks": len(fresh)}
            return out

        res = self._stage(
            f"targeted_r{r}", {"extra": self.cfg.extra_tasks()}, [f"analysis_r{r}"], fn
        )
        for d in res["tasks"]:  # also on a cached result, so later rounds dedupe identically
            self.tasks_by_id[d["task_id"]] = _task_from(d)
        return res

    def _stage_branch(
        self, r: int, parent: str, rows: Sequence[Mapping[str, Any]]
    ) -> dict[str, Any]:
        sb = self.deps.sandbox

        def fn() -> dict[str, Any]:
            if sb is None:
                return {"skipped": "no sandbox provided", "counters": {"branch_skipped": 1}}
            manifest = canonical_json({"round": r, "train_rows_sha256": _digest(list(rows))})
            uuid = self.deps.bridge.run(
                sb.branch(
                    parent, f"echo round-{r} > /round.txt",
                    files={f"/rounds/round{r}.json": manifest.encode()},
                )
            )  # fmt: skip
            return {"uuid": uuid, "parent": parent, "label": f"round-{r}",
                    "counters": {"branch_skipped": 0}}  # fmt: skip

        return self._stage(f"sandbox_branch_r{r}", {"parent": parent}, [f"targeted_r{r}"], fn)

    # ---- rounds ------------------------------------------------------------
    def _run_rounds(self, split: Mapping[str, Any], data: Mapping[str, Any]) -> dict[str, Any]:
        dev = [_task_from(d) for d in split["dev"]]
        train_fams = sorted({d["family"] for d in split["train"]})
        rows: list[dict[str, Any]] = list(data["rows"])
        rounds: list[dict[str, Any]] = []
        upstream = "teacher_data"
        parent_image = self.deps.sandbox_image
        stop = f"reached max_rounds={self.cfg.max_rounds}"
        for r in range(1, self.cfg.max_rounds + 1):
            ft = self._stage_finetune(r, rows, dev, upstream)
            dv = self._stage_dev_eval(r, ft, dev)
            rec: dict[str, Any] = {
                "round": r,
                "train_rows": len(rows),
                "dev_acc": dv["dev_acc"],
                "dev_failures": len(dv["failures"]),
                "adapter_sha256": ft["artifact"]["adapter_sha256"],
                "job_id": ft["artifact"]["job_id"],
                "finetune_stage": f"finetune_r{r}",
                "clusters": [],
            }
            rounds.append(rec)
            self.say(
                f"[round {r}] dev accuracy {dv['dev_acc']:.3f} ({len(dv['failures'])} failures)"
            )
            if dv["dev_acc"] >= self.cfg.dev_target_acc:
                stop = f"dev accuracy >= dev_target_acc={self.cfg.dev_target_acc}"
                break
            if not dv["failures"]:
                stop = "no dev failures"
                break
            if r == self.cfg.max_rounds:
                break
            an = self._stage_analysis(r, dv, train_fams, split["heldout_families"])
            rec["clusters"] = an["clusters"]
            if not an["target_families"]:
                stop = "failure analysis produced no usable target families"
                break
            tg = self._stage_targeted(r, an, split)
            rec["targeted_new_rows"] = len(tg["rows"])
            if not tg["rows"]:
                stop = "targeted generation produced no new verified rows"
                break
            br = self._stage_branch(r, parent_image, tg["rows"])
            if "uuid" in br:
                rec["sandbox_branch"] = {k: br[k] for k in ("uuid", "parent", "label")}
                parent_image = br["uuid"]
            rows = rows + list(tg["rows"])
            upstream = f"sandbox_branch_r{r}"
        best = max(rounds, key=lambda x: (x["dev_acc"], -x["round"]))  # ties: earliest round
        return {"rounds": rounds, "best_round": best["round"], "stop_reason": stop}

    # ---- stage 8: final held-out evaluation --------------------------------
    def _stage_final_eval(self, rounds: Mapping[str, Any]) -> dict[str, Any]:
        r = int(rounds["best_round"])
        ft_res = self._results[f"finetune_r{r}"]
        pipeline = self

        def fn() -> dict[str, Any]:
            trained = artifact_from_json(ft_res["artifact"], pipeline.run_dir)
            assert pipeline.deps.student_factory is not None  # noqa: S101
            assert pipeline.deps.base_factory is not None  # noqa: S101
            opened: list[StudentServer] = []
            try:
                server = pipeline.deps.student_factory(trained)
                opened.append(server)
                base_server = pipeline.deps.base_factory()
                opened.append(base_server)
                gens: dict[str, evaluator_mod.Generator] = {
                    "base": base_server,
                    "student": server,
                    "teacher": LLMGenerator(
                        pipeline.runner, "teacher", "eval_teacher", "final_eval"
                    ),
                }
                rep = evaluator_mod.evaluate(
                    pipeline.store, pipeline.run_id, gens, pipeline.deps.executor,
                    db_ref=pipeline.db_ref, schema_ddl=pipeline.ddl, gate_cfg=pipeline.config.gate,
                    trained=trained, expected=pipeline._expected(r),
                )  # fmt: skip
                rep = replace(
                    rep, artifact=_with_serving(rep.artifact, trained, server, base_server)
                )
            except BaseException:
                _close_all(opened, raise_first=False)
                raise
            _close_all(opened)
            return {
                "report": rep.to_json(),
                "candidate_round": r,
                "counters": {"heldout_n": rep.n,
                             **{f"unparseable_{m}": s.unparseable for m, s in rep.scores.items()},
                             **_generation_error_counter("student", server),
                             **_generation_error_counter("base", base_server)},
            }  # fmt: skip

        return self._stage(
            "final_eval",
            {"round": r, "gate": self.config.gate.model_dump()},
            [f"finetune_r{r}", "split"],
            fn,
        )

    # ---- report ------------------------------------------------------------
    def _cost_basis(self) -> dict[str, Any]:
        """Every cost figure here is an estimate; say so, with the basis of each cost line."""
        used = cost_by_model(self.store, self.run_id)
        sources = {m: self.config.prices[m].source for m in used if m in self.config.prices}
        names = "; ".join(sorted(set(sources.values()))) or "no priced model was used"
        lines = spend_lines(self.store, self.run_id)
        ft_bases = sorted({str(x["basis"]) for x in lines if str(x["kind"]).startswith("finetune")})
        sb_bases = sorted({str(x["basis"]) for x in lines if str(x["kind"]).startswith("sandbox")})
        return {
            "basis": "ESTIMATES, not billed amounts. LLM lines: measured token counts x the "
            f"configured price table (price source: {names}); fine-tune: "
            f"{', '.join(ft_bases) or 'no fine-tune ran'}; sandbox compute: "
            f"{', '.join(sb_bases) or 'not recorded'}",
            "price_sources": sources,
            "finetune_lines": [x for x in lines if str(x["kind"]).startswith("finetune")],
            "sandbox_lines": [x for x in lines if str(x["kind"]).startswith("sandbox")],
        }

    def _finetune_usd_text(self, rounds_run: int) -> str:
        lines = [x for x in spend_lines(self.store, self.run_id) if str(x["kind"]).startswith("f")]
        total = sum(float(x["usd"]) for x in lines)
        bases = sorted({str(x["basis"]) for x in lines}) or [BASIS_CEILING]
        return f"${total:.4f} [{', '.join(bases)}]; rounds run: {rounds_run}"

    def _student_cost_per_1k(self) -> Any:
        """Measured sandbox seconds per student sample x the sandbox price, or 'unavailable'."""
        secs, n = sandbox_seconds(self.store, self.run_id, "student")
        if self.ledger.sandbox_price is None or n == 0:
            return STUDENT_COST_UNAVAILABLE
        return {
            "basis": "billed-basis (measured seconds x console price)",
            "samples": n,
            "sandbox_seconds": secs,
            "usd_per_1k_tasks": self.ledger.estimate_sandbox_cost(secs) / n * 1000,
        }

    def _build_report(
        self,
        split: Mapping[str, Any],
        data: Mapping[str, Any],
        rounds: Mapping[str, Any],
        final: Mapping[str, Any],
    ) -> dict[str, Any]:
        ev = final["report"]
        n = int(ev["n"])
        teacher_use = final["metrics"]["usage"].get("eval_teacher", {})
        teacher_cost: dict[str, Any] = {
            "basis": "measured tokens x configured prices, from the held-out teacher pass",
            "held_out_tasks": n,
            "input_tokens": teacher_use.get("input_tokens", 0),
            "output_tokens": teacher_use.get("output_tokens", 0),
            "usd": teacher_use.get("usd_nano", 0) / 1e9,
            "usd_per_1k_tasks": (teacher_use.get("usd_nano", 0) / 1e9) / n * 1000 if n else None,
        }
        manifest_stages = [
            {"stage": s["stage"], "output_sha256": s["output_sha256"]}
            for s in stage_rows(self.store, self.run_id)
            if s["status"] == "complete"
        ]
        counters = {k: dict(v) for k, v in self.counters.items()}
        for stage, cs in counters.items():
            for key, val in cs.items():
                if val < 0:
                    raise PipelineError(f"negative counter {stage}.{key}={val}")
        llm_errors: Counter[str] = Counter()
        for res in self._results.values():
            llm_errors.update(res.get("metrics", {}).get("llm_errors", {}))
        retries = Counter[str]()
        for stage_usage in self.usage.values():
            for purpose, u in stage_usage.items():
                retries[f"{purpose}:failed_attempts"] += u.get("failed_attempts", 0)
                retries[f"{purpose}:schema_retries"] += u.get("schema_retries", 0)
        return {
            "label": DRY_RUN_LABEL if self.cfg.dry_run else None,
            "dry_run": self.cfg.dry_run,
            "run_id": self.run_id,
            "pack": self.cfg.pack,
            "decision": ev["gate"]["decision"],
            "decision_reasons": ev["gate"]["reasons"],
            "evaluation": ev,
            "candidate_round": final["candidate_round"],
            "candidate_selection": "best DEV accuracy across rounds (ties: earliest); held-out "
            "is scored once, for that candidate only",
            "data": {
                "train_tasks": len(split["train"]),
                "dev_tasks": len(split["dev"]),
                "heldout_tasks": len(split["heldout_task_ids"]),
                "heldout_families": split["heldout_families"],
                "heldout_class_counts": split["heldout_class_counts"],
                "heldout_sealed_sha256": split["sealed_sha256"],
                "teacher_verified_rows_round1": len(data["rows"]),
                "spot_check_file": "spot_check.json",
            },
            "headroom": {
                "base_dev_acc": self._results["headroom"]["base_dev_acc"],
                "max_allowed": self.cfg.headroom_max_base_acc,
            },
            "rounds": rounds["rounds"],
            "finetune": [
                self._results[rd["finetune_stage"]]["training"] for rd in rounds["rounds"]
            ],
            "rounds_stop_reason": rounds["stop_reason"],
            "counters": counters,
            "llm_errors_by_purpose": dict(llm_errors),
            "llm_attempt_counters": dict(retries),
            "cost": {
                **self._cost_basis(),
                "llm_by_model": cost_by_model(self.store, self.run_id),
                "run_total_usd": self.ledger.spent(),
                "run_cap_usd": self.ledger.run_cap,
                "finetune_usd": self._finetune_usd_text(len(rounds["rounds"])),
                "cost_per_1k_tasks": {
                    "teacher": teacher_cost,
                    "student": self._student_cost_per_1k(),
                },
            },
            "sandbox_lineage": [
                {k: v for k, v in res.items() if k in ("uuid", "parent", "label")}
                for name, res in sorted(self._results.items())
                if name.startswith("sandbox_branch_r") and "uuid" in res
            ],
            "config": {
                "pipeline": self.cfg.model_dump(mode="json"),
                "gate_thresholds": self.config.gate.model_dump(mode="json"),
                "seeds": {
                    "db_seed": self.cfg.db_seed,
                    "task_seed": self.cfg.seed,
                    "bootstrap_seed": self.config.gate.seed,
                },
                "model_ids": dict(self.config.model_ids),
                "prices": {k: v.model_dump() for k, v in self.config.prices.items()},
            },
            "stages": manifest_stages,
        }


def _pretty(obj: Any) -> str:
    import json

    return json.dumps(obj, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
