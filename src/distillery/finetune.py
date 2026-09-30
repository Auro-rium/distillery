"""Fine-tuning client for Nebius Token Factory (OpenAI-compatible fine-tuning API).

Doc sources (spikes/out): post-training_how-to-fine-tune.md (flow, hyperparameters),
post-training_models.md (which models), api-reference_fine-tuning_cancel-fine-tuning-job.md
(OpenAPI: cancel endpoint, status enum, hyperparameter bounds).

The client is synchronous and takes a duck-typed openai-like transport, so tests use a fake.
Callers in async code should run it via ``asyncio.to_thread``.
"""

from __future__ import annotations

import hashlib
import os
import random
import time
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import Any, Protocol

import openai
from pydantic import BaseModel, ConfigDict, Field

from distillery.llm import is_retryable

TERMINAL_STATUSES = frozenset({"succeeded", "failed", "cancelled"})
# Non-terminal per the OpenAPI enum / how-to: validating_files -> queued -> running.
ACTIVE_STATUSES = frozenset({"validating_files", "queued", "running"})
MIN_POLL_INTERVAL_S = 15.0  # doc: "Poll this endpoint every >= 15 seconds"
# Doc: every model supports these context lengths (post-training_models.md).
CONTEXT_LENGTHS = (8192, 16384, 32768, 65536, 131072)


class FineTuneError(Exception):
    pass


class JobFailedError(FineTuneError):
    def __init__(self, job_id: str, status: str, code: str | None, message: str | None) -> None:
        super().__init__(f"job {job_id} ended {status}: {code} {message}")
        self.job_id = job_id
        self.status = status


class PollTimeoutError(FineTuneError):
    pass


class HyperParameters(BaseModel):
    """Client-side mirror of the documented bounds (how-to page + OpenAPI schema)."""

    model_config = ConfigDict(extra="forbid")

    batch_size: int | None = Field(default=None, ge=1, le=64)  # OpenAPI 1..64 (doc "typical" 8-32)
    learning_rate: float | None = Field(default=None, ge=0)
    n_epochs: int | None = Field(default=None, ge=1, le=20)
    warmup_ratio: float | None = Field(default=None, ge=0, le=1)
    weight_decay: float | None = Field(default=None, ge=0)
    lora: bool | None = None
    lora_r: int | None = Field(default=None, ge=8, le=128)
    lora_alpha: int | None = Field(default=None, ge=8)
    lora_dropout: float | None = Field(default=None, ge=0, le=1)
    packing: bool | None = None
    max_grad_norm: float | None = Field(default=None, gt=0)  # OpenAPI: exclusiveMinimum 0
    context_length: int | None = Field(default=None, ge=8192, le=131072)

    def to_request(self) -> dict[str, Any]:
        data = self.model_dump(exclude_none=True)
        cl = data.get("context_length")
        if cl is not None and cl not in CONTEXT_LENGTHS:
            raise ValueError(f"context_length must be one of {CONTEXT_LENGTHS}, got {cl}")
        return data


@dataclass(frozen=True)
class JobInfo:
    id: str
    status: str
    model: str | None
    error_code: str | None
    error_message: str | None
    result_files: tuple[str, ...]
    trained_steps: int | None
    total_steps: int | None

    @property
    def terminal(self) -> bool:
        return self.status in TERMINAL_STATUSES


@dataclass(frozen=True)
class EventInfo:
    id: str | None
    created_at: int | None
    level: str | None
    message: str


@dataclass(frozen=True)
class CheckpointInfo:
    id: str
    step_number: int | None
    fine_tuned_model_checkpoint: str | None
    result_files: tuple[str, ...]
    metrics: Mapping[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class DownloadedFile:
    file_id: str
    path: Path
    sha256: str


@dataclass(frozen=True)
class TrainedArtifact:
    """What was trained. The evaluator must verify evaluated == trained via these hashes."""

    job_id: str
    checkpoint_id: str
    base_model: str | None
    fine_tuned_model_checkpoint: str | None
    files: tuple[DownloadedFile, ...]

    @property
    def file_sha256(self) -> dict[str, str]:
        return {f.path.name: f.sha256 for f in self.files}

    @property
    def adapter_sha256(self) -> str:
        """Single digest over all checkpoint files (sorted name + per-file sha256)."""
        h = hashlib.sha256()
        for name, digest in sorted(self.file_sha256.items()):
            h.update(f"{name}:{digest}\n".encode())
        return h.hexdigest()


class CancelFn(Protocol):
    def __call__(self, job_id: str) -> object: ...


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


@dataclass
class PaidJob:
    job_id: str
    succeeded: bool = False
    cancelled: bool = False
    cancel_error: str | None = None

    def mark_succeeded(self) -> None:
        self.succeeded = True


@contextmanager
def paid_job(create: Callable[[], str], cancel: CancelFn) -> Iterator[PaidJob]:
    """Guarantee the paid job is cancelled on any non-success exit.

    ``create`` starts the job and returns its id (if it raises, nothing was started).
    Inside the block call ``handle.mark_succeeded()`` once the job has succeeded and its
    artifacts are safely stored. Any exception, or leaving the block without marking
    success, triggers ``cancel(job_id)``. A cancel failure never masks the original error.
    """
    handle = PaidJob(create())
    try:
        yield handle
    except BaseException as exc:
        _cancel_quietly(handle, cancel, exc)
        raise
    else:
        if not handle.succeeded:
            _cancel_quietly(handle, cancel, None)
    # success path: nothing to cancel


def _cancel_quietly(handle: PaidJob, cancel: CancelFn, original: BaseException | None) -> None:
    try:
        cancel(handle.job_id)
        handle.cancelled = True
    except (openai.APIStatusError, openai.APIConnectionError) as exc:
        handle.cancel_error = f"{type(exc).__name__}: {exc}"
        if original is not None:
            original.add_note(f"ALSO: cancelling job {handle.job_id} failed: {handle.cancel_error}")
        else:
            raise FineTuneError(f"cancel of {handle.job_id} failed: {handle.cancel_error}") from exc


class FineTuneClient:
    def __init__(
        self,
        client: Any,
        *,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.monotonic,
        jitter: Callable[[], float] = random.random,
        max_attempts: int = 4,
        base_delay_s: float = 1.0,
        max_delay_s: float = 20.0,
    ) -> None:
        self._c = client
        self._sleep = sleep
        self._clock = clock
        self._jitter = jitter
        self._max_attempts = max_attempts
        self._base = base_delay_s
        self._max_delay = max_delay_s

    def _retry[T](self, fn: Callable[[], T]) -> T:
        """Bounded retry for idempotent calls: 429/5xx/timeouts only; 4xx is terminal."""
        for attempt in range(1, self._max_attempts + 1):
            try:
                return fn()
            except (openai.APIStatusError, openai.APIConnectionError) as exc:
                if not is_retryable(exc) or attempt == self._max_attempts:
                    raise
                delay = min(self._max_delay, self._base * 2 ** (attempt - 1))
                self._sleep(delay * (0.5 + 0.5 * self._jitter()))
        raise AssertionError("unreachable")  # pragma: no cover

    # -- files
    def upload(self, path: Path | str) -> str:
        p = Path(path)

        def go() -> str:
            with p.open("rb") as fh:
                return str(self._c.files.create(file=fh, purpose="fine-tune").id)

        return self._retry(go)

    # -- jobs
    def create_job(
        self,
        model: str,
        training_file: str,
        validation_file: str | None = None,
        hyperparameters: HyperParameters | None = None,
        suffix: str | None = None,
        seed: int | None = None,
    ) -> str:
        """Create a job (NOT auto-retried: a 5xx may have created a billable job)."""
        if suffix is not None and len(suffix) > 64:
            raise ValueError("suffix max length is 64 (OpenAPI)")
        req: dict[str, Any] = {"model": model, "training_file": training_file}
        if validation_file:
            req["validation_file"] = validation_file
        if hyperparameters is not None:
            req["hyperparameters"] = hyperparameters.to_request()
        if suffix:
            req["suffix"] = suffix
        if seed is not None:
            req["seed"] = seed
        return str(self._c.fine_tuning.jobs.create(**req).id)

    @staticmethod
    def _job_info(job: Any) -> JobInfo:
        err = getattr(job, "error", None)
        return JobInfo(
            id=str(job.id),
            status=str(job.status),
            model=getattr(job, "model", None),
            error_code=getattr(err, "code", None) if err else None,
            error_message=getattr(err, "message", None) if err else None,
            result_files=tuple(getattr(job, "result_files", None) or ()),
            trained_steps=getattr(job, "trained_steps", None),
            total_steps=getattr(job, "total_steps", None),
        )

    def get(self, job_id: str) -> JobInfo:
        return self._job_info(self._retry(lambda: self._c.fine_tuning.jobs.retrieve(job_id)))

    def poll(
        self,
        job_id: str,
        *,
        interval_s: float = MIN_POLL_INTERVAL_S,
        timeout_s: float | None = None,
        on_update: Callable[[JobInfo], None] | None = None,
        on_error: Callable[[BaseException], None] | None = None,
        transient_error_grace_s: float = 900.0,
    ) -> JobInfo:
        """Block until the job is terminal (succeeded / failed / cancelled).

        (The doc's sample loop ``while job.status in ["succeeded", ...]`` is inverted; this
        loops while the status is NOT terminal.)

        A retryable poll failure (connection error, 429/5xx; already retried a few times by
        ``get``) does not end the wait: the job runs (and bills) server-side regardless, and giving
        up here would make the caller cancel it, or leave it unclaimed if that cancel fails too
        (seen live: a 20 s DNS outage at minute 6 of a job that then succeeded). Polling continues
        for up to ``transient_error_grace_s`` of uninterrupted failures; terminal errors (4xx other
        than 429) raise at once.
        """
        if interval_s < MIN_POLL_INTERVAL_S:
            raise ValueError(f"poll interval must be >= {MIN_POLL_INTERVAL_S}s per docs")
        start = self._clock()
        failing_since: float | None = None
        while True:
            try:
                info = self.get(job_id)
            except (openai.APIStatusError, openai.APIConnectionError) as exc:
                now = self._clock()
                failing_since = now if failing_since is None else failing_since
                if (
                    not is_retryable(exc)
                    or now - failing_since + interval_s > transient_error_grace_s
                ):
                    raise
                if timeout_s is not None and now - start + interval_s > timeout_s:
                    raise PollTimeoutError(f"job {job_id} unreachable and unfinished") from exc
                if on_error is not None:
                    on_error(exc)
                self._sleep(interval_s)
                continue
            failing_since = None
            if on_update is not None:
                on_update(info)
            if info.terminal:
                return info
            if timeout_s is not None and self._clock() - start + interval_s > timeout_s:
                raise PollTimeoutError(f"job {job_id} still {info.status} after {timeout_s}s")
            self._sleep(interval_s)

    @staticmethod
    def require_success(info: JobInfo) -> JobInfo:
        if info.status != "succeeded":
            raise JobFailedError(info.id, info.status, info.error_code, info.error_message)
        return info

    def events(self, job_id: str, *, limit: int = 50, max_pages: int = 20) -> list[EventInfo]:
        out: list[EventInfo] = []
        after: str | None = None
        for _ in range(max_pages):
            kwargs: dict[str, Any] = {"limit": limit}
            if after:
                kwargs["after"] = after
            page = self._retry(partial(self._c.fine_tuning.jobs.list_events, job_id, **kwargs))
            data = list(page.data)
            out.extend(
                EventInfo(
                    getattr(e, "id", None),
                    getattr(e, "created_at", None),
                    getattr(e, "level", None),
                    str(e.message),
                )
                for e in data
            )
            if not getattr(page, "has_more", False) or not data:
                break
            after = getattr(data[-1], "id", None)
            if after is None:
                break
        return out

    def checkpoints(self, job_id: str) -> list[CheckpointInfo]:
        page = self._retry(lambda: self._c.fine_tuning.jobs.checkpoints.list(job_id))
        return [
            CheckpointInfo(
                id=str(c.id),
                step_number=getattr(c, "step_number", None),
                fine_tuned_model_checkpoint=getattr(c, "fine_tuned_model_checkpoint", None),
                result_files=tuple(c.result_files or ()),
                metrics=dict(getattr(c, "metrics", None) or {}),
            )
            for c in page.data
        ]

    def download_checkpoint(
        self, checkpoint: CheckpointInfo, directory: Path | str
    ) -> list[DownloadedFile]:
        """Download every result file of a checkpoint; returns paths + sha256 of each.

        Per the how-to, files are saved under ``basename(file.filename)``; the basename step
        also prevents path traversal from a hostile filename.
        """
        dest = Path(directory) / checkpoint.id
        dest.mkdir(parents=True, exist_ok=True)
        out: list[DownloadedFile] = []
        for file_id in checkpoint.result_files:
            meta = self._retry(partial(self._c.files.retrieve, file_id))
            name = os.path.basename(str(meta.filename))
            if not name or name in (".", ".."):
                raise FineTuneError(f"unusable filename for file {file_id}: {meta.filename!r}")
            target = dest / name
            content = self._retry(partial(self._c.files.content, file_id))
            content.write_to_file(str(target))
            out.append(DownloadedFile(file_id, target, sha256_file(target)))
        return out

    def trained_artifact(
        self, job: JobInfo, checkpoint: CheckpointInfo, directory: Path | str
    ) -> TrainedArtifact:
        files = self.download_checkpoint(checkpoint, directory)
        return TrainedArtifact(
            job_id=job.id,
            checkpoint_id=checkpoint.id,
            base_model=job.model,
            fine_tuned_model_checkpoint=checkpoint.fine_tuned_model_checkpoint,
            files=tuple(files),
        )

    def cancel(self, job_id: str) -> JobInfo:
        """POST /v1/fine_tuning/jobs/{id}/cancel (OpenAPI). Single attempt + bounded retry."""
        return self._job_info(self._retry(lambda: self._c.fine_tuning.jobs.cancel(job_id)))
