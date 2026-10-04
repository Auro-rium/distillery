"""One background worker thread that runs pipelines, one at a time."""

from __future__ import annotations

import functools
import os
import queue
import signal
import subprocess  # noqa: S404 - runs our own CLI with a fixed argv, never a shell
import sys
import threading
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

MAX_LOG_LINES = 2000
KILL_AFTER_INTERRUPT_S = 30.0


class RunConflictError(RuntimeError):
    pass


class QueueFullError(RuntimeError):
    pass


@dataclass
class Job:
    run_id: str
    scale: str
    dry_run: bool
    budget_usd: float | None = None
    finetune_estimate_usd: float | None = None
    state: str = "queued"  # queued | running | finished
    error: str | None = None
    logs: list[str] = field(default_factory=list)
    cancel_requested: bool = False
    interrupt: Callable[[], None] | None = None


Executor = Callable[[Job, Callable[[str], None]], int]


def redact(text: str, secrets: Sequence[str]) -> str:
    for s in secrets:
        if s:
            text = text.replace(s, "[redacted]")
    return text


def run_child(
    argv: Sequence[str],
    env: dict[str, str],
    job: Job,
    log: Callable[[str], None],
    kill_after_s: float,
) -> int:
    """Run ``argv`` in its OWN process group; ``job.interrupt`` sends SIGINT to the whole group
    and SIGKILLs it after ``kill_after_s`` if it is still alive."""
    proc = subprocess.Popen(  # noqa: S603
        argv, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
        start_new_session=True,
    )  # fmt: skip
    timers: list[threading.Timer] = []

    def _signal(sig: int) -> None:
        try:
            os.killpg(proc.pid, sig)
        except (ProcessLookupError, PermissionError):
            pass

    def interrupt() -> None:
        if proc.poll() is None:
            _signal(signal.SIGINT)
            t = threading.Timer(kill_after_s, lambda: _signal(signal.SIGKILL))
            t.daemon = True
            timers.append(t)
            t.start()

    job.interrupt = interrupt
    if job.cancel_requested:
        interrupt()
    try:
        for line in proc.stdout or ():
            log(line.rstrip("\n"))
        return proc.wait()
    finally:
        for t in timers:
            t.cancel()
        _signal(signal.SIGKILL)  # reap stragglers of the group (no-op if all exited)


def subprocess_executor(
    root: str,
    admin_token: str | None,
    secrets: Sequence[str],
    kill_after_s: float = KILL_AFTER_INTERRUPT_S,
) -> Executor:
    """Run ``python -m distillery run`` as a child so cancel can send SIGINT: the pipeline's
    paid-resource context managers then cancel provider jobs on the way out. Dry runs use the
    CLI's own dry-run defaults (no --seed / --max-rounds overrides)."""

    def run(job: Job, log: Callable[[str], None]) -> int:
        argv = [
            sys.executable, "-m", "distillery", "--root", root, "run", "--pack", "sql",
            "--scale", job.scale, "--run-id", job.run_id,
        ]  # fmt: skip
        env = dict(os.environ)
        env["PYTHONUNBUFFERED"] = "1"
        if job.dry_run:
            argv.append("--dry-run")
            env.pop("NEBIUS_API_KEY", None)
            env.pop("DISTILLERY_ADMIN_TOKEN_SUPPLIED", None)
        else:
            argv.append("--i-approve-spend")
            if admin_token:
                env["DISTILLERY_ADMIN_TOKEN_SUPPLIED"] = admin_token
            if job.budget_usd is not None:
                argv += ["--budget-usd", str(job.budget_usd)]
            if job.finetune_estimate_usd is not None:
                argv += ["--finetune-estimate-usd", str(job.finetune_estimate_usd)]
            if job.scale == "gated":  # pre-registered protocol: one round (DECISIONS 2026-10-03)
                argv += ["--max-rounds", "1"]
        return run_child(argv, env, job, log, kill_after_s)

    return run


class Worker:
    def __init__(self, executor: Executor, secrets: Sequence[str] = ()) -> None:
        self._executor = executor
        self._secrets = list(secrets)
        self._queue: queue.Queue[Job | None] = queue.Queue()
        self._jobs: dict[str, Job] = {}
        self._lock = threading.Lock()
        self._stopping = False
        self._thread = threading.Thread(target=self._loop, name="distillery-worker", daemon=True)
        self._thread.start()

    def stop(self, join_timeout_s: float = KILL_AFTER_INTERRUPT_S + 10.0) -> None:
        """Shut down: drop queued jobs, interrupt the running one (SIGINT, then SIGKILL after the
        executor's grace period) and join the worker thread."""
        with self._lock:
            self._stopping = True
            running = []
            for job in self._jobs.values():
                if job.state == "queued":
                    job.state = "finished"
                    job.error = "server shutting down"
                elif job.state == "running":
                    job.cancel_requested = True
                    running.append(job)
        self._queue.put(None)
        for job in running:
            if job.interrupt is not None:
                job.interrupt()
        self._thread.join(join_timeout_s)

    def jobs(self) -> list[Job]:
        with self._lock:
            return list(self._jobs.values())

    def get(self, run_id: str) -> Job | None:
        with self._lock:
            return self._jobs.get(run_id)

    def submit(self, job: Job, max_pending: int | None = None) -> None:
        with self._lock:
            if self._stopping:
                raise RunConflictError("server is shutting down")
            if max_pending is not None and (
                sum(1 for j in self._jobs.values() if j.state != "finished") >= max_pending
            ):
                raise QueueFullError(f"{max_pending} runs already queued or running")
            existing = self._jobs.get(job.run_id)
            if existing is not None and existing.state != "finished":
                raise RunConflictError(f"run {job.run_id} is already queued or running")
            if not job.dry_run and any(
                j.state != "finished" and not j.dry_run for j in self._jobs.values()
            ):
                raise RunConflictError("a live run is already active")
            self._jobs[job.run_id] = job
        self._queue.put(job)

    def cancel(self, run_id: str) -> str | None:
        """Returns the job state after the request, or None if the worker never saw the run."""
        with self._lock:
            job = self._jobs.get(run_id)
            if job is None:
                return None
            if job.state == "finished":
                return "finished"
            job.cancel_requested = True
            if job.state == "queued":
                job.state = "finished"
                job.error = "cancelled before start"
                return "finished"
        if job.interrupt is not None:
            job.interrupt()
        return "cancelling"

    def _log(self, job: Job, line: str) -> None:
        with self._lock:
            job.logs.append(redact(line, self._secrets))
            del job.logs[:-MAX_LOG_LINES]

    def _loop(self) -> None:
        while True:
            job = self._queue.get()
            if job is None:
                return
            with self._lock:
                if job.state != "queued":  # cancelled while waiting
                    continue
                job.state = "running"
                if self._stopping:
                    job.cancel_requested = True
            try:
                code = self._executor(job, functools.partial(self._log, job))
                if job.cancel_requested:
                    job.error = "cancelled"
                elif code != 0:
                    last = next((ln for ln in reversed(job.logs) if ln.strip()), "")
                    job.error = f"exit code {code}: {last}"[:500]
            except (OSError, RuntimeError, ValueError) as exc:
                job.error = redact(f"{type(exc).__name__}: {exc}", self._secrets)[:500]
            finally:
                with self._lock:
                    job.state = "finished"
