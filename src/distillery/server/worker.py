"""One background worker thread that runs pipelines, one at a time, under a supervisor.

Supervisor rules (plan A1): exit 75 (``EX_TEMPFAIL``, a retryable failure) or death by a signal
the server did not send restarts the same argv after a backoff (30 s, 60 s, 120 s, ...), at most
``MAX_RESTARTS`` times per run; the stage cache and fine-tune job adoption make that a resume.
Exit 0, 1 or 2, a user cancel, or a server shutdown are final for this process. A shutdown sends
SIGTERM (suspend: paid jobs are left running and are adopted on the next start); a user cancel
sends SIGINT (paid jobs are cancelled).
"""

from __future__ import annotations

import functools
import os
import queue
import re
import signal
import subprocess  # noqa: S404 - runs our own CLI with a fixed argv, never a shell
import sys
import threading
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

MAX_LOG_LINES = 2000
KILL_AFTER_INTERRUPT_S = 30.0
EXIT_RETRYABLE = 75  # distillery.errors.EXIT_RETRYABLE (not imported: keep the server light)
MAX_RESTARTS = 6
RESTART_BACKOFF_S = 30.0
_ANSI = re.compile(r"\x1b\[[0-9;?]*[ -/]*[@-~]")  # colour codes from child tracebacks


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
    pack: str = "sql"  # task pack (taskpacks.base registry name), passed to the CLI as --pack
    state: str = "queued"  # queued | running | finished
    error: str | None = None
    logs: list[str] = field(default_factory=list)
    cancel_requested: bool = False
    interrupt: Callable[[], None] | None = None  # SIGINT: user cancel (paid jobs cancelled)
    suspend: Callable[[], None] | None = None  # SIGTERM: shutdown (paid jobs kept, adopted later)
    suspend_requested: bool = False
    restarts: int = 0
    wake: threading.Event = field(default_factory=threading.Event, repr=False)


Executor = Callable[[Job, Callable[[str], None]], int]


def redact(text: str, secrets: Sequence[str]) -> str:
    for s in secrets:
        if s:
            text = text.replace(s, "[redacted]")
    return text


ChildHook = Callable[[str, Callable[[], None]], Callable[[], None]]


@dataclass(frozen=True)
class ChildInfo:
    """What the watchdog may know about the running child (plan A6): ``kill`` SIGKILLs its whole
    process group, which the supervisor sees as a negative exit code and restarts (a safe resume).
    It is deliberately NOT a SIGINT (that cancels paid jobs) or a SIGTERM (that is a suspend: no
    restart)."""

    pid: int
    started_at: float  # wall clock, comparable with the heartbeat file's timestamps
    alive: Callable[[], bool]
    kill: Callable[[], None]


WatchHook = Callable[[Job, ChildInfo], Callable[[], None]]  # returns a stop callback


def run_child(
    argv: Sequence[str],
    env: dict[str, str],
    job: Job,
    log: Callable[[str], None],
    kill_after_s: float,
    on_child: ChildHook | None = None,
    watch: WatchHook | None = None,
) -> int:
    """Run ``argv`` in its OWN process group; ``job.interrupt`` sends SIGINT to the whole group
    and SIGKILLs it after ``kill_after_s`` if it is still alive. ``on_child(run_id, kill)`` runs
    once the child exists (``kill`` SIGKILLs its group) and returns a stop callback (chaos
    supervisor, plan A5); ``watch(job, info)`` is the same idea with more to look at (the
    watchdog, plan A6)."""
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

    def _send(sig: int) -> None:
        if proc.poll() is None:
            _signal(sig)
            t = threading.Timer(kill_after_s, lambda: _signal(signal.SIGKILL))
            t.daemon = True
            timers.append(t)
            t.start()

    def interrupt() -> None:
        _send(signal.SIGINT)

    def suspend() -> None:
        _send(signal.SIGTERM)

    job.interrupt, job.suspend = interrupt, suspend
    if job.cancel_requested:
        interrupt()
    elif job.suspend_requested:
        suspend()
    stop_hook = on_child(job.run_id, lambda: _signal(signal.SIGKILL)) if on_child else None
    watch_stop = (
        watch(job, ChildInfo(proc.pid, time.time(), lambda: proc.poll() is None,
                             lambda: _signal(signal.SIGKILL)))
        if watch else None
    )  # fmt: skip
    try:
        for line in proc.stdout or ():
            log(line.rstrip("\n"))
        return proc.wait()
    finally:
        if stop_hook is not None:
            stop_hook()
        if watch_stop is not None:
            watch_stop()
        for t in timers:
            t.cancel()
        _signal(signal.SIGKILL)  # reap stragglers of the group (no-op if all exited)


def subprocess_executor(
    root: str,
    admin_token: str | None,
    secrets: Sequence[str],
    kill_after_s: float = KILL_AFTER_INTERRUPT_S,
    on_child: ChildHook | None = None,
    watch: WatchHook | None = None,
) -> Executor:
    """Run ``python -m distillery run`` as a child so cancel can send SIGINT: the pipeline's
    paid-resource context managers then cancel provider jobs on the way out. Dry runs use the
    CLI's own dry-run defaults (no --seed / --max-rounds overrides)."""

    def run(job: Job, log: Callable[[str], None]) -> int:
        argv = [
            sys.executable, "-m", "distillery", "--root", root, "run", "--pack", job.pack,
            "--scale", job.scale, "--run-id", job.run_id,
        ]  # fmt: skip
        env = dict(os.environ)
        env["PYTHONUNBUFFERED"] = "1"
        env["DISTILLERY_ATTEMPT"] = str(job.restarts + 1)  # shown in the child's heartbeat
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
        if watch is None:  # the pre-A6 call shape, kept for callers that stub ``run_child``
            return run_child(argv, env, job, log, kill_after_s, on_child)
        return run_child(argv, env, job, log, kill_after_s, on_child, watch)

    return run


AuditFn = Callable[[str, str, str, dict[str, object]], None]  # actor, action, run_id, detail
FinalFn = Callable[[Job, int | None], None]  # the run will not be restarted by this server


class Worker:
    def __init__(
        self,
        executor: Executor,
        secrets: Sequence[str] = (),
        *,
        max_restarts: int = MAX_RESTARTS,
        restart_backoff_s: float = RESTART_BACKOFF_S,
        on_audit: AuditFn | None = None,
        on_final: FinalFn | None = None,
    ) -> None:
        self._executor = executor
        self._max_restarts = max_restarts
        self._backoff_s = restart_backoff_s
        self._on_audit = on_audit
        self._on_final = on_final
        self._secrets = list(secrets)
        self._queue: queue.Queue[Job | None] = queue.Queue()
        self._jobs: dict[str, Job] = {}
        self._lock = threading.Lock()
        self._stopping = False
        self._thread = threading.Thread(target=self._loop, name="distillery-worker", daemon=True)
        self._thread.start()

    def stop(self, join_timeout_s: float = KILL_AFTER_INTERRUPT_S + 10.0) -> None:
        """Shut down: drop queued jobs, SUSPEND the running one (SIGTERM, then SIGKILL after the
        executor's grace period) and join the worker thread. SIGTERM, unlike a user cancel, leaves
        paid fine-tune jobs running, so a redeploy or restart does not pay for a second job: the
        next server start re-submits the run (``create_app`` reconcile) and it adopts them."""
        with self._lock:
            self._stopping = True
            running = []
            for job in self._jobs.values():
                if job.state == "queued":
                    job.state = "finished"
                    job.error = "server shutting down"
                elif job.state == "running":
                    job.suspend_requested = True
                    running.append(job)
        self._queue.put(None)
        for job in running:
            job.wake.set()  # a job waiting out a restart backoff stops waiting
            if job.suspend is not None:
                job.suspend()
        self._thread.join(join_timeout_s)

    @property
    def stopping(self) -> bool:
        return self._stopping

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
        job.wake.set()  # a job waiting out a restart backoff is not restarted
        if job.interrupt is not None:
            job.interrupt()
        return "cancelling"

    def _log(self, job: Job, line: str) -> None:
        with self._lock:
            job.logs.append(redact(_ANSI.sub("", line), self._secrets))
            del job.logs[:-MAX_LOG_LINES]

    def _audit(self, action: str, job: Job, **detail: object) -> None:
        if self._on_audit is not None:
            try:
                self._on_audit("supervisor", action, job.run_id, dict(detail))
            except Exception as exc:  # noqa: BLE001 - the audit log must never stop a run
                self._log(job, f"[supervisor] audit write failed: {type(exc).__name__}: {exc}")

    def _restartable(self, job: Job, code: int) -> bool:
        if job.cancel_requested or job.suspend_requested or self._stopping:
            return False  # a person or the server ended it: never restart
        return code == EXIT_RETRYABLE or code < 0  # retryable failure, or killed (SIGKILL / OOM)

    def _supervise(self, job: Job) -> int | None:
        """Run the job, restarting it while the rules allow. Returns the last exit code (None if
        the executor itself raised)."""
        code: int | None = None
        while True:
            try:
                code = self._executor(job, functools.partial(self._log, job))
            except (OSError, RuntimeError, ValueError) as exc:
                job.error = redact(f"{type(exc).__name__}: {exc}", self._secrets)[:500]
                return None
            if not self._restartable(job, code):
                return code
            if job.restarts >= self._max_restarts:
                self._audit("restarts_exhausted", job, exit_code=code, restarts=job.restarts)
                return code
            delay = self._backoff_s * 2**job.restarts
            job.restarts += 1
            self._log(
                job,
                f"[supervisor] exit {code}: restart {job.restarts}/{self._max_restarts} in "
                f"{delay:.0f} s (resume from the stage cache; paid jobs are adopted)",
            )
            self._audit("restart", job, exit_code=code, attempt=job.restarts, backoff_s=delay)
            if job.wake.wait(delay):  # cancel or shutdown during the backoff
                return code

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
                    job.suspend_requested = True
            code: int | None = None
            try:
                code = self._supervise(job)
                if job.cancel_requested:
                    job.error = "cancelled"
                elif job.suspend_requested:
                    job.error = "suspended: server shutting down; resumes on the next start"
                elif code is not None and code != 0:
                    last = next((ln for ln in reversed(job.logs) if ln.strip()), "")
                    job.error = f"exit code {code}: {last}"[:500]
            finally:
                with self._lock:
                    job.state = "finished"
                if not job.suspend_requested and self._on_final is not None:
                    try:
                        self._on_final(job, code)
                    except Exception as exc:  # noqa: BLE001
                        self._log(job, f"[supervisor] final hook failed: {exc}")
