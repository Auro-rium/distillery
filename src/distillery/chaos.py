"""Fault injection for the unattended-run proof (plan A5).

``DISTILLERY_CHAOS=<json plan>`` is honoured ONLY for run ids that start with ``chaos-`` (a dry
rehearsal needs the pipeline's mandatory ``dry-`` prefix, so ``dry-chaos-`` counts too). Any other
run id ignores the variable completely: nothing is wrapped, nothing is written. Plan keys (all
optional; unknown keys are refused so a typo cannot silently disable a fault)::

    {"llm_503":       {"after_calls": 5, "n": 3},      # n HTTP 503s from the LLM transport
     "net_window":    {"after_calls": 20, "seconds": 60},  # every LLM call fails for S seconds
     "kill":          ["teacher_data", "finetune_r1"], # supervisor SIGKILLs the child, once each
     "sandbox_fail":  {"stage": "dev_eval", "jobs": 2},  # k sandbox jobs return exit -1
     "hold_s": 0.0,                  # dry rehearsal only: the (fake) stage lingers this long so
     "poll_s": 0.2}                  # the supervisor can catch it; poll_s = supervisor probe period

Every injection is written to the audit log with actor ``chaos`` BEFORE it takes effect. Each fault
is injected at most the planned number of times across restarts of the child: the running totals
live in ``chaos_state.json`` and the kills in ``chaos_kill_<stage>.done`` marker files, both in the
run dir. The kill is done by the supervisor (the server process), never by the child itself, and
a fine-tune kill waits until a job exists, so the resume has something to adopt.
"""

from __future__ import annotations

import contextlib
import json
import os
import subprocess  # noqa: S404 - dry rehearsal only: runs the SQL runner script on a temp DB
import sys
import tempfile
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import httpx2
import openai

from distillery.orchestrator import Deps
from distillery.sandbox import FakeExecution, FakeSandbox, Job, RunResult, Sandbox
from distillery.sandbox_executor import DB_PATH, SCRIPT_PATH, SandboxExecutor
from distillery.store import atomic_write_bytes
from distillery.taskpacks.sql.executor import LocalExecutor

ENV = "DISTILLERY_CHAOS"
PREFIXES = ("chaos-", "dry-chaos-")
STATE_FILE = "chaos_state.json"
_KEYS = frozenset({"llm_503", "net_window", "kill", "sandbox_fail", "hold_s", "poll_s"})

AuditFn = Callable[[str, dict[str, Any]], None]  # action, detail (actor is always "chaos")


def is_chaos_run(run_id: str) -> bool:
    return run_id.startswith(PREFIXES)


@dataclass(frozen=True)
class ChaosPlan:
    llm_503_after: int = 0
    llm_503_n: int = 0
    net_after: int = 0
    net_seconds: float = 0.0
    kill_stages: tuple[str, ...] = ()
    sandbox_stage: str = ""
    sandbox_jobs: int = 0
    hold_s: float = 0.0
    poll_s: float = 0.2

    @classmethod
    def parse(cls, text: str) -> ChaosPlan:
        raw = json.loads(text)
        if not isinstance(raw, dict) or not set(raw) <= _KEYS:
            raise ValueError(f"{ENV}: a JSON object with keys from {sorted(_KEYS)} is required")
        a, w, s = raw.get("llm_503", {}), raw.get("net_window", {}), raw.get("sandbox_fail", {})
        return cls(
            llm_503_after=int(a.get("after_calls", 0)), llm_503_n=int(a.get("n", 0)),
            net_after=int(w.get("after_calls", 0)), net_seconds=float(w.get("seconds", 0.0)),
            kill_stages=tuple(str(x) for x in raw.get("kill", ())),
            sandbox_stage=str(s.get("stage", "")), sandbox_jobs=int(s.get("jobs", 0)),
            hold_s=float(raw.get("hold_s", 0.0)), poll_s=float(raw.get("poll_s", 0.2)),
        )  # fmt: skip


def plan_for(run_id: str, env: Mapping[str, str] | None = None) -> ChaosPlan | None:
    """The plan for this run, or None: no variable, or not a chaos run id (always ignored)."""
    text = (os.environ if env is None else env).get(ENV)
    if not text or not is_chaos_run(run_id):
        return None
    return ChaosPlan.parse(text)


def kill_marker(run_dir: Path, stage: str) -> Path:
    return run_dir / f"chaos_kill_{stage}.done"


class ChaosState:
    """Injection totals that survive a restart of the child (once-only across attempts)."""

    def __init__(self, run_dir: Path) -> None:
        self._path = run_dir / STATE_FILE
        self._lock = threading.Lock()

    def _read(self) -> dict[str, Any]:
        try:
            data = json.loads(self._path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}
        return data if isinstance(data, dict) else {}

    def get(self, key: str, default: Any = 0) -> Any:
        with self._lock:
            return self._read().get(key, default)

    def set(self, key: str, value: Any) -> None:
        with self._lock:
            data = self._read()
            data[key] = value
            atomic_write_bytes(self._path, json.dumps(data, sort_keys=True).encode())

    def bump(self, key: str) -> int:
        with self._lock:
            data = self._read()
            data[key] = int(data.get(key, 0)) + 1
            atomic_write_bytes(self._path, json.dumps(data, sort_keys=True).encode())
            return int(data[key])


_REQ = httpx2.Request("POST", "https://chaos.invalid/v1/chat/completions")


class ChaosTransport:
    """Wraps an async chat transport: (a) ``n`` HTTP 503s after ``after_calls`` calls of this
    process, (b) once, a window of ``seconds`` in which every call is a connection error."""

    def __init__(
        self, inner: Any, plan: ChaosPlan, state: ChaosState, audit: AuditFn,
        clock: Callable[[], float] = time.time,
    ) -> None:  # fmt: skip
        self._inner, self._plan, self._state, self._audit, self._clock = (
            inner, plan, state, audit, clock,
        )  # fmt: skip
        self._calls = 0
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)

    async def _create(self, **kw: Any) -> Any:
        self._calls += 1
        p, st = self._plan, self._state
        if p.net_seconds > 0 and self._calls > p.net_after:
            started = st.get("net_window_started_at", None)
            if started is None:
                started = self._clock()
                st.set("net_window_started_at", started)
                self._audit(
                    "inject_net_window", {"seconds": p.net_seconds, "model": kw.get("model")}
                )
            if self._clock() < float(started) + p.net_seconds:
                raise openai.APIConnectionError(request=_REQ)
        if self._calls > p.llm_503_after and int(st.get("llm_503_done", 0)) < p.llm_503_n:
            i = st.bump("llm_503_done")
            self._audit("inject_llm_503", {"n": i, "of": p.llm_503_n, "model": kw.get("model")})
            resp = httpx2.Response(503, request=_REQ, headers={"retry-after": "0"})
            raise openai.InternalServerError("chaos: injected 503", response=resp, body=None)
        return await self._inner.chat.completions.create(**kw)


class ChaosSandbox:
    """Wraps a ``Sandbox``: while the pipeline is in a stage starting with ``plan.sandbox_stage``,
    the next ``plan.sandbox_jobs`` jobs (counted across restarts) return exit -1 without running."""

    def __init__(
        self, inner: Sandbox, plan: ChaosPlan, state: ChaosState, audit: AuditFn,
        stage: Callable[[], str],
    ) -> None:  # fmt: skip
        self._inner, self._plan, self._state, self._audit, self._stage = (
            inner, plan, state, audit, stage,
        )  # fmt: skip

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)

    def _take(self) -> bool:
        p = self._plan
        if not p.sandbox_stage or not self._stage().startswith(p.sandbox_stage):
            return False
        if int(self._state.get("sandbox_failed", 0)) >= p.sandbox_jobs:
            return False
        i = self._state.bump("sandbox_failed")
        self._audit(
            "inject_sandbox_exit_-1", {"n": i, "of": p.sandbox_jobs, "stage": self._stage()}
        )
        return True

    @staticmethod
    def _lost() -> RunResult:
        return RunResult("", "chaos: injected sandbox failure", -1, None)

    async def run(self, image_ref: str, **kw: Any) -> RunResult:
        if self._take():
            return self._lost()
        return await self._inner.run(image_ref, **kw)

    async def run_batch(
        self, image_ref: str, jobs: Sequence[Job], concurrency: int = 10
    ) -> list[RunResult]:
        lost = [self._take() for _ in jobs]
        live = [j for j, x in zip(jobs, lost, strict=True) if not x]
        got = iter(await self._inner.run_batch(image_ref, live, concurrency) if live else [])
        return [self._lost() if x else next(got) for x in lost]


class ChaosFineTune:
    """Wraps a ``FineTuner``. With ``hold_s`` > 0 (dry rehearsal: the fake job finishes
    instantly), ``poll`` lingers while a fine-tune kill for this stage is still pending."""

    def __init__(self, inner: Any, plan: ChaosPlan, run_dir: Path, stage: Callable[[], str]):
        self._inner, self._plan, self._run_dir, self._stage = inner, plan, run_dir, stage

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)

    def poll(self, job_id: str, **kw: Any) -> Any:
        _hold(self._plan, self._run_dir, self._stage())
        return self._inner.poll(job_id, **kw)


def _hold(plan: ChaosPlan, run_dir: Path, stage: str) -> None:
    """Dry rehearsal only: linger (up to ``hold_s``) while the supervisor still owes this stage a
    kill; once its marker appears, wait a moment longer for the SIGKILL itself."""
    marker = kill_marker(run_dir, stage)
    if plan.hold_s <= 0 or stage not in plan.kill_stages or marker.exists():
        return
    end = time.monotonic() + plan.hold_s
    while time.monotonic() < end:
        if marker.exists():
            time.sleep(5.0)  # killed within milliseconds of the marker
            return
        time.sleep(0.05)


def _local_runner_handler(job: Job, fs: dict[str, bytes]) -> FakeExecution:
    """FakeSandbox handler for the SQL runner script: runs it for real on a temp copy of the DB."""
    if tuple(job.args) != (SCRIPT_PATH,):  # the image-building branch job: nothing to run
        return FakeExecution("", "", 0)
    with tempfile.TemporaryDirectory() as tmp:
        script, db = Path(tmp) / "runner.py", Path(tmp) / "db.sqlite"
        script.write_bytes(fs[SCRIPT_PATH])
        db.write_bytes(fs[DB_PATH])
        payload = json.loads(str(job.stdin))
        payload["db"] = str(db)
        proc = subprocess.run(  # noqa: S603
            [sys.executable, str(script)], input=json.dumps(payload), capture_output=True,
            text=True, timeout=120, check=False,
        )  # fmt: skip
    return FakeExecution(proc.stdout, proc.stderr, proc.returncode)


def instrument(
    deps: Deps, plan: ChaosPlan, run_dir: Path, audit: AuditFn, *, dry_run: bool
) -> Deps:
    """Return ``deps`` with the plan's injectors wired in (the pipeline sees the same interfaces).
    In a dry rehearsal the in-process executor is swapped for the real ``SandboxExecutor`` over
    a fake sandbox that runs the real runner script, so a sandbox fault goes through the same
    retry code as a live one."""
    state = ChaosState(run_dir)
    current = [""]
    stage = lambda: current[0]  # noqa: E731
    inner_on_stage = deps.on_stage

    def on_stage(name: str) -> None:
        current[0] = name
        if inner_on_stage is not None:
            inner_on_stage(name)
        if not name.startswith("finetune"):  # fine-tune holds in poll (after the job exists)
            _hold(plan, run_dir, name)

    deps.on_stage = on_stage
    deps.transport = ChaosTransport(deps.transport, plan, state, audit)
    deps.finetune = ChaosFineTune(deps.finetune, plan, run_dir, stage)
    if plan.sandbox_jobs > 0:
        if dry_run and isinstance(deps.executor, LocalExecutor):
            fake = FakeSandbox(_local_runner_handler)
            sandbox: Sandbox = ChaosSandbox(fake, plan, state, audit, stage)
            deps.sandbox = sandbox
            deps.executor = SandboxExecutor(
                sandbox, deps.sandbox_image, bridge=deps.bridge, infra_backoff_s=0.05
            )
        elif isinstance(deps.executor, SandboxExecutor) and deps.sandbox is not None:
            deps.sandbox = ChaosSandbox(deps.sandbox, plan, state, audit, stage)
            deps.executor.wrap_sandbox(deps.sandbox)
    return deps


# ------------------------------------------------------------------ the supervisor's kill


@dataclass
class ChaosSupervisor:
    """Server-side half of the plan: watches the run's progress (``probe``: running stage names
    and whether a fine-tune job has been recorded) and SIGKILLs the child once per planned stage.
    Plugged into ``subprocess_executor(on_child=...)``; inert for any non-chaos run id."""

    plan_for_run: Callable[[str], ChaosPlan | None]
    run_dir: Callable[[str], Path]
    probe: Callable[[str], tuple[list[str], bool]]  # run_id -> (running stages, job recorded)
    audit: Callable[[str, str, str, dict[str, Any]], None]  # actor, action, run_id, detail
    kills: list[str] = field(default_factory=list)

    def __call__(self, run_id: str, kill: Callable[[], None]) -> Callable[[], None]:
        plan = self.plan_for_run(run_id)
        if plan is None or not plan.kill_stages:
            return lambda: None
        stop = threading.Event()
        run_dir = self.run_dir(run_id)

        def watch() -> None:
            while not stop.wait(plan.poll_s):
                try:
                    running, job_recorded = self.probe(run_id)
                except Exception:  # noqa: BLE001, S112 - a probe hiccup must not stop the watcher
                    continue
                for st in plan.kill_stages:
                    marker = kill_marker(run_dir, st)
                    if marker.exists() or st not in running:
                        continue
                    if st.startswith("finetune") and not job_recorded:
                        continue
                    marker.write_text(str(time.time()))
                    with contextlib.suppress(Exception):
                        self.audit(
                            "chaos", "kill_child", run_id, {"stage": st, "signal": "SIGKILL"}
                        )
                    self.kills.append(st)
                    kill()
                    return  # the supervisor restarts the child; a new watcher covers the next stage

        t = threading.Thread(target=watch, name=f"chaos-watch-{run_id}", daemon=True)
        t.start()
        return stop.set


__all__ = [
    "ENV", "ChaosPlan", "ChaosSandbox", "ChaosState", "ChaosSupervisor", "ChaosTransport",
    "instrument", "is_chaos_run", "kill_marker", "plan_for",
]  # fmt: skip
