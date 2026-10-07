"""Run heartbeat: proof of life AND proof of progress, written by the pipeline process (plan A6).

While ``distillery run`` drives a run, a daemon thread rewrites ``run_dir/heartbeat.json``
atomically every ``HEARTBEAT_INTERVAL_S`` (``DISTILLERY_HEARTBEAT_INTERVAL_S``, default 10; 0 turns
it off). The watchdog in the server process reads it and tells two failures apart that look the
same from outside:

* ``hung``: the file stops changing (the process is wedged, or the GIL is held by a stuck C call);
* ``stalled``: the file keeps changing (the thread is alive) but ``progress`` does not.

``progress`` is a monotonically increasing counter of REAL forward progress, bumped from cheap hooks
(``bump(kind)``): every LLM call recorded, every sandbox job / generation batch completed, every
fine-tune poll response (a status update or a handled poll error: the job is being watched), every
stage start and finish, every pipeline log line. The heartbeat thread itself NEVER bumps it, or a
wedged pipeline would look alive.

``bump`` is a no-op unless a heartbeat is active in this process, so library use and tests that
call the pipeline directly are unaffected. A heartbeat failure never fails the run: it is logged
once and the thread keeps trying (a full disk may clear).
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import threading
import time
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import Any

FILE = "heartbeat.json"
ENV_INTERVAL = "DISTILLERY_HEARTBEAT_INTERVAL_S"
ENV_ATTEMPT = "DISTILLERY_ATTEMPT"  # set by the supervisor: 1 + restarts so far
DEFAULT_INTERVAL_S = 10.0

log = logging.getLogger("distillery.heartbeat")


def rss_mb(pid: int | str = "self") -> float | None:
    """Resident set size of a process in MB (Linux ``/proc``), None where it cannot be read."""
    try:
        for line in Path(f"/proc/{pid}/status").read_text(encoding="utf-8").splitlines():
            if line.startswith("VmRSS:"):
                return round(int(line.split()[1]) / 1024.0, 1)
    except (OSError, ValueError, IndexError):
        pass
    return None


class _State:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.active = False
        self.progress = 0
        self.last_progress_at = 0.0
        self.last_kind = ""
        self.stage = ""
        self.clock: Callable[[], float] = time.time

    def reset(self, clock: Callable[[], float]) -> None:
        with self.lock:
            self.clock = clock
            self.progress, self.last_progress_at = 0, clock()
            self.last_kind, self.stage = "", ""
            self.active = True


_S = _State()


def bump(kind: str = "") -> None:
    """Record one unit of real forward progress. No-op when no heartbeat is active."""
    s = _S
    if not s.active:
        return
    with s.lock:
        s.progress += 1
        s.last_progress_at = s.clock()
        s.last_kind = kind


def set_stage(name: str) -> None:
    """The stage now running (shown in the heartbeat). A stage start is also progress."""
    s = _S
    if not s.active:
        return
    with s.lock:
        s.stage = name
    bump("stage_start")


def snapshot() -> dict[str, Any]:
    s = _S
    with s.lock:
        return {
            "stage": s.stage, "progress": s.progress, "last_progress_at": s.last_progress_at,
            "last_kind": s.last_kind,
        }  # fmt: skip


def _write(path: Path, payload: dict[str, Any]) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    os.replace(tmp, path)  # atomic: the watchdog never reads a half-written file


class Heartbeat:
    def __init__(
        self, run_dir: Path, interval_s: float, attempt: int,
        clock: Callable[[], float] = time.time,
    ) -> None:  # fmt: skip
        self._path, self._interval, self._attempt, self._clock = (
            run_dir / FILE, interval_s, attempt, clock,
        )  # fmt: skip
        self._started_at = clock()
        self._stop = threading.Event()
        self._warned = False
        self._thread = threading.Thread(target=self._loop, name="distillery-heartbeat", daemon=True)

    def payload(self) -> dict[str, Any]:
        snap = snapshot()
        return {
            "pid": os.getpid(), "started_at": self._started_at, "at": self._clock(),
            "stage": snap["stage"], "progress": snap["progress"],
            "last_progress_at": snap["last_progress_at"], "last_kind": snap["last_kind"],
            "rss_mb": rss_mb(), "attempt": self._attempt,
        }  # fmt: skip

    def beat(self) -> None:
        try:
            _write(self._path, self.payload())
        except Exception as exc:  # noqa: BLE001 - a heartbeat failure must never fail the run
            if not self._warned:
                self._warned = True
                log.warning(
                    "heartbeat write failed (%s: %s); the run goes on", type(exc).__name__, exc
                )

    def _loop(self) -> None:
        while not self._stop.wait(self._interval):
            self.beat()

    def start(self) -> None:
        self.beat()
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread.is_alive():
            self._thread.join(timeout=2.0)
        self.beat()  # last state on disk: where the run ended


@contextmanager
def running(
    run_dir: Path, env: Mapping[str, str] | None = None, clock: Callable[[], float] = time.time
) -> Iterator[Heartbeat | None]:
    """Keep ``heartbeat.json`` fresh for the duration of the block (None when disabled)."""
    e = os.environ if env is None else env
    try:
        interval = float(e.get(ENV_INTERVAL) or DEFAULT_INTERVAL_S)
        attempt = int(e.get(ENV_ATTEMPT) or 1)
    except ValueError:
        interval, attempt = DEFAULT_INTERVAL_S, 1
    if interval <= 0:
        yield None
        return
    hb: Heartbeat | None = None
    try:
        run_dir.mkdir(parents=True, exist_ok=True)
        _S.reset(clock)
        hb = Heartbeat(run_dir, interval, attempt, clock)
        hb.start()
    except Exception as exc:  # noqa: BLE001
        log.warning("heartbeat could not start (%s: %s); the run goes on", type(exc).__name__, exc)
    try:
        yield hb
    finally:
        _S.active = False
        if hb is not None:
            with contextlib.suppress(Exception):
                hb.stop()


__all__ = ["FILE", "Heartbeat", "bump", "rss_mb", "running", "set_stage", "snapshot"]
