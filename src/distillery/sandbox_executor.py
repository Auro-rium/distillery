"""Executor backed by the Sandbox protocol, plus a small sync<->async bridge.

Design: the SQLite file and a tiny runner script are baked once into a sandbox image via
``Sandbox.branch`` (documented pattern: non-disposable run with ``files``). SQL is then executed
in batches: each disposable job gets a JSON list of statements on stdin and prints a JSON list of
outcomes. The runner script is NOT a fork of the verifier's logic: it is built by embedding the
source of ``taskpacks/sql/sqltext.py`` and ``taskpacks/sql/runner.py`` verbatim (imports of the
``distillery`` package stripped) and calling ``run_select``, so semantics are identical.

# UNVERIFIED (real Contree behaviour): the image must provide ``python3`` (>= 3.10, stdlib
# sqlite3); the base image choice is a Phase 0 item. Also unverified: that ``branch`` with
# ``files`` places them at the given absolute destinations, that stdin is delivered as given,
# and that stdout is not truncated for large result sets.
"""

from __future__ import annotations

import asyncio
import json
import re
import threading
from collections.abc import Coroutine, Sequence
from concurrent.futures import Future
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from distillery.sandbox import MAX_CONCURRENCY, Job, RunResult, Sandbox
from distillery.taskpacks.sql import runner as _runner_mod
from distillery.taskpacks.sql import sqltext as _sqltext_mod
from distillery.taskpacks.sql.runner import (
    DEFAULT_MAX_ROWS,
    DEFAULT_TIMEOUT_S,
    ErrorKind,
    ExecOutcome,
    Scalar,
)

WORK_DIR = "/work"
DB_PATH = f"{WORK_DIR}/db.sqlite"
SCRIPT_PATH = f"{WORK_DIR}/runner.py"
DEFAULT_STATEMENTS_PER_JOB = 25

_IMPORT_RE = re.compile(
    r"^from distillery\.taskpacks\.sql\.\w+ import (?:\([^)]*\)|[^\n]*)\n", re.MULTILINE
)
_FUTURE_RE = re.compile(r"^from __future__ import annotations\n", re.MULTILINE)

_MAIN = """

def _encode(v):
    if isinstance(v, bytes):
        return {"__bytes__": v.hex()}
    return v


def _main():
    import json
    import sys

    req = json.load(sys.stdin)
    out = []
    for sql in req["sqls"]:
        o = run_select(req["db"], sql, timeout_s=req["timeout_s"], max_rows=req["max_rows"])
        out.append(
            {
                "ok": o.ok,
                "columns": list(o.columns),
                "rows": [[_encode(c) for c in r] for r in o.rows],
                "error_kind": o.error_kind,
                "error": o.error,
                "elapsed_s": o.elapsed_s,
            }
        )
    json.dump(out, sys.stdout)


if __name__ == "__main__":
    _main()
"""


def build_runner_script() -> str:
    """Single-file runner: embedded sqltext + runner sources + a JSON stdin/stdout main."""
    parts: list[str] = []
    for mod in (_sqltext_mod, _runner_mod):
        src = Path(str(mod.__file__)).read_text(encoding="utf-8")
        src = _FUTURE_RE.sub("", src)
        src = _IMPORT_RE.sub("", src)
        parts.append(src)
    return "from __future__ import annotations\n" + "\n".join(parts) + _MAIN


class AsyncBridge:
    """Runs coroutines on one persistent background event loop, callable from sync code.

    A persistent loop (rather than ``asyncio.run`` per call) keeps async HTTP clients bound to
    one loop for the whole run, and works whether or not the caller already has a running loop.
    Do not call ``run`` from a coroutine executing on this bridge's own loop (would deadlock).
    """

    def __init__(self) -> None:
        self._loop = asyncio.new_event_loop()
        self._ready = threading.Event()
        self._thread = threading.Thread(target=self._serve, name="distillery-async", daemon=True)
        self._thread.start()
        self._ready.wait()
        self._closed = False

    def _serve(self) -> None:
        asyncio.set_event_loop(self._loop)
        self._loop.call_soon(self._ready.set)
        self._loop.run_forever()

    def run[T](self, coro: Coroutine[Any, Any, T]) -> T:
        if self._closed:
            coro.close()
            raise RuntimeError("AsyncBridge is closed")
        fut: Future[T] = asyncio.run_coroutine_threadsafe(coro, self._loop)
        try:
            return fut.result()
        except BaseException:
            # KeyboardInterrupt etc.: stop the in-flight coroutine before propagating.
            fut.cancel()
            raise

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._loop.call_soon_threadsafe(self._loop.stop)
        self._thread.join(timeout=5)
        if not self._thread.is_alive():
            self._loop.close()

    def __enter__(self) -> AsyncBridge:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()


def _decode_cell(v: Any) -> Scalar:
    if isinstance(v, dict) and "__bytes__" in v:
        return bytes.fromhex(str(v["__bytes__"]))
    if v is None or isinstance(v, int | float | str):
        return v
    raise ValueError(f"unexpected cell type {type(v).__name__}")


_KINDS: frozenset[str] = frozenset(
    {"syntax", "runtime", "timeout", "forbidden", "row_cap", "multi_statement"}
)


def parse_outcomes(stdout: str, expected: int) -> list[ExecOutcome]:
    """Parse the runner's JSON stdout; raises ValueError on any malformation."""
    data = json.loads(stdout)
    if not isinstance(data, list) or len(data) != expected:
        raise ValueError(f"expected a list of {expected} outcomes")
    out: list[ExecOutcome] = []
    for d in data:
        kind = d.get("error_kind")
        if kind is not None and kind not in _KINDS:
            raise ValueError(f"unknown error_kind {kind!r}")
        error_kind: ErrorKind | None = kind
        out.append(
            ExecOutcome(
                ok=bool(d["ok"]),
                columns=tuple(str(c) for c in d["columns"]),
                rows=tuple(tuple(_decode_cell(c) for c in r) for r in d["rows"]),
                error_kind=error_kind,
                error=str(d.get("error", "")),
                elapsed_s=float(d.get("elapsed_s", 0.0)),
            )
        )
    return out


@dataclass
class ExecutorStats:
    jobs: int = 0
    statements: int = 0
    infra_failures: int = 0  # jobs whose sandbox run/parse failed (statements reported as errors)
    images_built: int = 0


class SandboxExecutor:
    """``Executor`` over a ``Sandbox``. ``db_ref`` = registered key, else an existing file path."""

    def __init__(
        self,
        sandbox: Sandbox,
        base_image: str,
        *,
        bridge: AsyncBridge | None = None,
        python: str = "python3",
        timeout_s: float = DEFAULT_TIMEOUT_S,
        max_rows: int = DEFAULT_MAX_ROWS,
        statements_per_job: int = DEFAULT_STATEMENTS_PER_JOB,
        concurrency: int = MAX_CONCURRENCY,
    ) -> None:
        if not 1 <= concurrency <= MAX_CONCURRENCY:
            raise ValueError(f"concurrency must be in 1..{MAX_CONCURRENCY}")
        if statements_per_job < 1:
            raise ValueError("statements_per_job must be >= 1")
        self._sandbox = sandbox
        self._base_image = base_image
        self._own_bridge = bridge is None
        self._bridge = bridge or AsyncBridge()
        self._python = python
        self._timeout_s = timeout_s
        self._max_rows = max_rows
        self._per_job = statements_per_job
        self._concurrency = concurrency
        self._registered: dict[str, bytes | Path] = {}
        self._images: dict[str, str] = {}
        self._lock = threading.Lock()
        self.stats = ExecutorStats()

    def register(self, db_ref: str, data: bytes | Path) -> None:
        self._registered[db_ref] = data

    def close(self) -> None:
        if self._own_bridge:
            self._bridge.close()

    def _db_bytes(self, db_ref: str) -> bytes:
        src = self._registered.get(db_ref)
        if isinstance(src, bytes):
            return src
        return Path(src if src is not None else db_ref).read_bytes()

    async def _image_for(self, db_ref: str) -> str:
        with self._lock:
            cached = self._images.get(db_ref)
        if cached is not None:
            return cached
        image = await self._sandbox.branch(
            self._base_image,
            "true",
            files={DB_PATH: self._db_bytes(db_ref), SCRIPT_PATH: build_runner_script().encode()},
        )
        with self._lock:
            self._images[db_ref] = image
            self.stats.images_built += 1
        return image

    def _job(self, chunk: Sequence[str]) -> Job:
        payload = {
            "db": DB_PATH,
            "sqls": list(chunk),
            "timeout_s": self._timeout_s,
            "max_rows": self._max_rows,
        }
        return Job(
            command=self._python,
            args=(SCRIPT_PATH,),
            stdin=json.dumps(payload),
            # per-statement timeout is enforced in the runner; this is the whole-job backstop
            timeout=self._timeout_s * len(chunk) + 30.0,
        )

    @staticmethod
    def _infra_failure(n: int, why: str) -> list[ExecOutcome]:
        msg = f"sandbox infrastructure failure: {why}"
        return [ExecOutcome(False, error_kind="runtime", error=msg) for _ in range(n)]

    def _parse(self, res: RunResult, n: int) -> list[ExecOutcome]:
        if not res.ok:
            why = res.error or f"exit {res.exit_code}: {res.stderr[:200]!r}"
            self.stats.infra_failures += 1
            return self._infra_failure(n, why)
        try:
            return parse_outcomes(res.stdout, n)
        except (ValueError, KeyError, TypeError) as exc:
            self.stats.infra_failures += 1
            return self._infra_failure(n, f"unparseable runner output ({exc})")

    async def _run_async(self, db_ref: str, sqls: Sequence[str]) -> list[ExecOutcome]:
        image = await self._image_for(db_ref)
        chunks = [sqls[i : i + self._per_job] for i in range(0, len(sqls), self._per_job)]
        results = await self._sandbox.run_batch(
            image, [self._job(c) for c in chunks], concurrency=self._concurrency
        )
        out: list[ExecOutcome] = []
        for chunk, res in zip(chunks, results, strict=True):
            out.extend(self._parse(res, len(chunk)))
        self.stats.jobs += len(chunks)
        self.stats.statements += len(sqls)
        return out

    def run_batch(self, db_ref: str, sqls: Sequence[str]) -> list[ExecOutcome]:
        if not sqls:
            return []
        return self._bridge.run(self._run_async(db_ref, list(sqls)))
