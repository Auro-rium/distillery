# ruff: noqa: S101, S603
from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

from distillery.sandbox import FakeExecution, FakeSandbox, Job
from distillery.sandbox_executor import (
    DB_PATH,
    SCRIPT_PATH,
    AsyncBridge,
    SandboxExecutor,
    build_runner_script,
    parse_outcomes,
)
from distillery.taskpacks.sql import schema as s
from distillery.taskpacks.sql.executor import Executor, LocalExecutor
from distillery.taskpacks.sql.runner import run_select


def local_python_handler(job: Job, fs: dict[str, bytes]) -> FakeExecution:
    """Actually run the uploaded script with this interpreter (genuine round trip)."""
    if job.command is None:  # the image-building branch step (shell "true")
        return FakeExecution()
    with tempfile.TemporaryDirectory() as tmp:
        script = Path(tmp) / "runner.py"
        db = Path(tmp) / "db.sqlite"
        script.write_bytes(fs[SCRIPT_PATH])
        db.write_bytes(fs[DB_PATH])
        stdin = job.stdin if isinstance(job.stdin, str) else (job.stdin or b"").decode()
        stdin = stdin.replace(DB_PATH, str(db))
        proc = subprocess.run(  # noqa: S603
            [sys.executable, str(script)], input=stdin, capture_output=True, text=True, check=False
        )
    return FakeExecution(proc.stdout, proc.stderr, proc.returncode)


@pytest.fixture(scope="module")
def db(tmp_path_factory: pytest.TempPathFactory) -> Path:
    p = tmp_path_factory.mktemp("db") / "db.sqlite"
    s.write_database(0, p)
    return p


SQLS = [
    "SELECT COUNT(*) FROM accounts",
    "SELECT name FROM accounts ORDER BY id LIMIT 3",
    "SELECT * FROM nonexistent_table",
    "DELETE FROM accounts",
    "SELECT 1; SELECT 2",
    "SELECT 1.5, NULL, 'x', x'00ff'",
    "",
]


def test_protocol_conformance() -> None:
    ex: Executor = SandboxExecutor(FakeSandbox(local_python_handler), "img")
    assert hasattr(ex, "run_batch")
    ex.close()  # type: ignore[attr-defined]


def test_round_trip_matches_local_semantics(db: Path) -> None:
    ex = SandboxExecutor(FakeSandbox(local_python_handler), "img", statements_per_job=3)
    try:
        got = ex.run_batch(str(db), SQLS)
        want = LocalExecutor().run_batch(str(db), SQLS)
        assert len(got) == len(SQLS)
        for g, w in zip(got, want, strict=True):
            assert (g.ok, g.columns, g.rows, g.error_kind) == (
                w.ok,
                w.columns,
                w.rows,
                w.error_kind,
            )
        assert got[5].rows == ((1.5, None, "x", b"\x00\xff"),)
        assert got[3].error_kind == "forbidden"
        assert ex.stats.jobs == 3  # ceil(7/3) chunks
        assert ex.stats.images_built == 1
        assert ex.stats.infra_failures == 0
        ex.run_batch(str(db), ["SELECT 1"])
        assert ex.stats.images_built == 1  # image cached per db_ref
    finally:
        ex.close()


def test_empty_batch_and_registered_bytes(db: Path) -> None:
    ex = SandboxExecutor(FakeSandbox(local_python_handler), "img")
    try:
        assert ex.run_batch("x", []) == []
        ex.register("mem", db.read_bytes())
        assert ex.run_batch("mem", ["SELECT COUNT(*) FROM plans"])[0].ok
    finally:
        ex.close()


def test_runner_script_reuses_verifier_source() -> None:
    script = build_runner_script()
    assert "def run_select(" in script and "def split_statements(" in script
    assert "from distillery" not in script
    assert script.count("from __future__ import annotations") == 1
    compile(script, "runner.py", "exec")


def test_infra_failure_is_reported_not_raised(db: Path) -> None:
    def broken(job: Job, fs: dict[str, bytes]) -> FakeExecution:
        return (
            FakeExecution() if job.command is None else FakeExecution("", "python3: not found", 127)
        )

    def garbage(job: Job, fs: dict[str, bytes]) -> FakeExecution:
        return FakeExecution("not json", "", 0) if job.command else FakeExecution()

    for handler in (broken, garbage):
        ex = SandboxExecutor(FakeSandbox(handler), "img")
        try:
            out = ex.run_batch(str(db), ["SELECT 1", "SELECT 2"])
            assert [o.ok for o in out] == [False, False]
            assert all("sandbox infrastructure failure" in o.error for o in out)
            assert ex.stats.infra_failures == 1
        finally:
            ex.close()


def test_parse_outcomes_rejects_wrong_length() -> None:
    with pytest.raises(ValueError):
        parse_outcomes("[]", 1)


def test_concurrency_bounds() -> None:
    with pytest.raises(ValueError):
        SandboxExecutor(FakeSandbox(), "img", concurrency=41)


def test_async_bridge_runs_from_sync_and_from_running_loop() -> None:
    import asyncio

    async def add(a: int, b: int) -> int:
        await asyncio.sleep(0)
        return a + b

    with AsyncBridge() as br:
        assert br.run(add(1, 2)) == 3

        async def outer() -> int:
            return br.run(add(2, 3))  # caller has its own running loop

        assert asyncio.run(outer()) == 5
    with pytest.raises(RuntimeError):
        br.run(add(1, 1))


def test_matches_run_select_directly(db: Path) -> None:
    ex = SandboxExecutor(FakeSandbox(local_python_handler), "img")
    try:
        sql = "SELECT industry, COUNT(*) FROM accounts GROUP BY industry"
        assert ex.run_batch(str(db), [sql])[0].rows == run_select(db, sql).rows
    finally:
        ex.close()
