# ruff: noqa: S101, S604
from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any

import pytest

from distillery.sandbox import (
    ContreeSandbox,
    FakeExecution,
    FakeSandbox,
    Job,
    Sandbox,
    SandboxError,
    SandboxTimeoutError,
)


def run(coro: Any) -> Any:
    return asyncio.run(coro)


def test_protocol_conformance() -> None:
    sb: Sandbox = FakeSandbox()
    assert sb.lineage() == []


def test_run_output_and_exit_code() -> None:
    sb = FakeSandbox()
    r = run(sb.run("alpine:latest", shell="echo hi"))
    assert (r.stdout, r.exit_code, r.image_uuid) == ("hi\n", 0, None)
    r = run(sb.run("alpine:latest", shell="echo x; exit 3"))
    assert r.exit_code == 3 and not r.ok


def test_run_requires_shell_xor_command() -> None:
    with pytest.raises(ValueError):
        Job()
    with pytest.raises(ValueError):
        Job(shell="a", command="b")


def test_files_are_visible_to_command() -> None:
    sb = FakeSandbox()
    r = run(sb.run("img", shell="cat /data.txt", files={"data.txt": b"hello\n"}))
    assert r.stdout == "hello\n"


def test_branch_lineage_and_state_inheritance() -> None:
    async def go() -> tuple[FakeSandbox, str, list[str]]:
        sb = FakeSandbox()
        child = await sb.branch("base", "echo one > /f")
        outs = []
        gcs = []
        for letter in "ABC":
            gc = await sb.branch(child, f"echo {letter} >> /f")
            gcs.append(gc)
            r = await sb.run(gc, shell="cat /f")
            outs.append(r.stdout)
        assert len(set(gcs)) == 3
        return sb, child, outs

    sb, child, outs = run(go())
    assert outs == ["one\nA\n", "one\nB\n", "one\nC\n"]  # all branch from the same state
    lin = sb.lineage()
    by_uuid = {u: (p, label) for u, p, label in lin}
    assert by_uuid["base"] == (None, "root:base")
    assert by_uuid[child] == ("base", "echo one > /f")
    children = [(p, label) for u, (p, label) in by_uuid.items() if p == child]
    assert sorted(label for _, label in children) == [
        "echo A >> /f",
        "echo B >> /f",
        "echo C >> /f",
    ]
    assert len(lin) == 1 + 1 + 3


def test_identical_commands_give_identical_uuids() -> None:
    sb = FakeSandbox()
    a = run(sb.branch("base", "echo same"))
    b = run(sb.branch("base", "echo same"))
    c = run(sb.branch("base", "echo other"))
    assert a == b != c
    assert len(sb.lineage()) == 3  # base, a(==b), c


def test_branch_failure_is_error() -> None:
    with pytest.raises(SandboxError):
        run(FakeSandbox().branch("base", "exit 2"))


def test_run_timeout_raises() -> None:
    sb = FakeSandbox(lambda job, fs: FakeExecution(duration_s=1.0))
    with pytest.raises(SandboxTimeoutError):
        run(sb.run("img", shell="sleep", timeout=0.02))


def test_batch_order_and_concurrency_limit() -> None:
    sb = FakeSandbox(
        lambda job, fs: FakeExecution(stdout=(job.shell or "") + "\n", duration_s=0.01)
    )
    jobs = [Job(shell=f"j{i}") for i in range(25)]
    results = run(sb.run_batch("img", jobs, concurrency=5))
    assert [r.stdout for r in results] == [f"j{i}\n" for i in range(25)]
    assert sb.peak_inflight == 5


def test_batch_concurrency_bounds() -> None:
    sb = FakeSandbox()
    with pytest.raises(ValueError):
        run(sb.run_batch("img", [Job(shell="x")], concurrency=41))
    with pytest.raises(ValueError):
        run(sb.run_batch("img", [Job(shell="x")], concurrency=0))


def test_batch_timeout_isolated_to_one_job() -> None:
    def handler(job: Any, fs: Any) -> FakeExecution:
        return FakeExecution(stdout="ok\n", duration_s=1.0 if job.shell == "slow" else 0.0)

    sb = FakeSandbox(handler)
    res = run(sb.run_batch("img", [Job(shell="fast"), Job(shell="slow", timeout=0.02)]))
    assert res[0].ok and res[1].timed_out and res[1].exit_code == -1


def test_contree_sandbox_maps_sdk_calls() -> None:
    calls: list[dict[str, Any]] = []

    class Image:
        async def run(self, **kw: Any) -> Any:
            calls.append(kw)
            return SimpleNamespace(stdout=b"out", stderr="", exit_code=0, uuid="u-1")

    class Images:
        async def use(self, ref: str) -> Image:
            assert ref == "alpine:latest"
            return Image()

    sdk = SimpleNamespace(images=Images())
    sb = ContreeSandbox(lambda: "secret-key", "https://x.invalid", sdk=sdk)
    assert "secret-key" not in repr(sb)
    uuid = run(sb.branch("alpine:latest", "pip install x"))
    assert uuid == "u-1"
    assert calls[0] == {"disposable": False, "shell": "pip install x"}
    assert sb.lineage() == [
        ("alpine:latest", None, "root:alpine:latest"),
        ("u-1", "alpine:latest", "pip install x"),
    ]
    r = run(sb.run("alpine:latest", command="/bin/ls", args=["-l"], timeout=3))
    assert r.stdout == "out" and calls[1]["command"] == "/bin/ls" and calls[1]["timeout"] == 3
