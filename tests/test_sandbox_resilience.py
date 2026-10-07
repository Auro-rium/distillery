# ruff: noqa: S101
"""ContreeSandbox must survive transient transport errors (offline: stub SDKs only)."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any

import pytest
from contree_sdk.sdk.exceptions import ApiTimeoutError, ContreeTransportError

from distillery.sandbox import ContreeSandbox, Job, SandboxError

OK = SimpleNamespace(stdout=b"out", stderr="", exit_code=0, uuid="u-1")


def run(coro: Any) -> Any:
    return asyncio.run(coro)


def _sdk(fail_times: int, exc: Exception, calls: list[int]) -> Any:
    class Image:
        async def run(self, **kw: Any) -> Any:
            calls.append(1)
            if len(calls) <= fail_times:
                raise exc
            return OK

    class Images:
        async def use(self, ref: str) -> Image:
            return Image()

    return SimpleNamespace(images=Images())


def _sb(sdk: Any, **kw: Any) -> ContreeSandbox:
    return ContreeSandbox(lambda: "k", None, project_id="p", sdk=sdk, retry_backoff_s=0.0, **kw)


@pytest.mark.parametrize("exc", [ApiTimeoutError(timeout_type="read"), ContreeTransportError()])
def test_disposable_run_survives_transient_errors(exc: Exception) -> None:
    calls: list[int] = []
    sb = _sb(_sdk(2, exc, calls))
    res = run(sb.run("img", shell="x"))
    assert res.ok and res.stdout == "out" and len(calls) == 3


def test_disposable_run_that_always_fails_raises_sandbox_error_after_bounded_attempts() -> None:
    calls: list[int] = []
    sb = _sb(_sdk(10**6, ApiTimeoutError(timeout_type="read"), calls))
    with pytest.raises(SandboxError, match="ApiTimeoutError"):
        run(asyncio.wait_for(sb.run("img", shell="x"), timeout=5))
    assert len(calls) == 3  # 1 + RUN_RETRIES, then give up (no hang)
    results = run(sb.run_batch("img", [Job(shell="a"), Job(shell="b")]))
    assert [r.ok for r in results] == [False, False]  # failed results, siblings isolated


def test_branch_is_never_resubmitted() -> None:
    calls: list[int] = []
    sb = _sb(_sdk(1, ApiTimeoutError(timeout_type="read"), calls))
    with pytest.raises(SandboxError, match="not resubmitted"):
        run(sb.branch("img", "pip install x"))
    assert len(calls) == 1


def test_branch_that_uploads_files_is_retried_on_a_transport_error() -> None:
    calls: list[int] = []
    sb = _sb(_sdk(2, ContreeTransportError(), calls))
    uuid = run(sb.branch("img", "true", files={"/work/a": b"x"}))
    assert uuid == "u-1" and len(calls) == 3


def test_branch_upload_gives_up_after_bounded_attempts() -> None:
    calls: list[int] = []
    sb = _sb(_sdk(10**6, ApiTimeoutError(timeout_type="read"), calls))
    with pytest.raises(SandboxError, match="ApiTimeoutError"):
        run(asyncio.wait_for(sb.branch("img", "true", files={"/work/a": b"x"}), timeout=5))
    assert len(calls) == 4  # 1 + UPLOAD_RETRIES


def _poll_sdk(fail_times: int, calls: list[int]) -> Any:
    """A stub whose image.run polls an operation through ``_api.get_operation_status`` like the
    real SDK does, and fails if a poll error ever escapes."""

    async def get_operation_status(op: str) -> str:
        calls.append(1)
        if len(calls) <= fail_times:
            raise ApiTimeoutError(timeout_type="read")
        return f"done:{op}"

    api = SimpleNamespace(get_operation_status=get_operation_status)

    class Image:
        async def run(self, **kw: Any) -> Any:
            assert await api.get_operation_status("op-1") == "done:op-1"
            return OK

    class Images:
        async def use(self, ref: str) -> Image:
            return Image()

    return SimpleNamespace(images=Images(), _api=api)


def test_poll_errors_are_retried_in_place_without_resubmitting_the_operation() -> None:
    polls: list[int] = []
    sdk = _poll_sdk(4, polls)
    sb = _sb(sdk, poll_retries=6)
    res = run(sb.branch("img", "setup"))  # branch: no whole-run retry exists, polls still recover
    assert res == "u-1" and len(polls) == 5


def test_poll_that_always_fails_is_bounded_then_becomes_sandbox_error() -> None:
    polls: list[int] = []
    sb = _sb(_poll_sdk(10**6, polls), poll_retries=3)
    with pytest.raises(SandboxError, match="ApiTimeoutError"):
        run(asyncio.wait_for(sb.branch("img", "setup"), timeout=5))
    assert len(polls) == 4  # 1 + poll_retries, then the error is let through


def test_non_transient_poll_errors_pass_through_unretried() -> None:
    polls: list[int] = []

    async def get_operation_status(op: str) -> str:
        polls.append(1)
        raise ValueError("not transient")

    sdk = SimpleNamespace(
        images=SimpleNamespace(), _api=SimpleNamespace(get_operation_status=get_operation_status)
    )
    sb = _sb(sdk)
    wrapped = sb._get_sdk()._api.get_operation_status
    with pytest.raises(ValueError):
        run(wrapped("op"))
    assert len(polls) == 1
