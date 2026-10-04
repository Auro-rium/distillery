# ruff: noqa: S101, S604
from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any

import pytest

from distillery.sandbox import (
    TRUNCATE_OUTPUT_AT,
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
    assert calls[0] == {
        "disposable": False,
        "shell": "pip install x",
        "truncate_output_at": TRUNCATE_OUTPUT_AT,
    }
    assert sb.lineage() == [
        ("alpine:latest", None, "root:alpine:latest"),
        ("u-1", "alpine:latest", "pip install x"),
    ]
    r = run(sb.run("alpine:latest", command="/bin/ls", args=["-l"], timeout=3))
    assert r.stdout == "out" and calls[1]["command"] == "/bin/ls" and calls[1]["timeout"] == 3


# ---- real contree-sdk 0.3.6 wiring (stub SDK modules; no network) ---------------------------


def _install_stub_sdk(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    import sys
    from types import ModuleType

    seen: dict[str, Any] = {}

    class IAMAuth:
        def __init__(self, token: str, project_id: str, base_url: str) -> None:
            seen["auth"] = {"token": token, "project_id": project_id, "base_url": base_url}

    class ContreeConfig:
        def __init__(self, auth: Any, **kw: Any) -> None:
            seen["config_auth"] = auth
            seen["config_kw"] = kw

    class Contree:
        def __init__(self, config: Any = None, **kw: Any) -> None:
            seen["contree_args"] = (config, kw)
            self.images = SimpleNamespace()

    top, auth, cfg = ModuleType("contree_sdk"), ModuleType("contree_sdk.auth"), ModuleType("x")
    top.Contree = Contree  # type: ignore[attr-defined]
    auth.IAMAuth = IAMAuth  # type: ignore[attr-defined]
    cfg.ContreeConfig = ContreeConfig  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "contree_sdk", top)
    monkeypatch.setitem(sys.modules, "contree_sdk.auth", auth)
    monkeypatch.setitem(sys.modules, "contree_sdk.config", cfg)
    return seen


def test_contree_sandbox_builds_iam_config_with_project(monkeypatch: pytest.MonkeyPatch) -> None:
    seen = _install_stub_sdk(monkeypatch)
    monkeypatch.delenv("NEBIUS_PROJECT_ID", raising=False)
    monkeypatch.setenv("NEBIUS_AI_PROJECT", "project-e00abc")
    sb = ContreeSandbox(lambda: "secret-key", None)
    sb._get_sdk()
    assert seen["auth"] == {
        "token": "secret-key",
        "project_id": "project-e00abc",
        "base_url": "https://api.tokenfactory.nebius.com/sandboxes",
    }
    assert seen["config_auth"] is not None and seen["contree_args"][0] is not None
    assert "secret-key" not in repr(sb) and "project-e00abc" not in repr(sb)


def test_contree_sandbox_explicit_project_and_url(monkeypatch: pytest.MonkeyPatch) -> None:
    seen = _install_stub_sdk(monkeypatch)
    monkeypatch.delenv("NEBIUS_AI_PROJECT", raising=False)
    monkeypatch.delenv("NEBIUS_PROJECT_ID", raising=False)
    ContreeSandbox(lambda: "k", "https://x.invalid", project_id="p1")._get_sdk()
    assert seen["auth"]["project_id"] == "p1" and seen["auth"]["base_url"] == "https://x.invalid"


def test_contree_sandbox_reads_official_env_name_first(monkeypatch: pytest.MonkeyPatch) -> None:
    seen = _install_stub_sdk(monkeypatch)
    monkeypatch.setenv("NEBIUS_PROJECT_ID", "official")
    monkeypatch.setenv("NEBIUS_AI_PROJECT", "legacy")
    ContreeSandbox(lambda: "k", None)._get_sdk()
    assert seen["auth"]["project_id"] == "official"


def test_contree_sandbox_refuses_without_project(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_stub_sdk(monkeypatch)
    monkeypatch.delenv("NEBIUS_AI_PROJECT", raising=False)
    monkeypatch.delenv("NEBIUS_PROJECT_ID", raising=False)
    with pytest.raises(SandboxError, match="NEBIUS_PROJECT_ID"):
        ContreeSandbox(lambda: "k", None)._get_sdk()


def test_contree_sandbox_ensure_image_imports_via_oci() -> None:
    seen: list[str] = []

    class Images:
        async def oci(self, ref: str) -> Any:
            seen.append(ref)
            return SimpleNamespace(uuid="u-9")

    sb = ContreeSandbox(lambda: "k", None, project_id="p", sdk=SimpleNamespace(images=Images()))
    assert run(sb.ensure_image("python:3.12-slim")) == "u-9" and seen == ["python:3.12-slim"]


def test_contree_sandbox_maps_files_to_absolute_paths_and_stdin() -> None:
    calls: list[dict[str, Any]] = []

    class Image:
        async def run(self, **kw: Any) -> Any:
            calls.append(kw)
            return SimpleNamespace(stdout=None, stderr=None, exit_code=0, uuid=None)

    class Images:
        async def use(self, ref: str) -> Image:
            return Image()

    sb = ContreeSandbox(lambda: "k", None, project_id="p", sdk=SimpleNamespace(images=Images()))
    run(sb.run("img", shell="true", files={"work/gen.py": b"x", "/models/a": b"y"}, stdin="{}"))
    assert calls[0]["files"] == {"/work/gen.py": b"x", "/models/a": b"y"}
    assert calls[0]["stdin"] == "{}" and calls[0]["truncate_output_at"] >= 400_000


def _sdk_raising(exc: BaseException) -> Any:
    class Image:
        async def run(self, **kw: Any) -> Any:
            raise exc

    class Images:
        async def use(self, ref: str) -> Image:
            return Image()

    return SimpleNamespace(images=Images())


def test_sdk_operation_timeout_becomes_sandbox_timeout_and_is_isolated_in_batches() -> None:
    from uuid import uuid4

    from contree_sdk.sdk.exceptions.operation import OperationTimedOutError

    sdk = _sdk_raising(OperationTimedOutError(operation_uuid=uuid4()))
    sb = ContreeSandbox(lambda: "k", None, project_id="p", sdk=sdk)
    with pytest.raises(SandboxTimeoutError):
        run(sb.run("img", shell="x"))
    res = run(sb.run_batch("img", [Job(shell="a"), Job(shell="b")]))
    assert all(r.timed_out and r.exit_code == -1 and r.error for r in res)


def test_sdk_errors_become_sandbox_errors_without_leaking_siblings() -> None:
    from contree_sdk.sdk.exceptions import ContreeError
    from contree_sdk.sdk.exceptions.api import ApiStatusCodeError

    sdk = _sdk_raising(ApiStatusCodeError(status=500, error="boom"))
    sb = ContreeSandbox(lambda: "k", None, project_id="p", sdk=sdk)
    with pytest.raises(SandboxError, match="ApiStatusCodeError"):
        run(sb.run("img", shell="x"))
    res = run(sb.run_batch("img", [Job(shell="a"), Job(shell="b")]))
    assert [r.ok for r in res] == [False, False] and not any(r.timed_out for r in res)
    assert issubclass(ApiStatusCodeError, ContreeError)


def test_fake_sandbox_ensure_image_is_identity() -> None:
    sb: Sandbox = FakeSandbox()
    assert run(sb.ensure_image("docker://python:3.12-slim")) == "docker://python:3.12-slim"


def test_sdk_gets_a_transport_timeout_big_enough_for_adapter_uploads(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """P4: the SDK default (10 s) timed out uploading the LoRA adapter (POST /v1/files)."""
    seen = _install_stub_sdk(monkeypatch)
    monkeypatch.setenv("NEBIUS_AI_PROJECT", "project-e00abc")
    ContreeSandbox(lambda: "k", None)._get_sdk()
    assert seen["config_kw"]["transport_timeout"] >= 120
