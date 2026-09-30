# ruff: noqa: S101, S603, S105
from __future__ import annotations

import hmac
import json
import subprocess
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from distillery.cli import EXIT_FAIL, EXIT_OK, EXIT_REFUSED, main
from distillery.orchestrator import DRY_RUN_LABEL, Deps, PipelineError
from distillery.pipeline_fakes import FAKE_MODELS
from distillery.sandbox_executor import AsyncBridge

TOKEN = "s3cret-admin-token"


@pytest.fixture
def bridge() -> Iterator[AsyncBridge]:
    with AsyncBridge() as b:
        yield b


class Reached(PipelineError):
    """Raised by the injected deps factory: proves the gates were passed."""


@pytest.fixture
def live_env(tmp_path: Path) -> dict[str, str]:
    prices = tmp_path / "prices.json"
    price = {"input_per_mtok": 1.0, "output_per_mtok": 2.0, "source": "test", "date": "2026-09-29"}
    prices.write_text(json.dumps({m: price for m in FAKE_MODELS.values()}))
    env = {
        "DISTILLERY_ADMIN_TOKEN": TOKEN,
        "DISTILLERY_PRICES_FILE": str(prices),
        "PATH": "/usr/bin",
    }
    env.update({f"DISTILLERY_MODEL_{r.upper()}": m for r, m in FAKE_MODELS.items()})
    return env


def run_cli(tmp_path: Path, *args: str, **kw: Any) -> tuple[int, list[str]]:
    lines: list[str] = []
    code = main(["--root", str(tmp_path / "home"), *args], out=lines.append, **kw)
    return code, lines


LIVE = ("run", "--scale", "tiny", "--run-id", "real-1", "--finetune-estimate-usd", "1")


def factory_recorder(calls: list[str]) -> Any:
    def factory(_c: Any, _p: Any, _b: AsyncBridge) -> Deps:
        calls.append("called")
        raise Reached("deps factory reached")

    return factory


def test_live_run_without_token_is_refused(tmp_path: Path, live_env: dict[str, str]) -> None:
    calls: list[str] = []
    code, out = run_cli(
        tmp_path, *LIVE, "--i-approve-spend", env=live_env, deps_factory=factory_recorder(calls),
        token_prompt=lambda: "",
    )  # fmt: skip
    assert code == EXIT_REFUSED and "admin token" in out[-1] and calls == []
    assert not (tmp_path / "home" / "runs" / "real-1").exists()


def test_live_run_with_wrong_token_is_refused(tmp_path: Path, live_env: dict[str, str]) -> None:
    env = {**live_env, "DISTILLERY_ADMIN_TOKEN_SUPPLIED": "nope"}
    calls: list[str] = []
    code, _ = run_cli(
        tmp_path, *LIVE, "--i-approve-spend", env=env, deps_factory=factory_recorder(calls)
    )
    assert code == EXIT_REFUSED and calls == []


def test_no_admin_token_configured_refuses_everything(
    tmp_path: Path, live_env: dict[str, str]
) -> None:
    env = {k: v for k, v in live_env.items() if k != "DISTILLERY_ADMIN_TOKEN"}
    env["DISTILLERY_ADMIN_TOKEN_SUPPLIED"] = ""
    code, _ = run_cli(tmp_path, *LIVE, "--i-approve-spend", env=env, token_prompt=lambda: "")
    assert code == EXIT_REFUSED


def test_token_ok_but_no_approval_flag_is_refused_then_remembered(
    tmp_path: Path, live_env: dict[str, str]
) -> None:
    env = {**live_env, "DISTILLERY_ADMIN_TOKEN_SUPPLIED": TOKEN}
    calls: list[str] = []
    fac = factory_recorder(calls)
    code, out = run_cli(tmp_path, *LIVE, env=env, deps_factory=fac)
    assert code == EXIT_REFUSED and "--i-approve-spend" in out[-1] and calls == []
    code, _ = run_cli(tmp_path, *LIVE, "--i-approve-spend", env=env, deps_factory=fac)
    assert code == EXIT_FAIL and calls == ["called"]  # gates passed; factory raised Reached
    code, _ = run_cli(tmp_path, *LIVE, env=env, deps_factory=fac)  # approval remembered per run
    assert code == EXIT_FAIL and calls == ["called", "called"]


def test_token_check_is_constant_time_compare(
    tmp_path: Path, live_env: dict[str, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: list[tuple[bytes, bytes]] = []
    real = hmac.compare_digest

    def spy(a: bytes, b: bytes) -> bool:
        seen.append((a, b))
        return real(a, b)

    monkeypatch.setattr(hmac, "compare_digest", spy)
    env = {**live_env, "DISTILLERY_ADMIN_TOKEN_SUPPLIED": TOKEN}
    run_cli(tmp_path, *LIVE, env=env, deps_factory=factory_recorder([]))
    assert seen == [(TOKEN.encode(), TOKEN.encode())]


def test_live_run_without_serving_path_refuses_without_spending(
    tmp_path: Path, live_env: dict[str, str]
) -> None:
    env = {**live_env, "DISTILLERY_ADMIN_TOKEN_SUPPLIED": TOKEN}
    code, out = run_cli(tmp_path, *LIVE, "--i-approve-spend", env=env)
    assert code == EXIT_REFUSED and "refused" in out[-1]  # missing NEBIUS_* / sandbox config


def test_dry_run_prefix_and_flag_are_enforced(tmp_path: Path) -> None:
    code, out = run_cli(tmp_path, "run", "--dry-run", "--run-id", "sql-real", env={})
    assert code == EXIT_REFUSED and "dry" in out[-1]


def test_dry_run_status_report_and_module_entry(tmp_path: Path) -> None:
    code, out = run_cli(tmp_path, "run", "--pack", "sql", "--scale", "tiny", "--dry-run", env={})
    assert code == EXIT_OK, out
    text = "\n".join(out)
    assert text.count(DRY_RUN_LABEL) >= 3 and "decision:" in text
    root = tmp_path / "home" / "dry-runs" / "runs" / "dry-sql-tiny"
    report = json.loads((root / "report.json").read_text())
    assert report["label"] == DRY_RUN_LABEL and report["dry_run"] is True
    assert not (tmp_path / "home" / "runs").exists()  # never mixed with real runs

    code, out = run_cli(tmp_path, "status", "dry-sql-tiny", env={})
    assert code == EXIT_OK and any("final_eval" in line and "complete" in line for line in out)
    assert out[0] == DRY_RUN_LABEL
    code, out = run_cli(tmp_path, "report", "dry-sql-tiny", env={})
    assert code == EXIT_OK and DRY_RUN_LABEL in "\n".join(out)
    code, out = run_cli(tmp_path, "report", "dry-sql-tiny", "--json", env={})
    assert json.loads("\n".join(out))["run_id"] == "dry-sql-tiny"
    code, out = run_cli(tmp_path, "report", "dry-nope", env={})
    assert code == EXIT_FAIL

    proc = subprocess.run(
        [sys.executable, "-m", "distillery", "--root", str(tmp_path / "home")]
        + ["status", "dry-sql-tiny"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0 and "final_eval" in proc.stdout


# ---- live wiring (make_live_deps): offline, with FakeSandbox standing in for Nebius ---------------


def _live_config(**kw: Any) -> Any:
    from pydantic import SecretStr

    from distillery.config import Config

    base: dict[str, Any] = {
        "nebius_api_key": SecretStr("k-secret"),
        "nebius_base_url": "https://api.example/v1/",
        "nebius_project_id": SecretStr("proj-secret"),
        "model_ids": {"student": "Qwen/Qwen3-0.6B"},
    }
    base.update(kw)
    return Config(**base)


def test_make_live_deps_refuses_incomplete_config_before_touching_anything(
    bridge: AsyncBridge,
) -> None:
    from distillery.cli import make_live_deps
    from distillery.orchestrator import ConfigRefusal, PipelineConfig

    class Boom:
        def __getattr__(self, name: str) -> Any:
            raise AssertionError(f"sandbox touched: {name}")

    for missing, match in (
        ("nebius_api_key", "NEBIUS_API_KEY"),
        ("nebius_base_url", "NEBIUS_BASE_URL"),
        ("nebius_project_id", "NEBIUS_AI_PROJECT"),
    ):
        cfg = _live_config(**{missing: None})
        with pytest.raises(ConfigRefusal, match=match):
            make_live_deps(cfg, PipelineConfig(), bridge, sandbox=Boom())  # type: ignore[arg-type]


def test_make_live_deps_wires_executor_on_python_image_and_leaves_serving_to_pipeline(
    bridge: AsyncBridge,
) -> None:
    from distillery.cli import make_live_deps
    from distillery.finetune import FineTuneClient
    from distillery.orchestrator import PipelineConfig
    from distillery.sandbox import FakeSandbox
    from distillery.sandbox_executor import SandboxExecutor

    fake = FakeSandbox()
    deps = make_live_deps(_live_config(), PipelineConfig(), bridge, env={}, sandbox=fake)
    assert deps.sandbox is fake and deps.bridge is bridge
    assert deps.sandbox_image == "docker://python:3.12-slim"  # imported (ensure_image) up front
    assert isinstance(deps.executor, SandboxExecutor) and deps.executor._concurrency <= 20
    assert deps.executor._base_image == deps.sandbox_image
    assert deps.student_factory is None and deps.base_factory is None  # pipeline resolves them
    assert isinstance(deps.finetune, FineTuneClient)
    assert deps.finetune._c.max_retries == 0  # create_job must never be auto-retried (billable)
    assert deps.transport is not None


def test_make_live_deps_image_override_and_real_sandbox_construction(
    bridge: AsyncBridge, monkeypatch: pytest.MonkeyPatch
) -> None:
    import distillery.sandbox as sandbox_mod
    from distillery.cli import make_live_deps
    from distillery.orchestrator import PipelineConfig
    from distillery.sandbox import FakeSandbox

    seen: dict[str, Any] = {}

    class Recorder(FakeSandbox):
        def __init__(self, key_getter: Any, base_url: Any = None, **kw: Any) -> None:
            super().__init__()
            seen.update(key=key_getter(), url=base_url, **kw)

    monkeypatch.setattr(sandbox_mod, "ContreeSandbox", Recorder)
    env = {
        "DISTILLERY_SANDBOX_URL": "https://sb.example",
        "DISTILLERY_SANDBOX_IMAGE": "docker://x:1",
    }
    deps = make_live_deps(_live_config(), PipelineConfig(), bridge, env=env)
    assert deps.sandbox_image == "docker://x:1"
    assert seen == {"key": "k-secret", "url": "https://sb.example", "project_id": "proj-secret"}
    seen.clear()
    make_live_deps(_live_config(), PipelineConfig(), bridge, env={})
    assert seen["url"] is None  # SDK default base URL


def test_live_run_uses_sandbox_cpu_serving_and_dry_or_injected_runs_do_not(
    tmp_path: Path, live_env: dict[str, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    import distillery.cli as cli

    seen: list[str] = []

    def fake_live(config: Any, pcfg: Any, bridge: Any, **_kw: Any) -> Deps:
        seen.append(pcfg.student_serving)
        raise Reached("stop after wiring")

    monkeypatch.setattr(cli, "make_live_deps", fake_live)
    env = {**live_env, "DISTILLERY_ADMIN_TOKEN_SUPPLIED": TOKEN}
    code, _ = run_cli(tmp_path, *LIVE, "--i-approve-spend", env=env)
    assert code == EXIT_FAIL and seen == ["sandbox_cpu"]
    calls: list[str] = []
    seen_cfg: list[str] = []

    def injected(_c: Any, p: Any, _b: AsyncBridge) -> Deps:
        seen_cfg.append(p.student_serving)
        calls.append("called")
        raise Reached("injected")

    run_cli(tmp_path, *LIVE, env=env, deps_factory=injected)
    assert seen_cfg == ["injected"]  # a caller-supplied Deps keeps its own serving factories


def test_dry_run_and_status_label_costs_as_estimates(tmp_path: Path) -> None:
    code, out = run_cli(tmp_path, "run", "--pack", "sql", "--scale", "tiny", "--dry-run", env={})
    assert code == EXIT_OK
    assert any("ESTIMATE" in line and "spend" in line for line in out)
    code, out = run_cli(tmp_path, "status", "dry-sql-tiny", env={})
    assert any("ESTIMATE" in line and "spend" in line for line in out)
