# ruff: noqa: S101
"""A5: fault injection is inert for normal runs, injects exactly the planned faults once across
restarts, and the local dry rehearsal meets the pre-registered pass criteria."""

from __future__ import annotations

import asyncio
import importlib.util
import json
from pathlib import Path
from typing import Any

import openai
import pytest

from distillery import chaos
from distillery.cli import main
from distillery.sandbox import FakeSandbox, Job
from distillery.store import Store

PLAN = {
    "llm_503": {"after_calls": 1, "n": 2},
    "net_window": {"after_calls": 100, "seconds": 5},
    "kill": ["teacher_data"],
    "sandbox_fail": {"stage": "dev_eval", "jobs": 2},
}


class Inner:
    def __init__(self) -> None:
        self.calls = 0
        outer = self

        class C:
            @staticmethod
            async def create(**_: Any) -> str:
                outer.calls += 1
                return "ok"

        self.chat = type("Chat", (), {"completions": C})()


def _audit_into(log: list[tuple[str, dict[str, Any]]]) -> chaos.AuditFn:
    return lambda action, detail: log.append((action, detail))


def test_only_chaos_run_ids_honour_the_plan() -> None:
    env = {chaos.ENV: json.dumps(PLAN)}
    assert chaos.plan_for("chaos-sql-gated-1", env) is not None
    assert chaos.plan_for("dry-chaos-x", env) is not None
    for rid in ("sql-gated-20261006", "dry-sql-tiny", "xchaos-1", "chaos"):
        assert chaos.plan_for(rid, env) is None
    assert chaos.plan_for("chaos-a", {}) is None


def test_unknown_plan_key_is_refused() -> None:
    with pytest.raises(ValueError, match="keys from"):
        chaos.ChaosPlan.parse('{"llm_504": {}}')


def test_normal_dry_run_ignores_chaos_env(tmp_path: Path) -> None:
    lines: list[str] = []
    env = {chaos.ENV: json.dumps(PLAN)}
    code = main(
        ["--root", str(tmp_path), "run", "--scale", "tiny", "--dry-run"], env=env, out=lines.append
    )
    assert code == 0
    run_dir = tmp_path / "dry-runs" / "runs" / "dry-sql-tiny"
    assert not (run_dir / chaos.STATE_FILE).exists()
    assert not (run_dir / "fake_finetune_state.json").exists()
    store = Store(tmp_path / "dry-runs")
    assert store.list_audit("dry-sql-tiny") == []
    store.close()


def test_bad_plan_on_chaos_run_is_a_refusal(tmp_path: Path) -> None:
    lines: list[str] = []
    code = main(
        ["--root", str(tmp_path), "run", "--scale", "tiny", "--dry-run", "--run-id", "dry-chaos-b"],
        env={chaos.ENV: "{nope"},
        out=lines.append,
    )
    assert code == 2 and any("refused" in ln for ln in lines)


def test_transport_injects_each_fault_once_across_restarts(tmp_path: Path) -> None:
    plan = chaos.ChaosPlan.parse(
        json.dumps({**PLAN, "net_window": {"after_calls": 3, "seconds": 5}})
    )
    log: list[tuple[str, dict[str, Any]]] = []
    state = chaos.ChaosState(tmp_path)
    now = [1000.0]

    def attempt() -> list[str]:
        t = chaos.ChaosTransport(Inner(), plan, state, _audit_into(log), lambda: now[0])
        out = []
        for _ in range(8):
            try:
                out.append(asyncio.run(t.chat.completions.create(model="m")))
            except openai.InternalServerError:
                out.append("503")
            except openai.APIConnectionError:
                out.append("net")
        return out

    first = attempt()
    assert first[:3] == ["ok", "503", "503"]  # after_calls=1: the first call passes
    assert first[3] == "net"  # window opened at call 4 and still open
    now[0] += 6.0  # window over
    second = attempt()  # a "restarted child": counters reset, totals persist
    assert "503" not in second and "net" not in second
    kinds = [a for a, _ in log]
    assert kinds.count("inject_llm_503") == 2 and kinds.count("inject_net_window") == 1


def test_sandbox_fails_k_jobs_only_in_the_planned_stage(tmp_path: Path) -> None:
    plan = chaos.ChaosPlan.parse(json.dumps(PLAN))
    log: list[tuple[str, dict[str, Any]]] = []
    stage = ["teacher_data"]
    sb = chaos.ChaosSandbox(
        FakeSandbox(), plan, chaos.ChaosState(tmp_path), _audit_into(log), lambda: stage[0]
    )
    jobs = [Job(shell="echo hi") for _ in range(3)]

    async def go() -> list[int]:
        img = await sb.ensure_image("img")
        return [r.exit_code for r in await sb.run_batch(img, jobs)]

    assert asyncio.run(go()) == [0, 0, 0] and log == []
    stage[0] = "dev_eval_r1"
    assert asyncio.run(go()) == [-1, -1, 0]  # k=2 jobs lost, the rest ran
    assert asyncio.run(go()) == [0, 0, 0]  # never more than k in total
    assert [a for a, _ in log] == ["inject_sandbox_exit_-1"] * 2


def test_supervisor_is_inert_for_normal_runs(tmp_path: Path) -> None:
    sup = chaos.ChaosSupervisor(
        lambda rid: chaos.plan_for(rid, {chaos.ENV: json.dumps(PLAN)}),
        lambda rid: tmp_path,
        lambda rid: (["teacher_data"], True),
        lambda *a: None,
    )
    killed: list[int] = []
    stop = sup("sql-gated-1", lambda: killed.append(1))
    stop()
    assert killed == [] and not list(tmp_path.glob("chaos_kill_*"))


def test_local_rehearsal_meets_the_preregistered_criteria(tmp_path: Path) -> None:
    spec = importlib.util.spec_from_file_location(
        "a5_rehearsal", Path(__file__).parents[1] / "docs" / "proofs" / "a5_rehearsal.py"
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    result = mod.rehearse(tmp_path, timeout_s=240.0)
    assert result["checks"] == dict.fromkeys(result["checks"], True), result["checks"]
    assert result["supervisor_restarts"] >= 1
