# ruff: noqa: S101
"""Offline tests of scripts/spike_student_latency.py control flow (fakes only, no network)."""

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import pytest

from distillery.sandbox import FakeExecution, FakeSandbox, Job
from distillery.sandbox_executor import AsyncBridge
from distillery.student import StudentServingError

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "spike_student_latency.py"
spec = importlib.util.spec_from_file_location("spike_student_latency", SCRIPT)
assert spec is not None and spec.loader is not None
spike = importlib.util.module_from_spec(spec)
sys.modules["spike_student_latency"] = spike
spec.loader.exec_module(spike)


class Factory:
    def __init__(self, fail_servers: dict[int, Exception] | None = None) -> None:
        self.servers: list[Any] = []
        self.fail = fail_servers or {}

    def __call__(self, bs: int, conc: int, timeout_s: float) -> Any:
        s = spike.FakeSpikeServer(fail=self.fail.get(len(self.servers)))
        self.servers.append(s)
        return s


def run(factory: Any, **kw: Any) -> dict[str, Any]:
    return spike.run_spike(
        model="m/x",
        prompts=spike.fake_prompts(),
        build_deps=kw.pop("build_deps", lambda: "img"),
        make_server=factory,
        **kw,
    )


def test_success_path_runs_all_phases_and_closes() -> None:
    f = Factory()
    res = run(f)
    assert res["outcome"] == "completed"
    names = [p["name"] for p in res["phases"]]
    assert names == [
        "batch1_load",
        "batch2",
        "concurrency_1",
        "concurrency_5",
        "concurrency_10",
        "concurrency_20",
    ]
    assert res["phases"][0]["batch_size"] == 1 and res["phases"][0]["n"] == 1
    assert res["phases"][0]["per_sample_s"] == 2.0 and res["phases"][0]["peak_rss_mb"] == 3100
    assert res["phases"][1]["batch_size"] == 2 and res["phases"][1]["n"] == 2
    assert all(p["n"] <= 40 for p in res["phases"])
    assert res["total_generations"] <= 60
    assert all(s.closed for s in f.servers)
    assert res["build_s"] is not None


def test_generation_cap_is_hard() -> None:
    f = Factory()
    res = run(f, max_generations=10)
    assert res["total_generations"] <= 10
    assert res["outcome"] == "stopped_generation_cap"
    assert res["phases"][-1]["status"] == "skipped_generation_cap"
    assert all(s.closed for s in f.servers)


@pytest.mark.parametrize(
    ("exc", "outcome"),
    [
        (
            StudentServingError("generation batch failed (exit=137, timed_out=False): 'Killed'"),
            "aborted_oom",
        ),
        (StudentServingError("sample failed: MemoryError: Cannot allocate"), "aborted_oom"),
        (
            StudentServingError("generation batch failed (exit=-1, timed_out=True)"),
            "aborted_timeout",
        ),
        (StudentServingError("exit=1: traceback"), "aborted_error"),
    ],
)
def test_batch1_failure_aborts_and_skips_later_phases(exc: Exception, outcome: str) -> None:
    f = Factory({0: exc})
    res = run(f)
    assert res["outcome"] == outcome
    assert res["failure"]["phase"] == "batch1_load"
    assert [p["name"] for p in res["phases"]] == ["batch1_load"]
    assert res["phases"][0]["status"].startswith("failed_")
    assert len(f.servers) == 1 and f.servers[0].closed
    assert res["total_generations"] == 1  # the failed attempt still counts


def test_failure_mid_sweep_keeps_earlier_results() -> None:
    f = Factory({3: StudentServingError("exit=137 Killed")})  # concurrency_5
    res = run(f)
    assert res["outcome"] == "aborted_oom"
    assert [p["status"] for p in res["phases"]][:3] == ["ok", "ok", "ok"]
    assert res["phases"][-1]["name"] == "concurrency_5"
    assert len(res["phases"]) == 4
    assert all(s.closed for s in f.servers)


def test_build_failure_recorded_and_no_server_made() -> None:
    def boom() -> str:
        raise StudentServingError("pip install failed")

    f = Factory()
    res = run(f, build_deps=boom)
    assert res["outcome"] == "aborted_error" and res["failure"]["phase"] == "build"
    assert res["phases"] == [] and f.servers == []


def test_unexpected_exception_still_closes_servers() -> None:
    f = Factory({0: KeyError("bug")})
    with pytest.raises(KeyError):
        run(f)
    assert all(s.closed for s in f.servers)


def test_wall_clock_budget_stops() -> None:
    t = [0.0]

    def clock() -> float:
        t[0] += 100.0
        return t[0]

    res = run(Factory(), max_wall_s=250.0, clock=clock)
    assert res["outcome"] == "aborted_wall_clock"


def test_table_has_baseline_and_fake_label() -> None:
    res = run(Factory())
    table = spike.format_table(res, fake=True)
    assert "FAKE DATA" in table and "0.6B baseline" in table and "7.8" in table
    assert "FAKE" not in spike.format_table(res)


def test_dry_run_cli(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    out = tmp_path / "sub" / "r.json"
    assert spike.main(["--dry-run", "--out", str(out)]) == 0
    assert "FAKE DATA" in capsys.readouterr().out
    assert json.loads(out.read_text())["fake"] is True


def test_dry_run_writes_nothing_without_out(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    assert spike.main(["--dry-run"]) == 0
    assert not (tmp_path / "docs").exists()


def test_build_prompts_are_forty_template_prompts() -> None:
    prompts = spike.build_prompts()
    assert len(prompts) == 40 and prompts[0][0]["role"] == "system"


def test_live_wiring_with_fake_sandbox() -> None:
    """Real ServingImages/SandboxCpuStudent against FakeSandbox returning the script's JSON."""

    def handler(job: Job, fs: dict[str, bytes]) -> FakeExecution:
        if job.stdin is None:
            return FakeExecution()
        batch = json.loads(job.stdin)["messages_batch"]
        out = {
            "results": [{"text": "SELECT 1", "error": None} for _ in batch],
            "load_s": 1.5,
            "gen_s": 4.0,
            "n": len(batch),
            "peak_rss_mb": 3000,
        }
        return FakeExecution("DISTILLERY_OUT:" + json.dumps(out))

    with AsyncBridge() as bridge:
        sandbox = FakeSandbox(handler)
        image = bridge.run(sandbox.ensure_image("docker://base"))
        build, make = spike.live_wiring(sandbox, image, bridge, "Qwen/Qwen3-1.7B")
        res = spike.run_spike(
            model="Qwen/Qwen3-1.7B",
            prompts=spike.fake_prompts(),
            build_deps=build,
            make_server=make,
        )
    assert res["outcome"] == "completed"
    assert res["phases"][0]["per_sample_s"] == 4.0 and res["phases"][0]["peak_rss_mb"] == 3000
