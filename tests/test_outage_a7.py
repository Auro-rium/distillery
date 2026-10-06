# ruff: noqa: S101
"""A7: outage-aware restarts. Offline: a fake probe and a fake clock, no real sleeps."""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any

import pytest

from distillery.server.worker import Job, Worker, make_probe, provider_target


class Clock:
    def __init__(self) -> None:
        self.t = 0.0

    def __call__(self) -> float:
        return self.t


class Net:
    """Fake probe: unreachable for the first ``down`` probes (each probe advances the clock)."""

    def __init__(self, clock: Clock, down: int, tick: float = 30.0) -> None:
        self.clock, self.down, self.tick, self.calls = clock, down, tick, 0

    def __call__(self) -> str | None:
        self.calls += 1
        self.clock.t += self.tick
        if self.calls <= self.down:
            return "gaierror: [Errno -3] Temporary failure in name resolution"
        return None


class Script:
    def __init__(self, codes: list[int]) -> None:
        self.codes = codes

    def __call__(self, job: Job, log: Callable[[str], None]) -> int:
        return self.codes.pop(0)


def _wait(pred: Callable[[], bool], timeout: float = 10.0) -> None:
    end = time.monotonic() + timeout
    while not pred():
        assert time.monotonic() < end, "timed out"
        time.sleep(0.005)


def _go(codes: list[int], probe: Any, clock: Clock, **kw: Any) -> tuple[Worker, Job, list[Any]]:
    audits: list[Any] = []
    w = Worker(
        Script(codes), restart_backoff_s=0.0, on_audit=lambda *a: audits.append(a), probe=probe,
        outage_poll_s=0.0, clock=clock, **kw,
    )  # fmt: skip
    job = Job("r1", "tiny", False)
    w.submit(job)
    _wait(lambda: job.state == "finished")
    return w, job, audits


def _actions(audits: list[Any]) -> list[str]:
    return [a[1] for a in audits]


def test_outage_does_not_consume_restarts() -> None:
    clock = Clock()
    _, job, audits = _go([75, 0], Net(clock, down=5), clock, max_restarts=0)
    assert job.error is None and job.restarts == 0  # a zero budget would have ended the run
    acts = _actions(audits)
    assert acts.count("outage_wait") == 1 and acts.count("outage_over") == 1
    assert "restarts_exhausted" not in acts


def test_outage_audits_once_each_with_details() -> None:
    clock = Clock()
    _, job, audits = _go([75, 0], Net(clock, down=4), clock)
    wait = next(a for a in audits if a[1] == "outage_wait")
    over = next(a for a in audits if a[1] == "outage_over")
    assert "name resolution" in wait[3]["probe_error"] and wait[3]["since"] > 0
    assert over[3]["waited_s"] == pytest.approx(120.0)  # 4 later probes, 30 s fake ticks
    assert job.outage_waiting is False and job.outage_waited_total_s >= 120.0


def test_reachable_path_is_unchanged() -> None:
    clock = Clock()
    net = Net(clock, down=0)
    _, job, audits = _go([75, 75, 0], net, clock)
    assert job.restarts == 2 and net.calls == 2
    assert _actions(audits) == ["restart", "restart"]
    assert audits[0][3]["attempt"] == 1


def test_reachable_still_exhausts_the_budget() -> None:
    clock = Clock()
    _, job, audits = _go([75] * 5, Net(clock, down=0), clock, max_restarts=2)
    assert job.restarts == 2 and _actions(audits)[-1] == "restarts_exhausted"
    assert job.error and "exit code 75" in job.error


def test_cap_falls_back_to_the_counted_path() -> None:
    clock = Clock()
    w, job, audits = _go(
        [75] * 6, Net(clock, down=10**6), clock, max_restarts=2, max_outage_wait_s=100.0
    )
    acts = _actions(audits)
    assert acts.count("outage_wait") == 1 and acts.count("outage_cap") == 1
    assert acts.count("outage_over") == 0
    assert job.restarts == 2 and acts[-1] == "restarts_exhausted"  # counted only after the cap
    assert job.outage_waited_total_s >= 100.0
    assert w.provider_reachable is False


def test_cancel_during_the_wait_stops_it_at_once() -> None:
    audits: list[Any] = []
    w = Worker(
        Script([75, 0]), restart_backoff_s=0.0, on_audit=lambda *a: audits.append(a),
        probe=lambda: "down", outage_poll_s=3600.0,
    )  # fmt: skip
    job = Job("r1", "tiny", False)
    w.submit(job)
    _wait(lambda: job.outage_waiting)
    assert w.cancel("r1") == "cancelling"
    _wait(lambda: job.state == "finished")
    assert job.error == "cancelled" and job.restarts == 0 and not job.outage_waiting
    assert "outage_over" not in _actions(audits)


def test_server_stop_during_the_wait_suspends() -> None:
    w = Worker(Script([75, 0]), restart_backoff_s=0.0, probe=lambda: "down", outage_poll_s=3600.0)
    job = Job("r1", "tiny", False)
    w.submit(job)
    _wait(lambda: job.outage_waiting)
    w.stop(join_timeout_s=5)
    assert job.state == "finished" and job.suspend_requested and not job.outage_waiting


def test_watchdog_kill_during_an_outage_is_a_free_restart() -> None:
    clock = Clock()
    _, job, audits = _go([-9, 0], Net(clock, down=3), clock)  # SIGKILL by the watchdog
    assert job.restarts == 0 and job.error is None
    assert _actions(audits).count("outage_wait") == 1


def test_dry_runs_are_not_probed() -> None:
    clock = Clock()
    net = Net(clock, down=10**6)
    w = Worker(Script([75, 0]), restart_backoff_s=0.0, probe=net, outage_poll_s=0.0, clock=clock)
    job = Job("d1", "tiny", True)
    w.submit(job)
    _wait(lambda: job.state == "finished")
    assert net.calls == 0 and job.restarts == 1


def test_a_crashing_probe_counts_as_unreachable_not_a_crash() -> None:
    calls = {"n": 0}

    def probe() -> str | None:
        calls["n"] += 1
        if calls["n"] < 3:
            raise RuntimeError("boom")
        return None

    w = Worker(Script([75, 0]), restart_backoff_s=0.0, probe=probe, outage_poll_s=0.0)
    job = Job("r1", "tiny", False)
    w.submit(job)
    _wait(lambda: job.state == "finished")
    assert job.error is None and job.restarts == 0


def test_chaos_net_window_is_invisible_to_the_probe() -> None:
    """chaos net_window fails LLM calls inside the child; the supervisor probes from ITS OWN
    process, which that injection never touches, so the probe says reachable and the exit-75
    restart is counted as before."""
    clock = Clock()
    _, job, audits = _go([75, 0], Net(clock, down=0), clock)
    assert job.restarts == 1 and "outage_wait" not in _actions(audits)


def test_provider_target_and_real_probe_failure_is_offline() -> None:
    assert provider_target(None) == ("api.tokenfactory.nebius.com", 443, True)
    assert provider_target("https://api.x.example/v1/") == ("api.x.example", 443, True)
    assert provider_target("http://localhost:8080/v1") == ("localhost", 8080, False)
    # .invalid never resolves (RFC 6761), so no packet leaves for a real host
    assert make_probe("https://nonexistent.invalid/v1", timeout_s=1.0)() is not None
