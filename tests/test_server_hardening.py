"""Reviewer-finding regressions: examples, null fields, SSE, worker shutdown, anonymous limits,
playground cap race, dry-run parity with the CLI. Everything offline (dry pipeline, fakes)."""

from __future__ import annotations

import asyncio
import json
import os
import subprocess  # noqa: S404
import sys
import threading
import time
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from fastapi.testclient import TestClient
from server_helpers import (
    ADMIN,
    MODELS,
    FakeTransport,
    make_config,
    make_settings,
    scripted_executor,
    seed_finished_run,
)

from distillery.llm import LLMClient
from distillery.server import create_app, sse
from distillery.server.playground import DemoBudgetExhaustedError
from distillery.server.views import example_totals, filter_examples
from distillery.server.worker import run_child

REPO = Path(__file__).resolve().parents[1]


def _wait(client: TestClient, run_id: str, status: str, timeout: float = 120.0) -> dict[str, Any]:
    end = time.monotonic() + timeout
    d: dict[str, Any] = {}
    while time.monotonic() < end:
        d = client.get(f"/api/runs/{run_id}").json()
        if d["status"] == status:
            return d
        time.sleep(0.1)
    raise AssertionError(f"{run_id} never reached {status}: {d}")


def _dry_env() -> dict[str, str]:
    env = dict(os.environ)
    env.pop("NEBIUS_API_KEY", None)
    return env


# ---- 1/2/9: real pipeline through the worker's own executor ---------------------------------
def test_end_to_end_dry_pipeline_examples_and_cli_parity(tmp_path: Path) -> None:
    s = make_settings(tmp_path)  # executor=None -> the real subprocess executor
    rid = "dry-sql-tiny"
    with TestClient(create_app(s)) as c:
        assert (
            c.post("/api/runs", json={"scale": "tiny", "dry_run": True, "run_id": rid}).status_code
            == 202
        )
        _wait(c, rid, "complete")
        r = c.get(f"/api/runs/{rid}/examples?kind=all&limit=200")
        assert r.headers["x-examples-available"] == "true"
        items = r.json()
        assert items, "examples must be non-empty for a real dry run"
        assert {e["kind"] for e in items} <= {"fixed", "still_wrong", "regressed"}
        for kind in ("fixed", "still_wrong", "regressed"):
            got = c.get(f"/api/runs/{rid}/examples?kind={kind}&limit=200").json()
            assert got == [e for e in items if e["kind"] == kind]
        assert int(r.headers["x-examples-cap-per-kind"]) == 20
        totals = json.loads(r.headers["x-examples-totals"])
        assert totals["fixed"] >= sum(e["kind"] == "fixed" for e in items)
        api_report = c.get(f"/api/runs/{rid}/report").json()
    # The same run through the CLI (same run id) must give identical rounds: the worker adds no
    # seed / round overrides of its own.
    cli_root = tmp_path / "cli"
    subprocess.run(  # noqa: S603
        [sys.executable, "-m", "distillery", "--root", str(cli_root), "run", "--pack", "sql",
         "--scale", "tiny", "--run-id", rid, "--dry-run"],
        env=_dry_env(), check=True, capture_output=True, timeout=180,
    )  # fmt: skip
    cli_report = json.loads(
        (cli_root / "dry-runs" / "runs" / rid / "report.json").read_text(encoding="utf-8")
    )
    assert [r["dev_acc"] for r in api_report["rounds"]] == [
        r["dev_acc"] for r in cli_report["rounds"]
    ]
    assert api_report["config"]["pipeline"] == cli_report["config"]["pipeline"]
    assert api_report["decision"] == cli_report["decision"]


def test_filter_examples_reads_nested_and_toplevel_and_stored_kind() -> None:
    ex = [
        {"kind": "fixed", "task_id": "a", "base_ok": False, "student_ok": True},
        {"kind": "regressed", "task_id": "b", "base_ok": True, "student_ok": False},
    ]
    nested = {"evaluation": {"examples": ex}}
    top = {"examples": ex}
    for rep in (nested, top):
        items, ok = filter_examples(rep, "fixed", 10)
        assert ok and [e["task_id"] for e in items] == ["a"]
    assert filter_examples({"evaluation": {}}, "all", 10) == ([], False)
    # stored kind wins over a contradictory derivation
    odd = {"evaluation": {"examples": [{"kind": "still_wrong", "task_id": "z", "base_ok": True}]}}
    assert [e["task_id"] for e in filter_examples(odd, "still_wrong", 5)[0]] == ["z"]


def test_example_totals_only_from_recorded_gate_counts() -> None:
    gate = {"n": 20, "base_acc": 0.3, "student_only_vs_base": 10, "base_only_vs_student": 2}
    t = example_totals({"evaluation": {"gate": gate}})
    # base right 6 = 2 regressed + 4 both right; still_wrong = 20 - 4 - 10 - 2
    assert t == {"fixed": 10, "regressed": 2, "still_wrong": 4}
    assert example_totals({}) == {"fixed": None, "still_wrong": None, "regressed": None}


# ---- 3: null fields ---------------------------------------------------------------------------
def test_null_fields_are_null_not_invented(tmp_path: Path) -> None:
    s = make_settings(tmp_path)
    s.executor = scripted_executor(s.root)
    with TestClient(create_app(s)) as c:
        seed_finished_run(s.root, "dry-sql-tiny-n").close()
        d = c.get("/api/runs/dry-sql-tiny-n").json()
        assert d["sandbox"] == {"operations": None, "concurrency_peak": None}
        assert d["verifier"]["code"] is None
        for st in d["stages"]:  # completed: only the end time was stored
            assert st["ended_at"] and st["started_at"] is None
    with TestClient(
        create_app(make_settings(tmp_path / "x", config=make_config(model_ids={})))
    ) as c:
        assert c.get("/api/config").json()["models"] == {
            "planner": None, "teacher": None, "triage": None, "student": None,
        }  # fmt: skip


def test_running_stage_has_start_but_no_end(tmp_path: Path) -> None:
    gate = threading.Event()
    s = make_settings(tmp_path)
    s.executor = scripted_executor(s.root, gate)
    with TestClient(create_app(s)) as c:
        rid = c.post("/api/runs", json={"scale": "tiny", "dry_run": True}).json()["run_id"]
        _wait(c, rid, "running", 10)
        st = c.get(f"/api/runs/{rid}").json()["stages"][0]
        assert st["status"] == "done" or (st["started_at"] and st["ended_at"] is None)
        gate.set()
        _wait(c, rid, "complete", 10)


# ---- 4: SSE timestamps + last_event_id query --------------------------------------------------
def _events(text: str) -> list[dict[str, str]]:
    out = []
    for block in text.split("\n\n"):
        if block.strip() and not block.startswith(":"):
            out.append(dict(ln.partition(": ")[::2] for ln in block.splitlines()))
    return out


def test_sse_last_event_id_query_and_observed_at(tmp_path: Path) -> None:
    s = make_settings(tmp_path)
    s.executor = scripted_executor(s.root)
    with TestClient(create_app(s)) as c:
        rid = c.post("/api/runs", json={"scale": "tiny", "dry_run": True}).json()["run_id"]
        _wait(c, rid, "complete", 10)
        full = _events(c.get(f"/api/runs/{rid}/events").text)
        ids = [int(e["id"]) for e in full]
        via_q = _events(c.get(f"/api/runs/{rid}/events?last_event_id={ids[-2]}").text)
        assert [int(e["id"]) for e in via_q] == [ids[-1]]
        # the query parameter wins over the header
        both = c.get(
            f"/api/runs/{rid}/events?last_event_id={ids[-2]}", headers={"Last-Event-ID": "0"}
        )
        assert [int(e["id"]) for e in _events(both.text)] == [ids[-1]]
        for e in full:
            data = json.loads(e["data"])
            assert "ts" not in data and data["observed_at"]
        stage = json.loads(next(e for e in full if e["event"] == "stage")["data"])
        assert stage["at"]  # stored store timestamp of the transition


# ---- 5: SSE caps ------------------------------------------------------------------------------
def test_stream_limiter_global_and_per_ip() -> None:
    lim = sse.StreamLimiter(3, 2)
    assert lim.acquire("a") and lim.acquire("a") and not lim.acquire("a")
    assert lim.acquire("b") and not lim.acquire("c")  # global cap 3
    lim.release("a")
    assert lim.acquire("c")
    assert lim.active() == 3


def test_events_endpoint_returns_429_when_streams_exhausted(tmp_path: Path) -> None:
    s = make_settings(tmp_path, sse_max_streams=1, sse_max_streams_per_ip=1)
    s.executor = scripted_executor(s.root)
    app = create_app(s)
    with TestClient(app) as c:
        rid = c.post("/api/runs", json={"scale": "tiny", "dry_run": True}).json()["run_id"]
        _wait(c, rid, "complete", 10)
        assert app.state.stream_limiter.acquire("testclient")  # someone holds the only slot
        r = c.get(f"/api/runs/{rid}/events")
        assert r.status_code == 429 and r.json()["error"] == "too_many_streams"
        app.state.stream_limiter.release("testclient")
        assert c.get(f"/api/runs/{rid}/events").status_code == 200
        assert app.state.stream_limiter.active() == 0  # released after a finished stream


def test_astream_stops_when_client_disconnects_and_releases() -> None:
    closed: list[bool] = []
    calls = [0]

    async def gone() -> bool:
        calls[0] += 1
        return calls[0] >= 3

    async def run() -> list[str]:
        return [
            f
            async for f in sse.astream(
                lambda _a: ([], False),
                0,
                heartbeat_s=999,
                poll_s=0.001,
                is_disconnected=gone,
                on_close=lambda: closed.append(True),
            )  # fmt: skip
        ]

    assert asyncio.run(run()) == [] and closed == [True]


# ---- 6: shutdown ------------------------------------------------------------------------------
def test_lifespan_exit_kills_a_stubborn_live_child(tmp_path: Path) -> None:
    script = tmp_path / "stubborn.py"
    script.write_text(
        "import os, signal, time\nsignal.signal(signal.SIGINT, signal.SIG_IGN)\n"
        "print(os.getpid(), flush=True)\ntime.sleep(120)\n"
    )
    s = make_settings(tmp_path, shutdown_grace_s=0.5)
    s.executor = lambda job, log: run_child(
        [sys.executable, str(script)], dict(os.environ), job, log, s.shutdown_grace_s
    )
    app = create_app(s)
    worker = app.state.worker
    with TestClient(app) as c:
        rid = c.post(
            "/api/runs",
            json={"scale": "tiny", "dry_run": False, "approve_spend": True},
            headers={"X-Admin-Token": ADMIN},
        ).json()["run_id"]
        end = time.monotonic() + 10
        while not (job := worker.get(rid)) or not job.logs:
            assert time.monotonic() < end
            time.sleep(0.02)
        pid = int(job.logs[0])
        t0 = time.monotonic()
    assert time.monotonic() - t0 < 8
    assert not worker._thread.is_alive()
    with pytest.raises(ProcessLookupError):
        os.kill(pid, 0)
    assert worker.get(rid).state == "finished"  # type: ignore[union-attr]


# ---- 7: anonymous dry runs --------------------------------------------------------------------
def _dry_client(tmp_path: Path, gate: threading.Event | None = None, **kw: Any) -> TestClient:
    s = make_settings(tmp_path, **kw)
    s.executor = scripted_executor(s.root, gate)
    return TestClient(create_app(s))


def test_anonymous_dry_runs_only_tiny_unless_admin(tmp_path: Path) -> None:
    with _dry_client(tmp_path) as c:
        r = c.post("/api/runs", json={"scale": "small", "dry_run": True})
        assert r.status_code == 403 and r.json()["error"] == "admin_token_required"
        assert c.post("/api/runs", json={"scale": "tiny", "dry_run": True}).status_code == 202
        ok = c.post(
            "/api/runs", json={"scale": "small", "dry_run": True}, headers={"X-Admin-Token": ADMIN}
        )
        assert ok.status_code == 202
        bad = c.post(
            "/api/runs", json={"scale": "small", "dry_run": True}, headers={"X-Admin-Token": "no"}
        )
        assert bad.status_code == 403


def test_queue_depth_cap_returns_429(tmp_path: Path) -> None:
    gate = threading.Event()
    with _dry_client(tmp_path, gate, max_pending_jobs=2, dry_run_per_ip_per_hour=99) as c:
        body = {"scale": "tiny", "dry_run": True}
        assert c.post("/api/runs", json=body).status_code == 202
        assert c.post("/api/runs", json=body).status_code == 202
        r = c.post("/api/runs", json=body)
        assert r.status_code == 429 and r.json()["error"] == "queue_full"
        assert "retry-after" in r.headers
        gate.set()


def test_forwarded_for_ignored_unless_proxy_trusted(tmp_path: Path) -> None:
    body = {"scale": "tiny", "dry_run": True}
    with _dry_client(tmp_path / "a", dry_run_per_ip_per_hour=1) as c:
        assert (
            c.post("/api/runs", json=body, headers={"X-Forwarded-For": "1.1.1.1"}).status_code
            == 202
        )
        assert (
            c.post("/api/runs", json=body, headers={"X-Forwarded-For": "2.2.2.2"}).status_code
            == 429
        )
    with _dry_client(
        tmp_path / "b", dry_run_per_ip_per_hour=1, trusted_proxies=("testclient",)
    ) as c:
        assert (
            c.post("/api/runs", json=body, headers={"X-Forwarded-For": "1.1.1.1"}).status_code
            == 202
        )
        assert (
            c.post("/api/runs", json=body, headers={"X-Forwarded-For": "2.2.2.2"}).status_code
            == 202
        )
        assert (
            c.post("/api/runs", json=body, headers={"X-Forwarded-For": "2.2.2.2"}).status_code
            == 429
        )


# ---- 8: playground daily cap race -------------------------------------------------------------
class _SlowTransport(FakeTransport):
    async def _create(self, **kw: Any) -> Any:
        await asyncio.sleep(0.05)
        return await super()._create(**kw)


def test_parallel_playground_calls_cannot_exceed_daily_cap(tmp_path: Path) -> None:
    t = _SlowTransport()
    t.chat = SimpleNamespace(completions=SimpleNamespace(create=t._create))
    llm = LLMClient(t, MODELS)  # type: ignore[arg-type]
    probe = create_app(make_settings(tmp_path / "probe", playground_llm=llm))
    cfg = make_config(playground_daily_cap_usd=_cap_for(probe.state.playground))  # 2.5 estimates
    app = create_app(make_settings(tmp_path / "real", config=cfg, playground_llm=llm))
    pg = app.state.playground

    async def go() -> list[Any]:
        calls = (pg.answer("How many accounts are there?") for _ in range(6))
        return await asyncio.gather(*calls, return_exceptions=True)

    res = asyncio.run(go())
    ok = [r for r in res if isinstance(r, dict)]
    denied = [r for r in res if isinstance(r, DemoBudgetExhaustedError)]
    assert len(ok) == 2 and len(denied) == 4
    today = datetime.now(UTC).date().isoformat()
    assert pg.reader.real.spend_on_day(today, "playground") <= cfg.playground_daily_cap_usd
    assert pg._reserved == 0.0


def _cap_for(pg: Any) -> float:
    from distillery.prompts import build_messages
    from distillery.taskpacks.sql import schema as sql_schema

    msgs = build_messages(
        "How many accounts are there?", sql_schema.schema_ddl(0), role="eval_teacher"
    )
    est = pg.ledger.estimate_llm_cost("fake-t", sum(len(m["content"]) for m in msgs) // 3, 256)
    return est * 2.5
