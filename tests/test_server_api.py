from __future__ import annotations

import json
import threading
import time
from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from server_helpers import (
    ADMIN,
    make_settings,
    sample_report,
    scripted_executor,
    seed_finished_run,
)

from distillery.server import create_app
from distillery.server import sse as sse_mod
from distillery.store import Store


def _client(tmp_path: Path, **kw: object) -> TestClient:
    s = make_settings(tmp_path, **kw)
    s.executor = s.executor or scripted_executor(s.root)
    return TestClient(create_app(s))


@pytest.fixture
def client(tmp_path: Path) -> Iterator[TestClient]:
    with _client(tmp_path) as c:
        yield c


def wait_for(client: TestClient, run_id: str, status: str, timeout: float = 5.0) -> dict:
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        d = client.get(f"/api/runs/{run_id}").json()
        if d["status"] == status:
            return d
        time.sleep(0.02)
    raise AssertionError(f"{run_id} never reached {status}: {d}")


def test_health_and_config_have_no_secrets(client: TestClient) -> None:
    h = client.get("/api/health").json()
    assert h["ok"] is True and h["mode"] == "live" and h["version"]
    c = client.get("/api/config").json()
    assert set(c) == {"models", "thresholds", "run_cap_usd", "playground"}
    assert set(c["models"]) == {"planner", "teacher", "triage", "student"}
    assert set(c["thresholds"]) == {"ratio_lower_bound_min", "mcnemar_alpha", "bootstrap_resamples"}
    assert set(c["playground"]) == {
        "enabled",
        "per_ip_per_hour",
        "daily_cap_usd",
        "spent_today_usd",
        "models",
        "student_daily_cap",
        "student_per_ip_per_hour",
    }
    assert c["playground"]["enabled"] is False  # no LLM injected: honest


def test_replay_only_mode_without_key(tmp_path: Path) -> None:
    from distillery.config import Config

    with _client(tmp_path, config=Config()) as c:
        assert c.get("/api/health").json()["mode"] == "replay-only"


def test_default_replay_is_labelled_dry_run_sample(client: TestClient) -> None:
    items = client.get("/api/replay").json()
    assert len(items) == 1
    it = items[0]
    assert it["dry_run"] is True and it["recorded"] is False and it["recorded_at"] is None
    assert it["decision"] == sample_report()["decision"]
    rid = it["run_id"]
    rep = client.get(f"/api/runs/{rid}/report").json()
    assert rep["dry_run"] is True and rep["recorded"] is False
    assert rep["evaluation"] == sample_report()["evaluation"]  # verbatim
    detail = client.get(f"/api/runs/{rid}").json()
    assert detail["status"] == "complete" and detail["dry_run"] is True
    assert detail["stages"] and all(s["status"] == "done" for s in detail["stages"])
    assert detail["verifier"]["selftest"] is None  # not recorded in a report: not invented


def test_real_bundle_replaces_sample_and_is_recorded(tmp_path: Path) -> None:
    d = tmp_path / "replay" / "sql-full-20260101T000000Z"
    d.mkdir(parents=True)
    rep = sample_report()
    rep["dry_run"] = False
    (d / "report.json").write_text(json.dumps(rep))
    (d / "manifest.json").write_text(json.dumps({"recorded_at": "2026-01-01T00:00:00+00:00"}))
    with _client(tmp_path) as c:
        items = c.get("/api/replay").json()
        assert [i["run_id"] for i in items] == ["sql-full-20260101T000000Z"]
        assert items[0]["recorded"] is True and items[0]["dry_run"] is False
        assert items[0]["recorded_at"] == "2026-01-01T00:00:00+00:00"
        assert c.get("/api/runs/sql-full-20260101T000000Z/report").json()["recorded"] is True


def test_dry_run_lifecycle_and_derived_progress(tmp_path: Path) -> None:
    with _client(tmp_path) as c:
        r = c.post("/api/runs", json={"scale": "tiny", "dry_run": True})
        assert r.status_code == 202
        rid = r.json()["run_id"]
        assert rid.startswith("dry-")
        d = wait_for(c, rid, "complete")
        assert d["dry_run"] is True and d["error"] is None
        assert [s["name"] for s in d["stages"]] == [
            "schema",
            "gold_crosscheck",
            "verifier_selftest",
        ]
        assert d["spend"]["by_model"]["fake-t"]["calls"] == 1
        assert d["spend"]["finetune_usd_estimate"] == 2.0
        assert d["spend"]["total_usd"] == 2.5
        assert d["verifier"] == {
            "language": "sql",
            "code": None,
            "selftest": {"accepted_gold": 5, "rejected_corruptions": 17, "failures": 0},
        }
        runs = c.get("/api/runs").json()
        assert [x["run_id"] for x in runs] == [rid]
        assert runs[0]["status"] == "complete" and runs[0]["decision"] == "REJECT"
        assert runs[0]["dry_run"] is True and runs[0]["recorded"] is False
        rep = c.get(f"/api/runs/{rid}/report").json()
        assert rep["dry_run"] is True and rep["recorded"] is False and rep["recorded_at"] is None


def test_report_not_ready_until_complete(tmp_path: Path) -> None:
    gate = threading.Event()
    s = make_settings(tmp_path)
    s.executor = scripted_executor(s.root, gate)
    with TestClient(create_app(s)) as c:
        rid = c.post("/api/runs", json={"scale": "tiny", "dry_run": True}).json()["run_id"]
        wait_for(c, rid, "running")
        r = c.get(f"/api/runs/{rid}/report")
        assert r.status_code == 404 and r.json()["error"] == "report_not_ready"
        assert c.get(f"/api/runs/{rid}/examples").status_code == 404
        assert c.get(f"/api/runs/{rid}/tree").json() == {"nodes": []}
        gate.set()
        wait_for(c, rid, "complete")


def test_failed_run_reports_error(tmp_path: Path) -> None:
    s = make_settings(tmp_path)

    def boom(job, log):  # type: ignore[no-untyped-def]
        log("refused: nope")
        return 2

    s.executor = boom
    with TestClient(create_app(s)) as c:
        rid = c.post("/api/runs", json={"scale": "tiny", "dry_run": True}).json()["run_id"]
        d = wait_for(c, rid, "failed")
        assert "exit code 2" in d["error"] and "refused: nope" in d["error"]
        events = c.get(f"/api/runs/{rid}/events").text
        assert "event: done" in events and '"status":"failed"' in events


def test_run_driven_by_the_cli_reads_running_not_failed(tmp_path: Path) -> None:
    """No report and no in-process job: running only while the CLI process holds driver.pid."""
    from distillery.driver import driving

    s = make_settings(tmp_path)
    rid = "sql-tiny-cli"
    store = Store(s.root)
    store.create_run(rid)
    store.close()
    run_dir = s.root / "runs" / rid
    with TestClient(create_app(s)) as c:
        with driving(run_dir):
            assert c.get(f"/api/runs/{rid}").json()["status"] == "running"
        d = c.get(f"/api/runs/{rid}").json()
        assert d["status"] == "failed" and "interrupted" in d["error"]


def test_tree_from_real_lineage(client: TestClient, tmp_path: Path) -> None:
    rid = "dry-sql-tiny-abc"
    seed_finished_run(tmp_path / "data", rid).close()
    nodes = client.get(f"/api/runs/{rid}/tree").json()["nodes"]
    rep = sample_report()
    assert len(nodes) == 1 + len(rep["rounds"])  # base + one per round, nothing else
    base, r1, r2 = nodes
    assert base["parent_id"] is None and base["dev_score"] == rep["headroom"]["base_dev_acc"]
    assert r1["parent_id"] == base["id"] and r2["parent_id"] == r1["id"]
    lineage = rep["sandbox_lineage"][0]
    assert r1["id"] == lineage["uuid"] and r1["sandbox_image"] == lineage["uuid"]
    assert base["id"] == lineage["parent"]
    assert r1["dev_score"] == rep["rounds"][0]["dev_acc"]
    assert r1["data_delta"]["added_rows"] == rep["rounds"][0]["targeted_new_rows"]
    assert r2["sandbox_image"] is None and r2["data_delta"] is None  # no invented values
    assert [n["selected"] for n in nodes] == [False, False, True]
    assert all(n["cost_usd"] is None for n in nodes)


def test_examples_absent_then_present(client: TestClient, tmp_path: Path) -> None:
    seed_finished_run(tmp_path / "data", "dry-sql-tiny-a").close()
    r = client.get("/api/runs/dry-sql-tiny-a/examples")
    assert r.status_code == 200 and r.json() == []
    assert r.headers["x-examples-available"] == "false"

    def ex(i: int, base: bool, student: bool) -> dict:
        return {"task_id": f"t{i}", "base_ok": base, "student_ok": student, "teacher_ok": True}

    items = [ex(1, False, True), ex(2, True, False), ex(3, False, False), ex(4, True, True)]
    seed_finished_run(tmp_path / "data", "dry-sql-tiny-b", examples=items).close()

    def ids(kind: str, limit: int = 20) -> list[str]:
        r = client.get(f"/api/runs/dry-sql-tiny-b/examples?kind={kind}&limit={limit}")
        assert r.headers["x-examples-available"] == "true"
        return [e["task_id"] for e in r.json()]

    assert ids("fixed") == ["t1"]
    assert ids("still_wrong") == ["t3"]  # core: base wrong AND student wrong
    assert ids("regressed") == ["t2"]
    assert ids("all") == ["t1", "t2", "t3", "t4"]
    assert ids("all", 2) == ["t1", "t2"]
    assert client.get("/api/runs/dry-sql-tiny-b/examples?kind=bogus").status_code == 422


def test_unknown_run_and_unknown_endpoint(client: TestClient) -> None:
    for path in ("/api/runs/nope", "/api/runs/nope/report", "/api/runs/nope/tree",
                 "/api/runs/nope/events", "/api/nothing"):  # fmt: skip
        r = client.get(path)
        assert r.status_code == 404 and r.json()["error"] == "not_found", path


# ---- SSE -------------------------------------------------------------------
def _parse(text: str) -> list[dict]:
    out = []
    for block in text.split("\n\n"):
        if not block.strip() or block.startswith(":"):
            continue
        rec: dict = {}
        for line in block.splitlines():
            k, _, v = line.partition(": ")
            rec[k] = v
        out.append(rec)
    return out


def test_sse_full_stream_and_resume(client: TestClient, tmp_path: Path) -> None:
    rid = client.post("/api/runs", json={"scale": "tiny", "dry_run": True}).json()["run_id"]
    wait_for(client, rid, "complete")
    r = client.get(f"/api/runs/{rid}/events")
    assert r.headers["content-type"].startswith("text/event-stream")
    ev = _parse(r.text)
    ids = [int(e["id"]) for e in ev]
    assert ids == list(range(1, len(ids) + 1))
    kinds = [e["event"] for e in ev]
    assert kinds[-1] == "done" and kinds.count("done") == 1
    assert {"stage", "spend", "log"} <= set(kinds)
    done = json.loads(ev[-1]["data"])
    assert (
        done["status"] == "complete"
        and done["decision"] == "REJECT"
        and "observed_at" in done
        and "ts" not in done
    )
    stage = json.loads(next(e for e in ev if e["event"] == "stage")["data"])
    assert set(stage) == {"observed_at", "at", "name", "status"}
    # resume: only events after the given id
    r2 = client.get(f"/api/runs/{rid}/events", headers={"Last-Event-ID": str(ids[-2])})
    assert [int(e["id"]) for e in _parse(r2.text)] == [ids[-1]]
    r3 = client.get(f"/api/runs/{rid}/events", headers={"Last-Event-ID": "garbage"})
    assert len(_parse(r3.text)) == len(ids)


def test_sse_heartbeat_when_idle() -> None:
    now = [0.0]
    frames = sse_mod.stream(
        lambda _after: ([], False),
        0,
        heartbeat_s=15.0,
        poll_s=5.0,
        clock=lambda: now[0],
        sleep=lambda dt: now.__setitem__(0, now[0] + dt),
    )
    got = [next(frames) for _ in range(2)]
    assert got == [": heartbeat\n\n", ": heartbeat\n\n"]
    assert now[0] >= 15.0


def test_sse_stream_ends_after_done_and_emits_in_order() -> None:
    evs = [(1, "stage", {"ts": "t", "name": "a", "status": "done"}), (2, "done", {"ts": "t"})]
    out = list(sse_mod.stream(lambda after: ([e for e in evs if e[0] > after], True), 0,
                              heartbeat_s=15, poll_s=0))  # fmt: skip
    assert len(out) == 2 and out[0].startswith("id: 1\nevent: stage\ndata: {")


def test_sse_for_replay_sample(client: TestClient) -> None:
    ev = _parse(client.get("/api/runs/sample-dry-run/events").text)
    assert ev[-1]["event"] == "done"


# ---- worker ----------------------------------------------------------------
def test_second_live_run_conflicts_and_cancel_works(tmp_path: Path) -> None:
    gate = threading.Event()
    s = make_settings(tmp_path)
    s.executor = scripted_executor(s.root, gate)
    h = {"X-Admin-Token": ADMIN}
    body = {"scale": "tiny", "dry_run": False, "approve_spend": True}
    with TestClient(create_app(s)) as c:
        first = c.post("/api/runs", json=body, headers=h)
        assert first.status_code == 202
        rid = first.json()["run_id"]
        wait_for(c, rid, "running")
        second = c.post("/api/runs", json=body, headers=h)
        assert second.status_code == 409 and second.json()["error"] == "run_conflict"
        # dry runs are not blocked by a live run (they queue)
        assert c.post("/api/runs", json={"scale": "tiny", "dry_run": True}).status_code == 202
        assert c.post(f"/api/runs/{rid}/cancel").status_code == 401
        assert c.post(f"/api/runs/{rid}/cancel", headers={"X-Admin-Token": "x"}).status_code == 403
        r = c.post(f"/api/runs/{rid}/cancel", headers=h)
        assert r.status_code == 200 and r.json() == {"status": "cancelling"}
        d = wait_for(c, rid, "failed")
        assert d["error"] == "cancelled"
        assert c.post("/api/runs/nope/cancel", headers=h).status_code == 404


def test_starting_a_live_run_is_admin_gated(tmp_path: Path) -> None:
    body = {"scale": "tiny", "dry_run": False}  # no approve_spend: never actually starts a job
    with _client(tmp_path) as c:
        assert c.post("/api/runs", json=body).status_code == 401
        assert c.post("/api/runs", json=body, headers={"X-Admin-Token": "wrong"}).status_code == 403
        # with the right token the auth check passes and the next check (spend approval) answers
        r = c.post("/api/runs", json=body, headers={"X-Admin-Token": ADMIN})
        assert r.status_code == 400 and r.json()["error"] == "spend_not_approved"
        # anonymous dry runs are only allowed at scale 'tiny'
        small = {"scale": "small", "dry_run": True}
        assert c.post("/api/runs", json=small).status_code == 403
        assert c.get("/api/runs").json() == []  # nothing was started
