"""Telemetry: request ids, route templates (no raw ids), counters, SSE-safe, nothing secret."""

from __future__ import annotations

import json
import logging
from pathlib import Path

from fastapi.testclient import TestClient
from server_helpers import API_KEY, make_settings, seed_finished_run

from distillery.server import create_app
from distillery.server.telemetry import BUCKETS, Telemetry


def _client(tmp_path: Path) -> TestClient:
    seed_finished_run(tmp_path / "data", "dry-sql-tiny-t1").close()
    return TestClient(create_app(make_settings(tmp_path)))


def test_every_response_has_a_request_id_and_a_safe_inbound_id_is_kept(tmp_path: Path) -> None:
    with _client(tmp_path) as c:
        assert len(c.get("/api/health").headers["x-request-id"]) == 16
        kept = c.get("/api/health", headers={"X-Request-ID": "abc-12345678"})
        assert kept.headers["x-request-id"] == "abc-12345678"
        bad = c.get("/api/health", headers={"X-Request-ID": 'x y"; drop'})
        assert bad.headers["x-request-id"] != 'x y"; drop'


def test_counts_use_route_templates_not_raw_paths(tmp_path: Path) -> None:
    with _client(tmp_path) as c:
        c.get("/api/runs/dry-sql-tiny-t1")
        c.get("/api/runs/dry-sql-tiny-t1/report")
        c.get("/api/runs/no-such-run-xx")
        snap = c.get("/api/telemetry").json()
    routes = set(snap["by_route"])
    assert "/api/runs/{run_id}" in routes and "/api/runs/{run_id}/report" in routes
    assert not any("dry-sql-tiny-t1" in r for r in routes)  # ids never become labels
    assert snap["requests_total"] >= 3
    assert (
        snap["requests_by_status_class"]["2xx"] >= 2
        and snap["requests_by_status_class"]["4xx"] >= 1
    )
    assert snap["runs_by_status"]["complete"] == 1 and snap["in_flight"] >= 0


def test_prometheus_text_is_valid_and_holds_no_secret(tmp_path: Path) -> None:
    with _client(tmp_path) as c:
        c.get("/api/health")
        r = c.get("/api/metrics")
    assert r.headers["content-type"].startswith("text/plain")
    text = r.text
    assert 'distillery_http_requests_total{method="GET",route="/api/health",status="200"}' in text
    assert "distillery_http_request_duration_seconds_bucket" in text and 'le="+Inf"' in text
    assert API_KEY not in text


def test_event_stream_is_counted_but_kept_out_of_latency(tmp_path: Path) -> None:
    with _client(tmp_path) as c:
        c.get("/api/runs/dry-sql-tiny-t1/events")
        snap = c.get("/api/telemetry").json()
    r = snap["by_route"]["/api/runs/{run_id}/events"]
    assert r["requests"] == 1 and "latency_s_mean" not in r  # long-lived by design
    assert snap["event_streams_open"] == 0  # closed again


def test_access_log_line_is_json_without_ip_or_headers(tmp_path: Path, caplog) -> None:  # type: ignore[no-untyped-def]
    logging.getLogger("distillery.access").propagate = True  # let caplog see it
    with caplog.at_level(logging.INFO, logger="distillery.access"), _client(tmp_path) as c:
        c.get("/api/health?token=secret-in-query", headers={"Authorization": "Bearer zzz"})
    lines = [json.loads(r.message) for r in caplog.records if r.name == "distillery.access"]
    line = next(x for x in lines if x["route"] == "/api/health")
    assert set(line) == {"evt", "method", "route", "status", "ms", "stream", "request_id"}
    assert "secret-in-query" not in json.dumps(lines) and "zzz" not in json.dumps(lines)


def test_histogram_buckets_and_quantiles() -> None:
    t = Telemetry()
    for s in (0.005, 0.02, 0.02, 0.4, 7.0, 20.0):
        t.begin(False)
        t.end("GET", "/x", 200, s, False)
    snap = t.snapshot()["by_route"]["/x"]
    assert snap["requests"] == 6 and snap["errors_5xx"] == 0
    assert snap["latency_s_p50_upper_bound"] == 0.05  # 3rd of 6 falls in the <=0.05 bucket
    assert snap["latency_s_p95_upper_bound"] == float(
        "inf"
    )  # the 20 s request is over every bucket
    assert BUCKETS[0] == 0.01
