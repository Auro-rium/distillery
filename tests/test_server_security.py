from __future__ import annotations

import logging
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from server_helpers import (
    ADMIN,
    API_KEY,
    fake_llm,
    make_settings,
    scripted_executor,
    seed_finished_run,
)

from distillery.server import create_app

LIVE = {"scale": "tiny", "dry_run": False, "approve_spend": True}


def _app(tmp_path: Path, **kw: object) -> TestClient:
    s = make_settings(tmp_path, **kw)
    s.executor = scripted_executor(s.root)
    return TestClient(create_app(s))


def test_live_run_auth(tmp_path: Path) -> None:
    with _app(tmp_path) as c:
        r = c.post("/api/runs", json=LIVE)
        assert r.status_code == 401 and r.json()["error"] == "admin_token_required"
        r = c.post("/api/runs", json=LIVE, headers={"X-Admin-Token": "wrong"})
        assert r.status_code == 403 and r.json()["error"] == "forbidden"
        r = c.post(
            "/api/runs", json={**LIVE, "approve_spend": False}, headers={"X-Admin-Token": ADMIN}
        )
        assert r.status_code == 400 and r.json()["error"] == "spend_not_approved"
        r = c.post("/api/runs", json={**LIVE, "budget_usd": 1e9}, headers={"X-Admin-Token": ADMIN})
        assert r.status_code == 422
        assert c.get("/api/runs").json() == []  # nothing was started


def test_no_admin_token_configured_means_no_live_runs(tmp_path: Path) -> None:
    from server_helpers import make_config

    with _app(tmp_path, config=make_config(admin_token=None)) as c:
        r = c.post("/api/runs", json=LIVE, headers={"X-Admin-Token": ""})
        assert r.status_code == 401
        r = c.post("/api/runs", json=LIVE, headers={"X-Admin-Token": "anything"})
        assert r.status_code == 403


@pytest.mark.parametrize(
    "rid",
    [
        "../etc/passwd",
        "..",
        "a/b",
        "a b",
        "-x",
        ".hidden",
        "x" * 65,
        "playground",
        "a;rm",
        "%2e%2e",
    ],
)
def test_run_id_validation(tmp_path: Path, rid: str) -> None:
    with _app(tmp_path) as c:
        for suffix in ("", "/report", "/tree", "/examples", "/events"):
            r = c.get(f"/api/runs/{rid}{suffix}")
            assert r.status_code in (400, 404), (rid, suffix, r.status_code)
        r = c.post("/api/runs", json={"scale": "tiny", "dry_run": True, "run_id": rid})
        assert r.status_code in (422,), rid
        r = c.post(f"/api/runs/{rid}/cancel", headers={"X-Admin-Token": ADMIN})
        assert r.status_code in (400, 404)
    assert not (tmp_path / "data" / "runs" / "etc").exists()


def test_run_id_prefix_rules(tmp_path: Path) -> None:
    with _app(tmp_path) as c:
        r = c.post("/api/runs", json={"scale": "tiny", "dry_run": True, "run_id": "real-1"})
        assert r.status_code == 422
        r = c.post("/api/runs", json={**LIVE, "run_id": "dry-x"}, headers={"X-Admin-Token": ADMIN})
        assert r.status_code == 422


def test_completed_run_cannot_be_restarted(tmp_path: Path) -> None:
    with _app(tmp_path) as c:
        seed_finished_run(tmp_path / "data", "dry-sql-tiny-z").close()
        r = c.post("/api/runs", json={"scale": "tiny", "dry_run": True, "run_id": "dry-sql-tiny-z"})
        assert r.status_code == 409 and r.json()["error"] == "run_exists"


def test_body_size_cap(tmp_path: Path) -> None:
    with _app(tmp_path, max_body_bytes=200) as c:
        big = '{"scale":"tiny","dry_run":true,"run_id":"dry-' + "a" * 500 + '"}'
        r = c.post("/api/runs", content=big, headers={"content-type": "application/json"})
        assert r.status_code == 413 and r.json()["error"] == "payload_too_large"
        # chunked (no content-length) is capped too
        r = c.post("/api/playground", content=iter([b'{"question":"', b"a" * 300, b'"}']))
        assert r.status_code == 413
        r = c.post("/api/runs", content="not json", headers={"content-type": "application/json"})
        assert r.status_code == 400 and r.json()["error"] == "invalid_json"
        r = c.post("/api/runs", json={"scale": "huge", "dry_run": True})
        assert r.status_code == 422
        r = c.post("/api/runs", json={"scale": "tiny", "dry_run": True, "evil": 1})
        assert r.status_code == 422


def test_dry_run_rate_limit_per_ip(tmp_path: Path) -> None:
    now = [0.0]
    with _app(tmp_path, dry_run_per_ip_per_hour=2, clock=lambda: now[0]) as c:
        body = {"scale": "tiny", "dry_run": True}
        assert c.post("/api/runs", json=body).status_code == 202
        assert c.post("/api/runs", json=body).status_code == 202
        r = c.post("/api/runs", json=body)
        assert r.status_code == 429 and r.json()["error"] == "rate_limited"
        assert int(r.headers["retry-after"]) >= 1
        now[0] += 3601
        assert c.post("/api/runs", json=body).status_code == 202


def test_cors_same_origin_unless_dev_origin(tmp_path: Path) -> None:
    hdr = {"Origin": "http://localhost:5173"}
    with _app(tmp_path) as c:
        assert "access-control-allow-origin" not in c.get("/api/health", headers=hdr).headers
    with _app(tmp_path, dev_origin="http://localhost:5173") as c:
        r = c.get("/api/health", headers=hdr)
        assert r.headers["access-control-allow-origin"] == "http://localhost:5173"
        r = c.get("/api/health", headers={"Origin": "http://evil.example"})
        assert "access-control-allow-origin" not in r.headers


def test_no_secret_in_any_response_or_log(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.DEBUG)
    llm, _ = fake_llm()
    with _app(tmp_path, playground_llm=llm) as c:
        h = {"X-Admin-Token": ADMIN}
        rid = c.post("/api/runs", json={**LIVE, "run_id": "sql-tiny-sec"}, headers=h).json()[
            "run_id"
        ]
        for _ in range(200):
            if c.get(f"/api/runs/{rid}").json()["status"] == "complete":
                break
            time.sleep(0.02)
        texts = []
        paths = ["/api/health", "/api/config", "/api/runs", "/api/replay", f"/api/runs/{rid}",
                 f"/api/runs/{rid}/report", f"/api/runs/{rid}/tree", f"/api/runs/{rid}/examples",
                 f"/api/runs/{rid}/events", "/api/nope", "/api/runs/../x"]  # fmt: skip
        for p in paths:
            r = c.get(p)
            texts.append(r.text + str(dict(r.headers)))
        for r in (
            c.post("/api/runs", json=LIVE, headers={"X-Admin-Token": "bad" + ADMIN}),
            c.post("/api/playground", json={"question": "How many accounts are there?"}),
            c.post(f"/api/runs/{rid}/cancel", headers=h),
        ):
            texts.append(r.text + str(dict(r.headers)))
    blob = "\n".join(texts) + "\n".join(rec.getMessage() for rec in caplog.records)
    assert ADMIN not in blob and API_KEY not in blob
    assert "[redacted]" in texts[texts.index(next(t for t in texts if "starting" in t))]
