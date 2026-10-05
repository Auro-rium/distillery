from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from server_helpers import make_settings, seed_finished_run

from distillery.server import create_app

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("build_evidence", ROOT / "scripts/build_evidence.py")
assert _spec and _spec.loader
build_evidence = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(build_evidence)

COMMIT = "a" * 40
SEALED_KEYS = {"question", "gold_sql", "student_sql", "teacher_sql", "base_sql", "raw", "sql"}


def _walk_keys(v: object) -> set[str]:
    if isinstance(v, dict):
        return set(v) | {k for x in v.values() for k in _walk_keys(x)}
    if isinstance(v, list):
        return {k for x in v for k in _walk_keys(x)}
    return set()


def test_build_is_deterministic_and_pinned() -> None:
    a = build_evidence.build(COMMIT, "2026-01-01T00:00:00+00:00")
    b = build_evidence.build(COMMIT, "2026-01-01T00:00:00+00:00")
    assert json.dumps(a) == json.dumps(b)
    assert a["repo_url"] == "https://github.com/Auro-rium/distillery"
    assert [p["id"] for p in a["proofs"]] == ["W1-P1.11", "W1-P1.12", "W2", "W3", "W4", "W5"]
    for p in a["proofs"]:
        assert p["verdict"] in {"PASS", "FAIL", "PARTIAL"}
        assert p["evidence_url"] == f"{a['repo_url']}/blob/{COMMIT}/{p['evidence_path']}"
        assert (ROOT / p["evidence_path"]).is_file()
    assert len(a["proofs"][0]["series"]) == 10


def test_build_values_come_from_the_evidence_files() -> None:
    by = {p["id"]: p for p in build_evidence.build(COMMIT, "t")["proofs"]}
    assert by["W1-P1.11"]["headline"]["value"] == "64/64"
    assert by["W1-P1.12"]["headline"]["value"] == "142/150"
    assert by["W2"]["metrics"][0]["value"] == "85/86" and by["W2"]["verdict"] == "PARTIAL"
    assert by["W3"]["headline"]["value"] == "1321/1321"
    assert by["W4"]["headline"]["value"] == "PROMOTE"
    assert by["W5"]["headline"]["value"] == "PASS" and by["W5"]["verdict"] == "PARTIAL"


def test_no_sealed_fields() -> None:
    out = build_evidence.build(COMMIT, "t")
    assert not (_walk_keys(out) & SEALED_KEYS)
    text = json.dumps(out).upper()
    assert "SELECT " not in text


def test_committed_evidence_matches_builder() -> None:
    committed = json.loads((ROOT / "deploy/evidence.json").read_text(encoding="utf-8"))
    rebuilt = build_evidence.build(committed["commit"], committed["generated_at"])
    assert committed == rebuilt


def _client(tmp_path: Path) -> TestClient:
    return TestClient(create_app(make_settings(tmp_path)))


def test_evidence_endpoint_serves_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    f = tmp_path / "ev.json"
    f.write_text(json.dumps({"proofs": [], "commit": "x"}))
    monkeypatch.setenv("DISTILLERY_EVIDENCE_FILE", str(f))
    with _client(tmp_path) as c:
        r = c.get("/api/evidence")
        assert r.status_code == 200 and r.json() == {"proofs": [], "commit": "x"}


def test_evidence_endpoint_default_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("DISTILLERY_EVIDENCE_FILE", raising=False)
    with _client(tmp_path) as c:
        r = c.get("/api/evidence")
        assert r.status_code == 200 and len(r.json()["proofs"]) == 6


def test_evidence_endpoint_404_json(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DISTILLERY_EVIDENCE_FILE", str(tmp_path / "missing.json"))
    with _client(tmp_path) as c:
        r = c.get("/api/evidence")
        assert r.status_code == 404
        assert r.json()["error"] == "evidence_unavailable" and r.json()["message"]


def test_experiments_local_filters_names(tmp_path: Path) -> None:
    s = make_settings(tmp_path)
    store = seed_finished_run(s.root, "dry-sql-tiny-exp")
    store.add_experiment("dry-sql-tiny-exp", "finetune_job_started", {"job_id": "j1", "round": 1})
    store.add_experiment("dry-sql-tiny-exp", "serving_image", {"digest": "sha256:abc"})
    store.add_experiment("dry-sql-tiny-exp", "expected_artifact", {"round": 1})
    store.add_experiment("dry-sql-tiny-exp", "round_result", {"secret": "no"})
    store.close()
    with TestClient(create_app(s)) as c:
        r = c.get("/api/runs/dry-sql-tiny-exp/experiments")
        assert r.status_code == 200
        body = r.json()
        assert body["source"] == "store"
        assert [e["name"] for e in body["experiments"]] == [
            "finetune_job_started",
            "serving_image",
            "expected_artifact",
        ]
        assert body["experiments"][0]["data"] == {"job_id": "j1", "round": 1}
        assert body["experiments"][0]["created_at"]


def test_experiments_unknown_and_invalid(tmp_path: Path) -> None:
    with _client(tmp_path) as c:
        assert c.get("/api/runs/nope/experiments").status_code == 404
        assert c.get("/api/runs/bad.id/experiments").status_code in (400, 404)


def test_experiments_bundle(tmp_path: Path) -> None:
    s = make_settings(tmp_path)
    d = s.replay_dir / "rec-1"
    d.mkdir(parents=True)
    report = json.loads((ROOT / "docs/fixtures/sample-dry-run-report.json").read_text())
    (d / "report.json").write_text(json.dumps(report))
    (d / "manifest.json").write_text(json.dumps({"recorded_at": "2026-01-01T00:00:00Z"}))
    with TestClient(create_app(s)) as c:
        r = c.get("/api/runs/rec-1/experiments")
        assert r.json() == {"run_id": "rec-1", "source": "unavailable", "experiments": []}
        (d / "experiments.json").write_text(
            json.dumps(
                [
                    {"name": "finetune_job_closed", "data": {"job_id": "j"}},
                    {"name": "other", "data": {}},
                ]
            )
        )
        r = c.get("/api/runs/rec-1/experiments")
        assert r.json()["source"] == "bundle"
        assert [e["name"] for e in r.json()["experiments"]] == ["finetune_job_closed"]
