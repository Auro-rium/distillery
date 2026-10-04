from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from server_helpers import ADMIN, make_settings, seed_finished_run

from distillery import cli
from distillery.server import create_app
from distillery.server.reader import RunReader
from distillery.server.replay import ExportError, export_bundle


def _export(tmp_path: Path, run_id: str) -> None:
    env = {
        "DISTILLERY_HOME": str(tmp_path / "data"),
        "DISTILLERY_REPLAY_DIR": str(tmp_path / "replay"),
    }
    assert cli.main(["export-replay", run_id], env=env, out=lambda _s: None) == 0


def _bundle_only_client(tmp_path: Path) -> TestClient:
    s = make_settings(tmp_path / "empty")
    s.replay_dir = tmp_path / "replay"
    return TestClient(create_app(s))


def test_runs_lists_recorded_bundles_and_dedupes_local(tmp_path: Path) -> None:
    seed_finished_run(tmp_path / "data", "sql-tiny-rec", dry=False).close()
    _export(tmp_path, "sql-tiny-rec")
    with _bundle_only_client(tmp_path) as c:
        items = c.get("/api/runs").json()
        assert [i["run_id"] for i in items] == ["sql-tiny-rec"]
        assert items[0]["recorded"] is True and items[0]["recorded_at"]
        assert items[0]["dry_run"] is False and items[0]["status"] == "complete"
    s = make_settings(tmp_path)  # the same run is also local: listed once, as local
    s.replay_dir = tmp_path / "replay"
    with TestClient(create_app(s)) as c:
        items = c.get("/api/runs").json()
        assert [i["run_id"] for i in items] == ["sql-tiny-rec"]
        assert items[0]["recorded"] is False


def test_runs_does_not_list_the_unrecorded_sample(tmp_path: Path) -> None:
    with TestClient(create_app(make_settings(tmp_path))) as c:
        assert c.get("/api/runs").json() == []
        assert [i["run_id"] for i in c.get("/api/replay").json()] == ["sample-dry-run"]


def test_every_endpoint_serves_a_bundle_alone(tmp_path: Path) -> None:
    seed_finished_run(tmp_path / "data", "sql-tiny-rec", dry=False).close()
    _export(tmp_path, "sql-tiny-rec")
    with _bundle_only_client(tmp_path) as c:
        base = "/api/runs/sql-tiny-rec"
        assert c.get(base).json()["recorded"] is True
        assert c.get(f"{base}/report").json()["recorded"] is True
        assert c.get(f"{base}/tree").status_code == 200
        assert "event: done" in c.get(f"{base}/events").text
        ex = c.get(f"{base}/examples")
        assert ex.status_code == 200 and isinstance(ex.json(), list)
        assert ex.headers["X-Examples-Available"] in ("true", "false")
        assert "X-Examples-Totals" in ex.headers
        assert c.get(f"{base}/examples?kind=bogus").status_code == 422
        assert c.get("/api/runs/nope/examples").status_code == 404
        assert c.post(f"{base}/cancel", headers={"X-Admin-Token": ADMIN}).status_code == 404


def test_export_rewrites_spot_check_path_and_refuses_absolute_paths(tmp_path: Path) -> None:
    root = tmp_path / "data"
    store = seed_finished_run(root, "sql-tiny-abs", dry=False)
    rp = store.run_dir("sql-tiny-abs") / "report.json"
    rep = json.loads(rp.read_text())
    rep["data"]["spot_check_file"] = "/home/someone/runs/sql-tiny-abs/spot_check.json"
    rp.write_text(json.dumps(rep))
    store.close()
    s = make_settings(tmp_path)
    reader = RunReader(s)
    dest = export_bundle(
        reader,
        "sql-tiny-abs",
        tmp_path / "replay",
        allow_dry_run=False,
        recorded_at="2026-01-01T00:00:00Z",
    )
    assert (
        json.loads((dest / "report.json").read_text())["data"]["spot_check_file"]
        == "spot_check.json"
    )
    rep["config"]["leak"] = "see /tmp/x/y.json"
    rp.write_text(json.dumps(rep))
    with pytest.raises(ExportError, match="absolute path"):
        export_bundle(
            reader,
            "sql-tiny-abs",
            tmp_path / "replay2",
            allow_dry_run=False,
            recorded_at="2026-01-01T00:00:00Z",
        )
    assert not (tmp_path / "replay2").exists()
    reader.close()


def test_detail_serves_finetune_billed_and_ceiling_apart(tmp_path: Path) -> None:
    root = tmp_path / "data"
    store = seed_finished_run(root, "sql-tiny-ft", dry=False)  # seeds one 2.0 'finetune' row
    store.record_spend("sql-tiny-ft", "finetune_ceiling", None, 3.0)
    store.close()
    sp = RunReader(make_settings(tmp_path)).detail("sql-tiny-ft")["spend"]
    assert sp["finetune_billed_usd"] == 2.0
    assert sp["finetune_ceiling_usd"] == 3.0
    assert sp["finetune_usd_estimate"] == 5.0
