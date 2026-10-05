from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from server_helpers import make_config, make_settings, sample_report, seed_finished_run

from distillery import cli
from distillery.server import create_app


def _env(tmp_path: Path) -> dict[str, str]:
    return {
        "DISTILLERY_HOME": str(tmp_path / "data"),
        "DISTILLERY_REPLAY_DIR": str(tmp_path / "replay"),
    }


def test_export_refuses_dry_run_without_flag(tmp_path: Path) -> None:
    seed_finished_run(tmp_path / "data", "dry-sql-tiny-e").close()
    out: list[str] = []
    code = cli.main(["export-replay", "dry-sql-tiny-e"], env=_env(tmp_path), out=out.append)
    assert code == cli.EXIT_REFUSED and "dry run" in out[0]
    assert not (tmp_path / "replay").exists()


def test_export_dry_run_with_flag_stays_labelled_dry(tmp_path: Path) -> None:
    seed_finished_run(tmp_path / "data", "dry-sql-tiny-e").close()
    out: list[str] = []
    args = ["export-replay", "dry-sql-tiny-e", "--allow-dry-run"]
    assert cli.main(args, env=_env(tmp_path), out=out.append) == 0
    d = tmp_path / "replay" / "dry-sql-tiny-e"
    assert {p.name for p in d.iterdir()} == {
        "report.json", "manifest.json", "run.json", "events.json", "tree.json", "experiments.json"
    }  # fmt: skip
    manifest = json.loads((d / "manifest.json").read_text())
    assert manifest["recorded_at"] and manifest["source_run_id"] == "dry-sql-tiny-e"
    with TestClient(create_app(make_settings(tmp_path))) as c:
        items = c.get("/api/replay").json()
        by_id = {i["run_id"]: i for i in items}
        assert by_id["dry-sql-tiny-e"]["dry_run"] is True
        assert by_id["dry-sql-tiny-e"]["recorded_at"] == manifest["recorded_at"]
        # a dry-only replay dir still shows the labelled sample too
        assert "sample-dry-run" in by_id
        detail = c.get("/api/runs/dry-sql-tiny-e").json()  # local run wins over bundle
        assert detail["status"] == "complete"
        ev = c.get("/api/runs/sample-dry-run/events").text
        assert "event: done" in ev


def test_export_real_run_then_served_from_bundle_alone(tmp_path: Path) -> None:
    seed_finished_run(tmp_path / "data", "sql-tiny-real", dry=False).close()
    out: list[str] = []
    assert cli.main(["export-replay", "sql-tiny-real"], env=_env(tmp_path), out=out.append) == 0
    # serve from an EMPTY data dir: only the bundle exists
    s = make_settings(tmp_path / "other")
    s.replay_dir = tmp_path / "replay"
    with TestClient(create_app(s)) as c:
        items = c.get("/api/replay").json()
        assert [i["run_id"] for i in items] == ["sql-tiny-real"]  # sample hidden: a real one exists
        assert items[0]["recorded"] is True and items[0]["dry_run"] is False
        assert items[0]["recorded_at"]
        rep = c.get("/api/runs/sql-tiny-real/report").json()
        assert rep["recorded"] is True and rep["recorded_at"] == items[0]["recorded_at"]
        d = c.get("/api/runs/sql-tiny-real").json()
        assert d["recorded"] is True and d["status"] == "complete"
        assert d["spend"]["by_model"]["fake-t"]["calls"] == 1
        assert len(c.get("/api/runs/sql-tiny-real/tree").json()["nodes"]) == 1 + len(
            sample_report()["rounds"]
        )
        assert "event: done" in c.get("/api/runs/sql-tiny-real/events").text


def test_export_errors(tmp_path: Path) -> None:
    out: list[str] = []
    assert (
        cli.main(["export-replay", "nope"], env=_env(tmp_path), out=out.append) == cli.EXIT_REFUSED
    )
    assert (
        cli.main(["export-replay", "../x"], env=_env(tmp_path), out=out.append) == cli.EXIT_REFUSED
    )
    assert not (tmp_path / "replay").exists()


def test_spa_fallback_and_api_404(tmp_path: Path) -> None:
    dist = tmp_path / "dist"
    (dist / "assets").mkdir(parents=True)
    (dist / "index.html").write_text("<html>spa</html>")
    (dist / "assets" / "a.js").write_text("console.log(1)")
    (tmp_path / "secret.txt").write_text("nope")
    s = make_settings(tmp_path, config=make_config(), frontend_dist=dist)
    with TestClient(create_app(s)) as c:
        assert c.get("/").text == "<html>spa</html>"
        assert c.get("/replay/whatever").text == "<html>spa</html>"
        assert c.get("/assets/a.js").text == "console.log(1)"
        assert c.get("/../secret.txt").text == "<html>spa</html>"
        assert c.get("/%2e%2e/secret.txt").text == "<html>spa</html>"
        assert c.get("/api/health").json()["ok"] is True
        r = c.get("/api/unknown")
        assert r.status_code == 404 and r.json()["error"] == "not_found"


def test_no_static_when_dist_missing(tmp_path: Path) -> None:
    with TestClient(create_app(make_settings(tmp_path))) as c:
        assert c.get("/").status_code == 404


@pytest.mark.parametrize("cmd", ["serve", "export-replay"])
def test_cli_subcommands_registered(cmd: str) -> None:
    assert cli._parser().parse_args([cmd] + (["x"] if cmd == "export-replay" else [])).cmd == cmd
