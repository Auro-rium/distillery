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


def _set_artifact(root: Path, run_id: str, sha: str) -> None:
    """Make the finished (real) run's report carry a different trained artifact."""
    path = root / "runs" / run_id / "report.json"
    report = json.loads(path.read_text())
    report["evaluation"]["artifact"] = {**report["evaluation"]["artifact"], "adapter_sha256": sha}
    path.write_text(json.dumps(report))


def _bundle_bytes(d: Path) -> dict[str, bytes]:
    return {p.name: p.read_bytes() for p in sorted(d.iterdir())}


def test_export_default_bundle_has_no_bundle_id(tmp_path: Path) -> None:
    seed_finished_run(tmp_path / "data", "sql-tiny-real", dry=False).close()
    assert (
        cli.main(["export-replay", "sql-tiny-real"], env=_env(tmp_path), out=lambda _s: None) == 0
    )
    d = tmp_path / "replay" / "sql-tiny-real"
    manifest = json.loads((d / "manifest.json").read_text())
    assert "bundle_id" not in manifest and manifest["source_run_id"] == "sql-tiny-real"
    assert json.loads((d / "run.json").read_text())["run_id"] == "sql-tiny-real"
    assert json.loads((d / "report.json").read_text())["run_id"] == "sql-tiny-real"


def test_export_bundle_id_sits_beside_the_original(tmp_path: Path) -> None:
    root = tmp_path / "data"
    seed_finished_run(root, "sql-tiny-real", dry=False).close()
    env = _env(tmp_path)
    assert cli.main(["export-replay", "sql-tiny-real"], env=env, out=lambda _s: None) == 0
    first = _bundle_bytes(tmp_path / "replay" / "sql-tiny-real")
    # a later round that reuses the run id and trains a different artifact
    _set_artifact(root, "sql-tiny-real", "b" * 64)
    out: list[str] = []
    args = ["export-replay", "sql-tiny-real", "--bundle-id", "sql-tiny-n1"]
    assert cli.main(args, env=env, out=out.append) == 0
    assert out[0].endswith(str(tmp_path / "replay" / "sql-tiny-n1"))
    assert _bundle_bytes(tmp_path / "replay" / "sql-tiny-real") == first  # not touched
    d = tmp_path / "replay" / "sql-tiny-n1"
    manifest = json.loads((d / "manifest.json").read_text())
    assert manifest["run_id"] == manifest["bundle_id"] == "sql-tiny-n1"
    assert manifest["source_run_id"] == "sql-tiny-real"
    for name in ("report.json", "run.json"):
        assert json.loads((d / name).read_text())["run_id"] == "sql-tiny-n1"
    # the source id survives only where it is meant to: the manifest's provenance field
    for f in d.iterdir():
        text = f.read_text().replace('"source_run_id": "sql-tiny-real"', "")
        assert "sql-tiny-real" not in text, f.name
    # serve from an EMPTY data dir: both bundles are listed, each under its own id
    s = make_settings(tmp_path / "other")
    s.replay_dir = tmp_path / "replay"
    with TestClient(create_app(s)) as c:
        assert sorted(i["run_id"] for i in c.get("/api/replay").json()) == [
            "sql-tiny-n1",
            "sql-tiny-real",
        ]
        assert {i["run_id"] for i in c.get("/api/runs").json()} == {"sql-tiny-n1", "sql-tiny-real"}
        assert c.get("/api/runs/sql-tiny-n1").json()["run_id"] == "sql-tiny-n1"
        rep = c.get("/api/runs/sql-tiny-n1/report").json()
        assert rep["run_id"] == "sql-tiny-n1" and rep["recorded"] is True
        assert c.get("/api/runs/sql-tiny-n1/experiments").json()["run_id"] == "sql-tiny-n1"
        old = c.get("/api/runs/sql-tiny-real/report").json()
        assert old["run_id"] == "sql-tiny-real"
        assert old["evaluation"]["artifact"]["adapter_sha256"] != "b" * 64


def test_export_never_overwrites_a_different_artifact(tmp_path: Path) -> None:
    root = tmp_path / "data"
    seed_finished_run(root, "sql-tiny-real", dry=False).close()
    env = _env(tmp_path)
    assert cli.main(["export-replay", "sql-tiny-real"], env=env, out=lambda _s: None) == 0
    before = _bundle_bytes(tmp_path / "replay" / "sql-tiny-real")
    _set_artifact(root, "sql-tiny-real", "b" * 64)
    # without --bundle-id the run id names the bundle: that would overwrite the other artifact
    out: list[str] = []
    assert cli.main(["export-replay", "sql-tiny-real"], env=env, out=out.append) == cli.EXIT_REFUSED
    assert "different trained artifact" in out[0] and "--bundle-id" in out[0]
    # and so would an explicit id that points at the occupied directory
    args = ["export-replay", "sql-tiny-real", "--bundle-id", "sql-tiny-real"]
    assert cli.main(args, env=env, out=lambda _s: None) == cli.EXIT_REFUSED
    assert _bundle_bytes(tmp_path / "replay" / "sql-tiny-real") == before
    # the same artifact may refresh its own bundle
    args = ["export-replay", "sql-tiny-real", "--bundle-id", "sql-tiny-n1"]
    assert cli.main(args, env=env, out=lambda _s: None) == 0
    assert cli.main(args, env=env, out=lambda _s: None) == 0
    # a different artifact may not take over that custom bundle either
    _set_artifact(root, "sql-tiny-real", "c" * 64)
    assert cli.main(args, env=env, out=lambda _s: None) == cli.EXIT_REFUSED


@pytest.mark.parametrize("bad", ["../x", "bad.id", "playground", "x" * 65, "-lead", "humanset-abc"])
def test_export_refuses_invalid_bundle_id(tmp_path: Path, bad: str) -> None:
    seed_finished_run(tmp_path / "data", "sql-tiny-real", dry=False).close()
    out: list[str] = []
    args = ["export-replay", "sql-tiny-real", f"--bundle-id={bad}"]  # "=": "-lead" is not an option
    assert cli.main(args, env=_env(tmp_path), out=out.append) == cli.EXIT_REFUSED
    assert "invalid bundle id" in out[0]
    assert not (tmp_path / "replay").exists()


def test_export_bundle_id_must_agree_on_the_dry_prefix(tmp_path: Path) -> None:
    seed_finished_run(tmp_path / "data", "sql-tiny-real", dry=False).close()
    seed_finished_run(tmp_path / "data", "dry-sql-tiny-e").close()
    env = _env(tmp_path)
    out: list[str] = []
    args = ["export-replay", "sql-tiny-real", "--bundle-id", "dry-sql-tiny-real"]
    assert cli.main(args, env=env, out=out.append) == cli.EXIT_REFUSED and "prefix" in out[0]
    args = ["export-replay", "dry-sql-tiny-e", "--allow-dry-run", "--bundle-id", "sql-tiny-e"]
    assert cli.main(args, env=env, out=out.append) == cli.EXIT_REFUSED and "prefix" in out[1]
    assert not (tmp_path / "replay").exists()
    args = ["export-replay", "dry-sql-tiny-e", "--allow-dry-run", "--bundle-id", "dry-sql-tiny-x"]
    assert cli.main(args, env=env, out=out.append) == 0
    assert (tmp_path / "replay" / "dry-sql-tiny-x" / "report.json").is_file()


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
