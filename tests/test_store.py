# ruff: noqa: S101, S105, S106
import json
import sqlite3
from pathlib import Path
from typing import Any

import pytest

from distillery.store import HeldoutIntegrityError, Store, stage_key


def test_stage_key_canonical() -> None:
    assert stage_key("a", {"x": 1, "y": 2}) == stage_key("a", {"y": 2, "x": 1})
    assert stage_key("a", {"x": 1}) != stage_key("b", {"x": 1})
    assert stage_key("a", {"x": 1}) != stage_key("a", {"x": 2})


def test_get_or_run_idempotent(tmp_path: Path) -> None:
    s = Store(tmp_path)
    calls: list[int] = []

    def fn() -> dict[str, Any]:
        calls.append(1)
        return {"v": [1, 2]}

    assert s.get_or_run("r1", "gen", {"n": 1}, fn) == {"v": [1, 2]}
    assert s.get_or_run("r1", "gen", {"n": 1}, fn) == {"v": [1, 2]}
    assert len(calls) == 1
    # survives reopening the store
    s2 = Store(tmp_path)
    assert s2.get_or_run("r1", "gen", {"n": 1}, fn) == {"v": [1, 2]}
    assert len(calls) == 1
    manifest = json.loads((tmp_path / "runs" / "r1" / "manifest.json").read_text())
    assert manifest["stages"][0]["stage"] == "gen"
    digest = manifest["stages"][0]["output_sha256"]
    assert (tmp_path / "runs" / "r1" / "artifacts" / digest).exists()


def test_crash_not_marked_complete(tmp_path: Path) -> None:
    s = Store(tmp_path)
    attempts: list[int] = []

    def flaky() -> int:
        attempts.append(1)
        if len(attempts) == 1:
            raise RuntimeError("crash")
        return 42

    with pytest.raises(RuntimeError):
        s.get_or_run("r1", "st", {"a": 1}, flaky)
    assert s.get_or_run("r1", "st", {"a": 1}, flaky) == 42  # crashed stage is not complete
    assert s.get_or_run("r1", "st", {"a": 1}, flaky) == 42  # now cached: flaky not called again
    assert len(attempts) == 2


def test_keyboard_interrupt_not_complete(tmp_path: Path) -> None:
    s = Store(tmp_path)

    def fn() -> int:
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        s.get_or_run("r1", "st", {}, fn)
    assert s.get_or_run("r1", "st", {}, lambda: 7) == 7  # interrupted stage reruns


def test_changed_inputs_rerun(tmp_path: Path) -> None:
    s = Store(tmp_path)
    calls: list[int] = []

    def fn() -> int:
        calls.append(1)
        return len(calls)

    assert s.get_or_run("r1", "st", {"a": 1}, fn) == 1
    assert s.get_or_run("r1", "st", {"a": 2}, fn) == 2
    assert s.get_or_run("r1", "st", {"a": 1}, fn) == 1  # earlier result still cached
    assert len(calls) == 2


def test_corrupt_artifact_recomputes(tmp_path: Path) -> None:
    s = Store(tmp_path)
    s.get_or_run("r1", "st", {}, lambda: 1)
    for p in (tmp_path / "runs" / "r1" / "artifacts").iterdir():
        p.write_bytes(b"garbage")
    assert s.get_or_run("r1", "st", {}, lambda: 2) == 2


def test_no_tmp_files_left(tmp_path: Path) -> None:
    s = Store(tmp_path)
    s.get_or_run("r1", "st", {}, lambda: {"a": 1})
    s.seal_heldout("r1", [{"q": 1}])
    assert not list((tmp_path / "runs").rglob("*.tmp"))


def test_heldout_seal_and_load(tmp_path: Path) -> None:
    s = Store(tmp_path)
    items = [{"q": "a", "sql": "select 1"}, {"q": "b", "sql": "select 2"}]
    h = s.seal_heldout("r1", items)
    assert len(h) == 64
    assert s.seal_heldout("r1", items) == h  # idempotent
    assert s.load_heldout("r1") == items
    assert "heldout" in str(s._heldout_path("r1"))
    assert "artifacts" not in str(s._heldout_path("r1"))
    with pytest.raises(HeldoutIntegrityError):
        s.seal_heldout("r1", [{"q": "other"}])
    s._heldout_path("r1").write_bytes(b"[]")  # tamper
    with pytest.raises(HeldoutIntegrityError):
        s.load_heldout("r1")
    with pytest.raises(HeldoutIntegrityError):
        s.load_heldout("nope")


def test_run_id_validation(tmp_path: Path) -> None:
    s = Store(tmp_path)
    for bad in ("../x", "a/b", "", ".hidden"):
        with pytest.raises(ValueError):
            s.run_dir(bad)


def test_tables_and_spend(tmp_path: Path) -> None:
    s = Store(tmp_path)
    s.add_experiment("r1", "e1", {"acc": 0.5})
    assert s.list_experiments("r1") == [("e1", {"acc": 0.5})]
    s.record_llm_call("r1", "m", 1, 2, 0.1)
    s.record_spend("r1", "llm", "m", 0.1, 1, 2, day="2026-01-01")
    s.record_spend("r1", "playground", None, 0.2, day="2026-01-01")
    assert s.total_spend() == pytest.approx(0.3)
    assert s.total_spend("r1", exclude_kind="playground") == pytest.approx(0.1)
    assert s.spend_on_day("2026-01-01", "playground") == pytest.approx(0.2)


def test_stress_seal_and_load(tmp_path: Path) -> None:
    s = Store(tmp_path)
    items = [{"q": "a", "family": "f"}, {"q": "b", "family": "g"}]
    h = s.seal_stress("r1", items)
    assert len(h) == 64
    assert s.seal_stress("r1", items) == h  # idempotent
    assert s.load_stress("r1") == items
    assert Store(tmp_path).load_stress("r1") == items  # survives reopening
    assert "artifacts" not in str(s._stress_path("r1"))


def test_stress_none_when_never_sealed(tmp_path: Path) -> None:
    s = Store(tmp_path)
    assert s.load_stress("nope") is None
    s.seal_heldout("r1", [{"q": 1}])  # a sealed gate set does not imply a stress set
    assert s.load_stress("r1") is None


def test_stress_tamper_and_reseal_detected(tmp_path: Path) -> None:
    s = Store(tmp_path)
    s.seal_stress("r1", [{"q": "a"}])
    with pytest.raises(HeldoutIntegrityError):
        s.seal_stress("r1", [{"q": "other"}])
    s._stress_path("r1").write_bytes(b"[]")
    with pytest.raises(HeldoutIntegrityError):
        s.load_stress("r1")


def test_manifest_has_stress_sha_only_when_sealed(tmp_path: Path) -> None:
    s = Store(tmp_path)
    manifest = tmp_path / "runs" / "r1" / "manifest.json"
    s.seal_heldout("r1", [{"q": 1}])
    assert "stress_sha256" not in json.loads(manifest.read_text())
    h = s.seal_stress("r1", [{"q": 2}])
    m = json.loads(manifest.read_text())
    assert m["stress_sha256"] == h
    assert m["heldout_sha256"] != h


def test_human_seal_round_trip_tamper_and_reseal(tmp_path: Path) -> None:
    s = Store(tmp_path)
    assert s.load_human("nope") is None
    items = [{"task_id": "h-0-aaaa", "question": "q?"}]
    digest = s.seal_human("r1", items)
    assert s.seal_human("r1", items) == digest  # idempotent
    assert Store(tmp_path).load_human("r1") == items
    assert s.load_stress("r1") is None
    with pytest.raises(HeldoutIntegrityError):
        s.seal_human("r1", [{"task_id": "other"}])
    s._human_path("r1").write_bytes(b"[]")
    with pytest.raises(HeldoutIntegrityError):
        s.load_human("r1")


def test_manifest_has_human_sha_only_when_sealed(tmp_path: Path) -> None:
    s = Store(tmp_path)
    manifest = tmp_path / "runs" / "r1" / "manifest.json"
    s.seal_heldout("r1", [{"q": 1}])
    assert "human_sha256" not in json.loads(manifest.read_text())
    h = s.seal_human("r1", [{"q": 2}])
    assert json.loads(manifest.read_text())["human_sha256"] == h


def test_old_db_without_seals_table_opens(tmp_path: Path) -> None:
    s = Store(tmp_path)
    s.seal_heldout("r1", [{"q": 1}])
    s.close()
    db = sqlite3.connect(tmp_path / "index.sqlite")
    db.execute("DROP TABLE seals")
    db.commit()
    db.close()
    s2 = Store(tmp_path)  # CREATE TABLE IF NOT EXISTS recreates it
    assert s2.load_heldout("r1") == [{"q": 1}]
    assert s2.load_stress("r1") is None
    s2.seal_stress("r1", [{"q": 3}])
    assert s2.load_stress("r1") == [{"q": 3}]


def test_router_split_seal_round_trip_tamper_and_manifest(tmp_path: Path) -> None:
    s = Store(tmp_path)
    assert s.load_router_split("nope", "cal") is None and s.router_split_sha("nope", "cal") is None
    items = [{"task_id": "t1", "question": "q?"}]
    digest = s.seal_router_split("r1", "cal", items)
    assert s.seal_router_split("r1", "cal", items) == digest  # idempotent
    assert Store(tmp_path).load_router_split("r1", "cal") == items
    assert s.router_split_sha("r1", "cal") == digest
    assert s.load_router_split("r1", "test") is None
    other = s.seal_router_split("r1", "test", [{"task_id": "t2"}])
    assert other != digest
    m = json.loads((tmp_path / "runs" / "r1" / "manifest.json").read_text())
    assert (m["router_cal_sha256"], m["router_test_sha256"]) == (digest, other)
    rows = s._query("SELECT sha256 FROM seals WHERE run_id=? AND name=?", ("r1", "router_cal"))
    assert rows == [(digest,)]
    with pytest.raises(HeldoutIntegrityError):
        s.seal_router_split("r1", "cal", [{"task_id": "changed"}])
    s._router_path("r1", "cal").write_bytes(b"[]")
    with pytest.raises(HeldoutIntegrityError):
        s.load_router_split("r1", "cal")
    with pytest.raises(ValueError):
        s.seal_router_split("r1", "../x", [])
    # the router seals do not disturb the other sealed sets
    assert s.load_human("r1") is None and s.load_stress("r1") is None


def test_completed_stage_output_newest(tmp_path: Path) -> None:
    s = Store(tmp_path)
    assert s.completed_stage_output("r1", "split") is None
    s.get_or_run("r1", "split", {"v": 1}, lambda: {"a": 1})
    s.get_or_run("r1", "split", {"v": 2}, lambda: {"a": 2})
    assert s.completed_stage_output("r1", "split") == {"a": 2}
