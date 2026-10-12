# ruff: noqa: S101
"""The cross-process global spend ledger. Hermetic: every ledger lives under tmp_path (the
autouse fixture in conftest.py points DISTILLERY_GLOBAL_LEDGER at ``none`` otherwise)."""

from __future__ import annotations

import multiprocessing
import os
import sqlite3
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest
from server_helpers import make_config, make_settings

from distillery.budget import PLAYGROUND, BudgetExceeded, Ledger
from distillery.cli import main
from distillery.config import Config, Price
from distillery.global_ledger import (
    ENV_VAR,
    GlobalLedger,
    GlobalLedgerError,
    load_seed_file,
    resolve_path,
)
from distillery.server import create_app
from distillery.store import Store

REPO = Path(__file__).resolve().parent.parent
SEED_FILE = REPO / "deploy" / "ledger_seed.json"
JOIN_S = 120
PRICES = {"m": Price(input_per_mtok=2.0, output_per_mtok=8.0, source="test", date="2026-01-01")}


# ---- multiprocessing workers (module level: 'spawn' must be able to import them) ---------------
def _reserve_worker(path: str, i: int, barrier: Any, results: Any) -> None:
    gl = GlobalLedger(path)
    barrier.wait(timeout=JOIN_S)
    try:
        gl.reserve(f"holder-{i}", 1.0, 5.0)
    except BudgetExceeded:
        results.put((i, "exceeded"))
    else:
        results.put((i, "reserved"))


def _record_worker(path: str, i: int, per_proc: int, barrier: Any) -> None:
    gl = GlobalLedger(path)
    barrier.wait(timeout=JOIN_S)
    for k in range(per_proc):
        gl.record(f"/store/{i}", f"run-{i}", "llm", "m", 0.001 * (k + 1), 10, 5)


def _spawn(target: Any, argsets: list[tuple[Any, ...]]) -> list[Any]:
    ctx = multiprocessing.get_context("spawn")
    procs = [ctx.Process(target=target, args=a) for a in argsets]
    for p in procs:
        p.start()
    return procs


def _join_all(procs: list[Any]) -> None:
    for p in procs:
        p.join(JOIN_S)
    hung = [p.pid for p in procs if p.is_alive()]
    for p in procs:
        if p.is_alive():
            p.kill()
    assert not hung, f"worker processes hung: {hung}"
    assert [p.exitcode for p in procs] == [0] * len(procs)


# ---- the shared file ----------------------------------------------------------------------------
def test_concurrent_reserve_admits_exactly_the_cap(tmp_path: Path) -> None:
    path = tmp_path / "g.sqlite"
    gl = GlobalLedger(path)
    ctx = multiprocessing.get_context("spawn")
    barrier, results = ctx.Barrier(8), ctx.Queue()
    procs = _spawn(_reserve_worker, [(str(path), i, barrier, results) for i in range(8)])
    got = [results.get(timeout=JOIN_S) for _ in range(8)]
    _join_all(procs)
    outcome = [r for _, r in got]
    assert outcome.count("reserved") == 5 and outcome.count("exceeded") == 3
    assert gl.total_committed() == pytest.approx(5.0, abs=1e-9)
    assert len(gl.summary()["reservations"]) == 5


def test_concurrent_record_totals_the_exact_sum(tmp_path: Path) -> None:
    path = tmp_path / "g.sqlite"
    gl = GlobalLedger(path)
    n_proc, per_proc = 6, 25
    ctx = multiprocessing.get_context("spawn")
    barrier = ctx.Barrier(n_proc)
    procs = _spawn(_record_worker, [(str(path), i, per_proc, barrier) for i in range(n_proc)])
    _join_all(procs)
    expected = n_proc * sum(0.001 * (k + 1) for k in range(per_proc))
    assert gl.total_committed() == pytest.approx(expected, abs=1e-9)
    s = gl.summary()
    assert sum(k["n"] for k in s["by_kind"]) == n_proc * per_proc
    assert len(s["by_run"]) == n_proc


def test_reserve_refusal_writes_nothing_and_release_frees(tmp_path: Path) -> None:
    gl = GlobalLedger(tmp_path / "g.sqlite")
    rid = gl.reserve("a", 3.0, 5.0, note="three")
    with pytest.raises(BudgetExceeded, match="project cap"):
        gl.reserve("b", 2.5, 5.0)
    assert gl.total_committed() == pytest.approx(3.0)
    gl.reserve("c", 2.0, 5.0)  # exactly the cap is allowed
    assert gl.release(rid) is True and gl.release(rid) is False
    assert gl.total_committed() == pytest.approx(2.0)
    with pytest.raises(ValueError):
        gl.reserve("d", -1.0, 5.0)
    with pytest.raises(ValueError):
        gl.reserve("d", float("nan"), 5.0)


def test_expired_reservations_do_not_count_and_null_is_permanent(tmp_path: Path) -> None:
    now = [datetime(2026, 10, 12, 12, 0, 0, tzinfo=UTC)]  # microsecond == 0 on purpose
    gl = GlobalLedger(tmp_path / "g.sqlite", clock=lambda: now[0])
    gl.reserve("ttl", 2.0, 10.0, ttl_s=60)
    gl.reserve("forever", 1.0, 10.0)
    assert gl.total_committed() == pytest.approx(3.0)
    now[0] += timedelta(seconds=59)
    assert gl.total_committed() == pytest.approx(3.0)
    now[0] += timedelta(seconds=2)
    assert gl.total_committed() == pytest.approx(1.0)
    assert [r["holder"] for r in gl.summary()["reservations"]] == ["forever"]
    now[0] += timedelta(days=3650)
    assert gl.total_committed() == pytest.approx(1.0)
    gl.reserve("fits", 9.0, 10.0)  # the expired hold no longer blocks


def test_record_settles_its_reservation(tmp_path: Path) -> None:
    gl = GlobalLedger(tmp_path / "g.sqlite")
    rid = gl.reserve("call", 1.0, 5.0)
    gl.record("/s", "r", "llm", "m", 0.4, 1, 1, reservation_id=rid)
    assert gl.total_committed() == pytest.approx(1.0)  # 0.4 spent + 0.6 still held
    assert gl.summary()["reservations"][0]["usd"] == pytest.approx(0.6)
    gl.record("/s", "r", "llm", "m", 0.6, 1, 1, reservation_id=rid)
    assert gl.summary()["reservations"] == []  # used up: released
    assert gl.total_committed() == pytest.approx(1.0)
    rid2 = gl.reserve("call2", 1.0, 5.0)
    gl.record("/s", "r", "llm", "m", 0.1, 1, 1, reservation_id=rid2, release_remainder=True)
    assert gl.total_committed() == pytest.approx(1.1)  # the unused 0.9 is freed at once
    gl.record("/s", "r", "llm", "m", 0.1, reservation_id="gone")  # unknown id: still recorded
    assert gl.total_committed() == pytest.approx(1.2)
    with pytest.raises(ValueError):
        gl.record("/s", "r", "llm", "m", -0.1)


def test_record_is_never_refused_by_a_cap(tmp_path: Path) -> None:
    gl = GlobalLedger(tmp_path / "g.sqlite")
    gl.record("/s", "r", "llm", "m", 50.0)  # money already spent must be written
    assert gl.total_committed() == pytest.approx(50.0)


def test_summary_shape(tmp_path: Path) -> None:
    gl = GlobalLedger(tmp_path / "g.sqlite")
    gl.record("/s", "r1", "llm", "m", 1.0, 3, 4, note="x")
    gl.record("/s", "r1", "finetune", None, 2.0)
    gl.record("/t", "r2", "llm", "m", 0.5)
    gl.reserve("h", 0.25, 9.0, note="why")
    s = gl.summary()
    assert s["spend_usd"] == pytest.approx(3.5) and s["reserved_usd"] == pytest.approx(0.25)
    assert s["committed_usd"] == pytest.approx(3.75)
    assert {k["kind"]: (k["n"], k["usd"]) for k in s["by_kind"]} == {
        "llm": (2, 1.5),
        "finetune": (1, 2.0),
    }
    assert {r["run_id"]: r["usd"] for r in s["by_run"]} == {"r1": 3.0, "r2": 0.5}
    assert s["reservations"][0]["note"] == "why" and s["reservations"][0]["expires_at"] is None
    con = sqlite3.connect(tmp_path / "g.sqlite")
    try:
        assert con.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
    finally:
        con.close()


def test_unusable_file_fails_closed(tmp_path: Path) -> None:
    with pytest.raises(GlobalLedgerError):
        GlobalLedger(tmp_path)  # a directory
    bad = tmp_path / "bad.sqlite"
    bad.write_bytes(b"this is not a database" * 100)
    with pytest.raises(GlobalLedgerError):
        GlobalLedger(bad)


# ---- location rules -----------------------------------------------------------------------------
def test_resolve_path_rules(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(ENV_VAR, raising=False)
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    assert resolve_path({}) == tmp_path / "home" / ".distillery" / "global_ledger.sqlite"
    assert not (tmp_path / "home").exists()  # resolving creates nothing
    for off in ("none", "NONE", "off", "Off", "", "  "):
        assert resolve_path({ENV_VAR: off}) is None
    assert resolve_path({ENV_VAR: str(tmp_path / "x.sqlite")}) == tmp_path / "x.sqlite"
    # a mapping without the key falls back to the process environment, not to the home default
    monkeypatch.setenv(ENV_VAR, "none")
    assert resolve_path({}) is None and resolve_path(None) is None
    monkeypatch.setenv(ENV_VAR, str(tmp_path / "env.sqlite"))
    assert resolve_path({"OTHER": "1"}) == tmp_path / "env.sqlite"
    assert resolve_path({ENV_VAR: "none"}) is None  # the mapping wins when it has the key


def test_from_env_creates_parent_dir(tmp_path: Path) -> None:
    gl = GlobalLedger.from_env({ENV_VAR: str(tmp_path / "deep" / "dir" / "g.sqlite")})
    assert gl is not None and (tmp_path / "deep" / "dir" / "g.sqlite").exists()
    assert GlobalLedger.from_env({ENV_VAR: "none"}) is None


# ---- Ledger wiring ------------------------------------------------------------------------------
def _ledger(gl: GlobalLedger | None, *, run: str = "r", cap: float = 5.0) -> Ledger:
    return Ledger(run, 100.0, cap, 1.0, PRICES, global_ledger=gl)


def test_stale_snapshot_regression(tmp_path: Path) -> None:
    gl = GlobalLedger(tmp_path / "g.sqlite")
    a, b = _ledger(gl, run="a"), _ledger(gl, run="b")
    a.preflight(4.0)
    a.record("llm", "m", 4.0, 1, 1)
    with pytest.raises(BudgetExceeded, match="project cap"):
        b.preflight(2.0)  # b never saw the spend; the old snapshot would have let this through
    b.preflight(1.0)
    assert b.remaining() == pytest.approx(1.0)
    with pytest.raises(BudgetExceeded):
        b.preflight_playground(2.0)
    # the same scenario without a global ledger is the old snapshot behaviour (unchanged)
    old_a, old_b = _ledger(None, run="a"), _ledger(None, run="b")
    old_a.record("llm", "m", 4.0, 1, 1)
    old_b.preflight(2.0)
    assert old_b.remaining() == pytest.approx(5.0)


def test_global_total_includes_reservations_and_other_stores(tmp_path: Path) -> None:
    gl = GlobalLedger(tmp_path / "g.sqlite")
    gl.reserve("hosted-demo", 1.5, 10.0)
    gl.record("/other/store", "other-run", "llm", "m", 3.0)
    lg = _ledger(gl)
    assert lg.remaining() == pytest.approx(0.5)
    with pytest.raises(BudgetExceeded, match="project cap"):
        lg.preflight(0.6)
    lg.preflight(0.5)
    assert lg.spent() == 0.0  # the per-run view is this run's own spend only


def test_record_writes_local_store_and_global_ledger(tmp_path: Path) -> None:
    gl = GlobalLedger(tmp_path / "g.sqlite")
    store = Store(tmp_path / "store")
    lg = Ledger("r1", 10.0, 20.0, 1.0, PRICES, store, global_ledger=gl)
    lg.record("llm", "m", 0.25, 100, 50)
    lg.record(PLAYGROUND, "m", 0.05, 10, 5)
    assert store.total_spend() == pytest.approx(0.30)
    assert gl.total_committed() == pytest.approx(0.30)
    s = gl.summary()
    assert {r["run_id"] for r in s["by_run"]} == {"r1"}
    con = sqlite3.connect(tmp_path / "g.sqlite")
    try:
        rows = con.execute(
            "SELECT source_root, kind, model, input_tokens, output_tokens FROM spend ORDER BY id"
        ).fetchall()
    finally:
        con.close()
    assert rows == [
        (str(store.root), "llm", "m", 100, 50),
        (str(store.root), PLAYGROUND, "m", 10, 5),
    ]
    store.close()


def test_global_write_failure_raises_after_the_local_row_is_kept(tmp_path: Path) -> None:
    path = tmp_path / "g.sqlite"
    gl = GlobalLedger(path)
    store = Store(tmp_path / "store")
    lg = Ledger("r1", 10.0, 20.0, 1.0, PRICES, store, global_ledger=gl)
    for suffix in ("", "-wal", "-shm"):
        Path(f"{path}{suffix}").unlink(missing_ok=True)
    path.write_bytes(b"garbage" * 500)
    with pytest.raises(GlobalLedgerError):
        lg.record("llm", "m", 0.25, 1, 1)
    assert store.total_spend() == pytest.approx(0.25)  # the money was spent: the store has it
    with pytest.raises(GlobalLedgerError):
        lg.preflight(0.1)  # and a cap that cannot be read does not hold: fail closed
    store.close()


def test_from_config_attaches_only_for_real_store_backed_runs(tmp_path: Path) -> None:
    cfg = Config(project_cap_usd=5.0, run_cap_usd=5.0)
    path = tmp_path / "g.sqlite"
    env = {ENV_VAR: str(path)}
    store = Store(tmp_path / "store")
    real = Ledger.from_config("sql-gated-1", cfg, store, env=env)
    assert real.global_ledger is not None and real.global_ledger.path == path
    assert Ledger.from_config("sql-gated-1", cfg, env=env).global_ledger is None  # estimate-only
    assert Ledger.from_config("dry-sql-tiny-x", cfg, store, env=env).global_ledger is None
    assert Ledger.from_config("sql-1", cfg, store, env=env, dry_run=True).global_ledger is None
    assert Ledger.from_config("sql-1", cfg, store, env={ENV_VAR: "none"}).global_ledger is None
    assert Ledger.from_config("sql-1", cfg, store).global_ledger is None  # conftest: env is none
    store.close()


def test_dry_runs_never_write_to_the_global_ledger(tmp_path: Path) -> None:
    cfg = Config(project_cap_usd=5.0, run_cap_usd=5.0)
    path = tmp_path / "g.sqlite"
    env = {ENV_VAR: str(path)}
    GlobalLedger(path)  # exists, empty
    dry_store = Store(tmp_path / "dry-runs")
    dry = Ledger.from_config("dry-sql-tiny-x", cfg, dry_store, env=env)
    dry.record("llm", "fake", 123.0, 1, 1)  # fakes can price at anything: it is not money
    assert dry_store.total_spend() == pytest.approx(123.0)  # the dry store keeps its own books
    assert GlobalLedger(path).total_committed() == 0.0
    assert GlobalLedger(path).summary()["by_run"] == []
    real = Ledger.from_config("sql-real-1", cfg, Store(tmp_path / "real"), env=env)
    real.record("llm", "m", 0.5, 1, 1)
    assert GlobalLedger(path).total_committed() == pytest.approx(0.5)
    dry_store.close()


def test_disabled_mode_is_the_old_behaviour(tmp_path: Path) -> None:
    store = Store(tmp_path / "store")
    store.record_spend("earlier", "llm", "m", 1.5)
    cfg = Config(project_cap_usd=2.0, run_cap_usd=10.0)
    for off in ("none", "off", ""):
        lg = Ledger.from_config("r1", cfg, store, env={ENV_VAR: off})
        assert lg.global_ledger is None
        with pytest.raises(BudgetExceeded, match="project cap"):
            lg.preflight(0.6)
        lg.preflight(0.4)
        assert lg.remaining() == pytest.approx(0.5)
    # the snapshot is still a snapshot when disabled: spend elsewhere is not seen by preflight
    lg = Ledger.from_config("r1", cfg, store, env={ENV_VAR: "none"})
    store.record_spend("elsewhere", "llm", "m", 0.4)
    lg.preflight(0.5)
    assert not any(p.name.startswith("global_ledger") for p in tmp_path.rglob("*"))
    store.close()


# ---- the playground's project-cap check ---------------------------------------------------------
def _playground(tmp_path: Path, **cfg: Any) -> Any:
    settings = make_settings(tmp_path, config=make_config(project_cap_usd=1.0, **cfg))
    return create_app(settings).state.playground


def test_playground_reserve_project_check_is_fresh_with_the_global_ledger(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "g.sqlite"
    monkeypatch.setenv(ENV_VAR, str(path))
    pg = _playground(tmp_path / "s")
    assert pg.ledger.global_ledger is not None
    pg._reserve(0.1)
    pg._release(0.1)
    GlobalLedger(path).record("/another/store", "other-run", "llm", "m", 0.95)  # after start-up
    with pytest.raises(BudgetExceeded, match="project cap"):
        pg._reserve(0.1)
    assert pg._reserved == 0.0  # the refused estimate was not left reserved
    pg._reserve(0.04)  # still room for a smaller call


def test_playground_reserve_project_check_is_fresh_from_the_store_when_disabled(
    tmp_path: Path,
) -> None:
    pg = _playground(tmp_path)
    assert pg.ledger.global_ledger is None
    pg._reserve(0.1)
    pg._release(0.1)
    Store(tmp_path / "data").record_spend("other-process", "llm", "fake-t", 0.95)
    with pytest.raises(BudgetExceeded, match="project cap"):
        pg._reserve(0.1)
    pg._reserve(0.04)


# ---- seed file and CLI --------------------------------------------------------------------------
def test_seed_file_totals_15_6952_and_is_idempotent(tmp_path: Path) -> None:
    gl = GlobalLedger(tmp_path / "g.sqlite")
    spend, reservations = load_seed_file(SEED_FILE)
    assert len(spend) == 5 and [r["id"] for r in reservations] == ["hosted-demo"]
    res = gl.seed(spend, reservations)
    assert len(res.spend_inserted) == 5 and res.reservations_inserted == ["hosted-demo"]
    assert gl.total_committed() == pytest.approx(15.6952, abs=1e-9)
    again = gl.seed(spend, reservations)
    assert again.spend_inserted == [] and again.reservations_inserted == []
    assert len(again.spend_skipped) == 5 and again.reservations_skipped == ["hosted-demo"]
    assert gl.total_committed() == pytest.approx(15.6952, abs=1e-9)
    s = gl.summary()
    assert s["by_kind"] == [{"kind": "seed", "n": 5, "usd": pytest.approx(14.1952)}]
    (hosted,) = s["reservations"]
    assert hosted["id"] == "hosted-demo" and hosted["usd"] == 1.5 and hosted["expires_at"] is None
    assert hosted["note"] == "hosted demo allocation Oct 30 - Dec 15 2026 (Render playground)"
    by_run = {r["run_id"]: r["usd"] for r in s["by_run"]}
    assert by_run["sql-gated-20261004T155134Z-66ff347c"] == pytest.approx(4.2582)
    assert by_run["chaos-sql-gated-20261007b"] == pytest.approx(4.1918)
    # 19.5 cap leaves exactly the documented headroom
    with pytest.raises(BudgetExceeded):
        gl.reserve("x", 3.81, 19.5)
    gl.reserve("x", 3.80, 19.5)


def test_seed_rejects_bad_rows(tmp_path: Path) -> None:
    gl = GlobalLedger(tmp_path / "g.sqlite")
    with pytest.raises(ValueError, match="note"):
        gl.seed([{"usd": 1.0}], [])
    with pytest.raises(ValueError, match="unique"):
        gl.seed([{"usd": 1.0, "note": "a"}, {"usd": 2.0, "note": "a"}], [])
    with pytest.raises(ValueError):
        gl.seed([{"usd": -1.0, "note": "a"}], [])
    assert gl.total_committed() == 0.0
    bad = tmp_path / "bad.json"
    bad.write_text("[1, 2]")
    with pytest.raises(ValueError, match="object"):
        load_seed_file(bad)
    with pytest.raises(ValueError, match="cannot read"):
        load_seed_file(tmp_path / "missing.json")


def test_cli_ledger_seed_and_status(tmp_path: Path) -> None:
    path = tmp_path / "home" / "g.sqlite"
    env = {ENV_VAR: str(path), "DISTILLERY_PROJECT_CAP_USD": "19.5"}
    lines: list[str] = []

    def run(*argv: str, e: dict[str, str] = env) -> int:
        lines.clear()
        return main(list(argv), env=e, out=lines.append)

    assert run("ledger", "status") == 0 and "not created yet" in lines[0]
    assert not path.exists()  # status never creates the ledger
    assert run("ledger", "seed", "--file", str(SEED_FILE)) == 0
    assert "spend rows inserted 5, skipped 0" in "\n".join(lines)
    assert "committed usd: 15.6952" in "\n".join(lines)
    assert run("ledger", "seed", "--file", str(SEED_FILE)) == 0
    text = "\n".join(lines)
    assert "spend rows inserted 0, skipped 5" in text and "committed usd: 15.6952" in text
    assert run("ledger", "status") == 0
    text = "\n".join(lines)
    assert "committed usd: 15.6952 (spend 14.1952 + reservations 1.5000)" in text
    assert "project cap usd: 19.5000" in text and "remaining usd: 3.8048" in text
    assert "seed: n=5 usd=14.1952" in text
    assert "hosted-demo holder=hosted-demo usd=1.5000 until=permanent" in text
    assert run("ledger", "seed", "--file", str(tmp_path / "nope.json")) == 2  # EXIT_REFUSED


def test_cli_ledger_disabled(tmp_path: Path) -> None:
    lines: list[str] = []
    env = {ENV_VAR: "none"}
    assert main(["ledger", "status"], env=env, out=lines.append) == 0
    assert "disabled" in lines[0]
    assert main(["ledger", "seed", "--file", str(SEED_FILE)], env=env, out=lines.append) == 1
    assert not any(tmp_path.iterdir())


def test_the_autouse_fixture_keeps_tests_off_the_home_ledger() -> None:
    assert os.environ[ENV_VAR] == "none"
