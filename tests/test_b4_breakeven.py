# ruff: noqa: S101
from __future__ import annotations

import importlib.util
import sqlite3
from pathlib import Path

import pytest

from distillery.store import Store

SPEC = importlib.util.spec_from_file_location(
    "b4_breakeven", Path(__file__).resolve().parents[1] / "scripts" / "b4_breakeven.py"
)
assert SPEC and SPEC.loader
b4 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(b4)


def test_percentile_nearest_rank() -> None:
    v = [float(i) for i in range(1, 101)]
    assert (
        b4.percentile(v, 50) == 50 and b4.percentile(v, 95) == 95 and b4.percentile([3.0], 95) == 3
    )


def test_breakeven_formula_and_capacity() -> None:
    r = b4.breakeven(0.001, 2.0, 1000.0, "dedicated")
    assert r["breakeven_questions_per_day"] == pytest.approx(48000)  # 24 * 2 / 0.001
    assert r["capacity_check"]["status"] == "break-even unreachable on one host"  # 48000 > 24000
    assert b4.breakeven(0.001, 2.0, 5000.0)["capacity_check"]["status"] == "ok"
    cpu = b4.breakeven(0.001, 0.0, None)
    assert cpu["student_host"] == "sandbox CPU, not Dedicated"
    assert cpu["capacity_check"]["status"] == "unknown"
    with pytest.raises(ValueError):
        b4.breakeven(0.0, 1.0, None)


def test_teacher_stats_from_ledger_latency(tmp_path: Path) -> None:
    st = Store(tmp_path)
    for i in range(1, 101):
        st.record_llm_call("r", "m", 100, 10, 0.002, i / 10)
    st.record_llm_call("r", "m", 100, 10, 0.5, None)  # no latency: excluded
    st.record_llm_call("r", "other", 1, 1, 9.0, 1.0)
    st.close()
    t = b4.teacher_stats(tmp_path, "m")
    assert t["n_calls"] == 100 and t["latency_p50_s"] == pytest.approx(5.0)
    assert t["latency_p95_s"] == pytest.approx(9.5)
    assert t["usd_per_1k_questions"] == pytest.approx(2.0)
    assert b4.teacher_stats(tmp_path, "m", limit=10)["n_calls"] == 10
    with pytest.raises(SystemExit):
        b4.teacher_stats(tmp_path, "absent")


def test_old_store_without_latency_column_is_refused(tmp_path: Path) -> None:
    con = sqlite3.connect(tmp_path / "index.sqlite")
    con.execute(
        "CREATE TABLE llm_calls (id INTEGER PRIMARY KEY, run_id TEXT, model TEXT, usd REAL)"
    )
    con.commit()
    con.close()
    with pytest.raises(SystemExit, match="predates"):
        b4.teacher_stats(tmp_path, "m")
