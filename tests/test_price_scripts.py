# ruff: noqa: S101
import importlib.util
import sqlite3
import sys
from pathlib import Path
from types import ModuleType

import pytest

from distillery.config import FinetunePrice, Price, PriceFile, SandboxPrice
from distillery.orchestrator import SCALES
from distillery.store import Store

SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"


def load(name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


est = load("estimate_run_cost")
reb = load("rebase_ledger")


def px(i: float, o: float) -> Price:
    return Price(input_per_mtok=i, output_per_mtok=o, source="t", date="d")


FULL = PriceFile(
    llm={est.TEACHER: px(1, 3), est.PLANNER: px(3, 9)},
    finetune={est.STUDENT: FinetunePrice(usd_per_mtok_trained_tokens=2, source="t", date="d")},
    sandbox=SandboxPrice(usd_per_second=0.01, source="t", date="d"),
)
AVGS = {
    est.TEACHER: est.Avg(10, 1000.0, 400.0),
    est.PLANNER: est.Avg(5, 2000.0, 500.0),
}


def test_estimate_arithmetic_all_priced() -> None:
    sc = SCALES["mini"]  # train 120, dev 30, heldout 60
    lines = est.build_estimate(
        sc, FULL, AVGS, row_chars=300.0, chars_per_token=3.0, epochs=3, rounds=2,
        retries_factor=1.5, sec_per_generation=8.0, stress_n=0,
    )  # fmt: skip
    usd = {ln.name: ln.usd for ln in lines}
    per_teacher_call = (1000 * 1 + 400 * 3) / 1e6
    assert usd["teacher: train rows"] == pytest.approx(120 * 1.5 * per_teacher_call)
    assert usd["teacher: held-out/stress eval"] == pytest.approx(60 * per_teacher_call)
    assert usd["planner: failure analysis (upper bound)"] == pytest.approx(
        30 * 1 * (2000 * 3 + 500 * 9) / 1e6
    )
    tokens = 120 * 100 * 3 * 2  # 120 rows x 100 tokens x 3 epochs x 2 rounds
    assert usd["fine-tune"] == pytest.approx(tokens * 2 / 1e6)
    n_gen = (30 + 60) + (30 * 2 + 60)
    assert usd["sandbox generation"] == pytest.approx(n_gen * 8.0 * 0.01)
    assert all(ln.status.startswith("console price") for ln in lines)
    assert "TOTAL: $" in est.render(lines)


def test_stress_set_is_priced_by_default_and_assumed_prices_are_never_called_real() -> None:
    sc = SCALES["mini"]  # stress 20
    with_stress = {ln.name: ln for ln in est.build_estimate(sc, FULL, AVGS, row_chars=300.0)}
    none = {ln.name: ln for ln in est.build_estimate(sc, FULL, AVGS, row_chars=300.0, stress_n=0)}
    per_call = (1000 * 1 + 400 * 3) / 1e6
    assert with_stress["teacher: held-out/stress eval"].usd == pytest.approx((60 + 20) * per_call)
    assert with_stress["sandbox generation"].usd > none["sandbox generation"].usd
    assumed = PriceFile(
        llm={
            est.TEACHER: Price(
                input_per_mtok=1, output_per_mtok=3, source="ASSUMED ceiling", date="d"
            )
        }
    )
    lines = est.build_estimate(sc, assumed, AVGS, row_chars=300.0)
    teacher = next(ln for ln in lines if ln.name == "teacher: train rows")
    assert teacher.status == est.ASSUMED_PRICE and "console" not in teacher.status.split("(")[0]
    assert "ASSUMED ceiling prices" in est.render(lines)


def test_missing_prices_withhold_total() -> None:
    partial = PriceFile(llm={est.TEACHER: px(1, 3)})
    lines = est.build_estimate(SCALES["mini"], partial, AVGS, row_chars=300.0)
    text = est.render(lines)
    assert "TOTAL withheld" in text
    assert "TOTAL: $" not in text
    assert "SUBTOTAL of priced lines only" in text
    missing = {ln.name for ln in lines if ln.usd is None}
    assert missing == {"planner: failure analysis (upper bound)", "fine-tune", "sandbox generation"}
    assert all(est.MISSING_PRICE in ln.status for ln in lines if ln.usd is None)


def test_missing_measurement_is_not_priced_at_zero() -> None:
    lines = est.build_estimate(SCALES["tiny"], FULL, {}, row_chars=None)
    assert [ln.usd for ln in lines if ln.name.startswith(("teacher", "planner", "fine"))] == [
        None
    ] * 4
    assert "MISSING MEASUREMENT" in est.render(lines)


def make_run(root: Path) -> Store:
    s = Store(root)
    s.record_llm_call("r1", est.TEACHER, 1000, 300, 99.0)  # old (assumed) usd
    s.record_llm_call("r1", est.TEACHER, 3000, 100, 99.0)
    s.record_spend("r1", "llm", est.TEACHER, 5.0)
    s.record_spend("r1", "finetune_ceiling", None, 2.0)
    return s


def test_measure_avgs_and_row_chars(tmp_path: Path) -> None:
    make_run(tmp_path).close()
    avgs = est.measure_avgs(tmp_path, "r1")
    assert avgs[est.TEACHER] == est.Avg(2, 2000.0, 200.0)
    f = tmp_path / "train.jsonl"
    f.write_text("aaaa\nbbbbbb\n\n")
    assert est.avg_row_chars(f) == 5.0
    assert est.avg_row_chars(tmp_path / "nope.jsonl") is None


def test_rebase_report_then_apply_is_idempotent(tmp_path: Path) -> None:
    make_run(tmp_path).close()
    deltas = reb.compute(tmp_path, FULL, {"r1": [1_000_000]})
    d = deltas[0]
    new_llm = (4000 * 1 + 400 * 3) / 1e6
    assert d.old_usd == pytest.approx(7.0)
    assert d.new_usd == pytest.approx(new_llm + 1_000_000 * 2 / 1e6)
    # report-only: nothing written
    con = sqlite3.connect(tmp_path / "index.sqlite")
    assert con.execute("SELECT COUNT(*) FROM spend").fetchone()[0] == 2
    con.close()
    assert reb.apply(tmp_path, deltas) == 1
    con = sqlite3.connect(tmp_path / "index.sqlite")
    rows = con.execute("SELECT kind, usd FROM spend ORDER BY id").fetchall()
    con.close()
    assert [k for k, _ in rows] == ["llm", "finetune_ceiling", "rebase"]  # rows only appended
    assert rows[0][1] == 5.0
    again = reb.compute(tmp_path, FULL, {"r1": [1_000_000]})
    assert again[0].delta == pytest.approx(0.0)
    assert reb.apply(tmp_path, again) == 0


def test_rebase_refuses_unpriced_model_and_keeps_ceiling_without_tokens(tmp_path: Path) -> None:
    make_run(tmp_path).close()
    d = reb.compute(tmp_path, PriceFile(), None)[0]
    assert d.new_usd is None
    assert "no price" in d.note
    assert reb.apply(tmp_path, [d]) == 0
    d2 = reb.compute(tmp_path, FULL, None)[0]  # priced, but no trained_tokens known
    assert d2.new_usd == pytest.approx((4000 * 1 + 400 * 3) / 1e6 + 2.0)
