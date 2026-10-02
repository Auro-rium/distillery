# ruff: noqa: S101
import importlib.util
import json
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
    assert all(ln.status.startswith("listed price") for ln in lines)
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
    assert teacher.status == est.ASSUMED_PRICE and "listed" not in teacher.status
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


def test_measured_tokens_per_row_drives_the_fine_tune_line() -> None:
    sc = SCALES["mini"]  # train 120
    lines = est.build_estimate(
        sc, FULL, AVGS, row_chars=999999.0, tokens_per_row=960.0, epochs=3, rounds=1, stress_n=0
    )
    ft = next(ln for ln in lines if ln.name == "fine-tune")
    assert ft.usd == pytest.approx(120 * 960 * 3 * 2 / 1e6)  # price 2 per Mtok; chars ignored
    assert "MEASURED" in ft.quantity


# ---- gated (claims-real) design -----------------------------------------------------------------

GATED_AVGS = {**AVGS}
NANO_AVG = est.Avg(0, 200.0, 100.0)


def gated_prices() -> PriceFile:
    return PriceFile(
        llm={
            est.TEACHER: px(1, 3),
            est.PLANNER: px(3, 9),
            est.NANO: px(0.5, 1.0),
        },
        finetune={
            est.STUDENT_1_7B: FinetunePrice(usd_per_mtok_trained_tokens=2, source="t", date="d")
        },
        sandbox=SandboxPrice(usd_per_second=0.01, source="t", date="d"),
    )


def test_gated_estimate_arithmetic() -> None:
    sc = SCALES["gated"]
    lines = est.build_gated_estimate(
        sc, gated_prices(), GATED_AVGS, human_n=100, final_rows=1800, original_yield=0.9,
        sec_per_sample=20.0, nano_avg=NANO_AVG,
    )  # fmt: skip
    usd = {ln.name: ln.usd for ln in lines}
    t = (1000 * 1 + 400 * 3) / 1e6  # one teacher call
    assert usd["teacher: train originals"] == pytest.approx(650 * t)
    assert usd["nano: paraphrase calls"] == pytest.approx(650 * 0.9 * (200 * 0.5 + 100 * 1.0) / 1e6)
    assert usd["teacher: verify paraphrases (upper bound)"] == pytest.approx(650 * 0.9 * 3 * t)
    assert usd["teacher: draft human set (k=3)"] == pytest.approx(100 * 3 * t)
    assert usd["teacher: eval gate+stress+human"] == pytest.approx((300 + 100 + 100) * t)
    assert usd["planner: failure analysis (upper bound)"] == 0.0  # one round, no analysis
    assert usd["fine-tune"] == pytest.approx(1800 * 960 * 3 * 2 / 1e6)
    n_gen = 150 + 150 + 2 * 500
    assert est.gated_generations(sc, 100) == n_gen
    assert usd["sandbox generation"] == pytest.approx(n_gen * 20.0 * 0.01)
    nano = next(ln for ln in lines if ln.name.startswith("nano"))
    assert "USER-SUPPLIED" in nano.status
    assert "TOTAL: $" in est.render(lines)


def test_gated_nano_is_missing_measurement_not_a_guess() -> None:
    lines = est.build_gated_estimate(SCALES["gated"], gated_prices(), GATED_AVGS, human_n=50)
    nano = next(ln for ln in lines if ln.name.startswith("nano"))
    assert nano.usd is None and "MISSING MEASUREMENT" in nano.status
    text = est.render(lines)
    assert "TOTAL withheld" in text and "TOTAL: $" not in text


def test_finetune_ceiling_is_explicit_assumed_never_a_price_or_in_a_total() -> None:
    no_ft = PriceFile(llm=gated_prices().llm, sandbox=gated_prices().sandbox)
    lines = est.build_gated_estimate(
        SCALES["gated"], no_ft, GATED_AVGS, human_n=100, nano_avg=NANO_AVG,
        finetune_ceiling_usd=8.0,
    )  # fmt: skip
    ft = next(ln for ln in lines if ln.name == "fine-tune")
    assert ft.usd is None and est.MISSING_PRICE in ft.status
    ceil = next(ln for ln in lines if ln.informational)
    assert ceil.usd == 8.0 and ceil.status.startswith("ASSUMED") and "NOT a price" in ceil.status
    text = est.render(lines)
    assert "TOTAL withheld" in text and "TOTAL: $" not in text
    subtotal = sum(ln.usd for ln in lines if ln.usd is not None and not ln.informational)
    assert f"SUBTOTAL of priced lines only: ${subtotal:.4f}" in text  # ceiling not inside it
    assert "ASSUMED CEILING (not a price, not a quote)" in text
    none = est.build_gated_estimate(SCALES["gated"], no_ft, GATED_AVGS, human_n=100)
    assert not any(ln.informational for ln in none)  # shown only when asked for


def test_wall_clock_note_and_cli(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    note = est.wall_clock_note(1100, 0.22)
    assert "1.39 h" in note and "0.22 samples/s" in note
    s = Store(tmp_path)
    s.record_llm_call("r1", est.TEACHER, 1000, 300, 0.0)
    s.close()
    prices = tmp_path / "p.json"
    entry = {"input_per_mtok": 0.3, "output_per_mtok": 0.9, "source": "api", "date": "d"}
    prices.write_text(json.dumps({est.TEACHER: entry}))
    rc = est.main(
        [
            "--scale", "gated", "--prices", str(prices), "--root", str(tmp_path), "--run", "r1",
            "--human-n", "100", "--finetune-ceiling-usd", "8",
        ]
    )  # fmt: skip
    out = capsys.readouterr().out
    assert rc == 0 and "scale=gated" in out and "wall-clock" in out
    assert "MISSING MEASUREMENT" in out and "ASSUMED CEILING" in out and "TOTAL withheld" in out
