# ruff: noqa: S101, S105, S106
import logging
from pathlib import Path

import pytest

import sqlite3

from distillery.budget import (
    BASIS_BILLED,
    BASIS_CEILING,
    BudgetExceeded,
    Ledger,
    UnknownPriceError,
    paid_resource,
    sandbox_seconds,
    spend_lines,
)
from distillery.config import FinetunePrice, Price, SandboxPrice
from distillery.store import Store

PRICES = {"m": Price(input_per_mtok=2.0, output_per_mtok=8.0, source="test", date="2026-01-01")}


def cost_rows(store: Store) -> int:
    con = sqlite3.connect(store.root / "index.sqlite")
    try:
        return int(con.execute("SELECT COUNT(*) FROM llm_calls").fetchone()[0])
    finally:
        con.close()


def ledger(store: Store | None = None, run: float = 1.0, proj: float = 2.0) -> Ledger:
    return Ledger("r1", run, proj, 0.5, PRICES, store)


def test_estimate_and_unknown_price() -> None:
    lg = ledger()
    assert lg.estimate_llm_cost("m", 1_000_000, 500_000) == pytest.approx(6.0)
    with pytest.raises(UnknownPriceError):
        lg.estimate_llm_cost("nope", 1, 1)


def test_run_cap() -> None:
    lg = ledger()
    lg.preflight(0.9)
    lg.record("llm", "m", 0.9, 10, 10)
    assert lg.spent() == pytest.approx(0.9)
    with pytest.raises(BudgetExceeded, match="run cap"):
        lg.preflight(0.2)
    lg.preflight(0.1)


def test_project_cap_with_store_across_runs(tmp_path: Path) -> None:
    s = Store(tmp_path)
    s.record_spend("other", "llm", "m", 1.5)
    lg = ledger(s, run=10.0, proj=2.0)
    assert lg.project_spent() == pytest.approx(1.5)
    with pytest.raises(BudgetExceeded, match="project cap"):
        lg.preflight(0.6)
    lg.record("llm", "m", 0.4, 1, 1)
    assert s.total_spend("r1") == pytest.approx(0.4)
    # a fresh ledger sees persisted spend
    assert ledger(s, run=10.0, proj=2.0).spent() == pytest.approx(0.4)


def test_playground_daily_cap() -> None:
    lg = ledger()
    lg.preflight_playground(0.4, day="2026-01-01")
    lg.record("playground", "m", 0.4, day="2026-01-01")
    assert lg.spent() == 0.0  # not counted against the run cap
    with pytest.raises(BudgetExceeded, match="playground"):
        lg.preflight_playground(0.2, day="2026-01-01")
    lg.preflight_playground(0.2, day="2026-01-02")  # new day resets


def test_playground_persists(tmp_path: Path) -> None:
    s = Store(tmp_path)
    ledger(s).record("playground", None, 0.45, day="2026-01-01")
    with pytest.raises(BudgetExceeded):
        ledger(s).preflight_playground(0.1, day="2026-01-01")


def test_paid_resource_cancels_on_exception() -> None:
    cancelled: list[str] = []
    with pytest.raises(RuntimeError, match="boom"):
        with paid_resource(lambda: "job1", cancelled.append):
            raise RuntimeError("boom")
    assert cancelled == ["job1"]


def test_paid_resource_cancels_on_keyboard_interrupt_and_success() -> None:
    cancelled: list[str] = []
    with pytest.raises(KeyboardInterrupt):
        with paid_resource(lambda: "j", cancelled.append):
            raise KeyboardInterrupt
    with paid_resource(lambda: "k", cancelled.append) as r:
        assert r == "k"
    assert cancelled == ["j", "k"]


def test_cancel_error_does_not_mask_original(caplog: pytest.LogCaptureFixture) -> None:
    def bad_cancel(_: str) -> None:
        raise OSError("cancel failed")

    with caplog.at_level(logging.ERROR):
        with pytest.raises(RuntimeError, match="original"):
            with paid_resource(lambda: "j", bad_cancel):
                raise RuntimeError("original")
    assert "cancel of paid resource failed" in caplog.text


def test_create_failure_no_cancel() -> None:
    cancelled: list[str] = []

    def create() -> str:
        raise ValueError("no create")

    with pytest.raises(ValueError):
        with paid_resource(create, cancelled.append):
            pass
    assert cancelled == []


FT = {"base": FinetunePrice(usd_per_mtok_trained_tokens=4.0, source="console", date="2026-10-01")}
SB = SandboxPrice(usd_per_cpu_second=0.002, source="console", date="2026-10-01")


def priced(store: Store | None = None, *, ft: bool = True, sb: bool = True) -> Ledger:
    return Ledger(
        "r1", 5.0, 10.0, 0.5, PRICES, store, finetune_prices=FT if ft else None,
        sandbox_price=SB if sb else None,
    )  # fmt: skip


def test_finetune_estimate_is_trained_tokens_times_price() -> None:
    lg = priced()
    assert lg.estimate_finetune_cost("base", 396_885) == pytest.approx(396_885 * 4.0 / 1e6)
    assert lg.estimate_finetune_cost("base", 0) == 0.0


def test_unknown_finetune_or_sandbox_price_refuses() -> None:
    lg = priced(ft=False, sb=False)
    with pytest.raises(UnknownPriceError, match="fine-tune"):
        lg.estimate_finetune_cost("base", 10)
    with pytest.raises(UnknownPriceError, match="sandbox"):
        lg.estimate_sandbox_cost(10.0)
    with pytest.raises(UnknownPriceError):
        priced().estimate_finetune_cost("other-model", 10)


def test_sandbox_estimate() -> None:
    assert priced().estimate_sandbox_cost(1000 * 7.8) == pytest.approx(15.6)


def test_preflight_finetune_uses_planned_tokens_when_priced() -> None:
    lg = priced()
    usd, basis = lg.preflight_finetune("base", 1_000_000, fallback_usd=99.0)
    assert usd == pytest.approx(4.0)
    assert basis == BASIS_BILLED
    with pytest.raises(BudgetExceeded):
        lg.preflight_finetune("base", 2_000_000, fallback_usd=0.0)  # 8.0 > run cap 5.0


def test_preflight_finetune_falls_back_to_ceiling() -> None:
    lg = priced(ft=False)
    usd, basis = lg.preflight_finetune("base", 1_000_000, fallback_usd=2.0)
    assert (usd, basis) == (2.0, BASIS_CEILING)
    with pytest.raises(UnknownPriceError):
        lg.preflight_finetune("base", 1_000_000, fallback_usd=None)  # no price, no ceiling


def test_record_finetune_uses_actual_trained_tokens(tmp_path: Path) -> None:
    s = Store(tmp_path)
    lg = priced(s)
    usd, basis = lg.record_finetune("base", trained_tokens=115_611, ceiling_usd=2.0)
    assert usd == pytest.approx(115_611 * 4.0 / 1e6)
    assert basis == BASIS_BILLED
    assert lg.spent() == pytest.approx(usd)
    lines = spend_lines(s, "r1")
    assert lines == [
        {"kind": "finetune", "model": "base", "usd": pytest.approx(usd), "units": 115_611,
         "basis": BASIS_BILLED}  # fmt: skip
    ]
    # a fine-tune is not an LLM call
    assert cost_rows(s) == 0


def test_record_finetune_keeps_ceiling_when_price_or_tokens_missing(tmp_path: Path) -> None:
    s = Store(tmp_path)
    lg = priced(s, ft=False)
    assert lg.record_finetune("base", 100, ceiling_usd=2.0) == (2.0, BASIS_CEILING)
    lg2 = priced(s)  # priced, but the job reported no trained_tokens
    assert lg2.record_finetune("base", None, ceiling_usd=1.0) == (1.0, BASIS_CEILING)
    kinds = [x["kind"] for x in spend_lines(s, "r1")]
    assert kinds == ["finetune_ceiling", "finetune_ceiling"]
    with pytest.raises(UnknownPriceError):
        lg2.record_finetune("base", None, ceiling_usd=None)


def test_record_sandbox_priced_and_unpriced(tmp_path: Path) -> None:
    s = Store(tmp_path)
    assert priced(s).record_sandbox(10.0, "student", samples=2) == pytest.approx(0.02)
    assert priced(s, sb=False).record_sandbox(5.0, "student", samples=1) is None
    lines = spend_lines(s, "r1")
    assert [(x["kind"], x["usd"]) for x in lines] == [("sandbox", pytest.approx(0.02)),
                                                       ("sandbox_unpriced", 0.0)]  # fmt: skip
    assert [x["units"] for x in lines] == [10, 5]  # measured seconds are kept either way
    assert sandbox_seconds(s, "r1", "student") == (15.0, 3)
