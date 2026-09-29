# ruff: noqa: S101, S105, S106
import logging
from pathlib import Path

import pytest

from distillery.budget import BudgetExceeded, Ledger, UnknownPriceError, paid_resource
from distillery.config import Price
from distillery.store import Store

PRICES = {"m": Price(input_per_mtok=2.0, output_per_mtok=8.0, source="test", date="2026-01-01")}


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
