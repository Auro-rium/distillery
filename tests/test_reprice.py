# ruff: noqa: S101
from __future__ import annotations

import json
from pathlib import Path

import pytest

from distillery.budget import KIND_FINETUNE_CEILING, KIND_SANDBOX_UNPRICED
from distillery.cli import EXIT_FAIL, EXIT_OK, EXIT_REFUSED, main
from distillery.config import FinetunePrice, Price, PriceFile, SandboxPrice
from distillery.reprice import RepriceError, reprice_run
from distillery.store import Store

RUN = "sql-gated-t"
STUDENT = "Qwen/Qwen3-1.7B"


def _prices(*, ft: bool = True, sb: bool = True) -> PriceFile:
    return PriceFile(
        llm={"teacher-m": Price(input_per_mtok=1, output_per_mtok=3, source="api", date="d1")},
        finetune={
            STUDENT: FinetunePrice(usd_per_mtok_trained_tokens=2, source="console", as_of="d2")
        }
        if ft
        else {},
        sandbox=SandboxPrice(usd_per_second=0.01, source="spike", as_of="d3") if sb else None,
    )


@pytest.fixture
def store(tmp_path: Path) -> Store:
    st = Store(tmp_path)
    st.create_run(RUN)
    st.record_llm_call(RUN, "teacher-m", 1000, 400, 0.0)
    st.record_llm_call(RUN, "teacher-m", 3000, 600, 0.0)
    # what the original run recorded: a ceiling fine-tune row and unpriced sandbox seconds
    st.record_spend(RUN, KIND_FINETUNE_CEILING, STUDENT, 5.0, 2_000_000, 0, "2026-10-03")
    st.record_spend(RUN, KIND_SANDBOX_UNPRICED, "sandbox:student_dev", 0.0, 100_000, 10)
    st.record_spend(RUN, KIND_SANDBOX_UNPRICED, "sandbox:base_eval", 0.0, 50_000, 5)
    report = {
        "run_id": RUN,
        "cost": {"run_total_usd": 5.0, "basis": "original"},
        "finetune": [
            {"round": 1, "job_id": "j", "base_model": STUDENT, "trained_tokens": 2_000_000}
        ],
    }
    path = st.run_dir(RUN) / "report.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report))
    return st


def test_reprice_recomputes_from_measurements_and_keeps_original_cost(store: Store) -> None:
    rp = reprice_run(
        store, RUN, _prices(), prices_label="p.json", balance_before=20.0, balance_after=14.0
    )
    by_kind = {ln["kind"]: ln for ln in rp["lines"]}
    assert by_kind["llm"]["usd"] == pytest.approx((4000 * 1 + 1000 * 3) / 1e6)
    assert by_kind["finetune"]["usd"] == pytest.approx(2_000_000 * 2 / 1e6)
    assert by_kind["sandbox"]["sandbox_seconds"] == pytest.approx(150.0)
    assert by_kind["sandbox"]["usd"] == pytest.approx(1.5)
    assert by_kind["sandbox"]["price"] == {"source": "spike", "as_of": "d3"}
    assert by_kind["finetune"]["price"] == {"source": "console", "as_of": "d2"}
    assert rp["total_usd"] == pytest.approx(0.007 + 4.0 + 1.5)
    rec = rp["reconciliation"]
    assert rec["balance_delta_usd"] == pytest.approx(6.0)
    assert rec["unexplained_usd"] == pytest.approx(6.0 - rp["total_usd"])
    saved = json.loads((store.run_dir(RUN) / "report.json").read_text())
    assert saved["cost"] == {"run_total_usd": 5.0, "basis": "original"}  # never overwritten
    assert saved["cost_repriced"]["total_usd"] == rp["total_usd"]


def test_missing_prices_withhold_total_and_reconcile_against_subtotal(store: Store) -> None:
    rp = reprice_run(
        store,
        RUN,
        _prices(ft=False, sb=False),
        prices_label="p",
        balance_before=10,
        balance_after=9,
    )
    assert rp["total_usd"] is None and len(rp["missing"]) == 2
    assert rp["subtotal_usd"] == pytest.approx(0.007)
    assert "SUBTOTAL" in rp["reconciliation"]["compared_with"]


def test_balance_args_come_in_pairs_and_missing_report_fails(store: Store, tmp_path: Path) -> None:
    with pytest.raises(RepriceError):
        reprice_run(store, RUN, _prices(), prices_label="p", balance_before=1.0)
    other = Store(tmp_path / "empty")
    with pytest.raises(RepriceError):
        reprice_run(other, "nope", _prices(), prices_label="p")


def test_cli_reprice(store: Store, tmp_path: Path) -> None:
    pf = tmp_path / "prices.json"
    pf.write_text(
        json.dumps(
            {
                "teacher-m": {"input_per_mtok": 1, "output_per_mtok": 3, "source": "a", "date": "d"},
                "finetune": {
                    STUDENT: {"usd_per_mtok_trained_tokens": 2, "source": "c", "as_of": "d"}
                },
                "sandbox": {"usd_per_second": 0.01, "source": "s", "as_of": "d"},
            }
        )
    )
    lines: list[str] = []
    base = ["--root", str(store.root), "reprice", RUN, "--prices", str(pf)]
    ok = main([*base, "--balance-before", "20", "--balance-after", "14"], out=lines.append)
    assert ok == EXIT_OK
    assert any("TOTAL $" in ln for ln in lines)
    assert any("console balance fell" in ln for ln in lines)
    assert main([*base, "--balance-before", "20"], out=lines.append) == EXIT_REFUSED
    bad = main(["--root", str(store.root), "reprice", "zz", "--prices", str(pf)], out=lines.append)
    assert bad in (EXIT_REFUSED, EXIT_FAIL)
