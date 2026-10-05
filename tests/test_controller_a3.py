# ruff: noqa: S101
"""A3: the Ultra controller proposes, pure code validates, rejection falls back to the rules."""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from distillery.budget import BudgetExceeded
from distillery.controller import (
    Accepted,
    ControllerAction,
    ControllerHParams,
    ControllerState,
    Rejected,
    validate,
)
from distillery.orchestrator import Pipeline, Scale
from distillery.pipeline_fakes import DryRun, build_dry_run
from distillery.sandbox_executor import AsyncBridge
from distillery.store import Store

NANO = Scale(name="nano", train=16, dev=8, heldout=12, stress=6)


@pytest.fixture
def bridge() -> Iterator[AsyncBridge]:
    with AsyncBridge() as b:
        yield b


def state(**kw: Any) -> ControllerState:
    base: dict[str, Any] = {
        "round": 1, "max_rounds": 3, "dev_acc": 0.5, "dev_acc_by_family": {"a": 0.5},
        "clusters": [], "allowed_families": ("a", "b"), "remaining_usd": 10.0, "data_usd": 0.1,
        "ft_usd_per_epoch": 0.1, "ft_usd_fixed": None, "current_lr": 1e-4, "current_n_epochs": 3,
        "current_lora_r": 16, "batch_size": 4, "packing": False, "train_rows": 1000,
        "min_planned_steps": 300, "dry_run": False,
    }  # fmt: skip
    return ControllerState(**{**base, **kw})


def act(action: str = "adjust_hparams", family: str | None = None, **hp: Any) -> ControllerAction:
    return ControllerAction(
        action=action,
        family=family,
        hparams=ControllerHParams(**hp),
        rationale="t",  # type: ignore[arg-type]
    )


# ---- pure validate ------------------------------------------------------------------------------


def test_validate_accepts_in_bounds() -> None:
    v = validate(act(lr=2e-4, n_epochs=4, lora_r=32), state())
    assert isinstance(v, Accepted) and v.estimate_usd == pytest.approx(0.1 + 0.4)
    assert isinstance(validate(act("more_data", "a"), state()), Accepted)
    assert isinstance(validate(act("stop"), state(round=3, remaining_usd=0.0)), Accepted)


@pytest.mark.parametrize(
    ("action", "kw", "needle"),
    [
        (act(lr=5e-4), {}, "lr"),
        (act(n_epochs=5), {}, "n_epochs"),
        (act(lora_r=64), {}, "lora_r"),
        (act(), {}, "no hyperparameter"),
        (act("more_data", "heldout_fam"), {}, "not an allowed"),
        (act("more_data"), {}, "needs a family"),
        (act("new_round"), {"round": 3}, "max_rounds"),
        (act("new_round"), {"remaining_usd": 0.05}, "exceeds remaining"),
        (act(n_epochs=2), {"train_rows": 400}, "min_planned_steps"),  # 100 x 2 = 200 < 300
    ],
)
def test_validate_rejects_with_reason(
    action: ControllerAction, kw: dict[str, Any], needle: str
) -> None:
    v = validate(action, state(**kw))
    assert isinstance(v, Rejected) and needle in v.reason


def test_steps_floor_exempt_in_dry_run() -> None:
    assert isinstance(validate(act(n_epochs=2), state(train_rows=40, dry_run=True)), Accepted)


# ---- pipeline -----------------------------------------------------------------------------------


def run(tmp_path: Path, bridge: AsyncBridge, response: dict[str, Any] | None = None, **kw: Any):
    store = Store(tmp_path / "store")
    dr = build_dry_run(NANO, bridge, **kw)
    if response is not None:
        dr.transport.controller_response = json.dumps(response)
    pipe = Pipeline(dr.pipeline_cfg, dr.config, dr.deps, store, "dry-t", say=lambda _s: None)
    return pipe, dr, store, pipe.run()


def decisions(store: Store) -> list[dict[str, Any]]:
    return [d for n, d in store.list_experiments("dry-t") if n == "controller_decision"]


def hp_of(dr: DryRun) -> list[dict[str, Any]]:
    return [dr.finetune._hps[j] for j in dr.finetune.created]


@pytest.mark.parametrize(
    ("response", "needle"),
    [
        ({"action": "adjust_hparams", "hparams": {"lr": 9e-4}, "rationale": "x"}, "lr"),
        ({"action": "more_data", "family": "no_such_family", "rationale": "x"}, "not an allowed"),
    ],
)
def test_rejected_falls_back_and_is_logged(
    tmp_path: Path, bridge: AsyncBridge, response: dict[str, Any], needle: str
) -> None:
    pipe, dr, store, rep = run(tmp_path, bridge, response, controller=True)
    d = decisions(store)
    assert d and not d[0]["accepted"] and needle in d[0]["reason"] and d[0]["fallback_used"]
    assert d[0]["proposal"]["action"] == response["action"]
    audit = [a for a in store.list_audit("dry-t") if a["actor"] == "controller"]
    assert audit and audit[0]["action"] == "controller_decision"
    assert len({json.dumps(h, sort_keys=True) for h in hp_of(dr)}) == 1  # pinned hparams unchanged
    assert rep["rounds"][0]["controller"]["accepted"] is False
    assert pipe._results["controller_r1"]["effective"]["action"] == "new_round"


def test_heldout_or_stress_family_rejected(tmp_path: Path, bridge: AsyncBridge) -> None:
    from distillery.taskpacks.sql.questions import stress_families

    fam = stress_families()[0]
    _, _, store, _ = run(
        tmp_path, bridge, {"action": "more_data", "family": fam, "rationale": "x"}, controller=True
    )
    assert "not an allowed" in decisions(store)[0]["reason"]


def test_over_budget_rejected(tmp_path: Path, bridge: AsyncBridge) -> None:
    # Ceiling basis: every fine-tune is estimated at $3, but the cap leaves < $3 after round 1.
    # The fallback (rule-based) round 2 then trips the ledger preflight itself: the rules, not the
    # controller, own that refusal.
    with pytest.raises(BudgetExceeded):
        run(tmp_path, bridge, None, controller=True, finetune_estimate_usd=3.0, run_cap_usd=4.0)
    d = decisions(Store(tmp_path / "store"))
    assert d and not d[0]["accepted"] and "exceeds remaining cap" in d[0]["reason"]
    assert d[0]["fallback_used"]


def test_valid_adjust_reaches_finetune_request(tmp_path: Path, bridge: AsyncBridge) -> None:
    resp = {"action": "adjust_hparams", "hparams": {"lr": 2e-4, "n_epochs": 4, "lora_r": 32},
            "rationale": "x"}  # fmt: skip
    _, dr, store, _ = run(tmp_path, bridge, resp, controller=True)
    assert decisions(store)[0]["accepted"]
    hps = hp_of(dr)
    assert hps[0]["learning_rate"] == 1e-4 and hps[0]["n_epochs"] == 3  # round 1 is pinned
    assert hps[1]["learning_rate"] == 2e-4 and hps[1]["n_epochs"] == 4 and hps[1]["lora_r"] == 32
    assert hps[1]["lora_alpha"] == 2 * hps[0]["lora_alpha"]  # alpha/r ratio kept


def test_controller_off_is_unchanged(tmp_path: Path, bridge: AsyncBridge) -> None:
    _, dr_off, store_off, rep_off = run(tmp_path / "off", bridge)
    assert not decisions(store_off)
    assert not any(c["schema"] == "ControllerAction" for c in dr_off.transport.calls)
    assert not any("controller" in s for s in rep_off["stages"])
    assert all("controller" not in r for r in rep_off["rounds"])
    from distillery.orchestrator import PipelineConfig

    assert PipelineConfig().controller is False


def test_resume_replays_cached_decision(tmp_path: Path, bridge: AsyncBridge) -> None:
    resp = {"action": "adjust_hparams", "hparams": {"lr": 2e-4}, "rationale": "first"}
    _, _, store, rep1 = run(tmp_path, bridge, resp, controller=True)
    n = len(decisions(store))
    dr2 = build_dry_run(NANO, bridge, controller=True)
    dr2.transport.controller_response = json.dumps(
        {"action": "stop", "rationale": "would differ if asked again"}
    )
    pipe2 = Pipeline(dr2.pipeline_cfg, dr2.config, dr2.deps, store, "dry-t", say=lambda _s: None)
    rep2 = pipe2.run()
    assert pipe2.ran == [] and not any(
        c["schema"] == "ControllerAction" for c in dr2.transport.calls
    )
    assert len(decisions(store)) == n  # replay does not log again
    assert rep2["rounds"] == rep1["rounds"]
