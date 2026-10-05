"""A3: the Ultra controller. The planner may PROPOSE a round action; pure code decides.

``validate`` is a pure function of (action, state): no I/O, no clock, no model. A rejected proposal
falls back to the existing rule-based round decision (orchestrator). The bounds below are the
pre-registered ones (DECISIONS.md 2026-10-06) and are not configurable. The state shown to the
planner is built from DEV data only (accuracy per dev family, dev-failure clusters, budget, caps);
no sealed text (held-out, stress, human) is reachable from here. The gate never sees any of this.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

from pydantic import BaseModel, Field

# Pre-registered bounds (DECISIONS.md 2026-10-06, written before any controller run).
LR_CHOICES: tuple[float, ...] = (1e-4, 2e-4)
N_EPOCHS_CHOICES: tuple[int, ...] = (2, 3, 4)
LORA_R_CHOICES: tuple[int, ...] = (16, 32)

ActionName = Literal["more_data", "new_round", "adjust_hparams", "stop"]


class ControllerHParams(BaseModel):
    """Proposed hyperparameter overrides; ``None`` keeps the pinned value."""

    lr: float | None = None
    n_epochs: int | None = None
    lora_r: int | None = None

    def is_empty(self) -> bool:
        return self.lr is None and self.n_epochs is None and self.lora_r is None


class ControllerAction(BaseModel):
    action: ActionName
    family: str | None = None  # required for more_data; checked against the allowed families
    hparams: ControllerHParams = Field(default_factory=ControllerHParams)
    rationale: str = ""


@dataclass(frozen=True)
class ControllerState:
    """Everything validate needs (and, via ``prompt_view``, everything the planner sees)."""

    round: int
    max_rounds: int
    dev_acc: float
    dev_acc_by_family: Mapping[str, float]
    clusters: Sequence[Mapping[str, Any]]
    allowed_families: tuple[str, ...]  # train minus stress minus held-out
    remaining_usd: float  # min(run cap, project cap) minus spend
    data_usd: float  # estimated cost of one more targeted-data round
    ft_usd_per_epoch: float  # fine-tune estimate per epoch (price known) ...
    ft_usd_fixed: float | None  # ... or the operator ceiling (price unknown), per job
    current_lr: float | None
    current_n_epochs: int | None
    current_lora_r: int | None
    batch_size: int | None
    packing: bool | None
    train_rows: int
    min_planned_steps: int
    dry_run: bool
    bounds: Mapping[str, Sequence[float]] = field(
        default_factory=lambda: {
            "lr": LR_CHOICES,
            "n_epochs": N_EPOCHS_CHOICES,
            "lora_r": LORA_R_CHOICES,
        }  # fmt: skip
    )

    def prompt_view(self) -> dict[str, Any]:
        """Dev-derived facts only."""
        return {
            "round": self.round,
            "max_rounds": self.max_rounds,
            "dev_accuracy": round(self.dev_acc, 4),
            "dev_accuracy_by_family": {k: round(v, 4) for k, v in self.dev_acc_by_family.items()},
            "failure_clusters": [dict(c) for c in self.clusters],
            "allowed_families": list(self.allowed_families),
            "remaining_budget_usd": round(self.remaining_usd, 4),
            "current_hparams": {
                "lr": self.current_lr,
                "n_epochs": self.current_n_epochs,
                "lora_r": self.current_lora_r,
            },  # fmt: skip
            "hparam_bounds": {k: list(v) for k, v in self.bounds.items()},
        }


@dataclass(frozen=True)
class Accepted:
    action: ControllerAction
    estimate_usd: float


@dataclass(frozen=True)
class Rejected:
    reason: str


def estimate_usd(action: ControllerAction, state: ControllerState) -> float:
    """Cost of acting: one more targeted-data round plus the next fine-tune (``stop`` is free)."""
    if action.action == "stop":
        return 0.0
    if state.ft_usd_fixed is not None:
        ft = state.ft_usd_fixed
    else:
        epochs = action.hparams.n_epochs or state.current_n_epochs or 3
        ft = state.ft_usd_per_epoch * epochs
    return state.data_usd + ft


def _in(value: float, choices: Sequence[float]) -> bool:
    return any(math.isclose(value, c, rel_tol=1e-9) for c in choices)


def validate(action: ControllerAction, state: ControllerState) -> Accepted | Rejected:
    """Accept only an action that is inside the pre-registered bounds. All failures are reported."""
    if action.action == "stop":
        return Accepted(action, 0.0)
    why: list[str] = []
    if not state.round < state.max_rounds:
        why.append(f"round {state.round} is not below max_rounds={state.max_rounds}")
    hp = action.hparams
    if action.action == "more_data" and action.family is None:
        why.append("more_data needs a family")
    if action.family is not None and action.family not in state.allowed_families:
        why.append(
            f"family {action.family!r} is not an allowed training family "
            "(must be in train and not stress or held-out)"
        )
    if action.action == "adjust_hparams" and hp.is_empty():
        why.append("adjust_hparams proposed no hyperparameter")
    if hp.lr is not None and not _in(hp.lr, LR_CHOICES):
        why.append(f"lr {hp.lr!r} outside pre-registered {{{', '.join(map(str, LR_CHOICES))}}}")
    if hp.n_epochs is not None and hp.n_epochs not in N_EPOCHS_CHOICES:
        why.append(f"n_epochs {hp.n_epochs!r} outside pre-registered {list(N_EPOCHS_CHOICES)}")
    if hp.lora_r is not None and hp.lora_r not in LORA_R_CHOICES:
        why.append(f"lora_r {hp.lora_r!r} outside pre-registered {list(LORA_R_CHOICES)}")
    est = estimate_usd(action, state)
    if est > state.remaining_usd:
        why.append(f"estimated ${est:.4f} exceeds remaining cap ${state.remaining_usd:.4f}")
    if not state.dry_run and state.packing is False and state.batch_size:
        epochs = hp.n_epochs or state.current_n_epochs or 0
        steps = -(-state.train_rows // state.batch_size) * epochs
        if steps < state.min_planned_steps:
            why.append(
                f"planned optimizer steps {steps} below min_planned_steps={state.min_planned_steps}"
            )
    return Rejected("; ".join(why)) if why else Accepted(action, est)


def build_messages(state: ControllerState) -> list[dict[str, str]]:
    import json

    system = (
        "You are the round controller of a text-to-SQL distillation pipeline. Given dev-set "
        "results, propose ONE next action: more_data (extra verified training examples for one "
        "family), new_round (extra examples for all failing families), adjust_hparams (change "
        "lr / n_epochs / lora_r for the next fine-tune, within the bounds shown), or stop. "
        "Your proposal is checked by code and may be rejected; stay inside the bounds and "
        "allowed families. Give a short rationale."
    )
    user = json.dumps(state.prompt_view(), indent=2, sort_keys=True)
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


__all__ = [
    "LORA_R_CHOICES", "LR_CHOICES", "N_EPOCHS_CHOICES", "Accepted", "ControllerAction",
    "ControllerHParams", "ControllerState", "Rejected", "build_messages", "estimate_usd",
    "validate",
]  # fmt: skip
