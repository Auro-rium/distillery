"""Configuration: environment loader, price table, gate thresholds, admin token check."""

from __future__ import annotations

import hmac
import json
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Literal, get_args

from pydantic import BaseModel, ConfigDict, Field, SecretStr, ValidationError

Role = Literal["planner", "teacher", "triage", "student"]
ROLES: tuple[str, ...] = get_args(Role)


class ConfigError(RuntimeError):
    """Raised for missing or invalid configuration."""


class Price(BaseModel):
    """Per-million-token price for one model. Always carries provenance."""

    model_config = ConfigDict(frozen=True)

    input_per_mtok: float = Field(ge=0)
    output_per_mtok: float = Field(ge=0)
    source: str = Field(min_length=1)
    date: str = Field(min_length=1)


class GateThresholds(BaseModel):
    model_config = ConfigDict(frozen=True)

    ratio_lower_bound_min: float = 0.85
    mcnemar_alpha: float = 0.05
    bootstrap_resamples: int = Field(default=10000, ge=1)
    seed: int = 1234


class Config(BaseModel):
    nebius_api_key: SecretStr | None = None
    nebius_base_url: str | None = None
    admin_token: SecretStr | None = None
    run_cap_usd: float = 10.0
    project_cap_usd: float = 40.0
    playground_daily_cap_usd: float = 1.0
    model_ids: dict[str, str] = Field(default_factory=dict)
    prices: dict[str, Price] = Field(default_factory=dict)
    gate: GateThresholds = Field(default_factory=GateThresholds)

    def require_model(self, role: str) -> str:
        if role not in ROLES:
            raise ConfigError(f"unknown model role {role!r}; expected one of {ROLES}")
        model = self.model_ids.get(role)
        if not model:
            raise ConfigError(
                f"model id for role {role!r} is not configured; set DISTILLERY_MODEL_{role.upper()}"
            )
        return model

    def verify_admin_token(self, supplied: str) -> bool:
        return verify_admin_token(supplied, self.admin_token)


def verify_admin_token(supplied: str, expected: SecretStr | str | None) -> bool:
    """Constant-time comparison. Returns False if no admin token is configured."""
    if expected is None:
        return False
    exp = expected.get_secret_value() if isinstance(expected, SecretStr) else expected
    if not exp:
        return False
    return hmac.compare_digest(supplied.encode("utf-8"), exp.encode("utf-8"))


def load_prices(path: str | Path) -> dict[str, Price]:
    try:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as e:
        raise ConfigError(f"cannot read prices file {str(path)!r}: {e}") from e
    if not isinstance(raw, dict):
        raise ConfigError("prices file must be a JSON object keyed by model id")
    try:
        return {str(k): Price.model_validate(v) for k, v in raw.items()}
    except ValidationError as e:
        raise ConfigError(f"invalid price entry: {e}") from e


def _float_env(env: Mapping[str, str], name: str, default: float) -> float:
    val = env.get(name)
    if val is None or val == "":
        return default
    try:
        return float(val)
    except ValueError as e:
        raise ConfigError(f"{name} must be a number, got {val!r}") from e


def load_config(env: Mapping[str, str] | None = None) -> Config:
    e = os.environ if env is None else env
    key = e.get("NEBIUS_API_KEY")
    token = e.get("DISTILLERY_ADMIN_TOKEN")
    models = {
        r: e[f"DISTILLERY_MODEL_{r.upper()}"]
        for r in ROLES
        if e.get(f"DISTILLERY_MODEL_{r.upper()}")
    }
    prices_file = e.get("DISTILLERY_PRICES_FILE")
    gate_kwargs: dict[str, float | int] = {}
    if e.get("DISTILLERY_SEED"):
        try:
            gate_kwargs["seed"] = int(e["DISTILLERY_SEED"])
        except ValueError as ex:
            raise ConfigError("DISTILLERY_SEED must be an integer") from ex
    return Config(
        nebius_api_key=SecretStr(key) if key else None,
        nebius_base_url=e.get("NEBIUS_BASE_URL") or None,
        admin_token=SecretStr(token) if token else None,
        run_cap_usd=_float_env(e, "DISTILLERY_RUN_CAP_USD", 10.0),
        project_cap_usd=_float_env(e, "DISTILLERY_PROJECT_CAP_USD", 40.0),
        playground_daily_cap_usd=_float_env(e, "DISTILLERY_PLAYGROUND_DAILY_CAP_USD", 1.0),
        model_ids=models,
        prices=load_prices(prices_file) if prices_file else {},
        gate=GateThresholds(**gate_kwargs),  # type: ignore[arg-type]
    )
