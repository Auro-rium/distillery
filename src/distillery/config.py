"""Configuration: environment loader, price table, gate thresholds, admin token check."""

from __future__ import annotations

import hmac
import json
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Literal, get_args

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    SecretStr,
    ValidationError,
    model_validator,
)

Role = Literal["planner", "teacher", "triage", "student"]
ROLES: tuple[str, ...] = get_args(Role)


class ConfigError(RuntimeError):
    """Raised for missing or invalid configuration."""


def _provenance(data: Any) -> Any:
    """``date`` and ``as_of`` are the same fact (when the price was read); a price file may use
    either name and both are filled, so older files (``date`` only) stay valid."""
    if isinstance(data, dict):
        d, a = data.get("date"), data.get("as_of")
        if d is None and a is not None:
            data = {**data, "date": a}
        elif a is None and d is not None:
            data = {**data, "as_of": d}
    return data


def _check_provenance(source: str, date: str, as_of: str) -> None:
    if not source.strip() or not date.strip() or not as_of.strip():
        raise ValueError("price entries need non-blank source and as_of/date provenance")
    if date != as_of:
        raise ValueError(f"price entry has conflicting date {date!r} and as_of {as_of!r}")


class Price(BaseModel):
    """Per-million-token price for one model. Always carries provenance."""

    model_config = ConfigDict(frozen=True)

    input_per_mtok: float = Field(ge=0)
    output_per_mtok: float = Field(ge=0)
    source: str = Field(min_length=1)
    date: str = Field(min_length=1)


class FinetunePrice(BaseModel):
    """Fine-tune price for one base model, per million TRAINED tokens (as the job reports them)."""

    model_config = ConfigDict(frozen=True)

    usd_per_mtok_trained_tokens: float = Field(ge=0)
    source: str = Field(min_length=1)
    date: str = Field(min_length=1)
    as_of: str = Field(min_length=1)

    @model_validator(mode="before")
    @classmethod
    def _fill_provenance(cls, data: Any) -> Any:
        return _provenance(data)

    @model_validator(mode="after")
    def _provenance_ok(self) -> FinetunePrice:
        _check_provenance(self.source, self.date, self.as_of)
        return self


class SandboxPrice(BaseModel):
    """Sandbox compute price. Exactly one of the two rates; both are USD per second."""

    model_config = ConfigDict(frozen=True)

    usd_per_cpu_second: float | None = Field(default=None, ge=0)
    usd_per_second: float | None = Field(default=None, ge=0)
    source: str = Field(min_length=1)
    date: str = Field(min_length=1)
    as_of: str = Field(min_length=1)

    @model_validator(mode="before")
    @classmethod
    def _fill_provenance(cls, data: Any) -> Any:
        return _provenance(data)

    @model_validator(mode="after")
    def _provenance_ok(self) -> SandboxPrice:
        _check_provenance(self.source, self.date, self.as_of)
        return self

    @model_validator(mode="after")
    def _exactly_one_rate(self) -> SandboxPrice:
        if (self.usd_per_cpu_second is None) == (self.usd_per_second is None):
            raise ValueError(
                "sandbox price needs exactly one of usd_per_cpu_second, usd_per_second"
            )
        return self

    @property
    def per_second(self) -> float:
        """The rate applied to measured wall-clock seconds of one sandbox."""
        rate = self.usd_per_cpu_second if self.usd_per_second is None else self.usd_per_second
        assert rate is not None  # noqa: S101 - guaranteed by the validator
        return rate


class PriceFile(BaseModel):
    """A parsed price file: legacy per-model LLM prices plus the typed sections."""

    model_config = ConfigDict(frozen=True)

    llm: dict[str, Price] = Field(default_factory=dict)
    finetune: dict[str, FinetunePrice] = Field(default_factory=dict)
    sandbox: SandboxPrice | None = None


class GateThresholds(BaseModel):
    model_config = ConfigDict(frozen=True)

    ratio_lower_bound_min: float = 0.85
    mcnemar_alpha: float = 0.05
    bootstrap_resamples: int = Field(default=10000, ge=1)
    seed: int = 1234


class Config(BaseModel):
    nebius_api_key: SecretStr | None = None
    nebius_base_url: str | None = None
    nebius_project_id: SecretStr | None = None  # Sandboxes need it (the `Project` header)
    admin_token: SecretStr | None = None
    run_cap_usd: float = 10.0
    project_cap_usd: float = 40.0
    playground_daily_cap_usd: float = 1.0
    model_ids: dict[str, str] = Field(default_factory=dict)
    prices: dict[str, Price] = Field(default_factory=dict)
    finetune_prices: dict[str, FinetunePrice] = Field(default_factory=dict)
    sandbox_price: SandboxPrice | None = None
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


PRICE_SECTIONS = ("finetune", "sandbox")  # reserved top-level keys of the price file


def load_price_file(path: str | Path) -> PriceFile:
    """Read a price file. Flat per-model LLM entries stay valid; ``finetune`` and ``sandbox``
    are typed sections. Every entry must carry ``source`` and ``date``."""
    try:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as e:
        raise ConfigError(f"cannot read prices file {str(path)!r}: {e}") from e
    if not isinstance(raw, dict):
        raise ConfigError("prices file must be a JSON object keyed by model id")
    ft_raw = raw.get("finetune", {})
    if not isinstance(ft_raw, dict):
        raise ConfigError("prices file: 'finetune' must be an object keyed by base model id")
    try:
        llm = {str(k): Price.model_validate(v) for k, v in raw.items() if k not in PRICE_SECTIONS}
        finetune = {str(k): FinetunePrice.model_validate(v) for k, v in ft_raw.items()}
        sb_raw = raw.get("sandbox")
        sandbox = SandboxPrice.model_validate(sb_raw) if sb_raw is not None else None
    except ValidationError as e:
        raise ConfigError(f"invalid price entry: {e}") from e
    return PriceFile(llm=llm, finetune=finetune, sandbox=sandbox)


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
    project = e.get("NEBIUS_PROJECT_ID") or e.get("NEBIUS_AI_PROJECT")  # official name first
    prices_file = e.get("DISTILLERY_PRICES_FILE")
    price_file = load_price_file(prices_file) if prices_file else PriceFile()
    gate_kwargs: dict[str, float | int] = {}
    if e.get("DISTILLERY_SEED"):
        try:
            gate_kwargs["seed"] = int(e["DISTILLERY_SEED"])
        except ValueError as ex:
            raise ConfigError("DISTILLERY_SEED must be an integer") from ex
    return Config(
        nebius_api_key=SecretStr(key) if key else None,
        nebius_base_url=e.get("NEBIUS_BASE_URL") or None,
        nebius_project_id=SecretStr(project) if project else None,
        admin_token=SecretStr(token) if token else None,
        run_cap_usd=_float_env(e, "DISTILLERY_RUN_CAP_USD", 10.0),
        project_cap_usd=_float_env(e, "DISTILLERY_PROJECT_CAP_USD", 40.0),
        playground_daily_cap_usd=_float_env(e, "DISTILLERY_PLAYGROUND_DAILY_CAP_USD", 1.0),
        model_ids=models,
        prices=dict(price_file.llm),
        finetune_prices=dict(price_file.finetune),
        sandbox_price=price_file.sandbox,
        gate=GateThresholds(**gate_kwargs),  # type: ignore[arg-type]
    )
