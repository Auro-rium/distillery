"""Settings and small interfaces of the OpenAI-compatible gateway (kept free of heavy imports so
``settings.py`` can use them)."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

_REPO_ROOT = Path(__file__).resolve().parents[3]
SERVING_PRICE_FILE = _REPO_ROOT / "deploy" / "serving_price.json"


@dataclass(frozen=True)
class UpstreamResult:
    """What an upstream (Token Factory) chat call returns; usage is the provider's own count."""

    text: str
    input_tokens: int
    output_tokens: int
    finish_reason: str | None = "stop"


@dataclass(frozen=True)
class EmbedResult:
    vectors: list[list[float]]
    prompt_tokens: int


class Upstream(Protocol):
    """The real provider. ``complete`` takes only vetted ``params`` (``max_tokens``,
    ``temperature``, ``top_p``, ``stop``, ``seed``); ``embed`` serves the router's nn_sim."""

    async def embed(self, model: str, texts: Sequence[str]) -> EmbedResult: ...

    async def complete(
        self, model: str, messages: Sequence[Mapping[str, Any]], **params: Any
    ) -> UpstreamResult: ...


@dataclass
class GatewaySettings:
    # directory holding model.json + cert.json (+ one embed_index_*.bin) from `distillery router`
    router_dir: Path | None = None
    student_daily_max: int = 200  # DISTILLERY_GATEWAY_STUDENT_DAILY_MAX (sandbox calls per UTC day)
    per_ip_per_hour: int = 30  # requests per address per hour (the playground limiter class)
    # models the public may use through passthrough; empty = just the configured triage model
    passthrough_models: tuple[str, ...] = ()
    passthrough_max_tokens: int = 256  # clamp for the public (and the default when none is sent)
    admin_max_tokens: int = 4096  # clamp for admin passthrough
    skill_max_tokens: int = 2048  # output cap of the teacher call behind a skill request
    max_body_bytes: int = 65_536
    upstream: Upstream | None = None  # None = built from the Nebius config when possible
    # tests/overrides: a free, synchronous nn_sim. None = embed the question through the upstream
    # (priced, reserved against the daily cap, recorded) and compare with the router dir's index
    nn_sim: Callable[[str], float] | None = None
    serving_price_file: Path | None = SERVING_PRICE_FILE
    ip_salt: str | None = None  # None = random per process


def _int(env: Mapping[str, str], name: str, default: int, minimum: int = 0) -> int:
    raw = (env.get(name) or "").strip()
    if not raw:
        return default
    try:
        v = int(raw)
    except ValueError:
        raise ValueError(f"{name} must be an integer, got {raw!r}") from None
    if v < minimum:
        raise ValueError(f"{name} must be >= {minimum}, got {raw!r}")
    return v


def gateway_from_env(env: Mapping[str, str]) -> GatewaySettings:
    d = GatewaySettings()
    rdir = (env.get("DISTILLERY_ROUTER_DIR") or "").strip()
    price = (env.get("DISTILLERY_SERVING_PRICE_FILE") or "").strip()
    return GatewaySettings(
        router_dir=Path(rdir) if rdir else None,
        student_daily_max=_int(env, "DISTILLERY_GATEWAY_STUDENT_DAILY_MAX", d.student_daily_max),
        per_ip_per_hour=_int(env, "DISTILLERY_GATEWAY_PER_IP_PER_HOUR", d.per_ip_per_hour, 1),
        passthrough_models=tuple(
            m.strip()
            for m in (env.get("DISTILLERY_GATEWAY_PASSTHROUGH_MODELS") or "").split(",")
            if m.strip()
        ),
        passthrough_max_tokens=_int(
            env, "DISTILLERY_GATEWAY_PASSTHROUGH_MAX_TOKENS", d.passthrough_max_tokens, 1
        ),
        serving_price_file=Path(price) if price else SERVING_PRICE_FILE,
        ip_salt=(env.get("DISTILLERY_GATEWAY_IP_SALT") or None),
    )
