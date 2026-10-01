"""Server settings. ``create_app`` takes one of these so tests can inject fakes."""

from __future__ import annotations

import os
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import urlsplit

from distillery.config import Config, load_config
from distillery.llm import LLMClient, make_openai_client
from distillery.server.local_models import LocalModels
from distillery.server.local_models import from_env as local_models_from_env
from distillery.taskpacks.sql import schema as sql_schema

_REPO_ROOT = Path(__file__).resolve().parents[3]
SAMPLE_REPORT = _REPO_ROOT / "docs" / "fixtures" / "sample-dry-run-report.json"


def _utcnow() -> datetime:
    return datetime.now(UTC)


@dataclass
class ServerSettings:
    root: Path
    config: Config
    replay_dir: Path = Path("replay")
    sample_report: Path | None = SAMPLE_REPORT
    frontend_dist: Path | None = None
    # Exact origins allowed to call the API cross-origin (CORS). Empty = same origin only.
    allowed_origins: tuple[str, ...] = ()
    playground_llm: LLMClient | None = None
    # (job, log) -> exit code. None = run ``python -m distillery run`` in a subprocess.
    executor: Callable[..., int] | None = None
    playground_per_ip_per_hour: int = 10
    # Base + student on sandbox CPU (None = not wired). Their price is unknown, so they have their
    # own caps: questions that use them, per IP per hour and per UTC day (in memory, per process).
    playground_local: LocalModels | None = None
    playground_student_per_ip_per_hour: int = 3
    playground_student_daily_cap: int = 60
    dry_run_per_ip_per_hour: int = 6
    max_body_bytes: int = 16_384
    sse_max_streams: int = 32
    sse_max_streams_per_ip: int = 4
    max_pending_jobs: int = 5  # queued + running, enforced for anonymous dry runs
    shutdown_grace_s: float = 30.0  # SIGINT -> SIGKILL grace for a live child (cancel + shutdown)
    # Peer addresses whose X-Forwarded-For is believed. Default: none, the header is ignored.
    trusted_proxies: tuple[str, ...] = ()
    heartbeat_s: float = 15.0
    poll_s: float = 1.0
    demo_db_seed: int = 0
    clock: Callable[[], float] = time.monotonic
    now: Callable[[], datetime] = field(default=_utcnow)


_LOCAL_HOSTS = frozenset({"localhost", "127.0.0.1", "::1"})


def parse_allowed_origins(raw: str | None) -> tuple[str, ...]:
    """Comma-separated exact origins. No wildcards, no paths; https unless the host is local.
    Raises ValueError on a bad entry so a typo fails at startup instead of silently opening CORS."""
    out: list[str] = []
    for item in (raw or "").split(","):
        o = item.strip()
        if not o:
            continue
        try:
            u = urlsplit(o)
            host, _ = u.hostname, u.port  # .port raises ValueError on a bad port
        except ValueError:
            raise ValueError(f"invalid origin in DISTILLERY_ALLOWED_ORIGINS: {o!r}") from None
        if "*" in o or u.scheme not in ("http", "https") or not host:
            raise ValueError(f"origin must be an exact http(s) origin, no wildcard: {o!r}")
        if u.path or u.query or u.fragment or u.username or u.password:
            raise ValueError(f"origin must be scheme://host[:port] only: {o!r}")
        if u.scheme == "http" and host not in _LOCAL_HOSTS:
            raise ValueError(f"http origin is only allowed for localhost: {o!r}")
        origin = f"{u.scheme}://{u.netloc}".lower()
        if origin not in out:
            out.append(origin)
    return tuple(out)


def default_playground_llm(config: Config) -> LLMClient | None:
    """Teacher client, only when a key, base URL and teacher model are all configured.
    # UNVERIFIED against a real endpoint (no credentials)."""
    if config.nebius_api_key is None or not config.nebius_base_url:
        return None
    if not config.model_ids.get("teacher"):
        return None
    transport = make_openai_client(config.nebius_base_url, config.nebius_api_key.get_secret_value())
    return LLMClient(transport, config.model_ids)


def settings_from_env(
    env: Mapping[str, str] | None = None, root: Path | None = None
) -> ServerSettings:
    e = os.environ if env is None else env
    config = load_config(e)
    dist = _REPO_ROOT / "frontend" / "dist"
    return ServerSettings(
        root=root or Path(e.get("DISTILLERY_HOME") or ".distillery"),
        config=config,
        replay_dir=Path(e.get("DISTILLERY_REPLAY_DIR") or "replay"),
        frontend_dist=dist if (dist / "index.html").exists() else None,
        # DISTILLERY_DEV_ORIGIN is a deprecated alias, read only when the new variable is unset.
        allowed_origins=parse_allowed_origins(
            e.get("DISTILLERY_ALLOWED_ORIGINS")
            if e.get("DISTILLERY_ALLOWED_ORIGINS") is not None
            else e.get("DISTILLERY_DEV_ORIGIN")
        ),
        playground_llm=default_playground_llm(config),
        playground_local=local_models_from_env(
            config, e, demo_db=lambda: sql_schema.build_database(0)
        ),
        playground_student_per_ip_per_hour=int(
            e.get("DISTILLERY_PLAYGROUND_STUDENT_PER_IP_PER_HOUR") or 3
        ),
        playground_student_daily_cap=int(e.get("DISTILLERY_PLAYGROUND_STUDENT_DAILY_CAP") or 60),
        trusted_proxies=tuple(
            p.strip() for p in (e.get("DISTILLERY_TRUSTED_PROXIES") or "").split(",") if p.strip()
        ),
    )
