"""Server settings. ``create_app`` takes one of these so tests can inject fakes."""

from __future__ import annotations

import os
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from distillery.config import Config, load_config
from distillery.llm import LLMClient, make_openai_client

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
    dev_origin: str | None = None
    playground_llm: LLMClient | None = None
    # (job, log) -> exit code. None = run ``python -m distillery run`` in a subprocess.
    executor: Callable[..., int] | None = None
    playground_per_ip_per_hour: int = 10
    dry_run_per_ip_per_hour: int = 6
    max_body_bytes: int = 16_384
    heartbeat_s: float = 15.0
    poll_s: float = 1.0
    demo_db_seed: int = 0
    clock: Callable[[], float] = time.monotonic
    now: Callable[[], datetime] = field(default=_utcnow)


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
        dev_origin=e.get("DISTILLERY_DEV_ORIGIN") or None,
        playground_llm=default_playground_llm(config),
    )
