"""Shared fakes for the server tests. Everything here is fake; nothing touches the network."""

from __future__ import annotations

import json
import threading
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from pydantic import SecretStr

from distillery.config import Config, Price
from distillery.llm import LLMClient
from distillery.server.settings import SAMPLE_REPORT, ServerSettings
from distillery.server.worker import Job
from distillery.store import Store, atomic_write_bytes

ADMIN = "admin-token-XYZZY-123"
API_KEY = "sk-nebius-SECRET-987"
MODELS = {"planner": "fake-p", "teacher": "fake-t", "triage": "fake-n", "student": "fake-s"}
PRICE = Price(input_per_mtok=1.0, output_per_mtok=2.0, source="FAKE test price", date="2026-01-01")
QUESTION = "How many accounts are there?"
GOLD = "SELECT COUNT(*) FROM accounts"


def make_config(**kw: Any) -> Config:
    base: dict[str, Any] = {
        "nebius_api_key": SecretStr(API_KEY),
        "nebius_base_url": "https://example.invalid/v1",
        "admin_token": SecretStr(ADMIN),
        "model_ids": dict(MODELS),
        "prices": {m: PRICE for m in MODELS.values()},
        "playground_daily_cap_usd": 1.0,
    }
    base.update(kw)
    return Config(**base)


class FakeTransport:
    """Duck-typed async chat transport: always answers with ``sql``."""

    def __init__(self, sql: str = GOLD) -> None:
        self.sql = sql
        self.calls = 0
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    async def _create(self, **_: Any) -> Any:
        self.calls += 1
        text = f"```sql\n{self.sql}\n```"
        message = SimpleNamespace(content=text, refusal=None)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=message, finish_reason="stop")],
            usage=SimpleNamespace(prompt_tokens=1000, completion_tokens=100),
        )


def fake_llm(sql: str = GOLD) -> tuple[LLMClient, FakeTransport]:
    t = FakeTransport(sql)
    return LLMClient(t, MODELS), t  # type: ignore[arg-type]


def make_settings(tmp_path: Path, **kw: Any) -> ServerSettings:
    cfg = kw.pop("config", None) or make_config()
    return ServerSettings(
        root=tmp_path / "data",
        config=cfg,
        replay_dir=tmp_path / "replay",
        heartbeat_s=0.05,
        poll_s=0.01,
        **kw,
    )


def sample_report() -> dict[str, Any]:
    data: dict[str, Any] = json.loads(SAMPLE_REPORT.read_text(encoding="utf-8"))
    return data


def seed_finished_run(
    root: Path, run_id: str, *, examples: list[dict[str, Any]] | None = None, dry: bool = True
) -> Store:
    """Write what a finished pipeline leaves behind (store rows + report.json), using the sample
    report's shape. Test data, not results."""
    store = Store(root / "dry-runs" if dry else root)
    store.get_or_run(run_id, "schema", {}, lambda: {"ok": 1})
    store.get_or_run(
        run_id,
        "gold_crosscheck",
        {},
        lambda: {"accepted": [{"question": QUESTION, "gold_sql": GOLD, "requires_order": False}]},
    )
    store.get_or_run(
        run_id,
        "verifier_selftest",
        {},
        lambda: {"counters": {"tasks_sampled": 5, "corruptions_tested": 17}},
    )
    store.record_llm_call(run_id, "fake-t", 100, 10, 0.5)
    store.record_spend(run_id, "llm", "fake-t", 0.5, 100, 10)
    store.record_spend(run_id, "finetune", None, 2.0)
    report = sample_report()
    report["run_id"] = run_id
    report["dry_run"] = dry
    if examples is not None:
        # where the core really writes them (EvalReport.to_json -> report["evaluation"])
        report["evaluation"]["examples"] = examples
    atomic_write_bytes(store.run_dir(run_id) / "report.json", json.dumps(report).encode("utf-8"))
    return store


def scripted_executor(
    root: Path, gate: threading.Event | None = None
) -> Callable[[Job, Callable[[str], None]], int]:
    """Stands in for the pipeline subprocess: logs, optionally blocks on ``gate``, then leaves a
    finished run behind (or nothing when cancelled)."""

    def run(job: Job, log: Callable[[str], None]) -> int:
        log(f"starting {job.run_id} token={ADMIN}")  # the server must redact this
        store = Store(root / "dry-runs" if job.dry_run else root)
        store.get_or_run(job.run_id, "schema", {}, lambda: {"ok": 1})
        if gate is not None:
            job.interrupt = gate.set
            if job.cancel_requested:  # cancel may have landed before the interrupt was registered
                gate.set()
            gate.wait(10)
            if job.cancel_requested:
                store.close()
                return 130
        store.close()
        seed_finished_run(root, job.run_id, dry=job.dry_run).close()
        return 0

    return run
