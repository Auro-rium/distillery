"""Playground: teacher answers when configured; every SQL is verified by read-only execution."""

from __future__ import annotations

import threading
from datetime import UTC, datetime
from typing import Any

from starlette.concurrency import run_in_threadpool

from distillery.budget import PLAYGROUND, BudgetExceeded, Ledger, UnknownPriceError
from distillery.llm import LLMError
from distillery.prompts import build_messages, extract_sql
from distillery.server.limits import SlidingWindow
from distillery.server.reader import RunReader
from distillery.server.settings import ServerSettings
from distillery.taskpacks.sql import schema as sql_schema
from distillery.taskpacks.sql.runner import ExecOutcome, run_select
from distillery.taskpacks.sql.verifier import compare_outcomes

MAX_QUESTION_CHARS = 500
EST_OUTPUT_TOKENS = 256
PREVIEW_ROWS = 5
_UNAVAILABLE = {
    "base": "base model is not served by this demo (serving path unverified, spike S4)",
    "student": "student serving path not deployed (spike S4)",
}


def _today() -> str:
    return datetime.now(UTC).date().isoformat()


class DemoBudgetExhaustedError(RuntimeError):
    pass


def _cell(v: Any) -> Any:
    if isinstance(v, bytes):
        return v.hex()
    if isinstance(v, str) and len(v) > 200:
        return v[:200] + "..."
    return v


class Playground:
    def __init__(self, settings: ServerSettings, reader: RunReader) -> None:
        self.s = settings
        self.reader = reader
        self.ip_limit = SlidingWindow(settings.playground_per_ip_per_hour, 3600.0, settings.clock)
        self.ledger = Ledger.from_config("playground", settings.config, reader.real)
        self._db: bytes | None = None
        self._db_lock = threading.Lock()
        self._gold: dict[str, tuple[str, bool]] = {}
        self._gold_runs: set[str] = set()
        self._reserve_lock = threading.Lock()
        self._reserved = 0.0  # estimates of in-flight calls, not yet settled

    # ---- availability -----------------------------------------------------
    def teacher_unavailable_reason(self) -> str | None:
        cfg = self.s.config
        if self.s.playground_llm is None:
            return "teacher is not configured on this server (replay-only mode)"
        model = cfg.model_ids.get("teacher")
        if not model:
            return "no teacher model id configured"
        if model not in cfg.prices:
            return "no price configured for the teacher model; refusing to spend blind"
        return None

    # ---- verification -----------------------------------------------------
    def demo_db(self) -> bytes:
        with self._db_lock:
            if self._db is None:
                self._db = sql_schema.build_database(self.s.demo_db_seed)
            return self._db

    def _known_gold(self) -> dict[str, tuple[str, bool]]:
        """Gold SQL of every task from completed runs on the demo DB (real template gold)."""
        for item in self.reader.list_runs():
            rid = item["run_id"]
            if item["status"] != "complete" or rid in self._gold_runs:
                continue
            self._gold_runs.add(rid)
            report = self.reader.read_report(rid) or {}
            pipe = (report.get("config") or {}).get("pipeline") or {}
            if pipe.get("db_seed") != self.s.demo_db_seed:
                continue
            out = self.reader.stage_output(rid, "gold_crosscheck")
            for t in (out or {}).get("accepted", []) if isinstance(out, dict) else []:
                if isinstance(t, dict) and t.get("question") and t.get("gold_sql"):
                    self._gold.setdefault(
                        str(t["question"]).strip(),
                        (str(t["gold_sql"]), bool(t.get("requires_order"))),
                    )
        return self._gold

    def _verify(self, question: str, sql: str) -> tuple[bool | None, ExecOutcome]:
        db = self.demo_db()
        outcome = run_select(db, sql)
        gold = self._known_gold().get(question.strip())
        if gold is None:
            return None, outcome
        return compare_outcomes(outcome, run_select(db, gold[0]), gold[1]).ok, outcome

    # ---- request ----------------------------------------------------------
    async def answer(self, question: str) -> dict[str, Any]:
        results: dict[str, dict[str, Any]] = {
            role: _result(False, reason) for role, reason in _UNAVAILABLE.items()
        }
        cost = 0.0
        reason = self.teacher_unavailable_reason()
        if reason is not None:
            results["teacher"] = _result(False, reason)
        else:
            results["teacher"], cost = await self._teacher(question)
        order = {k: results[k] for k in ("teacher", "base", "student")}
        note = (
            "Only the teacher is served here; base and student are shown as unavailable, not faked."
        )
        return {"results": order, "cost_usd": cost, "note": note}

    def _reserve(self, estimate: float) -> None:
        """Atomically check (store spend + in-flight reservations + this estimate) against the
        daily cap and reserve the estimate. Spend is re-read from the store every time, so other
        processes' spend and our own settled calls are both counted."""
        with self._reserve_lock:
            spent = self.reader.real.spend_on_day(_today(), PLAYGROUND)
            spent = max(spent, self.ledger.playground_spent())
            pending = self._reserved + estimate
            cap = self.s.config.playground_daily_cap_usd
            if spent + pending > cap:
                raise BudgetExceeded(
                    f"playground daily cap ${cap:.2f}: spent ${spent:.4f} "
                    f"+ in-flight/est ${pending:.4f}"
                )
            self.ledger.preflight_playground(pending)  # project-wide cap too
            self._reserved += estimate

    def _release(self, estimate: float) -> None:
        with self._reserve_lock:
            self._reserved = max(0.0, self._reserved - estimate)

    async def _teacher(self, question: str) -> tuple[dict[str, Any], float]:
        llm = self.s.playground_llm
        if llm is None:  # guarded by teacher_unavailable_reason; explicit, not an assert
            return _result(False, "teacher is not configured"), 0.0
        model = self.s.config.model_ids["teacher"]
        messages = build_messages(
            question, sql_schema.schema_ddl(self.s.demo_db_seed), role="eval_teacher"
        )
        in_est = sum(len(m["content"]) for m in messages) // 3
        try:
            estimate = self.ledger.estimate_llm_cost(model, in_est, EST_OUTPUT_TOKENS)
            self._reserve(estimate)
        except BudgetExceeded as exc:
            raise DemoBudgetExhaustedError(str(exc)) from exc
        except UnknownPriceError as exc:
            return _result(False, str(exc)), 0.0
        try:
            try:
                res = await llm.chat("teacher", messages, purpose="playground")
            except LLMError as exc:
                return _result(True, None, error=f"teacher call failed: {type(exc).__name__}"), 0.0
            usd = self.ledger.estimate_llm_cost(model, res.input_tokens, res.output_tokens)
            # Record BEFORE releasing the reservation so the spend is never invisible.
            self.ledger.record(PLAYGROUND, model, usd, res.input_tokens, res.output_tokens)
        finally:
            self._release(estimate)
        sql = extract_sql(res.text)
        if sql is None:
            return _result(True, None, error="no SQL found in the model output"), usd
        verified, outcome = await run_in_threadpool(self._verify, question, sql)
        preview = None
        if outcome.ok:
            preview = {
                "columns": list(outcome.columns),
                "rows": [[_cell(c) for c in r] for r in outcome.rows[:PREVIEW_ROWS]],
                "row_count": len(outcome.rows),
            }
        return (
            _result(True, None, sql=sql, verified=verified, rows=preview,
                    error=None if outcome.ok else outcome.error[:300]),
            usd,
        )  # fmt: skip


def _result(
    available: bool,
    reason: str | None,
    *,
    sql: str | None = None,
    verified: bool | None = None,
    rows: dict[str, Any] | None = None,
    error: str | None = None,
) -> dict[str, Any]:
    return {
        "available": available,
        "reason": reason,
        "sql": sql,
        "verified": verified,
        "rows_preview": rows,
        "error": error,
    }
