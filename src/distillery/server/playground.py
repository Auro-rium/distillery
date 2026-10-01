"""Playground: teacher answers when configured; every SQL is verified by read-only execution."""

from __future__ import annotations

import asyncio
import threading
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Literal

from starlette.concurrency import run_in_threadpool

from distillery.budget import PLAYGROUND, BudgetExceeded, Ledger, UnknownPriceError
from distillery.finetune import sha256_file
from distillery.llm import LLMError
from distillery.orchestrator import artifact_from_json
from distillery.prompts import build_messages, extract_sql
from distillery.server.limits import SlidingWindow
from distillery.server.local_models import DISABLED_REASON, LocalModels
from distillery.server.reader import RunReader, valid_run_id
from distillery.server.settings import ServerSettings
from distillery.student import StudentServer, StudentServingError
from distillery.taskpacks.sql import schema as sql_schema
from distillery.taskpacks.sql.runner import ExecOutcome, run_select
from distillery.taskpacks.sql.verifier import compare_outcomes

MAX_QUESTION_CHARS = 500
EST_OUTPUT_TOKENS = 256
PREVIEW_ROWS = 5
DEMO_DB_REF = "playground-demo"
_RETRY_STATUS_S = 30.0  # a failed student lookup (missing run, bad adapter) is retried after this
State = Literal["ok", "unavailable", "rate_limited", "disabled", "error"]
LOCAL_ROLES = ("base", "student")
NOTE = (
    "Teacher runs on Token Factory (priced, capped). Base and student run on sandbox CPU when this "
    "server has them enabled; their cost is not measured and is not included in cost_usd. Every "
    "model that cannot answer says why; nothing is faked."
)


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
        self._gold_lock = threading.Lock()  # verification runs in several threads at once
        self._reserve_lock = threading.Lock()
        self._reserved = 0.0  # estimates of in-flight calls, not yet settled
        self.student_ip_limit = SlidingWindow(
            settings.playground_student_per_ip_per_hour, 3600.0, settings.clock
        )
        self._daily_lock = threading.Lock()
        self._daily: tuple[str, int] = ("", 0)  # (UTC date, student-using questions so far)
        self._student_lock = threading.Lock()
        self._student: tuple[StudentServer | None, str | None, float] | None = None

    def close(self) -> None:
        if self.s.playground_local is not None:
            self.s.playground_local.close()

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

    def _adapter_files(self, run_id: str) -> list[Path]:
        """The stored adapter of ``run_id``'s candidate round, files checked on disk. Raises
        ``_Unavailable`` with an honest reason. Never touches the sandbox."""
        if not valid_run_id(run_id):
            raise _Unavailable(f"invalid student run id {run_id!r}")
        report = self.reader.read_report(run_id)
        if report is None:
            raise _Unavailable(f"student run {run_id!r} has no report on this server")
        rnd = report.get("candidate_round")
        out = self.reader.stage_output(run_id, f"finetune_r{rnd}")
        if not isinstance(out, dict) or not isinstance(out.get("artifact"), dict):
            raise _Unavailable(f"run {run_id!r} has no stored adapter artifact (round {rnd})")
        run_dir = self.reader.store_for(run_id).run_dir(run_id)
        try:
            art = artifact_from_json(out["artifact"], run_dir)
        except (KeyError, TypeError, ValueError):
            raise _Unavailable(f"run {run_id!r}: stored adapter artifact is malformed") from None
        base = run_dir.resolve()
        files: list[Path] = []
        for f in art.files:
            path = f.path.resolve()
            if not path.is_relative_to(base) or not path.is_file():
                raise _Unavailable(f"adapter file {f.path.name!r} of run {run_id!r} is missing")
            if sha256_file(path) != f.sha256:
                raise _Unavailable(f"adapter file {f.path.name!r} does not match its stored sha256")
            files.append(path)
        return files

    def _student_status(self, loc: LocalModels) -> tuple[StudentServer | None, str | None]:
        with self._student_lock:
            now = self.s.clock()
            cached = self._student
            if cached is not None and (cached[0] is not None or now < cached[2]):
                return cached[0], cached[1]
            server: StudentServer | None = None
            reason: str | None = None
            if loc.student_run is None:
                reason = loc.student_unset_reason
            elif loc.student_factory is None:
                reason = "student serving is not wired on this server"
            else:
                try:
                    server = loc.student_factory(self._adapter_files(loc.student_run))
                except _Unavailable as exc:
                    reason = str(exc)
                except (StudentServingError, ValueError) as exc:
                    reason = f"adapter rejected: {str(exc)[:300]}"
            self._student = (server, reason, now + _RETRY_STATUS_S)
            return server, reason

    def local_status(self, role: str) -> tuple[State, str | None, StudentServer | None]:
        """(state, reason, server) of the base or the student. Cheap after the first call."""
        loc = self.s.playground_local
        if loc is None or not loc.enabled:
            return "disabled", (loc.disabled_reason if loc else DISABLED_REASON), None
        if role == "base":
            if loc.base is None:
                return "unavailable", loc.base_reason or "base model is not available", None
            return "ok", None, loc.base
        server, reason = self._student_status(loc)
        if server is None:
            return "unavailable", reason or "student is not available", None
        return "ok", None, server

    def statuses(self) -> dict[str, dict[str, Any]]:
        """Per-role state and reason for /api/config."""
        t = self.teacher_unavailable_reason()
        out = {"teacher": {"state": "unavailable" if t else "ok", "reason": t}}
        for role in LOCAL_ROLES:
            state, reason, _ = self.local_status(role)
            out[role] = {"state": state, "reason": reason}
        return out

    def any_available(self) -> bool:
        return any(v["state"] == "ok" for v in self.statuses().values())

    def _check_student_caps(self, ip: str) -> tuple[float, str] | None:
        """Consume one question's worth of the student caps, or say (retry seconds, why)."""
        now = self.s.now()
        today = now.astimezone(UTC).date().isoformat()
        with self._daily_lock:
            if self._daily[0] != today:
                self._daily = (today, 0)
            if self._daily[1] >= self.s.playground_student_daily_cap:
                midnight = datetime.combine(
                    now.astimezone(UTC).date() + timedelta(days=1), datetime.min.time(), UTC
                )
                secs = max(1.0, (midnight - now.astimezone(UTC)).total_seconds())
                return secs, (
                    f"daily cap of {self.s.playground_student_daily_cap} sandbox-model questions "
                    "reached; resets at 00:00 UTC"
                )
            retry = self.student_ip_limit.hit(ip)
            if retry is not None:
                return float(int(retry) + 1), (
                    f"limit of {self.s.playground_student_per_ip_per_hour} sandbox-model "
                    "questions per hour per address reached"
                )
            self._daily = (today, self._daily[1] + 1)
        return None

    # ---- verification -----------------------------------------------------
    def demo_db(self) -> bytes:
        with self._db_lock:
            if self._db is None:
                self._db = sql_schema.build_database(self.s.demo_db_seed)
            return self._db

    def _known_gold(self) -> dict[str, tuple[str, bool]]:
        """Gold SQL of every task from completed runs on the demo DB (real template gold)."""
        with self._gold_lock:
            return self._load_gold()

    def _load_gold(self) -> dict[str, tuple[str, bool]]:
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
        ex = self.s.playground_local.executor if self.s.playground_local else None
        outcome = ex.run_batch(DEMO_DB_REF, [sql])[0] if ex is not None else run_select(db, sql)
        gold = self._known_gold().get(question.strip())
        if gold is None:
            return None, outcome
        return compare_outcomes(outcome, run_select(db, gold[0]), gold[1]).ok, outcome

    # ---- request ----------------------------------------------------------
    async def answer(self, question: str, ip: str = "unknown") -> dict[str, Any]:
        results: dict[str, dict[str, Any]] = {}
        cost = 0.0
        reason = self.teacher_unavailable_reason()
        local: dict[str, StudentServer] = {}
        for role in LOCAL_ROLES:
            state, why, server = self.local_status(role)
            if server is not None:
                local[role] = server
            else:
                results[role] = _result(state, why)
        if local:
            limited = self._check_student_caps(ip)
            if limited is not None:
                for role in local:
                    results[role] = _result("rate_limited", limited[1], retry_after_s=limited[0])
                local = {}
        teacher_task = None
        if reason is not None:
            results["teacher"] = _result("unavailable", reason)
        else:
            teacher_task = asyncio.ensure_future(self._teacher(question))
        local_task = asyncio.ensure_future(self._local(question, local)) if local else None
        try:
            if teacher_task is not None:
                results["teacher"], cost = await teacher_task
        finally:
            if local_task is not None:
                results.update(await local_task)
        order = {k: results[k] for k in ("teacher", "base", "student")}
        return {"results": order, "cost_usd": cost, "sandbox_cost_usd": None, "note": NOTE}

    async def _local(
        self, question: str, servers: dict[str, StudentServer]
    ) -> dict[str, dict[str, Any]]:
        async def one(role: str, server: StudentServer) -> tuple[str, dict[str, Any]]:
            msgs = build_messages(
                question,
                sql_schema.schema_ddl(self.s.demo_db_seed),
                role=f"eval_{role}",  # type: ignore[arg-type]
            )
            try:
                outs = await run_in_threadpool(server.generate, [msgs])
            except Exception as exc:  # noqa: BLE001 - any serving failure is reported, never faked
                return role, _result("error", None, error=f"{role} failed: {str(exc)[:300]}")
            text = outs[0] if outs else ""
            if not text.strip():
                return role, _result("error", None, error=f"{role} returned no output")
            return role, await self._finish(question, text)

        pairs = await asyncio.gather(*(one(r, srv) for r, srv in servers.items()))
        return dict(pairs)

    async def _finish(self, question: str, text: str) -> dict[str, Any]:
        sql = extract_sql(text)
        if sql is None:
            return _result("error", None, error="no SQL found in the model output")
        verified, outcome = await run_in_threadpool(self._verify, question, sql)
        preview = None
        if outcome.ok:
            preview = {
                "columns": list(outcome.columns),
                "rows": [[_cell(c) for c in r] for r in outcome.rows[:PREVIEW_ROWS]],
                "row_count": len(outcome.rows),
            }
        return _result("ok", None, sql=sql, verified=verified, rows=preview,
                       error=None if outcome.ok else outcome.error[:300])  # fmt: skip

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
            return _result("unavailable", "teacher is not configured"), 0.0
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
            return _result("unavailable", str(exc)), 0.0
        try:
            try:
                res = await llm.chat("teacher", messages, purpose="playground")
            except LLMError as exc:
                return _result(
                    "error", None, error=f"teacher call failed: {type(exc).__name__}"
                ), 0.0
            usd = self.ledger.estimate_llm_cost(model, res.input_tokens, res.output_tokens)
            # Record BEFORE releasing the reservation so the spend is never invisible.
            self.ledger.record(PLAYGROUND, model, usd, res.input_tokens, res.output_tokens)
        finally:
            self._release(estimate)
        return await self._finish(question, res.text), usd


class _Unavailable(Exception):
    """Internal: a role cannot be served, with the reason to show."""


def _result(
    state: State,
    reason: str | None,
    *,
    sql: str | None = None,
    verified: bool | None = None,
    rows: dict[str, Any] | None = None,
    error: str | None = None,
    retry_after_s: float | None = None,
) -> dict[str, Any]:
    """``state`` is the distinct outcome; ``available`` (kept for older clients) is true when the
    model was actually asked (ok or error)."""
    out: dict[str, Any] = {
        "state": state,
        "available": state in ("ok", "error"),
        "reason": reason,
        "sql": sql,
        "verified": verified,
        "rows_preview": rows,
        "error": error,
    }
    if retry_after_s is not None:
        out["retry_after_s"] = retry_after_s
    return out
