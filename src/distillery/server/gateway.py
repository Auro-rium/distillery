"""OpenAI-compatible gateway: ``POST /v1/chat/completions``.

A client changes ONE thing (its base URL). Requests that are the certified ``acme-sql`` skill go
through the certified router: the student answers when its calibrated confidence reaches the
certified threshold and its SQL runs; every other request is handed back to the teacher. Anything
that is not the skill is a plain passthrough, allowed only for a short allowlist of cheap models
(max_tokens clamped) unless the caller holds the admin token: the hosted site is never an open
proxy on our key.

Every call, served or refused, is one row in ``gateway_calls`` (see ``Store.add_gateway_call``).
Savings are computed server-side from that table (``GET /api/gateway/savings``).
"""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
import math
import re
import secrets
import threading
import time
import uuid
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Literal

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from starlette.background import BackgroundTask
from starlette.concurrency import run_in_threadpool

from distillery.budget import PLAYGROUND, BudgetExceeded, UnknownPriceError
from distillery.drift import (
    AUDIT_AGREE,
    AUDIT_DISAGREE,
    AuditResult,
    DriftError,
    DriftEvent,
    DriftMonitor,
    DriftParams,
    DriftState,
    RouteEvent,
)
from distillery.prompts import SYSTEM_PROMPT, extract_sql, parse_user_content, strip_thinking
from distillery.router import RouterModel, decide, precheck
from distillery.server.gateway_config import (
    EmbedResult,
    GatewaySettings,
    Upstream,
    UpstreamResult,
)
from distillery.server.limits import SlidingWindow
from distillery.server.playground import (
    DEMO_DB_REF,
    EST_OUTPUT_TOKENS,
    MAX_QUESTION_CHARS,
    Playground,
    _cell,
)
from distillery.student import generate_detailed
from distillery.taskpacks.sql import schema as sql_schema
from distillery.taskpacks.sql.runner import ExecOutcome, run_select
from distillery.taskpacks.sql.verifier import compare_outcomes

log = logging.getLogger("distillery.gateway")

SKILL = "acme-sql"
MAX_ROWS = 20
NOT_CERTIFIED_REASON = "not certified: all traffic goes to the teacher"
UNKNOWN_SCHEMA_REASON = (
    "unknown database schema: the student is certified on the demo schema only → teacher"
)
STUDENT_DAILY_LIMIT_REASON = "student daily limit reached → teacher"
ROUTER_MODEL_FILES = ("model.json", "router_model.json")
SUPPORTED_PARAMS = ("temperature", "top_p", "stop", "seed")
UNSUPPORTED_FIELDS = ("tools", "tool_choice", "functions", "function_call")
SYSTEM_PROMPT_SHA256 = hashlib.sha256(SYSTEM_PROMPT.encode("utf-8")).hexdigest()
MAX_DDL_CHARS = 2 * len(sql_schema.schema_ddl(0))  # bigger "schemas" are not the skill
DDL_SHA256 = hashlib.sha256(sql_schema.schema_ddl(0).strip().encode("utf-8")).hexdigest()
BASIS_ACTUAL_TEACHER = "actual teacher call (measured tokens x configured price)"
BASIS_STUDENT_PENDING = "pending H1"
DRIFT_FILE = "drift.json"


class GatewayError(Exception):
    def __init__(
        self, status: int, code: str, message: str, headers: Mapping[str, str] | None = None
    ) -> None:
        super().__init__(message)
        self.status, self.code, self.message = status, code, message
        self.headers = dict(headers or {})


# ---- request parsing and skill detection ------------------------------------------------------


@dataclass(frozen=True)
class ChatRequest:
    model: str
    messages: tuple[dict[str, str], ...]
    params: dict[str, Any]
    max_tokens: int | None


@dataclass(frozen=True)
class SkillRequest:
    question: str
    ddl: str


def _content_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for p in content:
            if isinstance(p, dict) and p.get("type") == "text" and isinstance(p.get("text"), str):
                parts.append(p["text"])
            else:
                raise GatewayError(
                    400, "unsupported_content", "only text message content is supported"
                )
        return "".join(parts)
    raise GatewayError(400, "invalid_request", "message content must be a string")


def parse_chat_request(body: Any) -> ChatRequest:
    if not isinstance(body, dict):
        raise GatewayError(400, "invalid_json", "body must be a JSON object")
    model = body.get("model")
    if not isinstance(model, str) or not model or len(model) > 200:
        raise GatewayError(400, "invalid_request", "model must be a non-empty string")
    if body.get("stream"):
        raise GatewayError(
            400,
            "stream_not_supported",
            "stream=true is not supported by this gateway; send stream=false",
        )
    for f in UNSUPPORTED_FIELDS:
        if body.get(f) is not None:
            raise GatewayError(400, "unsupported_field", f"{f} is not supported by this gateway")
    if body.get("n") not in (None, 1):
        raise GatewayError(400, "unsupported_field", "n must be 1")
    raw = body.get("messages")
    if not isinstance(raw, list) or not raw or len(raw) > 64:
        raise GatewayError(400, "invalid_request", "messages must be a list of 1..64 items")
    msgs: list[dict[str, str]] = []
    for m in raw:
        if not isinstance(m, dict) or not isinstance(m.get("role"), str):
            raise GatewayError(400, "invalid_request", "each message needs a string role")
        msgs.append({"role": m["role"], "content": _content_text(m.get("content"))})
    mt = body.get("max_tokens", body.get("max_completion_tokens"))
    if mt is not None and (isinstance(mt, bool) or not isinstance(mt, int) or mt < 1):
        raise GatewayError(400, "invalid_request", "max_tokens must be a positive integer")
    params = {k: body[k] for k in SUPPORTED_PARAMS if body.get(k) is not None}
    return ChatRequest(model, tuple(msgs), params, mt)


def detect_skill(req: ChatRequest, teacher_model: str | None) -> SkillRequest | None:
    """The certified ``acme-sql`` skill iff: the model is the teacher id, the first message is a
    ``system`` message whose content hashes to ``SYSTEM_PROMPT``, there is exactly one user message
    (and nothing else: the student was trained on exactly ``[system, user]``), and the user message
    is exactly what ``prompts.build_messages`` writes for some (question, schema)."""
    if not teacher_model or req.model != teacher_model or len(req.messages) != 2:
        return None
    system, user = req.messages
    if system["role"] != "system" or user["role"] != "user":
        return None
    digest = hashlib.sha256(system["content"].encode("utf-8")).hexdigest()
    if not hmac.compare_digest(digest, SYSTEM_PROMPT_SHA256):
        return None
    parsed = parse_user_content(user["content"])
    if parsed is None or len(parsed[1]) > MAX_DDL_CHARS:
        return None
    return SkillRequest(question=parsed[0], ddl=parsed[1])


def prompt_sha256(messages: Any) -> str:
    blob = json.dumps(
        messages, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str
    )
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


# ---- router artifacts -------------------------------------------------------------------------


@dataclass(frozen=True)
class RouterState:
    """What the gateway loaded from ``DISTILLERY_ROUTER_DIR``. ``lambda_hat`` is None unless the
    model AND a certificate that pins exactly this model were loaded."""

    model: RouterModel | None = None
    lambda_hat: float | None = None
    alpha: float | None = None
    delta: float | None = None
    coverage: float | None = None
    selective_risk: float | None = None
    model_sha256: str | None = None
    cert_sha256: str | None = None  # sha256 of the cert.json bytes (what the drift guard pins)
    prereg_commit: str | None = None
    index_file: Path | None = None
    error: str | None = None  # why nothing is certified, when artifacts were configured

    @property
    def certified(self) -> bool:
        return self.model is not None and self.lambda_hat is not None


def _json_file(path: Path) -> dict[str, Any]:
    doc = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(doc, dict):
        raise ValueError(f"{path.name} must be a JSON object")
    return doc


def load_router(router_dir: Path | None) -> RouterState:
    """Read ``model.json`` + ``cert.json`` (the files ``distillery router fit/certify`` write).
    Never raises: a missing or inconsistent artifact means "not certified" with the reason in
    ``error`` (everything then goes to the teacher, the always-safe direction)."""
    if router_dir is None:
        return RouterState(error="no router artifacts configured (set DISTILLERY_ROUTER_DIR)")
    d = Path(router_dir)
    try:
        mpath = next((d / f for f in ROUTER_MODEL_FILES if (d / f).is_file()), None)
        if mpath is None:
            return RouterState(error=f"no model.json in {d}")
        mdoc = _json_file(mpath)
        model = RouterModel.from_json(mdoc.get("model", mdoc))
        sha = model.sha256()
        if "model_sha256" in mdoc and mdoc["model_sha256"] != sha:
            return RouterState(error="model.json does not match its recorded model sha256")
        index = sorted(d.glob("embed_index*.bin"))
        index_file = index[-1] if len(index) == 1 else None
        base = RouterState(model=model, model_sha256=sha, index_file=index_file)
        cpath = d / "cert.json"
        if not cpath.is_file():
            return replace(base, error="no cert.json: model is not certified")
        cert_sha = hashlib.sha256(cpath.read_bytes()).hexdigest()
        cdoc = _json_file(cpath)
        pinned = cdoc.get("model_sha256")
        if pinned != sha:
            return replace(base, error="cert.json was made for a different model than model.json")
        sub = cdoc.get("cert")
        cert: dict[str, Any] = sub if isinstance(sub, dict) else cdoc
        lam = cdoc.get("lambda_hat", cert.get("lambda_hat"))
        prereg = cdoc.get("prereg")
        commit = prereg.get("commit") if isinstance(prereg, dict) else cdoc.get("prereg_commit")

        def num(key: str) -> float | None:
            v = cdoc.get(key, cert.get(key))
            return float(v) if isinstance(v, int | float) and not isinstance(v, bool) else None

        lam_f = float(lam) if isinstance(lam, int | float) and not isinstance(lam, bool) else None
        if lam_f is not None and not math.isfinite(lam_f):
            return replace(base, error="cert.json lambda_hat is not finite")
        return RouterState(
            model=model,
            lambda_hat=lam_f,
            alpha=num("alpha"),
            delta=num("delta"),
            coverage=num("coverage"),
            selective_risk=num("selective_risk"),
            model_sha256=sha,
            cert_sha256=cert_sha,
            prereg_commit=str(commit) if commit else None,
            index_file=index_file,
            error=None if lam_f is not None else "certificate found no threshold: not certified",
        )
    except (OSError, ValueError, KeyError, TypeError) as exc:
        return RouterState(error=f"cannot load router artifacts: {type(exc).__name__}: {exc}"[:300])


def load_index(state: RouterState) -> Any | None:
    """The router dir's ``EmbeddingIndex`` (``embed_index_*.bin``), or None if there is not exactly
    one file or it does not load. Only the vectors are used here: the question is embedded on the
    request's own event loop through the upstream (see ``Gateway._embed_nn_sim``)."""
    if state.index_file is None:
        return None
    from distillery.embed_index import EmbeddingIndex, EmbedIndexError

    try:
        return EmbeddingIndex.load(state.index_file)
    except EmbedIndexError:
        return None


# ---- measured serving price -------------------------------------------------------------------


@dataclass(frozen=True)
class ServingPrice:
    student_usd_per_1k: float
    source: str
    as_of: str


def load_serving_price(path: Path | None) -> ServingPrice | None:
    """``deploy/serving_price.json``: ``{"student_usd_per_1k": <USD per 1000 student invocations>,
    "source": "<how it was measured>", "as_of": "<date>"}``. Absent file = pending H1 (None).
    A present but malformed file raises: a wrong price must not silently become "no price"."""
    if path is None or not Path(path).is_file():
        return None
    doc = _json_file(Path(path))
    v = doc.get("student_usd_per_1k")
    src, as_of = doc.get("source"), doc.get("as_of")
    if isinstance(v, bool) or not isinstance(v, int | float) or not math.isfinite(v) or v < 0:
        raise ValueError(f"{path}: student_usd_per_1k must be a number >= 0")
    if not (isinstance(src, str) and src.strip() and isinstance(as_of, str) and as_of.strip()):
        raise ValueError(f"{path}: source and as_of are required")
    return ServingPrice(float(v), src.strip(), as_of.strip())


# ---- upstream (Token Factory) -----------------------------------------------------------------


class OpenAIUpstream:
    """The real upstream: ``openai.AsyncOpenAI`` against the configured Token Factory base URL.
    # UNVERIFIED against a real endpoint here (no credentials in tests)."""

    def __init__(self, base_url: str, api_key: str, timeout_s: float = 120.0) -> None:
        import openai

        self._client = openai.AsyncOpenAI(
            base_url=base_url, api_key=api_key, timeout=timeout_s, max_retries=2
        )

    async def embed(self, model: str, texts: Sequence[str]) -> EmbedResult:
        resp = await self._client.embeddings.create(model=model, input=list(texts))
        rows = sorted(resp.data, key=lambda d: int(getattr(d, "index", 0) or 0))
        usage = resp.usage
        tokens = int(getattr(usage, "prompt_tokens", 0) or getattr(usage, "total_tokens", 0) or 0)
        return EmbedResult([[float(x) for x in d.embedding] for d in rows], tokens)

    async def complete(
        self, model: str, messages: Sequence[Mapping[str, Any]], **params: Any
    ) -> UpstreamResult:
        resp = await self._client.chat.completions.create(
            model=model,
            messages=[dict(m) for m in messages],  # type: ignore[misc]
            **params,
        )
        choice = resp.choices[0]
        usage = resp.usage
        return UpstreamResult(
            text=choice.message.content or "",
            input_tokens=int(getattr(usage, "prompt_tokens", 0) or 0),
            output_tokens=int(getattr(usage, "completion_tokens", 0) or 0),
            finish_reason=choice.finish_reason,
        )


# ---- the gateway ------------------------------------------------------------------------------


@dataclass
class CallLog:
    request_id: str
    ip_hash: str
    created_at: str
    day: str
    model: str = ""
    skill: bool = False
    prompt_sha256: str = ""
    served_by: str = "none"
    score: float | None = None
    lam: float | None = None
    reason: str | None = None
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    counterfactual_usd: float | None = None  # None = unavailable (see counterfactual_basis)
    counterfactual_basis: str | None = None
    embed_usd: float = 0.0  # the router's own nn_sim embedding, billed to this request
    student_called: bool = False
    latency_ms: float = 0.0
    status: str = "error"
    error: str | None = None
    drift_work: _DriftWork | None = None  # set when this request feeds the drift monitor

    @property
    def total_cost_usd(self) -> float:
        return self.cost_usd + self.embed_usd

    def row(self) -> dict[str, Any]:
        return {
            "created_at": self.created_at, "day": self.day, "request_id": self.request_id,
            "ip_hash": self.ip_hash, "model": self.model, "skill": int(self.skill),
            "prompt_sha256": self.prompt_sha256, "served_by": self.served_by,
            "score": self.score, "lambda": self.lam, "reason": self.reason,
            "input_tokens": self.input_tokens, "output_tokens": self.output_tokens,
            "cost_usd": self.total_cost_usd,
            "teacher_cost_usd_counterfactual": self.counterfactual_usd,
            "counterfactual_basis": self.counterfactual_basis,
            "student_called": int(self.student_called), "latency_ms": self.latency_ms,
            "status": self.status, "error": self.error,
        }  # fmt: skip


@dataclass
class _DriftWork:
    """What the drift monitor needs about one served skill request (fed after the response)."""

    request_id: str
    served_by: Literal["student", "teacher"]
    req: ChatRequest
    outcome: (
        ExecOutcome | None
    )  # the student's executed SQL (None: not executable / unknown schema)
    ddl_known: bool


@dataclass
class _Routed:
    served_by: str  # "student" | "teacher"
    text: str
    sql: str | None
    outcome: ExecOutcome | None
    usage: tuple[int, int]
    cost_usd: float
    counterfactual_usd: float | None
    counterfactual_basis: str
    score: float | None
    threshold: float | None
    reason: str
    short_circuit: bool
    usage_basis: str
    extra: dict[str, Any] = field(default_factory=dict)
    # True when the ROUTER decided (precheck / decide), False for infrastructure deferrals
    # (student unavailable, embedding failed, rollback ...): only decisions feed the drift monitor
    decided: bool = False


def _ascii(text: str) -> str:
    return (
        text.replace("→", "->").replace("≥", ">=").encode("ascii", "replace").decode("ascii")[:300]
    )


def _status_type(status: int) -> str:
    if status == 429:
        return "rate_limit_error"
    if status in (401, 403):
        return "permission_error"
    return "invalid_request_error" if status < 500 else "api_error"


class _Unroutable(Exception):
    """The router cannot decide; the message is the reason the teacher answers."""


class Gateway:
    def __init__(self, playground: Playground, ip_of: Callable[[Request], str]) -> None:
        self.pg = playground
        self.s = playground.s
        self.g: GatewaySettings = playground.s.gateway
        self.cfg = playground.s.config
        self.store = playground.reader.real
        self.ledger = playground.ledger
        self._ip_of = ip_of
        self.ip_limit = SlidingWindow(self.g.per_ip_per_hour, 3600.0, self.s.clock)
        self.teacher_model = self.cfg.model_ids.get("teacher")
        allow = self.g.passthrough_models or (
            (self.cfg.model_ids["triage"],) if self.cfg.model_ids.get("triage") else ()
        )
        self.allowlist = frozenset(allow)
        self.router = load_router(self.g.router_dir)
        self.serving_price = load_serving_price(self.g.serving_price_file)
        self.upstream: Upstream | None = self.g.upstream or self._default_upstream()
        self._salt = (self.g.ip_salt or secrets.token_hex(16)).encode()
        self._ddl_ok = {DDL_SHA256}
        self._student_lock = threading.Lock()
        self._student_day: tuple[str, int] | None = None
        self.nn_sim: Callable[[str], float] | None = self.g.nn_sim
        self.index = load_index(self.router) if self.router.certified else None
        self._drift_lock = threading.Lock()
        self._drift_stats: dict[str, Any] = {
            "audits": 0, "inconclusive": 0, "skipped_cap": 0, "skipped_error": 0, "spent_usd": 0.0,
        }  # fmt: skip
        self.drift: DriftMonitor | None = self._init_drift()

    # ---- drift guard ------------------------------------------------------------------------
    def _init_drift(self) -> DriftMonitor | None:
        """The monitor, restored from the persisted events (a restart never returns to CERTIFIED
        by itself). Config: ``GatewaySettings.drift`` (env ``DISTILLERY_DRIFT_*``), else
        ``drift.json`` in the router dir; neither = disabled, EXCEPT when the last persisted state
        is not CERTIFIED: then the params recorded in the rollback event keep the guard on (removing
        the config must not re-trust the student; only ``recertify`` does that)."""
        params: DriftParams | None = self.g.drift
        if params is None and self.g.router_dir is not None:
            f = Path(self.g.router_dir) / DRIFT_FILE
            if f.is_file():
                params = DriftParams.from_mapping(_json_file(f))  # malformed: stop the server
        persisted = self.store.drift_events()
        if params is None and persisted and persisted[-1]["state_to"] != "CERTIFIED":
            for e in reversed(persisted):
                rec = e["details"].get("params")
                if e["kind"] == "rollback" and isinstance(rec, dict):
                    params = DriftParams.from_mapping(rec)
                    log.warning(
                        "drift guard: no drift config, but the last state is %s; keeping the "
                        "guard on with the params recorded at the rollback",
                        persisted[-1]["state_to"],
                    )
                    break
        if params is None:
            return None
        mon = DriftMonitor(params, self.router.cert_sha256)
        mon.restore(persisted)
        self._persist_drift(mon.bind_cert(self.router.cert_sha256))
        return mon

    def _persist_drift(self, events: Sequence[DriftEvent]) -> None:
        for e in events:
            try:
                self.store.add_drift_event(e.to_row())
            except Exception:  # noqa: BLE001 - the in-memory state still holds; say it loudly
                log.exception("gateway: could not persist the drift event %s", e.kind)
            if e.kind == "rollback":
                log.warning("drift guard: ROLLBACK to teacher-only (%s): %s", e.detector, e.details)

    async def _drift_after(self, work: _DriftWork) -> None:
        """Runs after the response is sent: the optional audit, then feed the monitor."""
        mon = self.drift
        if mon is None:
            return
        try:
            audit: AuditResult | None = None
            if (
                work.served_by == "student"
                and mon.allows_student()
                and mon.wants_audit(work.request_id)
            ):
                audit = await self._audit(work)
            self._persist_drift(mon.observe(RouteEvent(work.served_by, work.request_id, audit)))
        except Exception:  # noqa: BLE001 - a background failure must never reach the client
            log.exception("gateway: drift bookkeeping failed for %s", work.request_id)

    async def _audit(self, work: _DriftWork) -> AuditResult | None:
        """Ask the teacher the same question (through the daily-cap reservation) and compare the two
        answers by EXECUTING both SQLs (the verifier's multiset comparison). None = no verdict
        (cap refused, upstream failed, or the teacher's SQL did not run): never a disagreement."""
        st = self._drift_stats
        if work.outcome is None or not work.ddl_known or not self.teacher_model:
            st["inconclusive"] += 1
            return None
        now = self._now()
        scratch = CallLog(work.request_id + ":audit", "", now.isoformat(), now.date().isoformat())
        params = dict(work.req.params)
        params["max_tokens"] = min(
            work.req.max_tokens or self.g.skill_max_tokens, self.g.skill_max_tokens
        )
        try:
            res, usd = await self._upstream_call(
                self.teacher_model, work.req.messages, params, scratch
            )
        except GatewayError as exc:  # cap reached (429), unpriced (503), upstream down (502)
            st["skipped_cap" if exc.code == "daily_budget_exhausted" else "skipped_error"] += 1
            log.warning("drift audit skipped for %s: %s", work.request_id, exc.code)
            return None
        st["audits"] += 1
        st["spent_usd"] += usd
        self.store.gateway_add_cost(work.request_id, usd)  # audits are part of the real cost
        teacher_out = await run_in_threadpool(self._verify, extract_sql(res.text), True)
        if teacher_out is None or not teacher_out.ok:
            st["inconclusive"] += 1
            return None
        same = compare_outcomes(work.outcome, teacher_out, False).ok
        return AUDIT_AGREE if same else AUDIT_DISAGREE

    def drift_status(self) -> dict[str, Any]:
        mon = self.drift
        events = [
            {k: e[k] for k in ("id", "created_at", "state_from", "state_to", "kind", "detector")}
            | {"details": e["details"]}
            for e in reversed(self.store.drift_events(limit=20))
        ]
        stats = dict(self._drift_stats)
        stats["spent_usd"] = round(stats["spent_usd"], 8)
        if mon is None:
            return {
                "enabled": False, "state": None, "allows_student": None, "reason": None,
                "detector": None, "params": None, "cert_sha256": self.router.cert_sha256,
                "rolled_back_cert_sha256": [], "window": None, "audits": stats, "events": events,
                "note": "drift guard not configured: set DISTILLERY_DRIFT_WINDOW and "
                "DISTILLERY_DRIFT_TAU (or a drift.json in the router dir); without it the gateway "
                "never rolls back by itself",
            }  # fmt: skip
        allowed = mon.allows_student()
        return {
            "enabled": True,
            "state": mon.state.value,
            "allows_student": allowed,
            "reason": None if allowed else mon.reason(),
            "detector": mon.last_detector,
            "params": mon.params.to_json(),
            "cert_sha256": mon.cert_sha256,
            "rolled_back_cert_sha256": list(mon.rolled_back_shas),
            "window": mon.window_stats(),
            "audits": stats,
            "events": events,
            "note": "window and audit counters are in memory (reset by a restart); the state and "
            "events are persisted",
        }

    def recertify(self, cert_sha256: Any, prereg_commit: Any) -> dict[str, Any]:
        """TEACHER -> CERTIFIED with a NEW certificate. The cert.json in the router dir is re-read
        (an operator drops the new artifacts there, then calls this); its sha256 must equal
        ``cert_sha256`` and must not be one this deployment already rolled back from."""
        mon = self.drift
        if mon is None:
            raise GatewayError(404, "drift_disabled", "the drift guard is not configured")
        with self._drift_lock:
            if mon.state is not DriftState.TEACHER:
                raise GatewayError(
                    409, "not_rolled_back", f"state is {mon.state.value}, not TEACHER"
                )
            if not isinstance(cert_sha256, str) or not re.fullmatch(r"[0-9a-f]{64}", cert_sha256):
                raise GatewayError(
                    422, "invalid_request", "cert_sha256 must be 64 lowercase hex characters"
                )
            if not isinstance(prereg_commit, str) or not prereg_commit.strip():
                raise GatewayError(422, "invalid_request", "prereg_commit is required")
            if cert_sha256 in mon.rolled_back_shas:
                raise GatewayError(
                    409, "same_certificate",
                    "that certificate was already rolled back: a NEW certificate is required",
                )  # fmt: skip
            new = load_router(self.g.router_dir)
            if new.cert_sha256 != cert_sha256:
                raise GatewayError(
                    409, "cert_mismatch",
                    f"cert.json in the router dir has sha256 {new.cert_sha256}, not the one given",
                )  # fmt: skip
            if not new.certified:
                raise GatewayError(409, "not_certified", new.error or "the certificate is empty")
            # swap the artifacts first: nothing is served by the student until the state flips
            self.router, self.index = new, load_index(new)
            try:
                events = mon.recertify(cert_sha256, prereg_commit)
            except DriftError as exc:
                raise GatewayError(409, "recertify_refused", str(exc)) from None
            self._persist_drift(events)
            try:
                self.store.add_audit(
                    "admin", "drift_recertify", None,
                    {"cert_sha256": cert_sha256, "prereg_commit": prereg_commit.strip()},
                )  # fmt: skip
            except Exception:  # noqa: BLE001
                log.exception("gateway: could not audit-log the recertify")
        return {
            "state": mon.state.value,
            "cert_sha256": cert_sha256,
            "prereg_commit": prereg_commit.strip(),
            "lambda": new.lambda_hat,
            "event": events[0].to_row(),
        }

    def _default_upstream(self) -> Upstream | None:
        key, url = self.cfg.nebius_api_key, self.cfg.nebius_base_url
        if key is None or not url:
            return None
        return OpenAIUpstream(url, key.get_secret_value())

    # ---- small helpers ----------------------------------------------------------------------
    def _now(self) -> datetime:
        return self.s.now().astimezone(UTC)

    def _until_midnight(self) -> int:
        now = self._now()
        nxt = datetime.combine(now.date() + timedelta(days=1), datetime.min.time(), UTC)
        return max(1, int((nxt - now).total_seconds()))

    def _ip_hash(self, ip: str) -> str:
        return hmac.new(self._salt, ip.encode(), hashlib.sha256).hexdigest()[:16]

    def _take_student_slot(self) -> bool:
        """Consume one of today's sandbox student calls. The day's count is seeded from the call
        log, so a restart does not reset the cap."""
        day = self._now().date().isoformat()
        with self._student_lock:
            if self._student_day is None or self._student_day[0] != day:
                self._student_day = (day, self.store.gateway_count(day, "student_called"))
            if self._student_day[1] >= self.g.student_daily_max:
                return False
            self._student_day = (day, self._student_day[1] + 1)
            return True

    def _admin(self, request: Request) -> bool:
        token = request.headers.get("x-admin-token")
        if token is not None:
            if not token:
                raise GatewayError(401, "admin_token_required", "X-Admin-Token header is empty")
            if not self.cfg.verify_admin_token(token):
                raise GatewayError(403, "forbidden", "invalid admin token")
            return True
        auth = request.headers.get("authorization", "")
        if auth[:7].lower() == "bearer ":
            return self.cfg.verify_admin_token(auth[7:].strip())
        return False

    async def _read_json(self, request: Request) -> Any:
        cap = self.g.max_body_bytes
        length = request.headers.get("content-length")
        if length is not None and length.isdigit() and int(length) > cap:
            raise GatewayError(413, "payload_too_large", f"body larger than {cap} bytes")
        buf = bytearray()
        async for chunk in request.stream():
            buf += chunk
            if len(buf) > cap:
                raise GatewayError(413, "payload_too_large", f"body larger than {cap} bytes")
        try:
            return json.loads(buf)
        except (ValueError, UnicodeDecodeError):
            raise GatewayError(400, "invalid_json", "body must be a JSON object") from None

    # ---- upstream call with the playground's daily-cap reservation --------------------------
    async def _upstream_call(
        self, model: str, messages: Sequence[Mapping[str, str]], params: dict[str, Any], lg: CallLog
    ) -> tuple[UpstreamResult, float]:
        if self.upstream is None:
            raise GatewayError(
                503, "upstream_unavailable", "no upstream model provider is configured"
            )
        in_est = sum(len(m["content"]) for m in messages) // 3
        try:
            # reserve against the output cap the call is allowed to reach, not a typical answer
            estimate = self.ledger.estimate_llm_cost(
                model, in_est, int(params.get("max_tokens") or EST_OUTPUT_TOKENS)
            )
        except UnknownPriceError:
            raise GatewayError(
                503,
                "price_unavailable",
                f"no price configured for {model!r}; refusing to spend blind",
            ) from None
        try:
            self.pg._reserve(estimate)  # noqa: SLF001 - the playground's atomic daily-cap check
        except BudgetExceeded:
            raise GatewayError(
                429, "daily_budget_exhausted",
                "the gateway's daily spend cap is reached; it resets at 00:00 UTC",
                {"Retry-After": str(self._until_midnight())},
            ) from None  # fmt: skip
        try:
            started = time.monotonic()
            try:
                res = await self.upstream.complete(model, messages, **params)
            except Exception as exc:  # noqa: BLE001 - any provider failure is a 502, never faked
                raise GatewayError(
                    502, "upstream_error", f"upstream call failed: {type(exc).__name__}"
                ) from exc
            latency = time.monotonic() - started
            usd = self.ledger.estimate_llm_cost(model, res.input_tokens, res.output_tokens)
            lg.input_tokens, lg.output_tokens, lg.cost_usd = (
                res.input_tokens,
                res.output_tokens,
                usd,
            )
            # recorded BEFORE the reservation is released, so spend is never invisible
            self.ledger.record(
                PLAYGROUND, model, usd, res.input_tokens, res.output_tokens, latency_s=latency
            )
        finally:
            self.pg._release(estimate)  # noqa: SLF001
        return res, usd

    # ---- entry point ------------------------------------------------------------------------
    async def handle(self, request: Request) -> JSONResponse:
        t0 = time.monotonic()
        now = self._now()
        lg = CallLog(
            request_id="req_" + uuid.uuid4().hex[:24],
            ip_hash=self._ip_hash(self._ip_of(request)),
            created_at=now.isoformat(),
            day=now.date().isoformat(),
        )
        try:
            body, headers = await self._serve(request, lg, t0)
            lg.status = "ok"
            resp = JSONResponse(body, headers=headers)
            if lg.drift_work is not None:  # the audit + monitor run after the client has its answer
                resp.background = BackgroundTask(self._drift_after, lg.drift_work)
        except GatewayError as exc:
            lg.status = "refused" if exc.status < 500 else "error"
            lg.error = exc.code
            lg.reason = lg.reason or exc.message[:300]
            lg.latency_ms = round((time.monotonic() - t0) * 1000, 3)
            resp = self._error(exc, lg.request_id)
        except Exception:  # noqa: BLE001
            log.exception("gateway: unexpected failure in %s", lg.request_id)
            lg.status, lg.error = "error", "internal_error"
            lg.latency_ms = round((time.monotonic() - t0) * 1000, 3)
            resp = self._error(GatewayError(500, "internal_error", "internal error"), lg.request_id)
        except BaseException:
            lg.status, lg.error = "error", "aborted"
            self._write(lg)
            raise
        self._write(lg)
        return resp

    def _write(self, lg: CallLog) -> None:
        try:
            self.store.add_gateway_call(lg.row())
        except Exception:  # noqa: BLE001 - the answer is already paid for; do not drop it
            log.exception("gateway: could not write the call log for %s", lg.request_id)

    @staticmethod
    def _error(exc: GatewayError, request_id: str) -> JSONResponse:
        body = {
            "error": {
                "message": exc.message, "type": _status_type(exc.status), "param": None,
                "code": exc.code,
            },
            "distillery": {"request_id": request_id},
        }  # fmt: skip
        return JSONResponse(
            body,
            status_code=exc.status,
            headers={**exc.headers, "x-distillery-request-id": request_id},
        )

    async def _serve(
        self, request: Request, lg: CallLog, t0: float
    ) -> tuple[dict[str, Any], dict[str, str]]:
        raw = await self._read_json(request)
        try:
            req = parse_chat_request(raw)
        except GatewayError:
            if isinstance(raw, dict):  # still say what was asked for in the log
                lg.model = str(raw.get("model", ""))[:200]
                lg.prompt_sha256 = prompt_sha256(raw.get("messages"))
            raise
        lg.model = req.model
        lg.prompt_sha256 = prompt_sha256(req.messages)
        admin = self._admin(request)
        if not admin:
            retry = self.ip_limit.hit(self._ip_of(request))
            if retry is not None:
                raise GatewayError(
                    429, "rate_limited",
                    f"limit of {self.g.per_ip_per_hour} requests per hour per address reached",
                    {"Retry-After": str(int(retry) + 1)},
                )  # fmt: skip
        skill = detect_skill(req, self.teacher_model)
        lg.skill = skill is not None
        if skill is not None:
            return await self._skill(req, skill, lg, t0)
        return await self._passthrough(req, admin, lg, t0)

    # ---- passthrough ------------------------------------------------------------------------
    async def _passthrough(
        self, req: ChatRequest, admin: bool, lg: CallLog, t0: float
    ) -> tuple[dict[str, Any], dict[str, str]]:
        if not admin and req.model not in self.allowlist:
            hint = (
                " The teacher model is served only for the certified acme-sql skill request."
                if req.model == self.teacher_model
                else ""
            )
            raise GatewayError(
                403, "model_not_allowed",
                f"model {req.model!r} is not available through this gateway without the admin "
                f"token (public models: {sorted(self.allowlist)}).{hint}",
            )  # fmt: skip
        cap = self.g.admin_max_tokens if admin else self.g.passthrough_max_tokens
        max_tokens = min(req.max_tokens or self.g.passthrough_max_tokens, cap)
        params = {**req.params, "max_tokens": max_tokens}
        lg.served_by = "passthrough"
        res, usd = await self._upstream_call(req.model, req.messages, params, lg)
        text = strip_thinking(res.text).strip()
        lg.counterfactual_usd, lg.counterfactual_basis = usd, "passthrough: same call either way"
        lg.latency_ms = round((time.monotonic() - t0) * 1000, 3)
        info = {
            "served_by": "passthrough", "skill": None, "cost_usd": lg.total_cost_usd,
            "latency_ms": lg.latency_ms, "request_id": lg.request_id,
            "max_tokens_applied": max_tokens, "admin": admin,
        }  # fmt: skip
        body = self._completion(req.model, text, res, lg, info)
        return body, self._headers(lg, info)

    # ---- the certified skill ----------------------------------------------------------------
    def _verify(self, sql: str | None, ddl_known: bool) -> ExecOutcome | None:
        if sql is None or not ddl_known:
            return None
        loc = self.s.playground_local  # same path as the playground: the sandbox executor if any
        if loc is not None and loc.executor is not None:
            return loc.executor.run_batch(DEMO_DB_REF, [sql])[0]
        return run_select(self.pg.demo_db(), sql)

    async def _nn_sim(self, question: str, lg: CallLog) -> float:
        """Max cosine similarity of the question to the training questions. Raises ``_Unroutable``
        (reason for the teacher) when it cannot be computed."""
        if self.g.nn_sim is not None:  # injected, free, synchronous (tests / overrides)
            try:
                value = float(await run_in_threadpool(self.g.nn_sim, question))
            except Exception:  # noqa: BLE001 - a lookup failure must not fail the request
                log.exception("gateway: nn_sim failed for %s", lg.request_id)
                raise _Unroutable("similarity lookup failed → teacher") from None
        elif self.index is not None and self.upstream is not None:
            value = await self._embed_nn_sim(question, lg)
        else:
            raise _Unroutable("similarity index unavailable: cannot route → teacher")
        if not math.isfinite(value):
            raise _Unroutable("similarity lookup failed → teacher")
        return value

    async def _embed_nn_sim(self, question: str, lg: CallLog) -> float:
        """Embed the question on THIS event loop through the upstream, with the same
        estimate -> reserve -> record -> release sequence as every other paid call; the cost is
        added to the request's ``cost_usd``. The comparison with the index is local."""
        from distillery.embed_index import EmbedIndexError

        assert self.index is not None and self.upstream is not None  # noqa: S101
        model = self.index.model
        in_est = len(question) // 3 + 8
        try:
            estimate = self.ledger.estimate_llm_cost(model, in_est, 0)
        except UnknownPriceError:
            raise _Unroutable("embedding model is unpriced: cannot route → teacher") from None
        try:
            self.pg._reserve(estimate)  # noqa: SLF001
        except BudgetExceeded:
            raise GatewayError(
                429, "daily_budget_exhausted",
                "the gateway's daily spend cap is reached; it resets at 00:00 UTC",
                {"Retry-After": str(self._until_midnight())},
            ) from None  # fmt: skip
        try:
            started = time.monotonic()
            try:
                res = await self.upstream.embed(model, [question])
                if len(res.vectors) != 1:
                    raise EmbedIndexError("wrong number of vectors")
            except Exception:  # noqa: BLE001
                log.exception("gateway: embedding failed for %s", lg.request_id)
                raise _Unroutable("similarity lookup failed → teacher") from None
            tokens = res.prompt_tokens or in_est
            usd = self.ledger.estimate_llm_cost(model, tokens, 0)
            lg.embed_usd += usd
            self.ledger.record(
                PLAYGROUND, model, usd, tokens, 0, latency_s=time.monotonic() - started
            )
        finally:
            self.pg._release(estimate)  # noqa: SLF001
        try:
            return float(self.index.nn_sim(self.index.prepare(res.vectors[0])))
        except EmbedIndexError:
            log.exception("gateway: embedding/index mismatch for %s", lg.request_id)
            raise _Unroutable("similarity lookup failed → teacher") from None

    async def _skill(
        self, req: ChatRequest, skill: SkillRequest, lg: CallLog, t0: float
    ) -> tuple[dict[str, Any], dict[str, str]]:
        if len(skill.question) > MAX_QUESTION_CHARS:
            raise GatewayError(
                400, "question_too_long", f"question is longer than {MAX_QUESTION_CHARS} characters"
            )
        ddl_known = hashlib.sha256(skill.ddl.encode("utf-8")).hexdigest() in self._ddl_ok
        routed = await self._route_skill(req, skill, ddl_known, lg)
        lg.served_by = routed.served_by
        lg.score, lg.lam, lg.reason = routed.score, routed.threshold, routed.reason
        lg.input_tokens, lg.output_tokens = routed.usage
        lg.counterfactual_usd = routed.counterfactual_usd
        lg.counterfactual_basis = routed.counterfactual_basis
        lg.latency_ms = round((time.monotonic() - t0) * 1000, 3)
        if (
            self.drift is not None
            and routed.decided
            and self.router.certified
            and routed.served_by in ("student", "teacher")
        ):
            lg.drift_work = _DriftWork(
                lg.request_id,
                "student" if routed.served_by == "student" else "teacher",
                req,
                routed.outcome if routed.served_by == "student" else None,
                ddl_known,
            )
        out = routed.outcome
        verified = bool(out is not None and out.ok and len(out.rows) >= 1)
        info: dict[str, Any] = {
            "served_by": routed.served_by, "skill": SKILL, "score": routed.score,
            "lambda": routed.threshold, "reason": routed.reason,
            "short_circuit": routed.short_circuit, "sql": routed.sql, "verified": verified,
            "columns": list(out.columns) if out is not None and out.ok else [],
            "rows": [[_cell(c) for c in r] for r in out.rows[:MAX_ROWS]]
            if out is not None and out.ok
            else [],
            "row_count": len(out.rows) if out is not None and out.ok else 0,
            "error": out.error[:300] if out is not None and not out.ok else None,
            "cost_usd": lg.total_cost_usd, "embed_cost_usd": lg.embed_usd,
            "teacher_cost_usd_counterfactual": routed.counterfactual_usd,
            "counterfactual_basis": routed.counterfactual_basis,
            "latency_ms": lg.latency_ms, "request_id": lg.request_id,
            "usage_basis": routed.usage_basis,
        }  # fmt: skip
        res = UpstreamResult(routed.text, *routed.usage)
        return self._completion(req.model, routed.text, res, lg, info), self._headers(lg, info)

    async def _route_skill(
        self, req: ChatRequest, skill: SkillRequest, ddl_known: bool, lg: CallLog
    ) -> _Routed:
        st = self.router
        lam = st.lambda_hat
        decision_reason = NOT_CERTIFIED_REASON
        score: float | None = None
        short = False
        nn: float | None = None
        if st.model is None or lam is None:
            if st.model is not None:
                decision_reason = decide(st.model, None, {}).reason
            return await self._via_teacher(req, ddl_known, lg, decision_reason, None, None, False)
        model = st.model
        if self.drift is not None and not self.drift.allows_student():
            # before any embedding or student spend: a rolled-back gateway is teacher-only
            return await self._via_teacher(
                req, ddl_known, lg, self.drift.reason(), None, lam, False
            )
        if not ddl_known:
            # certified on the demo schema only; no embedding or student slot is spent on a
            # request it can never serve, and it stays out of the drift monitor (not decided)
            return await self._via_teacher(
                req, ddl_known, lg, UNKNOWN_SCHEMA_REASON, None, lam, False
            )
        if "nn_sim" in model.feature_names:
            try:
                nn = await self._nn_sim(skill.question, lg)
            except _Unroutable as exc:
                return await self._via_teacher(req, ddl_known, lg, str(exc), None, lam, False)
        pre = precheck(model, lam, nn if nn is not None else 0.0)
        if pre is not None:
            return await self._via_teacher(
                req, ddl_known, lg, pre.reason, None, lam, True, decided=True
            )
        state, why, server = self.pg.local_status("student")
        if server is None:
            return await self._via_teacher(
                req, ddl_known, lg, f"student unavailable ({why}) → teacher", None, lam, False
            )
        if not self._take_student_slot():
            # a capacity limit, not a router decision: the teacher answers, outside the monitor
            return await self._via_teacher(
                req, ddl_known, lg, STUDENT_DAILY_LIMIT_REASON, None, lam, False
            )
        lg.student_called = True
        try:
            outs = await run_in_threadpool(generate_detailed, server, [list(req.messages)])
            out = outs[0]
        except Exception as exc:  # noqa: BLE001 - a serving failure defers to the teacher
            return await self._via_teacher(
                req, ddl_known, lg, f"student failed ({type(exc).__name__}) → teacher",
                None, lam, False,
            )  # fmt: skip
        if out.error or out.mean_logprob is None or not math.isfinite(out.mean_logprob):
            return await self._via_teacher(
                req, ddl_known, lg, "student confidence not measured → teacher", None, lam, False
            )
        sql = extract_sql(out.text)
        outcome = await run_in_threadpool(self._verify, sql, ddl_known)
        exec_ok = bool(outcome is not None and outcome.ok and len(outcome.rows) >= 1)
        signals = {
            "exec_ok": 1.0 if exec_ok else 0.0,
            "mean_logprob": float(out.mean_logprob),
            "nn_sim": nn if nn is not None else 0.0,
        }
        try:
            dec = decide(model, lam, signals)
        except KeyError:
            return await self._via_teacher(
                req,
                ddl_known,
                lg,
                "router needs a signal the gateway lacks → teacher",
                None,
                lam,
                False,
            )
        score, decision_reason, short = dec.score, dec.reason, False
        if dec.served_by == "student" and sql is not None:
            if self.drift is not None and not self.drift.allows_student():
                # another request rolled the gateway back while this one was in flight
                return await self._via_teacher(
                    req, ddl_known, lg, self.drift.reason(), score, lam, False
                )
            return self._student_routed(
                req, out.n_answer_tokens, sql, outcome, dec.score, lam, dec.reason
            )
        if (
            dec.served_by == "student"
        ):  # cannot happen with exec_ok in the model; never serve nothing
            decision_reason = "no SQL extracted from the student output → teacher"
        return await self._via_teacher(
            req, ddl_known, lg, decision_reason, score, lam, short, decided=True
        )

    def _student_routed(
        self, req: ChatRequest, n_answer_tokens: int | None, sql: str, outcome: ExecOutcome | None,
        score: float | None, lam: float | None, reason: str,
    ) -> _Routed:  # fmt: skip
        in_est = max(1, math.ceil(sum(len(m["content"]) for m in req.messages) / 3))
        out_tok = n_answer_tokens if n_answer_tokens else max(1, math.ceil(len(sql) / 3))
        cf, basis = self._counterfactual(in_est)
        return _Routed(
            "student", sql, sql, outcome, (in_est, out_tok), 0.0, cf, basis, score, lam, reason,
            False,
            "student: completion tokens measured by the student server, prompt tokens estimated "
            "(characters / 3); cost_usd excludes unpriced sandbox compute",
            decided=True,
        )  # fmt: skip

    def _counterfactual(self, in_est: int) -> tuple[float | None, str]:
        """What the teacher would have cost for this request, or None + why it is unavailable.
        Output length, in order: the mean of the teacher answers this gateway logged; else the
        measured mean of real teacher calls in pipeline runs (``llm_calls``); else unavailable
        (never an assumed length)."""
        if not self.teacher_model:
            return None, "unavailable: no teacher model configured"
        mean = self.store.gateway_teacher_output_mean()
        if mean is not None:
            out, src = (
                mean[0],
                f"{mean[0]:.1f} mean output tokens over {mean[1]} logged gateway teacher calls",
            )
        else:
            mean = self.store.llm_output_mean(self.teacher_model)
            if mean is None:
                return None, "unavailable: no measured teacher output length yet"
            out, src = (
                mean[0],
                (
                    f"{mean[0]:.1f} mean output tokens over {mean[1]} measured teacher calls in "
                    "pipeline runs (llm_calls)"
                ),
            )
        try:
            usd = self.ledger.estimate_llm_cost(self.teacher_model, in_est, round(out))
        except UnknownPriceError:
            return None, "unavailable: no teacher price configured"
        return usd, (f"teacher price x (estimated {in_est} prompt tokens [characters / 3] + {src})")

    async def _via_teacher(
        self, req: ChatRequest, ddl_known: bool, lg: CallLog, reason: str,
        score: float | None, lam: float | None, short: bool, *, decided: bool = False,
    ) -> _Routed:  # fmt: skip
        assert self.teacher_model is not None  # noqa: S101 - detect_skill needs it
        params = dict(req.params)
        params["max_tokens"] = min(
            req.max_tokens or self.g.skill_max_tokens, self.g.skill_max_tokens
        )
        res, usd = await self._upstream_call(self.teacher_model, req.messages, params, lg)
        text = strip_thinking(res.text).strip()
        sql = extract_sql(res.text)
        outcome = await run_in_threadpool(self._verify, sql, ddl_known)
        return _Routed(
            "teacher", text, sql, outcome, (res.input_tokens, res.output_tokens), usd, usd,
            BASIS_ACTUAL_TEACHER, score, lam, reason, short, "measured by the provider",
            decided=decided,
        )  # fmt: skip

    # ---- response shaping -------------------------------------------------------------------
    def _completion(
        self, model: str, text: str, res: UpstreamResult, lg: CallLog, info: dict[str, Any]
    ) -> dict[str, Any]:
        return {
            "id": "chatcmpl-" + lg.request_id[4:],
            "object": "chat.completion",
            "created": int(self._now().timestamp()),
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": text},
                    "finish_reason": res.finish_reason or "stop",
                }
            ],
            "usage": {
                "prompt_tokens": res.input_tokens,
                "completion_tokens": res.output_tokens,
                "total_tokens": res.input_tokens + res.output_tokens,
            },
            "distillery": info,
        }

    @staticmethod
    def _headers(lg: CallLog, info: dict[str, Any]) -> dict[str, str]:
        h = {
            "x-distillery-request-id": lg.request_id,
            "x-distillery-served-by": str(info["served_by"]),
            "x-distillery-cost-usd": f"{lg.total_cost_usd:.8f}",
            "x-distillery-latency-ms": f"{lg.latency_ms:.1f}",
        }
        if info.get("skill"):
            h["x-distillery-skill"] = SKILL
            h["x-distillery-reason"] = _ascii(str(info["reason"]))
            h["x-distillery-verified"] = str(bool(info["verified"])).lower()
            if info["score"] is not None:
                h["x-distillery-score"] = f"{info['score']:.6f}"
            if info["lambda"] is not None:
                h["x-distillery-lambda"] = f"{info['lambda']:.6f}"
        return h

    # ---- read-only endpoints ----------------------------------------------------------------
    def status(self) -> dict[str, Any]:
        st = self.router
        day = self._now().date().isoformat()
        return {
            "certified": st.certified,
            "reason": None if st.certified else (st.error or NOT_CERTIFIED_REASON),
            "lambda": st.lambda_hat,
            "coverage": st.coverage,
            "selective_risk": st.selective_risk,
            "alpha": st.alpha,
            "delta": st.delta,
            "model_sha256": st.model_sha256,
            "prereg_commit": st.prereg_commit,
            "similarity_index": self.g.nn_sim is not None or self.index is not None,
            "teacher_model": self.teacher_model,
            "passthrough_models": sorted(self.allowlist),
            "passthrough_max_tokens": self.g.passthrough_max_tokens,
            "student": self.pg.local_status("student")[0],
            "drift": {
                "enabled": self.drift is not None,
                "state": None if self.drift is None else self.drift.state.value,
            },
            "today": {
                "day": day,
                "served_by": self.store.gateway_served_by(day),
                "refused": self.store.gateway_count(day, "not_ok"),
                "student_calls": self.store.gateway_count(day, "student_called"),
                "student_daily_max": self.g.student_daily_max,
            },
        }

    def savings(self) -> dict[str, Any]:
        t = self.store.gateway_totals()
        price = self.serving_price
        pending = price is None
        per_call = 0.0 if price is None else price.student_usd_per_1k / 1000.0
        # like for like: only requests whose teacher-only cost could be computed
        n_cmp = t["comparable_requests"]
        actual = t["comparable_cost_usd"] + t["comparable_student_calls"] * per_call
        cf = t["counterfactual_usd"]
        saved = cf - actual
        pct = saved / cf * 100.0 if cf > 0 else None
        spent_total = t["spent_usd"] + t["skill_student_calls"] * per_call
        return {
            "window": "all logged skill requests since this gateway's call log began",
            "requests": t["requests"],
            "served_by": {"student": t["student"], "teacher": t["teacher"]},
            "student_calls": t["student_calls"],
            "refused": t["refused"],
            "requests_in_savings": n_cmp,
            "requests_excluded": t["requests"] - n_cmp,
            "actual_usd": round(actual, 8),
            "teacher_only_usd": round(cf, 8),
            "saved_usd": round(saved, 8),
            "saved_pct": None if pct is None else round(pct, 2),
            "actual_display": f"${actual:.4f}",
            "teacher_only_display": f"${cf:.4f}",
            "saved_display": f"${saved:.4f}",
            "saved_pct_display": None if pct is None else f"{pct:.1f}%",
            "total_spent_usd": round(spent_total, 8),
            "student_cost_basis": BASIS_STUDENT_PENDING
            if price is None
            else f"measured: {price.source} (as of {price.as_of})",
            "student_usd_per_1k": None if price is None else price.student_usd_per_1k,
            "student_cost_usd": None if price is None else round(t["student_calls"] * per_call, 8),
            "basis": {
                "actual": "provider-billed teacher $ (measured tokens x configured price) plus the "
                "router's own embedding $; "
                + (
                    "student serving cost NOT included (pending H1)"
                    if pending
                    else "plus student invocations x measured student_usd_per_1k"
                ),
                "teacher_only": "per request: the teacher's actual cost when it answered; else the "
                "teacher price x (estimated prompt tokens + the mean teacher output length: this "
                "gateway's logged teacher answers, else measured pipeline teacher calls). A "
                "request with no measured length is EXCLUDED from actual/teacher_only/saved and "
                "counted in "
                "requests_excluded; the basis of each row is in gateway_calls.counterfactual_basis",
                "saved": "teacher_only - actual over requests_in_savings"
                + ("; an upper bound while the student cost is pending H1" if pending else ""),
            },
            "passthrough_requests": t["passthrough_requests"],
            "passthrough_usd": round(t["passthrough_usd"], 8),
        }


def install(app: FastAPI, playground: Playground, ip_of: Callable[[Request], str]) -> Gateway:
    """Register the gateway routes. Call BEFORE the ``/api/{rest:path}`` and SPA catch-alls."""
    gw = Gateway(playground, ip_of)

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request) -> JSONResponse:
        return await gw.handle(request)

    @app.get("/api/gateway/status")
    def gateway_status() -> dict[str, Any]:
        return gw.status()

    @app.get("/api/gateway/savings")
    def gateway_savings() -> dict[str, Any]:
        return gw.savings()

    @app.get("/api/gateway/drift")
    def gateway_drift() -> dict[str, Any]:
        return gw.drift_status()

    @app.post("/api/gateway/drift/recertify")
    async def gateway_drift_recertify(request: Request) -> JSONResponse:
        def err(status: int, code: str, message: str) -> JSONResponse:
            return JSONResponse({"error": code, "message": message}, status_code=status)

        token = request.headers.get("x-admin-token")
        if not token:
            return err(401, "admin_token_required", "X-Admin-Token header required")
        if not gw.cfg.verify_admin_token(token):
            return err(403, "forbidden", "invalid admin token")
        try:
            body = await gw._read_json(request)  # noqa: SLF001 - same size cap and parsing
            if not isinstance(body, dict) or set(body) - {"cert_sha256", "prereg_commit"}:
                raise GatewayError(
                    422, "invalid_request", "body must be {cert_sha256, prereg_commit}"
                )
            out = await run_in_threadpool(
                gw.recertify, body.get("cert_sha256"), body.get("prereg_commit")
            )
        except GatewayError as exc:
            return err(exc.status, exc.code, exc.message)
        return JSONResponse(out)

    app.state.gateway = gw
    return gw


__all__ = ["Gateway", "detect_skill", "install", "load_router", "parse_chat_request"]
