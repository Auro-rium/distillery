"""FastAPI app factory implementing docs/API_CONTRACT.md."""

from __future__ import annotations

import json
import threading
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from importlib import metadata
from typing import Any, Literal

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse
from pydantic import BaseModel, ConfigDict, Field, ValidationError
from starlette.background import BackgroundTask
from starlette.concurrency import run_in_threadpool
from starlette.exceptions import HTTPException as StarletteHTTPException

from distillery.evaluator import EXAMPLES_PER_KIND
from distillery.orchestrator import DRY_PREFIX, SCALES
from distillery.server import sse
from distillery.server.limits import SlidingWindow
from distillery.server.playground import (
    MAX_QUESTION_CHARS,
    DemoBudgetExhaustedError,
    Playground,
)
from distillery.server.reader import RunReader, valid_run_id
from distillery.server.replay import Bundle, load_bundles
from distillery.server.settings import ServerSettings
from distillery.server.views import (
    EXAMPLE_KINDS,
    example_totals,
    filter_examples,
    tree_from_report,
)
from distillery.server.worker import (
    Job,
    QueueFullError,
    RunConflictError,
    Worker,
    subprocess_executor,
)


class ApiError(Exception):
    def __init__(
        self, status: int, code: str, message: str, headers: dict[str, str] | None = None
    ) -> None:
        self.status, self.code, self.message, self.headers = status, code, message, headers or {}


def _err(
    status: int, code: str, message: str, headers: dict[str, str] | None = None
) -> JSONResponse:
    return JSONResponse({"error": code, "message": message}, status_code=status, headers=headers)


class RunRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    pack: Literal["sql"] = "sql"
    scale: Literal["tiny", "small", "full"]
    dry_run: bool
    budget_usd: float | None = Field(default=None, gt=0)
    run_id: str | None = None
    approve_spend: bool = False
    finetune_estimate_usd: float | None = Field(default=None, ge=0)


class PlaygroundRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    question: str = Field(min_length=1, max_length=MAX_QUESTION_CHARS)


def client_ip(request: Request, trusted_proxies: tuple[str, ...]) -> str:
    """The peer address, unless the peer is a configured trusted proxy: then the right-most
    X-Forwarded-For hop that is not itself a trusted proxy. With no trusted proxies configured
    (the default) the header is ignored, so it cannot be used to dodge per-IP limits."""
    peer = request.client.host if request.client else "unknown"
    if peer not in trusted_proxies:
        return peer
    hops = [h.strip() for h in request.headers.get("x-forwarded-for", "").split(",") if h.strip()]
    for hop in reversed(hops):
        if hop not in trusted_proxies:
            return hop
    return peer


async def _read_body(request: Request, cap: int) -> Any:
    length = request.headers.get("content-length")
    if length is not None and length.isdigit() and int(length) > cap:
        raise ApiError(413, "payload_too_large", f"body larger than {cap} bytes")
    buf = bytearray()
    async for chunk in request.stream():
        buf += chunk
        if len(buf) > cap:
            raise ApiError(413, "payload_too_large", f"body larger than {cap} bytes")
    try:
        return json.loads(buf)
    except (ValueError, UnicodeDecodeError):
        raise ApiError(400, "invalid_json", "body must be a JSON object") from None


def _parse[M: BaseModel](model: type[M], raw: Any) -> M:
    try:
        return model.model_validate(raw)
    except ValidationError as exc:
        fields = "; ".join(
            f"{'.'.join(str(p) for p in e['loc'])}: {e['msg']}" for e in exc.errors()
        )
        raise ApiError(422, "invalid_request", fields) from None


def _version() -> str:
    try:
        return metadata.version("distillery")
    except metadata.PackageNotFoundError:
        return "0.0.0"


def create_app(settings: ServerSettings) -> FastAPI:
    cfg = settings.config
    secrets = [s.get_secret_value() for s in (cfg.nebius_api_key, cfg.admin_token) if s is not None]
    admin = cfg.admin_token.get_secret_value() if cfg.admin_token else None
    executor = settings.executor or subprocess_executor(
        str(settings.root), admin, secrets, settings.shutdown_grace_s
    )
    worker = Worker(executor, secrets)
    limiter = sse.StreamLimiter(settings.sse_max_streams, settings.sse_max_streams_per_ip)

    def _ip(request: Request) -> str:
        return client_ip(request, settings.trusted_proxies)

    reader = RunReader(settings, worker)
    playground = Playground(settings, reader)
    dry_limit = SlidingWindow(settings.dry_run_per_ip_per_hour, 3600.0, settings.clock)

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        yield
        await run_in_threadpool(worker.stop, settings.shutdown_grace_s + 10.0)
        reader.close()

    app = FastAPI(title="Distillery API", lifespan=lifespan, docs_url=None, redoc_url=None,
                  openapi_url=None)  # fmt: skip
    app.state.worker, app.state.reader, app.state.playground = worker, reader, playground
    app.state.stream_limiter = limiter

    if settings.allowed_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=list(settings.allowed_origins),
            allow_methods=["GET", "POST"],
            allow_headers=["X-Admin-Token", "Content-Type", "Last-Event-ID"],
            # Browsers hide these from cross-origin JS unless listed here.
            expose_headers=[
                "X-Examples-Available",
                "X-Examples-Cap-Per-Kind",
                "X-Examples-Totals",
                "Retry-After",
            ],
            allow_credentials=False,
        )

    @app.exception_handler(ApiError)
    async def _api_error(_: Request, exc: ApiError) -> JSONResponse:
        return _err(exc.status, exc.code, exc.message, exc.headers)

    @app.exception_handler(StarletteHTTPException)
    async def _http_error(_: Request, exc: StarletteHTTPException) -> JSONResponse:
        code = {404: "not_found", 405: "method_not_allowed"}.get(exc.status_code, "error")
        return _err(exc.status_code, code, str(exc.detail))

    @app.exception_handler(RequestValidationError)
    async def _validation_error(_: Request, exc: RequestValidationError) -> JSONResponse:
        return _err(422, "invalid_request", "invalid request")

    # ---- helpers ----------------------------------------------------------
    def check_id(run_id: str) -> None:
        if not valid_run_id(run_id):
            raise ApiError(
                400, "invalid_run_id", "run id must match [A-Za-z0-9][A-Za-z0-9_-]{0,63}"
            )

    def require_admin(request: Request) -> None:
        token = request.headers.get("x-admin-token")
        if not token:
            raise ApiError(401, "admin_token_required", "X-Admin-Token header required")
        if not cfg.verify_admin_token(token):
            raise ApiError(403, "forbidden", "invalid admin token")

    def bundle(run_id: str) -> Bundle | None:
        return load_bundles(settings.replay_dir, settings.sample_report).get(run_id)

    def local(run_id: str) -> bool:
        return reader.exists(run_id)

    def detail_of(run_id: str) -> dict[str, Any] | None:
        if local(run_id):
            return reader.detail(run_id)
        b = bundle(run_id)
        if b is None:
            return None
        return b.run or reader.replay_detail(run_id, b.report, b.recorded, b.recorded_at)

    def report_of(run_id: str) -> tuple[dict[str, Any], bool, str | None]:
        """(report, recorded, recorded_at); 404s for unknown or unfinished runs."""
        if local(run_id):
            rep = reader.read_report(run_id)
            if rep is None or reader.detail(run_id)["status"] != "complete":
                raise ApiError(404, "report_not_ready", "the run has not completed yet")
            return rep, False, None
        b = bundle(run_id)
        if b is None:
            raise ApiError(404, "not_found", "no such run")
        return b.report, b.recorded, b.recorded_at

    # ---- read endpoints ---------------------------------------------------
    @app.get("/api/health")
    def health() -> dict[str, Any]:
        live = cfg.nebius_api_key is not None and bool(cfg.nebius_base_url)
        return {"ok": True, "mode": "live" if live else "replay-only", "version": _version()}

    @app.get("/api/config")
    def config() -> dict[str, Any]:
        g = cfg.gate
        return {
            "models": {
                r: cfg.model_ids.get(r) for r in ("planner", "teacher", "triage", "student")
            },
            "thresholds": {
                "ratio_lower_bound_min": g.ratio_lower_bound_min,
                "mcnemar_alpha": g.mcnemar_alpha,
                "bootstrap_resamples": g.bootstrap_resamples,
            },
            "run_cap_usd": cfg.run_cap_usd,
            "playground": {
                "enabled": playground.teacher_unavailable_reason() is None,
                "per_ip_per_hour": settings.playground_per_ip_per_hour,
                "daily_cap_usd": cfg.playground_daily_cap_usd,
                "spent_today_usd": playground.ledger.playground_spent(),
            },
        }

    @app.get("/api/runs")
    def runs() -> list[dict[str, Any]]:
        items = reader.list_runs()
        have = {i["run_id"] for i in items}
        # Recorded bundles are served like runs; a local run of the same id wins. The built-in
        # unrecorded sample is not a run and stays on /api/replay only.
        items += [
            b.item()
            for b in load_bundles(settings.replay_dir, settings.sample_report).values()
            if b.recorded and b.run_id not in have
        ]
        items.sort(key=lambda i: i["created_at"] or "", reverse=True)
        return items

    @app.get("/api/runs/{run_id}")
    def run_detail(run_id: str) -> dict[str, Any]:
        check_id(run_id)
        d = detail_of(run_id)
        if d is None:
            raise ApiError(404, "not_found", "no such run")
        return d

    @app.get("/api/runs/{run_id}/events")
    def run_events(
        run_id: str, request: Request, last_event_id: str | None = None
    ) -> StreamingResponse:
        check_id(run_id)
        # ``?last_event_id=`` (for EventSource reconnects that cannot set headers) wins over the
        # Last-Event-ID header.
        raw = last_event_id if last_event_id is not None else request.headers.get("last-event-id")
        try:
            last = max(0, int(raw or "0"))
        except ValueError:
            last = 0
        b = None if local(run_id) else bundle(run_id)
        if not local(run_id) and b is None:
            raise ApiError(404, "not_found", "no such run")
        if b is not None and b.events is not None:
            stored = [(int(e["id"]), str(e["event"]), e["data"]) for e in b.events]

            def refresh(after: int) -> tuple[list[tuple[int, str, dict[str, Any]]], bool]:
                return [e for e in stored if e[0] > after], True

        else:
            key = run_id if b is None else f"replay:{run_id}"

            def refresh(after: int) -> tuple[list[tuple[int, str, dict[str, Any]]], bool]:
                if b is not None:
                    d = b.run or reader.replay_detail(run_id, b.report, b.recorded, b.recorded_at)
                    return reader.events_after(key, after, d, b.report.get("decision"), None)
                d = reader.detail(run_id)
                rep = reader.read_report(run_id) if d["status"] == "complete" else None
                return reader.events_after(
                    run_id, after, d, rep.get("decision") if rep else None, worker.get(run_id)
                )

        ip = _ip(request)
        if not limiter.acquire(ip):
            raise ApiError(429, "too_many_streams", "too many open event streams",
                           {"Retry-After": "5"})  # fmt: skip
        released = threading.Event()

        def release() -> None:
            if not released.is_set():
                released.set()
                limiter.release(ip)

        gen = sse.astream(
            refresh, last, heartbeat_s=settings.heartbeat_s, poll_s=settings.poll_s,
            is_disconnected=request.is_disconnected, on_close=release, clock=settings.clock,
        )  # fmt: skip
        return StreamingResponse(
            gen,
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
            background=BackgroundTask(release),
        )

    @app.get("/api/runs/{run_id}/report")
    def run_report(run_id: str) -> dict[str, Any]:
        check_id(run_id)
        rep, recorded, recorded_at = report_of(run_id)
        return {**rep, "dry_run": bool(rep.get("dry_run")), "recorded": recorded,
                "recorded_at": recorded_at}  # fmt: skip

    @app.get("/api/runs/{run_id}/tree")
    def run_tree(run_id: str) -> dict[str, Any]:
        check_id(run_id)
        if not local(run_id):
            b = bundle(run_id)
            if b is None:
                raise ApiError(404, "not_found", "no such run")
            return b.tree if b.tree is not None else tree_from_report(b.report)
        rep = reader.read_report(run_id)
        return tree_from_report(rep) if rep else {"nodes": []}

    @app.get("/api/runs/{run_id}/examples")
    def run_examples(run_id: str, kind: str = "all", limit: int = 20) -> JSONResponse:
        check_id(run_id)
        if kind not in EXAMPLE_KINDS:
            raise ApiError(422, "invalid_request", f"kind must be one of {list(EXAMPLE_KINDS)}")
        rep, _, _ = report_of(run_id)
        items, available = filter_examples(rep, kind, max(1, min(limit, 200)))
        headers = {
            "X-Examples-Available": str(available).lower(),
            "X-Examples-Cap-Per-Kind": str(EXAMPLES_PER_KIND),
            "X-Examples-Totals": json.dumps(example_totals(rep), separators=(",", ":")),
        }
        return JSONResponse(items, headers=headers)

    @app.get("/api/replay")
    def replay() -> list[dict[str, Any]]:
        return [
            b.item() for b in load_bundles(settings.replay_dir, settings.sample_report).values()
        ]

    # ---- write endpoints --------------------------------------------------
    @app.post("/api/runs", status_code=202)
    async def start_run(request: Request) -> dict[str, str]:
        body = _parse(RunRequest, await _read_body(request, settings.max_body_bytes))
        run_id = body.run_id
        is_admin = False
        if body.dry_run:
            # Anonymous callers: scale 'tiny' only and a bounded queue. A supplied admin token
            # lifts both (an invalid one is rejected, never silently downgraded).
            is_admin = request.headers.get("x-admin-token") is not None
            if is_admin:
                require_admin(request)
            elif body.scale != "tiny":
                raise ApiError(403, "admin_token_required",
                               "anonymous dry runs are limited to scale 'tiny'")  # fmt: skip
            retry = dry_limit.hit(_ip(request))
            if retry is not None:
                raise ApiError(429, "rate_limited", "too many dry runs from this address",
                               {"Retry-After": str(int(retry) + 1)})  # fmt: skip
        else:
            require_admin(request)
            if not body.approve_spend:
                raise ApiError(400, "spend_not_approved", "live runs need approve_spend: true")
            if body.budget_usd is not None and body.budget_usd > cfg.run_cap_usd:
                raise ApiError(422, "invalid_request",
                               f"budget_usd exceeds the run cap {cfg.run_cap_usd}")  # fmt: skip
        if run_id is None:
            tag = uuid.uuid4().hex[:8]
            run_id = f"dry-sql-{body.scale}-{tag}" if body.dry_run else (
                f"sql-{body.scale}-{settings.now().strftime('%Y%m%dT%H%M%SZ')}-{tag}"
            )  # fmt: skip
        if not valid_run_id(run_id):
            raise ApiError(
                422, "invalid_run_id", "run id must match [A-Za-z0-9][A-Za-z0-9_-]{0,63}"
            )
        if run_id.startswith(DRY_PREFIX) != body.dry_run:
            raise ApiError(
                422, "invalid_run_id", "dry runs need the 'dry-' id prefix, real runs must not"
            )
        if body.scale not in SCALES:
            raise ApiError(422, "invalid_request", "unknown scale")
        if reader.read_report(run_id) is not None:
            raise ApiError(409, "run_exists", "that run already completed")
        job = Job(run_id, body.scale, body.dry_run, body.budget_usd, body.finetune_estimate_usd)
        cap = settings.max_pending_jobs if body.dry_run and not is_admin else None
        try:
            worker.submit(job, max_pending=cap)
        except QueueFullError as exc:
            raise ApiError(429, "queue_full", str(exc), {"Retry-After": "30"}) from None
        except RunConflictError as exc:
            raise ApiError(409, "run_conflict", str(exc)) from None
        reader.reset_events(run_id)
        return {"run_id": run_id}

    @app.post("/api/runs/{run_id}/cancel")
    def cancel_run(run_id: str, request: Request) -> dict[str, str]:
        check_id(run_id)
        require_admin(request)
        status = worker.cancel(run_id)
        if status is None:
            if not local(run_id):
                raise ApiError(404, "not_found", "no such run")
            return {"status": "not_running"}
        return {"status": status}

    @app.post("/api/playground")
    async def play(request: Request) -> dict[str, Any]:
        retry = playground.ip_limit.hit(_ip(request))
        if retry is not None:
            raise ApiError(429, "rate_limited", "playground rate limit reached",
                           {"Retry-After": str(int(retry) + 1)})  # fmt: skip
        body = _parse(PlaygroundRequest, await _read_body(request, settings.max_body_bytes))
        try:
            return await playground.answer(body.question)
        except DemoBudgetExhaustedError:
            raise ApiError(503, "demo_budget_exhausted",
                           "demo budget exhausted, see replay") from None  # fmt: skip

    @app.api_route("/api/{rest:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH"])
    def api_not_found(rest: str) -> Response:
        return _err(404, "not_found", "no such endpoint")

    # ---- static frontend (SPA fallback) -----------------------------------
    dist = settings.frontend_dist
    if dist is not None and (dist / "index.html").exists():
        root = dist.resolve()

        @app.get("/{path:path}", include_in_schema=False)
        def spa(path: str) -> FileResponse:
            target = (root / path).resolve()
            if path and target.is_file() and target.is_relative_to(root):
                return FileResponse(target)
            return FileResponse(root / "index.html")

    return app


__all__ = ["ApiError", "create_app"]
