"""Request telemetry for the API: counters, latency histograms, gauges, structured access logs.

* One pure-ASGI middleware (not ``BaseHTTPMiddleware``, which would buffer the SSE event streams).
* Every response carries ``X-Request-ID`` (an inbound id is kept only if it is a safe token).
* One JSON log line per request: method, route TEMPLATE (``/api/runs/{run_id}``, never the raw path
  with ids or query strings), status, milliseconds, request id. No headers, bodies, tokens or IPs.
* ``GET /api/metrics`` (Prometheus text) and ``GET /api/telemetry`` (JSON) expose counters only:
  nothing secret, nothing per-visitor. Values are in-memory and reset on restart; they say so.
"""

from __future__ import annotations

import json
import logging
import re
import threading
import time
import uuid
from collections import Counter
from collections.abc import Callable
from typing import Any

from fastapi import FastAPI
from fastapi.responses import JSONResponse, PlainTextResponse
from starlette.types import ASGIApp, Message, Receive, Scope, Send

log = logging.getLogger("distillery.access")

BUCKETS = (0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0)  # seconds
_SAFE_ID = re.compile(r"^[A-Za-z0-9._-]{8,64}$")
_STREAM_SUFFIX = "/events"  # SSE: long-lived by design, counted but kept out of latency


class Telemetry:
    """Thread-safe in-memory metrics. Reset on restart (and labelled as such in the output)."""

    def __init__(self, clock: Callable[[], float] = time.monotonic) -> None:
        self._clock = clock
        self.started = clock()
        self._lock = threading.Lock()
        self.requests: Counter[tuple[str, str, int]] = Counter()
        self._hist: dict[str, list[int]] = {}
        self._sum: dict[str, float] = {}
        self._count: Counter[str] = Counter()
        self.inflight = 0
        self.streams_open = 0
        self.events: Counter[str] = Counter()  # named business events (playground answers, ...)
        # extra Prometheus lines computed at scrape time (run health, plan A6); run OUTSIDE the lock
        self.collectors: list[Callable[[], list[str]]] = []

    # -- recording ----------------------------------------------------------------------
    def begin(self, stream: bool) -> None:
        with self._lock:
            self.inflight += 1
            if stream:
                self.streams_open += 1

    def end(self, method: str, route: str, status: int, seconds: float, stream: bool) -> None:
        with self._lock:
            self.inflight -= 1
            if stream:
                self.streams_open -= 1
            self.requests[(method, route, status)] += 1
            if stream:
                return
            h = self._hist.setdefault(route, [0] * (len(BUCKETS) + 1))
            for i, b in enumerate(BUCKETS):
                if seconds <= b:
                    h[i] += 1
                    break
            else:
                h[-1] += 1
            self._sum[route] = self._sum.get(route, 0.0) + seconds
            self._count[route] += 1

    def incr(self, name: str, n: int = 1) -> None:
        with self._lock:
            self.events[name] += n

    # -- reading ------------------------------------------------------------------------
    def _quantile(self, route: str, q: float) -> float | None:
        h = self._hist.get(route)
        total = self._count.get(route, 0)
        if not h or not total:
            return None
        target, seen = q * total, 0
        for i, n in enumerate(h):
            seen += n
            if seen >= target:
                return BUCKETS[i] if i < len(BUCKETS) else float("inf")
        return None

    def snapshot(self, extra: dict[str, Any] | None = None) -> dict[str, Any]:
        with self._lock:
            by_class: Counter[str] = Counter()
            by_route: dict[str, dict[str, Any]] = {}
            for (_m, route, status), n in self.requests.items():
                by_class[f"{status // 100}xx"] += n
                r = by_route.setdefault(route, {"requests": 0, "errors_5xx": 0})
                r["requests"] += n
                if status >= 500:
                    r["errors_5xx"] += n
            for route, r in by_route.items():
                c = self._count.get(route, 0)
                if c:
                    r["latency_s_p50_upper_bound"] = self._quantile(route, 0.5)
                    r["latency_s_p95_upper_bound"] = self._quantile(route, 0.95)
                    r["latency_s_mean"] = round(self._sum[route] / c, 4)
            return {
                "note": "in-memory since process start; resets on restart; percentiles are "
                "histogram bucket upper bounds, not exact",
                "uptime_s": round(self._clock() - self.started, 1),
                "requests_total": sum(self.requests.values()),
                "requests_by_status_class": dict(by_class),
                "by_route": by_route,
                "in_flight": self.inflight,
                "event_streams_open": self.streams_open,
                "events": dict(self.events),
                **(extra or {}),
            }

    def prometheus(self) -> str:
        with self._lock:
            out = [
                "# HELP distillery_uptime_seconds Seconds since process start.",
                "# TYPE distillery_uptime_seconds gauge",
                f"distillery_uptime_seconds {self._clock() - self.started:.1f}",
                "# HELP distillery_http_requests_total Requests by method, route template, status.",
                "# TYPE distillery_http_requests_total counter",
            ]
            for (m, route, status), n in sorted(self.requests.items()):
                out.append(
                    f'distillery_http_requests_total{{method="{m}",route="{route}",'
                    f'status="{status}"}} {n}'
                )
            out += [
                "# HELP distillery_http_request_duration_seconds Latency (SSE streams excluded).",
                "# TYPE distillery_http_request_duration_seconds histogram",
            ]
            for route, h in sorted(self._hist.items()):
                cum = 0
                for i, b in enumerate(BUCKETS):
                    cum += h[i]
                    out.append(
                        f'distillery_http_request_duration_seconds_bucket{{route="{route}",'
                        f'le="{b}"}} {cum}'
                    )
                cum += h[-1]
                out.append(
                    f'distillery_http_request_duration_seconds_bucket{{route="{route}",'
                    f'le="+Inf"}} {cum}'
                )
                out.append(
                    f'distillery_http_request_duration_seconds_sum{{route="{route}"}} '
                    f"{self._sum[route]:.4f}"
                )
                out.append(
                    f'distillery_http_request_duration_seconds_count{{route="{route}"}} '
                    f"{self._count[route]}"
                )
            out += [
                "# TYPE distillery_http_in_flight gauge",
                f"distillery_http_in_flight {self.inflight}",
                "# TYPE distillery_event_streams_open gauge",
                f"distillery_event_streams_open {self.streams_open}",
                "# TYPE distillery_events_total counter",
            ]
            for name, n in sorted(self.events.items()):
                out.append(f'distillery_events_total{{name="{name}"}} {n}')
        for collect in self.collectors:
            try:
                out += collect()
            except Exception:  # noqa: BLE001, S112 - a collector must never break the scrape
                continue
        return "\n".join(out) + "\n"


class TelemetryMiddleware:
    """Pure ASGI: wraps ``send`` to read the status, never touches the body (SSE-safe)."""

    def __init__(self, app: ASGIApp, tele: Telemetry) -> None:
        self.app, self.tele = app, tele

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        path = str(scope.get("path", ""))
        stream = path.endswith(_STREAM_SUFFIX)
        supplied = dict(scope.get("headers") or []).get(b"x-request-id", b"").decode("latin-1")
        rid = supplied if _SAFE_ID.match(supplied) else uuid.uuid4().hex[:16]
        status = 500
        t0 = self.tele._clock()
        self.tele.begin(stream)

        async def send_wrapper(message: Message) -> None:
            nonlocal status
            if message["type"] == "http.response.start":
                status = int(message["status"])
                headers = list(message.get("headers") or [])
                headers.append((b"x-request-id", rid.encode("latin-1")))
                message = {**message, "headers": headers}
            await send(message)

        try:
            await self.app(scope, receive, send_wrapper)
        finally:
            seconds = self.tele._clock() - t0
            route = getattr(scope.get("route"), "path", None) or "unmatched"
            method = str(scope.get("method", "GET"))
            self.tele.end(method, route, status, seconds, stream)
            log.info(
                json.dumps(
                    {
                        "evt": "http",
                        "method": method,
                        "route": route,
                        "status": status,
                        "ms": round(seconds * 1000, 1),
                        "stream": stream,
                        "request_id": rid,
                    },
                    separators=(",", ":"),
                )
            )


def setup_logging() -> None:
    """Send access lines to stderr as bare JSON (idempotent; leaves other loggers alone)."""
    if log.handlers:
        return
    h = logging.StreamHandler()
    h.setFormatter(logging.Formatter("%(message)s"))
    log.addHandler(h)
    log.setLevel(logging.INFO)
    log.propagate = False


def install(app: FastAPI, extra: Callable[[], dict[str, Any]] | None = None) -> Telemetry:
    """Add the middleware and the two read-only endpoints. Call before any catch-all route."""
    tele = Telemetry()
    setup_logging()
    app.add_middleware(TelemetryMiddleware, tele=tele)

    @app.get("/api/telemetry")
    def telemetry_json() -> JSONResponse:
        return JSONResponse(tele.snapshot(extra() if extra else None))

    @app.get("/api/metrics")
    def metrics() -> PlainTextResponse:
        return PlainTextResponse(
            tele.prometheus(), media_type="text/plain; version=0.0.4; charset=utf-8"
        )

    app.state.telemetry = tele
    return tele
