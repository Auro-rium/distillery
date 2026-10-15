"""``POST /api/acme/ask``: the Acme analyst, run server-side for the website.

The browser cannot hold an API key, so the server runs the same two-call flow as
``distillery.acme.AcmeAnalyst`` (the SQL skill, then a one-line summary) against its OWN gateway.
The calls are real ``POST /v1/chat/completions`` requests delivered to this very ASGI app through
``httpx.ASGITransport``: same routing, same per-address limits and daily caps, same
``gateway_calls`` rows as a customer's HTTP request. Nothing here decides who answers.

The caller's address (already resolved by the server's trusted-proxy rules) is handed to the
gateway as the ASGI peer, so the gateway's per-IP limit counts the visitor, not the server. No
header of the incoming request (admin token included) is forwarded.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

import httpx
from fastapi import BackgroundTasks, FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field, ValidationError
from starlette.concurrency import run_in_threadpool

from distillery.acme import SUMMARY_MAX_TOKENS, summary_messages
from distillery.prompts import build_messages, extract_sql
from distillery.server.gateway import INTERNAL_HEADER
from distillery.server.playground import DEMO_DB_REF, MAX_QUESTION_CHARS, _cell
from distillery.taskpacks.sql import schema as sql_schema
from distillery.taskpacks.sql.runner import ExecOutcome, run_select

MAX_ROWS = 20  # same cap the gateway puts on the rows it returns
MAX_BODY_BYTES = 8 * 1024
GATEWAY_PATH = "/v1/chat/completions"


class AskRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    question: str = Field(min_length=1, max_length=MAX_QUESTION_CHARS)


def _err(
    status: int, code: str, message: str, headers: dict[str, str] | None = None, **extra: Any
) -> JSONResponse:
    return JSONResponse(
        {"error": code, "message": message, **extra}, status_code=status, headers=headers
    )


def _error_fields(res: httpx.Response) -> tuple[str, str, str | None]:
    """(code, message, request id) of a gateway (OpenAI-style) error response."""
    try:
        doc = res.json()
    except ValueError:
        doc = {}
    err = doc.get("error") if isinstance(doc, dict) else None
    meta = doc.get("distillery") if isinstance(doc, dict) else None
    code = err.get("code") if isinstance(err, dict) else None
    msg = err.get("message") if isinstance(err, dict) else None
    rid = meta.get("request_id") if isinstance(meta, dict) else None
    return (
        str(code or "gateway_error"),
        str(msg or "the gateway refused the request"),
        rid if isinstance(rid, str) else None,
    )


def _gateway_error(res: httpx.Response) -> JSONResponse:
    """Re-shape a gateway error into this API's ``{error, message}`` form, keeping the status,
    ``Retry-After`` and the gateway request id."""
    code, msg, rid = _error_fields(res)
    headers = {k: v for k, v in res.headers.items() if k.lower() == "retry-after"}
    return _err(res.status_code, code, msg, headers, request_id=rid)


def _completion(res: httpx.Response) -> tuple[str, dict[str, Any]] | None:
    try:
        doc = res.json()
        text = str(doc["choices"][0]["message"]["content"] or "")
        info = doc["distillery"]
    except (ValueError, KeyError, IndexError, TypeError):
        return None
    return (text, info) if isinstance(info, dict) else None


def _run_sql(request: Request, sql: str) -> ExecOutcome:
    """Read-only execution on the demo database; the sandbox executor when the server has one
    (the same choice the gateway makes), else the in-process read-only runner."""
    pg = request.app.state.playground
    loc = pg.s.playground_local
    if loc is not None and loc.executor is not None:
        return loc.executor.run_batch(DEMO_DB_REF, [sql])[0]  # type: ignore[no-any-return]
    return run_select(pg.demo_db(), sql)


def install(app: FastAPI, ip_of: Callable[[Request], str]) -> None:
    """Register ``POST /api/acme/ask``. Call after ``gateway.install`` and BEFORE the
    ``/api/{rest:path}`` and SPA catch-alls."""

    @app.post("/api/acme/ask")
    async def acme_ask(request: Request) -> JSONResponse:
        raw = bytearray()
        async for chunk in request.stream():
            raw += chunk
            if len(raw) > MAX_BODY_BYTES:
                return _err(413, "payload_too_large", f"body larger than {MAX_BODY_BYTES} bytes")
        try:
            body = AskRequest.model_validate(json.loads(raw))
        except ValidationError as exc:
            fields = "; ".join(
                f"{'.'.join(str(p) for p in e['loc'])}: {e['msg']}" for e in exc.errors()
            )
            return _err(422, "invalid_request", fields)
        except (ValueError, UnicodeDecodeError):
            return _err(400, "invalid_json", "body must be a JSON object")
        question = body.question.strip()
        if not question:
            return _err(422, "invalid_request", "question: must not be blank")

        gw = request.app.state.gateway
        teacher: str | None = gw.teacher_model
        if not teacher:
            return _err(503, "teacher_unavailable", "no teacher model is configured")
        ip = ip_of(request)

        followups = BackgroundTasks()
        resp = await _ask(app, gw, ip, question, teacher, request, followups)
        if followups.tasks:  # the drift audit runs after the visitor has the answer
            resp.background = followups
        return resp

    async def _ask(
        app: FastAPI, gw: Any, ip: str, question: str, teacher: str, request: Request,
        followups: BackgroundTasks,
    ) -> JSONResponse:  # fmt: skip
        transport = httpx.ASGITransport(app=app, client=(ip, 0), raise_app_exceptions=False)
        headers = {INTERNAL_HEADER: gw.internal_token}
        async with httpx.AsyncClient(
            transport=transport, base_url="http://acme.internal", headers=headers
        ) as http:
            calls: list[str] = []

            async def post(payload: dict[str, Any]) -> httpx.Response:
                res = await http.post(GATEWAY_PATH, json=payload)
                rid = res.headers.get("x-distillery-request-id")
                if rid:
                    calls.append(rid)
                    task = gw.take_drift_work(rid)
                    if task is not None:
                        followups.tasks.append(task)
                return res

            # 1. the SQL skill: byte-for-byte the request the Acme agent sends
            res = await post(
                {
                    "model": teacher,
                    "messages": build_messages(
                        question, sql_schema.schema_ddl(0), role="eval_teacher"
                    ),
                }
            )
            parsed = _completion(res) if res.status_code == 200 else None
            if parsed is None:
                if res.status_code == 200:
                    return _err(502, "gateway_error", "the gateway sent an unreadable answer")
                return _gateway_error(res)
            text, info = parsed

            # 2. the served SQL runs read-only on the demo database
            sql = extract_sql(text)
            columns: list[str] = []
            full_rows: list[Any] = []
            rows: list[list[Any]] = []
            row_count = 0
            error: str | None = None
            verified = False
            if sql is None:
                error = "the model returned no SQL"
            else:
                out = await run_in_threadpool(_run_sql, request, sql)
                if out.ok:
                    columns = list(out.columns)
                    full_rows = list(out.rows)  # uncapped: the summary prompt states the real count
                    rows = [[_cell(c) for c in r] for r in full_rows[:MAX_ROWS]]
                    row_count = len(full_rows)
                    verified = row_count >= 1
                else:
                    error = f"query failed: {out.error[:200]}"

            # 3. the one-line summary: a plain passthrough call on the cheap model
            summary: str | None = None
            summary_error: dict[str, Any] | None = None
            summary_cost: float | None = None
            triage: str | None = gw.cfg.model_ids.get("triage")
            if error is None and sql is not None:
                if not triage:
                    summary_error = {
                        "code": "summary_model_unavailable",
                        "message": "no summary model is configured",
                        "status": 503,
                    }
                else:
                    sres = await post(
                        {
                            "model": triage,
                            "messages": summary_messages(question, columns, full_rows),
                            "max_tokens": SUMMARY_MAX_TOKENS,
                        }
                    )
                    sparsed = _completion(sres) if sres.status_code == 200 else None
                    if sparsed is not None:
                        summary = sparsed[0].strip() or None
                        cost = sparsed[1].get("cost_usd")
                        summary_cost = float(cost) if isinstance(cost, int | float) else None
                    else:
                        code, msg, _ = _error_fields(sres)
                        summary_error = {
                            "code": code,
                            "message": msg,
                            "status": sres.status_code,
                            "retry_after": sres.headers.get("retry-after"),
                        }

        return JSONResponse(
            {
                "question": question,
                "answer": {
                    "sql": sql,
                    "columns": columns,
                    "rows": rows,
                    "row_count": row_count,
                    "verified": verified,
                    "error": error,
                },
                "summary": summary,
                "summary_error": summary_error,
                "summary_cost_usd": summary_cost,
                "distillery": info,
                "calls": calls,
            }
        )


__all__ = ["install"]
