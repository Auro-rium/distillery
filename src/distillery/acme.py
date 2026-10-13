"""The "Acme analyst": a tiny data agent that talks to an OpenAI-compatible endpoint through the
plain ``openai`` SDK and nothing else.

It is the demo of the pitch "change one base URL". The code below has NO knowledge of Distillery:
pointing the ``OpenAI`` client at Token Factory or at the Distillery gateway is the only
difference (``scripts/acme_agent.py --base-url ...``). Per question it makes two calls:

1. the SQL skill: ``prompts.build_messages(question, ddl, role="eval_teacher")`` to the teacher
   model (the request the certified gateway recognises);
2. the returned SQL runs read-only on the billing database (``taskpacks.sql.runner.run_select``);
3. the cheap summary model writes a one-line summary of the rows (a plain chat call).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from distillery.prompts import build_messages, extract_sql
from distillery.taskpacks.sql import schema as sql_schema
from distillery.taskpacks.sql.runner import DbRef, run_select

SUMMARY_SYSTEM = (
    "You are a data analyst at Acme. Given a question and the rows returned by a database query, "
    "reply with exactly one short sentence that answers the question. No preamble."
)
SUMMARY_ROWS = 10
SUMMARY_MAX_TOKENS = 128


@dataclass(frozen=True)
class AcmeAnswer:
    question: str
    sql: str | None
    columns: tuple[str, ...]
    rows: tuple[tuple[Any, ...], ...]
    summary: str | None
    error: str | None
    sql_usage: tuple[int, int]  # (prompt, completion) tokens the SQL call reported
    summary_usage: tuple[int, int]
    routing: dict[str, Any] | None  # a provider's extra top-level field, if it sends one


def _usage(resp: Any) -> tuple[int, int]:
    u = getattr(resp, "usage", None)
    return int(getattr(u, "prompt_tokens", 0) or 0), int(getattr(u, "completion_tokens", 0) or 0)


def _content(resp: Any) -> str:
    return str(resp.choices[0].message.content or "")


class AcmeAnalyst:
    def __init__(
        self,
        client: Any,
        *,
        sql_model: str,
        summary_model: str,
        db: DbRef | None = None,
        ddl: str | None = None,
    ) -> None:
        """``client`` is an ``openai.OpenAI``; ``db``/``ddl`` default to the demo billing DB."""
        self.client = client
        self.sql_model = sql_model
        self.summary_model = summary_model
        self.db: DbRef = db if db is not None else sql_schema.build_database(0)
        self.ddl = ddl if ddl is not None else sql_schema.schema_ddl(0)

    def ask(self, question: str) -> AcmeAnswer:
        messages = build_messages(question, self.ddl, role="eval_teacher")
        resp = self.client.chat.completions.create(model=self.sql_model, messages=messages)
        extra = getattr(resp, "model_extra", None) or {}
        routing = extra.get("distillery") if isinstance(extra, dict) else None
        sql = extract_sql(_content(resp))
        sql_usage = _usage(resp)
        route = routing if isinstance(routing, dict) else None

        def answer(
            *,
            error: str | None,
            cols: tuple[str, ...] = (),
            rows: tuple[tuple[Any, ...], ...] = (),
            summary: str | None = None,
            summary_usage: tuple[int, int] = (0, 0),
        ) -> AcmeAnswer:
            return AcmeAnswer(
                question, sql, cols, rows, summary, error, sql_usage, summary_usage, route
            )

        if sql is None:
            return answer(error="the model returned no SQL")
        out = run_select(self.db, sql)
        if not out.ok:
            return answer(error=f"query failed: {out.error[:200]}")
        summary, usage = self._summarize(question, out.columns, out.rows)
        return answer(
            error=None, cols=out.columns, rows=out.rows, summary=summary, summary_usage=usage
        )

    def _summarize(
        self, question: str, columns: Sequence[str], rows: Sequence[Sequence[Any]]
    ) -> tuple[str, tuple[int, int]]:
        resp = self.client.chat.completions.create(
            model=self.summary_model,
            messages=summary_messages(question, columns, rows),
            max_tokens=SUMMARY_MAX_TOKENS,
        )
        return _content(resp).strip(), _usage(resp)


def summary_messages(
    question: str, columns: Sequence[str], rows: Sequence[Sequence[Any]]
) -> list[dict[str, str]]:
    """The one-line-summary request (shared with the server-side ``/api/acme/ask`` flow)."""
    shown = [list(r) for r in rows[:SUMMARY_ROWS]]
    user = (
        f"Question: {question}\nColumns: {list(columns)}\n"
        f"Rows (first {len(shown)} of {len(rows)}): {shown}"
    )
    return [{"role": "system", "content": SUMMARY_SYSTEM}, {"role": "user", "content": user}]


def database_ref(path: str | None) -> DbRef | None:
    return Path(path) if path else None
