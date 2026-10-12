"""Run the Acme analyst agent against ANY OpenAI-compatible endpoint.

    # straight to Token Factory (the "before")
    uv run python scripts/acme_agent.py --base-url "$NEBIUS_BASE_URL" --api-key "$NEBIUS_API_KEY"
    # through the Distillery gateway (the "after": only the base URL changes)
    uv run python scripts/acme_agent.py --base-url https://<gateway-host>/v1 --api-key anything

The agent code (``distillery.acme``) is identical in both runs. The gateway needs no real key
from public callers; the models default to DISTILLERY_MODEL_TEACHER / DISTILLERY_MODEL_TRIAGE.
Each call here spends real money when pointed at a live endpoint: do it on purpose.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Sequence

from openai import OpenAI

from distillery.acme import AcmeAnalyst, database_ref

DEFAULT_QUESTIONS = (
    "How many accounts are there?",
    "How many invoices are overdue?",
    "Which country has the most accounts?",
)


def main(argv: Sequence[str] | None = None) -> int:
    env = os.environ
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--base-url", default=env.get("NEBIUS_BASE_URL"), help="OpenAI-compatible URL")
    p.add_argument("--api-key", default=env.get("OPENAI_API_KEY") or env.get("NEBIUS_API_KEY"))
    p.add_argument("--sql-model", default=env.get("DISTILLERY_MODEL_TEACHER"))
    p.add_argument("--summary-model", default=env.get("DISTILLERY_MODEL_TRIAGE"))
    p.add_argument("--db", default=None, help="SQLite billing database (default: the demo DB)")
    p.add_argument("--question", action="append", help="repeatable; default: three examples")
    p.add_argument("--json", action="store_true", help="one JSON object per question")
    args = p.parse_args(argv)
    missing = [
        n
        for n, v in (
            ("--base-url", args.base_url),
            ("--api-key", args.api_key),
            ("--sql-model", args.sql_model),
            ("--summary-model", args.summary_model),
        )
        if not v
    ]
    if missing:
        p.error(f"missing {', '.join(missing)} (or set the matching environment variable)")
    client = OpenAI(base_url=args.base_url, api_key=args.api_key)
    agent = AcmeAnalyst(
        client,
        sql_model=args.sql_model,
        summary_model=args.summary_model,
        db=database_ref(args.db),
    )
    code = 0
    for q in args.question or DEFAULT_QUESTIONS:
        ans = agent.ask(q)
        if ans.error:
            code = 1
        if args.json:
            print(json.dumps({**ans.__dict__, "rows": [list(r) for r in ans.rows]}, default=str))
            continue
        print(f"Q: {q}\n  SQL:     {ans.sql}\n  Summary: {ans.summary or ans.error}")
        if ans.routing:
            print(f"  Served by: {ans.routing.get('served_by')} ({ans.routing.get('reason')})")
    return code


if __name__ == "__main__":
    sys.exit(main())
