#!/usr/bin/env python3
"""W-E2E: the whole serving path, live, through a running Distillery server (pre-registered in
DECISIONS.md, 2026-10-16). SPENDS real money on the server's side: run it on purpose.

  # local live server (global ledger on), then:
  uv run python scripts/w_e2e.py --base-url http://127.0.0.1:8765 \\
      --burst-questions .distillery/router_burst/questions.txt --out docs/proofs/evidence/w_e2e.json
  # hosted check (no burst, so the public gateway is never rolled back on purpose):
  uv run python scripts/w_e2e.py --base-url https://distillery.onrender.com --hosted \\
      --out docs/proofs/evidence/w_e2e_hosted.json

Phases: (1) /api/gateway/status; (2) the 20 normal questions below on the demo schema, the first
three through the website's /api/acme/ask and the rest straight to /v1/chat/completions; (3) the 30
pre-registered credit-grant burst questions (asked against the demo schema, which has no credit
table), with the drift state read after every request; (4) drift, savings and the ledger delta.
Writes counts, verdicts, hashes and per-request routing; never the burst question text. Stops
early when the global ledger has moved by more than --ceiling-usd. Never overwrites --out.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx

from distillery.global_ledger import GlobalLedger
from distillery.prompts import build_messages
from distillery.taskpacks.sql import schema as sql_schema

STUDENT_BASE = "Qwen/Qwen3-1.7B"
N1_ADAPTER = "7d6ce813882bf3a211c84bfab7f8109b3bcdd6ca4617349a03bff58076b85193"
LAMBDA_HAT = 0.9172253676263808
BURST_SHA256 = "447103a535175859a3e9cee591e7cdcac438f11f89f06ca75161c3534b2878b2"
ROLLBACK_WITHIN = 25
VIA_WEBSITE = 3  # the first questions go through /api/acme/ask, the website's own path
NORMAL = (
    # the website's six example questions (frontend/src/screens/playground) and 14 fresh templated
    # draws: get_pack("sql").generate_tasks(<demo DB>, 14, 20261016, exclude_families=<stress>,
    # skeleton_cap=<default>), the in-distribution traffic the certificate covers
    "How many accounts are there?",
    "Which industry has the most active accounts?",
    "how many ppl are on the enterprise plan rn",
    "Show the name of every account in country GB whose status is suspended.",
    "List the ids of invoices with an amount above $1000 (amounts are stored in cents).",
    "Which healthcare accounts were created after 2022-12-19? Return account name and creation date.",  # noqa: E501
    "Show account name and seats for every subscription on a enterprise-tier plan with more than 18 seats, for active accounts.",  # noqa: E501
    "List invoice ids with the account name for paid invoices issued in 2025.",
    "Return every plan name that has trialing subscriptions along with how many.",
    "For urgent-priority tickets, give each category's average csat_score rounded to two decimal places.",  # noqa: E501
    "gimme the overdue invoices by currency, biggest first",
    "Among suspended accounts created before 2024-08-16, which account in each industry has the most users (count all user rows)? Break ties by lowest account_id. Return industry, account name and user count.",  # noqa: E501
    "Return the latest EUR-currency invoice per account (break date ties with the larger invoice_id): account_id, invoice_id, issued_date.",  # noqa: E501
    "How many closed high-priority tickets have no CSAT score?",
    "Which users in education accounts have never logged in? Return their emails.",
    "Per country, count the accounts with status 'active' where sales_rep is NULL.",
    "Per ticket category among normal tickets opened 2024-07-23 to 2025-05-24, give the number with no user_id next to the total number of tickets.",  # noqa: E501
    "Which accounts in GB have never opened a support ticket? Return names.",
    "users who never logged in, newest signups on top",
    "What is our net revenue retention by customer cohort?",
)


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _row(kind: str, i: int, status: int, d: dict[str, Any] | None, ms: float) -> dict[str, Any]:
    d = d or {}
    return {
        "phase": kind, "i": i, "http": status, "served_by": d.get("served_by"),
        "reason": d.get("reason"), "score": d.get("score"), "short_circuit": d.get("short_circuit"),
        "verified": d.get("verified"), "row_count": d.get("row_count"),
        "cost_usd": d.get("cost_usd"), "embed_cost_usd": d.get("embed_cost_usd"),
        "server_latency_ms": d.get("latency_ms"), "client_ms": round(ms, 1),
    }  # fmt: skip


def _prereg(commit: str, normal_sha: str) -> dict[str, str]:
    """The pre-registration commit: an ancestor of HEAD whose DECISIONS.md names the normal
    questions' sha256. Raises ``ValueError`` otherwise."""

    def git(*args: str) -> str:
        return subprocess.run(  # noqa: S603 - fixed git subcommands, no shell
            ["git", *args],  # noqa: S607
            check=True, capture_output=True, text=True,
        ).stdout.strip()  # fmt: skip

    try:
        full = git("rev-parse", "--verify", f"{commit}^{{commit}}")
        anc = subprocess.run(  # noqa: S603
            ["git", "merge-base", "--is-ancestor", full, "HEAD"],  # noqa: S607
            check=False,
        )
        text = git("show", f"{full}:DECISIONS.md")
        when = git("show", "-s", "--format=%cI", full)
    except subprocess.CalledProcessError as exc:
        raise ValueError(f"cannot read pre-registration commit {commit!r}: {exc}") from None
    if anc.returncode != 0:
        raise ValueError(f"{commit} is not an ancestor of HEAD")
    if normal_sha not in text:
        raise ValueError(f"DECISIONS.md at {commit} does not name the normal questions' sha256")
    return {"commit": full, "committed_at": when}


def _code_pin(prereg_commit: str) -> dict[str, str]:
    """The code that serves and scores must be the pre-registered code: a clean tree for src/,
    deploy/ and this script, and no change to them since the pre-registration commit."""
    paths = ["src", "deploy", "scripts/w_e2e.py"]
    dirty = subprocess.run(  # noqa: S603
        ["git", "status", "--porcelain", "--", *paths],  # noqa: S607
        check=True, capture_output=True, text=True,
    ).stdout.strip()  # fmt: skip
    if dirty:
        raise ValueError(f"uncommitted changes under {paths}: {dirty.splitlines()[:3]}")
    moved = subprocess.run(  # noqa: S603
        ["git", "diff", "--quiet", prereg_commit, "HEAD", "--", *paths],  # noqa: S607
        check=False,
    )
    if moved.returncode != 0:
        raise ValueError(f"{paths} changed since the pre-registration commit")
    head = subprocess.run(  # noqa: S603
        ["git", "rev-parse", "HEAD"],  # noqa: S607
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    return {"head": head, "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}


def _preconditions(
    config: dict[str, Any], status: dict[str, Any], drift: dict[str, Any], ceiling: float | None
) -> list[str]:
    """Free reads before any question is sent; a failure here is not a run (no evidence)."""
    bad = []
    if config.get("models", {}).get("student") != STUDENT_BASE:
        bad.append(f"student model is {config.get('models', {}).get('student')!r}")
    cap = config.get("playground", {}).get("daily_cap_usd")
    if ceiling is not None and not (isinstance(cap, int | float) and cap >= ceiling):
        bad.append(f"server daily cap {cap!r} is under the run ceiling {ceiling}")
    if not (status.get("certified") and status.get("lambda") == LAMBDA_HAT):
        bad.append("router is not certified at the pre-registered lambda")
    if (
        not status.get("student_adapter_sha256")
        == status.get("certified_adapter_sha256")
        == N1_ADAPTER
    ):
        bad.append("served or certified adapter is not N1")
    if status.get("student") != "ok":
        bad.append(f"student is {status.get('student')!r}")
    if drift.get("state") != "CERTIFIED":
        bad.append(f"drift state is {drift.get('state')!r}")
    return bad


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--base-url", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--burst-questions", type=Path, help="the sealed burst file (local run only)")
    ap.add_argument("--hosted", action="store_true", help="no burst, no ledger delta")
    ap.add_argument("--ceiling-usd", type=float, default=0.15)
    ap.add_argument("--prereg-commit", help="required for a scored run (checked against git)")
    ap.add_argument("--timeout-s", type=float, default=600.0)
    a = ap.parse_args(argv)
    if a.out.exists():
        print(f"error: {a.out} already exists; it is never overwritten", file=sys.stderr)
        return 2
    prereg = None
    if a.prereg_commit:
        try:
            prereg = _prereg(a.prereg_commit, _sha("\n".join(NORMAL)))
            code_pin = None if a.hosted else _code_pin(prereg["commit"])
        except ValueError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2
    else:
        code_pin = None
    burst: list[str] = []
    if not a.hosted:
        if a.burst_questions is None:
            print("error: --burst-questions is required without --hosted", file=sys.stderr)
            return 2
        text = a.burst_questions.read_text(encoding="utf-8")
        if hashlib.sha256(text.encode("utf-8")).hexdigest() != BURST_SHA256:
            print("error: burst questions do not match the pre-registered sha256", file=sys.stderr)
            return 2
        burst = [q for q in text.splitlines() if q.strip()]
    ledger = None if a.hosted else GlobalLedger.from_env()
    if not a.hosted and ledger is None:
        print(
            "error: a local run needs the global ledger (DISTILLERY_GLOBAL_LEDGER)", file=sys.stderr
        )
        return 2
    spend0 = ledger.total_spend() if ledger else None
    ddl = sql_schema.DDL
    base = a.base_url.rstrip("/")
    started = datetime.now(UTC).isoformat()
    rows: list[dict[str, Any]] = []
    drift_trace: list[dict[str, Any]] = []
    stopped: str | None = None

    with httpx.Client(timeout=a.timeout_s) as c:
        config = c.get(f"{base}/api/config").json()
        status = c.get(f"{base}/api/gateway/status").json()
        drift0 = c.get(f"{base}/api/gateway/drift").json()
        bad = _preconditions(config, status, drift0, None if a.hosted else a.ceiling_usd)
        if bad:
            print("error: preconditions failed, nothing sent: " + "; ".join(bad), file=sys.stderr)
            return 2
        teacher = status["teacher_model"]

        def ask(kind: str, i: int, question: str, website: bool) -> None:
            t0 = time.monotonic()
            if website:
                r = c.post(f"{base}/api/acme/ask", json={"question": question})
                d = r.json().get("distillery") if r.status_code == 200 else None
            else:
                body = {
                    "model": teacher,
                    "messages": build_messages(question, ddl, role="eval_teacher"),
                }
                r = c.post(f"{base}/v1/chat/completions", json=body)
                d = r.json().get("distillery") if r.status_code == 200 else None
            row = _row(kind, i, r.status_code, d, (time.monotonic() - t0) * 1000)
            row["via"] = "acme_ask" if website else "v1"
            rows.append(row)
            print(json.dumps({k: row[k] for k in ("phase", "i", "http", "served_by", "client_ms")}))

        def over_ceiling() -> bool:
            return ledger is not None and ledger.total_spend() - (spend0 or 0.0) > a.ceiling_usd

        for i, q in enumerate(NORMAL):
            ask("normal", i, q, website=i < VIA_WEBSITE)
            if over_ceiling():
                stopped = f"ledger moved by more than {a.ceiling_usd} USD"
                break
        drift_mid = c.get(f"{base}/api/gateway/drift").json()
        if stopped is None:
            for i, q in enumerate(burst):
                ask("burst", i, q, website=False)
                st = c.get(f"{base}/api/gateway/drift").json()
                drift_trace.append({"i": i, "state": st.get("state")})
                if over_ceiling():
                    stopped = f"ledger moved by more than {a.ceiling_usd} USD"
                    break
        drift1 = c.get(f"{base}/api/gateway/drift").json()
        savings = c.get(f"{base}/api/gateway/savings").json()
        status1 = c.get(f"{base}/api/gateway/status").json()

    spend1 = ledger.total_spend() if ledger else None
    normal = [r for r in rows if r["phase"] == "normal"]
    burst_rows = [r for r in rows if r["phase"] == "burst"]
    student = [r for r in normal if r["served_by"] == "student"]
    # deferred by a router decision: a score under lambda, or the short-circuit precheck
    decided_defer = [
        r for r in normal
        if r["served_by"] == "teacher" and (r["score"] is not None or r["short_circuit"])
    ]  # fmt: skip
    rb = next((t["i"] for t in drift_trace if t["state"] in ("ROLLBACK", "TEACHER")), None)
    after_rb = [r for r in burst_rows if rb is not None and r["i"] > rb]
    criteria = {
        "C1_certified_and_pinned": bool(
            status.get("certified")
            and status.get("lambda") == LAMBDA_HAT
            and status.get("student_adapter_sha256") == N1_ADAPTER
            and status.get("certified_adapter_sha256") == N1_ADAPTER
            and status.get("student") == "ok"
        ),
        "C2_student_served_verified": bool(student) and all(r["verified"] for r in student),
        "C3_router_deferred": bool(decided_defer),
        "C4_no_5xx": all(r["http"] < 500 for r in rows) and all(r["http"] == 200 for r in rows),
    }
    if not a.hosted:
        criteria["C5_drift"] = bool(
            drift0.get("state") == "CERTIFIED"
            and drift_mid.get("state") == "CERTIFIED"  # no rollback before the burst
            and rb is not None
            and rb + 1 <= ROLLBACK_WITHIN
            and not any(r["served_by"] == "student" for r in after_rb)
        )
        criteria["C6_spend_within_ceiling"] = (
            spend1 is not None and spend0 is not None and spend1 - spend0 <= a.ceiling_usd
        )
    if prereg is None:
        stopped = stopped or "no --prereg-commit: not a scored run"
    passed = stopped is None and all(criteria.values())
    lat = [r["client_ms"] for r in student]
    value = (
        f"{len(student)}/{len(normal)} normal questions served by the student (all verified: "
        f"{criteria['C2_student_served_verified']}), {len(decided_defer)} deferred by the router"
    )
    if not a.hosted:
        value += (
            f"; drift rollback on burst request {None if rb is None else rb + 1} "
            f"(limit {ROLLBACK_WITHIN}), 0 student answers after it: "
            f"{not any(r['served_by'] == 'student' for r in after_rb)}"
        )
    out = {
        "format": "distillery-w-e2e-v1",
        "mode": "hosted" if a.hosted else "local-live",
        "base_url": base,
        "started_at": started,
        "finished_at": datetime.now(UTC).isoformat(),
        "pass": passed,
        "prereg": prereg,
        "code": code_pin,
        "stopped": stopped,
        "value_display": value,
        "criteria": criteria,
        "status_before": {
            k: status.get(k)
            for k in (
                "certified",
                "lambda",
                "model_sha256",
                "prereg_commit",
                "student",
                "student_adapter_sha256",
                "certified_adapter_sha256",
                "drift",
            )
        },  # fmt: skip
        "status_after": {k: status1.get(k) for k in ("drift", "today")},
        "normal_questions_sha256": _sha("\n".join(NORMAL)),
        "burst_questions_sha256": BURST_SHA256 if burst else None,
        "counts": {
            "normal": len(normal),
            "normal_served_by_student": len(student),
            "normal_deferred_by_router": len(decided_defer),
            "burst": len(burst_rows),
            "burst_served_by_student": sum(r["served_by"] == "student" for r in burst_rows),
            "rollback_on_burst_request": None if rb is None else rb + 1,
            "student_answers_after_rollback": sum(r["served_by"] == "student" for r in after_rb),
        },
        "student_latency_ms": {
            "n": len(lat),
            "median": statistics.median(lat) if lat else None,
            "max": max(lat) if lat else None,
        },
        "drift_before_burst": drift_mid.get("state"),
        "drift_after": {k: drift1.get(k) for k in ("state", "audits", "detector")},
        "drift_trace": drift_trace,
        "savings": savings,
        "ledger": None
        if ledger is None
        else {
            "spend_before_usd": spend0,
            "spend_after_usd": spend1,
            "delta_usd": None if spend1 is None or spend0 is None else round(spend1 - spend0, 6),
            "ceiling_usd": a.ceiling_usd,
            "note": "sandbox seconds are recorded at 0 (unpriced): this delta is a lower bound",
        },  # fmt: skip
        "requests": rows,
    }
    a.out.parent.mkdir(parents=True, exist_ok=True)
    with a.out.open("x", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1, sort_keys=True)
        fh.write("\n")
    print(json.dumps({"pass": passed, "value": value, "criteria": criteria}, indent=1))
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
