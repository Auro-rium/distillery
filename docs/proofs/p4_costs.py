# ruff: noqa: S310, E501  (evidence tool: fixed https endpoint, long help strings)
"""P4 cost meter (read-only API calls): fine-tune trained_tokens + Sandboxes operations in a window.

Run:  set -a && . ./.env && set +a
      .venv/bin/python docs/proofs/p4_costs.py --since ISO --until ISO [--job ftjob-...]*
          [--ft-usd-per-mtok X] [--sandbox-usd-per-cpu-hour Y] [--out path.json]
consumed_cpu units are undocumented; this tool ASSUMES CPU-seconds (labelled as such). Prints JSON.
"""

import argparse
import json
import os
import urllib.parse
import urllib.request
from collections import defaultdict

import openai

from distillery.finetune import FineTuneClient

URL = "https://api.tokenfactory.nebius.com/sandboxes/v1/operations"


def fetch_ops(since: str, until: str) -> list[dict]:
    hdr = {
        "Authorization": f"Bearer {os.environ['NEBIUS_API_KEY']}",
        "Project": os.environ["NEBIUS_AI_PROJECT"],
    }
    ops, offset = [], 0
    while True:
        q = urllib.parse.urlencode({"since": since, "until": until, "limit": 200, "offset": offset})
        with urllib.request.urlopen(
            urllib.request.Request(f"{URL}?{q}", headers=hdr), timeout=60
        ) as r:
            body = json.load(r)
        page = (
            body
            if isinstance(body, list)
            else body.get("operations") or body.get("items") or body.get("data") or []
        )
        ops += page
        if len(page) < 200:
            return ops
        offset += len(page)


def num(v) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return 0.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--since", required=True)
    ap.add_argument("--until", required=True)
    ap.add_argument("--job", action="append", default=[])
    ap.add_argument("--ft-usd-per-mtok", type=float)
    ap.add_argument("--sandbox-usd-per-cpu-hour", type=float)
    ap.add_argument("--out")
    a = ap.parse_args()

    ft = FineTuneClient(
        openai.OpenAI(
            base_url=os.environ["NEBIUS_BASE_URL"],
            api_key=os.environ["NEBIUS_API_KEY"],
            max_retries=0,
        )
    )
    jobs = {j: num(ft.raw_job(j).get("trained_tokens")) for j in a.job}
    tokens = sum(jobs.values())

    ops = fetch_ops(a.since, a.until)
    by = defaultdict(lambda: {"ops": 0, "duration_sum": 0.0, "consumed_cpu_sum": 0.0})
    per_hour: dict[str, int] = defaultdict(int)
    for o in ops:
        s = by[str(o.get("status"))]
        s["ops"] += 1
        s["duration_sum"] += num(o.get("duration"))
        s["consumed_cpu_sum"] += num(o.get("consumed_cpu"))
        per_hour[
            str(o.get("created_at") or o.get("started_at") or o.get("start_time") or "?")[:13]
        ] += 1
    cpu_s = sum(v["consumed_cpu_sum"] for v in by.values())

    cost: dict = {}
    if a.ft_usd_per_mtok is not None:
        cost["finetune_usd"] = tokens / 1e6 * a.ft_usd_per_mtok
    else:
        cost["finetune_usd"] = "price not supplied"
    if a.sandbox_usd_per_cpu_hour is not None:
        cost["sandbox_usd"] = cpu_s / 3600 * a.sandbox_usd_per_cpu_hour
    else:
        cost["sandbox_usd"] = "price not supplied"
    if all(isinstance(v, float) for v in cost.values()):
        cost["total_usd"] = sum(cost.values())
    out = {
        "window": [a.since, a.until],
        "finetune": {"trained_tokens_by_job": jobs, "trained_tokens_total": tokens},
        "sandbox": {
            "total_ops": len(ops),
            "by_status": by,
            "consumed_cpu_total_ASSUMED_CPU_SECONDS": cpu_s,
            "ops_per_hour_utc": dict(sorted(per_hour.items())),
            "duration_unit": "as returned (undocumented)",
        },
        "cost": cost,
    }
    text = json.dumps(out, indent=1)
    if a.out:
        open(a.out, "w").write(text + "\n")
    print(text)


main()
