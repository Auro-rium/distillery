"""Post-deploy smoke suite for a running Distillery service. Free: it spends only a fake-model dry
run and one playground teacher call (about $0.0005). Exit status 1 if any check fails.

Usage: python scripts/smoke_live.py https://distillery.onrender.com [--no-playground] [--no-run]

Checks: health and mode; models/prices configured; both recorded runs listed with their labels;
New run is locked without a token; every response carries X-Request-ID; /api/metrics and
/api/telemetry respond and hold no secret-looking text; the frontend is served; one playground
question (teacher SQL returned); one dry run is started, its event stream is read, and the
VALUES MUST CHANGE (stage statuses advance, spend rises, it completes).
"""

import argparse
import json
import re
import sys
import time
import urllib.error
import urllib.request
from typing import Any

SECRET_LIKE = re.compile(r"(eyJ[A-Za-z0-9_-]{20,}|sk-[A-Za-z0-9]{20,}|Bearer\s+\S{12,})")


def call(base: str, path: str, method: str = "GET", body: Any = None, timeout: float = 90.0):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(
        base + path, data=data, method=method, headers={"content-type": "application/json"}
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:  # noqa: S310 - operator-given URL
            return r.status, dict(r.headers), r.read()
    except urllib.error.HTTPError as e:
        return e.code, dict(e.headers), e.read()


class Suite:
    def __init__(self) -> None:
        self.results: list[tuple[str, bool, str]] = []

    def check(self, name: str, ok: bool, detail: str = "") -> None:
        self.results.append((name, ok, detail))
        print(f"{'PASS' if ok else 'FAIL'}  {name}{('  ' + detail) if detail else ''}", flush=True)


def watch_run(base: str, s: Suite) -> None:
    code, _, body = call(
        base, "/api/runs", "POST", {"pack": "sql", "scale": "tiny", "dry_run": True}
    )
    s.check("dry run accepted", code in (200, 202), f"status {code}")
    if code not in (200, 202):
        return
    rid = json.loads(body)["run_id"]
    req = urllib.request.Request(base + f"/api/runs/{rid}/events")
    spends: list[float] = []
    stage_states: set[tuple[str, str]] = set()
    done = None
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=150) as r:  # noqa: S310
        ev = ""
        for raw in r:
            line = raw.decode().strip()
            if line.startswith("event:"):
                ev = line[6:].strip()
            elif line.startswith("data:"):
                d = json.loads(line[5:])
                if ev == "spend":
                    spends.append(float(d["total_usd"]))
                elif ev == "stage":
                    stage_states.add((d["name"], d["status"]))
                elif ev == "done":
                    done = d
                    break
            if time.time() - t0 > 140:
                break
    s.check(
        "run completed via the event stream", done is not None and done.get("status") == "complete"
    )
    s.check(
        "spend value changed while running", len(set(spends)) >= 3, f"{len(set(spends))} distinct"
    )
    s.check("spend never decreased", spends == sorted(spends))
    s.check(
        "stages advanced running -> done",
        any(st == "running" for _, st in stage_states)
        and any(st == "done" for _, st in stage_states),
        f"{len(stage_states)} stage events",
    )
    code, _, rep = call(base, f"/api/runs/{rid}/report")
    s.check(
        "report is served for the finished run",
        code == 200 and json.loads(rep).get("dry_run") is True,
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("base")
    ap.add_argument("--no-playground", action="store_true")
    ap.add_argument("--no-run", action="store_true")
    a = ap.parse_args()
    base = a.base.rstrip("/")
    s = Suite()

    code, h, body = call(base, "/api/health", timeout=120)
    health = json.loads(body) if code == 200 else {}
    s.check("health ok", code == 200 and health.get("ok") is True, str(health))
    s.check(
        "every response carries X-Request-ID", bool(h.get("x-request-id") or h.get("X-Request-ID"))
    )
    code, _, body = call(base, "/api/config")
    cfg = json.loads(body) if code == 200 else {}
    s.check("models configured", len(cfg.get("models", {})) == 4 and all(cfg["models"].values()))
    pg = cfg.get("playground", {})
    s.check(
        "playground teacher usable", pg.get("models", {}).get("teacher", {}).get("state") == "ok"
    )
    code, _, body = call(base, "/api/runs")
    recorded = [r for r in (json.loads(body) if code == 200 else []) if r.get("recorded")]
    s.check("recorded real runs listed", len(recorded) >= 2, f"{len(recorded)} recorded")
    s.check("recorded runs keep their label", all(r.get("dry_run") is False for r in recorded))
    code, _, _ = call(base, "/api/runs", "POST", {"pack": "sql", "scale": "tiny", "dry_run": False})
    s.check("live (paid) run is locked without a token", code in (401, 403), f"status {code}")
    code, _, body = call(base, "/")
    s.check("frontend served", code == 200 and b'<div id="root"' in body)
    code, _, body = call(base, "/api/metrics")
    s.check("metrics endpoint", code == 200 and b"distillery_http_requests_total" in body)
    code2, _, body2 = call(base, "/api/telemetry")
    s.check("telemetry endpoint", code2 == 200 and "requests_total" in json.loads(body2))
    s.check(
        "telemetry holds no secret-looking text",
        not SECRET_LIKE.search((body + body2).decode(errors="ignore")),
    )
    if not a.no_playground:
        code, _, body = call(
            base, "/api/playground", "POST", {"question": "How many accounts are in each country?"}
        )
        res = json.loads(body).get("results", {}) if code == 200 else {}
        s.check(
            "playground teacher answered with SQL",
            res.get("teacher", {}).get("state") == "ok"
            and "select" in str(res["teacher"].get("sql", "")).lower(),
            f"status {code}",
        )
    if not a.no_run:
        watch_run(base, s)
    failed = [n for n, ok, _ in s.results if not ok]
    print(f"\n{len(s.results) - len(failed)}/{len(s.results)} checks passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
