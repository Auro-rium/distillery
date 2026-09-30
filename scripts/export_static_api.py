"""Export the API's answers for FINISHED runs as static files, for a host with no backend (Vercel).

Runs the real API code against a real data directory (no API key, so nothing can spend), saves each
GET response byte for byte under frontend/public/static-api/, and rewrites the matching rewrites and
headers into frontend/vercel.json. Runs that are not finished are skipped: a frozen copy of a run in
progress would claim to be live. Re-run and redeploy to refresh.

Usage: python scripts/export_static_api.py --home .distillery/live [--only RUN_ID ...]
"""

import argparse
import json
import re
import shutil
import sys
from datetime import UTC, datetime
from pathlib import Path

from fastapi.testclient import TestClient

from distillery.server import create_app, settings_from_env

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "frontend" / "public" / "static-api"
VERCEL = ROOT / "frontend" / "vercel.json"
KINDS = ("all", "fixed", "still_wrong", "regressed")
GENERATED = "/static-api/"
JSON_TYPE = [
    {"key": "Content-Type", "value": "application/json"},
    {"key": "Cache-Control", "value": "no-cache"},
]
_ID = re.compile(r"^[A-Za-z0-9._-]+$")


def _write(rel: str, body: bytes) -> None:
    path = OUT / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(body)


def _recorded(body: bytes, at: str) -> bytes:
    """Label a frozen copy as recorded (the UI's banner comes from these payload fields)."""
    data = json.loads(body)
    for item in data if isinstance(data, list) else [data]:
        item["recorded"], item["recorded_at"] = True, at
    return json.dumps(data).encode()


def export(home: Path, only: list[str] | None) -> dict[str, object]:
    env = {"DISTILLERY_HOME": str(home)}  # no NEBIUS_API_KEY: replay-only, cannot spend
    app = create_app(settings_from_env(env, root=home))
    rewrites: list[dict[str, object]] = []
    headers: list[dict[str, object]] = []
    shutil.rmtree(OUT, ignore_errors=True)
    exported: list[str] = []
    at = datetime.now(UTC).replace(microsecond=0).isoformat()
    with TestClient(app) as c:
        for path, rel in (
            ("/api/health", "health"),
            ("/api/config", "config"),
            ("/api/replay", "replay"),
        ):
            r = c.get(path)
            r.raise_for_status()
            _write(rel, r.content)
            rewrites.append({"source": path, "destination": f"{GENERATED}{rel}"})
        runs = c.get("/api/runs").json()
        keep = []
        for run in runs:
            rid = run["run_id"]
            if only and rid not in only:
                continue
            if not _ID.match(rid) or run["status"] != "complete":
                print(f"skip {rid}: status {run['status']!r} (only finished runs are exported)")
                continue
            keep.append(run)
            base = f"/api/runs/{rid}"
            detail = c.get(base)
            detail.raise_for_status()
            _write(f"by-run/{rid}/detail", _recorded(detail.content, at))
            rewrites.append({"source": base, "destination": f"{GENERATED}by-run/{rid}/detail"})
            for name in ("report", "tree"):
                r = c.get(f"{base}/{name}")
                r.raise_for_status()
                _write(
                    f"by-run/{rid}/{name}",
                    _recorded(r.content, at) if name == "report" else r.content,
                )
                rewrites.append(
                    {"source": f"{base}/{name}", "destination": f"{GENERATED}by-run/{rid}/{name}"}
                )
            ev = c.get(f"{base}/events")
            ev.raise_for_status()
            _write(f"by-run/{rid}/events", ev.content)
            rewrites.append(
                {"source": f"{base}/events", "destination": f"{GENERATED}by-run/{rid}/events"}
            )
            headers.append(
                {
                    "source": f"{GENERATED}by-run/{rid}/events",
                    "headers": [{"key": "Content-Type", "value": ev.headers["content-type"]}],
                }
            )
            for kind in KINDS:
                r = c.get(f"{base}/examples", params={"kind": kind, "limit": 200})
                r.raise_for_status()
                _write(f"by-run/{rid}/examples-{kind}", r.content)
                rewrites.append(
                    {
                        "source": f"{base}/examples",
                        "has": [{"type": "query", "key": "kind", "value": kind}],
                        "destination": f"{GENERATED}by-run/{rid}/examples-{kind}",
                    }
                )
                headers.append(
                    {
                        "source": f"{GENERATED}by-run/{rid}/examples-{kind}",
                        "headers": [
                            {"key": k, "value": v}
                            for k, v in r.headers.items()
                            if k.lower().startswith("x-examples-")
                        ],
                    }
                )
            exported.append(rid)
        _write("runs-index", _recorded(json.dumps(keep).encode(), at))
        rewrites.append({"source": "/api/runs", "destination": f"{GENERATED}runs-index"})
    json_types = [
        {"key": "Content-Type", "value": "application/json"},
        {"key": "Cache-Control", "value": "no-cache"},
    ]
    headers.insert(
        0, {"source": f"{GENERATED}(.*)", "headers": json_types}
    )  # specific rules below win
    return {"rewrites": rewrites, "headers": headers, "runs": exported}


def patch_vercel(gen: dict[str, object]) -> None:
    cfg = json.loads(VERCEL.read_text())
    keep_rw = [
        r for r in cfg["rewrites"] if not str(r.get("destination", "")).startswith(GENERATED)
    ]
    keep_h = [h for h in cfg["headers"] if not str(h["source"]).startswith(("/api/", GENERATED))]
    # generated rewrites first (most specific), then the SPA fallback
    cfg["rewrites"] = [*gen["rewrites"], *keep_rw]  # type: ignore[misc]
    # generic header first, per-file headers after it so they override
    cfg["headers"] = [*keep_h, *gen["headers"]]  # type: ignore[misc]
    VERCEL.write_text(json.dumps(cfg, indent=2) + "\n")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--home", required=True, type=Path)
    ap.add_argument("--only", nargs="*")
    a = ap.parse_args()
    gen = export(a.home.resolve(), a.only)
    if not gen["runs"]:
        print("no finished runs exported; refusing to touch vercel.json", file=sys.stderr)
        return 1
    patch_vercel(gen)
    print(f"exported runs: {gen['runs']}; wrote {OUT} and updated {VERCEL}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
