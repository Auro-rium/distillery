# ruff: noqa: S101, E501, BLE001, S604
"""S2 (live): Sandboxes spike. Measures import, sqlite, CPU/RAM, egress, branching, 40-way parallelism.

Reads NEBIUS_API_KEY and NEBIUS_AI_PROJECT (or NEBIUS_PROJECT_ID) from the environment; never prints them.
Every command is tiny. Prints one JSON summary at the end.
"""

import asyncio
import json
import os
import time

from contree_sdk import Contree
from contree_sdk.auth import IAMAuth
from contree_sdk.config import ContreeConfig

PARALLEL = 40


def _txt(v: object) -> str:
    return v.decode(errors="replace") if isinstance(v, bytes) else str(v or "")


async def timed(coro):  # type: ignore[no-untyped-def]
    t = time.monotonic()
    try:
        return await coro, time.monotonic() - t, None
    except Exception as e:
        return None, time.monotonic() - t, f"{type(e).__name__}: {str(e)[:300]}"


async def main() -> None:
    proj = os.environ.get("NEBIUS_PROJECT_ID") or os.environ["NEBIUS_AI_PROJECT"]
    sdk = Contree(
        config=ContreeConfig(auth=IAMAuth(token=os.environ["NEBIUS_API_KEY"], project_id=proj))
    )
    out: dict = {}

    image, dt, err = await timed(sdk.images.oci("docker://python:3.12-slim"))
    out["import_python312_slim"] = {"seconds": round(dt, 2), "error": err}
    if err:
        print(json.dumps(out, indent=1, default=str))
        return

    async def sh(img, cmd):  # type: ignore[no-untyped-def]
        r, dt, err = await timed(img.run(shell=cmd, timeout=60))
        if err:
            return {"error": err, "seconds": round(dt, 2)}
        return {
            "exit": r.exit_code,
            "stdout": _txt(r.stdout)[:300].strip(),
            "stderr": _txt(r.stderr)[:200].strip(),
            "seconds": round(dt, 2),
            "uuid": getattr(r, "uuid", None),
        }

    out["sqlite_and_python"] = await sh(
        image,
        'python3 -c "import sqlite3,sys; print(sqlite3.sqlite_version, sys.version.split()[0])"',
    )
    out["cpu_ram"] = await sh(
        image, "nproc; grep -E 'MemTotal|MemAvailable' /proc/meminfo; df -h / | tail -1"
    )
    out["egress_pypi"] = await sh(
        image,
        "python3 -c \"import urllib.request as u; print(u.urlopen('https://pypi.org/simple/pip/', timeout=10).status)\"",
    )
    out["egress_huggingface"] = await sh(
        image,
        "python3 -c \"import urllib.request as u; print(u.urlopen('https://huggingface.co/api/models/Qwen/Qwen3-1.7B', timeout=10).status)\"",
    )
    out["egress_hf_cdn_head"] = await sh(
        image,
        "python3 -c \"import urllib.request as u; r=u.Request('https://huggingface.co/Qwen/Qwen3-1.7B/resolve/main/config.json', method='HEAD'); print(u.urlopen(r, timeout=10).status)\"",
    )

    # branching: one checkpoint, three children; same parent state, distinct uuids
    parent, dt, err = await timed(image.run(shell="echo base > /tmp/state.txt", disposable=False))
    out["branch_parent"] = {
        "uuid": getattr(parent, "uuid", None),
        "seconds": round(dt, 2),
        "error": err,
    }
    if parent is not None:
        kids = []
        for label in ("A", "B", "C"):
            k, dt, err = await timed(
                parent.run(
                    shell=f"echo {label} >> /tmp/state.txt && cat /tmp/state.txt", disposable=False
                )
            )
            kids.append(
                {
                    "label": label,
                    "uuid": getattr(k, "uuid", None),
                    "stdout": _txt(getattr(k, "stdout", "")).split(),
                    "seconds": round(dt, 2),
                    "error": err,
                }
            )
        out["branch_children"] = kids

    # 40 parallel disposable runs
    t0 = time.monotonic()
    results = await asyncio.gather(
        *[image.run(shell=f'python3 -c "print({i}*2)"', timeout=60) for i in range(PARALLEL)],
        return_exceptions=True,
    )
    total = time.monotonic() - t0
    ok = [
        r
        for r in results
        if not isinstance(r, Exception) and r.exit_code == 0 and _txt(r.stdout).strip().isdigit()
    ]
    out["parallel"] = {
        "n": PARALLEL,
        "ok": len(ok),
        "failed": PARALLEL - len(ok),
        "total_seconds": round(total, 2),
        "first_error": next(
            (f"{type(r).__name__}: {str(r)[:200]}" for r in results if isinstance(r, Exception)),
            None,
        ),
    }
    print(json.dumps(out, indent=1, default=str))


asyncio.run(main())
