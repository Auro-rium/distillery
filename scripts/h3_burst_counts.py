#!/usr/bin/env python3
"""Counts-only evidence for the W-D burst (free, no network): how the certified router routes the
30 scored burst questions, each counted once (the W-D replay draws them with replacement).

  uv run python scripts/h3_burst_counts.py --router-dir <run>/router --out counts.json

Reads ``signals_burst.json``, ``cert.json`` and ``model.json``; writes counts, hashes and the
pre-registration the burst was scored under, never question or SQL text. Never overwrites.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

from distillery.router import RouterModel
from distillery.router_eval import features_of, write_new_json


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--router-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)
    if a.out.exists():
        print(f"error: {a.out} already exists; it is never overwritten", file=sys.stderr)
        return 2
    d = a.router_dir
    sig = json.loads((d / "signals_burst.json").read_text(encoding="utf-8"))
    cert = json.loads((d / "cert.json").read_text(encoding="utf-8"))
    mdoc = json.loads((d / "model.json").read_text(encoding="utf-8"))
    model = RouterModel.from_json(mdoc["model"])
    if cert["model_sha256"] != model.sha256():
        print("error: cert.json was made for a different model than model.json", file=sys.stderr)
        return 2
    lam = float(cert["lambda_hat"])
    rows = sig["rows"]
    served = [r for r in rows if model.score(features_of(r)) >= lam]
    out = {
        "format": "distillery-h3-burst-counts-v1",
        "run_id": sig["run_id"],
        "prereg": {k: sig["prereg"][k] for k in ("commit", "committed_at")},
        "burst_sha256": sig["burst"]["sha256"],
        "adapter_sha256": sig["models"]["adapter_sha256"],
        "inputs": {
            "signals_burst": _sha(d / "signals_burst.json"),
            "cert": _sha(d / "cert.json"),
            "model": _sha(d / "model.json"),
        },
        "lambda_hat": lam,
        "n": len(rows),
        "student_correct": sum(bool(r["correct"]) for r in rows),
        "teacher_correct": sum(bool(r["teacher_correct"]) for r in rows),
        "exec_ok": sum(bool(r["exec_ok"]) for r in rows),
        "deferred": len(rows) - len(served),
        "served_by_student": len(served),
        "served_wrong": sum(not r["correct"] for r in served),
    }
    write_new_json(a.out, out)
    print(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
