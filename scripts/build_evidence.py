#!/usr/bin/env python3
"""Build deploy/evidence.json: the curated "proof ladder" shown by the UI.

Every number is read from docs/proofs/evidence/*.json (this script only knows keys, not values).
No sealed text (questions, SQL, model outputs) is copied. Output is deterministic for a commit:
`generated_at` is the commit time, not the wall clock.

    python scripts/build_evidence.py [--out deploy/evidence.json] [--commit SHA]
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
REPO_URL = "https://github.com/Auro-rium/distillery"
EVIDENCE_DIR = "docs/proofs/evidence"
DEFAULT_OUT = REPO_ROOT / "deploy" / "evidence.json"


def _load(name: str) -> dict[str, Any]:
    data = json.loads((REPO_ROOT / EVIDENCE_DIR / f"{name}.json").read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise TypeError(f"{name}.json must be a JSON object")
    return data


def _git(*args: str) -> str | None:
    try:
        out = subprocess.run(  # noqa: S603
            ["git", *args],  # noqa: S607
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return out.stdout.strip() or None


def _frac(num: int, den: int) -> str:
    return f"{num}/{den}"


def _parse_frac(s: str) -> tuple[int, int]:
    a, b = s.split("/")[0], s.split("/")[1].split()[0]
    return int(a), int(b)


def _pct(x: float, digits: int = 1) -> str:
    return f"{x * 100:.{digits}f}%"


def _proof(
    commit: str,
    pid: str,
    title: str,
    verdict: str,
    headline: dict[str, Any],
    metrics: list[dict[str, Any]],
    file: str,
    series: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    path = f"{EVIDENCE_DIR}/{file}.json"
    p: dict[str, Any] = {
        "id": pid,
        "title": title,
        "verdict": verdict,
        "headline": headline,
        "metrics": metrics,
    }
    if series is not None:
        p["series"] = series
    p["evidence_path"] = path
    p["evidence_url"] = f"{REPO_URL}/blob/{commit}/{path}"
    return p


def p1_11(commit: str) -> dict[str, Any]:
    d = _load("p1_11_overfit")
    a = d["models"]["adapter"]["train64"]
    b = d["models"]["base"]["train64"]
    ac, bc = round(a["accuracy"] * a["n"]), round(b["accuracy"] * b["n"])
    curve = [
        {"step": r["step"], "train_loss": r["train_loss"], "valid_loss": r["valid_loss"]}
        for r in d["loss_curve"]
    ]
    return _proof(
        commit, "W1-P1.11", "Fine-tune pipeline can overfit a tiny set",
        "PASS" if ac == a["n"] else "FAIL",
        {"label": "adapter on train set", "value": _frac(ac, a["n"])},
        [
            {"label": "base model on same set", "value": _frac(bc, b["n"])},
            {"label": "base model", "value": d["base_model"]},
            {"label": "trained steps", "value": d["job"]["trained_steps"]},
            {"label": "learning rate", "value": d["hyperparameters"]["learning_rate"]},
            {"label": "final train loss", "value": round(curve[-1]["train_loss"], 6)},
        ],
        "p1_11_overfit", curve,
    )  # fmt: skip


def p1_12(commit: str) -> dict[str, Any]:
    d = _load("p1_12_decision")
    win = d["arms"][d["winner"]]
    other = d["arms"]["b" if d["winner"] == "a" else "a"]
    return _proof(
        commit, "W1-P1.12", "Pilot: learning-rate arms on held-out dev",
        "PASS" if str(d["verdict"]).startswith("PASS") else "FAIL",
        {"label": f"arm {d['winner'].upper()} dev accuracy",
         "value": _frac(win["correct"], win["n"])},
        [
            {"label": "base model", "value": _frac(d["base"]["correct"], d["base"]["n"])},
            {"label": f"arm {d['winner'].upper()} learning rate", "value": win["lr"]},
            {"label": "other arm", "value": _frac(other["correct"], other["n"])},
            {"label": "other arm learning rate", "value": other["lr"]},
            {"label": "arm 95% interval (Wilson)",
             "value": f"{win['wilson95'][0]:.3f} to {win['wilson95'][1]:.3f}"},
        ],
        "p1_12_decision",
    )  # fmt: skip


def p2(commit: str) -> dict[str, Any]:
    d = _load("p2_data")
    lc, lt = _parse_frac(d["P2.1_label_correct"])
    sets = [d["P2.3_filter_recall_trained_rows"], d["P2.3_filter_recall_gated_train_pool_gold"]]
    caught = missed_den = 0
    for s in sets:
        c, t = _parse_frac(s["own_mutations"]["recall_excluding_equivalent_robust"])
        caught, missed_den = caught + c, missed_den + t
    return _proof(
        commit, "W2", "Training data: labels and filter",
        "PASS" if lc == lt and caught == missed_den else "PARTIAL",
        {"label": "labels verified correct", "value": _frac(lc, lt)},
        [
            {"label": "filter recall on true misses", "value": _frac(caught, missed_den)},
            {"label": "pool rows", "value": d["pool_rows"]},
            {"label": "accidental match rate under perturbation",
             "value": d["P2.2_perturbed"]["accidental_match_rate"]},
        ],
        "p2_data",
    )  # fmt: skip


def p3(commit: str) -> dict[str, Any]:
    d = _load("p3_calibrate")
    gc, gt = _parse_frac(d["P3.1_verifier"]["gold_vs_gold"])
    pairs = d["P3.1_verifier"]["handwritten_pairs"]
    power = d["P3.5_power"]["promote_rate"]
    key = "n=300 true_ratio=0.95"
    low = "n=300 true_ratio=0.8"
    ok = gc == gt and all(p["as_expected"] for p in pairs) and power[key] >= 0.95
    return _proof(
        commit, "W3", "Gate: verifier and statistical power",
        "PASS" if ok else "FAIL",
        {"label": "verifier gold vs gold", "value": _frac(gc, gt)},
        [
            {"label": "PROMOTE rate at true ratio 0.95, n=300", "value": _pct(power[key], 0)},
            {"label": "PROMOTE rate at true ratio 0.80, n=300", "value": _pct(power[low], 0)},
            {"label": "handwritten verifier pairs as expected",
             "value": _frac(sum(1 for p in pairs if p["as_expected"]), len(pairs))},
            {"label": "simulations per cell", "value": d["P3.5_power"]["sims_per_cell"]},
        ],
        "p3_calibrate",
    )  # fmt: skip


def p4(commit: str) -> dict[str, Any]:
    d = _load("p4_run_summary")
    g = d["gate_heldout"]
    return _proof(
        commit, "W4", "Full gated run on a sealed held-out set",
        "PASS" if d["decision"] == "PROMOTE" else "FAIL",
        {"label": "decision", "value": d["decision"]},
        [
            {"label": "student held-out accuracy", "value": _pct(g["student_acc"])},
            {"label": "teacher held-out accuracy", "value": _pct(g["teacher_acc"])},
            {"label": "base held-out accuracy", "value": _pct(g["base_acc"])},
            {"label": "student/teacher ratio (95% CI)",
             "value": f"{g['ratio_point']:.3f} ({g['ratio_lo']:.3f} to {g['ratio_hi']:.3f})"},
            {"label": "held-out n", "value": g["n"]},
            {"label": "run cost (estimate)", "value": round(d["cost"]["run_total_usd"], 2)},
        ],
        "p4_run_summary",
    )  # fmt: skip


def p5(commit: str) -> dict[str, Any]:
    d = _load("p5_live")
    checks = d["checks"]
    b3 = checks["B3_job_adopted_not_recreated"]
    results = [c["result"] for c in checks.values()]
    passed = results.count("PASS")
    return _proof(
        commit, "W5", "Kill and resume: fine-tune job adopted",
        "PASS" if passed == len(results) else "PARTIAL",
        {"label": "B3 job adopted, not recreated", "value": b3["result"]},
        [
            {"label": "fine-tune jobs started", "value": b3["finetune_job_started_events"]},
            {"label": "distinct job ids", "value": b3["distinct_job_ids"]},
            {"label": "adopted events", "value": len(b3["adopted_events"])},
            {"label": "kill-resume checks passed", "value": _frac(passed, len(results))},
            {"label": "checks failed",
             "value": ", ".join(sorted(re.sub(r"^([AB]\d+b?)_.*", r"\1", k)
                                      for k, c in checks.items() if c["result"] != "PASS"))},
        ],
        "p5_live",
    )  # fmt: skip


def build(commit: str, generated_at: str) -> dict[str, Any]:
    return {
        "generated_at": generated_at,
        "commit": commit,
        "repo_url": REPO_URL,
        "proofs": [f(commit) for f in (p1_11, p1_12, p2, p3, p4, p5)],
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--commit", default=None, help="commit sha to pin URLs to (default: HEAD)")
    args = ap.parse_args(argv)
    commit = args.commit or _git("rev-parse", "HEAD")
    if not commit:
        print("cannot determine the commit; pass --commit", file=sys.stderr)
        return 1
    when = _git("show", "-s", "--format=%cI", commit) or "unknown"
    text = json.dumps(build(commit, when), indent=2, ensure_ascii=False) + "\n"
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(text, encoding="utf-8")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
