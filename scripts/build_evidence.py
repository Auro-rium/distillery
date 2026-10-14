#!/usr/bin/env python3
"""Build deploy/evidence.json: the curated "proof ladder" shown by the UI.

Every number is read from docs/proofs/evidence/*.json (this script only knows keys, not values).
No sealed text (questions, SQL, model outputs) is copied. Output is deterministic for a commit:
`generated_at` is the commit time, not the wall clock.

`proofs` is the ladder; `receipts` is the same ground as one checkable row per claim (status,
the value as the evidence states it, the artifact and its permalink, the pre-registration commit,
sealed-set and adapter hashes, the gate's inputs, the ledger total). The key paths each receipt
reads are all in the KEYS table below; a claim whose evidence file does not exist yet is
"pending" (value null), never a guessed number.

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


def _load(name: str, root: Path | None = None) -> dict[str, Any]:
    data = json.loads(((root or REPO_ROOT) / EVIDENCE_DIR / f"{name}.json").read_text("utf-8"))
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


def a5(commit: str) -> dict[str, Any]:
    d = _load("a5_chaos")
    checks = d["checks"]
    met = sum(1 for v in checks.values() if v)
    return _proof(
        commit, "A5", "Chaos run: injected faults, no human action",
        "PASS" if d["pass"] else "FAIL",
        {"label": "chaos criteria met", "value": _frac(met, len(checks))},
        [
            {"label": "verdict reached", "value": d["decision"]},
            {"label": "faults injected", "value": sum(d["injections"].values())},
            {"label": "supervisor restarts", "value": d["supervisor_restarts"]},
            {"label": "fine-tune jobs billed", "value": len(d["finetune_jobs"])},
            {"label": "run cost (estimate)", "value": round(d["spend_total_usd"], 2)},
        ],
        "a5_chaos",
    )  # fmt: skip


def b2(commit: str) -> dict[str, Any]:
    d = _load("b2_fair_teacher_s0_s1")
    claims = d["analysis"]["claims"]["heldout"]
    v = claims["vs_strongest_teacher"]
    # PASS only when the pre-registered claim is made; an interval that includes 1.0 is inconclusive
    return _proof(
        commit, "B2", "Fair teacher: 8-shot Super vs the student",
        "PASS" if v["claim"] else "PARTIAL",
        {"label": "student/teacher ratio, lower bound", "value": round(v["ratio_lo"], 3)},
        [
            {"label": "student held-out accuracy", "value": _pct(claims["accuracy"]["student"])},
            {"label": f"strongest teacher ({v['teacher']}) accuracy",
             "value": _pct(claims["accuracy"][v["teacher"]])},
            {"label": "beat-the-teacher claim", "value": "made" if v["claim"] else "not made"},
            {"label": "P4 reproduced exactly",
             "value": "yes" if d["analysis"]["p4_reproduction"]["reproduced"] else "no"},
            {"label": "teacher setups deferred", "value": ", ".join(d["teacher_setups_deferred"])},
        ],
        "b2_fair_teacher_s0_s1",
    )  # fmt: skip


def gate_b(commit: str) -> dict[str, Any]:
    d = _load("gateb_agent_authored")
    g = d["gate"]
    return _proof(
        commit, "Gate B", "Natural phrasing: agent-authored questions",
        "PASS" if d["decision"] == "PROMOTE" else "FAIL",
        {"label": "decision", "value": d["decision"]},
        [
            {"label": "student accuracy", "value": _pct(g["student_acc"])},
            {"label": "base accuracy", "value": _pct(g["base_acc"])},
            {"label": "teacher accuracy (inflated)", "value": _pct(g["teacher_acc"])},
            {"label": "student/teacher ratio lower bound", "value": round(g["ratio_lo"], 3)},
            {"label": "McNemar p", "value": round(g["mcnemar_p"], 2)},
            {"label": "questions", "value": g["n"]},
        ],
        "gateb_agent_authored",
    )  # fmt: skip


def n1(commit: str) -> dict[str, Any]:
    d = _load("n1_natural_round")
    p = d["primary"]
    return _proof(
        commit, "N1", "Natural-phrasing training: fresh agent-authored set",
        "PASS" if p["decision"] == "PROMOTE" else "FAIL",
        {"label": "decision", "value": p["decision"]},
        [
            {"label": "student accuracy", "value": _pct(p["accuracy"]["student"])},
            {"label": "base accuracy", "value": _pct(p["accuracy"]["base"])},
            {"label": "teacher accuracy (inflated)", "value": _pct(p["accuracy"]["teacher"])},
            {"label": "student/teacher ratio lower bound", "value": round(p["ratio"]["lo"], 3)},
            {"label": "McNemar p", "value": f"{p['mcnemar']['p']:.1e}"},
            {"label": "questions", "value": p["n"]},
            {"label": "Gate B re-score of this adapter",
             "value": d["secondary"]["gate_b_rescore"]["status"]},
        ],
        "n1_natural_round",
    )  # fmt: skip


# ------------------------------------------------------------------------------------ receipts
# KEYS is the ONLY place that names keys inside the evidence files for receipts. A path is a dotted
# string; a numeric segment indexes a list ("preregistration.commits.0.sha"). Every path is read
# strictly: a file that exists but lacks a key stops the build (a KEYS mismatch must be seen, not
# hidden as "pending"). A file that does not exist is "pending": value null, nothing guessed.
#
# Per entry:
#   claim     one sentence, no numbers
#   file      evidence file stem under docs/proofs/evidence/ (existing proofs: the proof's own file)
#   sealed    name -> path of a sealed-set / question-file sha256
#   adapter   path of the adapter sha256
#   gate      name -> path of the gate's inputs, or one path to a dict copied verbatim
#   prereg    (commit path, committed_at path); only N1's evidence file records one
#   ledger    (file stem, path) of the USD total of the run behind the claim
#   reason    path of the evidence's own reason string (shown for a non-PASS verdict)
#   also      label of one extra proof metric appended to the value of a PARTIAL proof
# Pending-capable entries (the evidence file may not exist yet) also carry:
#   pass      path of a boolean: true -> verified, false -> failed (absent: present = verified)
#   fail_if   path whose truthy value means the run itself failed
#   value     template for value_display, filled from `value_keys` (name -> path)
# Sources of the key names (the code that WRITES the file), so they can be reconciled:
#   router_eval_test  `distillery router evaluate` body (router_cmd.py), router/eval_test.json
#   drift_burst       `scripts/drift_sim.py --out` document (distillery.drift_sim)
#   h3_burst_counts   `scripts/h3_burst_counts.py` (W-D's prereg, burst hashes, adapter)
#   extras_from       stem of a companion file the prereg/sealed/adapter paths are read from
#   h1_endpoint       `scripts/h1_endpoint.py` evidence (write_evidence)
#   w_e2e             PLACEHOLDER: no writer exists yet; path and keys are NOT decided
_GATE_P4 = {
    "n": "gate_heldout.n",
    "student_acc": "gate_heldout.student_acc",
    "teacher_acc": "gate_heldout.teacher_acc",
    "base_acc": "gate_heldout.base_acc",
    "ratio_point": "gate_heldout.ratio_point",
    "ratio_lo": "gate_heldout.ratio_lo",
    "ratio_hi": "gate_heldout.ratio_hi",
    "mcnemar_p": "gate_heldout.mcnemar_p",
    "ratio_lower_bound_min": "gate_heldout.thresholds.ratio_lower_bound_min",
    "mcnemar_alpha": "gate_heldout.thresholds.mcnemar_alpha",
}
_GATE_B = {k: v.replace("gate_heldout.", "gate.") for k, v in _GATE_P4.items()}
_N1 = "n1_natural_round"
_B2_VS = "analysis.claims.heldout.vs_strongest_teacher"
KEYS: dict[str, dict[str, Any]] = {
    "W1-P1.11": {
        "claim": "The fine-tune pipeline can overfit a tiny training set: the adapter reproduces "
        "the answers it was trained on.",
        "adapter": "adapter_sha256",
    },
    "W1-P1.12": {
        "claim": "A learning-rate pilot on a held-out dev set picks a working fine-tune setting.",
    },
    "W2": {
        "claim": "Training labels are correct and the filter catches every true miss.",
        "also": "filter recall on true misses",
    },
    "W3": {
        "claim": "The gate's verifier and its statistics work: gold matches gold, and the test has "
        "the power it was designed for.",
    },
    "W4": {
        "claim": "The gated run promotes the student on a sealed held-out set: its accuracy ratio "
        "to the teacher clears the pre-registered lower bound.",
        "sealed": {"heldout": "heldout_sealed_sha256", "stress": "stress.sha256"},
        "adapter": "artifact.adapter_sha256",
        "gate": _GATE_P4,
        "ledger": ("p4_run_summary", "cost.run_total_usd"),
    },
    "W5": {
        "claim": "A killed run resumes and adopts its fine-tune job instead of creating a second "
        "one, and every kill-resume check passes.",
        "also": "kill-resume checks passed",
        "ledger": ("p5_live", "checks.A2_spend_meter_matches_real_spend_rows.spend_total_usd"),
    },
    "A5": {
        "claim": "A chaos run with injected faults reaches a verdict with no human action.",
        "gate": "checks",
        "ledger": ("a5_chaos", "spend_total_usd"),
    },
    "B2": {
        "claim": "The student beats the strongest fair (few-shot) teacher: the lower bound of its "
        "accuracy ratio exceeds one.",
        "also": "beat-the-teacher claim",
        "sealed": {"heldout": "heldout_sealed_sha256"},
        "adapter": "adapter_sha256",
        "gate": {
            "n": f"{_B2_VS}.n",
            "student_correct": f"{_B2_VS}.student_correct",
            "teacher_correct": f"{_B2_VS}.teacher_correct",
            "ratio_point": f"{_B2_VS}.ratio_point",
            "ratio_lo": f"{_B2_VS}.ratio_lo",
            "ratio_hi": f"{_B2_VS}.ratio_hi",
            "claim_rule": f"{_B2_VS}.threshold",
        },
        "ledger": (_N1, "spend.efforts.B2.usd"),
    },
    "Gate B": {
        "claim": "The student holds up on naturally phrased, agent-authored questions: its "
        "accuracy ratio to the teacher clears the pre-registered lower bound.",
        "reason": "gate.reasons.0",
        "sealed": {"questions": "question_file_sha256", "confirmed_set": "confirmed_file_sha256"},
        "adapter": "adapter_sha256",
        "gate": _GATE_B,
        "ledger": (_N1, "spend.efforts.Gate B.usd"),
    },
    "N1": {
        "claim": "A student trained on naturally phrased questions clears the same pre-registered "
        "bound on a fresh agent-authored set.",
        "reason": "primary.reasons.0",
        "sealed": {
            "n1_eval": "primary.set.sealed_set_sha256",
            "heldout": "secondary.heldout.sealed_sha256",
            "stress": "secondary.stress.sealed_sha256",
        },
        "adapter": "artifact.adapter_sha256",
        "gate": {
            "n": "primary.n",
            "student_acc": "primary.accuracy.student",
            "teacher_acc": "primary.accuracy.teacher",
            "base_acc": "primary.accuracy.base",
            "ratio_point": "primary.ratio.point",
            "ratio_lo": "primary.ratio.lo",
            "ratio_hi": "primary.ratio.hi",
            "mcnemar_p": "primary.mcnemar.p",
            "ratio_lower_bound_min": "primary.thresholds.ratio_lower_bound_min",
            "mcnemar_alpha": "primary.thresholds.mcnemar_alpha",
        },
        "prereg": ("preregistration.commits.0.sha", "preregistration.commits.0.committed_at"),
        "ledger": (_N1, "spend.efforts.N1.usd"),
    },
    "W-C1": {
        "claim": "The router lets the student answer only where its selective error stays within "
        "alpha, at the minimum coverage, without losing system accuracy.",
        "file": "router_eval_test",
        "pass": "W-C1.pass",
        "value": "selective error {err} at coverage {cov} (alpha {alpha}); system accuracy "
        "{sys} vs teacher-only {teacher}",
        "value_keys": {
            "err": "W-C1.selective_error",
            "cov": "W-C1.coverage",
            "alpha": "W-C1.alpha",
            "sys": "W-C1.system_accuracy",
            "teacher": "W-C1.teacher_only_accuracy",
        },
        "sealed": {"test_split": "test_split_sha256"},
        "prereg": ("prereg.commit", "prereg.committed_at"),
        "gate": {
            "n_test": "n_test",
            "alpha": "alpha",
            "delta": "delta",
            "lambda_hat": "lambda_hat",
            "coverage": "W-C1.coverage",
            "min_coverage": "W-C1.min_coverage",
            "selective_error": "W-C1.selective_error",
            "cp_upper_95": "W-C1.cp_upper_95",
            "system_accuracy": "W-C1.system_accuracy",
            "teacher_only_accuracy": "W-C1.teacher_only_accuracy",
            "tolerance": "W-C1.tolerance",
        },
    },
    "W-C2": {
        "claim": "The router defers most of the answers the student gets wrong, and fewer of the "
        "ones it gets right.",
        "file": "router_eval_test",
        "pass": "W-C2.pass",
        "value": "{deferred} of {errors} wrong answers deferred (share {share}, lower 95% "
        "bound {lower}, minimum {min}); {ok_def} of {ok_n} right answers deferred",
        "value_keys": {
            "deferred": "W-C2.n_error_deferred",
            "errors": "W-C2.n_error_items",
            "share": "W-C2.deferred_share",
            "lower": "W-C2.deferred_share_lower_95",
            "min": "W-C2.deferred_share_min",
            "ok_def": "W-C2.n_correct_deferred",
            "ok_n": "W-C2.n_correct_items",
        },
        "sealed": {"test_split": "test_split_sha256"},
        "prereg": ("prereg.commit", "prereg.committed_at"),
        "gate": {
            "n_error_items": "W-C2.n_error_items",
            "n_error_deferred": "W-C2.n_error_deferred",
            "deferred_share": "W-C2.deferred_share",
            "deferred_share_min": "W-C2.deferred_share_min",
            "deferred_share_lower_95": "W-C2.deferred_share_lower_95",
            "n_correct_items": "W-C2.n_correct_items",
            "n_correct_deferred": "W-C2.n_correct_deferred",
            "deferral_rate_on_correct": "W-C2.deferral_rate_on_correct",
        },
    },
    "W-D": {
        "claim": "A burst of off-distribution traffic trips the drift guard and rolls the gateway "
        "back to the teacher.",
        "file": "drift_burst",
        "pass": "verdict.pass",
        "value": "rolled back by {detector} after {latency} burst requests; {wrong} wrong "
        "student answers served during the burst, {after} after the rollback, {false} false "
        "alarms before the burst",
        "value_keys": {
            "detector": "burst.detector",
            "latency": "burst.detection_latency_requests",
            "wrong": "burst.wrong_student_answers_served_during_burst",
            "after": "burst.student_answers_served_after_rollback",
            "false": "burst.false_alarms_before_burst",
        },
        # the burst was scored under the pre-registration; its sealed hashes are the committed
        # burst questions, references and DB (h3_burst_counts.json copies them from the signals)
        "extras_from": "h3_burst_counts",
        "prereg": ("prereg.commit", "prereg.committed_at"),
        "sealed": {
            "burst_questions": "burst_sha256.burst_questions",
            "burst_reference": "burst_sha256.burst_reference",
            "burst_db": "burst_sha256.burst_db",
        },
        "adapter": "adapter_sha256",
        "gate": "params",
    },
    "W-$": {
        "claim": "The student's cost per 1k requests is measured on a real endpoint at its "
        "measured throughput.",
        "file": "h1_endpoint",
        "fail_if": "error",
        "value": "student USD {usd} per 1k requests (hourly price {hourly} USD at {rph} requests "
        "per hour, projected)",
        "value_keys": {
            "usd": "costs.student.usd_per_1k",
            "hourly": "costs.student.hourly_usd",
            "rph": "costs.student.req_per_hour",
        },
        "gate": {
            "hourly_usd": "costs.student.hourly_usd",
            "req_per_hour": "costs.student.req_per_hour",
            "utilization": "costs.student.utilization",
        },
        "ledger": ("h1_endpoint", "ledger.delta_usd"),
    },
    "W-E2E": {
        "claim": "The whole path runs end to end on the live gateway: certified router, student, "
        "deferral and the drift guard.",
        "file": "w_e2e",  # PLACEHOLDER path and keys: nothing writes this file yet
        "pass": "pass",
        "value": "{value}",
        "value_keys": {"value": "value_display"},
    },
}
W_G_ARTIFACT = "tests/test_acme_wg.py"
W_G_CLAIM = "An unchanged agent works against the gateway after one base-URL change."
W_G_VALUE = "same agent code, base URL swapped; test passes"
PENDING_ORDER = ("W-C1", "W-C2", "W-D", "W-$", "W-G", "W-E2E")
# A proof whose own verdict is PARTIAL met only part of its claim: it is "failed" here, never
# "verified" (the verdict itself stays in `verdict` so the page can say "partly met").
PARTIAL_STATUS = "failed"
_NO_EXTRAS: dict[str, Any] = {
    "prereg": None,
    "sealed_sha256": {},
    "adapter_sha256": None,
    "gate_inputs": None,
    "ledger_total_usd": None,
}


def _dig(doc: Any, path: str, where: str) -> Any:
    cur = doc
    for seg in path.split("."):
        try:
            cur = cur[int(seg)] if isinstance(cur, list) else cur[seg]
        except (KeyError, IndexError, TypeError, ValueError) as exc:
            raise KeyError(f"{where}: key path {path!r} is missing at {seg!r}") from exc
    return cur


def _shown(v: Any) -> str:
    """A value as text in value_display: numbers and strings as the evidence has them, except a
    computed ratio with a long binary tail (0.041666666666666664), shown to 4 significant digits;
    the exact value stays in gate_inputs."""
    if v is None:
        return "none"
    if isinstance(v, float) and len(str(v)) > 10:
        return f"{v:.4g}"
    return str(v)


def _read(root: Path | None, stem: str) -> dict[str, Any] | None:
    f = (root or REPO_ROOT) / EVIDENCE_DIR / f"{stem}.json"
    return _load(stem, root) if f.is_file() else None


def _extras(
    spec: dict[str, Any], doc: dict[str, Any], stem: str, root: Path | None
) -> dict[str, Any]:
    """prereg, sealed hashes, adapter hash, gate inputs and ledger total, read by KEYS paths."""
    where = f"{stem}.json"
    src = spec.get("extras_from")  # prereg, sealed and adapter recorded in a companion file
    xdoc, xwhere = (doc, where) if src is None else (_load(src, root), f"{src}.json")

    def dig(path: str) -> Any:
        return _dig(doc, path, where)

    def xdig(path: str) -> Any:
        return _dig(xdoc, path, xwhere)

    prereg, gate, ledger = spec.get("prereg"), spec.get("gate"), spec.get("ledger")
    usd = None
    if ledger is not None:
        ldoc = doc if ledger[0] == stem else _read(root, ledger[0])
        usd = None if ldoc is None else _dig(ldoc, ledger[1], f"{ledger[0]}.json")
    gate_inputs: dict[str, Any] | None = None
    if isinstance(gate, str):
        gate_inputs = dict(dig(gate))
    elif gate is not None:
        gate_inputs = {k: dig(v) for k, v in gate.items()}
    return {
        "prereg": None
        if prereg is None
        else {"commit": xdig(prereg[0]), "committed_at": xdig(prereg[1])},
        "sealed_sha256": {k: xdig(v) for k, v in spec.get("sealed", {}).items()},
        "adapter_sha256": xdig(spec["adapter"]) if "adapter" in spec else None,
        "gate_inputs": gate_inputs,
        "ledger_total_usd": usd,
    }


def _receipt(
    rid: str,
    claim: str,
    commit: str,
    path: str,
    status: str,
    verdict: str | None,
    value: str | None,
    extras: dict[str, Any] | None,
) -> dict[str, Any]:
    return {
        "id": rid,
        "claim": claim,
        "status": status,
        "verdict": verdict,
        "value_display": value,
        "artifact": path,
        "permalink": f"{REPO_URL}/blob/{commit}/{path}",
        **(extras or _NO_EXTRAS),
    }


def _proof_receipt(p: dict[str, Any], commit: str, root: Path | None) -> dict[str, Any]:
    spec = KEYS[p["id"]]
    stem = Path(p["evidence_path"]).stem
    doc = _load(stem, root)
    verdict = p["verdict"]
    status = {"PASS": "verified", "FAIL": "failed", "PARTIAL": PARTIAL_STATUS}[verdict]
    h = p["headline"]
    value = f"{h['label']}: {_shown(h['value'])}"
    if verdict != "PASS" and "reason" in spec:  # the evidence's own words for the rejection
        value = f"{_shown(h['value'])}: {_dig(doc, spec['reason'], f'{stem}.json')}"
    elif verdict != "PASS" and "also" in spec:
        m = next(m for m in p["metrics"] if m["label"] == spec["also"])
        value += f"; {m['label']}: {_shown(m['value'])}"
    return _receipt(
        p["id"], spec["claim"], commit, p["evidence_path"], status, verdict, value,
        _extras(spec, doc, stem, root),
    )  # fmt: skip


def _pending_receipt(rid: str, commit: str, root: Path | None) -> dict[str, Any]:
    spec = KEYS[rid]
    stem = spec["file"]
    path = f"{EVIDENCE_DIR}/{stem}.json"
    doc = _read(root, stem)
    if doc is None:  # not run yet: pre-registered, no number
        return _receipt(rid, spec["claim"], commit, path, "pending", None, None, None)
    where = f"{stem}.json"
    fill = {k: _shown(_dig(doc, v, where)) for k, v in spec["value_keys"].items()}
    failed = "fail_if" in spec and bool(_dig(doc, spec["fail_if"], where))
    if "pass" in spec:
        failed = failed or not _dig(doc, spec["pass"], where)
    return _receipt(
        rid, spec["claim"], commit, path, "failed" if failed else "verified",
        "FAIL" if failed else "PASS", spec["value"].format(**fill), _extras(spec, doc, stem, root),
    )  # fmt: skip


def build_receipts(
    commit: str, proofs: list[dict[str, Any]], root: Path | None = None
) -> list[dict[str, Any]]:
    """One receipt per existing proof (ladder order), then the pre-registered claims."""
    out = [_proof_receipt(p, commit, root) for p in proofs]
    for rid in PENDING_ORDER:
        if rid != "W-G":
            out.append(_pending_receipt(rid, commit, root))
        elif (root or REPO_ROOT).joinpath(W_G_ARTIFACT).is_file():
            out.append(
                _receipt(rid, W_G_CLAIM, commit, W_G_ARTIFACT, "verified", None, W_G_VALUE, None)
            )
        else:
            raise FileNotFoundError(W_G_ARTIFACT)
    return out


def build(commit: str, generated_at: str, root: Path | None = None) -> dict[str, Any]:
    proofs = [f(commit) for f in (p1_11, p1_12, p2, p3, p4, p5, a5, b2, gate_b, n1)]
    return {
        "generated_at": generated_at,
        "commit": commit,
        "repo_url": REPO_URL,
        "proofs": proofs,
        "receipts": build_receipts(commit, proofs, root),
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
