# ruff: noqa: E501  # long evidence strings in a proof harness
"""N1 evidence: the finished natural-phrasing round, as one JSON file (read-only, no API calls, zero spend).

Reads the run's report.json and report_human.json, the P4 report of the same run id (for the stress
comparison), the SQLite spend ledgers of every effort so far, the N1 question/train files (hashes only)
and this repository's git history, and writes docs/proofs/evidence/n1_natural_round.json. Every number in
the output is read from those files or derived from them here; the script holds no result values. No
question text, SQL or model output is copied (the reports' `examples` are not read).

Run (any python with the repo installed; the stores are opened read-only and immutable):
    .venv/bin/python docs/proofs/n1_evidence.py \
        [--home /home/lenovo/Documents/distillery/.distillery] \
        [--report PATH] [--report-human PATH] [--p4-report PATH] \
        [--out docs/proofs/evidence/n1_natural_round.json]

N1 reused P4's run id and a copy of P4's store, so every run-level total inside N1's report.json
(`cost.run_total_usd`, `finetune_usd`, `llm_by_model`) is P4 + N1. N1's own spend is therefore taken from the
ledgers by row set-difference against the P4 ledger (the P4 rows must be a subset, or the script stops).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sqlite3
import subprocess  # noqa: S404
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
DEFAULT_HOME = Path("/home/lenovo/Documents/distillery/.distillery")
RUN_ID = "sql-gated-20261004T155134Z-66ff347c"
DEFAULT_OUT = REPO / "docs/proofs/evidence/n1_natural_round.json"
GATEB_EVIDENCE = REPO / "docs/proofs/evidence/gateb_agent_authored.json"
PRICES_FILE = REPO / "deploy/prices.json"
DECISIONS = REPO / "DECISIONS.md"

# The pre-registration chain (DECISIONS.md 2026-10-07), in the order it was written.
PREREG_COMMITS = (
    ("bcd5f2d", "pre-registration: rule, claim, budget, before any N1 question exists"),
    ("14f03c3", "N1-train and N1-eval question files fixed (sha256) before drafting"),
    ("dfcf7ad", "N1-eval confirmed set (N, sha256) recorded before training"),
    ("dbaa917", "N1-train rows (kept count, sha256) recorded before the fine-tune"),
)

# (effort, [(store dir under --home, baseline store dir under --home or None)]). A baseline store is the
# P4 store a later effort was copied from; the effort's own spend is the rows beyond it.
P4_STORE = "p5live"
EFFORTS: tuple[tuple[str, tuple[tuple[str, str | None], ...]], ...] = (
    ("P4", ((P4_STORE, None),)),
    ("B2", (("b2/store", P4_STORE),)),
    ("Gate B", (("gateb", None), ("gateb_store", P4_STORE))),
    ("A5", (("a5live", None),)),
    ("N1", (("n1eval", None), ("n1train_root", None), ("n1live", P4_STORE))),
)

SPEND_COLS = "id, run_id, kind, model, usd, input_tokens, output_tokens, day, created_at"


class EvidenceError(RuntimeError):
    pass


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise EvidenceError(f"{path.name} must be a JSON object")
    return data


def rel(path: Path, home: Path) -> str:
    """Provenance string without the machine's absolute path."""
    try:
        return ".distillery/" + path.resolve().relative_to(home.resolve()).as_posix()
    except ValueError:
        return path.name


def source(path: Path, home: Path) -> dict[str, str]:
    return {"path": rel(path, home), "sha256": sha256_file(path)}


def utc_dt(iso: str) -> datetime:
    return datetime.fromisoformat(iso).astimezone(UTC)


def utc(iso: str) -> str:
    return utc_dt(iso).isoformat()


def _db(path: Path) -> sqlite3.Connection:
    # immutable: the stores are finished; this opens them without ever writing a journal or a -shm file
    return sqlite3.connect(f"file:{path}?mode=ro&immutable=1", uri=True)


def spend_rows(index: Path) -> list[tuple[Any, ...]]:
    con = _db(index)
    try:
        return con.execute(f"SELECT {SPEND_COLS} FROM spend ORDER BY id").fetchall()  # noqa: S608
    finally:
        con.close()


def spend_delta(
    rows: list[tuple[Any, ...]], baseline: list[tuple[Any, ...]] | None
) -> list[tuple[Any, ...]]:
    """The rows of a store beyond its baseline. The baseline must be a subset (a copied store)."""
    if baseline is None:
        return list(rows)
    have = set(rows)
    missing = [r for r in baseline if r not in have]
    if missing:
        raise EvidenceError(
            f"baseline ledger is not a subset of the store ({len(missing)} baseline rows missing, "
            f"first id {missing[0][0]}); refusing to compute a delta"
        )
    base = set(baseline)
    return [r for r in rows if r not in base]


def nano(row: tuple[Any, ...]) -> int:
    """A ledger row's usd as an integer count of 1e-9 dollars, so sums are exact."""
    return round(float(row[4]) * 1e9)


def usd(rows: list[tuple[Any, ...]]) -> float:
    return round(sum(nano(r) for r in rows) / 1e9, 6)


def by_kind(rows: list[tuple[Any, ...]]) -> dict[str, dict[str, float | int]]:
    kinds = sorted({str(r[2]) for r in rows})
    return {
        k: {"rows": len(sub), "usd": usd(sub)}
        for k in kinds
        for sub in [[r for r in rows if str(r[2]) == k]]
    }


def ledger_stores() -> list[str]:
    return sorted({store for _name, parts in EFFORTS for store, _base in parts})


def spend_block(home: Path) -> dict[str, Any]:
    ledgers: dict[str, list[tuple[Any, ...]]] = {}
    for _name, parts in EFFORTS:
        for store, _base in parts:
            if store not in ledgers:
                ledgers[store] = spend_rows(home / store / "index.sqlite")
    efforts: dict[str, Any] = {}
    deltas: dict[str, list[tuple[Any, ...]]] = {}
    for name, parts in EFFORTS:
        comps = []
        all_rows: list[tuple[Any, ...]] = []
        for store, base in parts:
            d = spend_delta(ledgers[store], ledgers[base] if base else None)
            deltas[store] = d
            all_rows += d
            comps.append(
                {
                    "ledger": f".distillery/{store}/index.sqlite",
                    "counted": "all rows"
                    if base is None
                    else f"rows beyond the {base} ledger (copied from it)",
                    "rows": len(d),
                    "usd": usd(d),
                    "by_kind": by_kind(d),
                }
            )
        efforts[name] = {
            "usd": usd(all_rows),
            "usd_nano": sum(nano(r) for r in all_rows),
            "by_kind": by_kind(all_rows),
            "components": comps,
        }
    total = round(sum(e["usd_nano"] for e in efforts.values()) / 1e9, 6)
    return {
        "efforts": efforts,
        "project_total_usd": total,
        "_deltas": deltas,
    }  # _deltas: dropped by caller


def decisions_numbers() -> dict[str, float]:
    """Constants that only exist as prose in DECISIONS.md (the N1 pre-registration)."""
    text = DECISIONS.read_text(encoding="utf-8")
    pat = {
        "n1_ceiling_usd": r"Hard ceiling \$(\d+\.\d+) of new spend",
        "project_cap_usd": r"recorded project spend to date \$\d+\.\d+ of \$(\d+\.\d+)",
        "prereg_ratio_lower_bound_min": r"lower bound of student/teacher accuracy is >= (\d+\.\d+)",
        "prereg_mcnemar_alpha": r"exact McNemar p < (\d+\.\d+)",
        "prereg_recorded_spend_before_n1_usd": r"recorded project spend to date \$(\d+\.\d+) of \$\d+\.\d+",
    }
    out: dict[str, float] = {}
    for key, rx in pat.items():
        m = re.search(rx, text)
        if not m:
            raise EvidenceError(f"cannot find {key!r} in DECISIONS.md (pattern {rx!r})")
        out[key] = float(m.group(1))
    return out


def git_commit(sha: str) -> dict[str, str]:
    try:
        res = subprocess.run(  # noqa: S603
            ["git", "log", "-1", "--format=%H%x09%cI%x09%s", sha],  # noqa: S607
            cwd=REPO, capture_output=True, text=True, check=True,
        )  # fmt: skip
    except (OSError, subprocess.CalledProcessError) as exc:
        raise EvidenceError(f"cannot read commit {sha} from git history: {exc}") from exc
    full, committed, subject = res.stdout.strip().split("\t", 2)
    return {
        "sha": full,
        "committed_at": committed,
        "committed_at_utc": utc(committed),
        "subject": subject,
    }


def gate_view(g: dict[str, Any]) -> dict[str, Any]:
    th = g["thresholds"]
    return {
        "n": g["n"],
        "accuracy": {
            "base": g["base_acc"],
            "student": g["student_acc"],
            "teacher": g["teacher_acc"],
        },
        "student_ci_95": g["student_ci"],
        "ratio": {"point": g["ratio_point"], "lo": g["ratio_lo"], "hi": g["ratio_hi"]},
        "mcnemar": {
            "p": g["mcnemar_p"],
            "student_only_vs_base": g["student_only_vs_base"],
            "base_only_vs_student": g["base_only_vs_student"],
        },
        "thresholds": th,
        "bootstrap_resamples_used": g["bootstrap_resamples_used"],
        "decision": g["decision"],
        "reasons": g["reasons"],
        "checks": {
            "ratio_lower_bound_ok": g["ratio_lo"] >= th["ratio_lower_bound_min"],
            "mcnemar_ok": g["mcnemar_p"] < th["mcnemar_alpha"],
        },
    }


def build(home: Path, report_p: Path, human_p: Path, p4_p: Path) -> dict[str, Any]:
    rep, hum, p4 = load_json(report_p), load_json(human_p), load_json(p4_p)
    if not (rep["run_id"] == hum["run_id"] == p4["run_id"]):
        raise EvidenceError("report, report_human and the P4 report disagree on the run id")
    ev, hev, p4ev = rep["evaluation"], rep["evaluation"]["human"], p4["evaluation"]
    gb = hum["gate_b"]
    if gb != hev:
        raise EvidenceError("report_human.json gate_b differs from report.json evaluation.human")
    art = ev["artifact"]
    nat = rep["natural_train"]
    ft = next(f for f in rep["finetune"] if f["job_id"] == art["job_id"])
    cnt, dec = rep["counters"], decisions_numbers()
    th = hev["gate"]["thresholds"]

    # ---- spend, by ledger row set-difference ------------------------------------------------------
    sp = spend_block(home)
    deltas: dict[str, list[tuple[Any, ...]]] = sp.pop("_deltas")
    n1_live = deltas["n1live"]
    teacher_calls = rep["cost"]["cost_per_1k_tasks"]["teacher"]["items_scored"]
    human_calls = hev["teacher_usage_this_command"]["calls"]
    n1_llm_rows = sum(1 for r in n1_live if r[2] == "llm")
    ft_rows = [r for r in n1_live if r[2] == "finetune"]
    sandbox_price = load_json(PRICES_FILE)["sandbox"]
    sp["n1_checks"] = {
        "n1live_new_llm_rows": n1_llm_rows,
        "expected_final_eval_teacher_calls_plus_human_scoring_calls": teacher_calls + human_calls,
        "scored_once": n1_llm_rows == teacher_calls + human_calls,
        "n1live_new_finetune_rows": len(ft_rows),
        "n1_finetune_units_equal_trained_tokens": [int(r[5]) for r in ft_rows]
        == [ft["trained_tokens"]],
        "n1_finetune_usd_equals_report_line": [round(nano(r) / 1e9, 6) for r in ft_rows]
        == [
            round(x["usd"], 6)
            for x in rep["cost"]["finetune_lines"]
            if x["units"] == ft["trained_tokens"]
        ],
    }
    n1_usd = sp["efforts"]["N1"]["usd"]
    sp.update(
        {
            "n1_ceiling_usd": dec["n1_ceiling_usd"],
            "n1_ceiling_source": "DECISIONS.md 2026-10-07 pre-registration ('Hard ceiling ... of new spend')",
            "n1_within_ceiling": n1_usd <= dec["n1_ceiling_usd"],
            "project_cap_usd": dec["project_cap_usd"],
            "project_cap_source": "DECISIONS.md 2026-10-07 pre-registration ('of $19.50')",
            "project_headroom_usd": round(dec["project_cap_usd"] - sp["project_total_usd"], 6),
            "spend_recorded_in_prereg_before_n1_usd": dec["prereg_recorded_spend_before_n1_usd"],
            "basis": "LLM rows: measured tokens x the API price table (estimates); fine-tune rows: trained tokens x the console price (billed basis); sandbox rows are recorded at $0 (no console line seen, price unconfirmed), so every total here is a lower bound.",
            "sandbox_price_record": {
                "usd_per_second": sandbox_price["usd_per_second"],
                "as_of": sandbox_price["as_of"],
                "source": sandbox_price["source"],
            },
            "cumulative_fields_warning": f"report.json cost.run_total_usd ({rep['cost']['run_total_usd']:.4f}) and run-level spend in the replay bundle are P4 + N1 (N1 reused P4's run id and store copy); N1's own spend is efforts.N1.usd.",
            "report_cost_p4_plus_n1_usd": rep["cost"]["run_total_usd"],
        }
    )

    # ---- pre-registration chain vs the first fine-tune/score events --------------------------------
    commits = [{**git_commit(sha), "what": what} for sha, what in PREREG_COMMITS]
    con = _db(home / "n1live" / "index.sqlite")
    try:
        started = con.execute(
            "SELECT created_at FROM experiments WHERE name='finetune_job_started' AND data_json LIKE ? ORDER BY id",
            (f'%"{art["job_id"]}"%',),
        ).fetchall()
    finally:
        con.close()
    if len(started) != 1:
        raise EvidenceError(
            f"expected exactly one finetune_job_started for {art['job_id']}, found {len(started)}"
        )
    ft_started_dt = utc_dt(started[0][0])
    scored_dt = utc_dt(hev["scored_at"])
    ft_started, scored_at = ft_started_dt.isoformat(), scored_dt.isoformat()
    first_rows_dt = {
        store: min(utc_dt(r[8]) for r in spend_rows(home / store / "index.sqlite"))
        for store in ("n1eval", "n1train_root")
    }
    commit_dt = [utc_dt(c["committed_at"]) for c in commits]
    prereg = {
        "source": "DECISIONS.md 2026-10-07: pre-registration of round N1, then three amendments (question files, N1-eval confirmed set, N1-train rows)",
        "commits": commits,
        "finetune_job_started_at_utc": ft_started,
        "n1_eval_scored_at_utc": scored_at,
        "first_ledger_row_utc": {
            "n1eval_ledger (drafting of N1-eval)": first_rows_dt["n1eval"].isoformat(),
            "n1train_root_ledger (drafting of N1-train)": first_rows_dt["n1train_root"].isoformat(),
        },
        "order_checks": {
            "pre_registration_commit_before_first_n1_ledger_row": commit_dt[0]
            < min(first_rows_dt.values()),
            "all_commits_before_finetune_job_started": all(c < ft_started_dt for c in commit_dt),
            "all_commits_before_n1_eval_scored": all(c < scored_dt for c in commit_dt),
            "n1_eval_confirmed_set_commit_before_finetune_job_started": commit_dt[2]
            < ft_started_dt,
        },
        "claim_rule_unchanged": {
            "thresholds_in_report": th,
            "pre_registered_ratio_lower_bound_min": dec["prereg_ratio_lower_bound_min"],
            "pre_registered_mcnemar_alpha": dec["prereg_mcnemar_alpha"],
            "report_matches_pre_registration": th["ratio_lower_bound_min"]
            == dec["prereg_ratio_lower_bound_min"]
            and th["mcnemar_alpha"] == dec["prereg_mcnemar_alpha"],
            "gate_a_decision_unchanged": hum["gate_a_decision_unchanged"],
        },
    }

    # ---- files that fix the question/training sets -------------------------------------------------
    nt_file = home / "natural_train.json"
    nt = load_json(nt_file)
    files = {
        "natural_train.json": {
            **source(nt_file, home),
            "matches_report": sha256_file(nt_file) == nat["file_sha256"],
        },
        "natural/train_questions.txt": {
            **source(home / "natural/train_questions.txt", home),
            "matches_report": sha256_file(home / "natural/train_questions.txt")
            == nat["questions_sha256"],
        },
        "natural_eval/questions.txt": {
            **source(home / "natural_eval/questions.txt", home),
            "matches_report": sha256_file(home / "natural_eval/questions.txt")
            == hev["question_file_sha256"],
        },
    }
    if not all(f["matches_report"] for f in files.values()):
        raise EvidenceError(
            f"a question/training file does not match the hash in the report: {files}"
        )

    # ---- primary (pre-registered) ------------------------------------------------------------------
    pc = hev["counts"]
    primary = {
        "claim": f"PROMOTE on N1-eval only if the paired-bootstrap lower bound of student/teacher accuracy is >= {th['ratio_lower_bound_min']} AND the student beats base by exact McNemar p < {th['mcnemar_alpha']} (the existing gate, unchanged; DECISIONS.md 2026-10-07)",
        "set": {
            "name": "N1-eval",
            "label": "agent-authored (Claude subagent, blind to templates and to N1-train by instruction); NOT human-written",
            "confirmation": "mechanical: a teacher draft is confirmed iff its result equals the result of the author's blind reference SQL (DECISIONS.md 2026-10-07, N1-eval confirmed set); no confirmer discretion",
            "n": hev["n"],
            "questions_drafted": pc["questions"],
            "kept_after_drafting": pc["kept"],
            "confirmed": pc["confirmed"],
            "rejected_at_confirmation": pc["rejected"],
            "discarded_at_drafting": pc["discarded"],
            "discarded_by_reason": pc["discarded_by_reason"],
            "confirmed_file_sha256": hev["file_sha256"],
            "question_file_sha256": hev["question_file_sha256"],
            "sealed_set_sha256": hev["sha256"],
            "db_sha256": hev["db_sha256"],
            "dropped_exact_overlap": hev["dropped_exact_overlap"],
            "skeleton_in_train": hev["skeleton_in_train"],
            "sealed_before_training": prereg["order_checks"][
                "n1_eval_confirmed_set_commit_before_finetune_job_started"
            ],
            "generation_errors": hev["generation_errors"],
            "unparseable": hev["unparseable"],
        },
        "scored_at": hev["scored_at"],
        "scored_at_utc": scored_at,
        "teacher_model": hev["teacher_model"],
        **gate_view(hev["gate"]),
        "decision_human": rep["decision_human"],
        "spend_usd_scoring_command": hev["spend_usd_this_command"],
    }
    if primary["decision"] != rep["decision_human"]:
        raise EvidenceError("gate decision and decision_human disagree")

    # ---- secondary (reported, not the primary claim) -----------------------------------------------
    hg = ev["gate"]
    p4g = p4ev["gate"]
    ho_min = hg["thresholds"]["ratio_lower_bound_min"]
    heldout = {
        "role": "secondary, reported only: the original templated held-out must not regress (ratio lower bound >= the gate minimum)",
        "label": rep["data"]["heldout_kind"],
        "sealed_sha256": ev["heldout_sha256"],
        "same_set_as_p4": ev["heldout_sha256"] == p4ev["heldout_sha256"],
        "skeleton_overlap_with_train": rep["data"]["heldout_skeleton_overlap_rate"],
        **gate_view(hg),
        "pipeline_gate_decision": rep["decision"],
        "no_regression": {
            "ratio_lower_bound": hg["ratio_lo"],
            "minimum": ho_min,
            "passed": hg["ratio_lo"] >= ho_min,
        },
        "p4_same_set": {
            "student_accuracy": p4ev["accuracy"]["student"],
            "teacher_accuracy": p4ev["accuracy"]["teacher"],
            "base_accuracy": p4ev["accuracy"]["base"],
            "ratio_lo": p4g["ratio_lo"],
            "source": rel(p4_p, home),
        },
    }
    st, p4st = ev["stress"], p4ev["stress"]
    fams = sorted(st["accuracy_by_family"])
    stress = {
        "role": "reported only, never a gate input",
        "n": st["n"],
        "families": st["families"],
        "sealed_sha256": st["sha256"],
        "same_set_as_p4": st["sha256"] == p4st["sha256"],
        "accuracy": st["accuracy"],
        "accuracy_by_family": st["accuracy_by_family"],
        "unparseable": st["unparseable"],
        "p4_adapter_same_set": {
            "accuracy": p4st["accuracy"],
            "accuracy_by_family": p4st["accuracy_by_family"],
            "source": rel(p4_p, home),
        },
        "student_change_vs_p4": {
            "overall": round(st["accuracy"]["student"] - p4st["accuracy"]["student"], 6),
            **{
                f: round(
                    st["accuracy_by_family"][f]["student"]
                    - p4st["accuracy_by_family"][f]["student"],
                    6,
                )
                for f in fams
            },
        },
        "exact_question_overlap": {
            "natural_rows_dropped_for_equalling_a_stress_question": nat["dropped_by_overlap"][
                "stress"
            ],
            "natural_rows_dropped_for_equalling_any_sealed_question": nat["dropped_by_overlap"][
                "total"
            ],
            "concept_overlap_checked": False,
        },
        "reading_caveat": "the natural training questions can teach date and set concepts, so these families are no longer unseen concepts for N1; exact question overlap was checked, concept overlap was not",
    }
    gbe = load_json(GATEB_EVIDENCE)
    gate_b_p4 = {
        "adapter_sha256": gbe["adapter_sha256"],
        "set": "Gate B agent-authored set (docs/proofs/evidence/gateb_agent_authored.json)",
        "n": gbe["n"],
        "accuracy": gbe["accuracy"],
        "ratio_lo": gbe["gate"]["ratio_lo"],
        "mcnemar_p": gbe["gate"]["mcnemar_p"],
        "decision": gbe["decision"],
        "question_file_sha256": gbe["question_file_sha256"],
        "single_ledger_pass_usd": gbe["spend_usd_this_command"],
    }
    gate_b_rescore = {
        "status": "pending",
        "what": "re-score the N1 adapter on the Gate B set as a post-hoc transfer number (pre-registered as secondary, not a gate)",
        "reason": "needs a small paid run (teacher + sandbox) that has not been run or approved; no number exists",
        "reference_p4_adapter_on_gate_b": gate_b_p4,
        "n1_adapter_sha256": art["adapter_sha256"],
        "expected_cost_basis": "one earlier Gate B scoring pass (single_ledger_pass_usd above), teacher calls only",
    }

    # ---- artifact and training data ----------------------------------------------------------------
    artifact = {
        "adapter_sha256": art["adapter_sha256"],
        "job_id": art["job_id"],
        "checkpoint_id": art["checkpoint_id"],
        "trained_base_model": art["trained_base_model"],
        "trained_checkpoint_name": art["trained_checkpoint_name"],
        "differs_from_p4_adapter": art["adapter_sha256"] != p4ev["artifact"]["adapter_sha256"],
        "p4_adapter_sha256": p4ev["artifact"]["adapter_sha256"],
        "finetune": {
            "trained_tokens": ft["trained_tokens"],
            "trained_steps": ft["trained_steps"],
            "total_steps": ft["total_steps"],
            "hyperparameters": ft["hyperparameters"],
            "hyperparameters_equal_p4": ft["hyperparameters"]
            == p4["finetune"][0]["hyperparameters"],
            "loss_curve": ft["loss_curve"],
            "usd": usd(ft_rows),
            "usd_basis": "trained tokens x console price (billed basis)",
            "started_at_utc": ft_started,
        },
        "dev": {
            "base_accuracy": rep["headroom"]["base_dev_acc"],
            "student_accuracy": rep["rounds"][0]["dev_acc"],
            "n": cnt["dev_eval_r1"]["dev_n"],
            "failures": cnt["dev_eval_r1"]["dev_failures"],
        },
        "candidate_round": rep["candidate_round"],
        "rounds_stop_reason": rep["rounds_stop_reason"],
    }
    p4_rows = rep["paraphrase"]["final_train_rows"]
    training_data = {
        "train_rows_total": cnt["finetune_r1"]["train_rows"],
        "p4_rows_template_originals_plus_verified_paraphrases": p4_rows,
        "natural_rows_appended": nat["rows_appended"],
        "rows_add_up": p4_rows + nat["rows_appended"] == cnt["finetune_r1"]["train_rows"],
        "p4_rows_originals": rep["paraphrase"]["originals"],
        "p4_rows_paraphrases_kept": rep["paraphrase"]["rows_added"],
        "row_cap_reapplied_to_natural": nat["row_cap_applied"],
        "natural": {
            "label": nat["label"],
            "questions_drafted": nt["n_questions"],
            "teacher_drafts_per_question": nt["k"],
            "teacher_temperature": nt["temperature"],
            "teacher_model": nt["teacher_model"],
            "kept": len(nt["kept"]),
            "discarded": len(nt["discarded"]),
            "discarded_by_reason": nt["discards"],
            "file_sha256": nat["file_sha256"],
            "questions_sha256": nat["questions_sha256"],
            "reference_sha256": nat["reference_sha256"],
            "db_sha256": nat["db_sha256"],
            "rule": "a row is kept only if all k teacher drafts execute to the same non-empty result AND that result equals the author's blind reference SQL result",
            "dropped_by_overlap_with_sealed_sets": nat["dropped_by_overlap"],
            "heldout_skeleton_in_natural": nat["heldout_skeleton_in_natural"],
            "exclude_files_question_count": nat["exclude_questions"],
        },
        "identity_files": files,
    }

    disclosures = [
        "N1-eval and N1-train are agent-authored (Claude subagents), never human-written; the two authors share a model family, so a gap against a human-written set may remain.",
        "Gold SQL on N1-eval was drafted by the teacher and kept only where every draft executed, was non-empty and agreed with the author's reference: teacher accuracy on this set is inflated by construction, so the base and student numbers are the informative ones, and the same teacher is both the gold drafter and the compared model.",
        "The primary result is on a different set from Gate B (different questions, different author, different confirmation rule). N1's improvement over the P4 adapter is therefore not a measured improvement; only the Gate B re-score (pending) would put both adapters on one set.",
        "The held-out and stress sets are the same sealed sets P4 used, already seen by the P4 decision and by B2; they are regression checks here, not fresh evidence.",
        f"Stress families: natural training rows dropped for exactly matching a stress question: {nat['dropped_by_overlap']['stress']}; concept overlap was not checked, and the natural questions can teach date and set concepts. The stress change versus P4 must not be read as unseen-concept generalization.",
        f"One run, one seed, one fine-tune; the intervals are bootstrap intervals over the {hev['n']} N1-eval items, not run-to-run variance.",
        "Spend is a lower bound: the sandbox is recorded at $0 (price unconfirmed) and LLM lines are token-count estimates at API prices. N1's run-level fields in report.json (and in the replay bundle's run.json) are P4 + N1 cumulative because N1 reused P4's run id and store copy.",
        "The report's generic note on the human-set block ('confirmed by a human') does not apply here: confirmation of N1-eval was mechanical.",
    ]
    pending = [
        {
            "item": "Gate B re-score of the N1 adapter (post-hoc transfer number)",
            "status": gate_b_rescore["status"],
            "see": "secondary.gate_b_rescore",
        }
    ]

    return {
        "what": "Round N1: training on naturally phrased, teacher-verified questions, scored once on a fresh agent-authored set (DECISIONS.md 2026-10-07 pre-registration; result entry 2026-10-09)",
        "label": "AGENT-AUTHORED set (Claude subagents), NOT human-written",
        "run_id": rep["run_id"],
        "run_id_note": "N1 reused P4's run id on a copy of P4's store; the replay bundle is exported under its own bundle id",
        "sources": {
            "report": source(report_p, home),
            "report_human": source(human_p, home),
            "p4_report": source(p4_p, home),
            "ledgers": {st: source(home / st / "index.sqlite", home) for st in ledger_stores()},
            "decisions": "DECISIONS.md",
            "note": "paths are relative to the repository checkout that holds the gitignored .distillery/ directory; no sealed text is copied into this file",
        },
        "decision": {
            "primary": primary["decision"],
            "gate_a_pipeline_decision_unchanged": hum["gate_a_decision_unchanged"],
            "no_threshold_changed": prereg["claim_rule_unchanged"][
                "report_matches_pre_registration"
            ],
        },
        "primary": primary,
        "secondary": {"heldout": heldout, "stress": stress, "gate_b_rescore": gate_b_rescore},
        "artifact": artifact,
        "training_data": training_data,
        "spend": sp,
        "preregistration": prereg,
        "pending": pending,
        "disclosures": disclosures,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--home",
        type=Path,
        default=DEFAULT_HOME,
        help="the .distillery directory of the main checkout",
    )
    ap.add_argument(
        "--report",
        type=Path,
        default=None,
        help="N1 report.json (default: <home>/n1live/runs/<run>/report.json)",
    )
    ap.add_argument(
        "--report-human",
        type=Path,
        default=None,
        help="N1 report_human.json (default: next to --report)",
    )
    ap.add_argument(
        "--p4-report",
        type=Path,
        default=None,
        help="P4 report.json (default: <home>/p5live/runs/<run>/report.json)",
    )
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    a = ap.parse_args(argv)
    report = a.report or a.home / "n1live" / "runs" / RUN_ID / "report.json"
    human = a.report_human or report.with_name("report_human.json")
    p4 = a.p4_report or a.home / P4_STORE / "runs" / RUN_ID / "report.json"
    try:
        out = build(a.home, report, human, p4)
    except (EvidenceError, OSError, KeyError, sqlite3.Error) as exc:
        print(f"error: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    text = json.dumps(out, indent=1, ensure_ascii=False) + "\n"
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(text, encoding="utf-8")
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
