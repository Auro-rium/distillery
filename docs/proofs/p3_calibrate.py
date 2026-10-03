"""P3 evidence (free, local): verifier and gate calibration with the real gate code and config.

Run:  .venv/bin/python docs/proofs/p3_calibrate.py [--sims 200] [--workers 3]
Prints JSON. Uses the gated run's generated tasks (local SQLite, no LLM), the recorded per-item
vectors in docs/proofs/evidence/diag_*.json, and evaluate_gate with default GateThresholds.
"""

import json
import random
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from distillery.config import GateThresholds
from distillery.diagnostics import stage_artifact
from distillery.gate import evaluate_gate
from distillery.store import Store
from distillery.taskpacks.sql.runner import run_select
from distillery.taskpacks.sql.sqltext import has_top_level_order_by
from distillery.taskpacks.sql.verifier import compare_outcomes, corrupt_sql, execution_match

ROOT = Path(".distillery/live")
DB = ROOT / "runs" / "gated-1p7b-r1" / "db.sqlite"
CFG = GateThresholds()
arg = lambda k, d: int(sys.argv[sys.argv.index(k) + 1]) if k in sys.argv else d  # noqa: E731

# Hand-written pairs on the real schema: (name, gold, candidate, expected_match).
PAIRS = [
    ("join_order_and_aliases",
     "SELECT a.name FROM accounts a JOIN users u ON u.account_id = a.account_id WHERE u.role = 'owner'",
     "SELECT accounts.name FROM users JOIN accounts ON accounts.account_id = users.account_id "
     "WHERE users.role = 'owner'", True),
    ("in_vs_exists",
     "SELECT account_id FROM accounts WHERE account_id IN (SELECT account_id FROM users WHERE role = 'admin')",
     "SELECT a.account_id FROM accounts a WHERE EXISTS "
     "(SELECT 1 FROM users u WHERE u.account_id = a.account_id AND u.role = 'admin')", True),
    ("not_exists_vs_left_join_null",
     "SELECT a.account_id FROM accounts a WHERE NOT EXISTS (SELECT 1 FROM users u WHERE u.account_id = a.account_id)",
     "SELECT a.account_id FROM accounts a LEFT JOIN users u ON u.account_id = a.account_id "
     "WHERE u.user_id IS NULL", True),
    ("between_vs_range",
     "SELECT account_id FROM accounts WHERE created_at BETWEEN '2024-01-01' AND '2024-06-30'",
     "SELECT account_id FROM accounts WHERE created_at >= '2024-01-01' AND created_at <= '2024-06-30'", True),
    ("count_star_vs_count_pk_and_column_aliases",
     "SELECT country, COUNT(*) FROM accounts GROUP BY country",
     "SELECT country AS c, COUNT(account_id) AS n FROM accounts GROUP BY 1", True),
    ("cte_wrapper",
     "SELECT industry, COUNT(*) FROM accounts WHERE status = 'active' GROUP BY industry",
     "WITH x AS (SELECT * FROM accounts WHERE status = 'active') SELECT industry, COUNT(*) FROM x GROUP BY industry",
     True),
    ("column_order_swapped (positional by design: expected REJECT)",
     "SELECT country, COUNT(*) FROM accounts GROUP BY country",
     "SELECT COUNT(*), country FROM accounts GROUP BY country", False),
    ("order_by_desc_vs_asc on ordered gold (expected REJECT)",
     "SELECT account_id FROM accounts ORDER BY account_id LIMIT 5",
     "SELECT account_id FROM accounts ORDER BY account_id DESC LIMIT 5", False),
]


def verifier_checks(tasks: list[dict]) -> dict:
    gold_ok = gold_total = cands = rejected = wrap_total = wrap_ok = 0
    rng = random.Random(1234)
    for t in tasks:
        g = run_select(DB, t["gold_sql"])
        gold_total += 1
        gold_ok += compare_outcomes(g, g, t["requires_order"]).ok
        for c in corrupt_sql(t["gold_sql"], DB, rng, max_variants=4):
            cands += 1
            rejected += not execution_match(DB, c.sql, t["gold_sql"], t["requires_order"]).ok
        if not has_top_level_order_by(t["gold_sql"]):  # subquery wrap is equivalent only unordered
            wrap_total += 1
            body = t["gold_sql"].strip().rstrip(";")
            wrap_ok += execution_match(DB, f"SELECT * FROM ({body})", t["gold_sql"], False).ok
    pairs = []
    for name, gold, cand, expected in PAIRS:
        r = execution_match(DB, cand, gold, has_top_level_order_by(gold))
        pairs.append({"pair": name, "expected_match": expected, "got_match": r.ok,
                      "as_expected": r.ok == expected, "reason": r.reason})
    return {"gold_vs_gold": f"{gold_ok}/{gold_total}",
            "corruptions_rejected": f"{rejected}/{cands} (corrupt_sql keeps only variants the "
                                    "verifier rejects, so this is a consistency check, not recall)",
            "subquery_wrap_equivalent_accepted": f"{wrap_ok}/{wrap_total}",
            "handwritten_pairs": pairs}


def gate(base, student, teacher) -> dict:
    r = evaluate_gate(base, student, teacher, CFG)
    return {"decision": r.decision, "n": r.n, "acc": [r.base_acc, r.student_acc, r.teacher_acc],
            "ratio_ci": [r.ratio_lo, r.ratio_hi], "mcnemar_p": r.mcnemar_p, "reasons": r.reasons}


def recorded_gates() -> dict:
    d = json.loads(Path(".distillery/proofs/diag_sql-mini-live1_2026-10-01.json").read_text())
    vec = lambda m, s: {i["task_id"]: bool(i["correct"]) for i in d["models"][m][s]["items"]}  # noqa: E731
    out = {}
    # Held-out n=60: the run's teacher scored 60/60 (report.json), so its vector is all True.
    base, r1 = vec("base", "heldout"), vec("round1", "heldout")
    ids = sorted(base)
    t = [True] * len(ids)
    b = [base[i] for i in ids]
    out["heldout60_teacher_as_student"] = gate(b, t, t)
    out["heldout60_base_as_student"] = gate(b, b, t)
    out["heldout60_round1_student (the real REJECT)"] = gate(b, [r1[i] for i in ids], t)
    # Train rows n=117: teacher correct on every row by construction (rows kept only on match).
    bt, st = vec("base", "train[round1]"), vec("round1", "train")
    ids = sorted(bt)
    t = [True] * len(ids)
    out["train117_teacher_as_student"] = gate([bt[i] for i in ids], t, t)
    out["train117_base_as_student"] = gate([bt[i] for i in ids], [bt[i] for i in ids], t)
    # A/A: base scored twice on the same 60 held-out items (run final_eval vs the diagnosis job).
    out["aa_base_heldout_counts"] = {"run_final_eval_base_acc": 2 / 60,
                                     "diagnosis_base_acc": sum(b) / len(b)}
    return out


def one_sim(args) -> tuple[float, int, bool, float]:
    ratio, n, pt, pb, seed = args
    rng = random.Random(seed)
    teacher = [rng.random() < pt for _ in range(n)]
    student = [tc and rng.random() < ratio for tc in teacher]          # true ratio = `ratio`
    base = [rng.random() < pb for _ in range(n)]
    return ratio, n, evaluate_gate(base, student, teacher, CFG).decision == "PROMOTE", pb


def power(sims: int, workers: int) -> dict:
    jobs = [(r, n, 0.97, pb, 10_000 * k + i)
            for k, (r, n, pb) in enumerate((r, n, pb) for n in (300, 100) for pb in (0.15,)
                                           for r in (0.80, 0.85, 0.90, 0.95))
            for i in range(sims)]
    hits: dict[str, list[int]] = {}
    with ProcessPoolExecutor(workers) as ex:
        for r, n, ok, pb in ex.map(one_sim, jobs, chunksize=4):
            hits.setdefault(f"n={n} true_ratio={r}", [0, 0])
            hits[f"n={n} true_ratio={r}"][0] += ok
            hits[f"n={n} true_ratio={r}"][1] += 1
    return {"assumptions": "teacher acc 0.97, base acc 0.15 (independent), student correct only where "
                           "teacher is, with prob = true ratio; real GateThresholds (10k resamples)",
            "promote_rate": {k: round(a / b, 3) for k, (a, b) in hits.items()}, "sims_per_cell": sims}


if __name__ == "__main__":
    q = stage_artifact(Store(ROOT), "gated-1p7b-r1", "questions")
    out = {"P3.1_verifier": verifier_checks(q["tasks"] + q["stress_tasks"]),
           "P3.2-4_recorded": recorded_gates(),
           "P3.5_power": power(arg("--sims", 200), arg("--workers", 3))}
    print(json.dumps(out, indent=1))
