# ruff: noqa: E501, S311, S608, S101  (evidence script: literal SQL and seeded simulation)
"""P2 evidence (free, local, no LLM/network): training-data labels and sealed-set leakage.

Run:  .venv/bin/python docs/proofs/p2_data.py > docs/proofs/evidence/p2_data.json
Prints JSON (counts and rates only; sealed question/SQL text is never printed or written).
Writes a human spot-check of 30 TRAINING rows to .distillery/proofs/p2_spotcheck.md (gitignored).
Acceptance check used by the orchestrator (_teacher_rows): compare_outcomes(candidate, gold, requires_order).ok
"""

import difflib
import json
import random
import re
import shutil
import sqlite3
from collections import Counter
from pathlib import Path

from distillery.diagnostics import stage_artifact
from distillery.store import Store
from distillery.taskpacks.sql.runner import run_select
from distillery.taskpacks.sql.verifier import compare_outcomes, corrupt_sql

ROOT = Path(".distillery/live")
RUNS = ["sql-tiny-live1", "sql-mini-live1", "gated-1p7b-r1"]
DB = ROOT / "runs" / "gated-1p7b-r1" / "db.sqlite"
TMP = Path("/home/lenovo/.claude/jobs/958b3c06/tmp")
SEED = 1234
store = Store(ROOT)
QRE = re.compile(r"Question: (.*?)\n\n/no_think", re.S)


def norm(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip().rstrip(";")).lower()


def skel(s: str) -> str:
    s = re.sub(r"'(?:[^']|'')*'|\"(?:[^\"]|\"\")*\"", "?", s)
    s = re.sub(r"\b\d+(?:\.\d+)?\b", "?", s)
    return norm(s)


def same_db(db_hash_paths: list[Path]) -> bool:
    return len({p.read_bytes() for p in db_hash_paths}) == 1


def load_pool() -> tuple[list[dict], dict]:
    """Trained rows (teacher SQL) per smoke round, matched to template gold by question text."""
    pool, notes = [], {}
    for run in RUNS[:2]:
        q = stage_artifact(store, run, "questions")["tasks"]
        by_q: dict[str, list[dict]] = {}
        for t in q:
            by_q.setdefault(t["question"], []).append(t)
        for rd in sorted((ROOT / "runs" / run).glob("round*/train.jsonl")):
            n = unmatched = 0
            for line in rd.read_text().splitlines():
                m = json.loads(line)["messages"]
                mm = QRE.search(m[1]["content"])
                n += 1
                cands = by_q.get(mm.group(1).strip()) if mm else None
                if not cands:
                    unmatched += 1
                    continue
                pool.append({"run": run, "round": rd.parent.name, "question": cands[0]["question"],
                             "sql": m[2]["content"], "golds": [c["gold_sql"] for c in cands],
                             "family": cands[0]["family"], "requires_order": cands[0]["requires_order"]})
            notes[f"{run}/{rd.parent.name}"] = {"rows": n, "unmatched_to_template_task": unmatched}
    return pool, notes


def accepted(db, sql: str, gold_outcome, order: bool) -> bool:
    return compare_outcomes(run_select(db, sql), gold_outcome, order).ok


def label_check(db, pool) -> list[bool]:
    return [any(accepted(db, r["sql"], run_select(db, g), r["requires_order"]) for g in r["golds"]) for r in pool]


def perturb(src: Path, dst: Path, seed: int = SEED) -> dict:
    shutil.copy(src, dst)
    rng = random.Random(seed)
    con = sqlite3.connect(dst)
    cols = 0
    for (t,) in con.execute("select name from sqlite_master where type='table'").fetchall():
        info = con.execute(f'pragma table_info("{t}")').fetchall()
        pk = {c[1] for c in info if c[5]}
        for c in info:
            name = c[1]
            if name in pk or name.endswith("_id"):
                continue
            rowids = [r[0] for r in con.execute(f'select rowid from "{t}" order by rowid')]
            vals = [r[0] for r in con.execute(f'select "{name}" from "{t}" order by rowid')]
            rng.shuffle(vals)
            con.executemany(f'update "{t}" set "{name}"=? where rowid=?', zip(vals, rowids, strict=True))
            cols += 1
    con.commit()
    con.close()
    return {"shuffled_columns": cols}


MUT = [("flip_gt_to_lt", r"(?<![<>!=])>(?!=)", "<"), ("flip_lt_to_gt", r"(?<![<>!=])<(?![>=])", ">"),
       ("eq_to_neq", r"(?<![<>!])=(?!=)", "!="),
       ("and_to_or", r"\bAND\b", "OR"), ("desc_to_asc", r"\bDESC\b", "ASC"), ("drop_distinct", r"\bDISTINCT\b ?", ""),
       ("limit_plus_1", r"\bLIMIT (\d+)", lambda m: f"LIMIT {int(m.group(1)) + 1}"),
       ("number_plus_1", r"\b(\d+)\b", lambda m: str(int(m.group(1)) + 1))]


def own_mutant(sql: str, rng: random.Random, db) -> tuple[str, str] | None:
    order = list(MUT)
    rng.shuffle(order)
    for name, pat, rep in order:
        new, k = re.subn(pat, rep, sql, count=1, flags=re.I)
        if k and new != sql:
            return name, new
    return None


def sql_diff(a: str, b: str) -> str:
    ta, tb = a.split(), b.split()
    sm = difflib.SequenceMatcher(None, ta, tb)
    return "; ".join(f"{' '.join(ta[i1:i2])!r} -> {' '.join(tb[j1:j2])!r}" for t, i1, i2, j1, j2 in sm.get_opcodes() if t != "equal")


def recall(db, pool, pdbs: list[Path]) -> dict:
    rng = random.Random(SEED)
    sample = rng.sample(pool, max(1, len(pool) // 10))
    out = {"sampled": len(sample), "corrupt_sql": Counter(), "own_mutations": Counter()}
    acc_list, equiv = [], 0
    for r in sample:
        g = run_select(db, r["golds"][0])
        c = corrupt_sql(r["sql"], db, rng, max_variants=1)
        if c:
            out["corrupt_sql"]["injected"] += 1
            out["corrupt_sql"]["rejected"] += not accepted(db, c[0].sql, g, r["requires_order"])
        else:
            out["corrupt_sql"]["no_variant_available"] += 1
        mm = own_mutant(r["sql"], rng, db)
        m = mm[1] if mm else None
        if m is None:
            out["own_mutations"]["no_mutation_applicable"] += 1
            continue
        o = run_select(db, m)
        out["own_mutations"]["injected"] += 1
        if not o.ok:
            out["own_mutations"]["rejected_exec_error"] += 1
        elif compare_outcomes(o, g, r["requires_order"]).ok:
            out["own_mutations"]["accepted_wrongly_or_equivalent"] += 1
            robust = all(accepted(d, m, run_select(d, r["golds"][0]), r["requires_order"]) for d in pdbs)
            equiv += robust
            acc_list.append({"mutator": mm[0], "family": r.get("family"),
                             "class": "equivalent_robust" if robust else "true_miss",
                             "gold_sql": r["golds"][0], "mutant_sql": m, "diff": sql_diff(r["golds"][0], m)})
        else:
            out["own_mutations"]["rejected_result_mismatch"] += 1
    for k in ("corrupt_sql", "own_mutations"):
        d = dict(out[k])
        inj = d.get("injected", 0)
        rej = d.get("rejected", 0) if k == "corrupt_sql" else inj - d.get("accepted_wrongly_or_equivalent", 0)
        d["recall"] = f"{rej}/{inj}"
        d["recall_rate"] = round(rej / inj, 4) if inj else None
        out[k] = d
    om = out["own_mutations"]
    inj, acc = om.get("injected", 0), om.get("accepted_wrongly_or_equivalent", 0)
    om["accepted_equivalent_robust"] = equiv
    om["accepted_true_miss"] = acc - equiv
    om["recall_excluding_equivalent_robust"] = f"{inj - acc}/{inj - equiv}"
    om["recall_excluding_equivalent_robust_rate"] = round((inj - acc) / (inj - equiv), 4) if inj - equiv else None
    om["accepted_mutants"] = acc_list
    return out


def lost_rows(pool, ok, okp, pdb) -> list[dict]:
    res = []
    for r, a, b in zip(pool, ok, okp, strict=True):
        if not (a and not b):
            continue
        g, t = run_select(pdb, r["golds"][0]), run_select(pdb, r["sql"])
        res.append({"family": r["family"], "question": r["question"], "gold_sql": r["golds"][0], "teacher_sql": r["sql"],
                    "gold_perturbed_rows": [list(x) for x in g.rows[:8]], "teacher_perturbed_rows": [list(x) for x in t.rows[:8]],
                    "gold_n": len(g.rows), "teacher_n": len(t.rows)})
    return res


def leakage_and_coverage() -> dict:
    res = {}
    for run in RUNS:
        sp = stage_artifact(store, run, "split")
        train = sp["train"]
        sets = {"heldout": json.loads((ROOT / "runs" / run / "heldout" / "heldout.json").read_text())}
        sf = ROOT / "runs" / run / "heldout" / "stress.json"
        if sf.exists():
            sets["stress"] = json.loads(sf.read_text())
        tq = {norm(t["question"]) for t in train}
        tqs = {skel(t["question"]) for t in train}
        ts = {norm(t["gold_sql"]) for t in train}
        tss = {skel(t["gold_sql"]) for t in train}
        tt = {t["template"] for t in train}
        d = {"train_tasks_in_split": len(train), "train_family_counts": dict(Counter(t["family"] for t in train))}
        for name, items in sets.items():
            n = len(items)
            c = {
                "n": n,
                "exact_question": sum(norm(i["question"]) in tq for i in items),
                "masked_question": sum(skel(i["question"]) in tqs for i in items),
                "exact_sql": sum(norm(i["gold_sql"]) in ts for i in items),
                "masked_sql_skeleton": sum(skel(i["gold_sql"]) in tss for i in items),
                "same_template_name": sum(i["template"] in tt for i in items),
                "family_counts": dict(Counter(i["family"] for i in items)),
            }
            for k in ("exact_question", "masked_question", "exact_sql", "masked_sql_skeleton", "same_template_name"):
                c[k + "_rate"] = round(c[k] / n, 4)
            d[name] = c
        res[run] = d
    return res


def spotcheck(pool) -> None:
    rng = random.Random(SEED)
    rows = rng.sample(pool, 30)
    md = ["# P2 spot-check: 30 random TRAINING rows (seed 1234)\n", "Judge: does the SQL answer the question?\n"]
    for i, r in enumerate(rows, 1):
        o = run_select(DB, r["sql"])
        md += [f"## {i}. {r['run']}/{r['round']} ({r['family']})", f"Q: {r['question']}", "```sql", r["sql"], "```",
               f"columns: {list(o.columns)}  total rows: {len(o.rows)}"]
        md += [f"- {list(x)}" for x in o.rows[:5]] + [""]
    Path(".distillery/proofs").mkdir(parents=True, exist_ok=True)
    Path(".distillery/proofs/p2_spotcheck.md").write_text("\n".join(md))


if __name__ == "__main__":
    TMP.mkdir(parents=True, exist_ok=True)
    pool, notes = load_pool()
    dbs = [ROOT / "runs" / r / "db.sqlite" for r in RUNS]
    out = {"same_db_file_across_runs": same_db(dbs), "pool_rows": len(pool), "pool_notes": notes}
    ok = label_check(DB, pool)
    out["P2.1_label_correct"] = f"{sum(ok)}/{len(ok)}"
    pdb = TMP / "p2_perturbed.sqlite"
    out["P2.2_perturbation"] = perturb(DB, pdb)
    okp = label_check(pdb, pool)
    both = sum(a and b for a, b in zip(ok, okp, strict=True))
    lost = sum(a and not b for a, b in zip(ok, okp, strict=True))
    out["P2.2_perturbed"] = {
        "match_original": sum(ok), "match_perturbed": sum(okp), "match_both": both,
        "matched_original_not_perturbed": lost,
        "accidental_match_rate": round(lost / sum(ok), 4) if sum(ok) else None,
        "by_family_lost": dict(Counter(r["family"] for r, a, b in zip(pool, ok, okp, strict=True) if a and not b)),
    }
    # Filter recall on a pool of the trained rows; gated-run train pool (gold as stand-in) added separately.
    pdbs = [pdb]
    for sd in (SEED + 1, SEED + 2):
        extra = TMP / f"p2_perturbed_{sd}.sqlite"
        perturb(DB, extra, sd)
        pdbs.append(extra)
    out["P2.3_filter_recall_trained_rows"] = recall(DB, pool, pdbs)
    gt = stage_artifact(store, "gated-1p7b-r1", "split")["train"]
    gpool = [{"family": t["family"], "sql": t["gold_sql"], "golds": [t["gold_sql"]], "requires_order": t["requires_order"]} for t in gt]
    out["P2.3_filter_recall_gated_train_pool_gold"] = recall(DB, gpool, pdbs)
    out["P2.2_null_handling_lost_rows"] = lost_rows(pool, ok, okp, pdb)
    out["P2.4_P2.5_leakage_and_coverage"] = leakage_and_coverage()
    out["train_family_counts_trained_rows"] = dict(Counter(r["family"] for r in pool))
    spotcheck(pool)
    out["P2.6_spotcheck"] = ".distillery/proofs/p2_spotcheck.md (30 rows, not committed)"
    print(json.dumps(out, indent=1))
