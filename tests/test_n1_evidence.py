"""docs/proofs/n1_evidence.py and the committed docs/proofs/evidence/n1_natural_round.json.

The script reads gitignored stores, so CI cannot re-run it; these tests pin its pure logic (the spend
delta against a copied store) and check that the committed file is internally consistent and clean.
"""

from __future__ import annotations

import importlib.util
import json
import sqlite3
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("n1_evidence", ROOT / "docs/proofs/n1_evidence.py")
assert _spec and _spec.loader
n1_evidence = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(n1_evidence)

EVIDENCE = ROOT / "docs/proofs/evidence/n1_natural_round.json"
SEALED_KEYS = {
    "question",
    "gold_sql",
    "student_sql",
    "teacher_sql",
    "base_sql",
    "raw",
    "sql",
    "examples",
}


def _walk_keys(v: object) -> set[str]:
    if isinstance(v, dict):
        return set(v) | {k for x in v.values() for k in _walk_keys(x)}
    if isinstance(v, list):
        return {k for x in v for k in _walk_keys(x)}
    return set()


def _ledger(path: Path, rows: list[tuple[str, str, float]]) -> Path:
    path.mkdir(parents=True)
    con = sqlite3.connect(path / "index.sqlite")
    con.execute(
        "CREATE TABLE spend (id INTEGER PRIMARY KEY, run_id TEXT, kind TEXT, model TEXT, usd REAL, "
        "input_tokens INTEGER, output_tokens INTEGER, day TEXT, created_at TEXT)"
    )
    for i, (kind, model, usd) in enumerate(rows, 1):
        con.execute(
            "INSERT INTO spend VALUES (?,?,?,?,?,?,?,?,?)",
            (i, "r", kind, model, usd, 10, 1, "2026-10-07", f"2026-10-07T10:00:{i:02d}+00:00"),
        )
    con.commit()
    con.close()
    return path / "index.sqlite"


def test_spend_delta_is_the_rows_beyond_the_copied_baseline(tmp_path: Path) -> None:
    base = [("llm", "m", 0.5), ("finetune", "q", 2.0)]
    p4 = n1_evidence.spend_rows(_ledger(tmp_path / "p4", base))
    n1 = n1_evidence.spend_rows(
        _ledger(tmp_path / "n1", [*base, ("llm", "m", 0.25), ("sandbox", "s", 0.0)])
    )
    delta = n1_evidence.spend_delta(n1, p4)
    assert [(r[2], r[4]) for r in delta] == [("llm", 0.25), ("sandbox", 0.0)]
    assert n1_evidence.usd(delta) == 0.25
    assert n1_evidence.by_kind(delta) == {
        "llm": {"rows": 1, "usd": 0.25},
        "sandbox": {"rows": 1, "usd": 0.0},
    }
    assert n1_evidence.spend_delta(p4, None) == p4  # no baseline: every row counts


def test_spend_delta_refuses_a_store_that_is_not_a_copy(tmp_path: Path) -> None:
    p4 = n1_evidence.spend_rows(_ledger(tmp_path / "p4", [("llm", "m", 0.5)]))
    other = n1_evidence.spend_rows(_ledger(tmp_path / "other", [("llm", "m", 0.75)]))
    with pytest.raises(n1_evidence.EvidenceError, match="not a subset"):
        n1_evidence.spend_delta(other, p4)


def test_nano_sums_are_exact() -> None:
    rows: list[tuple[Any, ...]] = [(i, "r", "llm", "m", 0.1, 0, 0, "d", "t") for i in range(10)]
    assert n1_evidence.usd(rows) == 1.0  # ten dimes, no float drift


def test_ledgers_are_opened_without_writing(tmp_path: Path) -> None:
    index = _ledger(tmp_path / "s", [("llm", "m", 1.0)])
    before = (index.read_bytes(), sorted(p.name for p in index.parent.iterdir()))
    n1_evidence.spend_rows(index)
    assert (index.read_bytes(), sorted(p.name for p in index.parent.iterdir())) == before


def test_committed_evidence_is_consistent() -> None:
    d = json.loads(EVIDENCE.read_text(encoding="utf-8"))
    p = d["primary"]
    th = p["thresholds"]
    # the decision follows from the numbers and the unchanged thresholds
    assert p["checks"]["ratio_lower_bound_ok"] == (p["ratio"]["lo"] >= th["ratio_lower_bound_min"])
    assert p["checks"]["mcnemar_ok"] == (p["mcnemar"]["p"] < th["mcnemar_alpha"])
    assert p["decision"] == ("PROMOTE" if all(p["checks"].values()) else "REJECT")
    assert d["decision"]["primary"] == p["decision"] == p["decision_human"]
    assert d["decision"]["no_threshold_changed"] is True
    assert p["set"]["n"] == p["n"] and p["set"]["sealed_before_training"] is True
    assert p["mcnemar"]["student_only_vs_base"] > p["mcnemar"]["base_only_vs_student"]
    # secondary blocks are reported, never merged into the primary claim
    ho = d["secondary"]["heldout"]
    assert ho["no_regression"]["passed"] == (ho["ratio"]["lo"] >= ho["no_regression"]["minimum"])
    assert ho["same_set_as_p4"] is True and d["secondary"]["stress"]["same_set_as_p4"] is True
    assert d["secondary"]["gate_b_rescore"]["status"] == "pending"
    assert [x["status"] for x in d["pending"]] == ["pending"]
    # training rows and artifact
    t = d["training_data"]
    assert t["rows_add_up"] and t["train_rows_total"] == (
        t["p4_rows_template_originals_plus_verified_paraphrases"] + t["natural_rows_appended"]
    )
    assert all(f["matches_report"] for f in t["identity_files"].values())
    assert (
        d["artifact"]["differs_from_p4_adapter"]
        and d["artifact"]["finetune"]["hyperparameters_equal_p4"]
    )
    # spend: parts add up, N1 stays under its pre-registered ceiling, the checks passed
    s = d["spend"]
    parts = [e["usd"] for e in s["efforts"].values()]
    assert abs(sum(parts) - s["project_total_usd"]) < 1e-5
    assert (
        abs(sum(c["usd"] for c in s["efforts"]["N1"]["components"]) - s["efforts"]["N1"]["usd"])
        < 1e-5
    )
    assert s["n1_within_ceiling"] and s["n1_checks"]["scored_once"]
    assert abs(s["project_cap_usd"] - s["project_total_usd"] - s["project_headroom_usd"]) < 1e-5
    # pre-registration chain precedes training and scoring
    assert all(d["preregistration"]["order_checks"].values())
    assert d["preregistration"]["claim_rule_unchanged"]["report_matches_pre_registration"]


def test_committed_evidence_has_no_sealed_text_or_absolute_paths() -> None:
    text = EVIDENCE.read_text(encoding="utf-8")
    d = json.loads(text)
    assert not (_walk_keys(d) & SEALED_KEYS)
    assert "SELECT " not in text.upper()
    assert "/home/" not in text and "/Users/" not in text
