# ruff: noqa: S101
"""Receipts in deploy/evidence.json (scripts/build_evidence.py): one checkable row per claim.

Pending when the evidence file does not exist, verified/failed from its keys when it does, a
present file with a missing key stops the build, and the output is byte-for-byte deterministic.
The KEYS paths for files that do not exist yet are checked against the code that writes them.
"""

from __future__ import annotations

import importlib.util
import json
import shutil
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient
from server_helpers import make_settings
from test_router_eval import build_rows, model

from distillery.drift import DriftParams
from distillery.drift_sim import run_burst
from distillery.h1_measure import cost_per_1k
from distillery.router_eval import evaluate_test
from distillery.server import create_app

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("build_evidence", ROOT / "scripts/build_evidence.py")
assert _spec and _spec.loader
be = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(be)
_dspec = importlib.util.spec_from_file_location("drift_sim_cli", ROOT / "scripts/drift_sim.py")
assert _dspec and _dspec.loader
drift_script = importlib.util.module_from_spec(_dspec)
_dspec.loader.exec_module(drift_script)

COMMIT = "b" * 40
FIELDS = {
    "id", "claim", "status", "verdict", "value_display", "artifact", "permalink", "prereg",
    "sealed_sha256", "adapter_sha256", "gate_inputs", "ledger_total_usd",
}  # fmt: skip
PENDING = ["W-$", "W-E2E"]  # W-C1/W-C2/W-D scored 2026-10-14
EVIDENCE = Path(be.EVIDENCE_DIR)


def by_id(root: Path | None = None) -> dict[str, dict[str, Any]]:
    out = be.build(COMMIT, "2026-01-01T00:00:00+00:00", root)
    return {r["id"]: r for r in out["receipts"]}


@pytest.fixture
def sandbox(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A copy of the evidence files and the W-G test, so files can be added without touching the repo."""
    shutil.copytree(ROOT / EVIDENCE, tmp_path / EVIDENCE)
    (tmp_path / "tests").mkdir()
    shutil.copy(ROOT / be.W_G_ARTIFACT, tmp_path / be.W_G_ARTIFACT)
    monkeypatch.setattr(be, "REPO_ROOT", tmp_path)
    return tmp_path


def put(root: Path, stem: str, doc: dict[str, Any]) -> None:
    (root / EVIDENCE / f"{stem}.json").write_text(json.dumps(doc), encoding="utf-8")


# ---------------------------------------------------------------- the committed shape


def test_every_proof_has_a_receipt_then_the_preregistered_claims() -> None:
    out = be.build(COMMIT, "t")
    ids = [r["id"] for r in out["receipts"]]
    assert ids == [p["id"] for p in out["proofs"]] + ["W-C1", "W-C2", "W-D", "W-$", "W-G", "W-E2E"]
    assert len(set(ids)) == len(ids)
    for r in out["receipts"]:
        assert set(r) == FIELDS
        assert r["status"] in {"verified", "pending", "failed"}
        assert r["permalink"] == f"{out['repo_url']}/blob/{COMMIT}/{r['artifact']}"
        assert r["claim"].endswith(".") and not any(
            c.isdigit() for c in r["claim"].replace("1k", "")
        )
        assert isinstance(r["sealed_sha256"], dict)


def test_status_follows_the_proofs_own_verdict_and_nothing_partial_is_verified() -> None:
    out = be.build(COMMIT, "t")
    verdict = {p["id"]: p["verdict"] for p in out["proofs"]}
    for r in out["receipts"][: len(out["proofs"])]:
        assert r["verdict"] == verdict[r["id"]]
        assert (r["status"] == "verified") == (verdict[r["id"]] == "PASS"), r["id"]
        assert (ROOT / r["artifact"]).is_file()
    got = by_id()
    assert got["W2"]["status"] == got["W5"]["status"] == got["B2"]["status"] == "failed"
    assert got["W4"]["status"] == "verified"


def test_n1_is_failed_in_the_evidence_own_words() -> None:
    n1 = by_id()["N1"]
    assert n1["status"] == "failed" and n1["verdict"] == "FAIL"
    assert n1["value_display"].startswith("REJECT: ratio lower bound 0.495 < 0.85")
    assert n1["gate_inputs"]["ratio_lo"] == pytest.approx(0.4946236559139785)
    assert n1["gate_inputs"]["ratio_lower_bound_min"] == 0.85
    assert n1["prereg"]["commit"] == "bcd5f2df9388a9c31be5972530a3d5d20744ee5d"
    assert n1["prereg"]["committed_at"].startswith("2026-10-07")
    assert n1["adapter_sha256"] == (
        "7d6ce813882bf3a211c84bfab7f8109b3bcdd6ca4617349a03bff58076b85193"
    )
    assert set(n1["sealed_sha256"]) == {"n1_eval", "heldout", "stress"}
    assert all(len(v) == 64 for v in n1["sealed_sha256"].values())
    assert n1["ledger_total_usd"] == pytest.approx(3.828866)


def test_only_n1_and_the_router_record_a_preregistration_and_no_gate_input_is_null() -> None:
    got = by_id()
    assert [i for i, r in got.items() if r["prereg"] is not None] == ["N1", "W-C1", "W-C2"]
    for r in got.values():
        if r["gate_inputs"] is not None:
            assert all(v is not None for v in r["gate_inputs"].values()), r["id"]
    assert (
        got["W4"]["adapter_sha256"]
        == got["B2"]["adapter_sha256"]
        == got["Gate B"]["adapter_sha256"]
    )


def test_pending_claims_have_no_value_and_point_at_the_file_they_will_use() -> None:
    got = by_id()
    for rid in PENDING:
        r = got[rid]
        assert r["status"] == "pending" and r["verdict"] is None and r["value_display"] is None
        assert r["prereg"] is None and r["sealed_sha256"] == {} and r["gate_inputs"] is None
        assert r["adapter_sha256"] is None and r["ledger_total_usd"] is None
        assert (
            r["artifact"].startswith("docs/proofs/evidence/")
            and not (ROOT / r["artifact"]).exists()
        )
    assert got["W-C1"]["artifact"] == "docs/proofs/evidence/router_eval_test.json"
    assert got["W-C2"]["artifact"] == "docs/proofs/evidence/router_eval_test.json"
    assert got["W-D"]["artifact"] == "docs/proofs/evidence/drift_burst.json"
    assert got["W-$"]["artifact"] == "docs/proofs/evidence/h1_endpoint.json"


def test_w_g_is_verified_as_a_test() -> None:
    r = by_id()["W-G"]
    assert r["status"] == "verified" and r["artifact"] == "tests/test_acme_wg.py"
    assert r["value_display"] == "same agent code, base URL swapped; test passes"
    assert (ROOT / r["artifact"]).is_file()


def test_no_sealed_text_in_receipts() -> None:
    text = json.dumps(be.build(COMMIT, "t")["receipts"])
    assert "SELECT " not in text.upper()
    assert not (
        {"question", "gold_sql", "student_sql", "raw", "sql"}
        & _keys(be.build(COMMIT, "t")["receipts"])
    )


def _keys(v: object) -> set[str]:
    if isinstance(v, dict):
        return set(v) | {k for x in v.values() for k in _keys(x)}
    if isinstance(v, list):
        return {k for x in v for k in _keys(x)}
    return set()


# ---------------------------------------------------------------- determinism


def test_same_inputs_same_bytes(sandbox: Path) -> None:
    a = json.dumps(be.build(COMMIT, "t", sandbox), indent=2)
    b = json.dumps(be.build(COMMIT, "t", sandbox), indent=2)
    assert a == b


def test_main_writes_identical_bytes_for_the_same_commit(tmp_path: Path) -> None:
    one, two = tmp_path / "one.json", tmp_path / "two.json"
    assert be.main(["--out", str(one), "--commit", COMMIT]) == 0
    assert be.main(["--out", str(two), "--commit", COMMIT]) == 0
    assert one.read_bytes() == two.read_bytes()
    assert json.loads(one.read_text())["receipts"][0]["permalink"].startswith(be.REPO_URL)


# ---------------------------------------------------------------- pending -> verified / failed


def _router_doc(**over: Any) -> dict[str, Any]:
    """The `router evaluate` body: evaluate_test's result plus the keys router_cmd adds."""
    res = evaluate_test(model(), 0.5, build_rows(), alpha=0.05)
    res.pop("per_item")
    return {
        "prereg": {"commit": "c" * 40, "committed_at": "2026-10-20T10:00:00+05:30"},
        "test_split_sha256": "d" * 64,
        "alpha": 0.05,
        "delta": 0.1,
        **res,
        **over,
    }


def test_router_keys_match_the_code_that_writes_them(sandbox: Path) -> None:
    src = (ROOT / "src/distillery/router_cmd.py").read_text(encoding="utf-8")
    for key in ('"prereg"', '"test_split_sha256"', '"alpha"', '"delta"', "**res"):
        assert key in src
    doc = _router_doc()
    put(sandbox, "router_eval_test", doc)
    got = by_id(sandbox)
    c1, c2 = doc["W-C1"], doc["W-C2"]
    for rid, c in (("W-C1", c1), ("W-C2", c2)):
        r = got[rid]
        assert r["status"] == ("verified" if c["pass"] else "failed")
        assert r["verdict"] == ("PASS" if c["pass"] else "FAIL")
        assert r["sealed_sha256"] == {"test_split": "d" * 64}
        assert r["prereg"] == {"commit": "c" * 40, "committed_at": "2026-10-20T10:00:00+05:30"}
        assert r["adapter_sha256"] is None and r["ledger_total_usd"] is None
    assert f"selective error {c1['selective_error']:.4g} " in got["W-C1"]["value_display"]
    assert got["W-C1"]["gate_inputs"]["coverage"] == c1["coverage"]
    assert (
        f"{c2['n_error_deferred']} of {c2['n_error_items']} wrong answers"
        in got["W-C2"]["value_display"]
    )
    assert got["W-C1"]["artifact"] == "docs/proofs/evidence/router_eval_test.json"


def test_a_failing_router_test_is_failed_not_verified(sandbox: Path) -> None:
    doc = _router_doc()
    doc["W-C1"]["pass"] = False
    put(sandbox, "router_eval_test", doc)
    got = by_id(sandbox)
    assert got["W-C1"]["status"] == "failed" and got["W-C1"]["value_display"]
    assert got["W-C2"]["status"] == ("verified" if doc["W-C2"]["pass"] else "failed")


def test_drift_burst_keys_match_the_simulator(sandbox: Path) -> None:
    fit = [
        {"task_id": f"t{i}", "served_by": "teacher" if i % 2 else "student", "correct": True}
        for i in range(200)
    ]
    burst = [{"task_id": f"b{i}", "served_by": "teacher", "correct": False} for i in range(50)]
    p = DriftParams(
        window_w=20, tau=0.9, audit_rate=0.0, audit_window_m=10, audit_k=4, min_requests=20
    )
    rep = run_burst(fit, burst, p, n_before=100, n_burst=60, seed=1)
    doc = {
        "params": p.to_json(),
        "inputs": {"fit_rows": {"sha256": "e" * 64}, "burst_rows": {"sha256": "f" * 64}},
        "burst": rep,
        "verdict": drift_script.verdict(rep, 25, 3),
    }
    put(sandbox, "drift_burst", doc)
    r = by_id(sandbox)["W-D"]
    assert rep["rolled_back"] and r["status"] == "verified"
    assert r["sealed_sha256"] == {"fit_rows": "e" * 64, "burst_rows": "f" * 64}
    assert r["gate_inputs"] == p.to_json()
    assert str(rep["detection_latency_requests"]) in r["value_display"]
    doc["verdict"] = drift_script.verdict({**rep, "detection_latency_requests": 26}, 25, 3)
    put(sandbox, "drift_burst", doc)
    assert by_id(sandbox)["W-D"]["status"] == "failed"


def test_h1_keys_match_the_measurement_code(sandbox: Path) -> None:
    doc = {
        "costs": {"student": cost_per_1k(2.5, 1200.0), "projected": True},
        "ledger": {"delta_usd": 0.75},
        "error": None,
    }
    put(sandbox, "h1_endpoint", doc)
    r = by_id(sandbox)["W-$"]
    assert r["status"] == "verified"
    assert f"student USD {doc['costs']['student']['usd_per_1k']:.4g} " in r["value_display"]
    assert r["ledger_total_usd"] == 0.75
    assert r["gate_inputs"]["hourly_usd"] == 2.5
    doc["error"] = "endpoint never became ready"
    put(sandbox, "h1_endpoint", doc)
    assert by_id(sandbox)["W-$"]["status"] == "failed"


def test_a_present_file_with_a_missing_key_stops_the_build(sandbox: Path) -> None:
    doc = _router_doc()
    del doc["W-C2"]["deferred_share_lower_95"]
    put(sandbox, "router_eval_test", doc)
    with pytest.raises(KeyError, match=r"router_eval_test\.json.*deferred_share_lower_95"):
        be.build(COMMIT, "t", sandbox)


def test_an_existing_proof_with_a_missing_key_stops_the_build(sandbox: Path) -> None:
    f = sandbox / EVIDENCE / "n1_natural_round.json"
    doc = json.loads(f.read_text())
    del doc["primary"]["reasons"]
    f.write_text(json.dumps(doc))
    with pytest.raises(KeyError, match="reasons"):
        be.build(COMMIT, "t", sandbox)


def test_w_g_needs_its_test_file(sandbox: Path) -> None:
    (sandbox / be.W_G_ARTIFACT).unlink()
    with pytest.raises(FileNotFoundError):
        be.build(COMMIT, "t", sandbox)


def test_the_existing_evidence_endpoint_serves_the_committed_receipts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("DISTILLERY_EVIDENCE_FILE", raising=False)
    committed = json.loads((ROOT / "deploy/evidence.json").read_text(encoding="utf-8"))
    with TestClient(create_app(make_settings(tmp_path))) as c:
        body = c.get("/api/evidence").json()
    assert body["receipts"] == committed["receipts"]
    assert [r["id"] for r in body["receipts"]][-6:] == list(be.PENDING_ORDER)
    # W-C1/W-C2 point at one file, W-G at a test: the pending ones are never linked as evidence
    assert {r["status"] for r in body["receipts"] if r["id"] in PENDING} == {"pending"}
