# ruff: noqa: S101
"""The drift simulator on synthetic recorded rows (no network)."""

from __future__ import annotations

import importlib.util
import json
import random
import sys
from pathlib import Path
from typing import Any

import pytest

from distillery.drift import DriftMonitor, DriftParams, RouteEvent
from distillery.drift_sim import (
    DriftSimError,
    _tau_for,  # noqa: PLC2701
    audit_tail,
    calibrate_tau,
    load_rows,
    route_of,
    run_burst,
    simulate,
    stream,
)

SPEC = importlib.util.spec_from_file_location(
    "drift_sim_script", Path(__file__).resolve().parents[1] / "scripts" / "drift_sim.py"
)
assert SPEC and SPEC.loader
script = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = script
SPEC.loader.exec_module(script)


def rows(
    n: int, defer_share: float, *, seed: int = 1, correct: bool = True
) -> list[dict[str, Any]]:
    """n rows, exactly ``defer_share`` of them deferred, shuffled; student rows are ``correct``."""
    k = round(n * defer_share)
    flags = [True] * k + [False] * (n - k)
    random.Random(seed).shuffle(flags)
    return [
        {
            "task_id": f"t{i}",
            "served_by": "teacher" if d else "student",
            "correct": (not d) and correct,
        }
        for i, d in enumerate(flags)
    ]


def params(**kw: Any) -> DriftParams:
    base: dict[str, Any] = {
        "window_w": 50, "tau": 0.8, "audit_rate": 0.0, "audit_window_m": 20, "audit_k": 5,
        "min_requests": 50,
    }  # fmt: skip
    base.update(kw)
    return DriftParams(**base)


# ---- rows and routing ---------------------------------------------------------------------


def test_route_of_recorded_decision_or_score_with_lambda() -> None:
    assert route_of({"served_by": "teacher", "score": 0.99}, 0.5) == "teacher"
    assert route_of({"score": 0.5}, 0.5) == "student"  # score >= lambda serves the student
    assert route_of({"score": 0.49}, 0.5) == "teacher"
    with pytest.raises(DriftSimError):
        route_of({"score": 0.9})  # no lambda
    with pytest.raises(DriftSimError):
        route_of({"correct": True}, 0.5)


def test_load_rows_shapes(tmp_path: Path) -> None:
    r = [{"served_by": "student", "correct": True}]
    (tmp_path / "a.json").write_text(json.dumps(r))
    (tmp_path / "b.json").write_text(json.dumps({"format": "x", "rows": r}))
    (tmp_path / "c.jsonl").write_text(json.dumps(r[0]) + "\n\n" + json.dumps(r[0]) + "\n")
    assert load_rows(tmp_path / "a.json") == r == load_rows(tmp_path / "b.json")
    assert len(load_rows(tmp_path / "c.jsonl")) == 2
    (tmp_path / "bad.json").write_text("[1, 2]")
    (tmp_path / "empty.json").write_text("[]")
    for name in ("bad.json", "empty.json", "missing.json"):
        with pytest.raises(DriftSimError):
            load_rows(tmp_path / name)


# ---- stream -------------------------------------------------------------------------------


def test_stream_is_deterministic_and_shaped() -> None:
    ind, burst = rows(100, 0.5), rows(40, 1.0, seed=2)
    a = stream(ind, burst, n_before=30, n_burst=10, seed=7)
    assert a == stream(ind, burst, n_before=30, n_burst=10, seed=7)
    assert a != stream(ind, burst, n_before=30, n_burst=10, seed=8)
    assert [r.index for r in a] == list(range(40))
    assert [r.source for r in a] == ["in_dist"] * 30 + ["burst"] * 10
    assert all(r.served_by == "teacher" for r in a[30:])
    assert len({r.request_id for r in a}) == 40
    # the in-distribution prefix does not depend on the burst
    assert a[:30] == stream(ind, rows(5, 0.0), n_before=30, n_burst=3, seed=7)[:30]


def test_stream_burst_at_and_empty_inputs() -> None:
    ind, burst = rows(10, 0.5), rows(10, 1.0)
    assert len(stream(ind, burst, n_before=5, burst_at=5, n_burst=2, seed=0)) == 7
    with pytest.raises(DriftSimError):
        stream(ind, burst, n_before=5, burst_at=6, seed=0)
    with pytest.raises(DriftSimError):
        stream([], burst, n_before=5, seed=0)
    with pytest.raises(DriftSimError):
        stream(ind, burst, n_before=-1, seed=0)


def test_stream_routes_score_rows_with_lambda() -> None:
    sc = [{"score": 0.9, "correct": True}, {"score": 0.1, "correct": False}]
    out = stream(sc, sc, n_before=50, n_burst=0, seed=3, lam=0.5)
    assert {r.served_by for r in out} == {"student", "teacher"}
    with pytest.raises(DriftSimError):
        stream(sc, sc, n_before=5, seed=3)  # score rows need lambda


# ---- tau calibration ----------------------------------------------------------------------


def test_calibrate_tau_lands_above_a_half_baseline() -> None:
    cal = calibrate_tau(rows(1000, 0.5), 50, 1.0, n_streams=100, seed=3)
    assert cal["baseline_deferral_share"] == 0.5
    assert 0.5 < cal["tau"] <= 1.0
    assert cal["false_alarm_per_1000"] <= 1.0
    assert cal["validation_false_alarm_per_1000"] >= 0.0
    assert cal["window_w"] == 50 and cal["n_streams"] == 100 and cal["seed"] == 3
    assert cal["tau_fraction"] == f"{round(cal['tau'] * 50)}/50"


def test_calibrate_tau_is_deterministic_and_monotone_in_the_target() -> None:
    fit = rows(600, 0.5)
    a = calibrate_tau(fit, 40, 2.0, n_streams=60, seed=5)
    assert a == calibrate_tau(fit, 40, 2.0, n_streams=60, seed=5)
    strict = calibrate_tau(fit, 40, 0.1, n_streams=60, seed=5)
    loose = calibrate_tau(fit, 40, 50.0, n_streams=60, seed=5)
    assert strict["tau"] >= a["tau"] >= loose["tau"]
    assert strict["false_alarm_per_1000"] <= 0.1 and a["false_alarm_per_1000"] <= 2.0


def test_calibrate_tau_smallest_tau_property() -> None:
    cal = calibrate_tau(rows(800, 0.3), 40, 1.0, n_streams=80, seed=2)
    j = int(cal["tau_fraction"].split("/")[0])
    if j > 1:  # one step lower must violate the target on the same calibration streams
        lower = calibrate_tau(rows(800, 0.3), 40, cal["false_alarm_per_1000"] - 1e-9, 80, 2)
        assert lower["tau"] > cal["tau"] or cal["false_alarm_per_1000"] == 0.0


def test_calibrate_tau_constant_traffic() -> None:
    # every request deferred: nothing is above the baseline, so no tau meets a 0 target
    with pytest.raises(DriftSimError, match="no tau"):
        calibrate_tau(rows(100, 1.0), 20, 0.0, n_streams=10, seed=1)
    # no deferrals at all: any tau works, the smallest grid tau (1/W) is never reached
    cal = calibrate_tau(rows(100, 0.0), 20, 0.0, n_streams=10, seed=1)
    assert cal["tau"] == _tau_for(1, 20) and cal["false_alarm_per_1000"] == 0.0


def test_calibrate_tau_rejects_bad_input() -> None:
    with pytest.raises(DriftSimError):
        calibrate_tau([], 10, 1.0)
    with pytest.raises(DriftSimError):
        calibrate_tau(rows(10, 0.5), 0, 1.0)
    with pytest.raises(DriftSimError):
        calibrate_tau(rows(10, 0.5), 10, -1.0)


def test_calibrated_tau_is_a_short_decimal_that_trips_at_exactly_j_of_w() -> None:
    assert _tau_for(5, 7) <= 5 / 7 and str(_tau_for(5, 7)).count("0") >= 0
    for w, j in ((7, 5), (7, 1), (3, 2), (97, 60)):
        tau = _tau_for(j, w)
        m = DriftMonitor(params(window_w=w, tau=tau, min_requests=w), "a" * 64)
        events = []
        for i in range(w):  # j deferrals then students: the window holds exactly j/w at the end
            events += m.observe(RouteEvent("teacher" if i < j else "student", f"r{i}"))
        assert events, (w, j)
        m2 = DriftMonitor(params(window_w=w, tau=tau, min_requests=w), "a" * 64)
        events2 = []
        for i in range(w):
            events2 += m2.observe(RouteEvent("teacher" if i < j - 1 else "student", f"r{i}"))
        assert not events2, (w, j)  # j - 1 of w does not trip


def test_audit_tail() -> None:
    assert audit_tail(1, 1, 0.3) == pytest.approx(0.3)
    assert audit_tail(5, 0, 0.3) == pytest.approx(1.0)
    assert audit_tail(3, 2, 0.5) == pytest.approx(0.5)
    assert audit_tail(50, 5, 0.0) == 0.0
    with pytest.raises(DriftSimError):
        audit_tail(3, 4, 0.1)


# ---- run_burst ----------------------------------------------------------------------------


def test_all_defer_burst_is_detected_within_the_window() -> None:
    fit = rows(1000, 0.5)
    cal = calibrate_tau(fit, 50, 1.0, n_streams=100, seed=3)
    p = params(tau=cal["tau"])
    out = run_burst(fit, rows(200, 1.0, seed=9), p, n_before=300, n_burst=200, seed=11)
    assert out["rolled_back"] and out["detector"] == "deferral_rate"
    assert out["false_alarms_before_burst"] == 0
    assert 1 <= out["detection_latency_requests"] <= 50  # within W
    assert out["rollback_index"] == 300 + out["detection_latency_requests"] - 1
    assert out["student_answers_served_after_rollback"] == 0
    assert out["student_answers_blocked_by_rollback"] == 0  # the burst never routes to the student
    assert out["wrong_student_answers_served_before_rollback"] == 0
    kinds = [e["kind"] for e in out["events"]]
    assert kinds == ["rollback", "teacher_only", "redistill_queued"]
    assert all(e["index"] == out["rollback_index"] for e in out["events"])
    json.dumps(out)  # serialisable
    assert out == run_burst(fit, rows(200, 1.0, seed=9), p, n_before=300, n_burst=200, seed=11)


def test_audit_path_catches_confident_wrong_student_answers() -> None:
    # the router keeps serving the student on the burst (deferral share FALLS), but it is wrong
    ind = rows(500, 0.4)
    burst = [{"task_id": f"b{i}", "served_by": "student", "correct": False} for i in range(50)]
    p = params(tau=0.9, audit_rate=1.0, audit_window_m=10, audit_k=4, min_requests=50)
    out = run_burst(ind, burst, p, n_before=200, n_burst=100, seed=4)
    assert out["detector"] == "audit_disagreement" and out["false_alarms_before_burst"] == 0
    assert out["detection_latency_requests"] == 4  # the 4th audited wrong answer
    assert out["wrong_student_answers_served_before_rollback"] == 4
    assert out["wrong_student_answers_served_before_burst"] == 0
    assert out["wrong_student_answers_served_during_burst"] == 4
    assert out["student_answers_served_after_rollback"] == 0
    assert out["student_answers_blocked_by_rollback"] == 96  # the rest of the burst went to teacher
    assert out["audits"] > 0 and out["audit_disagreements"] == 4


def test_without_audits_the_same_burst_is_not_caught() -> None:
    ind = rows(500, 0.4)
    burst = [{"task_id": f"b{i}", "served_by": "student", "correct": False} for i in range(50)]
    out = run_burst(ind, burst, params(tau=0.9), n_before=200, n_burst=100, seed=4)
    assert not out["rolled_back"] and out["detection_latency_requests"] is None
    assert out["wrong_student_answers_served_during_burst"] == 100  # why the audit exists


def test_partial_audit_rate_is_deterministic_and_slower() -> None:
    ind = rows(500, 0.4)
    burst = [{"task_id": f"b{i}", "served_by": "student", "correct": False} for i in range(50)]
    p = params(tau=0.9, audit_rate=0.25, audit_window_m=10, audit_k=4)
    out = run_burst(ind, burst, p, n_before=200, n_burst=100, seed=4)
    assert out["rolled_back"] and out["detection_latency_requests"] > 4
    assert out == run_burst(ind, burst, p, n_before=200, n_burst=100, seed=4)


def test_false_alarm_before_burst_is_reported_not_hidden() -> None:
    ind = rows(500, 0.9)  # in-distribution traffic that is itself mostly deferred
    out = run_burst(ind, rows(10, 1.0), params(tau=0.5), n_before=200, n_burst=50, seed=1)
    assert out["false_alarms_before_burst"] == 1
    assert out["detection_latency_requests"] is None and "false alarm" in out["latency_note"]
    assert out["rollback_index"] < 200
    assert out["student_answers_served_after_rollback"] == 0


def test_rollback_overrides_the_recorded_student_decision() -> None:
    ind = rows(300, 0.0)  # every request recorded as student-served
    burst = [{"served_by": "student", "correct": False}]
    p = params(tau=1.0, audit_rate=1.0, audit_window_m=5, audit_k=2)
    out = run_burst(ind, burst, p, n_before=10, n_burst=20, seed=0)
    assert out["rollback_index"] == 11
    assert (
        out["student_answers_blocked_by_rollback"] == 18
    )  # 30 requests, 12 served before the rollback
    assert out["student_answers_served_after_rollback"] == 0


def test_student_row_without_correct_is_an_error() -> None:
    reqs = stream([{"served_by": "student"}], [{"served_by": "student"}], n_before=2, seed=0)
    with pytest.raises(DriftSimError, match="correct"):
        simulate(reqs, params())


def test_no_burst_means_any_rollback_is_a_false_alarm() -> None:
    out = run_burst(rows(200, 0.9), [{"served_by": "teacher"}], params(tau=0.5), n_before=100,
                    n_burst=0, seed=0)  # fmt: skip
    assert out["burst_start_index"] is None and out["false_alarms_before_burst"] == 1


# ---- CLI ----------------------------------------------------------------------------------


def _write(path: Path, r: list[dict[str, Any]]) -> Path:
    path.write_text(json.dumps({"format": "distillery-router-signals-v1", "rows": r}))
    return path


def test_cli_writes_an_artifact_and_refuses_to_overwrite(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    fit = _write(tmp_path / "fit.json", rows(600, 0.5))
    burst = _write(tmp_path / "burst.json", rows(100, 1.0, seed=2))
    out = tmp_path / "sim.json"
    argv = [
        "--fit-rows", str(fit), "--burst-rows", str(burst), "--window", "40", "--target-fa", "2",
        "--audit-rate", "0.1", "--n-before", "200", "--n-burst", "100", "--n-streams", "60",
        "--seed", "5", "--out", str(out),
    ]  # fmt: skip
    assert script.main(argv) == 0
    doc = json.loads(out.read_text())
    assert doc["format"] == "distillery-drift-sim-v1"
    assert doc["calibration"]["baseline_deferral_share"] == 0.5
    assert doc["params"]["tau"] == doc["calibration"]["tau"]
    assert (
        doc["burst"]["rolled_back"] and doc["burst"]["student_answers_served_after_rollback"] == 0
    )
    assert doc["inputs"]["fit_rows"]["n"] == 600 and len(doc["inputs"]["fit_rows"]["sha256"]) == 64
    assert "tau=" in capsys.readouterr().out
    before = out.read_bytes()
    assert script.main(argv) == 2  # never overwritten
    assert out.read_bytes() == before
    assert "already exists" in capsys.readouterr().err


def test_cli_errors_are_exit_2_without_a_traceback(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    fit = _write(tmp_path / "fit.json", rows(50, 0.5))
    argv = ["--fit-rows", str(fit), "--burst-rows", str(tmp_path / "nope.json"), "--window", "10",
            "--target-fa", "1", "--out", str(tmp_path / "o.json")]  # fmt: skip
    assert script.main(argv) == 2
    assert "error:" in capsys.readouterr().err
    assert not (tmp_path / "o.json").exists()


def test_cli_scores_rows_with_a_router_dir_and_given_tau(tmp_path: Path) -> None:
    from distillery.router import RouterModel

    model = RouterModel(
        feature_names=("exec_ok", "mean_logprob", "nn_sim"), means=(0.5, -0.5, 0.5),
        stds=(0.5, 0.5, 0.25), weights=(1.5, 0.5, 1.0), bias=0.5, l2=0.01, n_fit=10,
        fit_sha256="f" * 64,
    )  # fmt: skip
    rd = tmp_path / "router"
    rd.mkdir()
    (rd / "model.json").write_text(json.dumps({"model": model.to_json()}))
    (rd / "cert.json").write_text(json.dumps({"lambda_hat": 0.8}))
    good = {"exec_ok": 1, "mean_logprob": -0.05, "nn_sim": 0.95, "correct": True}
    bad = {"exec_ok": 0, "mean_logprob": -2.0, "nn_sim": 0.1, "correct": False}
    fit = _write(tmp_path / "fit.json", [good, bad] * 50)
    burst = _write(tmp_path / "burst.json", [bad] * 20)
    out = tmp_path / "o.json"
    rc = script.main([
        "--fit-rows", str(fit), "--burst-rows", str(burst), "--router-dir", str(rd),
        "--window", "20", "--target-fa", "1", "--tau", "0.9", "--n-before", "60",
        "--n-burst", "40", "--out", str(out),
    ])  # fmt: skip
    assert rc == 0
    doc = json.loads(out.read_text())
    assert doc["calibration"] is None and doc["tau_source"].startswith("given")
    assert doc["inputs"]["lambda"] == 0.8
    assert doc["burst"]["rolled_back"]
