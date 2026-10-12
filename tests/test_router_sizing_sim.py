# ruff: noqa: S101
from __future__ import annotations

import importlib.util
import json
import math
import random
import sys
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location(
    "router_sizing_sim", Path(__file__).resolve().parents[1] / "scripts" / "router_sizing_sim.py"
)
assert SPEC and SPEC.loader
sim = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = sim  # dataclasses resolve annotations through sys.modules
SPEC.loader.exec_module(sim)


def test_parse_mix_and_levels() -> None:
    fams = sim.parse_mix(sim.DEFAULT_MIX)
    assert [(f.name, f.share, f.accuracy) for f in fams] == [
        ("templated", 0.5, 0.92),
        ("natural", 0.25, 0.57),
        ("stress", 0.25, 0.66),
    ]
    assert sim.parse_mix("a:1.0:0.5")[0].nn_mean == 0.6
    for bad in ("a:0.5:0.9", "a:1:2", "a:b", "a:0.5:0.9,b:0.6:0.9"):
        with pytest.raises(ValueError):
            sim.parse_mix(bad)
    levels = sim.make_levels(0.20, 0.90, 0.02)
    assert len(levels) == 36 and levels[0] == 0.2 and levels[-1] == 0.9
    assert all(a < b for a, b in zip(levels, levels[1:], strict=False))
    with pytest.raises(ValueError):
        sim.make_levels(0.5, 0.2, 0.1)


def test_toy_signals_are_valid_and_informative() -> None:
    rng = random.Random(0)
    rows, labels = sim.draw_toy(rng, sim.parse_mix(sim.DEFAULT_MIX), 4000, 1.0)
    assert all(
        r["mean_logprob"] <= 0 and -1 <= r["nn_sim"] <= 1 and r["exec_ok"] in (0.0, 1.0)
        for r in rows
    )
    assert sum(labels) / len(labels) == pytest.approx(0.768, abs=0.03)
    ok_exec = [r["exec_ok"] for r, y in zip(rows, labels, strict=True) if y]
    bad_exec = [r["exec_ok"] for r, y in zip(rows, labels, strict=True) if not y]
    assert sum(ok_exec) / len(ok_exec) > sum(bad_exec) / len(bad_exec) + 0.1


def test_run_is_seeded_and_sane() -> None:
    fams = sim.parse_mix(sim.DEFAULT_MIX)
    make = sim.toy_rep_factory(fams, 1.0, 600, 400, sim.make_levels(0.2, 0.9, 0.05))
    kw = {"alpha": 0.08, "delta": 0.05, "cal_sizes": [300, 600], "reps": 6, "seed": 3}
    a = sim.run(make, **kw)
    b = sim.run(make, **kw)
    assert a == b  # same seed, same numbers (nan-free here)
    assert [r["cal_size"] for r in a] == [300, 600]
    for r in a:
        assert 0.0 <= r["p_fail"] <= 1.0 and 0.0 <= r["mean_cal_coverage"] <= 1.0
        assert r["p_cert_and_test_over_alpha"] <= 1.0 - r["p_fail"] + 1e-12
    text = sim.format_table(a)
    assert "cal_n" in text and "300" in text and "600" in text


def test_no_signal_and_perfect_signal_extremes() -> None:
    fams = sim.parse_mix("all:1.0:0.7:0.6")
    kw = {"alpha": 0.08, "delta": 0.05, "cal_sizes": [500], "reps": 5, "seed": 1}
    levels = sim.make_levels(0.2, 0.9, 0.05)
    flat = sim.run(sim.toy_rep_factory(fams, 0.0, 800, 400, levels), **kw)[0]
    assert flat["p_fail"] == 1.0  # signals carry nothing: 30% errors everywhere, never certified
    assert math.isnan(flat["mean_test_risk_certified"])
    strong = sim.run(sim.toy_rep_factory(fams, 4.0, 800, 400, levels), **kw)[0]
    assert strong["p_fail"] < 1.0 and strong["mean_cal_coverage"] > 0.0


def test_from_fit_hook_and_cli(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    rng = random.Random(4)
    rows, labels = sim.draw_toy(rng, sim.parse_mix(sim.DEFAULT_MIX), 700, 1.5)
    items = [{**r, "correct": y} for r, y in zip(rows, labels, strict=True)]
    path = tmp_path / "fit.json"
    path.write_text(json.dumps({"items": items}), encoding="utf-8")
    loaded_rows, loaded_labels = sim.load_fit_items(path)
    assert loaded_rows == rows and loaded_labels == labels
    oof = sim.out_of_fold_scores(loaded_rows, loaded_labels)
    assert len(oof) == 700 and all(0.0 < s < 1.0 for s in oof)
    rc = sim.main(
        [
            "--from-fit",
            str(path),
            "--reps",
            "4",
            "--cal-sizes",
            "100",
            "--test-size",
            "100",
            "--json",
        ]
    )
    assert rc == 0
    out = json.loads(capsys.readouterr().out)
    assert len(out) == 1 and out[0]["cal_size"] == 100 and out[0]["reps"] == 4
    path.write_text(json.dumps(items[:5]), encoding="utf-8")
    with pytest.raises(ValueError, match="at least 20"):
        sim.load_fit_items(path)
