# ruff: noqa: S101, S603
from __future__ import annotations

import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import pytest

from distillery.budget import BudgetExceeded
from distillery.evaluator import ArtifactMismatchError
from distillery.finetune import PINNED_HYPERPARAMETERS, HyperParameters, JobFailedError
from distillery.orchestrator import (
    DRY_RUN_LABEL,
    SCALES,
    STUDENT_COST_UNAVAILABLE,
    ConfigRefusal,
    Pipeline,
    PipelineConfig,
    Scale,
    ShortfallError,
    TaskTooEasyError,
    VerifierSelfTestError,
    plan_split,
)
from distillery.pipeline_fakes import DryRun, build_dry_run
from distillery.sandbox import FakeExecution, FakeSandbox, Job
from distillery.sandbox_executor import DB_PATH, SCRIPT_PATH, AsyncBridge, SandboxExecutor
from distillery.store import Store
from distillery.taskpacks.sql import schema as sql_schema
from distillery.taskpacks.sql.executor import LocalExecutor
from distillery.taskpacks.sql.questions import (
    DEFAULT_SKELETON_CAP,
    generate_tasks,
    stress_families,
)

NANO = Scale(name="nano", train=16, dev=8, heldout=12, stress=6)


_FAST_DIRS: list[Path] = []


def fast_dir(tmp_path: Path, name: str) -> Path:
    """Store dir on tmpfs when available: every LLM call commits to SQLite (fsync), which makes
    full pipeline runs several times slower on a disk. Falls back to tmp_path."""
    shm = Path("/dev/shm")  # noqa: S108
    if shm.is_dir() and os.access(shm, os.W_OK):
        d = Path(tempfile.mkdtemp(prefix="distillery-test-", dir=shm))
        _FAST_DIRS.append(d)
        return d / name
    return tmp_path / name


@pytest.fixture(autouse=True, scope="module")
def _cleanup_fast_dirs() -> Iterator[None]:
    yield
    while _FAST_DIRS:
        shutil.rmtree(_FAST_DIRS.pop(), ignore_errors=True)


class Boom(RuntimeError):
    pass


@pytest.fixture
def bridge() -> Iterator[AsyncBridge]:
    with AsyncBridge() as b:
        yield b


def make(
    tmp_path: Path,
    bridge: AsyncBridge,
    *,
    run_id: str = "dry-t",
    scale: Scale = NANO,
    dry: DryRun | None = None,
    **kw: Any,
) -> tuple[Pipeline, DryRun, Store]:
    store = Store(fast_dir(tmp_path, "store"))
    dr = dry or build_dry_run(scale, bridge, **kw)
    pipe = Pipeline(dr.pipeline_cfg, dr.config, dr.deps, store, run_id, say=lambda _s: None)
    return pipe, dr, store


@pytest.fixture(scope="module")
def reference(tmp_path_factory: pytest.TempPathFactory) -> Iterator[dict[str, Any]]:
    tmp = tmp_path_factory.mktemp("ref")
    with AsyncBridge() as bridge:
        pipe, dr, store = make(tmp, bridge)
        report = pipe.run()
        yield {"pipe": pipe, "dr": dr, "store": store, "report": report}
        store.close()


# ---- end to end ---------------------------------------------------------------------------------


def test_dry_run_report_has_all_required_fields(reference: dict[str, Any]) -> None:
    rep, pipe, store = reference["report"], reference["pipe"], reference["store"]
    assert rep["label"] == DRY_RUN_LABEL and rep["dry_run"] is True
    assert rep["decision"] in {"PROMOTE", "REJECT"} and rep["decision_reasons"]
    for key in (
        "evaluation", "rounds", "counters", "cost", "config", "stages", "data", "headroom",
        "llm_errors_by_purpose", "llm_attempt_counters", "sandbox_lineage", "candidate_selection",
    ):  # fmt: skip
        assert key in rep, key
    cfg = rep["config"]
    assert cfg["seeds"]["task_seed"] == 1234 and "bootstrap_seed" in cfg["seeds"]
    assert cfg["model_ids"]["planner"] == "fake-planner"
    assert cfg["gate_thresholds"]["ratio_lower_bound_min"] == 0.85
    assert rep["evaluation"]["gate"]["decision"] == rep["decision"]
    cost = rep["cost"]
    assert set(cost["llm_by_model"]) >= {"fake-planner", "fake-teacher"}
    assert "fake-student-base" not in cost["llm_by_model"]  # base goes through base_factory
    assert cost["cost_per_1k_tasks"]["student"] == STUDENT_COST_UNAVAILABLE
    t = cost["cost_per_1k_tasks"]["teacher"]
    assert t["input_tokens"] > 0 and t["usd_per_1k_tasks"] > 0
    assert (pipe.run_dir / "report.json").exists()
    manifest = json.loads((pipe.run_dir / "manifest.json").read_text())
    assert manifest["dry_run"] is True and manifest["label"] == DRY_RUN_LABEL
    assert manifest["heldout_sha256"] == rep["data"]["heldout_sealed_sha256"]
    spot = json.loads((pipe.run_dir / "spot_check.json").read_text())["items"]
    assert len(spot) == min(30, NANO.heldout)
    assert store.total_spend("dry-t") >= 0


def test_counters_are_nonnegative_ints_and_cover_every_drop(reference: dict[str, Any]) -> None:
    counters = reference["report"]["counters"]
    for stage, cs in counters.items():
        for key, val in cs.items():
            assert isinstance(val, int) and val >= 0, (stage, key, val)
    assert {"dropped_error", "dropped_empty", "dropped_duplicate"} <= set(counters["questions"])
    cc = counters["gold_crosscheck"]
    assert cc["checked"] == cc["accepted"] + cc["discarded"]
    assert cc["discarded"] == sum(v for k, v in cc.items() if k.startswith("discard_"))
    td = counters["teacher_data"]
    assert td["tasks"] == td["verified"] + td["dropped"]
    assert td["triage_calls"] > 0  # triage use is counted
    assert reference["report"]["llm_attempt_counters"]
    assert reference["dr"].transport.calls


def test_heldout_sealed_and_sized(reference: dict[str, Any]) -> None:
    rep, store = reference["report"], reference["store"]
    items = store.load_heldout("dry-t")
    assert len(items) == NANO.heldout == rep["evaluation"]["n"]
    assert rep["data"]["train_tasks"] == NANO.train and rep["data"]["dev_tasks"] == NANO.dev
    # the gate set is purely in-distribution: every family in it is also trained on
    assert {i["heldout_class"] for i in items} == {"in_distribution"}
    split = reference["pipe"]._results["split"]
    train_fams = {d["family"] for d in split["train"]}
    assert {i["family"] for i in items} <= train_fams
    assert rep["data"]["heldout_class_counts"] == {"in_distribution": NANO.heldout}
    assert split["heldout_families"] == []
    # never a gate question that was in train (questions are unique across the pool)
    train_q = {d["question"] for d in split["train"]}
    assert not {i["question"] for i in items} & train_q
    assert 0.0 <= rep["data"]["heldout_skeleton_overlap_rate"] <= 1.0  # measured, reported


def test_stress_set_sealed_separately_and_reported_not_gated(reference: dict[str, Any]) -> None:
    rep, store, pipe = reference["report"], reference["store"], reference["pipe"]
    split = pipe._results["split"]
    stress = store.load_stress("dry-t")
    assert stress is not None and len(stress) == NANO.stress == rep["data"]["stress_tasks"]
    assert {i["heldout_class"] for i in stress} == {"stress"}
    fams = {i["family"] for i in stress}
    gate_fams = {i["family"] for i in store.load_heldout("dry-t")}
    train_fams = {d["family"] for d in (*split["train"], *split["dev"])}
    assert fams == set(rep["data"]["stress_families"])
    assert not fams & (train_fams | gate_fams)  # reserved families appear nowhere else
    assert {i["task_id"] for i in stress}.isdisjoint(split["heldout_task_ids"])
    ev = rep["evaluation"]
    assert ev["stress"]["n"] == NANO.stress and set(ev["stress"]["accuracy"]) == {
        "base", "student", "teacher"
    }  # fmt: skip
    # the gate is computed from the gate set alone
    assert ev["n"] == NANO.heldout and len(ev["gate"]["reasons"]) >= 0
    assert rep["data"]["stress_sealed_sha256"] == split["stress_sealed_sha256"]


def test_failure_analysis_prompts_contain_dev_items_only(reference: dict[str, Any]) -> None:
    dr: DryRun = reference["dr"]
    heldout = reference["store"].load_heldout("dry-t")
    dev = reference["pipe"]._results["split"]["dev"]
    analysis = [c for c in dr.transport.calls if c["schema"] == "FailureClusters"]
    assert analysis, "the fake student is weak enough that a failure analysis must have run"
    dev_q = {d["question"] for d in dev}
    for call in analysis:
        assert call["model"] == "fake-planner"
        text = "\n".join(m["content"] for m in call["messages"])
        for it in heldout:
            assert it["question"] not in text and it["gold_sql"] not in text
            allowed = re.findall(r"Allowed families: (.*)", text)[0].split(", ")
            for fam in reference["report"]["data"]["stress_families"]:
                assert fam not in allowed  # reserved stress families are never offered
        asked = re.findall(r"\] Q: (.*)", text)
        assert asked and set(asked) <= dev_q


def test_no_model_touches_gold_and_planner_never_sees_heldout(reference: dict[str, Any]) -> None:
    """Gold comes from templates: no model answers a question to vet it (the Ultra/Super
    agreement filter is gone), and the planner only ever sees dev failures, never held-out."""
    heldout = {i["question"] for i in reference["store"].load_heldout("dry-t")}
    planner = [c for c in reference["dr"].transport.calls if c["model"] == "fake-planner"]
    assert planner and all(c["schema"] == "FailureClusters" for c in planner)  # analysis only
    for c in planner:
        assert not any(q in m["content"] for m in c["messages"] for q in heldout)
    cc = reference["report"]["counters"]["gold_crosscheck"]
    assert cc["accepted"] + cc["discarded"] == cc["checked"]
    assert not any(k.startswith("discard_") and "planner" in k for k in cc)


def test_evaluated_artifact_is_trained_artifact(reference: dict[str, Any]) -> None:
    rep, pipe = reference["report"], reference["pipe"]
    r = rep["candidate_round"]
    trained = pipe._results[f"finetune_r{r}"]["artifact"]
    ev = rep["evaluation"]["artifact"]
    assert ev["adapter_sha256"] == trained["adapter_sha256"]
    assert ev["job_id"] == trained["job_id"] and ev["checkpoint_id"] == trained["checkpoint_id"]
    names = [n for n, _ in reference["store"].list_experiments("dry-t")]
    assert names.count("expected_artifact") == len(rep["rounds"])
    assert (
        rep["candidate_round"]
        == max(rep["rounds"], key=lambda x: (x["dev_acc"], -x["round"]))["round"]
    )


def test_rounds_record_lineage_and_stay_within_max(reference: dict[str, Any]) -> None:
    rep = reference["report"]
    assert 1 <= len(rep["rounds"]) <= 3
    lin = rep["sandbox_lineage"]
    assert len(lin) == len(rep["rounds"]) - 1 >= 1
    assert lin[0]["parent"] == "fake-base-image" and lin[0]["label"] == "round-1"
    assert all(k in lin[0] for k in ("uuid", "parent", "label"))


def test_dry_run_flag_and_prefix_enforced(tmp_path: Path, bridge: AsyncBridge) -> None:
    pipe, _dr, _s = make(tmp_path, bridge, run_id="sql-real-looking")
    with pytest.raises(ConfigRefusal, match="dry"):
        pipe.run()


# ---- resumability -------------------------------------------------------------------------------


def test_resume_after_mid_stage_kill_skips_done_stages_and_matches_hashes(
    tmp_path: Path, bridge: AsyncBridge, reference: dict[str, Any]
) -> None:
    def kill(stage: str) -> None:
        if stage == "teacher_data":
            raise Boom("injected")

    pipe1, dr1, store = make(tmp_path, bridge, on_stage=kill)
    with pytest.raises(Boom):
        pipe1.run()
    assert pipe1.ran[-1] == "teacher_data"
    done = set(pipe1.ran) - {"teacher_data"}
    assert {"schema", "questions", "gold_crosscheck", "split", "headroom"} <= done
    status = {s[0]: s[1] for s in _stages(store)}
    assert status["teacher_data"] == "failed" and status["headroom"] == "complete"

    dr2 = build_dry_run(NANO, bridge)
    pipe2 = Pipeline(dr2.pipeline_cfg, dr2.config, dr2.deps, store, "dry-t", say=lambda _s: None)
    report = pipe2.run()
    assert not done & set(pipe2.ran), "completed stages must be skipped"
    assert "teacher_data" in pipe2.ran
    # no planner call at all in the resumed process except failure analysis
    assert all(c["schema"] == "FailureClusters" for c in dr2.transport.calls
               if c["model"] == "fake-planner")  # fmt: skip
    for stage, digest in pipe2.hashes.items():
        assert digest == reference["pipe"].hashes[stage], stage
    assert report["decision"] == reference["report"]["decision"]
    assert report["evaluation"] == reference["report"]["evaluation"]


def test_rerun_of_finished_run_runs_nothing(tmp_path: Path, bridge: AsyncBridge) -> None:
    pipe, dr, store = make(tmp_path, bridge)
    first = pipe.run()
    n_calls = len(dr.transport.calls)
    dr2 = build_dry_run(NANO, bridge)
    pipe2 = Pipeline(dr2.pipeline_cfg, dr2.config, dr2.deps, store, "dry-t", say=lambda _s: None)
    second = pipe2.run()
    assert pipe2.ran == [] and dr2.transport.calls == [] and n_calls > 0
    assert second["stages"] == first["stages"]
    assert second["cost"]["llm_by_model"] == first["cost"]["llm_by_model"]  # not double counted


def _stages(store: Store) -> list[tuple[str, str]]:
    from distillery.orchestrator import stage_rows

    return [(s["stage"], s["status"]) for s in stage_rows(store, "dry-t")]


# ---- fine-tune safety ---------------------------------------------------------------------------


def test_failed_finetune_job_is_cancelled(tmp_path: Path, bridge: AsyncBridge) -> None:
    pipe, dr, store = make(tmp_path, bridge, fail_rounds=frozenset({1}))
    with pytest.raises(JobFailedError):
        pipe.run()
    assert len(dr.finetune.created) == 1
    assert dr.finetune.cancelled == dr.finetune.created
    closed = [d for n, d in store.list_experiments("dry-t") if n == "finetune_job_closed"]
    assert closed and closed[0]["outcome"] == "aborted"
    assert not [n for n, _ in store.list_experiments("dry-t") if n == "expected_artifact"]


def test_interrupt_during_finetune_cancels_job(tmp_path: Path, bridge: AsyncBridge) -> None:
    pipe, dr, _s = make(tmp_path, bridge)

    def interrupt(*_a: Any, **_k: Any) -> Any:
        raise KeyboardInterrupt

    dr.finetune.checkpoints = interrupt  # type: ignore[method-assign]
    with pytest.raises(KeyboardInterrupt):
        pipe.run()
    assert dr.finetune.cancelled == dr.finetune.created and len(dr.finetune.created) == 1


def test_orphaned_job_from_killed_process_is_cancelled_on_resume(
    tmp_path: Path, bridge: AsyncBridge
) -> None:
    pipe, dr, store = make(tmp_path, bridge)
    store.add_experiment("dry-t", "finetune_job_started", {"round": 1, "job_id": "ftjob-ghost"})
    pipe.run()
    assert "ftjob-ghost" in dr.finetune.cancelled
    assert pipe.counters["finetune_r1"]["orphan_jobs_cancelled"] == 1


def test_budget_exceeded_refuses_before_any_llm_call(tmp_path: Path, bridge: AsyncBridge) -> None:
    pipe, dr, _s = make(tmp_path, bridge, run_cap_usd=1e-9)
    with pytest.raises(BudgetExceeded):
        pipe.run()
    assert dr.transport.calls == []


def _live_like(tmp_path: Path, bridge: AsyncBridge, **pcfg: Any) -> tuple[Pipeline, DryRun]:
    """A dry-run rig whose pipeline config is NOT dry, to exercise the live-only preflight."""
    dr = build_dry_run(NANO, bridge, **pcfg)
    cfg = dr.pipeline_cfg.model_copy(update={"dry_run": False})
    store = Store(fast_dir(tmp_path, "store"))
    return Pipeline(cfg, dr.config, dr.deps, store, "t", say=lambda _s: None), dr


def test_min_planned_steps_refuses_before_any_upload(tmp_path: Path, bridge: AsyncBridge) -> None:
    pipe, dr = _live_like(tmp_path, bridge)  # 16 rows / batch 16 x 3 epochs = 3 < 50
    with pytest.raises(ConfigRefusal, match="planned optimizer steps"):
        pipe._finetune_preflight(16)
    assert dr.finetune.uploads == {} and dr.finetune.created == []


def test_min_planned_steps_accepts_enough_rows(tmp_path: Path, bridge: AsyncBridge) -> None:
    pipe, _dr = _live_like(tmp_path, bridge)
    pipe._finetune_preflight(16 * 17)  # 17 x 3 = 51 steps
    with pytest.raises(ConfigRefusal):
        pipe._finetune_preflight(16 * 16)  # 48


def test_packing_refused_unless_allowed(tmp_path: Path, bridge: AsyncBridge) -> None:
    hp = PINNED_HYPERPARAMETERS.model_copy(update={"packing": True})
    pipe, _dr = _live_like(tmp_path, bridge, hyperparameters=hp)
    with pytest.raises(ConfigRefusal, match="packing"):
        pipe._finetune_preflight(10_000)
    pipe.cfg = pipe.cfg.model_copy(update={"allow_packing": True})
    pipe._finetune_preflight(10_000)


def test_default_style_hyperparameters_refused_even_in_dry_run(
    tmp_path: Path, bridge: AsyncBridge
) -> None:
    pipe, _dr, _s = make(tmp_path, bridge, hyperparameters=HyperParameters(lora=True, n_epochs=3))
    with pytest.raises(ConfigRefusal, match="explicit"):
        pipe.run()


def test_report_has_finetune_section(reference: dict[str, Any]) -> None:
    rep = reference["report"]
    assert len(rep["finetune"]) == len(rep["rounds"])
    rec = rep["finetune"][0]
    assert rec["round"] == 1 and rec["job_id"] == rep["rounds"][0]["job_id"]
    assert rec["hyperparameters"]["batch_size"] == 16 and rec["hyperparameters"]["packing"] is False
    assert rec["trained_steps"] and rec["trained_tokens"] and rec["loss_curve"] and rec["events"]
    assert rep["config"]["pipeline"]["hyperparameters"]["learning_rate"] == 1e-4


def test_finetune_estimate_over_cap_refuses_before_upload(
    tmp_path: Path, bridge: AsyncBridge
) -> None:
    pipe, dr, store = make(tmp_path, bridge, finetune_estimate_usd=100.0)
    with pytest.raises(BudgetExceeded):
        pipe.run()
    assert dr.finetune.uploads == {} and dr.finetune.created == []
    assert store.total_spend("dry-t") < 100.0


def test_evaluating_a_tampered_artifact_fails(tmp_path: Path, bridge: AsyncBridge) -> None:
    holder: dict[str, Pipeline] = {}

    def tamper(stage: str) -> None:
        if stage == "final_eval":
            for name, res in holder["p"]._results.items():
                if name.startswith("finetune_r"):
                    (holder["p"].run_dir / res["artifact"]["files"][1]["path"]).write_text(
                        "swapped weights"
                    )

    pipe, _dr, _s = make(tmp_path, bridge, on_stage=tamper)
    holder["p"] = pipe
    with pytest.raises(ArtifactMismatchError):
        pipe.run()


# ---- loud failures ------------------------------------------------------------------------------


def test_task_too_easy_fails_loudly(tmp_path: Path, bridge: AsyncBridge) -> None:
    pipe, _dr, _s = make(tmp_path, bridge, error_rates={"fake-student-base": 0.0})
    with pytest.raises(TaskTooEasyError, match="task too easy"):
        pipe.run()


def test_verifier_selftest_fails_if_a_corruption_is_accepted(
    tmp_path: Path, bridge: AsyncBridge
) -> None:
    class Liar(LocalExecutor):
        lie = False

        def run_batch(self, db_ref: str, sqls: Any) -> Any:
            out = super().run_batch(db_ref, sqls)
            return [out[0]] * len(out) if self.lie and out else out

    liar = Liar()
    pipe, dr, _s = make(
        tmp_path, bridge, on_stage=lambda s: setattr(liar, "lie", s == "verifier_selftest")
    )
    dr.deps.executor = liar
    with pytest.raises(VerifierSelfTestError, match="ACCEPTED"):
        pipe.run()


def test_serving_path_missing_refuses_before_spending(tmp_path: Path, bridge: AsyncBridge) -> None:
    from distillery.orchestrator import ServingUnavailableError

    pipe, dr, _s = make(tmp_path, bridge)
    dr.deps.student_factory = None
    with pytest.raises(ServingUnavailableError):
        pipe.run()
    assert dr.transport.calls == []


def test_base_serving_missing_refuses_before_spending(tmp_path: Path, bridge: AsyncBridge) -> None:
    from distillery.orchestrator import ServingUnavailableError

    pipe, dr, _s = make(tmp_path, bridge)
    dr.deps.base_factory = None
    with pytest.raises(ServingUnavailableError):
        pipe.run()
    assert dr.transport.calls == []


def test_base_never_goes_through_llm_client_and_is_closed(reference: dict[str, Any]) -> None:
    dr = reference["dr"]
    assert dr.base.servers and all(s.closed and s.calls >= 1 for s in dr.base.servers)
    assert all(c["model"] != "fake-student-base" for c in dr.transport.calls)


def test_report_carries_capped_examples_after_scoring(reference: dict[str, Any]) -> None:
    ex = reference["report"]["evaluation"]["examples"]
    assert 0 < len(ex) <= 60
    assert {"task_id", "gold_sql", "base_sql", "student_sql", "teacher_sql"} <= set(ex[0])


# ---- pure split planning ------------------------------------------------------------------------


def test_plan_split_in_distribution_exact_sizes_and_stress(tmp_path: Path) -> None:
    db = tmp_path / "d.sqlite"
    sql_schema.write_database(0, db)
    reserved = stress_families()
    tasks = generate_tasks(str(db), 120, 7, exclude_families=reserved).tasks
    stress = generate_tasks(str(db), 40, 8, families=reserved).tasks
    for scale in (NANO, SCALES["tiny"]):
        plan = plan_split(tasks, scale, 3, stress)
        assert (len(plan.train), len(plan.dev), len(plan.heldout), len(plan.stress)) == (
            scale.train, scale.dev, scale.heldout, scale.stress,
        )  # fmt: skip
        ids = [t.task_id for t in (*plan.train, *plan.dev, *plan.heldout, *plan.stress)]
        assert len(ids) == len(set(ids))
        assert plan.heldout_families == []  # the gate holds out no family
        assert {t.family for t in plan.heldout} <= {t.family for t in plan.train}
        assert {t.family for t in plan.stress} <= set(reserved)
        assert not {t.family for t in plan.stress} & {t.family for t in plan.train}
        assert plan.counters["heldout_questions_seen_in_train"] == 0
        assert 0 <= plan.counters["heldout_skeleton_in_train"] <= scale.heldout
    with pytest.raises(ShortfallError):
        plan_split(tasks[:30], SCALES["tiny"], 3, stress)
    with pytest.raises(ShortfallError, match="leaked"):  # a stress family in the gate pool
        plan_split([*tasks, stress[0]], SCALES["tiny"], 3, stress)


# ---- integration through the sandbox executor ---------------------------------------------------


def _python_handler(job: Job, fs: dict[str, bytes]) -> FakeExecution:
    if job.command is None:
        return FakeExecution()
    with tempfile.TemporaryDirectory() as tmp:
        script, db = Path(tmp) / "runner.py", Path(tmp) / "db.sqlite"
        script.write_bytes(fs[SCRIPT_PATH])
        db.write_bytes(fs[DB_PATH])
        stdin = str(job.stdin).replace(DB_PATH, str(db))
        proc = subprocess.run(
            [sys.executable, str(script)], input=stdin, capture_output=True, text=True, check=False
        )
    return FakeExecution(proc.stdout, proc.stderr, proc.returncode)


def test_pipeline_through_sandbox_executor_matches_local(
    tmp_path: Path, bridge: AsyncBridge, reference: dict[str, Any]
) -> None:
    dr = build_dry_run(NANO, bridge)
    ex = SandboxExecutor(FakeSandbox(_python_handler), "img", bridge=bridge)
    dr.deps.executor = ex
    store = Store(fast_dir(tmp_path, "s2"))
    pipe = Pipeline(dr.pipeline_cfg, dr.config, dr.deps, store, "dry-t", say=lambda _s: None)
    rep = pipe.run()
    assert ex.stats.infra_failures == 0 and ex.stats.statements > 0
    assert rep["evaluation"] == reference["report"]["evaluation"]
    for stage in ("gold_crosscheck", "verifier_selftest", "split", "teacher_data"):
        assert pipe.hashes[stage] == reference["pipe"].hashes[stage], stage


def test_orchestrator_never_mentions_the_sealed_reader() -> None:
    src = (Path(__file__).parents[1] / "src/distillery/orchestrator.py").read_text()
    assert "load_" + "heldout" not in src
    _: Callable[[], None] = lambda: None  # noqa: E731


# ---- student_serving option (wiring only; real serving is UNVERIFIED) ----------------------------


def test_serving_default_is_injected_and_untouched(tmp_path: Path, bridge: AsyncBridge) -> None:
    pipe, dr, _s = make(tmp_path, bridge)
    assert pipe.cfg.student_serving == "injected"
    pipe._resolve_serving()
    assert pipe.deps is dr.deps and pipe.deps.student_factory is dr.students


def test_serving_sandbox_cpu_builds_factories_without_mutating_deps(
    tmp_path: Path, bridge: AsyncBridge
) -> None:
    from distillery.sandbox_student import SandboxCpuStudent

    dry = build_dry_run(NANO, bridge, student_serving="sandbox_cpu")
    dry.deps.student_factory = dry.deps.base_factory = None
    pipe, dr, _s = make(tmp_path, bridge, dry=dry)
    pipe._resolve_serving()
    assert dr.deps.student_factory is None  # caller's Deps untouched (resumable)
    assert pipe.deps.student_factory is not None and pipe.deps.base_factory is not None
    base = pipe.deps.base_factory()
    assert isinstance(base, SandboxCpuStudent) and base._adapter_files == ()
    assert base._image is None  # nothing built until first generate


def test_serving_modes_refuse_ambiguity_or_missing_config(
    tmp_path: Path, bridge: AsyncBridge
) -> None:
    dry = build_dry_run(NANO, bridge, student_serving="sandbox_cpu")
    pipe, _dr, _s = make(tmp_path, bridge, dry=dry)  # fake factories still set
    with pytest.raises(ConfigRefusal, match="already has serving factories"):
        pipe._resolve_serving()
    dry = build_dry_run(NANO, bridge, student_serving="sandbox_cpu")
    dry.deps.student_factory = dry.deps.base_factory = None
    dry.deps.sandbox = None
    pipe, dr, _s = make(tmp_path, bridge, dry=dry)
    with pytest.raises(ConfigRefusal, match="needs Deps.sandbox"):
        pipe.run()
    assert dr.transport.calls == []


# ---- final-eval server cleanup: every (paid) server is closed even when a close raises ---------


class _Srv:
    def __init__(self, name: str, log: list[str], close_error: bool = False) -> None:
        self.name, self.log, self.close_error = name, log, close_error

    def generate(self, batch: Any) -> list[str]:
        raise AssertionError("not reached")

    def close(self) -> None:
        self.log.append(f"close:{self.name}")
        if self.close_error:
            raise RuntimeError(f"{self.name} delete failed")


def _run_final_eval_with(
    tmp_path: Path, bridge: AsyncBridge, *, student: Any, base: Any
) -> tuple[list[str], BaseException | None]:
    state = {"final": False}
    pipe, dr, _s = make(
        tmp_path, bridge, on_stage=lambda st: state.update(final=st == "final_eval")
    )
    log: list[str] = []
    real_student, real_base = dr.deps.student_factory, dr.deps.base_factory
    assert real_student is not None and real_base is not None
    dr.deps.student_factory = lambda t: student(log) if state["final"] else real_student(t)
    dr.deps.base_factory = lambda: base(log) if state["final"] else real_base()
    err: BaseException | None = None
    try:
        pipe.run()
    except BaseException as e:  # noqa: BLE001
        err = e
    return log, err


def test_student_close_error_does_not_leak_base_server(tmp_path: Path, bridge: AsyncBridge) -> None:
    log, err = _run_final_eval_with(
        tmp_path, bridge,
        student=lambda lg: _Srv("student", lg, close_error=True),
        base=lambda lg: _Srv("base", lg),
    )  # fmt: skip
    # generation fails first (stub raises); both closes must still run, in either case
    assert "close:student" in log and "close:base" in log
    assert err is not None


def test_close_error_is_raised_after_all_servers_closed(
    tmp_path: Path, bridge: AsyncBridge
) -> None:
    from distillery.orchestrator import _close_all

    log: list[str] = []
    servers = [_Srv("a", log, True), _Srv("b", log), _Srv("c", log, True)]
    with pytest.raises(RuntimeError, match="a delete failed"):
        _close_all(servers)
    assert log == ["close:a", "close:b", "close:c"]  # first error raised, none skipped


def test_base_factory_failure_closes_the_student_server(
    tmp_path: Path, bridge: AsyncBridge
) -> None:
    def boom(_lg: list[str]) -> Any:
        raise RuntimeError("base create failed")

    log, err = _run_final_eval_with(
        tmp_path, bridge, student=lambda lg: _Srv("student", lg), base=boom
    )
    assert log == ["close:student"] and isinstance(err, RuntimeError)
    assert "base create failed" in str(err)


def test_base_factory_failure_and_student_close_failure_both_surface(
    tmp_path: Path, bridge: AsyncBridge
) -> None:
    def boom(_lg: list[str]) -> Any:
        raise RuntimeError("base create failed")

    log, err = _run_final_eval_with(
        tmp_path, bridge, student=lambda lg: _Srv("student", lg, True), base=boom
    )
    assert log == ["close:student"]
    assert isinstance(err, RuntimeError) and "base create failed" in str(err)


def _trained_art(base: str | None, ckpt: str | None) -> Any:
    from distillery.finetune import TrainedArtifact

    return TrainedArtifact(
        job_id="j", checkpoint_id="c", base_model=base, fine_tuned_model_checkpoint=ckpt, files=()
    )


def test_report_artifact_records_trained_model_names() -> None:
    from distillery.orchestrator import _with_serving

    out = _with_serving({"job_id": "j"}, _trained_art("base/x", "ft-ckpt"))
    assert out == {
        "job_id": "j",
        "trained_base_model": "base/x",
        "trained_checkpoint_name": "ft-ckpt",
    }


# ---- live sandbox-CPU serving wiring (offline; FakeSandbox stands in for Nebius) -----------------


def _gen_handler(counts: dict[str, int]) -> Callable[[Job, dict[str, bytes]], FakeExecution]:
    def handler(job: Job, fs: dict[str, bytes]) -> FakeExecution:
        shell = job.shell or ""
        if "pip install" in shell:
            counts["pip"] = counts.get("pip", 0) + 1
            return FakeExecution()
        if "gen.py" in shell:
            n = len(json.loads(str(job.stdin))["messages_batch"])
            res = {"results": [{"text": "SELECT 1", "error": None} for _ in range(n)], "n": n}
            return FakeExecution("DISTILLERY_OUT:" + json.dumps(res) + "\n")
        counts["layer"] = counts.get("layer", 0) + 1
        return FakeExecution()

    return handler


def _sandbox_cpu_pipe(
    tmp_path: Path, bridge: AsyncBridge, counts: dict[str, int], **cfg_kw: Any
) -> tuple[Pipeline, Store]:
    dry = build_dry_run(NANO, bridge, student_serving="sandbox_cpu", **cfg_kw)
    dry.deps.student_factory = dry.deps.base_factory = None
    dry.deps.sandbox = FakeSandbox(_gen_handler(counts))
    pipe, _dr, store = make(tmp_path, bridge, dry=dry)
    return pipe, store


def test_sandbox_cpu_serving_builds_the_heavy_image_once_for_base_and_student(
    tmp_path: Path, bridge: AsyncBridge
) -> None:
    from distillery.finetune import DownloadedFile, TrainedArtifact

    counts: dict[str, int] = {}
    pipe, store = _sandbox_cpu_pipe(tmp_path, bridge, counts)
    pipe._resolve_serving()
    store.create_run(pipe.run_id)
    student_model = pipe.config.require_model("student")
    d = tmp_path / "ckpt"
    d.mkdir()
    (d / "adapter_config.json").write_text(json.dumps({"base_model_name_or_path": student_model}))
    (d / "adapter_model.safetensors").write_bytes(b"w")
    trained = TrainedArtifact(
        "j", "c", student_model, None,
        tuple(DownloadedFile(p.name, p, "0" * 64) for p in sorted(d.iterdir())),
    )  # fmt: skip
    assert pipe.deps.base_factory is not None and pipe.deps.student_factory is not None
    for _ in range(2):  # headroom, then final_eval, each builds its own server objects
        assert pipe.deps.base_factory().generate([[{"role": "user", "content": "q"}]]) == [
            "SELECT 1"
        ]
        assert pipe.deps.student_factory(trained).generate([[{"role": "user", "content": "q"}]])
    assert counts == {"pip": 1, "layer": 1}  # one pip+weights build, one adapter layer
    built = [d for n, d in store.list_experiments(pipe.run_id) if n == "serving_image"]
    assert [b["kind"] for b in built] == ["deps", "adapter"] and all(b["image"] for b in built)


def test_sandbox_cpu_defaults_are_the_measured_recipe() -> None:
    from distillery.orchestrator import PipelineConfig

    cfg = PipelineConfig()
    assert cfg.student_max_new_tokens == 160 and cfg.student_batch_size <= 4
    assert 1 <= cfg.student_concurrency <= 20
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        PipelineConfig(student_concurrency=21)


class _WithErrors:
    """StudentServer wrapper reporting sandbox generation errors like SandboxCpuStudent does."""

    def __init__(self, inner: Any, errors: int) -> None:
        self._inner, self.generation_errors = inner, errors

    def generate(self, batch: Any) -> list[str]:
        return list(self._inner.generate(batch))

    def close(self) -> None:
        self._inner.close()


def test_generation_errors_are_counted_per_stage_in_the_report(
    tmp_path: Path, bridge: AsyncBridge
) -> None:
    pipe, dr, _s = make(tmp_path, bridge)
    real_student, real_base = dr.deps.student_factory, dr.deps.base_factory
    assert real_student is not None and real_base is not None
    dr.deps.student_factory = lambda t: _WithErrors(real_student(t), 2)
    dr.deps.base_factory = lambda: _WithErrors(real_base(), 1)
    counters = pipe.run()["counters"]
    assert counters["headroom"]["base_generation_errors"] == 1
    assert counters["dev_eval_r1"]["student_generation_errors"] == 2
    assert counters["final_eval"]["student_generation_errors"] == 2
    assert counters["final_eval"]["base_generation_errors"] == 1


def test_servers_without_generation_errors_add_no_counter_keys(
    reference: dict[str, Any],
) -> None:
    for cs in reference["report"]["counters"].values():
        assert not [k for k in cs if k.endswith("generation_errors")]


def test_report_cost_says_it_is_an_estimate_and_names_the_price_source(
    reference: dict[str, Any],
) -> None:
    cost = reference["report"]["cost"]
    assert "ESTIMATE" in cost["basis"] and "FAKE dry-run price" in cost["basis"]
    assert cost["price_sources"]["fake-planner"] == "FAKE dry-run price, not a real price"
    assert "unavailable" in STUDENT_COST_UNAVAILABLE and "sandbox" in STUDENT_COST_UNAVAILABLE


# ---- a job left ambiguous by a failed cancel is adopted on resume, never paid for twice ----------


def _outage_first_attempt(tmp_path: Path, bridge: AsyncBridge, **kw: Any) -> tuple[Store, str]:
    """Run 1: polling dies with a connection error and the cancel fails too (the job lives on)."""
    import httpx2
    import openai

    pipe, dr, store = make(tmp_path, bridge, **kw)
    down = openai.APIConnectionError(request=httpx2.Request("GET", "https://x.invalid"))

    def poll_dies(*_a: Any, **_k: Any) -> Any:
        raise down

    def cancel_dies(_job_id: str) -> Any:
        raise down

    dr.finetune.poll = poll_dies  # type: ignore[method-assign]
    dr.finetune.cancel = cancel_dies  # type: ignore[method-assign]
    with pytest.raises(openai.APIConnectionError):
        pipe.run()
    (jid,) = dr.finetune.created
    closed = [d for n, d in store.list_experiments("dry-t") if n == "finetune_job_closed"]
    assert closed[-1]["outcome"] == "aborted"
    return store, jid


def _resume(store: Store, bridge: AsyncBridge, **kw: Any) -> tuple[Pipeline, Any, dict[str, Any]]:
    dr = build_dry_run(NANO, bridge, **kw)
    pipe = Pipeline(dr.pipeline_cfg, dr.config, dr.deps, store, "dry-t", say=lambda _s: None)
    return pipe, dr, pipe.run()


@pytest.mark.parametrize("status", ["succeeded", "running"])
def test_resume_adopts_the_job_instead_of_creating_a_second_one(
    tmp_path: Path, bridge: AsyncBridge, status: str
) -> None:
    from distillery.orchestrator import cost_by_model

    store, jid = _outage_first_attempt(tmp_path, bridge, finetune_estimate_usd=1.5)
    dr2 = build_dry_run(NANO, bridge, finetune_estimate_usd=1.5)
    dr2.finetune.job_status[jid] = status
    pipe2 = Pipeline(dr2.pipeline_cfg, dr2.config, dr2.deps, store, "dry-t", say=lambda _s: None)
    pipe2.run()
    assert not [s for s in dr2.finetune.suffixes.values() if s.endswith("-r1")]  # no new r1 job
    assert not [
        p for p in dr2.finetune.uploads.values() if "round1" in str(p)
    ]  # nothing re-uploaded
    adopted = [
        d["job_id"] for n, d in store.list_experiments("dry-t") if n == "finetune_job_adopted"
    ]
    assert adopted == [jid]
    expected = [d for n, d in store.list_experiments("dry-t") if n == "expected_artifact"]
    assert expected[0]["job_id"] == jid
    llm_usd = sum(m["usd"] for m in cost_by_model(store, "dry-t").values())
    finetune_usd = store.total_spend("dry-t") - llm_usd
    assert finetune_usd == pytest.approx(1.5 * len(expected))  # once per completed round


@pytest.mark.parametrize("status", ["cancelled", "failed"])
def test_resume_does_not_adopt_a_dead_job(tmp_path: Path, bridge: AsyncBridge, status: str) -> None:
    store, jid = _outage_first_attempt(tmp_path, bridge)
    dr2 = build_dry_run(NANO, bridge)
    dr2.finetune.job_status[jid] = status
    pipe2 = Pipeline(dr2.pipeline_cfg, dr2.config, dr2.deps, store, "dry-t", say=lambda _s: None)
    pipe2.run()
    assert dr2.finetune.created  # a fresh job was created for round 1
    assert not [d for n, d in store.list_experiments("dry-t") if n == "finetune_job_adopted"]


# ---- gated scale and teacher cost denominator ---------------------------------------------------


def test_gated_scale_is_pinned_and_reachable_by_plan_split_arithmetic(tmp_path: Path) -> None:
    sc = SCALES["gated"]
    assert (sc.train, sc.dev, sc.heldout, sc.stress) == (650, 150, 300, 100)
    assert (SCALES["full"].train, SCALES["mini"].train) == (1800, 120)  # others untouched
    oversample = PipelineConfig().oversample
    assert oversample == 1.1
    n_total = math.ceil((sc.train + sc.dev + sc.heldout) * oversample)
    n_stress = math.ceil(sc.stress * oversample)
    assert (n_total, n_stress) == (1210, 111)  # ceil(100 x 1.1) is 111 in floating point
    db = tmp_path / "d.sqlite"
    sql_schema.write_database(0, db)
    reserved = stress_families()
    seed = PipelineConfig().seed  # same generation arguments as Pipeline._stage_questions
    tasks = generate_tasks(
        str(db), n_total, seed, exclude_families=reserved, skeleton_cap=DEFAULT_SKELETON_CAP
    ).tasks
    stress = generate_tasks(
        str(db), n_stress, seed + 7, families=reserved, template_share=0.1, family_share=1.0
    ).tasks
    plan = plan_split(tasks, sc, seed, stress)
    assert (len(plan.train), len(plan.dev), len(plan.heldout), len(plan.stress)) == (
        650, 150, 300, 100,
    )  # fmt: skip


def test_dry_run_at_gated_stand_in_scale_and_teacher_cost_denominator(
    tmp_path: Path, bridge: AsyncBridge
) -> None:
    stand_in = Scale(name="gated-mini", train=26, dev=6, heldout=12, stress=6)
    pipe, _dr, store = make(tmp_path, bridge, scale=stand_in)
    rep = pipe.run()
    store.close()
    t = rep["cost"]["cost_per_1k_tasks"]["teacher"]
    assert t["items_scored_breakdown"] == {"gate": 12, "stress": 6}
    assert t["items_scored"] == 18
    assert t["usd_per_1k_tasks"] == pytest.approx(t["usd"] / 18 * 1000)
    assert "18 items" in t["usd_per_1k_tasks_basis"]
