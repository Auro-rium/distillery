# ruff: noqa: S101, S603
from __future__ import annotations

import json
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
from distillery.finetune import JobFailedError
from distillery.orchestrator import (
    DRY_RUN_LABEL,
    SCALES,
    STUDENT_COST_UNAVAILABLE,
    ConfigRefusal,
    Pipeline,
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
from distillery.taskpacks.sql.questions import generate_tasks

NANO = Scale(name="nano", train=16, dev=8, heldout=12)


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
    assert set(cost["llm_by_model"]) >= {"fake-planner", "fake-teacher", "fake-student-base"}
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
    fams = {i["family"] for i in items if i["heldout_class"] == "unseen_family"}
    train_fams = {d["family"] for d in reference["pipe"]._results["split"]["train"]}
    assert fams and not fams & train_fams
    unseen = sum(rep["data"]["heldout_class_counts"].get(k, 0) for k in ("unseen_family",))
    assert unseen / NANO.heldout >= 0.2


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
            if it["heldout_class"] == "unseen_family":  # held-out families are never offered
                allowed = re.findall(r"Allowed families: (.*)", text)[0].split(", ")
                assert it["family"] not in allowed
        asked = re.findall(r"\] Q: (.*)", text)
        assert asked and set(asked) <= dev_q


def test_planner_never_sees_heldout_after_crosscheck(reference: dict[str, Any]) -> None:
    """Every non-crosscheck planner prompt (analysis) is held-out free (see DECISIONS)."""
    heldout = {i["question"] for i in reference["store"].load_heldout("dry-t")}
    planner = [c for c in reference["dr"].transport.calls if c["model"] == "fake-planner"]
    crosscheck = [c for c in planner if c["schema"] == "SqlAnswer"]
    other = [c for c in planner if c["schema"] != "SqlAnswer"]
    assert crosscheck and other
    for c in other:
        assert not any(q in m["content"] for m in c["messages"] for q in heldout)


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
    assert not [c for c in dr2.transport.calls if c["schema"] == "SqlAnswer"
                and c["model"] == "fake-planner"]  # fmt: skip
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


def test_ultra_verifier_hook_is_documented_not_silent(tmp_path: Path, bridge: AsyncBridge) -> None:
    pipe, _dr, _s = make(tmp_path, bridge, ultra_verifier_authoring=True)
    with pytest.raises(NotImplementedError, match="hook"):
        pipe.run()


# ---- pure split planning ------------------------------------------------------------------------


def test_plan_split_exact_sizes_and_unseen_families(tmp_path: Path) -> None:
    db = tmp_path / "d.sqlite"
    sql_schema.write_database(0, db)
    tasks = generate_tasks(str(db), 120, 7).tasks
    for scale in (NANO, SCALES["tiny"]):
        plan = plan_split(tasks, scale, 3)
        assert (len(plan.train), len(plan.dev), len(plan.heldout)) == (
            scale.train, scale.dev, scale.heldout,
        )  # fmt: skip
        ids = [t.task_id for t in (*plan.train, *plan.dev, *plan.heldout)]
        assert len(ids) == len(set(ids))
        train_fams = {t.family for t in (*plan.train, *plan.dev)}
        assert not train_fams & set(plan.heldout_families)
        unseen = [t for t in plan.heldout if t.family in plan.heldout_families]
        assert len(unseen) / scale.heldout >= 0.2
    with pytest.raises(ShortfallError):
        plan_split(tasks[:30], SCALES["tiny"], 3)


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
