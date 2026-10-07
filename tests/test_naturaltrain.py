# ruff: noqa: S101
"""N1 natural training rows (DECISIONS.md 2026-10-07): the keep rule, the draft/check script, and
the pipeline hook (overlap drop, cache isolation, identity, report). Offline, fakes only."""

from __future__ import annotations

import importlib.util
import json
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from distillery import naturaltrain as nt
from distillery.cli import EXIT_OK, EXIT_REFUSED
from distillery.cli import main as cli_main
from distillery.orchestrator import ConfigRefusal, Pipeline, Scale
from distillery.pipeline_fakes import build_dry_run, question_of
from distillery.prompts import to_training_row
from distillery.sandbox_executor import AsyncBridge
from distillery.store import Store, sha256_hex
from distillery.taskpacks.sql import schema as sql_schema
from distillery.taskpacks.sql.executor import LocalExecutor
from distillery.taskpacks.sql.human import load_question_file
from tests.test_cli_human import ScriptedTransport, factory_for
from tests.test_cli_human import live_env as live_env  # noqa: F401 - pytest fixture

NANO = Scale(name="nano", train=16, dev=8, heldout=12, stress=6)
DB = sql_schema.build_database(0)
DB_SHA = sha256_hex(DB)
DDL = sql_schema.schema_ddl(0)
COUNT_A = "SELECT COUNT(*) FROM accounts"
COUNT_U = "SELECT COUNT(*) FROM users"
EMPTY = "SELECT plan_id FROM plans WHERE 1=0"
BAD = "SELECT nope FROM nowhere"
PRE_FINETUNE = (
    "schema", "questions", "gold_crosscheck", "verifier_selftest", "split", "headroom",
    "teacher_data", "paraphrase",
)  # fmt: skip
SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "natural_train.py"


@dataclass(frozen=True)
class _Q:
    question: str


def natural_doc(
    path: Path, pairs: Sequence[tuple[str, str]], *, db_sha: str = DB_SHA, ddl: str = DDL
) -> Path:
    doc = {
        "format": nt.NATURAL_FORMAT,
        "label": nt.LABEL,
        "db_sha256": db_sha,
        "questions_sha256": "a" * 64,
        "reference_sha256": "b" * 64,
        "kept": [to_training_row(_Q(q), sql, schema_ddl=ddl) for q, sql in pairs],
        "kept_meta": [{"i": i, "question": q, "sql": sql} for i, (q, sql) in enumerate(pairs)],
        "discards": {},
        "spend": {},
    }
    path.write_text(json.dumps(doc))
    return path


NOVEL = [
    ("[N] how many accounts do we have?", COUNT_A),
    ("[N] how many users are there?", COUNT_U),
    ("[N] count the plans", "SELECT COUNT(*) FROM plans"),
]


# ---------------------------------------------------------------- the keep rule


class StubRunner:
    """``humanset._Runner``: question -> k replies (None = an LLM error)."""

    def __init__(self, script: Mapping[str, Sequence[str | None]]) -> None:
        self.script = script
        self.seen: dict[str, int] = {}
        self.temperatures: list[float | None] = []

    def map(self, role: str, batch: Any, *, purpose: str, stage: str, temperature: Any = None):  # type: ignore[no-untyped-def]
        assert role == "teacher"
        self.temperatures.append(temperature)
        out = []
        for msgs in batch:
            q = question_of(msgs)
            assert q is not None
            i = self.seen.get(q, 0)
            self.seen[q] = i + 1
            sql = self.script[q][i]
            out.append(None if sql is None else SimpleNamespace(text=f"```sql\n{sql}\n```"))
        return out


KEEP_CASES: dict[str, tuple[list[str | None], str, str | None]] = {
    # question: (3 drafts, reference SQL, expected discard reason or None = kept)
    "keep?": ([COUNT_A] * 3, COUNT_A, None),
    "keep unordered?": (
        ["SELECT plan_id FROM plans ORDER BY plan_id DESC"] * 3, "SELECT plan_id FROM plans", None,
    ),
    "disagree?": ([COUNT_A, COUNT_A, COUNT_U], COUNT_A, "disagree"),
    "empty?": ([EMPTY] * 3, EMPTY, "empty"),
    "exec error?": ([BAD] * 3, COUNT_A, "exec_error"),
    "llm error?": ([COUNT_A, None, COUNT_A], COUNT_A, "llm_error"),
    "mismatch?": ([COUNT_A] * 3, COUNT_U, "ref_mismatch"),
    "order mismatch?": (
        ["SELECT plan_id FROM plans ORDER BY plan_id"] * 3,
        "SELECT plan_id FROM plans ORDER BY plan_id DESC", "ref_mismatch",
    ),
    "ref error?": ([COUNT_A] * 3, BAD, "ref_error"),
    "ref empty?": ([COUNT_A] * 3, EMPTY, "ref_empty"),
}  # fmt: skip


def _draft(tmp: Path, cases: Mapping[str, tuple[list[str | None], str, str | None]]):  # type: ignore[no-untyped-def]
    qpath = tmp / "q.txt"
    qpath.write_text("\n".join(cases) + "\n")
    refs = {q: nt.Reference(i, q, ref) for i, (q, (_d, ref, _r)) in enumerate(cases.items())}
    runner = StubRunner({q: d for q, (d, _ref, _r) in cases.items()})
    doc = nt.draft_natural(
        load_question_file(qpath), refs, runner=runner, executor=LocalExecutor({"db": DB}),
        db_ref="db", schema_ddl=DDL, db_sha256=DB_SHA, teacher_model="fake-teacher",
    )  # fmt: skip
    return doc, runner


def test_keep_rule_requires_agreement_non_empty_execution_and_reference_equality(
    tmp_path: Path,
) -> None:
    doc, runner = _draft(tmp_path, KEEP_CASES)
    kept = [m["question"] for m in doc["kept_meta"]]
    assert kept == [q for q, (_d, _r, why) in KEEP_CASES.items() if why is None]
    reasons = {d["question"]: d["reason"] for d in doc["discarded"]}
    assert reasons == {q: why for q, (_d, _r, why) in KEEP_CASES.items() if why is not None}
    assert sum(doc["discards"].values()) == len(reasons)
    assert set(doc["discards"]) == set(nt.DISCARD_REASONS)
    assert runner.temperatures == [nt.DRAFT_TEMPERATURE] and doc["k"] == 3
    # every kept row is the pipeline's training row, built from the teacher's first draft
    for row, m in zip(doc["kept"], doc["kept_meta"], strict=True):
        assert row == to_training_row(_Q(m["question"]), m["sql"], schema_ddl=DDL)
        assert m["sql"] == KEEP_CASES[m["question"]][0][0]


def test_missing_reference_is_refused(tmp_path: Path) -> None:
    qpath = tmp_path / "q.txt"
    qpath.write_text("a?\nb?\n")
    with pytest.raises(nt.NaturalTrainError, match="no reference"):
        nt.draft_natural(
            load_question_file(qpath), {"a?": nt.Reference(0, "a?", COUNT_A)},
            runner=StubRunner({}), executor=LocalExecutor({"db": DB}), db_ref="db",
            schema_ddl=DDL, db_sha256=DB_SHA, teacher_model="t",
        )  # fmt: skip


def test_reference_loader_refuses_duplicates_and_bad_lines(tmp_path: Path) -> None:
    p = tmp_path / "r.jsonl"
    p.write_text(json.dumps({"i": 0, "question": "a  ?", "reference_sql": COUNT_A}) + "\n")
    refs, sha = nt.load_references(p)
    assert set(refs) == {"a ?"} and sha == sha256_hex(p.read_bytes())
    p.write_text(p.read_text() + json.dumps({"i": 1, "question": "a ?", "reference_sql": "x"}))
    with pytest.raises(nt.NaturalTrainError, match="duplicate"):
        nt.load_references(p)
    p.write_text('{"i": 0}\n')
    with pytest.raises(nt.NaturalTrainError, match="reference line 1"):
        nt.load_references(p)


# ---------------------------------------------------------------- the script


def _script() -> Any:
    spec = importlib.util.spec_from_file_location("natural_train_script", SCRIPT)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


DRAFT_SCRIPT: dict[str, list[Any]] = {
    "keep?": [COUNT_A] * 3,
    "mismatch?": [COUNT_A] * 3,
    "disagree?": [COUNT_A, COUNT_A, COUNT_U],
}
DRAFT_REFS = {"keep?": COUNT_A, "mismatch?": COUNT_U, "disagree?": COUNT_A}


def _inputs(tmp: Path) -> tuple[Path, Path, Path]:
    q = tmp / "questions.txt"
    q.write_text("\n".join(DRAFT_SCRIPT) + "\n")
    r = tmp / "reference.jsonl"
    r.write_text(
        "".join(
            json.dumps({"i": i, "question": k, "reference_sql": v}) + "\n"
            for i, (k, v) in enumerate(DRAFT_REFS.items())
        )
    )
    db = tmp / "db.sqlite"
    db.write_bytes(DB)
    return q, r, db


def _run_script(tmp: Path, *args: str, **kw: Any) -> tuple[int, list[str]]:
    lines: list[str] = []
    code = _script().main(list(args), out=lines.append, **kw)
    return code, lines


def _draft_args(tmp: Path, budget: str = "1.0", *extra: str) -> list[str]:
    q, r, db = _inputs(tmp)
    return [
        "draft", "--questions", str(q), "--reference", str(r), "--db", str(db),
        "--out", str(tmp / "natural_train.json"), "--root", str(tmp / "home"),
        "--budget-usd", budget, *extra,
    ]  # fmt: skip


def test_draft_writes_the_natural_file_and_records_spend(
    tmp_path: Path, live_env: dict[str, str]
) -> None:
    transport = ScriptedTransport(DRAFT_SCRIPT)
    code, out = _run_script(
        tmp_path, *_draft_args(tmp_path, "1.0", "--i-approve-spend"),
        env=live_env, deps_factory=factory_for(transport),
    )  # fmt: skip
    assert code == EXIT_OK, out
    assert any(line.startswith("preflight:") for line in out)
    doc = json.loads((tmp_path / "natural_train.json").read_text())
    q, r, db = tmp_path / "questions.txt", tmp_path / "reference.jsonl", tmp_path / "db.sqlite"
    assert doc["questions_sha256"] == sha256_hex(q.read_bytes())
    assert doc["reference_sha256"] == sha256_hex(r.read_bytes())
    assert doc["db_sha256"] == sha256_hex(db.read_bytes()) == DB_SHA
    assert [m["question"] for m in doc["kept_meta"]] == ["keep?"]
    assert doc["kept"] == [to_training_row(_Q("keep?"), COUNT_A, schema_ddl=DDL)]
    assert doc["discards"]["ref_mismatch"] == 1 and doc["discards"]["disagree"] == 1
    assert all(c["temperature"] == 0.8 for c in transport.calls) and len(transport.calls) == 9
    run_id = doc["spend"]["run_id"]
    assert run_id == f"{nt.RUN_PREFIX}{doc['questions_sha256'][:8]}"
    assert doc["spend"]["usd"] > 0 and doc["spend"]["budget_usd"] == 1.0
    store = Store(tmp_path / "home")
    assert store.total_spend(run_id) == pytest.approx(doc["spend"]["usd"])
    store.close()
    # what the pipeline reads back
    ns = nt.read_natural(tmp_path / "natural_train.json")
    assert ns.questions == ("keep?",) and ns.db_sha256 == DB_SHA


def test_draft_refusals_happen_before_any_call(tmp_path: Path, live_env: dict[str, str]) -> None:
    transport = ScriptedTransport(DRAFT_SCRIPT)
    fac = factory_for(transport)
    no_token = {**live_env, "DISTILLERY_ADMIN_TOKEN_SUPPLIED": "nope"}
    code, out = _run_script(tmp_path, *_draft_args(tmp_path, "1.0", "--i-approve-spend"),
                            env=no_token, deps_factory=fac)  # fmt: skip
    assert code == EXIT_REFUSED and "admin token" in out[-1]
    code, out = _run_script(tmp_path, *_draft_args(tmp_path), env=live_env, deps_factory=fac)
    assert code == EXIT_REFUSED and "--i-approve-spend" in out[-1]
    code, out = _run_script(tmp_path, *_draft_args(tmp_path, "0.0000001", "--i-approve-spend"),
                            env=live_env, deps_factory=fac)  # fmt: skip
    assert code == EXIT_REFUSED and "exceeds the remaining budget" in out[-1]
    code, out = _run_script(
        tmp_path, *_draft_args(tmp_path, "1.0", "--i-approve-spend", "--db-seed", "7"),
        env=live_env, deps_factory=fac,
    )  # fmt: skip
    assert code == EXIT_REFUSED and "--db-seed" in out[-1]
    with pytest.raises(SystemExit):  # --budget-usd is required
        _script().main(["draft", "--questions", "q", "--reference", "r", "--db", "d", "--out", "o"])
    assert transport.calls == [] and not (tmp_path / "natural_train.json").exists()


def test_check_reports_exact_and_skeleton_overlap(tmp_path: Path) -> None:
    nat = natural_doc(tmp_path / "n.json", [
        ("How many accounts are on the pro plan?", COUNT_A),
        ("[N] something else entirely", COUNT_U),
    ])  # fmt: skip
    sealed = tmp_path / "sealed.txt"
    sealed.write_text(
        "How many  accounts are on the pro plan?\nHow many accounts are on the free plan?\nnew?\n"
    )
    out_json = tmp_path / "check.json"
    code, out = _run_script(
        tmp_path, "check", "--natural", str(nat), "--sealed", str(sealed), "--out", str(out_json)
    )
    assert code == EXIT_OK
    rep = json.loads(out_json.read_text())
    s = rep["sets"][str(sealed)]
    assert s["sealed_n"] == 3 and s["exact_overlap"] == 1
    assert s["exact_overlap_questions"] == ["How many accounts are on the pro plan?"]
    from distillery.taskpacks.sql.questions import skeleton

    nat_sk = {skeleton(q) for q, _ in [("How many accounts are on the pro plan?", ""),
                                        ("[N] something else entirely", "")]}  # fmt: skip
    want = sum(
        skeleton(q) in nat_sk
        for q in ["How many accounts are on the pro plan?",
                  "How many accounts are on the free plan?", "new?"]
    )  # fmt: skip
    assert s["skeleton_in_train"] == want >= 1
    assert any("WARNING: 1 exact overlap" in line for line in out)


def test_question_texts_reads_every_supported_format(tmp_path: Path) -> None:
    (tmp_path / "a.txt").write_text("# c\nq1\nq2\n")
    (tmp_path / "b.json").write_text(json.dumps({"items": [{"question": "q1 "}]}))
    (tmp_path / "c.jsonl").write_text('{"question": "q3"}\n{"question": "q4"}\n')
    (tmp_path / "d.json").write_text(json.dumps(["q5"]))
    assert nt.read_question_texts(tmp_path / "a.txt") == ["q1", "q2"]
    assert nt.read_question_texts(tmp_path / "b.json") == ["q1"]
    assert nt.read_question_texts(tmp_path / "c.jsonl") == ["q3", "q4"]
    assert nt.read_question_texts(tmp_path / "d.json") == ["q5"]


# ---------------------------------------------------------------- the pipeline hook


def _pipe(store: Store, bridge: AsyncBridge, run_id: str, **kw: Any) -> Pipeline:
    dr = build_dry_run(NANO, bridge, **kw)
    return Pipeline(dr.pipeline_cfg, dr.config, dr.deps, store, run_id, say=lambda _s: None)


def _keys(store: Store, run_id: str) -> dict[str, set[str]]:
    out: dict[str, set[str]] = {}
    for stage, key in store._query(  # noqa: SLF001 - the stage cache keys themselves
        "SELECT stage, input_hash FROM stages WHERE run_id=?", (run_id,)
    ):
        out.setdefault(stage, set()).add(key)
    return out


@pytest.fixture(scope="module")
def n1(tmp_path_factory: pytest.TempPathFactory) -> Iterator[dict[str, Any]]:
    """A plain dry run (stands in for P4), then the same run id in the same store with natural
    rows (stands in for N1 on a copy of P4's store)."""
    tmp = tmp_path_factory.mktemp("n1")
    store = Store(tmp / "store")
    with AsyncBridge() as bridge:
        base = _pipe(store, bridge, "dry-n1")
        base_report = base.run()
        base_keys = _keys(store, "dry-n1")
        heldout = store.load_heldout("dry-n1")
        stress = store.load_stress("dry-n1") or []
        dev = base._results["split"]["dev"]  # noqa: SLF001
        excl = tmp / "gateb_questions.txt"
        excl.write_text("[N]  a sealed   gate b question\n")
        pairs = [
            *NOVEL,
            (heldout[0]["question"], heldout[0]["gold_sql"]),
            (stress[0]["question"], stress[0]["gold_sql"]),
            (dev[0]["question"], dev[0]["gold_sql"]),
            ("[N] a sealed gate b question", COUNT_A),
        ]
        nat = natural_doc(tmp / "natural_train.json", pairs)
        pipe = _pipe(
            store, bridge, "dry-n1",
            natural_train_path=str(nat), natural_exclude_paths=(str(excl),),
        )  # fmt: skip
        report = pipe.run()
        yield {
            "tmp": tmp, "store": store, "base": base, "base_report": base_report,
            "base_keys": base_keys, "pipe": pipe, "report": report, "nat": nat,
            "excl": excl, "bridge": bridge,
        }  # fmt: skip
    store.close()


def test_natural_rerun_reuses_every_pre_finetune_stage_and_retrains(n1: dict[str, Any]) -> None:
    pipe, store = n1["pipe"], n1["store"]
    assert not set(PRE_FINETUNE) & set(pipe.ran)  # all cache hits
    assert "natural_train" in pipe.ran and "finetune_r1" in pipe.ran
    keys = _keys(store, "dry-n1")
    for stage in PRE_FINETUNE:
        assert keys[stage] == n1["base_keys"][stage], stage
    assert len(keys["finetune_r1"]) == 2  # P4's key untouched, plus the N1 one


def test_finetune_sees_base_plus_kept_rows_and_trains_into_its_own_dir(n1: dict[str, Any]) -> None:
    pipe, rep = n1["pipe"], n1["report"]
    base_rows = len(n1["base"]._results["paraphrase"]["rows"])  # noqa: SLF001
    ft = pipe._results["finetune_r1"]  # noqa: SLF001
    assert ft["train_rows"] == base_rows + len(NOVEL) == rep["rounds"][0]["train_rows"]
    sha = pipe.cfg.natural_train_sha256
    rdir = pipe.run_dir / f"round1-n{sha[:8]}"
    assert len((rdir / "train.jsonl").read_text().splitlines()) == base_rows + len(NOVEL)
    assert len((pipe.run_dir / "round1" / "train.jsonl").read_text().splitlines()) == base_rows
    base_ft = n1["base"]._results["finetune_r1"]["artifact"]  # noqa: SLF001
    assert ft["artifact"]["job_id"] != base_ft["job_id"]  # a new fine-tune, not the cached one


def test_overlap_drop_is_counted_and_reported(n1: dict[str, Any]) -> None:
    block = n1["report"]["natural_train"]
    assert block["dropped_by_overlap"] == {
        "heldout": 1, "stress": 1, "human": 0, "dev": 1, "exclude_files": 1, "total": 4,
    }  # fmt: skip
    assert block["rows_in_file"] == len(NOVEL) + 4 and block["rows_appended"] == len(NOVEL)
    assert block["file_sha256"] == sha256_hex(n1["nat"].read_bytes())
    assert block["exclude_files_sha256"] == [sha256_hex(n1["excl"].read_bytes())]
    assert block["db_sha256"] == DB_SHA and block["label"] == "agent-authored"
    assert block["row_cap_applied"] is False
    assert block["questions_sha256"] == "a" * 64 and block["reference_sha256"] == "b" * 64
    assert n1["report"]["counters"]["natural_train"]["natural_dropped_total"] == 4
    # identity: the sha is in the config dump; machine-local paths are not
    pipe_cfg = n1["report"]["config"]["pipeline"]
    assert pipe_cfg["natural_train_sha256"] == block["file_sha256"]
    text = json.dumps(n1["report"])
    assert "natural_train_path" not in text and str(n1["tmp"]) not in text
    meta = json.loads((n1["pipe"].run_dir / "run_meta.json").read_text())
    assert meta["config"]["natural_train_sha256"] == block["file_sha256"]
    assert "natural_train" not in n1["base_report"]  # a plain run's report is unchanged


def test_expected_artifact_is_tagged_and_never_confused_across_runs(n1: dict[str, Any]) -> None:
    store, pipe, base = n1["store"], n1["pipe"], n1["base"]
    recs = [
        d
        for n, d in store.list_experiments("dry-n1")
        if n == "expected_artifact" and d["round"] == 1
    ]
    assert [d.get("natural_train_sha256") for d in recs] == [
        None, pipe.cfg.natural_train_sha256,
    ]  # fmt: skip
    for p in (pipe, base):
        for rd in p._results:  # noqa: SLF001
            if rd.startswith("finetune_r"):
                r = int(rd.removeprefix("finetune_r"))
                art = p._results[rd]["artifact"]  # noqa: SLF001
                exp = p._expected(r)  # noqa: SLF001
                assert (exp.job_id, exp.adapter_sha256) == (art["job_id"], art["adapter_sha256"])
    cand = n1["report"]["candidate_round"]
    want = pipe._results[f"finetune_r{cand}"]["artifact"]  # noqa: SLF001
    assert n1["report"]["evaluation"]["artifact"]["job_id"] == want["job_id"]


def test_score_human_after_n1_scores_the_n1_adapter(n1: dict[str, Any], tmp_path: Path) -> None:
    """N1-eval is scored afterwards with score-human, which rebuilds the pipeline config from the
    report: it must pick the adapter trained on the natural rows."""
    from distillery.humanscore import score_human_run
    from tests.test_orchestrator_human import confirmed_file

    f = confirmed_file(tmp_path / "n1_eval.json")
    with AsyncBridge() as bridge:
        dr = build_dry_run(NANO, bridge, human_set_path=f)
        block = score_human_run(n1["store"], "dry-n1", f, dr.config, dr.deps, say=lambda _s: None)
    assert block["n"] > 0
    rep = json.loads((n1["pipe"].run_dir / "report_human.json").read_text())
    cand = n1["report"]["candidate_round"]
    want = n1["pipe"]._results[f"finetune_r{cand}"]["artifact"]  # noqa: SLF001
    assert json.dumps(rep).count(want["adapter_sha256"]) >= 1


def test_separate_dry_runs_with_and_without_natural_share_pre_finetune_keys(
    tmp_path: Path,
) -> None:
    nat = natural_doc(tmp_path / "n.json", NOVEL)
    with AsyncBridge() as bridge:
        s1, s2 = Store(tmp_path / "a"), Store(tmp_path / "b")
        _pipe(s1, bridge, "dry-k").run()
        p2 = _pipe(s2, bridge, "dry-k", natural_train_path=str(nat))
        p2.run()
        k1, k2 = _keys(s1, "dry-k"), _keys(s2, "dry-k")
        s1.close()
        s2.close()
    for stage in PRE_FINETUNE:
        assert k1[stage] == k2[stage], stage
    assert k1["finetune_r1"] != k2["finetune_r1"]
    assert "natural_train" in k2 and "natural_train" not in k1


def test_db_mismatch_refuses_before_any_paid_stage(tmp_path: Path) -> None:
    nat = natural_doc(tmp_path / "n.json", NOVEL, db_sha="0" * 64)
    with AsyncBridge() as bridge:
        store = Store(tmp_path / "s")
        pipe = _pipe(store, bridge, "dry-db", natural_train_path=str(nat))
        with pytest.raises(ConfigRefusal, match="different database"):
            pipe.run()
        assert pipe.ran == []  # refused before any stage
        assert not (pipe.run_dir / "run_meta.json").exists()  # and before the identity record
        # a refused file never pins the run id: the corrected file runs in the same store
        good = natural_doc(tmp_path / "good.json", NOVEL)
        pipe2 = _pipe(store, bridge, "dry-db", natural_train_path=str(good))
        rep = pipe2.run()
        assert rep["natural_train"]["rows_appended"] == len(NOVEL)
        store.close()


def test_resume_with_the_same_natural_file_is_a_full_cache_hit(n1: dict[str, Any]) -> None:
    """The autonomy path: a restart with the same flags adopts every finished stage and starts
    no second fine-tune."""
    pipe = _pipe(
        n1["store"], n1["bridge"], "dry-n1",
        natural_train_path=str(n1["nat"]), natural_exclude_paths=(str(n1["excl"]),),
    )  # fmt: skip
    rep = pipe.run()
    assert pipe.ran == []
    assert rep["natural_train"] == n1["report"]["natural_train"]


def test_rows_not_in_the_run_training_format_are_refused(tmp_path: Path) -> None:
    nat = natural_doc(tmp_path / "n.json", NOVEL, ddl=DDL + "\n-- drift")
    with AsyncBridge() as bridge:
        store = Store(tmp_path / "s")
        pipe = _pipe(store, bridge, "dry-fmt", natural_train_path=str(nat))
        with pytest.raises(ConfigRefusal, match="training format"):
            pipe.run()
        store.close()


def test_a_different_natural_file_on_resume_is_refused(n1: dict[str, Any]) -> None:
    other = natural_doc(n1["tmp"] / "other.json", NOVEL[:1])
    for kw in ({"natural_train_path": str(other)}, {}):
        pipe = _pipe(n1["store"], n1["bridge"], "dry-n1", **kw)
        with pytest.raises(ConfigRefusal, match="natural train file sha256"):
            pipe.run()


def test_configured_sha_must_match_the_file(tmp_path: Path) -> None:
    nat = natural_doc(tmp_path / "n.json", NOVEL)
    with AsyncBridge() as bridge:
        store = Store(tmp_path / "s")
        pipe = _pipe(store, bridge, "dry-sha", natural_train_path=str(nat),
                     natural_train_sha256="f" * 64)  # fmt: skip
        with pytest.raises(ConfigRefusal, match="configured"):
            pipe.run()
        store.close()


# ---------------------------------------------------------------- the CLI


def test_cli_dry_run_end_to_end_with_natural_rows(tmp_path: Path) -> None:
    nat = natural_doc(tmp_path / "n.json", NOVEL)
    excl = tmp_path / "x.txt"
    excl.write_text(NOVEL[0][0] + "\n")
    lines: list[str] = []
    code = cli_main(
        ["--root", str(tmp_path / "home"), "run", "--dry-run", "--scale", "tiny",
         "--natural-train", str(nat), "--natural-exclude", str(excl)],
        env={}, out=lines.append,
    )  # fmt: skip
    assert code == EXIT_OK, lines[-20:]
    report_path = Path(lines[-1].split("report: ", 1)[1])
    rep = json.loads(report_path.read_text())
    block = rep["natural_train"]
    assert block["rows_appended"] == len(NOVEL) - 1
    assert block["dropped_by_overlap"]["exclude_files"] == 1
    para_rows = rep["paraphrase"]["final_train_rows"]
    assert rep["rounds"][0]["train_rows"] == para_rows + len(NOVEL) - 1
    assert rep["config"]["pipeline"]["natural_train_sha256"] == sha256_hex(nat.read_bytes())


def test_cli_refuses_exclude_without_natural_and_missing_files(tmp_path: Path) -> None:
    lines: list[str] = []
    code = cli_main(["--root", str(tmp_path), "run", "--dry-run", "--natural-exclude", "x"],
                    env={}, out=lines.append)  # fmt: skip
    assert code == EXIT_REFUSED and "--natural-train" in lines[-1]
    code = cli_main(["--root", str(tmp_path), "run", "--dry-run", "--natural-train",
                     str(tmp_path / "missing.json")], env={}, out=lines.append)  # fmt: skip
    assert code == EXIT_REFUSED and "not found" in lines[-1]
