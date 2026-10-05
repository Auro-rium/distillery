"""Workstream B: the fair teacher comparison as library code (``scripts/fair_teacher.py`` = CLI).

Pre-registered in DECISIONS.md (2026-10-05) BEFORE any scoring. Setups: S0 Super zero-shot (the P4
prompt), S1 Super with k=8 examples, S2 Ultra with k=8 examples (Ultra mapped onto the teacher
role), Student (the promoted adapter), plus the base model so the P4 gate can be recomputed. The k=8
examples are one fixed set drawn from the TRAIN split only. The strongest teacher is picked on dev
(150), never on held-out. "Beats the teacher" needs the lower bound of the student/strongest-teacher
accuracy-ratio CI > 1.0 with the gate's own bootstrap (resamples and seed). The held-out re-score is
post-hoc; the P4 gate decision stands unchanged.

The source run is READ-ONLY. ``prepare_workspace`` copies its store into the output directory and
everything (ledger rows, seals, vectors) is written there; ``tree_fingerprint`` proves the source
did not change. All sealed-set access stays in ``evaluator.py`` (``score_setups``).
"""

from __future__ import annotations

import hashlib
import json
import shutil
import sqlite3
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from distillery import evaluator as evaluator_mod
from distillery.budget import Ledger, UnknownPriceError
from distillery.config import Config, GateThresholds
from distillery.evaluator import Generator, PromptBuilder, Setup
from distillery.gate import evaluate_gate
from distillery.llm import LLMClient
from distillery.misses import render_table, summarize_misses
from distillery.orchestrator import (
    DRY_RUN_LABEL,
    SCALES,
    Deps,
    LLMGenerator,
    LLMRunner,
    Pipeline,
    PipelineConfig,
    _close_all,
    artifact_from_json,
)
from distillery.prompts import (
    FEWSHOT_K,
    FEWSHOT_SEED,
    build_fewshot_messages,
    build_messages,
    select_fewshot_examples,
)
from distillery.stats import paired_bootstrap_ratio_ci
from distillery.store import Store, atomic_write_bytes, canonical_json, sha256_hex
from distillery.taskpacks.sql import schema as sql_schema

S0, S1, S2 = "S0", "S1", "S2"
TEACHER_SETUPS = (S0, S1, S2)  # increasing strength/cost; also the dev tie-break order (see below)
STUDENT, BASE = "student", "base"
STAGE = "fair_teacher"
PURPOSES = {S0: "fair_s0", S1: "fair_s1", S2: "fair_s2"}
CLAIM_RATIO_MIN = 1.0  # pre-registered: ratio CI lower bound must be STRICTLY above this

# What P4 recorded for the gate set (docs/proofs/evidence/p4_run_summary.json): 300 items,
# student 92.3% (277), teacher 90.0% (270), base 25.7% (77). Recomputing the gate from the saved
# per-item vectors of S0/student/base must reproduce these counts.
P4_REFERENCE: dict[str, Any] = {
    "n": 300,
    "correct": {STUDENT: 277, "teacher": 270, BASE: 77},
    "source": "docs/proofs/evidence/p4_run_summary.json (gate_heldout)",
}
P4_MEASURED_TEACHER_OUT_TOKENS_PER_CALL = 409  # 1,079,412 out / 2,638 calls over the whole P4 run
FINETUNE_FREE_NOTE = "student re-generation runs on sandbox CPU (unbilled beta, 2-3 h wall)"


class FairTeacherRefusal(RuntimeError):
    """A precondition failed; nothing was scored or spent."""


# ---------------------------------------------------------------- pure analysis


def pick_strongest(dev_block: Mapping[str, Any]) -> dict[str, Any]:
    """The strongest teacher by DEV accuracy. A tie goes to the later setup (S2 > S1 > S0): the
    claim must clear the harder opponent, so a tie never makes the comparison easier."""
    acc = {n: float(dev_block["setups"][n]["accuracy"]) for n in TEACHER_SETUPS}
    best = max(acc.values())
    chosen = [n for n in TEACHER_SETUPS if acc[n] == best][-1]
    return {
        "setup": chosen,
        "dev_accuracy": acc,
        "tie_break": "later setup wins a tie (S2 > S1 > S0), recorded in DECISIONS.md",
        "selected_on": "dev",
    }


def claim_rule(
    student: Sequence[bool], teacher: Sequence[bool], cfg: GateThresholds
) -> dict[str, Any]:
    """Pre-registered claim: lower bound of the student/teacher ratio CI > 1.0 (same bootstrap as
    the gate: ``cfg.bootstrap_resamples`` resamples, ``cfg.seed``)."""
    if sum(teacher) == 0:
        return {
            "claim": False,
            "reason": "teacher accuracy is 0; ratio undefined",
            "n": len(teacher),
        }
    ci = paired_bootstrap_ratio_ci(student, teacher, cfg.bootstrap_resamples, cfg.seed)
    return {
        "claim": ci.lo > CLAIM_RATIO_MIN,
        "ratio_point": ci.point,
        "ratio_lo": ci.lo,
        "ratio_hi": ci.hi,
        "threshold": f"ratio lower bound > {CLAIM_RATIO_MIN}",
        "student_correct": sum(student),
        "teacher_correct": sum(teacher),
        "n": len(teacher),
        "bootstrap_resamples": cfg.bootstrap_resamples,
        "seed": cfg.seed,
    }


def _vectors(block: Mapping[str, Any], names: Sequence[str]) -> list[list[bool]]:
    return [[bool(x) for x in block["setups"][n]["correct"]] for n in names]


def recompute_gate(
    payload: Mapping[str, Any],
    cfg: GateThresholds,
    *,
    set_name: str = "heldout",
    teacher: str = S0,
) -> dict[str, Any]:
    """``gate.evaluate_gate`` recomputed from SAVED per-item vectors (base, student, teacher)."""
    block = payload["sets"][set_name]
    if block.get("status") != "scored":
        raise FairTeacherRefusal(f"set {set_name!r} is not scored: {block.get('reason')}")
    base, student, teach = _vectors(block, (BASE, STUDENT, teacher))
    return evaluate_gate(base, student, teach, cfg).model_dump(mode="json")


def check_p4_reproduction(
    payload: Mapping[str, Any],
    cfg: GateThresholds,
    reference: Mapping[str, Any] = P4_REFERENCE,
) -> dict[str, Any]:
    """Does the gate recomputed from the saved S0/student/base vectors match P4's 92.3% / 90.0% /
    25.7%? Compares correct COUNTS (exact); a deviation is reported, never hidden: a regenerated
    zero-shot teacher is allowed to differ by sampling noise, but then the claim must say so."""
    gate = recompute_gate(payload, cfg)
    n = int(gate["n"])
    got = {
        STUDENT: round(gate["student_acc"] * n),
        "teacher": round(gate["teacher_acc"] * n),
        BASE: round(gate["base_acc"] * n),
    }
    want = dict(reference["correct"])
    return {
        "reproduced": n == reference["n"] and got == want,
        "n": n,
        "expected_n": reference["n"],
        "correct": got,
        "expected_correct": want,
        "accuracy": {k: v / n for k, v in got.items()},
        "differences": {k: got[k] - want[k] for k in want},
        "gate_decision_recomputed": gate["decision"],
        "gate": gate,
        "reference": reference["source"],
    }


def before_after(
    before: Mapping[str, Any], after: Mapping[str, Any], names: Sequence[str]
) -> dict[str, Any]:
    """Accuracy of every setup before and after a re-score (B3), per scored set."""
    out: dict[str, Any] = {}
    for set_name, b in before["sets"].items():
        a = after["sets"].get(set_name)
        if b.get("status") != "scored" or a is None or a.get("status") != "scored":
            continue
        out[set_name] = {
            n: {
                "before": sum(b["setups"][n]["correct"]),
                "after": sum(a["setups"][n]["correct"]),
                "n": b["n"],
            }
            for n in names
        }
    return out


def analyse(
    payload: Mapping[str, Any],
    dev_block: Mapping[str, Any],
    cfg: GateThresholds,
) -> dict[str, Any]:
    """Everything derived from saved vectors: strongest teacher, claim rule on each scored set,
    the P4 reproduction check. Pure, so it can be re-run from the JSON alone."""
    strongest = pick_strongest(dev_block)
    claims: dict[str, Any] = {}
    for set_name, block in payload["sets"].items():
        if block.get("status") != "scored":
            claims[set_name] = {"status": "pending", "reason": block.get("reason")}
            continue
        student = [bool(x) for x in block["setups"][STUDENT]["correct"]]
        per = {
            n: claim_rule(student, [bool(x) for x in block["setups"][n]["correct"]], cfg)
            for n in TEACHER_SETUPS
        }
        claims[set_name] = {
            "status": "scored",
            "n": block["n"],
            "accuracy": {n: block["setups"][n]["accuracy"] for n in block["setups"]},
            "vs_strongest_teacher": {"teacher": strongest["setup"], **per[strongest["setup"]]},
            "vs_each_teacher_for_information": per,
        }
    heldout = payload["sets"]["heldout"]
    return {
        "strongest_teacher": strongest,
        "claims": claims,
        "claim_rule": f"student/teacher ratio CI lower bound > {CLAIM_RATIO_MIN}, gate bootstrap",
        "p4_reproduction": check_p4_reproduction(payload, cfg)
        if heldout.get("status") == "scored"
        else None,
        "disclosure": "post-hoc re-score; the original P4 gate decision stands unchanged",
    }


def misses_report(payload: Mapping[str, Any]) -> dict[str, Any]:
    """B3: the pre-classified misses of every scored set, summarised, with the review table."""
    entries = [
        m for b in payload["sets"].values() if b.get("status") == "scored" for m in b["misses"]
    ]
    summary = summarize_misses(entries)
    return {
        "entries": entries,
        "summary": summary,
        "table_md": render_table(summary["review"]),
        "note": "rule-based, no LLM; a class is a hint for review, not a verdict",
    }


# ---------------------------------------------------------------- pre-flight estimate


def tree_fingerprint(root: Path) -> str:
    """sha256 over (relative path, size, mtime_ns) of every file under ``root``: proves the source
    run was not written to. Contents are not read (the sealed files stay unread)."""
    h = hashlib.sha256()
    for p in sorted(root.rglob("*")):
        if p.is_file():
            st = p.stat()
            h.update(f"{p.relative_to(root)}\0{st.st_size}\0{st.st_mtime_ns}\n".encode())
    return h.hexdigest()


def _ro_connect(root: Path) -> sqlite3.Connection:
    return sqlite3.connect(f"file:{root / 'index.sqlite'}?mode=ro", uri=True)


def read_outputs_readonly(root: Path, run_id: str, stage: str) -> list[dict[str, Any]]:
    """Completed outputs of a stage (newest first), read without opening a ``Store``."""
    con = _ro_connect(root)
    try:
        rows = con.execute(
            "SELECT output_sha256 FROM stages WHERE run_id=? AND stage=? AND status='complete' "
            "ORDER BY updated_at DESC",
            (run_id, stage),
        ).fetchall()
    finally:
        con.close()
    out: list[dict[str, Any]] = []
    for (digest,) in rows:
        out.append(json.loads((root / "runs" / run_id / "artifacts" / digest).read_bytes()))
    return out


def spent_readonly(root: Path) -> tuple[float, float]:
    """(spend of every run, spend of the playground) in the source store, read-only."""
    con = _ro_connect(root)
    try:
        (total,) = con.execute("SELECT COALESCE(SUM(usd),0) FROM spend").fetchone()
    finally:
        con.close()
    return float(total), 0.0


def _tokens(messages: Sequence[Mapping[str, str]], chars_per_token: int) -> int:
    return sum(len(str(m.get("content", ""))) for m in messages) // chars_per_token


def estimate_table(
    root: Path,
    run_id: str,
    config: Config,
    *,
    human_n: int = 75,
    dev_n: int | None = None,
    k: int = FEWSHOT_K,
) -> dict[str, Any]:
    """The pre-flight cost table (no calls, no sealed read). Token counts come from the REAL
    prompts built on the run's train/dev items (held-out prompts have the same schema and the same
    question style, so the dev mean stands in for them); prices come from the budget ledger's
    price table. Output tokens use the pipeline's budget figure (``est_output_tokens``)."""
    report = json.loads((root / "runs" / run_id / "report.json").read_text())
    pcfg = PipelineConfig.model_validate(report["config"]["pipeline"])
    sealed = report["data"]["heldout_sealed_sha256"]
    splits = [
        s for s in read_outputs_readonly(root, run_id, "split") if s["sealed_sha256"] == sealed
    ]
    schemas = read_outputs_readonly(root, run_id, "schema")
    if not splits or not schemas:
        raise FairTeacherRefusal(f"run {run_id}: no split/schema stage output to size prompts from")
    ddl = sql_schema.schema_ddl(int(schemas[0]["db_seed"]))
    train, dev = splits[0]["train"], splits[0]["dev"][:dev_n] if dev_n else splits[0]["dev"]
    shots = select_fewshot_examples(train, k)
    n_gate = int(report["data"].get("heldout_n") or pcfg.scale.heldout)
    cpt = pcfg.chars_per_token
    zero = [_tokens(build_messages(d["question"], ddl, role="eval_teacher"), cpt) for d in dev]
    few = [_tokens(build_fewshot_messages(d["question"], ddl, shots), cpt) for d in dev]
    mean_zero, mean_few = sum(zero) / len(zero), sum(few) / len(few)
    ledger = Ledger.from_config(run_id, config)
    super_id, ultra_id = config.require_model("teacher"), config.require_model("planner")

    def usd(model: str, n: int, mean_in: float, out_per: int) -> float | None:
        try:
            return ledger.estimate_llm_cost(model, round(n * mean_in), n * out_per)
        except UnknownPriceError:
            return None

    out_est = pcfg.est_output_tokens
    rows: list[dict[str, Any]] = []

    def row(item: str, model: str, n: int, mean_in: float) -> None:
        rows.append(
            {
                "item": item,
                "model": model,
                "calls": n,
                "input_tokens": round(n * mean_in),
                "output_tokens": n * out_est,
                "usd_budget_basis": usd(model, n, mean_in, out_est),
                "usd_p4_measured_output": usd(
                    model, n, mean_in, P4_MEASURED_TEACHER_OUT_TOKENS_PER_CALL
                ),
            }
        )

    row("S0 held-out (Super zero-shot)", super_id, n_gate, mean_zero)
    row(f"S1 held-out (Super, k={k})", super_id, n_gate, mean_few)
    row(f"S2 held-out (Ultra, k={k})", ultra_id, n_gate, mean_few)
    row("Dev selection S0", super_id, len(dev), mean_zero)
    row(f"Dev selection S1 (k={k})", super_id, len(dev), mean_few)
    row(f"Dev selection S2 (k={k}, Ultra)", ultra_id, len(dev), mean_few)
    row(f"Human set S0 (n={human_n} assumed; pending)", super_id, human_n, mean_zero)
    row("Human set S1 (assumed)", super_id, human_n, mean_few)
    row("Human set S2 (assumed)", ultra_id, human_n, mean_few)

    def total(key: str, only: Callable[[str], bool] = lambda _i: True) -> float | None:
        vals = [r[key] for r in rows if only(r["item"])]
        return None if any(v is None for v in vals) else float(sum(vals))

    spent_all, _ = spent_readonly(root)
    return {
        "rows": rows,
        "total_usd_budget_basis": total("usd_budget_basis"),
        "total_usd_p4_measured_output": total("usd_p4_measured_output"),
        "total_without_human_usd_budget_basis": total(
            "usd_budget_basis", lambda i: not i.startswith("Human")
        ),
        "mean_input_tokens": {"zero_shot": mean_zero, f"few_shot_k{k}": mean_few},
        "output_tokens_per_call_budget": out_est,
        "output_tokens_per_call_p4_measured": P4_MEASURED_TEACHER_OUT_TOKENS_PER_CALL,
        "plan_reference": (
            "plan B2 estimate: S0 ~0.15, S1 ~0.6, S2 ~2.5, dev ~0.8, human drafting <0.5 = ~4.5"
        ),
        "not_in_table": "human question drafting (<$0.5, separate command)",
        "student_regeneration": FINETUNE_FREE_NOTE,
        "caps": {
            "run_cap_usd": config.run_cap_usd,
            "project_cap_usd": config.project_cap_usd,
            "spent_in_source_store_usd": spent_all,
            "project_headroom_usd": config.project_cap_usd - spent_all,
        },
        "prices": {
            m: {"in_per_mtok": p.input_per_mtok, "out_per_mtok": p.output_per_mtok, "date": p.date}
            for m, p in config.prices.items()
            if m in (super_id, ultra_id)
        },
        "token_basis": f"chars/{cpt} on the real prompts (dev mean); sealed items not read",
    }


def format_estimate(est: Mapping[str, Any]) -> str:
    def f(x: float | None) -> str:
        return "no price" if x is None else f"${x:,.3f}"

    lines = [
        "Pre-flight estimate (NO calls made; approval required before any live run)",
        f"{'item':44} {'calls':>6} {'in_tok':>10} {'out_tok':>9} "
        f"{'usd(budget)':>12} {'usd(P4 out)':>12}",
    ]
    for r in est["rows"]:
        lines.append(
            f"{r['item']:44} {r['calls']:6d} {r['input_tokens']:10d} {r['output_tokens']:9d} "
            f"{f(r['usd_budget_basis']):>12} {f(r['usd_p4_measured_output']):>12}"
        )
    lines += [
        f"{'TOTAL':44} {'':6} {'':10} {'':9} {f(est['total_usd_budget_basis']):>12} "
        f"{f(est['total_usd_p4_measured_output']):>12}",
        f"total without the (pending) human set: {f(est['total_without_human_usd_budget_basis'])}",
        f"output tokens/call: budget {est['output_tokens_per_call_budget']}, "
        f"P4 measured {est['output_tokens_per_call_p4_measured']}",
        f"plan reference: {est['plan_reference']}",
        f"not in table: {est['not_in_table']}; student: {est['student_regeneration']}",
        f"caps: run ${est['caps']['run_cap_usd']:.2f}, "
        f"project ${est['caps']['project_cap_usd']:.2f}, "
        f"spent so far ${est['caps']['spent_in_source_store_usd']:.3f}, "
        f"headroom ${est['caps']['project_headroom_usd']:.3f}",
        f"basis: {est['token_basis']}",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------- workspace and caching


def prepare_workspace(root: Path, out: Path, run_id: str) -> Path:
    """Copy the source store (index + this run's directory) to ``out/store`` so that the source is
    never opened for writing. Reused when it already exists (a resume). Returns the store root."""
    root, out = root.resolve(), out.resolve()
    if out == root or root in out.parents or out in root.parents:
        raise FairTeacherRefusal(
            "--out must be a separate directory, not inside --root (or vice versa)"
        )
    src_run = root / "runs" / run_id
    if not (src_run / "report.json").is_file() or not (root / "index.sqlite").is_file():
        raise FairTeacherRefusal(f"{root}: no finished run {run_id!r} (report.json / index.sqlite)")
    work = out / "store"
    if not (work / "runs" / run_id / "report.json").is_file():
        work.mkdir(parents=True, exist_ok=True)
        shutil.copy2(root / "index.sqlite", work / "index.sqlite")
        shutil.copytree(src_run, work / "runs" / run_id)
    return work


def find_run_id(root: Path) -> str:
    """The single finished run (with a report.json) under ``root``."""
    done = sorted(p.name for p in (root / "runs").glob("*") if (p / "report.json").is_file())
    if len(done) != 1:
        raise FairTeacherRefusal(f"pass --run-id: {len(done)} finished runs under {root}: {done}")
    return done[0]


class CachedGenerator:
    """Per-model output cache, like ``docs/proofs/p1_overfit.py`` (a crash never discards finished
    work). Keyed by the sha256 of the exact prompt batch, so a different set or a different prompt
    can never reuse a stale file."""

    def __init__(self, inner: Generator, cache_dir: Path, name: str) -> None:
        self._inner, self._dir, self._name = inner, cache_dir, name

    def generate(self, messages_batch: Sequence[Sequence[Mapping[str, Any]]]) -> list[str]:
        digest = sha256_hex(canonical_json([list(b) for b in messages_batch]).encode("utf-8"))
        path = self._dir / f"{self._name}-{digest[:16]}.json"
        if path.is_file():
            cached = json.loads(path.read_text(encoding="utf-8"))
            if cached.get("batch_sha256") == digest and len(cached["outputs"]) == len(
                messages_batch
            ):
                return [str(o) for o in cached["outputs"]]
        outs = self._inner.generate(messages_batch)  # type: ignore[arg-type]
        atomic_write_bytes(
            path, json.dumps({"batch_sha256": digest, "outputs": list(outs)}).encode("utf-8")
        )
        return list(outs)


# ---------------------------------------------------------------- the run


def build_teacher_generators(pipe: Pipeline, cache_dir: Path) -> dict[str, Generator]:
    """S0 and S1 use the run's teacher role (Super); S2 maps the planner model (Ultra) onto the
    teacher role in a second client that shares the SAME ledger, so every call is capped,
    recorded (tokens, usd, latency) and counted together."""
    deps, config = pipe.deps, pipe.config
    ids = {**config.model_ids, "teacher": config.require_model("planner")}
    kw: dict[str, Any] = {} if deps.llm_sleep is None else {"sleep": deps.llm_sleep}
    llm_ultra = LLMClient(
        deps.transport, ids, sink=pipe.sink, pricing=pipe.ledger.estimate_llm_cost, **kw
    )
    runner_ultra = LLMRunner(llm_ultra, pipe.ledger, deps.bridge, pipe.cfg)
    raw: dict[str, Generator] = {
        S0: LLMGenerator(pipe.runner, "teacher", PURPOSES[S0], STAGE),
        S1: LLMGenerator(pipe.runner, "teacher", PURPOSES[S1], STAGE),
        S2: LLMGenerator(runner_ultra, "teacher", PURPOSES[S2], STAGE),
    }
    return {n: CachedGenerator(g, cache_dir, n) for n, g in raw.items()}


def teacher_setups(
    gens: Mapping[str, Generator], ddl: str, shots: Sequence[Mapping[str, str]]
) -> list[Setup]:
    zero: PromptBuilder = lambda it: build_messages(  # noqa: E731
        str(it["question"]), ddl, role="eval_teacher"
    )
    few: PromptBuilder = lambda it: build_fewshot_messages(str(it["question"]), ddl, shots)  # noqa: E731
    return [Setup(S0, gens[S0], zero), Setup(S1, gens[S1], few), Setup(S2, gens[S2], few)]


def _dump(path: Path, obj: Any) -> None:
    atomic_write_bytes(
        path, (json.dumps(obj, indent=1, ensure_ascii=False, default=str) + "\n").encode()
    )


def run_fair_teacher(
    root: Path,
    out: Path,
    run_id: str,
    config: Config,
    deps: Deps,
    *,
    say: Callable[[str], None] = print,
    k: int = FEWSHOT_K,
    seed: int = FEWSHOT_SEED,
) -> dict[str, Any]:
    """Rebuild the finished run (like ``humanscore.score_human_run``), pick the strongest teacher
    on dev, score base/student/S0/S1/S2 on held-out (and the human set when sealed), save per-item
    vectors, recompute the gate, apply the claim rule and pre-classify misses. Writes only under
    ``out``; ``root`` is checked unchanged at the end."""
    from distillery.humanscore import _outputs as outputs_of  # the same stage readers

    before_fp = tree_fingerprint(root)
    work = prepare_workspace(root, out, run_id)
    out.mkdir(parents=True, exist_ok=True)
    cache_dir = out / "cache"
    store = Store(work)
    try:
        report = json.loads((store.run_dir(run_id) / "report.json").read_text(encoding="utf-8"))
        cfg_block = report["config"]
        gate_cfg = GateThresholds.model_validate(cfg_block["gate_thresholds"])
        pcfg = PipelineConfig.model_validate(cfg_block["pipeline"])
        pipe = Pipeline(
            pcfg, config.model_copy(update={"gate": gate_cfg}), deps, store, run_id, say=say
        )
        schema = outputs_of(store, run_id, "schema")[0]
        pipe.ddl = sql_schema.schema_ddl(int(schema["db_seed"]))
        sealed_sha = report["data"]["heldout_sealed_sha256"]
        split = next(
            s for s in outputs_of(store, run_id, "split") if s["sealed_sha256"] == sealed_sha
        )
        r = int(report["candidate_round"])
        expected = pipe._expected(r)  # noqa: SLF001 - the run's own recorded identity
        ft = [
            o
            for o in outputs_of(store, run_id, f"finetune_r{r}")
            if o["artifact"]["adapter_sha256"] == expected.adapter_sha256
        ]
        if not ft:
            raise FairTeacherRefusal(
                f"run {run_id}: no finetune_r{r} output matches expected_artifact"
            )
        trained = artifact_from_json(ft[0]["artifact"], store.run_dir(run_id))
        evaluator_mod.verify_artifact(trained, expected)
        # the student base is whatever was TRAINED, not whatever the environment's role says
        pipe.config = pipe.config.model_copy(
            update={"model_ids": {**pipe.config.model_ids, "student": trained.base_model}}
        )
        pipe._resolve_serving()  # noqa: SLF001 - same live wiring as final_eval
        deps = pipe.deps

        shots = select_fewshot_examples(split["train"], k, seed)
        _dump(
            out / "fewshot_examples.json",
            {"k": k, "seed": seed, "from": "train split only", "examples": shots},
        )
        say(f"[fair] {k} examples drawn from train (seed {seed}): {[e['family'] for e in shots]}")
        gens = build_teacher_generators(pipe, cache_dir)
        setups_t = teacher_setups(gens, pipe.ddl, shots)

        spent0 = pipe.ledger.spent()
        snap0 = pipe.sink.snapshot()
        say("[fair] dev selection pass (150 dev items, unsealed)")
        dev_block = evaluator_mod.score_open_set(
            split["dev"], setups_t, deps.executor, db_ref=pipe.db_ref, schema_ddl=pipe.ddl,
            set_name="dev",
        )  # fmt: skip
        _dump(out / "dev_vectors.json", dev_block)
        strongest = pick_strongest(dev_block)
        say(f"[fair] strongest teacher on dev: {strongest['setup']} {strongest['dev_accuracy']}")

        if deps.student_factory is None or deps.base_factory is None:
            raise FairTeacherRefusal("no student/base serving path is available")
        opened: list[Any] = []
        try:
            student = deps.student_factory(trained)
            opened.append(student)
            base = deps.base_factory()
            opened.append(base)
            zero: PromptBuilder = lambda it: build_messages(  # noqa: E731
                str(it["question"]), pipe.ddl, role="eval_student"
            )
            setups = [
                Setup(BASE, CachedGenerator(base, cache_dir, BASE), zero),
                Setup(STUDENT, CachedGenerator(student, cache_dir, STUDENT), zero),
                *setups_t,
            ]
            payload = evaluator_mod.score_setups(
                store, run_id, setups, deps.executor, db_ref=pipe.db_ref, schema_ddl=pipe.ddl,
                on_set=lambda name, _b: say(f"[fair] scored {name}"),
            )  # fmt: skip
        except BaseException:
            _close_all(opened, raise_first=False)
            raise
        _close_all(opened)
        payload["adapter_sha256"] = trained.adapter_sha256
        payload["fewshot_task_ids"] = [e["task_id"] for e in shots]
        _dump(out / "vectors.json", payload)

        analysis = analyse(payload, dev_block, gate_cfg)
        miss = misses_report(payload)
        _dump(out / "b3_misses.json", {k_: v for k_, v in miss.items() if k_ != "table_md"})
        atomic_write_bytes(out / "b3_misses.md", miss["table_md"].encode("utf-8"))
        usage = pipe.sink.snapshot()
        evidence = {
            "run_id": run_id,
            "dry_run": pcfg.dry_run,
            "label": DRY_RUN_LABEL if pcfg.dry_run else None,
            "adapter_sha256": trained.adapter_sha256,
            "heldout_sealed_sha256": sealed_sha,
            "student_base_model": trained.base_model,
            "teacher_model": config.require_model("teacher"),
            "ultra_model": config.require_model("planner"),
            "fewshot": {"k": k, "seed": seed, "task_ids": [e["task_id"] for e in shots]},
            "dev_block_file": "dev_vectors.json",
            "vectors_file": "vectors.json",
            "analysis": analysis,
            "b3_summary": miss["summary"]["counts"],
            "spend_usd_this_command": pipe.ledger.spent() - spent0,
            "usage_by_purpose": {p: usage.get(p, {}) for p in PURPOSES.values()},
            "usage_before": {p: snap0.get(p, {}) for p in PURPOSES.values()},
            "llm_errors": dict(pipe.runner.errors),
            "source_tree_unchanged": tree_fingerprint(root) == before_fp,
        }
        if not evidence["source_tree_unchanged"]:
            raise FairTeacherRefusal("the read-only source run changed during scoring")
        _dump(out / "b_fair_teacher.json", evidence)
        return evidence
    finally:
        store.close()


def rescore_from_cache(
    out: Path,
    run_id: str,
    config: Config,
    deps: Deps,
    *,
    saved: Path | None = None,
    say: Callable[[str], None] = print,
) -> dict[str, Any]:
    """B3 re-score: re-execute and re-verify the CACHED outputs in ``vectors.json`` with the
    current verifier. No model is called (setups carry no generator); reports before/after for
    every setup. ``saved`` defaults to ``out/vectors.json``; the result goes to
    ``vectors_rescored.json`` (the original is never overwritten)."""
    src = saved or out / "vectors.json"
    payload = json.loads(src.read_text(encoding="utf-8"))
    store = Store(out / "store")
    try:
        meta = json.loads((store.run_dir(run_id) / "report.json").read_text())
        pipe = Pipeline(
            PipelineConfig.model_validate(meta["config"]["pipeline"]),
            config, deps, store, run_id, say=say,
        )  # fmt: skip
        no_model: PromptBuilder = lambda _it: []  # noqa: E731
        setups = [Setup(n, None, no_model) for n in payload["setups"]]
        after = evaluator_mod.score_setups(
            store, run_id, setups, deps.executor, db_ref=pipe.db_ref, schema_ddl="", cached=payload
        )
    finally:
        store.close()
    after["adapter_sha256"] = payload.get("adapter_sha256")
    _dump(out / "vectors_rescored.json", after)
    result = {
        "before_after_correct": before_after(payload, after, payload["setups"]),
        "model_calls": 0,
    }
    _dump(out / "rescore_summary.json", result)
    return result


def dry_run(
    out: Path, *, human_set: Path | None = None, say: Callable[[str], None] = print
) -> dict[str, Any]:
    """Offline, free: build a finished dry run with the fake models, then run the whole fair
    comparison and the cached re-score on it. Everything is FAKE and labelled; it exercises the
    code path (copy, few-shot, dev pick, vectors, gate recompute, claim rule, B3), not a result."""
    from distillery.pipeline_fakes import build_dry_run
    from distillery.sandbox_executor import AsyncBridge

    run_id = "dry-fair-teacher"
    src = out / "dry_source"
    overrides: dict[str, Any] = {} if human_set is None else {"human_set_path": human_set}
    with AsyncBridge() as bridge:
        dr = build_dry_run(SCALES["tiny"], bridge, **overrides)
        store = Store(src)
        try:
            Pipeline(dr.pipeline_cfg, dr.config, dr.deps, store, run_id, say=lambda _m: None).run()
        finally:
            store.close()
        evidence = run_fair_teacher(src, out / "work", run_id, dr.config, dr.deps, say=say)
        evidence["rescore"] = rescore_from_cache(out / "work", run_id, dr.config, dr.deps, say=say)
    return evidence
