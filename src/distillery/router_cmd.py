"""``distillery router ...``: build-splits, signals, fit, certify, evaluate.

Thin command layer over ``router_data`` (splits, signals), ``router_eval`` (fit/certify/evaluate,
pre-registration receipt) and ``evaluator`` (the only reader of sealed sets). Artifacts live under
``<run dir>/router/``:

    splits.json          counts and sha256 of the sealed CAL/TEST splits
    signals_<split>.json per-item signal rows (fit, cal, test)
    model.json           the fitted router (FIT rows only)
    cert.json            Learn-then-Test certificate (CAL rows only)
    eval_test.json       the ONE-SHOT test evaluation (W-C1, W-C2)

Order of use: build-splits -> commit the printed sha256 in DECISIONS.md -> signals (fit, cal, test;
cal/test need ``--prereg-commit``) -> fit -> certify -> evaluate. Everything that spends goes
through a ``Ledger.from_config(..., store, env=env)`` (global ledger + project cap) and is
preflighted before the first paid call.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from distillery import errors, evaluator
from distillery import router_data as rd
from distillery import router_eval as re_
from distillery.budget import BudgetExceeded, Ledger, UnknownPriceError
from distillery.config import Config, ConfigError, load_config
from distillery.embed_index import DEFAULT_DIMS, EmbeddingIndex, IndexSimilarity, cache_key
from distillery.humanset import HumanSetError
from distillery.orchestrator import ConfigRefusal
from distillery.router import RouterModel
from distillery.store import Store, sha256_hex
from distillery.taskpacks.base import get_pack
from distillery.taskpacks.sql import schema as sql_schema
from distillery.taskpacks.sql.executor import Executor, LocalExecutor

EXIT_OK, EXIT_FAIL, EXIT_REFUSED = errors.EXIT_OK, errors.EXIT_FAIL, errors.EXIT_REFUSED
DEFAULT_EMBED_MODEL = "Qwen/Qwen3-Embedding-8B"
EMBED_PURPOSE = "router_embed"
TEACHER_PURPOSE = "router_teacher"
CHARS_PER_TOKEN = 3  # the pipeline's own budget convention
EST_OUTPUT_TOKENS = 256
Say = Callable[[str], None]


@dataclass
class RouterRuntime:
    """What a paid ``signals`` run needs. Built by a factory (live wiring or a test's fakes)."""

    student: Any  # .generate(messages_batch) -> list[str | object with .text/.mean_logprob]
    teacher: Any | None  # .generate(messages_batch) -> list[str]; None when not needed
    executor: Executor
    embed_many: Callable[[list[str]], list[list[float]]]
    ledger: Ledger
    schema_ddl: str
    db_ref: str
    models: dict[str, str] = field(default_factory=dict)  # recorded in the artifact
    close: Callable[[], None] = lambda: None


@dataclass
class RuntimeContext:
    args: argparse.Namespace
    config: Config
    store: Store
    run_id: str
    env: Mapping[str, str]
    say: Say
    want_teacher: bool
    ledger: Ledger  # preflighted; the runtime must bill through this one


RuntimeFactory = Callable[[RuntimeContext], RouterRuntime]
DepsBuilder = Callable[[Config, Any, Any], Any]  # (config, pipeline_cfg, bridge) -> Deps


class Refused(Exception):
    """A refusal with its message; the command prints it and exits EXIT_REFUSED."""


# ---------------------------------------------------------------- parser


def add_router_parser(sub: Any) -> None:
    r = sub.add_parser("router", help="certified router data pipeline (splits, signals, certify)")
    rs = r.add_subparsers(dest="router_cmd", required=True)

    def common(p: argparse.ArgumentParser) -> None:
        p.add_argument(
            "--run", required=True, help="the N1 run id whose store holds the sealed sets"
        )
        p.add_argument(
            "--repo", default=".", help="git repository holding DECISIONS.md (default .)"
        )

    bs = rs.add_parser(
        "build-splits", help="draw, filter and seal the fresh CAL/TEST splits (free)"
    )
    common(bs)
    bs.add_argument(
        "--natural",
        action="append",
        default=[],
        metavar="FILE",
        help="confirmed natural-question file (human_confirmed.json format); repeatable",
    )
    bs.add_argument(
        "--human-file",
        action="append",
        default=[],
        metavar="FILE",
        help="confirmed human file that is ALREADY used (Gate B): excluded from CAL/TEST "
        "and part of FIT; repeatable",
    )
    bs.add_argument("--seed", type=int, default=rd.SPLIT_SEED)
    bs.add_argument("--cal-n", type=int, default=500)
    bs.add_argument("--test-n", type=int, default=500)
    bs.add_argument("--mix-templated", type=float, default=0.5)
    bs.add_argument("--mix-natural", type=float, default=0.25)
    bs.add_argument("--mix-stress", type=float, default=0.25)
    bs.add_argument("--oversample", type=float, default=3.0)
    bs.add_argument(
        "--round-dir",
        action="append",
        default=None,
        metavar="DIR",
        help="training round dir(s) under the run dir (default: every round*)",
    )

    sg = rs.add_parser("signals", help="compute per-item signals for a split (PAID)")
    common(sg)
    sg.add_argument("--split", required=True, choices=["fit", "cal", "test"])
    sg.add_argument(
        "--human-file",
        action="append",
        default=[],
        metavar="FILE",
        help="(fit) extra confirmed human file(s) in the FIT split, e.g. Gate B",
    )
    sg.add_argument("--prereg-commit", default=None, help="required for cal/test")
    sg.add_argument("--embed-model", default=DEFAULT_EMBED_MODEL)
    sg.add_argument("--embed-dims", type=int, default=DEFAULT_DIMS)
    sg.add_argument("--round-dir", action="append", default=None, metavar="DIR")
    sg.add_argument("--budget-usd", type=float, default=None, help="per-run spend cap")
    sg.add_argument("--i-approve-spend", action="store_true")
    sg.add_argument("--overwrite", action="store_true", help="(fit only) replace signals_fit.json")
    sg.add_argument(
        "--missing-logprob", type=float, default=None,
        help="impute this mean_logprob for outputs without one (default: refuse)",
    )  # fmt: skip

    ft = rs.add_parser("fit", help="fit the router on the FIT signals only (free)")
    common(ft)
    ft.add_argument("--l2", type=float, default=1e-2)
    ft.add_argument("--overwrite", action="store_true")

    ce = rs.add_parser("certify", help="Learn-then-Test certificate on the CAL signals (free)")
    common(ce)
    ce.add_argument("--alpha", type=float, required=True)
    ce.add_argument("--delta", type=float, required=True)
    ce.add_argument("--levels", default=None, help="comma-separated coverage levels in (0,1]")
    ce.add_argument("--prereg-commit", default=None)

    ev = rs.add_parser("evaluate", help="the ONE-SHOT test evaluation (W-C1, W-C2)")
    common(ev)
    ev.add_argument("--split", required=True, choices=["test"])
    ev.add_argument("--prereg-commit", default=None)


# ---------------------------------------------------------------- shared helpers


def _store_for(root: Path, run_id: str) -> Store:
    from distillery.orchestrator import DRY_PREFIX

    return Store(root / "dry-runs" if run_id.startswith(DRY_PREFIX) else root)


def _json_file(path: Path) -> dict[str, Any]:
    try:
        doc: dict[str, Any] = json.loads(path.read_bytes())
    except (OSError, ValueError) as exc:
        raise Refused(f"cannot read {path.name}: {exc}") from None
    return doc


def _file_sha(path: Path) -> str:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError as exc:
        raise Refused(f"cannot read {path.name}: {exc}") from None


def _db(store: Store, run_id: str) -> tuple[str, str]:
    """(db path, sha256) of the run's database, checked against the schema stage record."""
    path = store.run_dir(run_id) / "db.sqlite"
    if not path.is_file():
        raise Refused(f"run {run_id} has no db.sqlite")
    sha = sha256_hex(path.read_bytes())
    rec = store.completed_stage_output(run_id, "schema")
    if rec is not None and rec.get("db_sha256") != sha:
        raise Refused(f"run {run_id}: db.sqlite differs from the recorded schema stage")
    return str(path), sha


def _need_files(paths: Sequence[str]) -> None:
    for f in paths:
        if not Path(f).is_file():
            raise Refused(f"file not found: {f}")


def _levels(raw: str | None) -> tuple[float, ...]:
    if raw is None:
        return re_.DEFAULT_LEVELS
    try:
        return tuple(float(x) for x in raw.split(",") if x.strip())
    except ValueError:
        raise Refused(f"--levels must be comma-separated numbers, got {raw!r}") from None


# ---------------------------------------------------------------- build-splits


def _build_splits(args: argparse.Namespace, store: Store, out: Say) -> int:
    _need_files([*args.natural, *args.human_file])
    db_ref, db_sha = _db(store, args.run)
    params = rd.SplitParams(
        seed=args.seed,
        cal_n=args.cal_n,
        test_n=args.test_n,
        mix={
            "templated": args.mix_templated,
            "natural": args.mix_natural,
            "stress": args.mix_stress,
        },
        oversample=args.oversample,
    )
    rep = rd.build_splits(
        store,
        args.run,
        executor=LocalExecutor(),
        db_ref=db_ref,
        db_sha=db_sha,
        natural_files=args.natural,
        human_files=args.human_file,
        params=params,
        round_dirs=args.round_dir,
    )
    out(
        f"run {args.run}: training questions {rep['training']['questions']} "
        f"({rep['training']['rows']} rows in {', '.join(rep['training']['row_files'])})"
    )
    for kind, k in rep["per_kind"].items():
        out(
            f"  {kind}: candidates {k['candidates']}, "
            f"dropped {json.dumps(k['dropped'], sort_keys=True)}, "
            f"pool {k['pool']} (target {k['target_pool']}, shortfall {k['shortfall']}) "
            f"-> cal {k['cal']} / test {k['test']}"
        )
    out(f"realised mix: {json.dumps(rep['realised_mix'], sort_keys=True)}")
    out(
        "skeleton overlap per reference set (reported, not dropped): "
        + json.dumps(
            {
                r: {k: v["skeleton"] for k, v in per.items()}
                for r, per in rep["overlap"]["vs_reference_sets"].items()
            },
            sort_keys=True,
        )
    )
    out(
        f"skeleton overlap with training (reported, not dropped): "
        f"{json.dumps(rep['overlap']['skeleton_in_train'], sort_keys=True)}"
    )
    out(f"cal  n={rep['n']['cal']}  sealed sha256: {rep['cal_sha256']}")
    out(f"test n={rep['n']['test']}  sealed sha256: {rep['test_sha256']}")
    out("next: put both sha256 into DECISIONS.md, commit, and pass that commit as --prereg-commit")
    out(f"report: {rd.router_dir(store, args.run) / 'splits.json'}")
    return EXIT_OK


# ---------------------------------------------------------------- signals (paid)


def _estimate_usd(
    ledger: Ledger,
    *,
    embed_model: str,
    embed_texts: Sequence[str],
    teacher_model: str | None,
    teacher_prompts: Sequence[str],
) -> dict[str, Any]:
    """Budget-basis estimate (prompt chars / 3 tokens, 256 output tokens per teacher call)."""
    e_tok = sum(len(t) for t in embed_texts) // CHARS_PER_TOKEN
    embed_usd = ledger.estimate_llm_cost(embed_model, e_tok, 0)
    teacher_usd, t_in = 0.0, 0
    if teacher_model is not None:
        t_in = sum(len(t) for t in teacher_prompts) // CHARS_PER_TOKEN
        teacher_usd = ledger.estimate_llm_cost(
            teacher_model, t_in, EST_OUTPUT_TOKENS * len(teacher_prompts)
        )
    return {
        "embed_tokens": e_tok,
        "embed_usd": embed_usd,
        "teacher_calls": len(teacher_prompts),
        "teacher_input_tokens": t_in,
        "teacher_usd": teacher_usd,
        "total_usd": embed_usd + teacher_usd,
        "basis": f"prompt chars / {CHARS_PER_TOKEN} tokens, {EST_OUTPUT_TOKENS} output tokens per "
        "teacher call, configured prices (ESTIMATE); student sandbox seconds are billed at close "
        "and not estimated here",
    }


def _signals(
    args: argparse.Namespace,
    env: Mapping[str, str],
    store: Store,
    out: Say,
    *,
    factory: RuntimeFactory,
    verify_admin: Callable[[Config], bool],
) -> int:
    split, run_id = args.split, args.run
    rdir = rd.router_dir(store, run_id)
    path = rdir / f"signals_{split}.json"
    if path.exists() and not (split == "fit" and args.overwrite):
        raise Refused(f"{path.name} already exists; signals are never recomputed over an artifact")
    if (
        split == "fit"
        and path.exists()
        and any((rdir / f).exists() for f in ("model.json", "signals_test.json", "cert.json"))
    ):
        raise Refused("signals_fit.json is already used (model/test artifacts exist); not replaced")
    if split != "fit" and args.human_file:
        raise Refused("--human-file only applies to --split fit (cal/test are sealed)")
    prereg: dict[str, Any] | None = None
    if split != "fit":
        prereg = re_.verify_prereg(
            args.repo, args.prereg_commit, {split: store.router_split_sha(run_id, split)}
        )
        out(
            f"pre-registration {prereg['commit'][:12]} ({prereg['committed_at']}) contains the "
            f"sealed {split} sha256"
        )
    _need_files(args.human_file)
    db_ref, db_sha = _db(store, run_id)
    config = load_config(env)
    if not verify_admin(config):
        raise Refused("signals spends money and requires the admin token; none/incorrect supplied")
    if not args.i_approve_spend:
        raise Refused("signals spends money; it requires --i-approve-spend")
    if args.budget_usd is not None:
        config = config.model_copy(update={"run_cap_usd": args.budget_usd})

    # sealed/fit items: only the evaluator reads them
    items = (
        evaluator.router_fit_items(store, run_id, args.human_file, db_sha=db_sha)
        if split == "fit"
        else evaluator.router_split_items(store, run_id, split)
    )
    train = rd.training_questions(store, run_id, round_dirs=args.round_dir)
    key = cache_key(args.embed_model, args.embed_dims, train.train)
    cache = rdir / f"embed_index_{key[:16]}.bin"
    pk = get_pack("sql")
    ddl = sql_schema.schema_ddl(_db_seed(store, run_id))
    want_teacher = split == "test"
    teacher_model = config.model_ids.get("teacher") if want_teacher else None
    if want_teacher and teacher_model is None:
        raise Refused("no teacher model configured; test signals need teacher_correct")
    # ONE ledger for the whole command (global ledger + project cap), preflighted BEFORE the paid
    # runtime (sandbox images, student serving) is built; the runtime reuses it.
    ledger = Ledger.from_config(run_id, config, store, env=env)
    est = _estimate_usd(
        ledger,
        embed_model=args.embed_model,
        embed_texts=[
            *(() if cache.is_file() else train.train),
            *(str(i["question"]) for i in items),
        ],
        teacher_model=teacher_model,
        teacher_prompts=[
            "".join(
                m["content"]
                for m in pk.build_messages(str(i["question"]), ddl, role="eval_teacher")
            )
            for i in items
        ]
        if want_teacher
        else [],
    )
    out(
        f"estimate usd {est['total_usd']:.6f} (embeddings {est['embed_usd']:.6f}, teacher "
        f"{est['teacher_usd']:.6f}) - {est['basis']}"
    )
    ledger.preflight(est["total_usd"])  # BudgetExceeded -> refused, nothing spent yet
    spent0 = ledger.spent()
    ctx = RuntimeContext(args, config, store, run_id, env, out, want_teacher, ledger)
    rt = factory(ctx)
    try:
        index = EmbeddingIndex.build_sync(
            train.train,
            rt.embed_many,
            model=args.embed_model,
            dims=args.embed_dims,
            cache_path=cache,
        )
        out(
            f"embedding index: {len(index)} training questions, dims {index.dims}, "
            f"key {index.key[:12]}"
        )
        rows = rd.compute_signals(
            items,
            rt.student,
            rt.executor,
            IndexSimilarity(index, rt.embed_many),
            rt.db_ref,
            schema_ddl=rt.schema_ddl or ddl,
            teacher=rt.teacher,
            pack=pk,
            split=split,
            require_mean_logprob=True,
            missing_logprob=args.missing_logprob,
        )
    finally:
        rt.close()  # bills the sandbox seconds of the student
    spent = ledger.spent() - spent0
    split_sha = store.router_split_sha(run_id, split) if split != "fit" else None
    extra: dict[str, Any] = {
        "run_id": run_id,
        "prereg": prereg,
        "embed": {
            "model": args.embed_model,
            "dims": args.embed_dims,
            "index_key": index.key,
            "training_questions": len(index),
        },
        "models": rt.models,
        "kinds": dict(sorted(Counter(str(r["kind"]) for r in rows).items())),
        "missing_logprob": args.missing_logprob,
        "estimate_usd": est["total_usd"],
        "spend_usd_this_command": spent,
        "fit_sources": sorted({str(i.get("source")) for i in items}) if split == "fit" else None,
    }
    sha = rd.write_signal_rows(path, split=split, split_sha256=split_sha, rows=rows, extra=extra)
    out(
        f"{split}: {len(rows)} rows, exec_ok {sum(r['exec_ok'] for r in rows)}, student correct "
        f"{sum(bool(r['correct']) for r in rows)}, spend usd (ESTIMATE) {spent:.6f}"
    )
    out(f"wrote {path} sha256 {sha}")
    return EXIT_OK


def _db_seed(store: Store, run_id: str) -> int:
    rec = store.completed_stage_output(run_id, "schema") or {}
    return int(rec.get("db_seed", 0))


# ---------------------------------------------------------------- fit / certify / evaluate


def _fit(args: argparse.Namespace, store: Store, out: Say) -> int:
    rdir = rd.router_dir(store, args.run)
    sig = rdir / "signals_fit.json"
    if not sig.is_file():
        raise Refused("no signals_fit.json: run `router signals --split fit` first")
    path = rdir / "model.json"
    if path.exists():
        if not args.overwrite:
            raise Refused("model.json already exists (use --overwrite before certifying)")
        if (rdir / "cert.json").exists() or (rdir / "signals_test.json").exists():
            raise Refused(
                "model.json cannot be replaced once a certificate or test signals exist "
                "(test labels have been seen)"
            )
        path.unlink()
    doc = rd.read_signal_rows(sig)
    model = re_.fit_model(doc["rows"], l2=args.l2)
    body = {
        "format": "distillery-router-model-v1",
        "run_id": args.run,
        "model": model.to_json(),
        "model_sha256": model.sha256(),
        "fit_signals_sha256": _file_sha(sig),
        "fit_split": "fit (sealed held-out + stress + human sets of the run, plus human files)",
        "n_fit": len(doc["rows"]),
        "student_accuracy_fit": sum(bool(r["correct"]) for r in doc["rows"]) / len(doc["rows"]),
    }
    sha = re_.write_new_json(path, body)
    out(
        f"fitted on {body['n_fit']} FIT rows; weights "
        f"{dict(zip(model.feature_names, [round(w, 4) for w in model.weights], strict=True))}, "
        f"bias {model.bias:.4f}"
    )
    out(f"model sha256: {model.sha256()}")
    out(f"wrote {path} (file sha256 {sha})")
    return EXIT_OK


def _load_model(store: Store, run_id: str) -> tuple[RouterModel, dict[str, Any], Path]:
    path = rd.router_dir(store, run_id) / "model.json"
    if not path.is_file():
        raise Refused("no model.json: run `router fit` first")
    doc = _json_file(path)
    model = RouterModel.from_json(doc["model"])
    if model.sha256() != doc["model_sha256"]:
        raise Refused("model.json does not match its recorded model sha256")
    return model, doc, path


def _signals_for(store: Store, run_id: str, split: str) -> tuple[dict[str, Any], str]:
    path = rd.router_dir(store, run_id) / f"signals_{split}.json"
    if not path.is_file():
        raise Refused(f"no signals_{split}.json: run `router signals --split {split}` first")
    doc = rd.read_signal_rows(path)
    if split != "fit":
        sealed = store.router_split_sha(run_id, split)
        if doc.get("split_sha256") != sealed:
            raise Refused(f"signals_{split}.json was computed on a different {split} split")
    return doc, _file_sha(path)


def _certify(args: argparse.Namespace, store: Store, out: Say) -> int:
    run_id = args.run
    path = rd.router_dir(store, run_id) / "cert.json"
    if path.exists():
        raise Refused("cert.json already exists; the certificate is never recomputed")
    prereg = re_.verify_prereg(
        args.repo, args.prereg_commit, {"cal": store.router_split_sha(run_id, "cal")}
    )
    model, mdoc, mpath = _load_model(store, run_id)
    fit_doc, fit_sha = _signals_for(store, run_id, "fit")
    if fit_sha != mdoc["fit_signals_sha256"]:
        raise Refused("signals_fit.json changed since the model was fitted")
    cal_doc, cal_sha = _signals_for(store, run_id, "cal")
    res = re_.certify_cal(
        model,
        fit_doc["rows"],
        cal_doc["rows"],
        alpha=args.alpha,
        delta=args.delta,
        levels=_levels(args.levels),
    )
    body = {
        "format": "distillery-router-cert-v1",
        "run_id": run_id,
        "prereg": prereg,
        "model_file_sha256": _file_sha(mpath),
        "fit_signals_sha256": fit_sha,
        "cal_signals_sha256": cal_sha,
        "cal_split_sha256": store.router_split_sha(run_id, "cal"),
        **res,
    }
    sha = re_.write_new_json(path, body)
    lam = res["lambda_hat"]
    cert = res["cert"]
    out(f"alpha {args.alpha} delta {args.delta} grid {len(res['grid'])} thresholds")
    if lam is None:
        out("NOT certified: no threshold passed; all traffic goes to the teacher")
    else:
        out(
            f"lambda_hat {lam:.6f}  cal coverage {cert['coverage']:.3f}  "
            f"cal selective risk {cert['selective_risk']:.4f}"
        )
    out(f"wrote {path} (file sha256 {sha})")
    return EXIT_OK


def _evaluate(args: argparse.Namespace, store: Store, out: Say) -> int:
    run_id = args.run
    rdir = rd.router_dir(store, run_id)
    path = rdir / "eval_test.json"
    if path.exists():
        raise Refused("eval_test.json already exists: the test split is evaluated exactly once")
    cert_path = rdir / "cert.json"
    if not cert_path.is_file():
        raise Refused("no cert.json: certify on the calibration split before touching test")
    cert = _json_file(cert_path)
    prereg = re_.verify_prereg(
        args.repo,
        args.prereg_commit,
        {
            "cal": store.router_split_sha(run_id, "cal"),
            "test": store.router_split_sha(run_id, "test"),
        },
    )
    model, mdoc, mpath = _load_model(store, run_id)
    if cert["model_sha256"] != model.sha256():
        raise Refused("cert.json was made for a different model than model.json")
    test_doc, test_sha = _signals_for(store, run_id, "test")
    res = re_.evaluate_test(model, cert["lambda_hat"], test_doc["rows"], alpha=float(cert["alpha"]))
    body = {
        "format": "distillery-router-eval-v1",
        "run_id": run_id,
        "split": "test",
        "prereg": prereg,
        "cert_file_sha256": _file_sha(cert_path),
        "model_file_sha256": _file_sha(mpath),
        "test_signals_sha256": test_sha,
        "test_split_sha256": store.router_split_sha(run_id, "test"),
        "alpha": cert["alpha"],
        "delta": cert["delta"],
        **res,
    }
    sha = re_.write_new_json(path, body)  # O_EXCL: a second evaluation cannot overwrite this one
    c1, c2 = res["W-C1"], res["W-C2"]
    out(
        f"W-C1 {'PASS' if c1['pass'] else 'FAIL'}: coverage {c1['coverage']:.3f} "
        f"(min {c1['min_coverage']}), selective error {c1['selective_error']} "
        f"(alpha {c1['alpha']}), CP95 upper {c1['cp_upper_95']:.4f}, "
        f"system {c1['system_accuracy']:.4f} "
        f"vs teacher-only {c1['teacher_only_accuracy']:.4f}"
    )
    out(
        f"W-C2 {'PASS' if c2['pass'] else 'FAIL'}: deferred share {c2['deferred_share']} "
        f"(lower95 {c2['deferred_share_lower_95']}), deferral on correct "
        f"{c2['deferral_rate_on_correct']}, stress routed to student "
        f"{c2['stress_routing_share_student']} (not gated)"
    )
    out(f"wrote {path} (file sha256 {sha})")
    return EXIT_OK


# ---------------------------------------------------------------- entry


def run_router(
    args: argparse.Namespace,
    env: Mapping[str, str],
    out: Say,
    *,
    root: Path,
    runtime_factory: RuntimeFactory | None,
    deps_builder: DepsBuilder,
    verify_admin: Callable[[Config], bool],
) -> int:
    store = _store_for(root, args.run)
    try:
        cmd = args.router_cmd
        if cmd == "build-splits":
            return _build_splits(args, store, out)
        if cmd == "signals":
            factory = runtime_factory or (lambda ctx: live_runtime(ctx, deps_builder))
            return _signals(args, env, store, out, factory=factory, verify_admin=verify_admin)
        if cmd == "fit":
            return _fit(args, store, out)
        if cmd == "certify":
            return _certify(args, store, out)
        return _evaluate(args, store, out)
    except (
        Refused,
        re_.RouterRefusal,
        rd.RouterDataError,
        HumanSetError,
        ConfigError,
        BudgetExceeded,
        UnknownPriceError,
        ConfigRefusal,
        ValueError,
    ) as exc:
        out(f"refused: {exc}")
        return EXIT_REFUSED
    finally:
        store.close()


# ---------------------------------------------------------------- live wiring (UNVERIFIED live)


def live_runtime(ctx: RuntimeContext, deps_builder: DepsBuilder) -> RouterRuntime:
    """The paid wiring: the run's trained student served in sandboxes (as ``final_eval`` /
    ``score-human`` do), the teacher through the shared LLM client, embeddings through the same
    client. ONE Ledger (``Ledger.from_config(..., store, env=env)``) backs all of it, so the
    global ledger and the project cap apply to students, teacher and embeddings alike.
    UNVERIFIED live until its first real run."""
    from distillery.humanscore import HumanScoreRefusal, _outputs, load_report
    from distillery.llm import LLMClient
    from distillery.orchestrator import (
        LedgerSink,
        LLMGenerator,
        LLMRunner,
        Pipeline,
        PipelineConfig,
        PipelineError,
        _close_all,
        artifact_from_json,
    )
    from distillery.sandbox_executor import AsyncBridge

    store, run_id, config = ctx.store, ctx.run_id, ctx.config
    try:
        report = load_report(store, run_id)
        pcfg = PipelineConfig.model_validate(report["config"]["pipeline"])
        bridge = AsyncBridge()
        try:
            deps = deps_builder(config, pcfg, bridge)
            pipe = Pipeline(pcfg, config, deps, store, run_id, say=ctx.say)
            ledger = ctx.ledger
            pipe.ledger = ledger
            pipe.sink = LedgerSink(ledger)
            kw: dict[str, Any] = {} if deps.llm_sleep is None else {"sleep": deps.llm_sleep}
            pipe.llm = LLMClient(
                deps.transport,
                config.model_ids,
                sink=pipe.sink,
                pricing=ledger.estimate_llm_cost,
                **kw,
            )
            pipe.runner = LLMRunner(pipe.llm, ledger, bridge, pcfg)
            schema = (_outputs(store, run_id, "schema") or [{}])[0]
            db_path = store.run_dir(run_id) / "db.sqlite"
            if not schema or sha256_hex(db_path.read_bytes()) != schema.get("db_sha256"):
                raise HumanScoreRefusal(
                    f"run {run_id}: db.sqlite is missing or differs from record"
                )
            pipe.ddl = sql_schema.schema_ddl(int(schema["db_seed"]))
            r = int(report["candidate_round"])
            expected = pipe._expected(r)  # noqa: SLF001 - the run's own recorded identity
            ft = [
                o
                for o in _outputs(store, run_id, f"finetune_r{r}")
                if o["artifact"]["adapter_sha256"] == expected.adapter_sha256
                and o["artifact"]["job_id"] == expected.job_id
            ]
            if not ft:
                raise HumanScoreRefusal(f"run {run_id}: no finetune_r{r} matches expected_artifact")
            trained = artifact_from_json(ft[0]["artifact"], store.run_dir(run_id))
            evaluator.verify_artifact(trained, expected)
            pipe.config = pipe.config.model_copy(
                update={"model_ids": {**pipe.config.model_ids, "student": trained.base_model}}
            )
            pipe._resolve_serving()  # noqa: SLF001 - same live wiring as final_eval
            if pipe.deps.student_factory is None:
                raise HumanScoreRefusal("no student serving path is available")
            student = pipe.deps.student_factory(trained)
        except BaseException:
            bridge.close()
            raise
    except (HumanScoreRefusal, PipelineError, evaluator.ArtifactMismatchError) as exc:
        raise Refused(str(exc)) from None
    teacher = (
        LLMGenerator(pipe.runner, "teacher", TEACHER_PURPOSE, "router_signals")
        if ctx.want_teacher
        else None
    )

    def embed_many(texts: list[str]) -> list[list[float]]:
        return bridge.run(pipe.llm.embed(texts, model=ctx.args.embed_model, purpose=EMBED_PURPOSE))

    def close() -> None:
        try:
            _close_all([student], raise_first=False)
        finally:
            bridge.close()

    return RouterRuntime(
        student=student,
        teacher=teacher,
        executor=pipe.deps.executor,
        embed_many=embed_many,
        ledger=ledger,
        schema_ddl=pipe.ddl,
        db_ref=pipe.db_ref,
        models={
            "student_base": str(trained.base_model),
            "adapter_sha256": str(trained.adapter_sha256),
            "teacher": config.model_ids.get("teacher", ""),
            "embedding": ctx.args.embed_model,
        },
        close=close,
    )
