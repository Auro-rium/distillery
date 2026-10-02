"""Command line: ``python -m distillery run|status|report``.

Anything that can spend money (a non-dry run) needs (1) the admin token, checked in constant time
against ``DISTILLERY_ADMIN_TOKEN``, and (2) the explicit ``--i-approve-spend`` flag the first time
a given run id is started (approval is remembered per run so it can be resumed).

The supplied token is read from ``DISTILLERY_ADMIN_TOKEN_SUPPLIED`` or, if unset and stdin is a
terminal, from a hidden prompt; it is never taken from argv (shell history / process list).
"""

from __future__ import annotations

import argparse
import getpass
import json
import os
import sys
from collections.abc import Callable, Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from distillery import humanset
from distillery.budget import BudgetExceeded, Ledger, spend_lines
from distillery.config import Config, ConfigError, load_config
from distillery.driver import driving
from distillery.llm import LLMClient
from distillery.orchestrator import (
    DRY_PREFIX,
    DRY_RUN_LABEL,
    SCALES,
    ConfigRefusal,
    Deps,
    LedgerSink,
    LLMRunner,
    Pipeline,
    PipelineConfig,
    PipelineError,
    cost_by_model,
    stage_rows,
)
from distillery.pipeline_fakes import build_dry_run, dry_run_id
from distillery.sandbox import Sandbox
from distillery.sandbox_executor import AsyncBridge, SandboxExecutor
from distillery.store import Store, atomic_write_bytes, sha256_hex
from distillery.taskpacks.sql import schema as sql_schema
from distillery.taskpacks.sql.human import QuestionFileError, load_question_file

DepsFactory = Callable[[Config, PipelineConfig, AsyncBridge], Deps]
EXIT_OK, EXIT_FAIL, EXIT_REFUSED = 0, 1, 2


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="distillery", description="Sandbox-verified distillation")
    p.add_argument(
        "--root", default=None, help="data dir (default: $DISTILLERY_HOME or .distillery)"
    )
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="run the pipeline")
    r.add_argument("--pack", choices=["sql"], default="sql")
    r.add_argument("--scale", choices=sorted(SCALES), default="tiny")
    r.add_argument("--dry-run", action="store_true", help="offline, fake models, no spend")
    r.add_argument("--run-id", default=None)
    r.add_argument("--budget-usd", type=float, default=None, help="per-run spend cap")
    r.add_argument("--i-approve-spend", action="store_true")
    r.add_argument("--finetune-estimate-usd", type=float, default=None)
    r.add_argument(
        "--min-planned-steps", type=int, default=None,
        help="refuse a live fine-tune planning fewer optimizer steps than this (default 50)",
    )  # fmt: skip
    r.add_argument("--max-rounds", type=int, default=3)
    r.add_argument("--max-base-acc", type=float, default=0.80, help="headroom threshold on dev")
    r.add_argument("--seed", type=int, default=1234)
    r.add_argument(
        "--human-set", default=None, metavar="FILE",
        help="human_confirmed.json from confirm-heldout --finalize: sealed as Gate B",
    )  # fmt: skip
    s = sub.add_parser("status", help="show stages and spend of a run")
    s.add_argument("run")
    g = sub.add_parser("report", help="print the report of a finished run")
    g.add_argument("run")
    g.add_argument("--json", action="store_true", help="print the raw report.json")
    rp = sub.add_parser(
        "reprice", help="recompute a finished run's cost at a price file into cost_repriced"
    )
    rp.add_argument("run")
    rp.add_argument("--prices", required=True, help="price file (finetune/sandbox/llm sections)")
    rp.add_argument("--balance-before", type=float, default=None, help="console balance, USD")
    rp.add_argument("--balance-after", type=float, default=None, help="console balance, USD")
    rp.add_argument("--no-write", action="store_true", help="print only; leave report.json alone")
    sv = sub.add_parser("serve", help="run the HTTP API server (docs/API_CONTRACT.md)")
    sv.add_argument("--host", default="127.0.0.1")
    sv.add_argument("--port", type=int, default=8000)
    dh = sub.add_parser(
        "draft-heldout", help="teacher drafts gold SQL for a human question file (spends)"
    )
    dh.add_argument("--questions", required=True, metavar="FILE", help="one question per line")
    dh.add_argument("--budget-usd", type=float, default=None, help="spend cap for this drafting")
    dh.add_argument("--i-approve-spend", action="store_true")
    dh.add_argument("--db-seed", type=int, default=0)
    ch = sub.add_parser(
        "confirm-heldout", help="review drafted gold SQL, then --finalize human_confirmed.json"
    )
    ch.add_argument(
        "--set", default=None, metavar="ID", help="humanset/<id> (default: the only one)"
    )
    ch.add_argument("--finalize", action="store_true", help="write human_confirmed.json and stop")
    ex = sub.add_parser("export-replay", help="bundle a finished run into replay/")
    ex.add_argument("run")
    ex.add_argument("--replay-dir", default=None, help="default: $DISTILLERY_REPLAY_DIR or replay")
    ex.add_argument("--allow-dry-run", action="store_true", help="export a dry run (labelled)")
    return p


def _root(args: argparse.Namespace, env: Mapping[str, str]) -> Path:
    return Path(args.root or env.get("DISTILLERY_HOME") or ".distillery")


def _store_for(root: Path, run_id: str) -> Store:
    """Dry runs live in their own store so they can never mix with real ones."""
    return Store(root / "dry-runs" if run_id.startswith(DRY_PREFIX) else root)


SANDBOX_BASE_IMAGE = "docker://python:3.12-slim"  # the ref the S2/S4 spikes verified (has sqlite)
EXECUTOR_CONCURRENCY = 10  # SQL-runner jobs in flight (beta limit: 50 operations in total)


def make_live_deps(
    config: Config,
    pcfg: PipelineConfig,
    bridge: AsyncBridge,
    *,
    env: Mapping[str, str] | None = None,
    sandbox: Sandbox | None = None,
) -> Deps:
    """Real wiring: Token Factory for inference and fine-tuning, Nebius Sandboxes for everything
    that is executed (SQL runner, base and student serving on CPU).

    Refuses before touching any network when the key, base URL or sandbox project id is missing.
    The student/base factories are left ``None`` on purpose: the live pipeline config asks for
    ``student_serving="sandbox_cpu"`` and the pipeline builds them (one shared heavy image).
    ``sandbox`` is injectable for offline tests. UNVERIFIED live until the first real run: this
    exact function against the real fine-tune API.
    """
    import openai

    from distillery.finetune import FineTuneClient
    from distillery.llm import make_openai_client
    from distillery.sandbox import ContreeSandbox

    e: Mapping[str, str] = os.environ if env is None else env
    key, project = config.nebius_api_key, config.nebius_project_id
    if key is None or not config.nebius_base_url:
        raise ConfigRefusal("NEBIUS_API_KEY and NEBIUS_BASE_URL are required for a live run")
    if project is None:
        raise ConfigRefusal(
            "NEBIUS_AI_PROJECT (or NEBIUS_PROJECT_ID) is required for a live run: the Sandboxes "
            "API rejects requests without a Project header"
        )
    secret = key.get_secret_value()
    if sandbox is None:
        sandbox = ContreeSandbox(
            lambda: secret,
            e.get("DISTILLERY_SANDBOX_URL") or None,  # None = the SDK default base URL
            project_id=project.get_secret_value(),
        )
    image = bridge.run(
        sandbox.ensure_image(e.get("DISTILLERY_SANDBOX_IMAGE") or SANDBOX_BASE_IMAGE)
    )
    return Deps(
        transport=make_openai_client(config.nebius_base_url, secret),
        # max_retries=0: FineTuneClient owns the retry policy; create_job must never be retried
        finetune=FineTuneClient(
            openai.OpenAI(base_url=config.nebius_base_url, api_key=secret, max_retries=0)
        ),
        executor=SandboxExecutor(sandbox, image, bridge=bridge, concurrency=EXECUTOR_CONCURRENCY),
        bridge=bridge,
        student_factory=None,
        sandbox=sandbox,
        sandbox_image=image,
    )


def _supplied_token(env: Mapping[str, str], prompt: Callable[[], str] | None) -> str:
    tok = env.get("DISTILLERY_ADMIN_TOKEN_SUPPLIED")
    if tok:
        return tok
    if prompt is not None:
        return prompt()
    if sys.stdin.isatty():
        return getpass.getpass("Admin token: ")
    return ""


def _summary(report: Mapping[str, Any], out: Callable[[str], None]) -> None:
    if report.get("dry_run"):
        out("=" * 64)
        out(DRY_RUN_LABEL)
        out("=" * 64)
    ev = report["evaluation"]
    acc = ev["accuracy"]
    out(f"run: {report['run_id']}   decision: {report['decision']}")
    for reason in report["decision_reasons"]:
        out(f"  - {reason}")
    out(
        f"held-out n={ev['n']} accuracy base={acc['base']:.3f} student={acc['student']:.3f} "
        f"teacher={acc['teacher']:.3f}"
    )
    out(f"held-out sealed sha256: {report['data']['heldout_sealed_sha256']}")
    human = ev.get("human")
    if human:
        hacc = human["accuracy"]
        out(
            f"human held-out (Gate B) decision: {report.get('decision_human')} n={human['n']} "
            f"accuracy base={hacc['base']:.3f} student={hacc['student']:.3f} "
            f"teacher={hacc['teacher']:.3f}"
        )
        for reason in human["gate"]["reasons"]:
            out(f"  - {reason}")
    for rd in report["rounds"]:
        out(
            f"round {rd['round']}: train_rows={rd['train_rows']} dev_acc={rd['dev_acc']:.3f} "
            f"failures={rd['dev_failures']}"
        )
    out(
        f"rounds stopped: {report['rounds_stop_reason']}; "
        f"candidate round {report['candidate_round']}"
    )
    drops = {
        s: {k: v for k, v in c.items() if "drop" in k or "discard" in k or "error" in k}
        for s, c in report["counters"].items()
    }
    out("drop/error counters: " + json.dumps({s: c for s, c in drops.items() if c}, sort_keys=True))
    cost = report["cost"]
    out(
        f"llm+run spend usd (ESTIMATE, not a billed amount): {cost['run_total_usd']:.6f} "
        f"(cap {cost['run_cap_usd']})"
    )
    if cost.get("basis"):
        out(f"  {cost['basis']}")
    out(f"fine-tune cost: {cost['finetune_usd']}")
    out(f"student cost per 1k tasks: {cost['cost_per_1k_tasks']['student']}")
    if report.get("dry_run"):
        out(DRY_RUN_LABEL)


def _cmd_run(
    args: argparse.Namespace,
    env: Mapping[str, str],
    deps_factory: DepsFactory | None,
    out: Callable[[str], None],
    token_prompt: Callable[[], str] | None,
) -> int:
    root = _root(args, env)
    dry = bool(args.dry_run)
    run_id = args.run_id or (
        dry_run_id(args.scale)
        if dry
        else f"sql-{args.scale}-{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}"
    )
    try:
        config = load_config(env)
    except ConfigError as exc:
        out(f"configuration error: {exc}")
        return EXIT_REFUSED

    if not dry:
        # 1. admin token (constant-time compare inside config.verify_admin_token)
        if not config.verify_admin_token(_supplied_token(env, token_prompt)):
            out("refused: a live (spending) run requires the admin token; none/incorrect supplied")
            return EXIT_REFUSED
        # 2. explicit first-time spend approval, remembered per run id
        store = _store_for(root, run_id)
        marker = store.run_dir(run_id) / "spend_approved.json"
        if not marker.exists():
            if not args.i_approve_spend:
                out("refused: first start of a spending run requires --i-approve-spend")
                store.close()
                return EXIT_REFUSED
            atomic_write_bytes(marker, json.dumps({"run_id": run_id}).encode())
        store.close()
        if args.budget_usd is not None:
            config = config.model_copy(update={"run_cap_usd": args.budget_usd})
    elif args.budget_usd is not None:
        out("note: --budget-usd is ignored in a dry run (no spend)")

    human_set = Path(args.human_set) if args.human_set else None
    if human_set is not None and not human_set.is_file():
        out(f"refused: human set file not found: {human_set}")
        return EXIT_REFUSED
    pcfg = PipelineConfig(
        human_set_path=human_set,
        scale=SCALES[args.scale],
        dry_run=dry,
        # the real live path serves base and student on sandbox CPU; a dry run or a caller-supplied
        # Deps (tests, library use) brings its own serving factories
        student_serving="injected" if dry or deps_factory is not None else "sandbox_cpu",
        seed=args.seed,
        max_rounds=args.max_rounds,
        headroom_max_base_acc=args.max_base_acc,
        finetune_estimate_usd=args.finetune_estimate_usd,
        **({} if args.min_planned_steps is None else {"min_planned_steps": args.min_planned_steps}),
    )
    store = _store_for(root, run_id)
    try:
        with AsyncBridge() as bridge:
            if dry:
                dr = build_dry_run(
                    pcfg.scale, bridge, seed=args.seed, max_rounds=args.max_rounds,
                    headroom_max_base_acc=args.max_base_acc, human_set_path=human_set,
                )  # fmt: skip
                config, pcfg, deps = dr.config, dr.pipeline_cfg, dr.deps
            elif deps_factory is not None:
                deps = deps_factory(config, pcfg, bridge)
            else:
                deps = make_live_deps(config, pcfg, bridge, env=env)
            with driving(store.run_dir(run_id)):
                report = Pipeline(pcfg, config, deps, store, run_id, say=out).run()
    except (ConfigRefusal, ConfigError, humanset.HumanSetError) as exc:
        out(f"refused: {exc}")
        return EXIT_REFUSED
    except PipelineError as exc:
        out(f"FAILED: {type(exc).__name__}: {exc}")
        return EXIT_FAIL
    finally:
        store.close()
    _summary(report, out)
    out(f"report: {store.run_dir(run_id) / 'report.json'}")
    return EXIT_OK


def _cmd_status(
    args: argparse.Namespace, env: Mapping[str, str], out: Callable[[str], None]
) -> int:
    root = _root(args, env)
    store = _store_for(root, args.run)
    try:
        if not store.run_dir(args.run).exists():
            out(f"no such run: {args.run}")
            return EXIT_FAIL
        if args.run.startswith(DRY_PREFIX):
            out(DRY_RUN_LABEL)
        for s in stage_rows(store, args.run):
            err = f"  error: {s['error']}" if s["error"] else ""
            out(f"{s['stage']:<22} {s['status']:<9} {s['updated_at']}{err}")
        out(
            f"spend usd (ESTIMATE from the configured price table, see report config.prices for "
            f"the source; each fine-tune/sandbox line below states its basis): "
            f"{store.total_spend(args.run):.6f}"
        )
        for model, c in cost_by_model(store, args.run).items():
            out(f"  {model}: calls={c['calls']} usd={c['usd']:.6f}")
        for line in spend_lines(store, args.run):
            out(f"  {line['kind']} {line['model']}: usd={line['usd']:.6f} [{line['basis']}]")
    finally:
        store.close()
    return EXIT_OK


def _cmd_report(
    args: argparse.Namespace, env: Mapping[str, str], out: Callable[[str], None]
) -> int:
    root = _root(args, env)
    store = _store_for(root, args.run)
    try:
        path = store.run_dir(args.run) / "report.json"
        if not path.exists():
            out(f"no report for run {args.run} (not finished?); try: distillery status {args.run}")
            return EXIT_FAIL
        text = path.read_text(encoding="utf-8")
    finally:
        store.close()
    if args.json:
        out(text)
    else:
        _summary(json.loads(text), out)
    return EXIT_OK


def _cmd_reprice(
    args: argparse.Namespace, env: Mapping[str, str], out: Callable[[str], None]
) -> int:
    from distillery.config import load_price_file
    from distillery.reprice import RepriceError, render, reprice_run

    root = _root(args, env)
    store = _store_for(root, args.run)
    try:
        prices = load_price_file(args.prices)
        rp = reprice_run(
            store,
            args.run,
            prices,
            prices_label=str(args.prices),
            balance_before=args.balance_before,
            balance_after=args.balance_after,
            write=not args.no_write,
        )
    except (ConfigError, RepriceError) as exc:
        out(f"refused: {exc}")
        return EXIT_REFUSED
    finally:
        store.close()
    for line in render(rp):
        out(line)
    return EXIT_OK


def _cmd_serve(args: argparse.Namespace, env: Mapping[str, str]) -> int:
    import uvicorn

    from distillery.server import create_app, settings_from_env

    app = create_app(settings_from_env(env, root=_root(args, env)))
    uvicorn.run(app, host=args.host, port=args.port)
    return EXIT_OK


def _cmd_draft_heldout(
    args: argparse.Namespace,
    env: Mapping[str, str],
    deps_factory: DepsFactory | None,
    out: Callable[[str], None],
    token_prompt: Callable[[], str] | None,
) -> int:
    """Teacher drafts gold SQL for the human questions. Spends, so: admin token plus
    --i-approve-spend every time, a ledger cap, and spend recorded under ``humanset-<sha8>``."""
    root = _root(args, env)
    try:
        config = load_config(env)
        teacher = config.require_model("teacher")
    except ConfigError as exc:
        out(f"configuration error: {exc}")
        return EXIT_REFUSED
    if not config.verify_admin_token(_supplied_token(env, token_prompt)):
        out("refused: drafting spends money and requires the admin token; none/incorrect supplied")
        return EXIT_REFUSED
    if not args.i_approve_spend:
        out("refused: drafting requires --i-approve-spend")
        return EXIT_REFUSED
    try:
        qf = load_question_file(args.questions)
    except QuestionFileError as exc:
        out(f"refused: {exc}")
        return EXIT_REFUSED
    if args.budget_usd is not None:
        config = config.model_copy(update={"run_cap_usd": args.budget_usd})
    set_id = qf.file_sha256[:8]
    run_id = f"{humanset.RUN_PREFIX}{set_id}"
    db = sql_schema.build_database(args.db_seed)
    db_path = humanset.set_dir(root, set_id) / "db.sqlite"
    atomic_write_bytes(db_path, db)
    pcfg = PipelineConfig(student_serving="injected" if deps_factory is not None else "sandbox_cpu")
    store = Store(root)
    try:
        with AsyncBridge() as bridge:
            deps = (
                deps_factory(config, pcfg, bridge)
                if deps_factory is not None
                else make_live_deps(config, pcfg, bridge, env=env)
            )
            ledger = Ledger.from_config(run_id, config, store)
            llm_kwargs: dict[str, Any] = {}
            if deps.llm_sleep is not None:
                llm_kwargs["sleep"] = deps.llm_sleep
            llm = LLMClient(
                deps.transport, config.model_ids, sink=LedgerSink(ledger),
                pricing=ledger.estimate_llm_cost, **llm_kwargs,
            )  # fmt: skip
            runner = LLMRunner(llm, ledger, bridge, pcfg)
            out(
                f"drafting {len(qf.questions)} questions x {humanset.K_CANDIDATES} "
                f"(teacher {teacher})"
            )
            drafts = humanset.draft_questions(
                qf, runner=runner, executor=deps.executor, db_ref=str(db_path),
                schema_ddl=sql_schema.schema_ddl(args.db_seed), db_sha256=sha256_hex(db),
                teacher_model=teacher,
            )  # fmt: skip
            spent = ledger.spent()
    except BudgetExceeded as exc:
        out(f"refused: budget exceeded before drafting finished: {exc}")
        return EXIT_REFUSED
    except (ConfigRefusal, ConfigError) as exc:
        out(f"refused: {exc}")
        return EXIT_REFUSED
    finally:
        store.close()
    path = humanset.write_drafts(root, drafts)
    out(f"set {set_id}: {len(drafts['kept'])} kept, {len(drafts['discarded'])} discarded")
    out("discarded by reason: " + json.dumps(drafts["discarded_by_reason"], sort_keys=True))
    out(f"spend usd (ESTIMATE, run {run_id}): {spent:.6f}")
    out(f"drafts: {path}")
    out(f"next: python -m distillery confirm-heldout --set {set_id}")
    return EXIT_OK


def _cmd_confirm_heldout(
    args: argparse.Namespace,
    env: Mapping[str, str],
    out: Callable[[str], None],
    ask: Callable[[str], str],
) -> int:
    root = _root(args, env)
    try:
        set_id = args.set
        if set_id is None:
            sets = humanset.list_sets(root)
            if not sets:
                out("no human sets found: run draft-heldout first")
                return EXIT_FAIL
            if len(sets) > 1:
                out(f"several human sets, pick one with --set: {', '.join(sets)}")
                return EXIT_FAIL
            set_id = sets[0]
        if args.finalize:
            path = humanset.finalize(root, set_id)
            counts = json.loads(path.read_text(encoding="utf-8"))["counts"]
            out(
                f"confirmed: {counts['confirmed']}, rejected: {counts['rejected']}, "
                f"skipped: {counts['skipped']}, undecided: {counts['undecided']}, "
                f"kept: {counts['kept']}, discarded: {counts['discarded']}"
            )
            out(f"wrote {path}")
            out(f"use it with: python -m distillery run --human-set {path} ...")
            return EXIT_OK
        summary = humanset.confirm_drafts(root, set_id, ask=ask, out=out)
    except humanset.HumanSetError as exc:
        out(f"error: {exc}")
        return EXIT_FAIL
    out(
        f"confirmed: {summary['confirmed']}, rejected: {summary['rejected']}, "
        f"skipped: {summary['skipped']}, undecided: {summary['undecided']}"
        + (" (quit early; rerun to resume)" if summary["quit"] else "")
    )
    return EXIT_OK


def _cmd_export_replay(
    args: argparse.Namespace, env: Mapping[str, str], out: Callable[[str], None]
) -> int:
    from distillery.server.reader import RunReader
    from distillery.server.replay import ExportError, export_bundle
    from distillery.server.settings import ServerSettings

    root = _root(args, env)
    try:
        config = load_config(env)
    except ConfigError as exc:
        out(f"configuration error: {exc}")
        return EXIT_REFUSED
    replay_dir = Path(args.replay_dir or env.get("DISTILLERY_REPLAY_DIR") or "replay")
    reader = RunReader(ServerSettings(root=root, config=config))
    try:
        dest = export_bundle(
            reader,
            args.run,
            replay_dir,
            allow_dry_run=args.allow_dry_run,
            recorded_at=datetime.now(UTC).isoformat(),
        )
    except ExportError as exc:
        out(f"refused: {exc}")
        return EXIT_REFUSED
    finally:
        reader.close()
    out(f"exported {args.run} -> {dest}")
    return EXIT_OK


def main(
    argv: Sequence[str] | None = None,
    *,
    env: Mapping[str, str] | None = None,
    deps_factory: DepsFactory | None = None,
    out: Callable[[str], None] = print,
    token_prompt: Callable[[], str] | None = None,
    ask: Callable[[str], str] = input,
) -> int:
    e: Mapping[str, str] = os.environ if env is None else env
    args = _parser().parse_args(argv)
    if args.cmd == "run":
        return _cmd_run(args, e, deps_factory, out, token_prompt)
    if args.cmd == "draft-heldout":
        return _cmd_draft_heldout(args, e, deps_factory, out, token_prompt)
    if args.cmd == "confirm-heldout":
        return _cmd_confirm_heldout(args, e, out, ask)
    if args.cmd == "serve":
        return _cmd_serve(args, e)
    if args.cmd == "export-replay":
        return _cmd_export_replay(args, e, out)
    if args.cmd == "reprice":
        return _cmd_reprice(args, e, out)
    if args.cmd == "status":
        return _cmd_status(args, e, out)
    return _cmd_report(args, e, out)
