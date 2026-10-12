#!/usr/bin/env python3
"""N1 natural-phrasing training rows (DECISIONS.md 2026-10-07). Library: distillery.naturaltrain.

Subcommands:
  draft   the teacher drafts SQL (k=3, temperature 0.8, the training prompt) for every authored
          question; a row is kept only if all drafts execute, are non-empty, agree AND equal the
          reference SQL's result. SPENDS: needs the admin token (DISTILLERY_ADMIN_TOKEN_SUPPLIED or
          a hidden prompt), --i-approve-spend every time and a --budget-usd cap; the preflight
          estimate is printed and the draft refuses before any call when it exceeds the cap.
          Spend is recorded in the --root ledger under naturaltrain-<questions sha8>.
  check   offline overlap report (exact normalised match + held-out skeleton measure) between a
          natural_train.json and sealed question files. No calls.

Use the kept file with: python -m distillery run ... --natural-train natural_train.json
  --natural-exclude <sealed question file> [...]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from distillery import naturaltrain as nt
from distillery.budget import BudgetExceeded, Ledger, UnknownPriceError
from distillery.cli import DepsFactory, _supplied_token, make_live_deps
from distillery.config import ConfigError, load_config
from distillery.llm import LLMClient
from distillery.orchestrator import ConfigRefusal, LedgerSink, LLMRunner, PipelineConfig
from distillery.sandbox_executor import AsyncBridge
from distillery.store import Store, sha256_hex
from distillery.taskpacks.base import get_pack
from distillery.taskpacks.sql import schema as sql_schema
from distillery.taskpacks.sql.human import QuestionFileError, load_question_file

EXIT_OK, EXIT_FAIL, EXIT_REFUSED = 0, 1, 2
_ENV_PREFIXES = ("DISTILLERY_", "NEBIUS_")


def _load_env(base: Mapping[str, str], path: str | None) -> dict[str, str]:
    env = dict(base)
    if path:
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                if k.startswith(_ENV_PREFIXES) and v.strip():
                    env.setdefault(k.strip(), v.strip())
    return env


def _parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("draft", help="teacher-draft and reference-verify rows (spends)")
    d.add_argument("--questions", required=True, help="question file, one per line")
    d.add_argument("--reference", required=True, help="reference.jsonl: i, question, reference_sql")
    d.add_argument("--db", required=True, help="the run db.sqlite (checked vs --db-seed)")
    d.add_argument("--db-seed", type=int, default=0)
    d.add_argument("--out", required=True, help="where to write natural_train.json")
    d.add_argument("--root", default=None, help="ledger store (default $DISTILLERY_HOME)")
    d.add_argument("--budget-usd", type=float, required=True, help="hard cap for this drafting")
    d.add_argument("--i-approve-spend", action="store_true")
    d.add_argument("--env-file", default=None)
    c = sub.add_parser("check", help="offline overlap report against sealed question files")
    c.add_argument("--natural", required=True, help="natural_train.json")
    c.add_argument("--sealed", action="append", required=True, help="sealed file (repeatable)")
    c.add_argument("--out", default=None, help="also write the JSON report here")
    return ap


def _cmd_draft(  # noqa: C901, PLR0911
    a: argparse.Namespace,
    env: Mapping[str, str],
    deps_factory: DepsFactory | None,
    out: Callable[[str], None],
    token_prompt: Callable[[], str] | None,
) -> int:
    try:
        config = load_config(env)
        teacher = config.require_model("teacher")
    except ConfigError as exc:
        out(f"configuration error: {exc}")
        return EXIT_REFUSED
    if not config.verify_admin_token(_supplied_token(env, token_prompt)):
        out("refused: drafting spends money and requires the admin token; none/incorrect supplied")
        return EXIT_REFUSED
    if not a.i_approve_spend:
        out("refused: drafting requires --i-approve-spend")
        return EXIT_REFUSED
    if a.budget_usd <= 0:
        out("refused: --budget-usd must be positive")
        return EXIT_REFUSED
    try:
        qf = load_question_file(a.questions)
        refs, ref_sha = nt.load_references(a.reference)
        missing = [q.question for q in qf.questions if q.question not in refs]
        if missing:
            raise nt.NaturalTrainError(f"{len(missing)} question(s) have no reference SQL")
        db_path = Path(a.db)
        db_sha = nt.file_sha256(db_path)
    except (QuestionFileError, nt.NaturalTrainError) as exc:
        out(f"refused: {exc}")
        return EXIT_REFUSED
    if sha256_hex(sql_schema.build_database(a.db_seed)) != db_sha:
        out(f"refused: {db_path} is not the database built from --db-seed {a.db_seed}")
        return EXIT_REFUSED
    ddl = sql_schema.schema_ddl(a.db_seed)
    config = config.model_copy(update={"run_cap_usd": a.budget_usd})
    run_id = f"{nt.RUN_PREFIX}{qf.file_sha256[:8]}"
    pcfg = PipelineConfig(student_serving="injected" if deps_factory is not None else "sandbox_cpu")
    root = Path(a.root or env.get("DISTILLERY_HOME") or ".distillery")
    store = Store(root)
    try:
        ledger = Ledger.from_config(run_id, config, store, env=env)
        try:
            est = nt.estimate_usd(
                [q.question for q in qf.questions], ddl,
                lambda i, o: ledger.estimate_llm_cost(teacher, i, o),
                chars_per_token=pcfg.chars_per_token, out_tokens_per_call=pcfg.est_output_tokens,
            )  # fmt: skip
        except UnknownPriceError as exc:
            out(f"refused: {exc}")
            return EXIT_REFUSED
        before = ledger.spent()
        out(
            f"preflight: {est['calls']} teacher calls ({teacher}), estimate ${est['usd']:.4f} "
            f"[{est['basis']}]; cap ${a.budget_usd:.4f}, already spent under {run_id} "
            f"${before:.4f}"
        )
        if est["usd"] > a.budget_usd - before:
            out("refused: the estimate exceeds the remaining budget; nothing was called")
            return EXIT_REFUSED
        with AsyncBridge() as bridge:
            deps = (
                deps_factory(config, pcfg, bridge)
                if deps_factory is not None
                else make_live_deps(config, pcfg, bridge, env=env)
            )
            llm_kwargs: dict[str, Any] = {}
            if deps.llm_sleep is not None:
                llm_kwargs["sleep"] = deps.llm_sleep
            llm = LLMClient(
                deps.transport, config.model_ids, sink=LedgerSink(ledger),
                pricing=ledger.estimate_llm_cost, **llm_kwargs,
            )  # fmt: skip
            runner = LLMRunner(llm, ledger, bridge, pcfg)
            doc = nt.draft_natural(
                qf, refs, runner=runner, executor=deps.executor, db_ref=str(db_path),
                schema_ddl=ddl, db_sha256=db_sha, teacher_model=teacher,
            )  # fmt: skip
            spent = ledger.spent() - before
    except BudgetExceeded as exc:
        out(f"refused: budget exceeded before drafting finished: {exc}")
        return EXIT_REFUSED
    except (ConfigRefusal, ConfigError, nt.NaturalTrainError) as exc:
        out(f"refused: {exc}")
        return EXIT_REFUSED
    finally:
        store.close()
    doc = {
        **doc,
        "questions_sha256": qf.file_sha256,
        "reference_sha256": ref_sha,
        "spend": {
            "run_id": run_id,
            "usd": spent,
            "estimate": est,
            "budget_usd": a.budget_usd,
            "basis": "ESTIMATE: measured tokens x configured price table, not a billed amount",
        },
    }
    path = nt.write_natural(a.out, doc)
    out(f"kept {len(doc['kept'])} of {doc['n_questions']} questions")
    out("discards by reason: " + json.dumps(doc["discards"], sort_keys=True))
    out(f"spend usd (ESTIMATE, ledger run {run_id}): {spent:.6f}")
    out(f"wrote {path}")
    out(f"next: python {Path(__file__).name} check --natural {path} --sealed <sealed file> ...")
    return EXIT_OK


def _cmd_check(a: argparse.Namespace, out: Callable[[str], None]) -> int:
    try:
        rep = nt.check_overlap(a.natural, a.sealed, get_pack("sql").skeleton)
    except nt.NaturalTrainError as exc:
        out(f"refused: {exc}")
        return EXIT_REFUSED
    text = json.dumps(rep, indent=2, sort_keys=True, ensure_ascii=False)
    if a.out:
        Path(a.out).write_text(text + "\n", encoding="utf-8")
    out(text)
    for name, s in rep["sets"].items():
        if s["exact_overlap"]:
            out(f"WARNING: {s['exact_overlap']} exact overlap(s) with {name}")
    return EXIT_OK


def main(
    argv: Sequence[str] | None = None,
    *,
    env: Mapping[str, str] | None = None,
    deps_factory: DepsFactory | None = None,
    out: Callable[[str], None] = print,
    token_prompt: Callable[[], str] | None = None,
) -> int:
    a = _parser().parse_args(argv)
    if a.cmd == "check":
        return _cmd_check(a, out)
    e = _load_env(os.environ if env is None else env, a.env_file)
    return _cmd_draft(a, e, deps_factory, out, token_prompt)


if __name__ == "__main__":
    sys.exit(main())
