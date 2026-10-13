#!/usr/bin/env python3
"""H1: serve Qwen3-1.7B + the N1 LoRA on a Nebius AI Cloud endpoint (L40S), measure it, delete it.

Two modes:

  --plan-only         prints the exact `nebius ai endpoint ...` commands (token redacted) and the
                      preflight estimate, then exits. No CLI call, no network, no spend.
  (live)              REQUIRES --i-approve-spend. Creates the endpoint, measures throughput/latency
                      and parity, ALWAYS deletes it (and confirms via `list`), records the spend in
                      the ledger and writes docs/proofs/evidence/h1_endpoint.json (+ the raw
                      per-request samples in h1_endpoint_samples.jsonl beside it).

Inputs: --prompts is a JSON/JSONL file of test-split items ({question, schema_ddl, gold_sql,
requires_order}); --sandbox-outputs is the sandbox student's outputs for the same items (a JSON
list of strings, same order); --db is the SQLite file the items run against (accuracy/parity).
--root is the store holding the ledger (and the teacher's cost rows for --teacher-model).

  python scripts/h1_endpoint.py --plan-only --vllm-tag vX.Y.Z --hourly-usd 1.90 \
      --adapter-s3 s3://BUCKET --max-lora-rank 64

Exit codes: 0 ok, 1 failed (endpoint deleted), 2 DELETE NOT CONFIRMED (may still be billing),
3 usage / refused.
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from distillery import aicloud_endpoint as ae
from distillery import h1_measure as hm
from distillery.budget import BudgetExceeded, Ledger
from distillery.config import ConfigError, load_config
from distillery.store import Store
from distillery.taskpacks.sql.executor import LocalExecutor

EXIT_OK, EXIT_ERROR, EXIT_DELETE_UNCONFIRMED, EXIT_USAGE = 0, 1, 2, 3
DEFAULT_OUT_DIR = "docs/proofs/evidence"
EVIDENCE_NAME = "h1_endpoint.json"
SAMPLES_NAME = "h1_endpoint_samples.jsonl"
# The delete reserve (default 5 min) must cover: one in-flight request at the deadline
# (REQUEST_TIMEOUT_S), the delete call (seconds), and the `list` confirmation (<= 0.6 x reserve).
# A hung delete subprocess (180 s per attempt, 3 attempts) can still overrun; that path raises.
REQUEST_TIMEOUT_S = 60.0

ClientFactory = Callable[[ae.EndpointHandle], hm.ChatClient]


def _ints(text: str) -> tuple[int, ...]:
    return tuple(int(x) for x in text.split(",") if x.strip())


def _floats(text: str) -> tuple[float, ...]:
    return tuple(float(x) for x in text.split(",") if x.strip())


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    g = ap.add_argument_group("endpoint plan")
    g.add_argument("--vllm-tag", required=True, help="explicit vllm/vllm-openai tag (not latest)")
    g.add_argument("--hourly-usd", type=float, required=True, help="console price of the endpoint")
    g.add_argument("--adapter-s3", required=True, help="s3://BUCKET[/prefix] holding the adapter")
    g.add_argument("--adapter-s3-profile", default=None)
    g.add_argument(
        "--max-lora-rank",
        type=int,
        default=ae.DEFAULT_MAX_LORA_RANK,
        help="N1 adapter_config.json 'r' (default %(default)s)",
    )
    g.add_argument("--platform", default=ae.DEFAULT_PLATFORM)
    g.add_argument("--preset", default=None)
    g.add_argument("--preemptible", action="store_true")
    g.add_argument("--disk-size", default="100Gi")
    g.add_argument("--name", default=None, help="endpoint name (default: random suffix)")
    g.add_argument("--time-box-minutes", type=float, default=45.0)
    g.add_argument("--ready-timeout-minutes", type=float, default=25.0)
    g.add_argument("--delete-reserve-minutes", type=float, default=5.0)
    ap.add_argument("--plan-only", action="store_true", help="print commands + estimate and exit")
    ap.add_argument("--i-approve-spend", action="store_true", help="REQUIRED for the live path")
    m = ap.add_argument_group("measurement")
    m.add_argument("--prompts", default=None, help="JSON/JSONL test-split items")
    m.add_argument("--sandbox-outputs", default=None, help="sandbox student outputs (JSON list)")
    m.add_argument("--db", default=None, help="SQLite file the items run against")
    m.add_argument(
        "--schema-ddl",
        default=None,
        help="file with the schema DDL, for items that carry no schema_ddl of their own",
    )
    m.add_argument("--concurrency", type=_ints, default=hm.DEFAULT_LEVELS)
    m.add_argument("--max-tokens", type=int, default=hm.DEFAULT_MAX_TOKENS)
    c = ap.add_argument_group("cost (projections)")
    c.add_argument("--root", default=None, help="store root: the ledger + teacher cost source")
    c.add_argument("--run-id", default="h1-endpoint")
    c.add_argument("--teacher-model", default=None)
    c.add_argument("--teacher-root", default=None, help="read-only store for teacher cost rows")
    c.add_argument("--teacher-run-id", default=None)
    c.add_argument("--teacher-limit", type=int, default=None)
    c.add_argument("--coverage", type=float, default=None, help="share routed to the student")
    c.add_argument("--audit-rate", type=float, default=0.05)
    c.add_argument("--volumes", type=_floats, default=(1_000.0, 10_000.0, 100_000.0, 1_000_000.0))
    ap.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    return ap


def plan_from_args(a: argparse.Namespace) -> ae.EndpointPlan:
    kw: dict[str, Any] = {"name": a.name} if a.name else {}
    return ae.EndpointPlan(
        vllm_tag=a.vllm_tag,
        hourly_usd=a.hourly_usd,
        adapter_s3_source=a.adapter_s3,
        adapter_s3_profile=a.adapter_s3_profile,
        max_lora_rank=a.max_lora_rank,
        platform=a.platform,
        preset=a.preset,
        preemptible=a.preemptible,
        disk_size=a.disk_size,
        time_box_minutes=a.time_box_minutes,
        ready_timeout_minutes=a.ready_timeout_minutes,
        delete_reserve_minutes=a.delete_reserve_minutes,
        **kw,
    )


def plan_only_text(plan: ae.EndpointPlan, headroom: float | None = None) -> str:
    tok = plan.auth_token
    lines = [
        "H1 AI Cloud endpoint plan (plan-only: nothing was run, no network, no spend)",
        f"  image              {plan.image}",
        f"  name               {plan.name}",
        f"  platform / preset  {plan.platform} / {plan.preset or '(default)'}",
        f"  capacity           {'preemptible' if plan.preemptible else 'on-demand'}",
        f"  base model         {plan.base_model}",
        f"  adapter            {plan.served_model_name} <- {plan.volume()}",
        f"  max LoRA rank      {plan.max_lora_rank}",
        f"  auth token         {ae.REDACTED} (generated per run, never printed)",
        f"  hourly price       ${plan.hourly_usd:.4f}/h (operator-supplied)",
        f"  time box           {plan.time_box_minutes:g} min (hard, create -> confirmed delete)",
        f"  measuring stops    at {plan.time_box_minutes - plan.delete_reserve_minutes:g} min "
        f"(last {plan.delete_reserve_minutes:g} min reserved for delete + confirm)",
        f"  preflight estimate ${plan.estimated_cost_usd:.4f} "
        f"(= ${plan.hourly_usd:.4f}/h x {plan.time_box_minutes:g}/60)",
    ]
    if headroom is not None:
        lines.append(f"  ledger headroom    ${headroom:.4f}")
    lines += [
        "commands (get/list/delete are UNVERIFIED: check each with --help before the live run):",
        f"  create  {ae.render_argv(ae.create_argv(plan), tok)}",
        f"  get     {ae.render_argv(ae.get_argv('<ID>'))}",
        f"  list    {ae.render_argv(ae.list_argv())}",
        f"  delete  {ae.render_argv(ae.delete_argv('<ID>'))}",
    ]
    return "\n".join(lines)


@dataclass
class Prepared:
    items: list[dict[str, Any]]
    prompts: list[hm.ChatMessages]
    sandbox: list[str] | None
    teacher: dict[str, Any] | None


class Refused(Exception):
    """An input is unusable; raised BEFORE anything is created or spent."""


def prepare(a: argparse.Namespace) -> Prepared:
    """Load and validate every input while it is still free to refuse."""
    try:
        items = hm.load_items(Path(a.prompts))
        ddl = Path(a.schema_ddl).read_text(encoding="utf-8") if a.schema_ddl else None
        prompts = hm.build_prompts(items, default_schema_ddl=ddl)
        sandbox = hm.load_outputs(Path(a.sandbox_outputs)) if a.sandbox_outputs else None
    except (OSError, ValueError) as exc:
        raise Refused(f"cannot read the inputs: {exc}") from exc
    if not items:
        raise Refused("--prompts holds no items")
    if sandbox is not None and len(sandbox) != len(items):
        raise Refused(f"--sandbox-outputs has {len(sandbox)} outputs for {len(items)} items")
    if sandbox is not None and not a.db:
        raise Refused("--sandbox-outputs needs --db (parity executes both sides)")
    if a.db:
        if not Path(a.db).is_file():
            raise Refused(f"--db {a.db} is not a file")
        bad = [
            i for i, it in enumerate(items) if "gold_sql" not in it or "requires_order" not in it
        ]
        if bad:
            raise Refused(f"items {bad[:5]} lack gold_sql/requires_order (needed with --db)")
    if not a.concurrency or any(lv < 1 for lv in a.concurrency):
        raise Refused("--concurrency levels must be >= 1")
    if a.coverage is not None and not 0 <= a.coverage <= 1:
        raise Refused("--coverage must be in [0, 1]")
    if not 0 <= a.audit_rate <= 1:
        raise Refused("--audit-rate must be in [0, 1]")
    if any(v <= 0 for v in a.volumes):
        raise Refused("--volumes must be positive")
    if a.coverage is not None and not a.teacher_model:
        raise Refused("--coverage needs --teacher-model (the blended curve compares to it)")
    teacher = None
    if a.teacher_model:
        root = a.teacher_root or a.root
        if not root:
            raise Refused("--teacher-model needs --teacher-root or --root")
        try:  # read-only and free: a missing teacher must not be discovered after the spend
            teacher = hm.teacher_cost_per_1k(
                Path(root), a.teacher_model, a.teacher_run_id, a.teacher_limit
            )
        except (SystemExit, OSError, sqlite3.Error) as exc:
            raise Refused(f"teacher cost unavailable for {a.teacher_model!r}: {exc}") from exc
    return Prepared(items, prompts, sandbox, teacher)


def costs_block(
    a: argparse.Namespace,
    result: hm.MeasureResult | None,
    plan: ae.EndpointPlan,
    teacher: dict[str, Any] | None,
) -> dict[str, Any]:
    out: dict[str, Any] = {"hourly_usd": plan.hourly_usd, "projected": True}
    rph = hm.best_req_per_hour(result) if result else None
    if rph is None:
        out["note"] = "no error-free throughput level: no cost projection"
        return out
    student = hm.cost_per_1k(plan.hourly_usd, rph)
    out["student"] = student
    if teacher is None:
        return out
    out["teacher"] = teacher
    out["student_vs_teacher_per_1k_ratio"] = student["usd_per_1k"] / teacher["usd_per_1k"]
    if a.coverage is not None:
        t1k, capacity = teacher["usd_per_1k"], rph * 24
        out["blended_busy"] = hm.blended_curve(
            a.coverage,
            student["usd_per_1k"],
            t1k,
            a.audit_rate,
            a.volumes,
            student_capacity_per_day=capacity,
        )
        out["blended_always_on"] = hm.blended_curve(
            a.coverage,
            0.0,
            t1k,
            a.audit_rate,
            a.volumes,
            student_fixed_daily_usd=24 * plan.hourly_usd,
            student_capacity_per_day=capacity,
        )
    return out


def accuracy_and_parity(
    a: argparse.Namespace,
    result: hm.MeasureResult,
    items: Sequence[Mapping[str, Any]],
    sandbox: Sequence[str] | None,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    """(endpoint accuracy, parity) from the first concurrency level's outputs; offline."""
    if not a.db or not result.levels:
        return None, None
    outputs = result.outputs_at(result.levels[0].concurrency, len(items))
    if outputs is None:
        return {"note": "first level has missing/failed requests: no accuracy"}, None
    ex = LocalExecutor()
    db = str(Path(a.db).resolve())
    acc = hm.accuracy(outputs, items, executor=ex, db_ref=db)
    acc.pop("correct")
    par = (
        hm.parity(outputs, sandbox, items, executor=ex, db_ref=db) if sandbox is not None else None
    )
    return acc, par


def write_evidence(out_dir: Path, evidence: dict[str, Any], samples: list[dict[str, Any]]) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / SAMPLES_NAME).write_text(
        "".join(json.dumps(s, sort_keys=True) + "\n" for s in samples), encoding="utf-8"
    )
    path = out_dir / EVIDENCE_NAME
    path.write_text(json.dumps(evidence, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    return path


_ANALYSIS_ERRORS = (ValueError, KeyError, TypeError, OSError, sqlite3.Error, ZeroDivisionError)


def run_live(
    a: argparse.Namespace,
    plan: ae.EndpointPlan,
    ledger: Ledger,
    *,
    runner: ae.CommandRunner,
    client_factory: ClientFactory,
    clock: ae.Clock,
    out: Callable[[str], None] = print,
) -> int:
    try:
        prep = prepare(a)  # every input is validated BEFORE the first paid call
    except Refused as exc:
        out(f"refused: {exc}")
        return EXIT_USAGE
    spent_before = ledger.spent()
    result: hm.MeasureResult | None = None
    handle: ae.EndpointHandle | None = None
    created: list[str] = []
    error: str | None = None
    delete_unconfirmed = False
    try:
        with ae.aicloud_endpoint(
            plan,
            runner,
            ledger,
            clock,
            probe=lambda h: client_factory(h).lists_model(),
            on_created=created.append,
        ) as handle:
            result = hm.measure_throughput(
                client_factory(handle),
                prep.prompts,
                a.concurrency,
                a.max_tokens,
                monotonic=clock.monotonic,
                should_stop=handle.expired,
            )
    except ae.EndpointDeleteError as exc:
        error, delete_unconfirmed = str(exc), True
    except (ae.AICloudEndpointError, BudgetExceeded, hm.ChatError) as exc:
        error = f"{type(exc).__name__}: {exc}"
    except BaseException as exc:  # the artifact is still written; the endpoint is already deleted
        error = f"{type(exc).__name__}: {exc}"
        _write(a, plan, ledger, spent_before, result, handle, created, error, prep, None, None, [])
        raise

    acc = par = None
    analysis_errors: list[str] = []
    if result is not None and not delete_unconfirmed:
        try:  # the money is spent and the endpoint is gone: never lose the measurements
            acc, par = accuracy_and_parity(a, result, prep.items, prep.sandbox)
        except _ANALYSIS_ERRORS as exc:
            analysis_errors.append(f"accuracy/parity: {type(exc).__name__}: {exc}")
    path = _write(
        a, plan, ledger, spent_before, result, handle, created, error, prep, acc, par,
        analysis_errors,
    )  # fmt: skip
    out(f"wrote {path}")
    if delete_unconfirmed:
        out(f"DELETE NOT CONFIRMED: {error}")
        return EXIT_DELETE_UNCONFIRMED
    if error:
        out(f"FAILED (endpoint deleted): {error}")
        return EXIT_ERROR
    return EXIT_OK


def _write(
    a: argparse.Namespace,
    plan: ae.EndpointPlan,
    ledger: Ledger,
    spent_before: float,
    result: hm.MeasureResult | None,
    handle: ae.EndpointHandle | None,
    created: list[str],
    error: str | None,
    prep: Prepared,
    acc: dict[str, Any] | None,
    par: dict[str, Any] | None,
    analysis_errors: list[str],
) -> Path:
    if handle is not None:
        lifecycle = handle.lifecycle()
    else:  # never came up: the id (if the create answered) and the delete verdict are all we have
        lifecycle = {
            "endpoint_id": created[0] if created else None,
            "name": plan.name,
            "deleted_confirmed": error is not None and "may still be billing" not in error,
            "error": error,
        }
    try:
        costs = costs_block(a, result, plan, prep.teacher)
    except _ANALYSIS_ERRORS as exc:
        costs = {"projected": True, "error": f"{type(exc).__name__}: {exc}"}
        analysis_errors.append(f"costs: {type(exc).__name__}: {exc}")
    spent_after = ledger.spent()
    evidence = {
        "proof": "H1",
        "plan": plan.public_dict(),
        "lifecycle": lifecycle,
        "measurements": result.to_json() if result else None,
        "endpoint_accuracy": acc,
        "parity": par,
        "costs": costs,
        "ledger": {
            "run_id": ledger.run_id,
            "kind": ae.KIND_AICLOUD_ENDPOINT,
            "spent_before_usd": spent_before,
            "spent_after_usd": spent_after,
            "delta_usd": spent_after - spent_before,
        },
        "n_items": len(prep.items),
        "error": error,
        "analysis_errors": analysis_errors,
    }
    return write_evidence(Path(a.out_dir), evidence, result.samples if result else [])


def main(
    argv: Sequence[str] | None = None,
    *,
    env: Mapping[str, str] | None = None,
    runner: ae.CommandRunner | None = None,
    client_factory: ClientFactory | None = None,
    clock: ae.Clock | None = None,
    ledger: Ledger | None = None,
    out: Callable[[str], None] = print,
) -> int:
    a = build_parser().parse_args(argv)
    env = os.environ if env is None else env
    try:
        plan = plan_from_args(a)
    except ValidationError as exc:
        out(f"refused: invalid plan: {exc}")
        return EXIT_USAGE
    try:
        lg = ledger if ledger is not None else _make_ledger(a, env)
    except ConfigError as exc:
        out(f"refused: {exc}")
        return EXIT_USAGE
    if a.plan_only:
        out(plan_only_text(plan, lg.remaining() if lg is not None else None))
        if lg is not None:
            try:
                lg.preflight(plan.estimated_cost_usd)
                out("  preflight          would PASS")
            except BudgetExceeded as exc:
                out(f"  preflight          would be REFUSED: {exc}")
        return EXIT_OK
    if not a.i_approve_spend:
        out("refused: the live path spends money; pass --i-approve-spend (or use --plan-only)")
        return EXIT_USAGE
    if lg is None or not a.prompts:
        out("refused: the live path needs --root (the ledger) and --prompts")
        return EXIT_USAGE
    factory = client_factory or (
        lambda h: hm.HttpChatClient(h.base_url, h.token, timeout_s=REQUEST_TIMEOUT_S)
    )
    return run_live(
        a,
        plan,
        lg,
        runner=runner or ae.SubprocessRunner(),
        client_factory=factory,
        clock=clock or ae.Clock(),
        out=out,
    )


def _make_ledger(a: argparse.Namespace, env: Mapping[str, str]) -> Ledger | None:
    if not a.root:
        return None
    return Ledger.from_config(a.run_id, load_config(env), Store(Path(a.root)), env=env)


if __name__ == "__main__":
    raise SystemExit(main())
