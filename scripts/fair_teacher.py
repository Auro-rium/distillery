#!/usr/bin/env python3
"""B2: the fair teacher comparison (S0 / S1 / S2 / student) on the promoted run. Pre-registered in
DECISIONS.md (2026-10-05).

Modes (exactly one):
  --estimate   pre-flight cost table from real prompt token counts and the configured prices. No
               calls, no sealed read. Show it and get approval BEFORE any live run.
  --dry-run    the whole pipeline on fake models in --out: free, labelled FAKE; proves the path.
  --rescore    B3: re-execute and re-verify the cached outputs in OUT/vectors.json with the current
               verifier (re-execution only: no model is called); before/after for every setup.
  (default)    the LIVE run: spends. Needs --i-approve-spend and the admin token in
               DISTILLERY_ADMIN_TOKEN_SUPPLIED; run cap from DISTILLERY_RUN_CAP_USD or --budget-usd.

--root is the source store (e.g. .distillery/p5live) and is READ-ONLY: it is copied to OUT/store and
everything is written under OUT (keep OUT out of git: it holds a copy of the sealed set). The
promoted run is auto-detected when --root holds exactly one finished run.

Environment: model ids and prices come from the usual variables (DISTILLERY_MODEL_*,
DISTILLERY_PRICES_FILE, NEBIUS_*); --env-file loads KEY=VALUE lines for names it recognises only.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from distillery import fairteacher as ft
from distillery.config import ConfigError, load_config

_ENV_PREFIXES = ("DISTILLERY_", "NEBIUS_")


def _load_env(path: str | None) -> dict[str, str]:
    env = dict(os.environ)
    if path:
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                if k.startswith(_ENV_PREFIXES) and v.strip():
                    env.setdefault(k.strip(), v.strip())
    return env


def main(argv: list[str] | None = None) -> int:  # noqa: C901, PLR0911, PLR0912
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--root", help="source store (READ-ONLY), e.g. .distillery/p5live")
    ap.add_argument("--run-id", help="default: the single finished run under --root")
    ap.add_argument("--out", required=True, help="output dir (all writes go here)")
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument(
        "--estimate", action="store_true", help="print the pre-flight table; no calls"
    )
    mode.add_argument("--dry-run", action="store_true", help="fake models, free")
    mode.add_argument("--rescore", action="store_true", help="re-verify cached outputs; no calls")
    ap.add_argument("--human-n", type=int, default=75, help="estimate: assumed human-set size")
    ap.add_argument("--env-file", default=None)
    ap.add_argument("--budget-usd", type=float, default=None, help="live: per-run spend cap")
    ap.add_argument("--i-approve-spend", action="store_true")
    ap.add_argument(
        "--student-concurrency",
        type=int,
        default=None,
        help="sandbox generations in flight per model (default: the run's own); lower it when a "
        "live run shares the sandbox operation cap",
    )
    ap.add_argument(
        "--setups",
        default=",".join(ft.TEACHER_SETUPS),
        help="teacher setups to score, comma separated (default all; a left-out setup is recorded "
        "as deferred, DECISIONS.md 2026-10-06)",
    )
    a = ap.parse_args(argv)
    out = Path(a.out)

    if a.dry_run:
        out.mkdir(parents=True, exist_ok=True)
        ev = ft.dry_run(out)
        print(
            json.dumps({"label": ev["label"], "analysis": ev["analysis"]["claims"]}, indent=1)[
                :1500
            ]
        )
        print(f"dry run OK: evidence in {out / 'work' / 'b_fair_teacher.json'}")
        return 0

    if not a.root:
        ap.error("--root is required (except with --dry-run)")
    root = Path(a.root)
    env = _load_env(a.env_file)
    try:
        config = load_config(env)
        run_id = a.run_id or ft.find_run_id(root)
        if a.estimate:
            est = ft.estimate_table(root, run_id, config, human_n=a.human_n)
            out.mkdir(parents=True, exist_ok=True)
            (out / "estimate.json").write_text(json.dumps(est, indent=1), encoding="utf-8")
            print(ft.format_estimate(est))
            return 0
        from distillery.cli import make_live_deps
        from distillery.orchestrator import PipelineConfig
        from distillery.sandbox_executor import AsyncBridge

        report = json.loads((root / "runs" / run_id / "report.json").read_text(encoding="utf-8"))
        pcfg = PipelineConfig.model_validate(report["config"]["pipeline"])
        if not a.rescore:  # a live run spends: token and explicit approval every time
            if not config.verify_admin_token(env.get("DISTILLERY_ADMIN_TOKEN_SUPPLIED", "")):
                print(
                    "refused: a live run spends; set DISTILLERY_ADMIN_TOKEN_SUPPLIED",
                    file=sys.stderr,
                )
                return 2
            if not a.i_approve_spend:
                print(
                    "refused: a live run spends; pass --i-approve-spend after the estimate",
                    file=sys.stderr,
                )
                return 2
            if a.budget_usd is not None:
                config = config.model_copy(update={"run_cap_usd": a.budget_usd})
        with AsyncBridge() as bridge:
            deps = make_live_deps(config, pcfg, bridge, env=env)
            if a.rescore:
                res = ft.rescore_from_cache(out, run_id, config, deps)
            else:
                names = tuple(x.strip() for x in a.setups.split(",") if x.strip())
                if not names or any(x not in ft.TEACHER_SETUPS for x in names):
                    print(
                        f"refused: --setups must name some of {ft.TEACHER_SETUPS}", file=sys.stderr
                    )
                    return 2
                res = ft.run_fair_teacher(
                    root, out, run_id, config, deps, teacher_names=names,
                    student_concurrency=a.student_concurrency,
                )  # fmt: skip
        print(json.dumps(res, indent=1, default=str)[:4000])
        return 0
    except (ft.FairTeacherRefusal, ConfigError) as exc:
        print(f"refused: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
