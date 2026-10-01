"""Read-only, no-spend diagnostics of a finished run. Prints JSON.

  jobs     RUN_ID [--job-id ID ...]   raw fine-tune job object, events and the per-step loss
                                      (uses NEBIUS_API_KEY / NEBIUS_BASE_URL from the environment;
                                      they are never printed). Default: every job in the run's
                                      `experiments` table.
  prompt   RUN_ID [--round N]         first training row vs the rebuilt eval prompt
                                      (byte-identical?) and both rendered
                                      through the checkpoint's chat template.
  overlap  RUN_ID [A B]               dev tasks failed in each of two dev-eval stages (stage
                                      names, default dev_eval_r1 dev_eval_r2, or artifact
                                      hashes) + overlap.

Common: --root DIR (default .distillery, holds index.sqlite and runs/).
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

from distillery import diagnostics as d
from distillery.store import Store
from distillery.taskpacks.sql import schema as sql_schema


def cmd_jobs(store: Store, run_id: str, job_ids: list[str]) -> dict[str, Any]:
    import openai

    from distillery.finetune import FineTuneClient

    if not job_ids:
        job_ids = [
            str(data["job_id"])
            for name, data in store.list_experiments(run_id)
            if name in ("finetune_job_started", "finetune_job_adopted")
        ]
    key, base = os.environ.get("NEBIUS_API_KEY"), os.environ.get("NEBIUS_BASE_URL")
    if not key or not base:
        raise SystemExit("NEBIUS_API_KEY and NEBIUS_BASE_URL must be set in the environment")
    client = FineTuneClient(openai.OpenAI(base_url=base, api_key=key, max_retries=0))
    out: dict[str, Any] = {}
    for jid in dict.fromkeys(job_ids):
        raw = client.raw_job(jid)
        info = client.get(jid)
        out[jid] = {
            "status": info.status,
            "resolved_hyperparameters": info.hyperparameters,
            "trained_tokens": info.trained_tokens,
            "trained_steps": info.trained_steps,
            "total_steps": info.total_steps,
            "loss_curve": client.loss_curve(jid),
            "events": [
                {"created_at": e.created_at, "level": e.level, "message": e.message}
                for e in client.events(jid)
            ],
            "raw_job": raw,
        }
    return out


def cmd_prompt(store: Store, run_id: str, rnd: int, db_seed: int) -> dict[str, Any]:
    rdir = store.run_dir(run_id) / f"round{rnd}"
    rows = d.read_jsonl(rdir / "train.jsonl")
    if not rows:
        raise SystemExit(f"{rdir / 'train.jsonl'} is empty")
    cmp = d.compare_prompts(rows[0], sql_schema.schema_ddl(db_seed))
    out: dict[str, Any] = {
        "byte_identical": cmp["byte_identical"],
        "question": cmp["question"],
        "train_prompt_messages": cmp["train_prompt"],
        "eval_prompt_messages": cmp["eval_prompt"],
        "train_assistant": cmp["train_assistant"],
    }
    templates = sorted((rdir / "checkpoints").glob("*/chat_template.jinja"))
    if not templates:
        out["rendered"] = {"error": f"no chat_template.jinja under {rdir / 'checkpoints'}"}
    else:
        out["chat_template"] = str(templates[0])
        out["rendered"] = d.render_all(templates[0].read_text(encoding="utf-8"), cmp)
        try:  # optional: token counts for a token-for-token comparison
            from tokenizers import Tokenizer  # type: ignore[import-not-found,unused-ignore]

            tok_path = templates[0].parent / "tokenizer.json"
            if tok_path.exists() and "error" not in out["rendered"]:
                tk = Tokenizer.from_file(str(tok_path))
                out["encoded"] = {
                    k: tk.encode(v, add_special_tokens=False).ids
                    for k, v in out["rendered"].items()
                }
        except ImportError:
            out["encoded"] = "tokenizers not installed"
    return out


def cmd_overlap(store: Store, run_id: str, a: str, b: str) -> dict[str, Any]:
    return {
        "a": a,
        "b": b,
        **d.failure_overlap(d.stage_artifact(store, run_id, a), d.stage_artifact(store, run_id, b)),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--root", default=".distillery")
    sub = ap.add_subparsers(dest="cmd", required=True)
    j = sub.add_parser("jobs")
    j.add_argument("run_id")
    j.add_argument("--job-id", action="append", default=[])
    p = sub.add_parser("prompt")
    p.add_argument("run_id")
    p.add_argument("--round", type=int, default=1)
    p.add_argument("--db-seed", type=int, default=0)
    o = sub.add_parser("overlap")
    o.add_argument("run_id")
    o.add_argument("a", nargs="?", default="dev_eval_r1")
    o.add_argument("b", nargs="?", default="dev_eval_r2")
    args = ap.parse_args(argv)
    store = Store(Path(args.root))
    try:
        if args.cmd == "jobs":
            res = cmd_jobs(store, args.run_id, args.job_id)
        elif args.cmd == "prompt":
            res = cmd_prompt(store, args.run_id, args.round, args.db_seed)
        else:
            res = cmd_overlap(store, args.run_id, args.a, args.b)
    finally:
        store.close()
    print(json.dumps(res, indent=2, ensure_ascii=False, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
