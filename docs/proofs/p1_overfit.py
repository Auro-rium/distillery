# ruff: noqa: E501, S311  (evidence script: seeded shuffle, long lines)
"""P1.11 overfit instrument: can the Qwen3-1.7B student memorise 64 training rows?

Fine-tunes Qwen/Qwen3-1.7B (LoRA) on 64 TRAIN rows of run gated-1p7b-r1 for 10 epochs at
batch size 2 (320 optimizer steps), then scores the base model and the adapter on those same 64
prompts in the sandbox. Completion = the task's template gold SQL (no LLM spend). Only
``split["train"]`` is read; no sealed set (held-out, stress, human) is ever touched. The pass bar
(adapter >= 90% on the 64 rows) is judged by the controller, not by this script.

Modes:
  plan  (default)  NO network, NO spend. Selects the rows, writes train.jsonl / valid.jsonl into
                   --out-dir and prints row count, per-row token p50/max, planned steps,
                   estimated trained tokens and estimated sandbox generations.
  run              SPENDS: one paid fine-tune job plus 2 x 64 sandbox generations. Refuses to start
                   without --i-approve-spend. Needs env NEBIUS_API_KEY, NEBIUS_BASE_URL and
                   NEBIUS_PROJECT_ID / NEBIUS_AI_PROJECT (never printed), and the student model
                   (DISTILLERY_MODEL_STUDENT) must be Qwen/Qwen3-1.7B. Crash safe: job.json is
                   written the moment the job exists, and an existing job.json is ADOPTED (polled)
                   instead of creating a second job. Writes docs/proofs/evidence/p1_11_overfit.json.

Usage:
  .venv/bin/python docs/proofs/p1_overfit.py plan --tokenizer DIR
  .venv/bin/python docs/proofs/p1_overfit.py run --i-approve-spend
"""

import argparse
import json
import os
import random
import statistics
import sys
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from distillery.diagnostics import render_chat_template, stage_artifact
from distillery.finetune import HyperParameters, paid_job, planned_steps
from distillery.prompts import to_training_row
from distillery.store import Store
from distillery.taskpacks.sql.schema import schema_ddl

RUN_ID = "gated-1p7b-r1"
MODEL = "Qwen/Qwen3-1.7B"
N_TRAIN, N_VALID, SEED = 64, 16, 1234
SUFFIX = "distillery-p1-11-overfit"
POLL_INTERVAL_S = 30.0
MIN_STEPS = 300
HP = HyperParameters(
    lora=True, lora_r=16, lora_alpha=32, lora_dropout=0.0, learning_rate=2e-4, batch_size=2,
    n_epochs=10, packing=False, warmup_ratio=0.0, weight_decay=0.0, max_grad_norm=1.0,
    context_length=8192,
)  # fmt: skip
ADAPTER_NAMES = ("adapter_config.json", "adapter_model.safetensors")
EVIDENCE = Path(__file__).resolve().parent / "evidence" / "p1_11_overfit.json"


def select_tasks(
    train: Sequence[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """64 train tasks round-robin over sorted families, then 16 more (disjoint) the same way.

    Within a family the tasks follow one ``random.Random(SEED)`` shuffle; a single round-robin draw
    of N_TRAIN + N_VALID tasks is split, so the two selections are disjoint and each is balanced.
    """
    rng = random.Random(SEED)
    queues: dict[str, list[dict[str, Any]]] = {}
    for fam in sorted({str(t["family"]) for t in train}):
        q = sorted((t for t in train if str(t["family"]) == fam), key=lambda t: str(t["task_id"]))
        rng.shuffle(q)
        queues[fam] = q
    drawn: list[dict[str, Any]] = []
    while len(drawn) < N_TRAIN + N_VALID and any(queues.values()):
        for fam in sorted(queues):
            if queues[fam] and len(drawn) < N_TRAIN + N_VALID:
                drawn.append(queues[fam].pop(0))
    if len(drawn) < N_TRAIN + N_VALID:
        raise SystemExit(f"only {len(drawn)} train tasks available, need {N_TRAIN + N_VALID}")
    return drawn[:N_TRAIN], drawn[N_TRAIN:]


def require_steps(steps: int | None) -> None:
    if steps is None or steps < MIN_STEPS:
        raise SystemExit(f"planned steps {steps} < {MIN_STEPS}")


def to_row(task: dict[str, Any], ddl: str) -> dict[str, Any]:
    return to_training_row(
        SimpleNamespace(question=task["question"]), str(task["gold_sql"]), schema_ddl=ddl
    )


def write_jsonl(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), "utf-8")


def pct(xs: Sequence[int]) -> dict[str, int]:
    s = sorted(xs)
    return {"p50": int(statistics.median(s)), "max": s[-1]}


def row_token_counts(
    rows: Sequence[dict[str, Any]], tokenizer_dir: Path | None
) -> tuple[list[int], str]:
    """Per-row token counts: rendered chat template + real tokenizer when available, else chars/3."""
    if tokenizer_dir is not None:
        from tokenizers import Tokenizer

        tok = Tokenizer.from_file(str(tokenizer_dir / "tokenizer.json"))
        cfg = json.loads((tokenizer_dir / "tokenizer_config.json").read_text("utf-8"))
        template = cfg["chat_template"]
        counts = [
            len(
                tok.encode(
                    render_chat_template(template, r["messages"], add_generation_prompt=False),
                    add_special_tokens=False,
                ).ids
            )
            for r in rows
        ]
        return counts, "tokenizer+chat_template"
    return [-(-sum(len(m["content"]) for m in r["messages"]) // 3) for r in rows], "chars/3"


def prepare(args: argparse.Namespace) -> tuple[list[dict[str, Any]], Path, Path]:
    """Select rows (train split only) and write train.jsonl / valid.jsonl."""
    store = Store(Path(args.root))
    try:
        train = stage_artifact(store, RUN_ID, "split")["train"]
    finally:
        store.close()
    train_sel, valid_sel = select_tasks(train)
    ddl = schema_ddl(0)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    train_path, valid_path = out / "train.jsonl", out / "valid.jsonl"
    write_jsonl(train_path, [to_row(t, ddl) for t in train_sel])
    write_jsonl(valid_path, [to_row(t, ddl) for t in valid_sel])
    return train_sel, train_path, valid_path


def plan(args: argparse.Namespace) -> int:
    train_sel, train_path, valid_path = prepare(args)
    rows = [json.loads(x) for x in train_path.read_text("utf-8").splitlines()]
    tokenizer_dir = Path(args.tokenizer) if args.tokenizer else None
    counts, method = row_token_counts(rows, tokenizer_dir)
    steps = planned_steps(len(rows), HP)
    require_steps(steps)
    fam_counts: dict[str, int] = {}
    for t in train_sel:
        fam_counts[str(t["family"])] = fam_counts.get(str(t["family"]), 0) + 1
    print(
        json.dumps(
            {
                "mode": "plan",
                "train_rows": len(rows),
                "valid_rows": len(valid_path.read_text("utf-8").splitlines()),
                "families": dict(sorted(fam_counts.items())),
                "token_method": method,
                "row_tokens": pct(counts),
                "planned_steps": steps,
                "estimated_trained_tokens": sum(counts) * int(HP.n_epochs or 0),
                "estimated_sandbox_generations": len(rows) * 2,
                "hyperparameters": HP.to_request(),
                "files": {"train": str(train_path), "valid": str(valid_path)},
            },
            indent=2,
        )
    )
    return 0


def run(args: argparse.Namespace) -> int:
    if not args.i_approve_spend:
        raise SystemExit("run SPENDS (paid fine-tune + sandbox compute); pass --i-approve-spend")
    # Network-capable modules are imported only here so plan mode can never reach the network.
    import openai

    from distillery.cli import SANDBOX_BASE_IMAGE
    from distillery.config import load_config
    from distillery.evaluator import diagnose_generate
    from distillery.finetune import FineTuneClient, JobInfo
    from distillery.sandbox import ContreeSandbox
    from distillery.sandbox_executor import AsyncBridge
    from distillery.sandbox_student import SandboxCpuStudent, ServingImages
    from distillery.taskpacks.sql.executor import LocalExecutor

    config = load_config()
    key, base_url = os.environ.get("NEBIUS_API_KEY"), os.environ.get("NEBIUS_BASE_URL")
    if not key or not base_url:
        raise SystemExit("NEBIUS_API_KEY and NEBIUS_BASE_URL must be set in the environment")
    if config.nebius_api_key is None or config.nebius_project_id is None:
        raise SystemExit("NEBIUS_API_KEY and NEBIUS_PROJECT_ID must be set in the environment")
    base_model = config.require_model("student")
    if base_model != MODEL:
        raise SystemExit(f"student model must be {MODEL}, configured: {base_model}")
    steps = planned_steps(N_TRAIN, HP)
    require_steps(steps)

    train_sel, train_path, valid_path = prepare(args)
    out = Path(args.out_dir)
    ft = FineTuneClient(openai.OpenAI(base_url=base_url, api_key=key, max_retries=0))
    job_json = out / "job.json"

    def create() -> str:
        if job_json.exists():  # crash recovery: adopt, never create a second billable job
            adopted = str(json.loads(job_json.read_text("utf-8"))["job_id"])
            print(f"adopting existing job {adopted} from {job_json}", flush=True)
            return adopted
        train_id, valid_id = ft.upload(train_path), ft.upload(valid_path)
        job_id = ft.create_job(
            MODEL, train_id, valid_id, hyperparameters=HP, suffix=SUFFIX, seed=SEED
        )
        job_json.write_text(
            json.dumps({"job_id": job_id, "created_at": datetime.now(UTC).isoformat()}), "utf-8"
        )  # IMMEDIATELY, before anything else can fail
        print(f"created job {job_id} (recorded in {job_json})", flush=True)
        return job_id

    def cancel(job_id: str) -> object:
        info = ft.get(job_id)  # an adopted/finished job must not be "cancelled" after the fact
        return info if info.terminal else ft.cancel(job_id)

    last: list[str] = []

    def on_update(info: JobInfo) -> None:
        if not last or last[-1] != info.status:
            last.append(info.status)
            print(
                f"{datetime.now(UTC).isoformat(timespec='seconds')} status={info.status} "
                f"steps={info.trained_steps}/{info.total_steps}",
                flush=True,
            )

    with paid_job(create, cancel) as handle:
        job_id = handle.job_id
        info = ft.require_success(ft.poll(job_id, interval_s=POLL_INTERVAL_S, on_update=on_update))
        cks = [c for c in ft.checkpoints(job_id) if c.step_number is not None and c.result_files]
        if not cks:
            raise SystemExit(f"job {job_id} has no downloadable checkpoint")
        ckpt = max(cks, key=lambda c: c.step_number or 0)
        artifact = ft.trained_artifact(info, ckpt, out)
        handle.mark_succeeded()
    files = {f.path.name: f.path for f in artifact.files}
    missing = [n for n in ADAPTER_NAMES if n not in files]
    if missing:
        raise SystemExit(f"checkpoint {ckpt.id} lacks {missing}; got {sorted(files)}")
    adapter_files = [files[n] for n in ADAPTER_NAMES]
    raw = ft.raw_job(job_id)
    loss_curve = ft.loss_curve(job_id)
    print(f"checkpoint {ckpt.id} step {ckpt.step_number} downloaded; scoring", flush=True)

    items = [
        {
            "task_id": t["task_id"],
            "gold_sql": t["gold_sql"],
            "requires_order": bool(t["requires_order"]),
            "question": t["question"],
        }
        for t in train_sel
    ]
    db_path = Path(args.root) / "runs" / RUN_ID / "db.sqlite"
    servers: dict[str, Any] = {}
    with AsyncBridge() as bridge:
        sandbox = ContreeSandbox(
            lambda: config.nebius_api_key.get_secret_value() if config.nebius_api_key else "",
            os.environ.get("DISTILLERY_SANDBOX_URL") or None,
            project_id=config.nebius_project_id.get_secret_value(),
        )
        image = bridge.run(
            sandbox.ensure_image(os.environ.get("DISTILLERY_SANDBOX_IMAGE") or SANDBOX_BASE_IMAGE)
        )
        images = ServingImages(sandbox, image, bridge, base_model=base_model)

        def make_server(files_: Sequence[Path]) -> Any:
            return SandboxCpuStudent(
                sandbox, image, bridge, base_model=base_model, adapter_files=files_,
                batch_size=2, concurrency=10, max_new_tokens=160, images=images,
            )  # fmt: skip

        try:
            servers = {"base": make_server(()), "adapter": make_server(adapter_files)}
            result = diagnose_generate(
                servers, {"train64": items}, LocalExecutor(),
                db_ref=str(db_path), schema_ddl=schema_ddl(0),
            )  # fmt: skip
        finally:
            for s in servers.values():
                s.close()

    def timing_sum(server: Any, k: str) -> float:
        return sum(float(t.get(k) or 0) for t in getattr(server, "timings", []))

    accuracy = {m: result["models"][m]["train64"]["accuracy"] for m in result["models"]}
    evidence = {
        "run_id": RUN_ID,
        "base_model": base_model,
        "hyperparameters": HP.to_request(),
        "seed": SEED,
        "planned_steps": steps,
        "job_id": job_id,
        "checkpoint": {"id": ckpt.id, "step_number": ckpt.step_number},
        "adapter_sha256": artifact.adapter_sha256,
        "job": {
            "status": raw.get("status"),
            "resolved_hyperparameters": raw.get("hyperparameters"),
            "trained_tokens": raw.get("trained_tokens"),
            "trained_steps": raw.get("trained_steps"),
        },
        "loss_curve": loss_curve,
        "accuracy": accuracy,
        "identical_to_base": result.get("identical_to_base"),
        "cap_hits": {m: getattr(s, "cap_hits", None) for m, s in servers.items()},
        "sandbox_timings_sum": {
            m: {k: timing_sum(s, k) for k in ("load_s", "gen_s")} for m, s in servers.items()
        },
        "models": result["models"],
    }
    EVIDENCE.parent.mkdir(parents=True, exist_ok=True)
    EVIDENCE.write_text(json.dumps(evidence, indent=1, ensure_ascii=False, default=str), "utf-8")
    print(json.dumps({"evidence": str(EVIDENCE), "accuracy": accuracy, "job_id": job_id}, indent=2))
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("mode", nargs="?", choices=("plan", "run"), default="plan")
    ap.add_argument("--root", default=".distillery/live")
    ap.add_argument("--out-dir", default=".distillery/proofs/p1_11")
    ap.add_argument(
        "--tokenizer", default=None, help="dir with tokenizer.json + tokenizer_config.json"
    )
    ap.add_argument("--i-approve-spend", action="store_true")
    args = ap.parse_args(argv)
    return run(args) if args.mode == "run" else plan(args)


if __name__ == "__main__":
    sys.exit(main())
