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
import time
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
    context_length=16384,  # provider: batch_size x context_length >= 32768 (422 otherwise)
)  # fmt: skip
ADAPTER_NAMES = ("adapter_config.json", "adapter_model.safetensors")
EVIDENCE_DIR = Path(__file__).resolve().parent / "evidence"
EVIDENCE = EVIDENCE_DIR / "p1_11_overfit.json"
EVAL, SET_NAME, BASE_CACHE_DIR = "train", "train64", Path(".distillery/proofs/p1_11")


def _pilot_hp(lr: float) -> HyperParameters:  # P1.12 arms, DECISIONS.md 2026-10-04 (fixed)
    return HyperParameters(
        lora=True, lora_r=16, lora_alpha=32, lora_dropout=0.0, learning_rate=lr, batch_size=4,
        n_epochs=4, packing=False, warmup_ratio=0.0, weight_decay=0.0, max_grad_norm=1.0,
        context_length=8192,
    )  # fmt: skip


# name -> (n_train, hyperparameters, eval split, set name, evidence file, out dir, base cache dir)
PRESETS: dict[str, tuple[int, HyperParameters, str, str, str, str, str]] = {
    "p1_11": (
        64,
        HP,
        "train",
        "train64",
        "p1_11_overfit.json",
        ".distillery/proofs/p1_11",
        ".distillery/proofs/p1_11",
    ),
    "p1_12a": (
        300,
        _pilot_hp(1e-4),
        "dev",
        "dev150",
        "p1_12_arm_a.json",
        ".distillery/proofs/p1_12/a",
        ".distillery/proofs/p1_12",
    ),
    "p1_12b": (
        300,
        _pilot_hp(2e-4),
        "dev",
        "dev150",
        "p1_12_arm_b.json",
        ".distillery/proofs/p1_12/b",
        ".distillery/proofs/p1_12",
    ),
}


def apply_preset(name: str) -> str:
    """Set the module constants for one pre-registered configuration; returns its out dir."""
    global N_TRAIN, HP, EVAL, SET_NAME, EVIDENCE, BASE_CACHE_DIR, SUFFIX
    n, hp, ev, set_name, evidence, out_dir, base_cache = PRESETS[name]
    N_TRAIN, HP, EVAL, SET_NAME = n, hp, ev, set_name
    EVIDENCE, BASE_CACHE_DIR = EVIDENCE_DIR / evidence, Path(base_cache)
    SUFFIX = f"distillery-{name.replace('_', '-')}"
    return out_dir


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


def eval_tasks(args: argparse.Namespace, train_sel: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """P1.11 scores the trained rows themselves; P1.12 scores split["dev"] (never trained on)."""
    if EVAL == "train":
        return train_sel
    store = Store(Path(args.root))
    try:
        return list(stage_artifact(store, RUN_ID, "split")["dev"])
    finally:
        store.close()


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
                "eval_set": SET_NAME,
                "estimated_sandbox_generations": len(eval_tasks(args, train_sel)) * 2,
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
    from distillery.evaluator import diagnose_generate, identical_rate
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
        for t in eval_tasks(args, train_sel)
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

        # One model at a time, adapter first (it decides the verdict); each model's scores are
        # cached in out-dir as soon as they exist, so a crash never discards finished work.
        models: dict[str, Any] = {}
        for name, files_ in (("adapter", adapter_files), ("base", ())):
            cache = (BASE_CACHE_DIR if name == "base" else out) / f"score_{name}.json"
            lock = cache.with_suffix(".lock")
            while name == "base" and lock.exists() and not cache.exists():  # other arm scoring it
                time.sleep(30)
            if cache.exists():
                models[name] = json.loads(cache.read_text())
                print(f"{name}: reusing cached scores from {cache}", flush=True)
                continue
            if name == "base":
                cache.parent.mkdir(parents=True, exist_ok=True)
                lock.write_text(str(os.getpid()))
            servers[name] = make_server(files_)
            try:
                models[name] = diagnose_generate(
                    {name: servers[name]}, {SET_NAME: items}, LocalExecutor(),
                    db_ref=str(db_path), schema_ddl=schema_ddl(0),
                )["models"][name]  # fmt: skip
            finally:
                servers[name].close()
            cache.write_text(json.dumps(models[name]))
            lock.unlink(missing_ok=True)
            print(f"{name}: accuracy {models[name][SET_NAME]['accuracy']:.3f}", flush=True)
        base_items, ad_items = (
            models["base"][SET_NAME]["items"],
            models["adapter"][SET_NAME]["items"],
        )
        result = {
            "models": models,
            "identical_to_base": {
                "adapter": {
                    f: identical_rate([r[f] for r in base_items], [r[f] for r in ad_items])
                    for f in ("raw", "sql")
                }
            },
        }

    def timing_sum(server: Any, k: str) -> float:
        return sum(float(t.get(k) or 0) for t in getattr(server, "timings", []))

    accuracy = {m: result["models"][m][SET_NAME]["accuracy"] for m in result["models"]}
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


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    p, d = k / n, 1 + z * z / n
    c, h = (p + z * z / (2 * n)) / d, z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / d
    return c - h, c + h


def decide() -> int:
    """P1.12 decision rule (DECISIONS.md 2026-10-04): qualify = Wilson CI entirely above base's;
    winner = higher dev accuracy among qualifiers, tie -> lower learning rate; none -> stop."""
    arms = {a: json.loads((EVIDENCE_DIR / f"p1_12_arm_{a}.json").read_text()) for a in ("a", "b")}

    def k(m: dict[str, Any]) -> int:
        return sum(bool(i["correct"]) for i in m["dev150"]["items"])

    base = arms["a"]["models"]["base"]
    n = len(base["dev150"]["items"])
    b_ci = wilson(k(base), n)
    rows = {}
    for a, ev in arms.items():
        ad = ev["models"]["adapter"]
        ci = wilson(k(ad), n)
        rows[a] = {"lr": ev["hyperparameters"]["learning_rate"], "correct": k(ad), "n": n,
                   "acc": k(ad) / n, "wilson95": ci, "qualifies": ci[0] > b_ci[1]}  # fmt: skip
    q = [a for a in rows if rows[a]["qualifies"]]
    winner = min(q, key=lambda a: (-rows[a]["correct"], rows[a]["lr"])) if q else None
    verdict = (
        f"PASS: arm {winner} (lr {rows[winner]['lr']})"
        if winner
        else "FAIL: no arm beats base (stop)"
    )
    out = {"base": {"correct": k(base), "n": n, "acc": k(base) / n, "wilson95": b_ci},
           "arms": rows, "winner": winner, "verdict": verdict}  # fmt: skip
    path = EVIDENCE_DIR / "p1_12_decision.json"
    path.write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("mode", nargs="?", choices=("plan", "run", "decide"), default="plan")
    ap.add_argument("--preset", choices=sorted(PRESETS), default="p1_11")
    ap.add_argument("--root", default=".distillery/live")
    ap.add_argument("--out-dir", default=None, help="default: the preset's out dir")
    ap.add_argument(
        "--tokenizer", default=None, help="dir with tokenizer.json + tokenizer_config.json"
    )
    ap.add_argument("--i-approve-spend", action="store_true")
    args = ap.parse_args(argv)
    if args.mode == "decide":
        return decide()
    out_dir = apply_preset(args.preset)
    args.out_dir = args.out_dir or out_dir
    return run(args) if args.mode == "run" else plan(args)


if __name__ == "__main__":
    sys.exit(main())
