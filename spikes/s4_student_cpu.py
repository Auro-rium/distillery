# ruff: noqa: S101, E501, BLE001, S604, S108
"""S4 (live): can a small Qwen run on Sandbox CPU? Measures RAM, load time, seconds/sample.

Everything heavy runs INSIDE a Nebius sandbox (weights come from huggingface.co over the sandbox's egress;
nothing is downloaded to this machine). Base image -> deps image (branch) -> one image per model (branch).
Scores the base model's SQL locally against the gold results (a first look at headroom, small sample).

Usage: python spikes/s4_student_cpu.py --i-approve-spend [--models Qwen/Qwen3-0.6B,Qwen/Qwen3-1.7B] [--n 12]
Reads NEBIUS_API_KEY and NEBIUS_PROJECT_ID / NEBIUS_AI_PROJECT from the environment; never prints them.
"""

import argparse
import asyncio
import json
import os
import tempfile
import time

from contree_sdk import Contree
from contree_sdk.auth import IAMAuth
from contree_sdk.config import ContreeConfig

from distillery.prompts import build_messages, extract_sql
from distillery.taskpacks.sql import questions as q
from distillery.taskpacks.sql import schema as s
from distillery.taskpacks.sql.executor import LocalExecutor
from distillery.taskpacks.sql.verifier import compare_outcomes

BENCH = r"""
import json, os, resource, sys, time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

mdir, dtype, n, maxnew = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
torch.set_num_threads(os.cpu_count() or 4)
dt = {"bf16": torch.bfloat16, "fp32": torch.float32}[dtype]
t = time.time()
tok = AutoTokenizer.from_pretrained(mdir)
model = AutoModelForCausalLM.from_pretrained(mdir, torch_dtype=dt)
model.eval()
load_s = time.time() - t
rss_load_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
print("LOADED", json.dumps({"load_s": round(load_s, 1), "rss_mb": round(rss_load_mb)}), flush=True)
prompts = json.load(open("/prompts.json"))[:n]
outs = []
for p in prompts:
    text = tok.apply_chat_template(p["messages"], tokenize=False, add_generation_prompt=True, enable_thinking=False)
    ids = tok(text, return_tensors="pt")
    t = time.time()
    with torch.no_grad():
        g = model.generate(**ids, max_new_tokens=maxnew, do_sample=False)
    sec = time.time() - t
    new = g[0][ids["input_ids"].shape[1]:]
    outs.append({"id": p["id"], "output": tok.decode(new, skip_special_tokens=True),
                 "prompt_tokens": int(ids["input_ids"].shape[1]), "new_tokens": int(len(new)), "seconds": round(sec, 2)})
    print("SAMPLE", outs[-1]["id"], outs[-1]["prompt_tokens"], outs[-1]["new_tokens"], outs[-1]["seconds"], flush=True)
peak_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
print("RESULT_JSON=" + json.dumps({"dtype": dtype, "load_s": round(load_s, 1), "peak_rss_mb": round(peak_mb), "samples": outs}))
"""

DEPS = (
    "pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch "
    "&& pip install --no-cache-dir transformers accelerate peft safetensors huggingface_hub"
)


def _txt(v: object) -> str:
    return v.decode(errors="replace") if isinstance(v, bytes) else str(v or "")


def tail(v: object, n: int = 600) -> str:
    return _txt(v).strip()[-n:]


def prep_prompts(n: int) -> list[dict]:
    d = tempfile.mkdtemp()
    db = os.path.join(d, "db.sqlite")
    s.write_database(0, db)
    ddl = s.schema_ddl(0)
    rep = q.generate_tasks(db, 200, 0)
    by_family: dict[str, list] = {}
    for t in rep.tasks:
        by_family.setdefault(t.family, []).append(t)
    picked = []
    i = 0
    while len(picked) < n and any(by_family.values()):  # round-robin across families
        fam = sorted(by_family)[i % len(by_family)]
        if by_family[fam]:
            picked.append(by_family[fam].pop(0))
        i += 1
    prompts = [
        {
            "id": t.task_id,
            "family": t.family,
            "difficulty": t.difficulty,
            "requires_order": t.requires_order,
            "gold_sql": t.gold_sql,
            "question": t.question,
            "messages": build_messages(t.question, ddl, role="eval_base"),
        }
        for t in picked
    ]
    for p in prompts:
        p["_db"] = db
    return prompts


def score_locally(prompts: list[dict], samples: list[dict]) -> dict:
    by_id = {p["id"]: p for p in prompts}
    ex = LocalExecutor()
    ok = unparse = 0
    per_family: dict[str, list[bool]] = {}
    for smp in samples:
        p = by_id[smp["id"]]
        sql = extract_sql(smp["output"])
        if sql is None:
            unparse += 1
            per_family.setdefault(p["family"], []).append(False)
            continue
        cand, gold = ex.run_batch(p["_db"], [sql, p["gold_sql"]])
        good = compare_outcomes(cand, gold, p["requires_order"]).ok
        ok += good
        per_family.setdefault(p["family"], []).append(good)
    n = len(samples)
    return {
        "n": n,
        "correct": ok,
        "unparseable": unparse,
        "accuracy": round(ok / n, 3) if n else None,
        "by_family": {k: f"{sum(v)}/{len(v)}" for k, v in sorted(per_family.items())},
    }


async def timed(label: str, coro):  # type: ignore[no-untyped-def]
    t = time.monotonic()
    print(f"[{label}] start", flush=True)
    try:
        r = await coro
        print(
            f"[{label}] done in {time.monotonic() - t:.1f}s exit={getattr(r, 'exit_code', '?')}",
            flush=True,
        )
        return r
    except Exception as e:
        print(
            f"[{label}] ERROR after {time.monotonic() - t:.1f}s: {type(e).__name__}: {str(e)[:300]}",
            flush=True,
        )
        return None


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--i-approve-spend", action="store_true")
    ap.add_argument("--models", default="Qwen/Qwen3-0.6B,Qwen/Qwen3-1.7B")
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--max-new", type=int, default=160)
    a = ap.parse_args()
    if not a.i_approve_spend:
        raise SystemExit("refusing to use paid sandbox compute without --i-approve-spend")

    prompts = prep_prompts(a.n)
    wire = [{k: v for k, v in p.items() if k in ("id", "messages")} for p in prompts]
    proj = os.environ.get("NEBIUS_PROJECT_ID") or os.environ["NEBIUS_AI_PROJECT"]
    sdk = Contree(
        config=ContreeConfig(auth=IAMAuth(token=os.environ["NEBIUS_API_KEY"], project_id=proj))
    )
    files = {"bench.py": BENCH.encode(), "prompts.json": json.dumps(wire).encode()}
    summary: dict = {"n_prompts": len(prompts)}

    base = await sdk.images.oci("docker://python:3.12-slim")
    deps = await timed(
        "install deps (torch cpu, transformers, peft)",
        base.run(shell=DEPS, disposable=False, timeout=1500),
    )
    if deps is None or deps.exit_code != 0:
        print("deps failed:", tail(getattr(deps, "stderr", ""), 1500))
        return
    summary["deps_image"] = str(deps.uuid)

    for model in a.models.split(","):
        key = model.split("/")[-1]
        dl_cmd = (
            'HF_HUB_DISABLE_TELEMETRY=1 python -c "from huggingface_hub import snapshot_download as d; '
            f"print(d('{model}', local_dir='/models/{key}'))\" && du -sh /models/{key}"
        )
        dl = await timed(
            f"download {model} inside sandbox",
            deps.run(shell=dl_cmd, disposable=False, timeout=1500),
        )
        if dl is None or dl.exit_code != 0:
            summary[key] = {"error": "download failed", "stderr": tail(getattr(dl, "stderr", ""))}
            continue
        summary[key] = {"download_stdout": tail(dl.stdout, 200), "runs": {}}
        for dtype in ("bf16",):
            r = await timed(
                f"bench {key} {dtype} (n={a.n})",
                dl.run(
                    shell=f"python /bench.py /models/{key} {dtype} {a.n} {a.max_new}",
                    files=files,
                    timeout=1700,
                    truncate_output_at=400000,
                ),
            )
            entry: dict = {"exit": getattr(r, "exit_code", None)}
            out = _txt(getattr(r, "stdout", ""))
            entry["stderr_tail"] = tail(getattr(r, "stderr", ""), 500)
            marker = "RESULT_JSON="
            if marker in out:
                res = json.loads(out.split(marker, 1)[1].strip().splitlines()[0])
                secs = [x["seconds"] for x in res["samples"]]
                toks = sum(x["new_tokens"] for x in res["samples"])
                entry.update(
                    {
                        "load_s": res["load_s"],
                        "peak_rss_mb": res["peak_rss_mb"],
                        "avg_sample_s": round(sum(secs) / len(secs), 2),
                        "max_sample_s": max(secs),
                        "avg_prompt_tokens": round(
                            sum(x["prompt_tokens"] for x in res["samples"]) / len(secs)
                        ),
                        "gen_tokens_per_s": round(toks / sum(secs), 2),
                        "local_score_base_model": score_locally(prompts, res["samples"]),
                        "sample_outputs": [x["output"][:160] for x in res["samples"][:2]],
                    }
                )
            else:
                entry["stdout_tail"] = tail(out, 500)
            summary[key]["runs"][dtype] = entry
            print(json.dumps({key: entry}, indent=1, default=str), flush=True)

    print("FINAL_SUMMARY=" + json.dumps(summary, default=str))


asyncio.run(main())
