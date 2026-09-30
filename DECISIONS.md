# DECISIONS

## 2026-09-29: Repo location and license holder
Repo created at `~/Documents/distillery` (no location was specified). LICENSE names "Distillery contributors" as
copyright holder rather than a person, since no legal name was provided; change before submission if desired.

## 2026-09-29: Orchestrator, CLI, dry-run (decisions made while building them)
- **Stage order.** Split & seal runs before teacher data (the teacher only ever sees train tasks) and the headroom
  check ("task too easy" if base dev accuracy >= 0.80, configurable) runs right after the split because it needs the dev set.
- **Cross-check semantics.** "Identical results" is judged on execution results (planner, teacher and template gold must
  all match under `compare_outcomes`), not SQL text, since equivalent SQL rarely matches textually. Discard reasons are
  counted; if planner and teacher agree with each other but not with the template gold the task is discarded AND listed as
  `suspect_gold` (probable template bug). Cross-check runs on all generated questions, including future held-out ones
  (label quality), so "the planner never sees held-out" is enforced for failure analysis and every later planner call
  (tested), not for the gold cross-check.
- **Verifier authoring by Ultra: not built.** For the SQL pack the verifier is the deterministic execution match. The hook is
  `PipelineConfig.ultra_verifier_authoring`; setting it raises NotImplementedError (never silently ignored). The self-test
  (gold accepted, every `corrupt_sql` variant rejected) runs through the pipeline's real Executor, and also checks the executor
  agrees with the local reference, so it doubles as a sandbox-path test.
- **Rounds.** `max_rounds` = total fine-tune rounds including the first. Each round: dev eval -> (planner clusters dev failures
  into families to target, only train families offered) -> targeted generation filtered against held-out ids/families/gold
  hashes -> teacher + execution verification -> `Sandbox.branch` (lineage uuid/parent/label in the report) -> re-fine-tune from
  the base on the cumulative rows. Targeted tasks are not LLM cross-checked (template gold + teacher verification only).
  The held-out set is scored ONCE, for the round with the best DEV accuracy (ties: earliest), never per round: selecting on
  held-out would burn it.
- **Triage (Nano) is advisory.** It flags teacher outputs with dirty format (counted, sample capped) but never drops
  execution-verified rows.
- **Fine-tune cost is an explicit estimate.** The docs give no fine-tune price, so a live run must pass
  `--finetune-estimate-usd`; it is preflighted before upload/job creation and recorded as spend `finetune` (reported as an estimate).
  Jobs started but never closed (process killed) are cancelled on resume; every non-success exit cancels via `finetune.paid_job`.
- **Base model is called through the inference API** (role `student` in `LLMClient`) for base eval and headroom. UNVERIFIED that
  Qwen3-1.7B is served serverless. Student cost per 1k tasks is the literal string "unavailable: serving path undecided (spike S4)".
- **Live runs refuse to start without a student serving path** (`student_factory` is None until S4), before any spend.
- **Dry runs are physically separate**: run ids must start with `dry-` (and real ones must not), their own store root
  (`<root>/dry-runs`), `dry_run: true` + label stamped into `manifest.json`, `run_meta.json`, `report.json`, and printed at the
  start and end of console output. Fake price rows carry provenance "FAKE dry-run price".
- **Admin gate.** Expected token from `DISTILLERY_ADMIN_TOKEN`; supplied token from `DISTILLERY_ADMIN_TOKEN_SUPPLIED` or a hidden
  prompt, never argv. `--i-approve-spend` is required the first time a run id starts and is remembered per run.
- **Stage artifacts store checkpoint paths relative to the run dir** so hashes are reproducible across store locations.
- Budget preflight is per chunk of 50 LLM calls (estimate = prompt chars / 3 + configured output tokens, x price); actual usage
  is recorded from each response.

## 2026-09-30: Student model changed to Qwen/Qwen3-30B-A3B-Instruct-2507
Qwen3-1.7B is fine-tunable but NOT on the serverless inference API (live-verified), so the base model could not be scored
without building our own serving. `Qwen/Qwen3-30B-A3B-Instruct-2507` is on the serverless list AND listed as LoRA-fine-tunable
(post-training/models), so base and teacher/Nemotron all go through the same inference API; it is an Instruct model, so no
thinking-mode toggle is needed. Set via `DISTILLERY_MODEL_STUDENT` (no code change). Trade-offs, all UNVERIFIED: (1) it is a
30B MoE (3B active), not a "small" model, so the cost story is "3B-active vs 55B-active teacher", not tiny-vs-huge; (2) base
accuracy may already be high, so the headroom check (base <= 60% on dev) may fail and the task set would need to get harder;
(3) MoE LoRA fine-tuning price and time are unknown; (4) serving the fine-tuned adapter still needs a dedicated endpoint
(custom weights "on request"), so S4 is still open; only the BASE is now servable serverless.

## 2026-09-30 (later): Student reverted to Qwen/Qwen3-1.7B (fallback Qwen3-0.6B)
The 30B-A3B student above is withdrawn: the product claim is distilling into a genuinely SMALL model, and a 30B MoE undercuts it.
Consequence, accepted: Qwen3-1.7B is not on the serverless API, so BOTH the base and the fine-tuned student must be served by our
own path, chosen in spike S4 (sandbox CPU with peft, or a dedicated endpoint with custom weights, or a serverless GPU job). Base and
student must be scored on the same serving path. The 30B decision stays in the log as a considered-and-rejected option.

## 2026-09-30: All model weights live on Nebius, never on the dev machine
Base weights for the small student (Qwen/Qwen3-1.7B, public, Apache-2.0, ungated on Hugging Face) are NOT downloaded locally
(a partial local download was started and deleted). The whole agentic system, including serving, runs in Nebius cloud only.
So the weights get fetched INSIDE Nebius: either by the Sandbox image build (needs sandbox egress to huggingface.co, unverified,
spike S2/S4) or by a dedicated endpoint / serverless job pulling from Hugging Face (custom weights path, on request). If sandbox
egress is blocked, fallback is uploading weights to Nebius storage from a Nebius-hosted step, decided in S4.

## 2026-09-30: Student is Qwen/Qwen3-0.6B; the 1.7B benchmark was stopped by the user
S4 measured Qwen3-0.6B bf16 on a 4-CPU/~4 GB sandbox: peak RSS 2.09 GB, 7.8 tok/s, ~7.8 s/sample at ~893-token prompts. The Qwen3-1.7B
benchmark was started, then stopped on the user's instruction and its sandbox operation cancelled (op 01a0f149, status CANCELLED);
1.7B fit on this sandbox is therefore UNMEASURED (estimate: ~4 GB+, likely OOM-tight). Student for the live end-to-end run: Qwen3-0.6B.
