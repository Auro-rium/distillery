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
