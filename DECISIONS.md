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
- **Verifier authoring by Ultra: not built.** For the SQL pack the verifier is the deterministic execution match. (The former
  `PipelineConfig.ultra_verifier_authoring` hook, which only raised NotImplementedError, was removed as dead code.) The self-test
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

## 2026-09-30: PRE-REGISTRATION for the next (single, gated) run. Written BEFORE any new data is generated or any result seen.
Everything below is fixed now. Results are reported whatever they are; nothing here is changed after seeing them.

**Why a new run.** Two live runs (tiny n=20, mini n=60) ended REJECT with student 15%/5% vs teacher 95%/100%. Read-only inspection of the
three real fine-tune jobs found the training was almost a no-op: the pipeline sent only `lora` and `n_epochs`, so the provider defaults
applied (batch_size 8, learning_rate 1e-5, lora_r 8, lora_alpha 8, packing TRUE, context 8192) and 40/117/138 rows trained for only
3/6/9 optimizer steps (trained_tokens 115,611 / 336,855 / 396,885). Loss fell slightly (mini r1 train 0.674 to 0.609). The same 3 of
30 dev tasks were solved after round 1 and after round 2. So those runs do not show that a 0.6B student cannot learn this task; they
show it was barely trained. Cause is a pipeline defect (hyperparameters implicit), fixed by making them explicit and guarded.

**Diagnostic thresholds (fixed before the diagnostics ran).** (b) student accuracy on its OWN training rows below 50% is "low";
(c) base-vs-student identical raw-output rate above 90% on dev+held-out is "high". Either one, or a train/eval prompt-template mismatch
found in (e), means: find and fix the pipeline defect (with a regression test) before any further run.

**Benchmark.**
- GATE set: sealed, in-distribution held-out, n >= 300. Every family in it also appears in training. Its (question, gold SQL) pairs never
  appear in train or dev. Same-skeleton overlap (question with literals masked) is ALLOWED but its rate is measured and reported.
- STRESS set: n ~ 100 from families reserved for it and excluded from train, dev and the gate set. Sealed separately. Reported separately
  (base/student/teacher). NEVER an input to the gate decision. Reserved families are chosen by the fixed rule
  `random.Random(777).sample(sorted(all_families), k=2)` (constants STRESS_SEED=777, STRESS_FAMILY_COUNT=2), not by past failures.
- DEV: 150 tasks (drives round selection only). TRAIN: 1,500 to 2,000 verified rows (target 1,800).
- GATE RULE, UNCHANGED from the original spec and never to be changed after results: PROMOTE iff the paired-bootstrap lower bound of
  (student accuracy / teacher accuracy) >= 0.85 AND exact McNemar p < 0.05 against the base model; 10,000 resamples, bootstrap seed 1234.
- Gold labels come from parameterized templates executed in code (no LLM produces gold). Questions use the hand-written phrasing variants
  in the templates (no LLM paraphrase). Teacher SQL (Nemotron Super) is kept as training data only if its execution result equals the
  template gold. The Ultra/Super agreement filter is removed and Ultra is not used in any bulk loop. Consequence accepted in advance:
  teacher accuracy on the gate set will be below the previous 95%-100%, because tasks the big models could not solve are no longer filtered.
- Templates: the template list is frozen after the offline audit (`scripts/audit_templates.py`); ORDER BY ties and under-capacity templates
  are reported by the audit and any template removed is listed here before generation.

**Student and training (no sweep).** Base model Qwen/Qwen3-0.6B, LoRA, served on Nebius Sandbox CPU for both base and student, evaluated in parallel sandboxes.
lora_r 16, learning_rate 1e-4, n_epochs 3 (as instructed), plus the values that were previously implicit and are now pinned:
batch_size 16, packing false, lora_alpha 16, lora_dropout 0.0, warmup_ratio 0.0, weight_decay 0.0, max_grad_norm 1.0, context_length 8192.
Planned steps = ceil(rows / 16) x 3, about 170 to 375 (the pipeline refuses a live fine-tune below 50 planned steps). max_rounds = 1 for the
gated run (the sandbox-branch tree already exists from `sql-mini-live1`). Fixed seeds: db_seed 0, task seed 1234, bootstrap seed 1234.

**Cost.** No run starts before a pre-flight estimate at REAL console prices is approved by the user. Until console prices are pasted, all
money figures are labelled ceiling estimates and are not quoted as costs.

**What a REJECT would mean.** If the correctly trained student still fails the gate, that is the result: report it, with the stress-set numbers,
and do not alter the gate, the sets, the templates or the hyperparameters to chase a PROMOTE.

## 2026-10-01: Amendments recorded BEFORE any new data is generated (diagnostics results, audit, and two implementation details)
**Diagnostics (b)(c)(e) ran on the real mini-run adapters (sandbox, approved).** (b) FAILED the pre-registered threshold: the student scored
16.2% (round 1 adapter, 117 rows) and 17.4% (round 2 adapter, 138 rows) on its OWN training rows (base: 8.5% / 8.0%), under 50%. (c) did NOT trip:
extracted-SQL identical rate student-vs-base was 0.9% to 3.3% (adapter is applied, not a no-op). (e) did not trip: training and eval prompts are
byte-identical, `/no_think` is in both, the chat template renders `<think>\n\n</think>\n\n` before the answer exactly as the eval prompt ends with
thinking off. So the defect found is the one already identified (implicit hyperparameters, 3-9 optimizer steps at lr 1e-5); it is fixed
(explicit pinned hyperparameters, packing off, 50-step floor, regression tests). That this fully explains the weak student is NOT yet proven; the
properly configured run is the test.

**Template audit (`scripts/audit_templates.py`, offline, 89 templates, 300 draws each, seed 4242).** It found one real defect: template
`accounts_above_industry_avg_users` had a second phrasing that omitted the country filter its gold applies (a hidden filter, so no model could
answer it). Fixed by naming the country in the wording. After the fix: 0 templates with gold errors, 0 ORDER BY ties (checked by re-running every
gold on a reversed-row-order copy of the database), 0 LIMIT-without-ORDER, 0 questions whose gold variants return different results. The
template list is FROZEN at 89 names, sha256 of the sorted names `d06fa8221b54d6ca482cf4537ff87ca229638b800540093255f56c61e38d7320`. The audit
is a CI test, so a later template cannot reintroduce a hidden filter.

**Measured at the pre-registered scale (offline dry run, structure only, fake models).** 1,800 train / 150 dev / 300 gate / 100 stress tasks
are all reachable. The stress rule resolves to `date_math` and `set_ops`. The 1,800 training tasks contain only 141 distinct question skeletons and
97.3% of gate questions share a skeleton with a training question. This was accepted in advance (hand-written phrasings, skeleton overlap
reported); it means the gate measures generalization across literals within known question shapes, not novel query structure. It must be stated
wherever gate results are shown. The stress set is the only unseen-structure signal.

**Implementation details fixed now.** (1) `oversample` 1.6 -> 1.1: it existed to absorb the removed agreement filter and 1.6 is not reachable
(the 10 training families supply ~2.8k tasks). (2) The stress draw uses template_share 0.1 and family_share 1.0 because with only two families
the default caps starve the pool. (3) One anchor training task per family is placed in train first, so every gate family is covered by training by
construction. (4) Teacher yield: a train row exists only if the teacher's SQL matches template gold, so verified rows = 1,800 x teacher accuracy;
if that falls below 1,500 the pre-registered range is missed and this is reported, not worked around.

## 2026-10-03: PRE-REGISTRATION for the single paid run, version 2 (student Qwen3-1.7B, human gate). Written BEFORE any paid run, any human-gold drafting, and any result.
This entry supersedes the 2026-09-30 pre-registration where they differ; everything not mentioned here stays as pre-registered then.

**Why this run exists.** The template-only gate cannot support the product claim (97.3% of its questions share a skeleton with training). This run adds a human-written held-out set (Gate B) and paraphrase-augmented training. Nothing here is changed after seeing results; a REJECT is reported as the result.

**Student: Qwen/Qwen3-1.7B (user decision, 2026-10-03), LoRA, served on Nebius Sandbox CPU for base and student.** Basis: live spike (STATUS.md, `docs/spikes/s5-qwen3-1_7b.json`): the base model fits a 4 vCPU / 4 GB sandbox with peak RSS about 3.8 GB (93%, almost no headroom), 10.6 to 28.8 s per sample depending on concurrency, best throughput 0.22 samples/s at concurrency 20. Accepted risk: a LoRA adapter adds memory, so the student may OOM where the base did not; an untrained-adapter memory pre-flight runs before the paid fine-tune (and an OOM there is reported and stops the run). Qwen3-0.6B is not used for this run.

**Gates (same thresholds for both, unchanged from the original spec).**
- Gate A (pipeline validation): the existing template gate, sealed in-distribution set n = 300.
- Gate B (the product claim): the human-written set, confirmed item by item by the user; same rule: PROMOTE iff the paired-bootstrap lower bound of student/teacher accuracy >= 0.85 AND exact McNemar p < 0.05 against base; 10,000 resamples, seed 1234.
- The stress set (n about 100, `date_math` and `set_ops`) is reported separately and is never a gate input. The student is not required to beat the teacher: the gate asks for at least 85% of teacher accuracy (lower bound) and a significant win over base. Beating the teacher outright is not a gate condition and would be reported as such if it happened.
- Gate B size: N_human is the number of confirmed items in the user's file. It is recorded here (count and sha256 of `human_confirmed.json`) by amendment BEFORE the paid run starts; if N_human is small the 0.85 lower bound is hard to reach, and that is accepted in advance, not adjusted for.

**Human set procedure.** The user supplies the questions (no LLM writes or edits them). Gold is drafted by the teacher at temperature 0.8, k = 3 samples, kept only if all three execute OK to the same non-empty result (multiset comparison; ordered only if all three have a top-level ORDER BY); discards are counted by reason. The user then marks each draft correct or incorrect; only confirmed items are sealed (own sha256, evaluator-only access). Disclosed bias: only questions the teacher can solve survive drafting, so teacher accuracy on Gate B is high by construction and hard questions are under-represented; discard and rejection counts are reported.

**Training data.** Scale `gated`: 650 template train tasks, dev 150, gate 300, stress 100, oversample 1.1. Teacher rows as before (kept only if execution equals template gold). Paraphrase augmentation: Nano writes up to 3 paraphrases per verified train question (it sees train question text only); a paraphrase is kept only if the teacher's SQL for it executes to the original template gold. Verified originals are always kept; paraphrases are sampled with seed 1234 to fill the cap `train_row_cap = 2000`. Target total 1,500 to 2,000 rows; if verified rows fall below 1,500 that is reported, not worked around. The paraphraser never sees any sealed-set content (tests enforce it).

**Training configuration (no sweep).** LoRA r 16, alpha 16, learning rate 1e-4, 3 epochs, batch 16, packing off, warmup 0.0, weight decay 0.0, max grad norm 1.0, lora dropout 0.0, context 8192. Planned steps = ceil(rows/16) x 3, about 280 to 375; the pipeline refuses below 50. `max_rounds = 1`. Seeds: db 0, task 1234, paraphrase 1234, bootstrap 1234. The run id is fresh (the human set changes the cache key of `split` and everything after it).

**Money.** No paid call before the pre-flight estimate is shown and approved by the user. The fine-tune and sandbox prices are not available from the API; they are derived from console balance deltas and recorded in config with source and date, and the report cost block is recomputed from them (`distillery reprice`). Until then the fine-tune line is an explicitly assumed ceiling.

**What would change the plan.** Only a failure of the pipeline itself (for example the adapter OOM pre-flight, or a refused preflight). Model quality is not a reason to change thresholds, sets, templates or hyperparameters.

## 2026-10-03 (amendment, recorded BEFORE the paid run starts): Gate B is scored after training, because no question file exists yet
No human question file had been supplied when the user approved the run and asked for a result without waiting. Adding `--human-set` to `run` would re-key `split` and every stage after it, so it cannot be added to a finished run without repeating paid work. Therefore:
- **Run 1 produces Gate A and the stress set only.** Its `decision` is the Gate A decision (pipeline validation) and is labelled as such; it is NOT the product-claim verdict.
- **Gate B is added afterwards by `distillery score-human <run> --human-set FILE`**: the user's confirmed set is sealed AFTER training and after the Gate A decision, and scored once against the same trained adapter, with the same thresholds and bootstrap seed. Nothing in training, selection, or tuning can depend on it; the report says it was sealed after the fact. A Gate B decision is only claimed once that command has run on the user's confirmed file.
- Everything else in the 2026-10-03 pre-registration stands: student Qwen3-1.7B, scale `gated`, paraphrase rule and cap, hyperparameters, max_rounds 1, thresholds, "a REJECT is the result".
- **Cap and ceiling actually used:** the user approved the estimate on 2026-10-03; the per-run cap is raised from the .env value ($5) to $9.00 for this run via `--budget-usd`, and the fine-tune cost ceiling is $6.00 (`--finetune-estimate-usd 6`, an assumed ceiling, not a price). The project cap stays $19.50.

## 2026-10-03 (amendment): planned-steps floor 50 -> 300 (proof P1.7)
The smoke runs trained 3, 6 and 9 optimizer steps at provider defaults, far below what the plan intends (steps >= 300). `PipelineConfig.min_planned_steps` now defaults to 300, so any non-dry run planning fewer than 300 optimizer steps is refused before upload (dry runs are exempt).
- **Consequence:** at the pinned batch 16 x 3 epochs a run needs >= 1,600 rows. With fewer, rows, epochs or batch size must change deliberately, decided by proof P1.12 and pre-registered before the run; a short run is refused, not silently undertrained.

## 2026-10-04 (pre-registration, recorded BEFORE any pilot job or pilot data exists): P1.12 hyperparameter pilot
**Why.** P1.11 passed (Qwen3-1.7B, 64 rows, lr 2e-4, r16/α32, batch 2, 320 steps: adapter 64/64 vs base 17/64), so training works and the smoke failure was undertraining. The pinned configuration (lr 1e-4, r16/α16, batch 16, 3 epochs) gives fewer than 300 steps unless a run has >= 1,600 rows, and it has never been run. P1.12 chooses the learning rate on dev data, once, before P4. This amends the "no sweep" line of the training configuration; nothing else (gate, thresholds, sets, templates, seeds) changes.

**Data.** Source: run `gated-1p7b-r1`, `split["train"]` only (650 tasks). 300 tasks chosen round-robin over sorted families, tasks within a family in a `random.Random(1234)` shuffle (the P1.11 procedure). Completion = the task's template `gold_sql` (no teacher spend; P4 trains on teacher SQL as registered, and that difference is accepted). Prompt = `to_training_row`/`build_messages` (eval-identical formatting, proven P1.2). Selection: `split["dev"]` (150 tasks, never trained on). Sealed sets (held-out, stress, human) are not touched.

**Arms (2, fixed; the third grid point 5e-4 is dropped in advance because P1.11 fit at 2e-4 and valid loss was already noisy there).** Common: LoRA r 16, alpha 32, dropout 0.0, batch 4, n_epochs 4, packing false, warmup 0.0, weight decay 0.0, max grad norm 1.0, context 8192 (batch x context = 32,768, the provider minimum). Planned steps = 300/4 x 4 = **300**. Arm A: learning rate **1e-4**. Arm B: learning rate **2e-4**. Each job is scored at its final (max step_number) checkpoint; no early stopping, no checkpoint picking.

**Scoring.** Base and both arms generate on the 150 dev prompts through the production sandbox path (merged serving, greedy, 160 new tokens) and are scored by `execution_match` (proven P3.1). Metric: dev accuracy, with a 95% Wilson interval per model.

**Decision rule (fixed now).**
1. An arm qualifies only if its Wilson interval lies entirely above base's (non-overlapping).
2. The winner is the qualifying arm with the higher dev accuracy; on a tie (equal counts), the lower learning rate.
3. If no arm qualifies, stop and report (grounding prompt), with no third arm and no rerun.

**Carried into P4 (fixed now).** P4 uses the winner's learning rate with r 16, alpha 32, batch 4, packing false, context 8192 and the registered 3 epochs; at the P4 data size (>= 1,600 teacher rows) that is >= 1,200 planned steps. P4's own pre-flight estimate is approved separately.

**Cost (estimate, approval required).** About 300 x 966 tokens x 4 epochs ≈ 1.16M trained tokens per arm, 2.32M for both (P1.11 was 0.615M). Sandbox: 150 dev prompts x 3 models at batch 2 = 225 generation jobs (P1.11: 32 jobs used load 288 s + generation 1,305 s). Dollar figures need the console prices from P1.11.

## 2026-10-04 (result + carry-over, recorded BEFORE P4 starts): P1.12 picked learning rate 1e-4
P1.12 ran exactly as pre-registered (`docs/proofs/evidence/p1_12_decision.json`): dev accuracy base 33/150 (Wilson [0.161, 0.293]), arm A lr 1e-4 **142/150** ([0.898, 0.973]), arm B lr 2e-4 141/150 ([0.890, 0.968]). Both qualify; A wins (higher count). Per the carry-over rule, `PINNED_HYPERPARAMETERS` is now LoRA r 16, alpha 32, learning rate 1e-4, batch 4, 3 epochs, packing false, context 8192, warmup 0, weight decay 0, max grad norm 1.0, dropout 0. With the 300-step floor, a run needs >= 400 training rows (was 1,600 at batch 16). Caveats recorded now: dev shares templates with train (in-distribution), and the pilot trained on gold SQL while P4 trains on teacher SQL. Gate, thresholds, sets, seeds and max_rounds 1 are unchanged.

## 2026-10-06 (pre-registration, recorded BEFORE any controller run): A3 Ultra controller bounds
The planner (Ultra) may propose a round action, but only code decides. Bounds fixed now, before any controller run:
- **Hyperparameters:** learning rate in {1e-4, 2e-4}; n_epochs in {2, 3, 4}; lora_r in {16, 32}. `min_planned_steps` is still enforced at fine-tune time (an adjust that would plan fewer steps is refused as before).
- **Budget:** the action's preflight estimate must fit the remaining cap; otherwise it is rejected.
- **Rounds:** the current round r must satisfy r < max_rounds.
- **Family:** must be in train \ stress \ held-out families; anything else is rejected.
- **Rejection:** a rejected action falls back to the existing rule-based decision; the reason is logged.
- **Default off:** `PipelineConfig.controller` defaults to False; with it off, behaviour and report are unchanged. The gate stays pure code and never sees controller output.

## 2026-10-06 (pre-registration, recorded BEFORE any A5 chaos run, local or hosted): A5 chaos pass criteria
A5: pre-register chaos pass criteria. A `chaos-` run is judged PASS only if ALL of the following hold; a failure is reported as the result, and the criteria are not changed afterwards.
- a verdict is reached (PROMOTE or REJECT both count);
- every injected fault is visible in the audit log and was recovered;
- the audit log shows exactly one `admin` action (the start) and zero others;
- exactly one fine-tune job was billed;
- spend <= cap.

Fault plan (from the A5 plan): (a) the LLM transport wrapper raises 503 for N calls; (b) a connection-error window of 60 s (network drop); (c) the supervisor SIGKILLs the child once in `teacher_data` and once in `finetune_r1` (tests adoption); (d) the sandbox returns exit -1 for k jobs in `dev_eval`. `DISTILLERY_CHAOS=<json plan>` is honoured only for run ids that start with `chaos-`. Each injection is logged as audit actor `chaos`.

## 2026-10-06 (A4): hosting stays on free Render; the autonomy claim is bounded by it
User decision: stay on free Render. Changes: `autoDeploy: false` (a push can no longer restart the instance under a running pipeline; the live service `distillery`, `srv-dauuokk1nsns73fqcv70`, differs in name from the blueprint's `distillery-api`, so its dashboard Auto-Deploy must also be switched Off by hand), a GitHub Actions keep-alive hitting `/api/health` every 10 minutes, the supervisor in-process with the server, and the audit log (`GET /api/runs/{id}/audit`).
**Stated limit:** the free instance's disk is wiped when the **instance** restarts. Autonomy is therefore proven for worker crashes, provider and network faults and sandbox failures inside one instance lifetime. A host restart is out of scope unless a persistent disk is added later. Memory headroom on the 512 MB instance: `docs/proofs/evidence/a4_memory.json`.
