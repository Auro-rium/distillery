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

**Student: Qwen/Qwen3-1.7B (user decision, 2026-10-03), LoRA, served on Nebius Sandbox CPU for base and student.** Basis: live spike (`docs/spikes/s5-qwen3-1_7b.json`): the base model fits a 4 vCPU / 4 GB sandbox with peak RSS about 3.8 GB (93%, almost no headroom), 10.6 to 28.8 s per sample depending on concurrency, best throughput 0.22 samples/s at concurrency 20. Accepted risk: a LoRA adapter adds memory, so the student may OOM where the base did not; an untrained-adapter memory pre-flight runs before the paid fine-tune (and an OOM there is reported and stops the run). Qwen3-0.6B is not used for this run.

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

## 2026-10-05 (pre-registration, recorded BEFORE any B2 scoring): fair teacher comparison (workstream B)
**Why.** The P4 gate compared the student with a zero-shot teacher. B asks whether the student still wins against a teacher that is given examples and against the stronger teacher model. It is cheap and needs no retraining.

**Setups.**
- S0 = Super zero-shot (the P4 prompt).
- S1 = Super with k=8 examples.
- S2 = Ultra with k=8 examples (the Ultra model id is mapped onto the teacher role).
- Student = the promoted adapter `4bebc793...`.

**The k=8 examples.** One fixed set for every question, drawn deterministically (seed 1234), stratified across train families, from the train split only.

**"Strongest teacher" is picked on dev (150), not held-out**, so held-out is not used for selection.

**Claim rule.** "Beats the teacher" only if the lower bound of the student / strongest-teacher ratio CI is **> 1.0**, using the same bootstrap as the gate.

**Disclosure.** The held-out re-score is a post-hoc analysis. The original P4 gate decision stands unchanged.

**Verifier fixes (B3)** apply symmetrically to every model: any confirmed false-negative fix to `taskpacks/sql/verifier.py` is applied once and every setup is re-scored from its cached outputs, with before/after reported for every model. No thresholds, splits or metrics change.

## 2026-10-06 (amendment to the 2026-10-05 B pre-registration, recorded BEFORE any B2 scoring): implementation details fixed by the code
These close gaps the pre-registration left open; no model has been called and no result exists. They change no threshold, split or metric.
- **Dev tie-break.** If two or more teacher setups tie on dev accuracy, the LATER one (S2 over S1 over S0) is the "strongest teacher": the claim must clear the harder opponent, so a tie never makes the comparison easier.
- **Few-shot layout.** The k=8 examples are chat turns: `[system, (user, assistant) x 8, user]`. Each example user turn has exactly the zero-shot format (schema, `Question:`, `/no_think`); the assistant turn is the template gold SQL of that TRAIN task; the final user turn is byte-identical to the S0 prompt. Selection (`prompts.select_fewshot_examples`): families sorted then shuffled with `random.Random(1234)`, tasks within a family sorted by id then shuffled with the same generator, round-robin over families, so 8 of the 10 train families appear once.
- **What is scored.** Held-out (300) for base, student, S0, S1, S2; dev (150) for S0, S1, S2 only (selection). The base is re-scored only so that the P4 gate (base 25.7%, student 92.3%, S0 90.0%) can be recomputed from the saved per-item vectors; if the regenerated counts differ from P4's (277, 270, 77 of 300) the difference is reported next to the claim, not hidden.
- **Claim rule.** `student_correct/teacher_correct` ratio, `paired_bootstrap_ratio_ci` with the gate's own `bootstrap_resamples` and `seed`; claim only if the lower bound is strictly above 1.0 (a bound of exactly 1.0 is not a claim). The comparison is against the strongest teacher chosen on dev; the other two teachers are reported for information only.
- **Human set.** Scored with the same setups only if it is already sealed in the run; otherwise recorded as `pending`, never as a result.
- **B3.** Rule-based classification (`misses.py`) is a hint for the user's review and changes no score. Only confirmed verifier false negatives are fixed, once, in `taskpacks/sql/verifier.py`, and every model is re-scored from its cached outputs (re-execution only).

## 2026-10-06 (C3, recorded before any live tool-call run and before the committed dry-run evidence): tool-call stress families
The tool-call pack reserves its stress families by the SAME written rule as the SQL pack: `random.Random(STRESS_SEED).sample(sorted(FAMILIES), k=STRESS_FAMILY_COUNT)` with `STRESS_SEED=777`, `STRESS_FAMILY_COUNT=2` (`taskpacks/toolcall/questions.py`). Over the 8 families (`bulk, conditional, cross_entity, dependent_chain, error_guard, lookup_then_act, multi_step, single_call`) the rule resolves to **`dependent_chain` and `single_call`**. They never appear in train, dev or the gate set; the stress set is reported separately and is never a gate input. This draw is final: no re-draw, whatever any run shows (a stress set that turns out easy or hard is reported as such). Gate thresholds, splits and metrics are the SQL pack's, unchanged. The rule and its constants were committed with the pack (C1-C2), before any pipeline run; the offline dry run (fake models, no result) reproduced the same two families in its report (`docs/proofs/evidence/c3_toolcall_dry_run.json`).

## 2026-10-07 (amendment to the A3 pre-registration, recorded BEFORE any controller run): lora_alpha follows lora_r
When a validated `adjust_hparams` changes `lora_r`, `lora_alpha` is rescaled to keep the pinned alpha/r ratio (`max(8, round(alpha * r_new / r_old))`, orchestrator `_apply_hparams`). This was an implementation choice not stated in the 2026-10-06 bounds; it is fixed here before any run uses the controller. `lora_alpha` is not itself a controller-proposable parameter.

## 2026-10-06 (amendment to the 2026-10-03 Gate B procedure, recorded BEFORE any question is written, drafted or scored): Gate B questions are agent-authored
The user delegated writing the Gate B question file to a Claude subagent (Opus 5.5), because no human question file exists. This replaces "the user supplies the questions (no LLM writes or edits them)" and "confirmed item by item by the user". Everything else in the 2026-10-03 Gate B rule stands (gold drafted by the teacher at temperature 0.8, k = 3, kept only if all three execute to the same non-empty result; same thresholds, 10,000 resamples, seed 1234; sealed after training, scored once; disclosed teacher-solvable bias).
- **Label.** This set is called the **agent-authored set** in every report and claim. It is never called human-written. The code keeps its `human*` names for file formats only.
- **Blindness (instruction-based, disclosed).** The author gets the schema DDL and a plain-language description of the workload only. It is told not to open the question templates, training/dev rows, run stores or sealed files. This cannot be enforced technically; it is disclosed as such.
- **Size.** Target 100 questions, fixed in the file before drafting. N and the file sha256 are recorded here by amendment before scoring.
- **Confirmation.** The authoring subagent (which knows each question's intent) reviews every teacher draft (question, SQL, result rows) and confirms or rejects it; decisions go to `decisions.jsonl`. The user is not the confirmer.
- **No targeting.** The author is not told which families the student trained on, nor told to avoid the stress concepts. It tags the SQL constructs each question needs in a separate file used only for post-hoc analysis, never as a gate input.

## 2026-10-06 (amendment to the B pre-registration, recorded BEFORE any B2 scoring): S2 (Ultra, k=8) deferred on budget
User decision: the full B2 estimate ($9.30, of which Ultra few-shot is ~$6.90) does not fit the $19.50 project cap next to the A5 live run. B2 runs now with **S0 and S1 only**; S2 runs later only if the console balance allows, in the same output directory (outputs are cached by prompt digest, so S0, S1, base and student are not regenerated).
- **Strongest teacher (now)** is picked on dev among the scored setups (S0, S1), tie to the later one, as pre-registered. Any claim made now reads "vs the strongest of zero-shot and 8-shot Super"; it is **not** a claim against Ultra, and the report lists S2 as `deferred`.
- **If S2 is scored later,** the dev pick is redone over S0, S1, S2 with the same rule, and the claim is re-stated against that pick. The earlier S0/S1-only result is kept in the record, not replaced.
- `--student-concurrency` lowers in-flight sandbox generations so B2 does not crowd A5 out of the sandbox operation cap; it changes speed only, not outputs.
- Nothing else changes: same few-shot set (seed 1234, k=8, train only), same claim rule (ratio CI lower bound > 1.0), same gate bootstrap.

## 2026-10-06 (amendment, recorded BEFORE drafting or scoring): agent-authored Gate B file fixed
`.distillery/humanset/questions.txt` (gitignored, kept out of git like every sealed source): **N = 100 questions, sha256 `f791e32967f0e59aad7da28f8e53c4e183f5b99134662fc738b888d80c24da0a`**. Author's split: 25 simple, 49 medium, 26 hard (target was 25/45/30; reported, not changed). The author's reference SQL (`reference.jsonl`) is used only to review teacher drafts during confirmation; it is never a gate input and never shown to a model. The file is not edited after this entry; drafting discards and confirm rejections are reported as counts.

## 2026-10-06 (amendment, recorded BEFORE Gate B is scored): agent-authored Gate B confirmed set
Drafting (teacher k=3, temp 0.8): 100 questions, 87 kept, 13 discarded (12 disagree, 1 empty). Independent check: all 87 kept drafts execute to the same result as the author's blind reference SQL. Author review: 86 confirmed, 1 rejected (`h-60-8f3b5adc`: right only because no account in this db has two ended subscriptions). **Gate B N = 86**, `human_confirmed.json` sha256 `68816123b3ab5c9fd322fd5d3fe256fa8931ade9517e944538ca9bd83d34d892`, db sha256 `03864579…`. Scored once against the promoted P4 adapter with the pre-registered thresholds, in a copy of the P4 store (the source store is read-only for B2). Disclosed: teacher-solvable bias (13% discarded at drafting), agent-authored, confirmer = author.

## 2026-10-06 (A7: outage-aware restarts, recorded after the A5 attempt that exhausted restarts on a local DNS outage)
A local DNS outage ("[Errno -3] Temporary failure in name resolution" reaching the Token Factory host) made every attempt fail in the `headroom` stage and burned 5 of the 6 supervisor restarts in about 25 minutes while nothing was wrong with the run. **Rule:** before the supervisor COUNTS a restart (exit 75 or death by a signal, the watchdog's SIGKILL included), it probes the provider host from the server process (DNS + TCP + TLS to the host of `NEBIUS_BASE_URL`, default `api.tokenfactory.nebius.com:443`). If the host is unreachable the restart is NOT counted: the run waits, probing every `OUTAGE_POLL_S` (30 s), audits `supervisor:outage_wait` once and `supervisor:outage_over` once, and restarts immediately when the host answers. The total outage wait per run is capped at `MAX_OUTAGE_WAIT_S` (6 h, env `DISTILLERY_MAX_OUTAGE_WAIT_S`); after that the normal counted-restart path applies. If the host is reachable, behaviour is unchanged (a retryable failure on a working network is a real provider or infrastructure problem and consumes the budget). Dry runs make no provider calls and are not probed. Cancel, suspend and server stop interrupt the wait at once. This changes no pre-registered criterion: no threshold, split, metric, seed or protocol; it only decides when an identical resume is attempted.

## 2026-10-07 (pre-registration of round N1, recorded BEFORE any N1 question exists, is drafted, trained on or scored): natural-phrasing training data
**Why.** P4 PROMOTEd on the templated gate set (92.3% vs base 25.7%) but scored 40.7% vs base 38.4% on the 86-question agent-authored Gate B set (REJECT, McNemar p 0.84) and 16% on stress. Reading: the student learned template phrasing, not the task. N1 tests whether training on naturally phrased, teacher-verified questions closes that gap.
- **Training addition.** A Claude subagent (Opus 5.5, label **agent-authored**, never "human") writes N1-train: about 320 natural questions over the same schema, blind by instruction to the templates, the P4 training/dev rows, the Gate B questions and the N1-eval set. It also writes a reference SQL per question. Teacher (Nemotron 3 Super) drafts k=3 at temperature 0.8; a row is kept only if all three execute to the same non-empty result AND that result equals the reference SQL's result. Kept rows are appended to the P4 training rows (originals + verified paraphrases), nothing removed. Rows whose normalised question equals any Gate B, N1-eval, held-out or stress question are dropped and counted.
- **Scoring set.** N1-eval: at least 80 questions authored by a second, separate subagent, blind to N1-train and to the Gate B set, drafted/confirmed by the same rule as Gate B. It is fresh: the Gate B set is NOT used to judge N1 (it already informed the design). Scored once, sealed before training, base/student/teacher with the same prompt.
- **Primary claim (fixed now, no change after results).** PROMOTE on N1-eval only if the paired-bootstrap lower bound of student/teacher accuracy is >= 0.85 AND student beats base by exact McNemar p < 0.05 (the existing gate, unchanged). **Secondary (reported, not a gate):** the original 300-item held-out must not regress (ratio lower bound >= 0.85), and Gate B (n=86) is re-scored and reported as a post-hoc transfer number only. Any other outcome is REJECT and is reported as such.
- **Setup.** Same base (Qwen3-1.7B), same hyperparameters as P4 (lr 1e-4 pinned by P1.12), one fine-tune round, no controller changes. Reuses the P4 cached stages; only rows, fine-tune and evaluation are new.
- **Budget.** Hard ceiling $5.00 of new spend (est. fine-tune ~$2.6, teacher drafting ~$0.8, scoring ~$0.4); recorded project spend to date $10.32 of $19.50. Preflight estimate is printed and checked before each paid step; stop on any overrun.
- **Disclosure.** Authors and confirmer are Claude subagents, not humans; the teacher that drafts gold is also the teacher the student is compared with (selection bias, as for Gate B); N1-train and N1-eval share an author model family, so a gap vs a human-written set may remain.

## 2026-10-07 (amendment, recorded BEFORE any N1 drafting, training or scoring): N1 question files fixed
Both files are agent-authored (Claude Opus 5.5 subagents, blind to templates and to each other by instruction; each read only `schema.sql` and the db read-only). Kept out of git like every sealed source (`.distillery/`).
- **N1-train:** `.distillery/natural/train_questions.txt`, **N = 320**, sha256 `4ccc1b198bd095ff9905da8373a96d6555feb9f6f8ac91610a7e998325fff504` (64 simple / 144 medium / 112 hard). Reference file 0-based.
- **N1-eval:** `.distillery/natural_eval/questions.txt`, **N = 110 drafted** (at least 80 expected after teacher confirmation; the confirmed N and sha are recorded by a further amendment before scoring), sha256 `375e07ae97c6ec048a6e63c5b2727b5200b5020ba1b9778cd642691f23421ad7` (22 simple / 48 medium / 40 hard). Reference file `i` is 1-based.
- Checked offline: zero exact (normalised) overlap between N1-train, N1-eval and the Gate B file.
- Unchanged: gate thresholds, claim rule, $5.00 ceiling, one round, P4 hyperparameters.

## 2026-10-07 (amendment, recorded BEFORE N1 training or any N1 scoring): N1-eval confirmed set
Teacher drafting (k=3, temp 0.8) of the 110 N1-eval questions: 98 kept, 12 discarded (8 disagree, 4 exec_error). Spend $0.2065 (estimate, ledger `humanset-375e07ae`).
Confirmation is **mechanical, not by the authoring agent**: a kept draft is confirmed iff its result equals the result of the author's blind reference SQL (`compare_outcomes`, ordered only when the reference has a top-level ORDER BY). All 98 confirmed, 0 rejected. This is stricter than Gate B's judgment-based confirm on one axis (an independent reference must agree) and removes the confirmer's discretion.
**N1-eval confirmed set: N = 98, `human_confirmed.json` sha256 `3ab039710037195134128743f82b1434c939d1e24efb6ff5ece484f0dc875beb`** (db sha256 prefix `03864579a97c4336`). Same bias as Gate B, disclosed: gold was drafted by the teacher and kept only where all three drafts executed non-empty and agreed with the reference, so the teacher's accuracy on this set is inflated and the base/student numbers are the informative ones.
Round spend so far: $0.21 of the $5.00 ceiling.

## 2026-10-07 (amendment, recorded BEFORE the N1 fine-tune starts or anything is scored): N1-train rows fixed
Teacher drafting of N1-train (k=3, temp 0.8, kept only if all drafts equal the reference result): **249 of 320 kept** (discarded: 48 disagree, 12 exec_error, 9 ref_mismatch, 2 empty). Spend $0.683 (estimate). `natural_train.json` sha256 `bd327eec4ce554eeb9806ce3e11e1646e3461df22f25bed5460b62073291996b`. Overlap check vs the Gate B file and N1-eval: 0 exact, 0 skeleton. These 249 rows are appended to the P4 rows; the run is a copy of the P4 store under P4's run id so stages up to paraphrase hit the cache. Round spend so far $0.89 of the $5.00 ceiling; remaining plan: fine-tune (~$2.7) + final-eval teacher (~$0.25) + N1-eval scoring (~$0.1) ≈ $4.0 total.

## 2026-10-12 (result of round N1, scored 2026-10-09T04:55Z and recorded after scoring; no threshold changed): N1-eval REJECT
Evidence: `docs/proofs/evidence/n1_natural_round.json` (written by `docs/proofs/n1_evidence.py` from the reports and ledgers). Run `sql-gated-20261004T155134Z-66ff347c` (N1 reuses P4's run id on a copy of its store), adapter `7d6ce813…`, job `ftjob-6cfd065b…`, 2,230 rows (1,981 P4 + 249 natural, file `bd327eec…`), P4 hyperparameters, 1,674 steps. Gate A decision unchanged (PROMOTE).
- **Primary.** N1-eval, n = 98, agent-authored, mechanically confirmed (`3ab03971…c875beb`), scored once: base 0.296, student 0.571 (95% CI 0.473 to 0.665), teacher 0.959. Student/teacher ratio point 0.596, lower bound **0.495 < 0.85**; McNemar exact p = 4.6e-07 (student-only 29, base-only 2). The ratio bound fails, so the pre-registered rule gives **REJECT**. Thresholds, seed and resamples are the pre-registered ones.
- **Secondary (reported only).** The 300 templated held-out (same set as P4): base 0.2567, student 0.920, teacher 0.900, ratio lower bound 0.989, McNemar 201 vs 2: no regression (P4: student 0.9233, lower bound 0.993).
- **Stress (reported only, never a gate input).** n = 100: base 0.09, student 0.66, teacher 0.98; date_math 0.645 (P4 0.210), set_ops 0.684 (P4 0.079); P4 overall 0.16. Exact question overlap with stress was checked (0 rows dropped); concept overlap was not, and the natural questions can teach date and set concepts, so these families are no longer unseen concepts for N1.
- **Spend** (store ledgers; sandbox is recorded at $0, so all figures are lower bounds). N1 $3.8289 of the $5.00 ceiling: drafting N1-eval 0.2065 and N1-train 0.6826, plus the N1 run beyond P4's stages 2.9398 (fine-tune 2.6288, LLM 0.3110 incl. scoring N1-eval 0.0610). Project total since P4 $14.1952 of $19.50 (P4 4.2582, B2 1.6495, Gate B 0.2668, A5 4.1918, N1 3.8289; rounded to 4 d.p., A5 as recorded in `a5_chaos.json` (ledger 4.191885); exact total 14.195248). Correction: Gate B's ledger total is 0.2668 (drafting 0.1743 plus two scoring passes, 0.0925), not 0.22, so the $10.32 recorded in the N1 pre-registration was $10.37 (4 d.p.: 10.3664).
- **Pending.** Gate B (n = 86) re-score of the N1 adapter as a post-hoc transfer number: needs a small paid run (one earlier Gate B pass cost about $0.05), not run, no number exists.
- **Reading.** Naturally phrased training data moved the student from indistinguishable-from-base on Gate B (P4 adapter: 0.407 vs base 0.384, McNemar p 0.84) to significantly better than base on N1-eval (0.571 vs 0.296), but it stays below the 0.85 ratio bound. The two sets differ, so this is not a measured improvement. Teacher accuracy on both sets is inflated by construction (gold drafted by the teacher and kept only where its drafts agree), so the base and student numbers are the informative ones. One run, one seed.

## 2026-10-13 (pre-registration, recorded BEFORE any calibration, test or burst item is scored): H3 certified router
This commit's hash is what `distillery router signals|certify|evaluate --prereg-commit` checks: it must be an ancestor of HEAD and this text must contain each sealed sha256 below. Advisor-reviewed before commit. Nothing below changes after any calibration, test or burst item is scored.

**What is certified.**
- Adapter: N1, `7d6ce813…` (job `ftjob-6cfd065b…`), base Qwen/Qwen3-1.7B.
- Serving path: Nebius Sandbox CPU, transformers bf16, greedy, max_new_tokens 160, merged adapter. This is the same path the hosted router uses.
- Teacher: Nemotron 3 Super (`nvidia/nemotron-3-super-120b-a12b`), zero-shot, the canonical eval prompt.

**Guarantee.**
- With probability at least 1 − δ over the draw of the calibration set, the selective error of student-served answers is at most α, for traffic drawn from the calibration mix (Learn-then-Test, fixed-sequence binomial tests).
- α = **0.08**, δ = **0.05**.

**Derivation of α** (already-recorded numbers from the N1 run report only; no new data).
- Teacher error: 10.0% on the 300 templated held-out (0.900), 4.1% on N1-eval (0.959), 2.0% on stress (0.98).
- Weighted 0.50 / 0.25 / 0.25 plus 2 points: 0.0500 + 0.0103 + 0.0050 + 0.02 = 0.0853, **rounded down to 0.08**.
- Disclosure: natural items survive drafting only when the teacher's drafts agree, which deflates teacher error on natural items. Calibration and test natural items pass the same filter, so α and the guarantee describe the same filtered mix.

**Splits.**
- **Fit** (already scored in earlier rounds; used only to fit the combiner, fix the grid and set the drift parameters; never for certification or any pass/fail number): held-out 300, stress 100, Gate B 86, N1-eval 98 = 584 rows, `signals_fit.json` sha256 `fe9cf67142f37c5c4f759b5c3b8faccc68dba4649655f12609c55d5665722247`. Regenerating them reproduced N1's recorded scores exactly (held-out 276/300, stress 66/100, N1-eval 56/98; criterion ±2 items, written before the run).
- **Calibration:** n = 489, sha256 `fe725827586278074b8847a0aeaec85d548c928eb31d8ef700b78e42925db137` (templated 250 / natural 114 / stress 125).
- **Test:** n = 490, sha256 `abc6e2943d3b68b0b4e513e7050de9690abf16d05a19a8bfbfe8860ab68de75d` (templated 250 / natural 115 / stress 125).
- Natural fell 21 short of the 250 target: 300 written → 239 kept by drafting (38 disagree, 22 exec error, 1 no SQL; $0.686) → 229 mechanically confirmed against the author's blind reference SQL (10 rejected; `human_confirmed.json` sha256 `5b307813ba16b5f52ca6492a49908e0684bab09b02c785d78dec5c59377c9f30`, question file `094e5fc0…`).
- How each kind was built: templated = fresh draws from the 10 training families, seed 20261012; stress = fresh `date_math` + `set_ops` draws, excluding the sealed 100; natural = written by a blind author from the schema and DB only, drafted by the teacher (k=3, temp 0.8, kept if all agree), mechanically confirmed. One pool per kind, split at random (seeded) into calibration and test, so the two are exchangeable.
- **Overlap removed** (exact) against N1's 2,297 training questions (the split's train stage plus both round training files, incl. the 249 natural), dev, gate, stress, Gate B, N1-eval, and calibration vs test. Drops: templated 157 (train 101, gate 38, dev 18), stress 17 (sealed stress set), natural 0. Skeleton overlap is reported, not dropped: templated items share skeletons with training by construction (243 per split); natural and stress share none with training. Builder report: `router/splits.json` in the N1 run dir.

**Signals and combiner (fixed now).**
- `exec_ok`: the extracted SQL runs read-only with no error and returns ≥ 1 row.
- `mean_logprob`: the mean token log-probability of the answer tokens, think block excluded.
- `nn_sim`: the maximum cosine similarity to the same 2,297 training questions, Qwen3-Embedding-8B truncated to 1024 dims (first 1024 components, re-normalised).
- **Combiner:** logistic regression (IRLS, L2 = 0.01) on standardised features, fitted on **fit only**: weights exec_ok 2.9737, mean_logprob 0.7096, nn_sim 0.6034, bias 1.2858. Model sha256 `5592a87b02131075dfd5a21c516604784e8c438a7dd7c3b73adf0ce6ccfb364c` (copy: `docs/proofs/evidence/h3_router_model.json`).
- **Short-circuit:** if even exec_ok = 1 and logprob = 0 cannot reach λ̂ at this nn_sim, the request goes straight to the teacher. Decisions are identical with and without it.

**Certification** (`distillery router certify --alpha 0.08 --delta 0.05`, calibration only).
- Grid: thresholds at fit-split coverage 0.20, 0.22, …, 0.90, most to least conservative.
- At each step p = BinomCDF(errors; n_selected, α); reject if p ≤ δ; stop at the first non-rejection. λ̂ = the last threshold that rejected; if none did, nothing is served by the student.
- Sizing on the real fit signals (`scripts/h3_router_sizing.py`, 1,000 bootstrap pipelines of the 5-fold out-of-fold scores, evidence `docs/proofs/evidence/h3_router_sizing.json`): P(certification fails) 0.000; mean test coverage 0.575; mean test selective error 0.051; P(test point estimate > 0.08 | certified) 0.191; **P(W-C1 risk and coverage clauses pass) ≈ 0.78**. Certifying at a stricter internal α was simulated (0.07 → 0.82, 0.06 → 0.83, 0.05 → 0.73) and **not adopted**: the guarantee is stated at 0.08 and certifying at 0.08 is the honest test of it. The bootstrap resamples about 290 items per half with replacement, so it overstates test variance somewhat.

**Test, scored ONCE.** No refit, recertify or threshold change after any test item is scored. The student and the teacher are both scored on every test item.

**W-C1 (pass = all three):**
1. Selective error point estimate on test ≤ 0.08.
2. Test coverage ≥ 0.30 (a vacuous pass is FAIL).
3. System accuracy (student where served, teacher otherwise) ≥ teacher-only accuracy on test − 0.01.
Also reported: the Clopper–Pearson 95% upper bound on selective error; system accuracy vs 8-shot Super (B2 S1, 0.933 on held-out), context only.

**W-C2 (pass = both):** let E = test items of kind natural or stress the student gets wrong.
1. At least 80% of E are deferred to the teacher.
2. The one-sided 95% Clopper–Pearson **lower** bound on that deferred share **exceeds** the deferral rate on the student's correct natural + stress test items.
Reported, not gated: counts per family; the stress routing share; natural-phrasing errors deferred vs served wrong.

**W-D drift.**
- **Burst:** 30 questions, fixed before any scoring: a new product line, credit grants. Written by a blind author from the schema and DB only, against a copy of the billing DB plus one new table `credit_grants` (376 seeded rows, `build_extension.py`, Random(20261012)). Every reference SQL needs that table. No N1 training, natural, Gate B or N1-eval question mentions credits (0 exact, 0 skeleton overlap). sha256:
  - extended DB `e50f7b39e1a39065c0fd1922f9bc4b6c1aa719dca72b48d2d89d980ed15a50cd`
  - questions `447103a535175859a3e9cee591e7cdcac438f11f89f06ca75161c3534b2878b2`
  - references `cfe2526a3bba767697c563caeb30aaf4de03d6689919d74b48e11b687aa6fb0c`
  - During the burst the prompt's DDL includes `extension.sql` and SQL runs on the extended DB. Scored by `distillery router signals --split burst` (refuses unless this commit holds all three hashes); student and teacher answer every burst item, so the replay is free.
- **Detector (fixed now; evidence `docs/proofs/evidence/h3_drift_params.json`, `scripts/h3_drift_params.py`, fit split only):**
  - (a) deferral share over the last **W = 20** requests ≥ **τ**, armed after **20** requests. τ is read from the table below at the **certified grid level**: `levels[i]` in `cert.json`, where `i` is the position of λ̂ in `cert.json`'s `grid` (by position, never by float matching); each τ is the smallest value with ≤ **0.1 false rollbacks per 1,000** in-distribution fit requests (iid resampling; the fit rows are stored grouped by source, so a block bootstrap would invent runs of deferrals).

    | certified fit-coverage level | τ |
    |---|---|
    | 0.20–0.32 | none feasible: W-D is FAIL (not run) |
    | 0.34–0.40 | 1.00 |
    | 0.42–0.48 | 0.95 |
    | 0.50–0.54 | 0.90 |
    | 0.56–0.60 | 0.85 |
    | 0.62–0.64 | 0.80 |
    | 0.66–0.68 | 0.75 |
    | 0.70–0.72 | 0.70 |
    | 0.74–0.76 | 0.65 |
    | 0.78–0.80 | 0.60 |
    | 0.82–0.84 | 0.55 |
    | 0.86 | 0.50 |
    | 0.88 | 0.45 |
    | 0.90 | 0.40 |

  - (b) the teacher shadows **10%** of student-served requests; **≥ 10** audit disagreements in the last **50** audited answers trip it (P(Binomial(50, 0.08) ≥ 10) = 0.0056 per window; the earlier draft's k = 2 would give 0.917 per window). In this 230-request replay only about 10–13 answers are audited, so (b) essentially cannot trip and **W-D is decided by detector (a) alone**; (b) is there for long-run hosted traffic.
  - Either one trips: CERTIFIED → ROLLBACK → TEACHER. Re-distil is queued, not run. Returning to CERTIFIED needs a new gate.
- **Replay (one pre-registered run, after `router certify`):** with `R = .distillery/n1live/runs/sql-gated-20261004T155134Z-66ff347c/router`,
  `uv run python scripts/drift_sim.py --fit-rows $R/signals_fit.json --in-dist-rows $R/signals_test.json --burst-rows $R/signals_burst.json --router-dir $R --target-fa 0.1 --tau <τ from the table> --window 20 --min-requests 20 --audit-rate 0.10 --audit-window 50 --audit-k 10 --n-before 200 --n-burst 30 --seed 20261013 --out docs/proofs/evidence/drift_burst.json`
  (λ̂ comes from `$R/cert.json`; rows are scored with `$R/model.json`). In-distribution rows are the 490 scored test rows and burst rows the 30 scored burst rows; the simulator draws both **with replacement**, so some burst questions repeat and some are absent in the 30-request burst. Audits replay as disagree iff the recorded student answer is wrong. The command was smoke-tested on fit rows before this commit.
- **Pass (all):**
  1. No rollback before the first burst request (a false alarm is FAIL).
  2. Rollback within **N = 25** requests of the first burst request, counting it as 1.
  3. Wrong student answers served during the burst before rollback ≤ **3**.
  If certification fails (no λ̂) or the certified level has no feasible τ, W-D is FAIL.
- Simulation behind N and the bound (synthetic bursts, 300 seeds, level 0.56 / τ 0.85): burst deferral 1.0 → median 14, p95 17, 0 wrong; 0.95 → p95 19, p95 wrong 3; 0.9 → p95 23, 7/300 missed; 0.8 → 105/300 missed. Pre-burst false alarms 3/300 per 200-request prefix. So W-D passes if the router defers about ≥ 90–95% of burst requests and fails otherwise; it tests the router's out-of-distribution deferral plus the state machine.
- Reported, not gated: 0 student answers after rollback (holds by construction); the same replay over seeds 0–299 (latency quantiles, miss rate, false alarms); the false-alarm count on a 1,000-request test-row stream without a burst; a second scenario whose burst is the fit family with the highest student error, **stress `date_math`** (fit error 22/62 = 0.355), drawn from the test rows of that family.

**Recorded per item:** output sha256, signals, score, route, correctness, so the served path is reproducible.

**Disclosures.**
- The guarantee covers the calibration mix only, not arbitrary traffic. One adapter, one calibration draw.
- Fit sets were used in earlier gates.
- N1 scores 0.66 on stress, so stress is **not** an unseen family for N1; no "unseen family" claim is made.
- The teacher touched the natural cal/test items during drafting (k=3, temp 0.8, drafting prompt). Teacher-only accuracy on test comes from a separate zero-shot scoring pass and on natural items is near 1.0 by construction.
- Gate B re-score of N1 was done without new calls and is unpaired (`docs/proofs/evidence/n1_gate_b_unpaired.json`: 66/86 = 0.767); it is a fit-split number, not an H3 result.

**Spend pre-flight** (approved 2026-10-13 as step C): student generation on cal 489 + test 490 + burst 30 on Sandbox CPU (sandbox seconds unpriced, recorded at $0), embeddings ≈ $0.0005, teacher zero-shot on test 490 + burst 30 ≈ $0.33. Total ≈ $0.33 of the remaining $3.12 under the global ledger (committed $16.38 incl. the $1.50 hosted reserve, cap $19.50). Each `router signals` call carries a run-total cap (`--budget-usd`) of the run's ledger total plus $0.40.

## 2026-10-14 (result of H3 calibration and test, scored 2026-10-13 / 2026-10-14 under pre-registration 6b37025; no threshold, feature or model changed): W-C1 PASS, W-C2 PASS
- **Certification** (calibration n=489, signals sha256 `875cfdcb…dd358`, `cert.json` sha256 `ebacbc2e…71d4fb`): λ̂ 0.9172 at grid position 15 (fit-coverage level 0.50). The student is selected on 226 calibration items with 10 errors (risk 0.044, Clopper–Pearson 95% upper 0.074, p 0.025 ≤ δ 0.05); the next step (238 items, 13 errors, p 0.088) did not reject. Copy: `docs/proofs/evidence/router_cert.json`.
- **Test, evaluated once** (n=490, signals sha256 `0187648e…d3c`, `docs/proofs/evidence/router_eval_test.json` sha256 `860299db…285db1`). Student-only accuracy 356/490 = 0.727; teacher-only 0.935.
  - **W-C1 PASS** (all three): selective error 10/240 = 0.042 ≤ 0.08 (CP 95% upper 0.070); coverage 0.490 ≥ 0.30; system accuracy 0.951 ≥ teacher-only 0.935 − 0.01. Context only: 8-shot Super (B2 S1) scored 0.933 on held-out, a different set.
  - **W-C2 PASS** (both): of 117 natural + stress items the student gets wrong, 115 are deferred (0.983 ≥ 0.80); CP 95% lower bound 0.947 > the deferral rate on the student's 123 correct natural + stress items, 113/123 = 0.919.
  - **Plain reading:** the separation clause passes by a small margin. On natural and stress traffic the router defers almost everything, right or wrong (stress routed to the student: 0.064). Of the 240 answers the student serves, 228 are templated, in-distribution questions. Per family: natural 62/62 wrong deferred, 49/53 right deferred; date_math 33/34 and 37/38; set_ops 20/21 and 27/32.
- **Attempts disclosed.** Test scoring was started three times. 2026-10-13 15:18 and 16:00 IST: both died on local DNS failures (embedding retries exhausted; sandbox transport error) during student generation, before any teacher call and before any row was written or seen. The retry fix `a9b65a1` (waits between re-submissions; longer API retries) touches no scoring, routing or certification code. The third attempt (2026-10-14 14:05) completed. Test-split spend ≈ $0.319 (estimate, global ledger).
- **W-D:** pending. The burst scoring run (2026-10-14 14:40) was stopped by the host for low memory before any burst row was written; nothing was scored.

## 2026-10-14 (result of W-D, run once under pre-registration 6b37025 after certification; no parameter changed): W-D PASS
- **Burst scoring:** 30 rows, student correct 11/30, signals sha256 `f1624361…63a4f`, ≈ $0.023. Run 2026-10-14 15:40–15:45 IST; the earlier 14:40 attempt was stopped by the host for low memory before any row was written.
- **τ:** λ̂ sits at grid position 15 of `cert.json`, level 0.50 → τ 0.90 from the pre-registered table.
- **Replay:** the pre-registered command, plus `--pass-within 25 --max-wrong-during-burst 3`, which only evaluate the pre-registered pass rule (`docs/proofs/evidence/drift_burst.json`, sha256 `fec60523…f538e2a`). Detector (a) tripped on burst request 13 (rollback index 212). No rollback before the burst; 0 wrong student answers served during the burst; 0 after the rollback. The 4 wrong student answers in the run were served before the burst, on in-distribution traffic. Verdict: **PASS** (all three clauses).
- The router defers each of the 30 distinct burst questions on its own (score < λ̂ for every one), so the student never answers a credit-grants question even before the rollback. Counts (no question text): `docs/proofs/evidence/h3_burst_counts.json` (`scripts/h3_burst_counts.py`): n 30, student correct 11, teacher correct 27, deferred 30, served by the student 0. That file also copies the pre-registration and the three pre-registered burst hashes, which the W-D receipt cites.
- **Reported, not gated** (`docs/proofs/evidence/h3_drift_report.json`, sha256 `ce95658d…a5ad`): same replay over seeds 0–299: 295 pass, 5 fail on a false alarm before the burst, 0 misses, rollback median 15 / p95 18 / max 18, 0 wrong during any burst. 1,000-request test stream without a burst: no rollback. date_math burst (72 test rows): 293/300 pass, 1 miss, max 3 wrong during the burst.
- H3 + W-D spend: drafting $0.6860 + signals and scoring $0.3433 = $1.0293. Global ledger $15.2245 + $1.50 reserve, cap $19.50.

## 2026-10-16 (pre-registration, recorded BEFORE any W-E2E request is sent): W-E2E, the live end-to-end run
The user approved live end-to-end runs on Nebius and the Render deploy on 2026-10-16 ("make live e2e runs on the nebius … deploy it on the render"). This commit's hash is what `scripts/w_e2e.py --prereg-commit` checks: it must be an ancestor of HEAD and this text must contain the normal questions' sha256 below. Nothing below changes after the first request is sent.

**Claim.** The whole serving path runs live: the certified router (H3, λ̂ 0.9172, model `5592a87b…`) loaded with the exact N1 checkpoint it was certified on, the student answering on Nebius Sandbox CPU, deferral to the teacher (Nemotron 3 Super on Token Factory), the website's own endpoint, and the drift guard rolling the gateway back on a burst. It is a systems claim; accuracy is W-C1's and is not re-measured here.

**Setup (local live server, global ledger on).** `python -m distillery serve --host 127.0.0.1 --port 8765` from commit HEAD with `.env` sourced (key names only: `NEBIUS_API_KEY`, `NEBIUS_AI_PROJECT`, `NEBIUS_BASE_URL`, model ids) plus: `DISTILLERY_HOME=<fresh empty dir>`, `DISTILLERY_PRICES_FILE=deploy/prices.json` (as on Render), `DISTILLERY_ROUTER_DIR=deploy/router`, `DISTILLERY_PLAYGROUND_STUDENT_DIR=deploy/student` (checkpoint digest `7d6ce813882bf3a211c84bfab7f8109b3bcdd6ca4617349a03bff58076b85193`, pinned by `deploy/router/student.json`), `DISTILLERY_PLAYGROUND_SANDBOX_MODELS=1`, `DISTILLERY_PLAYGROUND_DAILY_CAP_USD=0.15` (the server's own spend ceiling for the day), `DISTILLERY_GATEWAY_PER_IP_PER_HOUR=200` (the default 30 would refuse a 50-request run from one address; disclosed). The overrides are exported after `.env` is sourced, so they win. Drift parameters from `deploy/router/drift.json`, the W-D values: W 20, τ 0.90, armed after 20, audit 10% / M 50 / k 10.

**Run.** `uv run python scripts/w_e2e.py --base-url http://127.0.0.1:8765 --burst-questions .distillery/router_burst/questions.txt --prereg-commit <this commit> --out docs/proofs/evidence/w_e2e.json`, once.
- Normal phase, 20 questions in the fixed order in `scripts/w_e2e.py` (sha256 of the newline-joined list `0b207859f80c5bb18fba993f5179ab29c02992120ed5e1d0bce84ed72bd74bcb`): the website's six example questions and 14 fresh templated draws (`generate_tasks(<demo DB>, 14, seed 20261016, exclude_families=<stress>, skeleton_cap=<default>)`, the traffic the certificate covers). The demo DB is byte-identical to the run DB (sha256 prefix `03864579`). Free check before this commit: 1 of the 20 ("Return every plan name that has trialing subscriptions along with how many.") is word for word an N1 training question; 0 match dev. The first three go through `POST /api/acme/ask` (the website path), the rest straight to `POST /v1/chat/completions`.
- Burst phase: the 30 pre-registered credit-grant questions (sha256 `447103a5…78b2`, W-D's set), asked against the demo schema. **Different from W-D:** the gateway routes only the schema it was certified on (an unknown schema goes to the teacher outside the drift monitor), so the burst prompt carries the demo DDL, which has no `credit_grants` table; burst answers are not scored. The drift state is read after every burst request. The burst text never goes into the evidence file.

**Pass rule (all of C1–C6; computed by the script exactly as written):**
- C1: `/api/gateway/status` before the run: certified, λ = 0.9172253676263808, served and certified adapter digests both `7d6ce813…`, student "ok".
- C2: at least 1 normal question served by the student, and every student-served answer verified (its SQL ran on the demo DB with ≥ 1 row).
- C3: at least 1 normal question deferred by a router decision (score under λ̂, or the short-circuit precheck).
- C4: every request answered HTTP 200 (a 429 or any 5xx fails).
- C5: drift state CERTIFIED before the run and after the normal phase (no rollback before the burst); rollback (ROLLBACK or TEACHER) within 25 burst requests counting the first as 1; 0 student-served answers after the rollback.
- C6: global-ledger spend delta ≤ $0.15. The script stops when the ledger has moved by more than $0.15 (a stop is FAIL).
- **Reported, not gated:** per-request route, score, latency (student median and max), audits, savings, the ledger delta.
- **Free preconditions (not a run):** before any question is sent the script reads `/api/config`, `/api/gateway/status` and `/api/gateway/drift` and exits 2 without writing evidence unless the student model is `Qwen/Qwen3-1.7B`, the server's daily cap is ≥ the $0.15 ceiling, C1 holds and drift is CERTIFIED. It also refuses to start unless `src/`, `deploy/` and `scripts/w_e2e.py` are committed and unchanged since this commit, and it records HEAD and its own sha256. A stop at these reads is not a run.
- **Repeat rule:** if the run dies on a local network or host failure with no HTTP response (as on 2026-10-13), it may be started again from a fresh `DISTILLERY_HOME`, disclosed here; any HTTP response, pass or fail, stands.

**Unscored live smoke, before this commit (not evidence).** The serving path (shipped checkpoint → sandbox student → gateway) had only been tested with fakes, so 3 fresh templated draws (seed 20261017, not in the list above) went through `/v1` on a throwaway server home: 2 served by the student (scores 0.947 and 0.974, both verified; 148 s for the first, which built the serving image, then 50 s), 1 deferred by the precheck (6.6 s). For comparison, templated calibration rows: median score 0.958, 86% ≥ λ̂ (natural 7%, stress 3%), so the live path scores like the certified one. Smoke spend ≈ $0.0007. Expected, not gated: about 12 of the 14 templated questions served by the student, most of the 6 website examples deferred, rollback around burst request 15–18.

**Hosted check (reported, separate).** After the Render deploy: the same script with `--hosted` against `https://distillery.onrender.com` (normal phase only, C1–C4, no burst, so the public gateway is never rolled back on purpose) → `docs/proofs/evidence/w_e2e_hosted.json`. Hosted spend is not in the laptop ledger; it comes out of the $1.50 hosted reservation and its daily cap.

**Spend pre-flight.** Local: about 30–40 teacher calls (Super, ≈ $0.0005 each: about 1,200 input and 150 output tokens), 3 summaries (Nano, ≈ $0.001 each), 50 embeddings (≈ $0.00002) → ≈ $0.03; hard ceiling $0.15. Hosted: ≈ $0.01 inside its daily cap. Sandbox seconds for about 20–35 student generations plus one serving-image build are unpriced (recorded at $0, so the ledger delta is a lower bound). Global ledger before: spend $15.2245, committed $16.7245 with the $1.50 reserve, cap $19.50.

## 2026-10-16 (result of W-E2E, run once under pre-registration 705876d; nothing changed after the first request): W-E2E PASS
- **Run:** 2026-10-16, local live server from commit `705876d` (recorded in the evidence with the script's sha256), fresh `DISTILLERY_HOME`, global ledger on. One run; no repeat was needed. Evidence `docs/proofs/evidence/w_e2e.json` (sha256 `3ef1ab31…74ba`).
- **C1–C6 all true.** Certified at λ̂ 0.9172, served and certified checkpoint both `7d6ce813…`.
- **Normal phase (20):** 12 served by the student, all 12 verified, all 12 templated (12 of the 14 templated draws; the other 2 scored 0.828 and 0.904 and were deferred). All 6 website example questions went to the teacher: 4 by the similarity precheck, 2 on low confidence (0.858, 0.807). No rollback before the burst (deferral share 0.29 after the 14th counted request).
- **Burst (30):** rollback (TEACHER) on burst request 14, limit 25; 0 student answers in the burst.
- **Reported, not gated:** student-served answers took a median 38.8 s (max 72.5 s) client-side on Sandbox CPU; the first website question took 228.7 s (cold serving image, then a low-confidence deferral). Burst requests 2.5–10.6 s. 2 audits, 0 disagreements, $0.0010. Global ledger delta $0.0331 (ceiling $0.15); with the unscored smoke, spend is now $15.2583 + $1.50 reserve, cap $19.50. Sandbox seconds are unpriced.
- **Reading:** the live path does what the offline evaluation said: it serves templated, in-distribution questions and hands the naturally phrased website examples to the teacher. The website's example questions therefore show the teacher, not the student; the hosted check and the website should include templated examples so a visitor can see both paths.

## 2026-10-22 (hosted deploy and hosted W-E2E check): PASS
- **Deploy:** Render service `distillery`, commit `e09c606`, after the dashboard fixes `DISTILLERY_PLAYGROUND_DAILY_CAP_USD` 0.5 -> 0.02 and `DISTILLERY_MODEL_STUDENT` `Qwen/Qwen3-0.6B` -> `Qwen/Qwen3-1.7B` (Auto-Deploy was already off). `scripts/smoke_live.py --no-run`: 18/18 checks passed (cap 0.02, certified N1 adapter `7d6ce813…`, drift guard CERTIFIED, New run locked with 401).
- **Hosted check:** `scripts/w_e2e.py --hosted --prereg-commit 705876d`, normal phase only (no burst, so the public gateway was not rolled back on purpose). C1–C4 all true: 12 of 20 normal questions served by the student (all verified), 8 deferred to the teacher, no 5xx. Student-served requests took roughly 32–69 s client-side on Sandbox CPU. Evidence `docs/proofs/evidence/w_e2e_hosted.json`. Hosted spend is not in the laptop ledger (about $0.01 of the day's $0.02 hosted cap).
