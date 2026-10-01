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
