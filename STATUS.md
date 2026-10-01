# STATUS

**Date:** 2026-10-01 · **Phase:** diagnostics done, benchmark rebuilt and pre-registered, **no paid run until the cost estimate is approved** · **Result claims:** none (two smoke runs, both REJECT, both undertrained)

## Where each step of the plan stands
1. **Free diagnostics: done** (evidence below). 2. **Pre-registration: done** (DECISIONS.md, committed before any new data). 3. **Gold from templates: done** (code, audit, tests). 4. **Student config: done in code** (pinned hyperparameters, 50-step floor, base/student/teacher scored concurrently). 5. **Cost estimate: blocked on you** (console prices). 6. **Real backend: partly done** (see bottom); host and `vercel.json` decisions are yours.

## Step 1 diagnostics (real data, 2026-09-30/10-01)
- **(a) prices:** LLM prices found via the API (see Step 5). Billed spend, the fine-tune price and the sandbox price: console only, not done.
- **(b) student on its OWN training rows (sandbox, run on the two real mini-run adapters): 16.2% (round 1, 117 rows) and 17.4% (round 2, 138 rows); base on the same rows 8.5% / 8.0%.** Threshold was "low if < 50%": **tripped.**
- **(c) identical-string rate student vs base (extracted SQL): 0.9% to 3.3%** on train/dev/held-out. Threshold "high if > 90%": not tripped. The adapter is applied and changes outputs.
- **(d) loss curves (real job objects):** tiny 3 steps (train 0.771 to 0.751), mini r1 6 steps (0.674 to 0.609), mini r2 9 steps (0.687 to 0.566). Loss fell, from only 3 to 9 optimizer steps.
- **(e) prompt render: byte-identical.** Training row and rebuilt eval prompt are identical (system 215 chars, user 3886 chars, same text). Rendered through the checkpoint's own chat template with jinja2: `/no_think` is in both, and the thinking-off eval prompt ends with `<think>\n\n</think>\n\n`, the same block the training render puts before the answer. No template mismatch.
- **(f)** the same 3 of 30 dev tasks were solved after round 1 and round 2.
- **Reading:** root cause is the provider defaults the pipeline silently accepted (batch 8, lr 1e-5, lora_r 8, alpha 8, packing on): 3 to 9 optimizer steps. That is consistent with every number above (student ~2x base on its own training data, but far from fitting it). It is **not proven** to be the whole explanation; the properly configured run is the test. Fixed with explicit pinned hyperparameters, packing off, a refusal below 50 planned steps, and a `finetune` report section.

## Benchmark as now built (offline, tested; fake-model dry run at full scale passes)
- Gold = template SQL, executed in the sandbox; no model writes or vets it. The Ultra/Super agreement filter is gone; Ultra is only used for failure analysis.
- Gate set: 300, in-distribution, every family also in train, questions never in train. **Stress set: 100 from reserved families `date_math`, `set_ops`** (rule `Random(777).sample`, verified), sealed separately, scored outside the gate.
- Scale: train 1800 / dev 150 / gate 300 / stress 100, oversample 1.1.
- **Measured caveat:** 1800 training tasks contain only 141 distinct question skeletons and **97.3% of gate questions share a skeleton with a training question** (same wording, other literals). The gate measures generalisation across literals within known question shapes; only the stress set tests unseen structure.
- Template audit (89 templates, 300 draws each): found and fixed one real defect (a phrasing that hid a country filter the gold applied); now 0 gold errors, 0 ORDER BY ties, 0 LIMIT-without-ORDER, 0 ambiguous questions. It runs in CI.
- Teacher yield risk: a train row exists only if the teacher's SQL matches gold, so rows = 1800 x teacher accuracy; if under 1500 the pre-registered range is missed and that is reported.

## Step 5: pre-flight estimate (NOT approved, nothing spent)
**LLM prices are now real**, read from the API itself (`GET /v1/models?verbose=true`, 2026-10-01): Nano $0.06 in / $0.24 out, Super $0.30 / $0.90, Ultra $1.00 / $3.00 per Mtok. My earlier ceilings were 3 to 10x too high. Re-priced, the two smoke runs consumed **$1.46 of LLM calls** (not about $4.2).
`scripts/estimate_run_cost.py --scale full --rounds 1` (measured token averages from the real mini run): teacher on 1800 train rows **$1.13**, teacher on 400 gate/stress items **$0.25**, planner $0 (no analysis in a 1-round run), subtotal **$1.38**.
**Still unpriced, and not findable without the console:** the SFT/LoRA fine-tune price (about 5.18M trained tokens for the run: 1800 rows x 960 tokens/row, measured by the three real jobs at 963/960/959, x 3 epochs) and the Sandbox price (1,100 generations x 7.8 s = 8,580 s). I searched the docs, the OpenAPI spec, the model endpoints and the pricing site: no usage or billing endpoint exists, and the only fine-tune price in the API is for a different product (speculative-decoder training). So no total is shown.
**What fits, as arithmetic not a quote:** working budget $9.50 ($19.50 cap minus the $10 demo reserve), minus $1.46 spent on LLM calls, minus the $1.38 above. That leaves about $6.66 for the earlier fine-tune jobs (0.85M tokens) plus the new one (5.18M tokens) plus sandbox, so if sandbox is free everything fits only when the fine-tune price is under about **$1.10 per Mtok trained tokens**. The console price decides it.

## What the earlier smoke runs do and do not show
Both are plumbing proofs, not results: `sql-tiny-live1` (n=20: base 0.15, student 0.15, teacher 0.95, REJECT) and `sql-mini-live1` (n=60, 2 rounds: base 0.033, student 0.05, teacher 1.0, REJECT, real sandbox-branch tree). Estimated (not billed) spend about $3.07 and $6.46 at ceiling prices.

## Step 6 backend / deploy
**Live at https://distillery.onrender.com** (one Render Docker service serves the API and the built frontend from one origin; Vercel and the static snapshot are removed from the repo). Checked live: `/api/health` ok, both recorded runs listed, a POST for a new run without a token is refused (401), frontend serves. Hackathon credit is Token Factory only, so Nebius AI Cloud hosting was not used (see docs/DEPLOY.md).
Code done: `/api/runs` lists recorded bundles (`replay/`), relative `spot_check_file`, playground teacher plus opt-in sandbox base/student (unverified live), New run locked (no admin token is set on the service; live runs start from the CLI on a trusted machine).
Verified live on 2026-10-01 after setting the env vars through the Render API: mode `live`, models configured, playground teacher `ok` with a $0.50 daily cap, one real question answered (teacher SQL, $0.0005), base/student shown as `disabled`, both recorded runs listed, New run refused without a token (401). Render CLI is logged in on this machine (service `distillery`, `srv-dauuokk1nsns73fqcv70`, auto-deploy on push to `main`). The old Vercel project `frontend` still exists in your Vercel account but is no longer part of the build; delete it there if you want.

## Config
- Student `Qwen/Qwen3-0.6B` (LoRA) on Nebius Sandbox CPU. Budget: $19.50 project cap, $10 reserved for the demo. Live home (spend ledger): `.distillery/live` (gitignored). Prices: `.distillery/prices.json` and `deploy/prices.json` (LLM prices from the API listing; no fine-tune or sandbox price yet).

## Blockers (you)
Billed spend and the fine-tune and sandbox prices from the console; approval of the step-5 estimate; demo video; Devpost submit; make the repo public.
