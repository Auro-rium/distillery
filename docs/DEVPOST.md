# Distillery: Devpost submission (Coding and Agentic Engineering track)

Paste-ready text for the Nebius x NVIDIA Global AI Hackathon. Rules for this file: every number is copied from a recorded artifact and cited; anything not yet measured is a `⟨PENDING:id⟩` marker or says "pending". Do not replace a marker with a number that is not in `docs/proofs/evidence/*.json`.

## Submission checklist

- [ ] Deadline: Fri 2026-10-30 10:00 PT (judging 2026-12-01 to 12-15; the demo must stay up, free and unrestricted, until then).
- [ ] Track: Coding and Agentic Engineering.
- [ ] Working demo URL: `⟨DEMO_URL⟩`
- [ ] Public repository with the license visible (MIT, `LICENSE`): `⟨REPO_URL⟩`
- [ ] README covering Nemotron + Token Factory use and setup: `README.md` (done; re-read before submitting)
- [ ] FEEDBACK (mandatory; also the Most Valuable Feedback prize): `FEEDBACK.md` (done)
- [ ] "Significantly updated" disclosure (lineage note): below (done)
- [ ] Demo video under 3:00, public on YouTube: `⟨VIDEO_URL⟩` (made by the team)
- [ ] Every `⟨PENDING:...⟩` below filled from a recorded artifact, or the sentence left as "pending"
- [ ] Stage 1 fit sentence pasted (below)

## Lineage note (required)

This concept was first explored in an earlier AWS-based prototype (Aug 2026) that never completed an end-to-end run. Distillery is a from-scratch rebuild started Sep 29, 2026 on Nebius Token Factory; no code was carried over.

## Stage 1 fit

Distillery is an autonomous agent that writes SQL, runs and tests it in Nebius Sandboxes, fine-tunes a model on Token Factory, gates it and certifies it, and it uses Token Factory and NVIDIA Nemotron 3 at runtime (Super answers every deferred gateway request and Nano writes the Acme analyst's summaries; Ultra plans the agent loop), so it fits the Coding and Agentic Engineering track and the required stack.

## Tagline

Change one base URL and Distillery swaps your big model for a certified specialist that hands back anything it can't guarantee.

## Inspiration

Teams pay a large model to do the same narrow job millions of times. Distilling that job into a small model is well understood, but trusting the result is not: most distillation demos stop at "it scored well on a set we wrote". We wanted an agent that does the whole loop and, more importantly, one that is allowed to say no, both when deciding whether to ship a model and, later, on every single request.

## What it does

Given a narrow task (today: text-to-SQL over a synthetic SQLite database), the Distillery agent runs the loop by itself:

1. **Writes and tests code.** Gold SQL comes from parameterized templates and is executed in a Nebius Sandbox; the Nemotron 3 Super teacher writes SQL for the training tasks, and a row is kept only if it executes to the gold result. Nemotron 3 Ultra clusters the dev failures into task families to target next; Nemotron 3 Nano triages formatting.
2. **Fine-tunes and gates.** It fine-tunes `Qwen/Qwen3-1.7B` (LoRA) on Token Factory, branches the sandbox so every round is a node in an experiment tree, opens a sealed held-out set once, and a pure-Python gate (no LLM) decides PROMOTE or REJECT: the student/teacher accuracy ratio must have a lower bound of at least 0.85 and the student must beat the base model on an exact McNemar test.
3. **Certifies a router.** The student does not pass the gate on naturally phrased questions (see results), so the product is not "swap the model". A router scores each request from three signals (does the student's SQL run, the student's mean token log-probability, similarity of the question to the training questions via `Qwen/Qwen3-Embedding-8B`). A Learn-then-Test threshold, certified on a sealed calibration split, decides whether the student may answer.
4. **Delivers it through a gateway.** `POST /v1/chat/completions` is OpenAI-compatible. An unchanged agent (our Acme analyst, plain `openai` SDK) points at the gateway by changing its base URL; the gateway recognises the certified skill, serves the student where certified and hands everything else back to the teacher. Each response says who answered and why, and savings are computed server-side from the call log.
5. **Guards itself.** A drift guard watches the deferral rate and audits student answers against the teacher; if traffic stops looking like the calibration mix it rolls the gateway back to the teacher by itself and queues (does not run) a re-distill.
6. **Shows its receipts.** The `/receipts` page lists every claim with its value, a commit-pinned artifact, its pre-registration commit, the sealed-set and adapter hashes and the ledger total. Claims that have not been scored are shown as pending, with no number.

## Results

Student is `Qwen/Qwen3-1.7B` + LoRA, teacher is Nemotron 3 Super. All values are from the cited file.

| ID | Result | Status | Source |
|---|---|---|---|
| P4 | Sealed templated held-out, n=300: base 0.257, student 0.923, teacher 0.900; ratio lower bound 0.993 (bar 0.85); McNemar 202 vs 2; run cost $4.26. Stress set (n=100, not a gate input): student 0.16 | PROMOTE, in-distribution only | `docs/proofs/evidence/p4_run_summary.json` |
| A5 | Fresh run under 8 injected faults: 4 supervisor restarts, 1 fine-tune job billed, verdict PROMOTE (lower bound 1.022), $4.1918 | PASS (5/5 criteria) | `docs/proofs/evidence/a5_chaos.json` |
| B2 | Student 0.9233 vs a fair 8-shot Super teacher 0.9333; ratio lower bound 0.957 (a claim needs > 1.0) | no beat-the-teacher claim | `docs/proofs/evidence/b2_fair_teacher_s0_s1.json` |
| Gate B | 86 natural, agent-authored questions: base 0.384, student 0.407; ratio lower bound 0.306; McNemar p 0.84 | REJECT | `docs/proofs/evidence/gateb_agent_authored.json` |
| N1 | Natural-phrasing training, fresh agent-authored set (n=98): base 0.296, student 0.571, teacher 0.959; ratio lower bound 0.495; McNemar p 4.6e-07 | REJECT (above base, below the bar) | `docs/proofs/evidence/n1_natural_round.json` |
| W-C1 | Certified router on the sealed test split (n=490): student served 240 (coverage 0.490, minimum 0.30) with 10 wrong (selective error 0.042, 95% upper bound 0.070, alpha 0.08); system accuracy 0.951 vs teacher-only 0.935 | PASS | `docs/proofs/evidence/router_eval_test.json` |
| W-C2 | Natural + stress test items: 115 of 117 wrong student answers deferred (0.983, lower 95% bound 0.947, minimum 0.80); 113 of 123 right answers deferred too (0.919), so the router mostly defers this traffic | PASS (small separation) | `docs/proofs/evidence/router_eval_test.json` |
| W-D | Burst of 30 off-distribution questions (new `credit_grants` table) after 200 test requests: rollback on burst request 13 (limit 25), 0 false alarms, 0 wrong student answers during the burst; each of the 30 distinct burst questions is deferred. 295 of 300 seeds pass | PASS | `docs/proofs/evidence/drift_burst.json`, `docs/proofs/evidence/h3_drift_report.json`, `docs/proofs/evidence/h3_burst_counts.json` |
| W-$ | Student cost per 1k requests on an AI Cloud L40S vLLM endpoint | `⟨PENDING:W-$⟩` not yet measured | `h1_endpoint.json` (not yet written) |
| W-G | An unchanged agent works against the gateway after one base-URL change | verified as a test | `tests/test_acme_wg.py` |
| W-E2E | Live local server on Token Factory + Nebius Sandbox, pinned N1 checkpoint: 12 of 20 normal questions served by the student (all 12 verified; 12 of the 14 templated ones), 8 deferred by the router (all 6 of the website's example questions); 30-question burst: rollback on burst request 14 (limit 25), 0 student answers during or after it; student answers took a median 39 s (max 73 s) on sandbox CPU; spend $0.0331 | PASS | `docs/proofs/evidence/w_e2e.json` |

Spend to date is $15.2583 of the 19.50 cap, plus $1.50 reserved for hosting (global ledger, `STATUS.md`; sandbox is recorded at $0, so it is a lower bound).

## How we built it

- **Backend:** Python 3.12+, a staged, resumable orchestrator; a SQLite store with content-addressed artifacts; a FastAPI server (SSE progress, gateway, Acme API, receipts); a React/TypeScript frontend (Mission, Evidence, Acme analyst, Receipts, run screens).
- **Nemotron (NVIDIA):** Ultra `nvidia/Nemotron-3-Ultra-550b-a55b` plans, Super `nvidia/nemotron-3-super-120b-a12b` teaches and answers deferred requests at runtime, Nano `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B` triages and writes one-line summaries.
- **Token Factory:** inference and `json_schema` structured output for the agents; the fine-tuning API (LoRA on `Qwen/Qwen3-1.7B`); Sandboxes to execute, test and branch; embeddings (`Qwen/Qwen3-Embedding-8B`) for the router signal. **Nebius AI Cloud:** a time-boxed L40S vLLM + LoRA endpoint harness to price the student, always deleted in `finally` (offline-tested; not yet run live).
- **Integrity in code, not convention:** only `evaluator.py` can read sealed sets (AST test); train and eval prompts come from one builder; the evaluated adapter is SHA-256-checked against the trained one; the router refuses to certify without a pre-registration commit that is an ancestor of HEAD; paid jobs are cancelled in `finally`; a cross-process ledger enforces the spend cap; no LLM sits in the gate.

## Challenges

- **Our own student failed the honest test.** It PROMOTEd on templated questions and was REJECTed on natural phrasing twice (Gate B, then N1 after training on natural data). We kept both failures visible and built the certified router around them instead of weakening the gate.
- **The first fine-tunes learned nothing**: provider defaults trained for 3 to 9 steps with no warning. We found it from the job objects, pinned the hyperparameters and proved the pipeline can fit data (P1.11: adapter 64/64 on its training prompts) before spending on P4 (`docs/proofs/evidence/p1_11_overfit.json`).
- **A fine-tunable small model is not servable on the serverless API**, and custom weights on Dedicated endpoints are beta and on request. We served on Sandbox CPU and built the AI Cloud harness to price a GPU path (see `FEEDBACK.md`).
- **Keeping an unattended run alive** on a free host: supervisor restarts, outage-aware retries, job adoption after a kill (A5).
- **Not fooling ourselves:** the natural-question sets are agent-authored (a Claude subagent), not human-written, and their teacher accuracy is inflated by construction; we say so wherever the numbers appear.

## Accomplishments

- A complete agent loop that ran live end to end for $4.26 (P4) and recovered from 8 injected faults with one human action, the start (A5).
- A gate that rejects: two of our own results are REJECT and are published as such.
- A certified router that keeps the student's served error within its bound on a sealed test split (W-C1, W-C2), a drift guard that rolls back on a burst (W-D), and the whole path run live end to end through the gateway for $0.0331 (W-E2E).
- An unchanged agent works against the gateway after one base-URL change (W-G, a test).
- Detailed feedback on Token Factory, Sandboxes, fine-tuning, embeddings and AI Cloud in `FEEDBACK.md`.

## What we learned

- Training-set phrasing decides everything: a student that matches the teacher on templated questions is indistinguishable from base (Gate B: 0.407 vs 0.384) on natural ones, and natural training data helps (N1: 0.571 vs base 0.296) but not enough to replace the teacher.
- So the useful product is a router that knows when not to answer, with a certificate, not a smaller model on faith.
- Nano spends far more output tokens than input on short answers (P4: 684 calls, 126,538 input and 753,151 output tokens, `p4_run_summary.json`), so "cheap" is not "few tokens".
- Pre-registering thresholds and sealing sets before scoring changed what we were willing to publish.

## What's next

Measure the student's serving cost on a GPU endpoint (W-$; the student currently runs on sandbox CPU, tens of seconds per answer); widen the certificate beyond templated traffic; more task packs and a human-written natural question set.

## Built with

Nebius Token Factory (inference, structured output, fine-tuning, Sandboxes, embeddings), Nebius AI Cloud, NVIDIA Nemotron 3 Ultra, Super and Nano, `Qwen/Qwen3-1.7B`, `Qwen/Qwen3-Embedding-8B`, Python, FastAPI, SQLite, React, TypeScript, Vite, Docker, Render.

## Links

- Live demo: `⟨DEMO_URL⟩`
- Repository: `⟨REPO_URL⟩`
- Demo video (under 3:00): `⟨VIDEO_URL⟩`
- Recorded run bundle: `replay/sql-gated-20261004T155134Z-66ff347c/`

## Testing instructions for judges

No account or key is needed to try the demo.

1. Open `⟨DEMO_URL⟩`. The landing page is **Mission** (how it works, the proof ladder, stored runs); **Evidence** opens the featured recorded run's report. Recorded runs carry the banner "Recorded run · <time> · real Token Factory jobs"; a dry-run sample, if shown, carries "DRY RUN — fake models, numbers are NOT results". Check which banner you see.
2. Open **Acme analyst** (`/playground`) and ask a question. It makes real gateway calls, capped per address per hour and per day. The answer card says whether the student or the teacher answered and why. Until a certificate is loaded it says "Not certified yet: all traffic goes to the teacher".
3. Open **Receipts** (`/receipts`): every claim with its artifact link and hashes; pending claims show no number.
4. Run locally without keys (fake models, plumbing only):
   ```bash
   git clone ⟨REPO_URL⟩ && cd distillery
   uv sync --extra dev
   uv run python -m distillery run --pack sql --scale tiny --dry-run
   uv run python -m pytest -q -p no:cacheprovider
   ```
5. Live runs need `NEBIUS_API_KEY`, `NEBIUS_AI_PROJECT`, a prices file and the admin token; they are locked on the public demo. The free host sleeps when idle, so the first request can take about a minute.

## Track fit: Coding and Agentic Engineering

An agent plans, writes SQL, runs and tests it in sandboxes, fine-tunes a model, analyses its own failures, gates and certifies the result, and keeps watching it in production, with a deterministic gate and a certified threshold as the safety boundary.
