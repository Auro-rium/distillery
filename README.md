# Distillery

**Change one base URL and Distillery swaps your big model for a certified specialist that hands back anything it can't guarantee.**

Distillery is an autonomous agent that distils a narrow, repetitive job (today: text-to-SQL over a synthetic SQLite database) from a big Nemotron model into a small fine-tuned student (`Qwen/Qwen3-1.7B`, LoRA), then ships it behind an OpenAI-compatible gateway. The agent writes the SQL, runs and tests it in Nebius Sandboxes, fine-tunes on Token Factory, gates the result with a pure-Python test, and certifies a router that serves the student only where it can bound the error rate; everything else goes back to the teacher. A drift guard rolls the gateway back to the teacher on its own.

Track: Coding and Agentic Engineering (Nebius x NVIDIA Global AI Hackathon). Demo: `⟨DEMO_URL⟩`. Video: `⟨VIDEO_URL⟩`.

> **Honest status.** The agent loop, the gate, the autonomy run, the gateway and the receipts page are built and measured. The student PROMOTEd on templated questions and **fails on naturally phrased ones** (Gate B and N1 are both REJECT, below). The certified router exists *because* the student fails on natural phrasing: it serves the student only where it is certified and defers the rest. Its test-split results, the drift burst, the serving cost and the end-to-end run are **pre-registered and not yet measured**; they appear as `pending` and nowhere as achieved.

## What it does

```
generate tasks -> execute gold in Sandboxes -> seal gate/stress sets -> teacher data (execution-verified)
  -> fine-tune student (Token Factory) -> dev eval -> failure analysis -> next round (sandbox branch)
  -> sealed held-out, scored once -> GATE (pure Python) -> PROMOTE / REJECT
  -> certified router (Learn-then-Test threshold on a sealed calibration split)
  -> gateway POST /v1/chat/completions: student if certified, else teacher
  -> drift guard CERTIFIED -> ROLLBACK -> TEACHER  -> receipts (every claim, value, hash, commit)
```

- **Agent loop** (`orchestrator.py`, optional bounded Ultra controller `controller.py`, off by default): planner clusters dev failures into families, teacher writes SQL, only rows that execute correctly are kept, each round branches the sandbox into an experiment tree. Supervised, resumable, outage-aware (`server/worker.py`, `driver.py`).
- **Gate** (`gate.py`, `stats.py`): PROMOTE only if the paired-bootstrap lower bound of student/teacher accuracy is at least 0.85 and the student beats base by exact McNemar (alpha 0.05). No LLM in the gate. The held-out set is readable only by `evaluator.py` (AST-tested).
- **Certified router** (`router.py`, `router_data.py`, `router_eval.py`, `stats.certify`): a logistic score over three signals (execution OK, student mean log-prob, nearest-neighbour similarity to training questions via `Qwen/Qwen3-Embedding-8B`), with the serve-the-student threshold certified by Learn-then-Test on sealed CAL/TEST splits whose hashes and pre-registration commit are checked. Planned parameters: alpha 0.08, delta 0.05 ([STATUS.md](STATUS.md)).
- **Gateway** (`server/gateway.py`): the certified `acme-sql` skill is recognised by the sha256 of its system prompt; `distillery` extension in the response says who answered; every call is a `gateway_calls` row; savings are computed server-side. With no loaded certificate, every request goes to the teacher.
- **Acme analyst** (`acme.py`, `server/acme_api.py`, `scripts/acme_agent.py`): a plain `openai`-SDK agent. Pointing it at Token Factory or at the gateway differs only in `--base-url`.
- **Drift guard** (`drift.py`, `drift_sim.py`): deferral-rate and teacher-audit detectors; rollback is automatic, the way back is a new certificate (`recertify`); a re-distill is queued, never run automatically.
- **Money** (`global_ledger.py`, `budget.py`): one cross-process ledger, caps, preflight, paid jobs cancelled in `finally`.
- **Receipts** (`scripts/build_evidence.py`, `/receipts`, `GET /api/evidence`): one row per claim with value, commit-pinned artifact, pre-registration commit, sealed-set and adapter hashes, ledger total. Pending claims show no number.

## Results

Every number is copied from the file in the Source column. Student is `Qwen/Qwen3-1.7B` + LoRA; teacher is Nemotron 3 Super. Evidence directory: [docs/proofs/evidence](docs/proofs/evidence).

| ID | Claim | Result | Status | Source |
|---|---|---|---|---|
| P4 | Student promoted on a sealed templated held-out (n=300) | base 0.257, student 0.923, teacher 0.900; ratio lower bound 0.993 (bar 0.85); McNemar 202 vs 2; run cost $4.26. Stress set (n=100, not a gate input): student 0.16 | **PROMOTE**, in-distribution only (skeleton overlap with train 0.9467) | [p4_run_summary.json](docs/proofs/evidence/p4_run_summary.json) |
| A5 | Autonomy under injected faults, no human action | 8 injected faults, 4 supervisor restarts, 1 fine-tune job billed, verdict PROMOTE (lower bound 1.022), spend $4.1918 | **PASS** (5/5 criteria) | [a5_chaos.json](docs/proofs/evidence/a5_chaos.json) |
| B2 | Student beats a fair (8-shot Super) teacher | student 0.9233 vs 0.9333; ratio lower bound 0.957 (claim needs > 1.0) | **no claim made** | [b2_fair_teacher_s0_s1.json](docs/proofs/evidence/b2_fair_teacher_s0_s1.json) |
| Gate B | Student holds up on 86 natural, agent-authored questions | base 0.384, student 0.407; ratio lower bound 0.306; McNemar p 0.84 (teacher 0.988 is inflated: it drafted the gold) | **REJECT** | [gateb_agent_authored.json](docs/proofs/evidence/gateb_agent_authored.json) |
| N1 | Training on natural phrasing closes the gap (fresh agent-authored set, n=98) | base 0.296, student 0.571, teacher 0.959; ratio lower bound 0.495 < 0.85; McNemar p 4.6e-07 | **REJECT** (better than base, short of the bar) | [n1_natural_round.json](docs/proofs/evidence/n1_natural_round.json) |
| W-C1 | Certified router: selective error within alpha at minimum coverage, system accuracy not below teacher-only | `⟨PENDING:W-C1⟩` | pending: pre-registered, not yet scored | `router_eval_test.json` (does not exist yet) |
| W-C2 | Router defers most wrong student answers, few right ones | `⟨PENDING:W-C2⟩` | pending: pre-registered, not yet scored | `router_eval_test.json` (does not exist yet) |
| W-D | Drift burst trips the guard and rolls back to the teacher | `⟨PENDING:W-D⟩` | pending: not yet measured | `drift_burst.json` (does not exist yet) |
| W-$ | Student cost per 1k requests on a time-boxed AI Cloud L40S vLLM endpoint | `⟨PENDING:W-$⟩` | pending: not yet measured | `h1_endpoint.json` (does not exist yet) |
| W-G | An unchanged agent works against the gateway after one base-URL change | same agent code, base URL swapped; test passes | verified as a test | [tests/test_acme_wg.py](tests/test_acme_wg.py) |
| W-E2E | Live end to end: certified router, student, deferral, drift guard | `⟨PENDING:W-E2E⟩` | pending: not yet measured | `w_e2e.json` (does not exist yet) |

Spend to date (store ledgers; sandbox is recorded at $0, so this is a lower bound): $14.1952 of the 19.50 cap ([STATUS.md](STATUS.md)). Claims, procedures and pre-registrations: [DECISIONS.md](DECISIONS.md); the 25-claim audit: [GROUND_TRUTH.md](GROUND_TRUTH.md). Cost per 1k tasks for the student is not reported until W-$ exists.

## Nebius and NVIDIA usage

| Role | Model id | Job |
|---|---|---|
| Planner | `nvidia/Nemotron-3-Ultra-550b-a55b` (Nemotron 3 Ultra) | clusters dev failures into task families; optional bounded controller |
| Teacher | `nvidia/nemotron-3-super-120b-a12b` (Nemotron 3 Super) | writes training SQL; scored as the ceiling; at runtime it answers every request the router defers |
| Triage | `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B` (Nemotron 3 Nano) | format triage; one-line summaries in the Acme analyst; the only public passthrough model |
| Student | `Qwen/Qwen3-1.7B` + LoRA | the small model we distil into |
| Router embeddings | `Qwen/Qwen3-Embedding-8B` | `nn_sim` signal (Token Factory embeddings) |

Ids are set in [render.yaml](render.yaml) and read through `DISTILLERY_MODEL_PLANNER|TEACHER|TRIAGE|STUDENT`; prices in [deploy/prices.json](deploy/prices.json).

- **Token Factory inference and structured output:** all three Nemotron models, `json_schema` strict mode.
- **Token Factory fine-tuning:** LoRA on Qwen3-1.7B (the P4, A5 and N1 runs); checkpoints SHA-256-checked.
- **Token Factory Sandboxes:** every gold, training row and evaluation is executed there; branching is the experiment tree; base and student run on sandbox CPU today.
- **Token Factory embeddings:** `Qwen/Qwen3-Embedding-8B`.
- **Nebius AI Cloud (H1):** a time-boxed L40S vLLM + LoRA endpoint harness (`aicloud_endpoint.py`, `h1_measure.py`, `scripts/h1_endpoint.py`), always deleted in `finally`. Offline-tested; not yet run live (W-$ pending).

Friction and findings: [FEEDBACK.md](FEEDBACK.md).

## Quickstart

```bash
uv sync --extra dev
cp .env.example .env     # fill in; .env is gitignored
(cd frontend && npm install)
```

Variables in `.env.example` (names only): `NEBIUS_API_KEY`, `NEBIUS_BASE_URL`, `NEBIUS_AI_PROJECT`, `DISTILLERY_SANDBOX_URL`, `DISTILLERY_SANDBOX_IMAGE`, `DISTILLERY_ADMIN_TOKEN`, `DISTILLERY_RUN_CAP_USD`, `DISTILLERY_PROJECT_CAP_USD`, `DISTILLERY_PLAYGROUND_DAILY_CAP_USD`, `DISTILLERY_PRICES_FILE`, `DISTILLERY_MODEL_PLANNER`, `DISTILLERY_MODEL_TEACHER`, `DISTILLERY_MODEL_TRIAGE`, `DISTILLERY_MODEL_STUDENT`, `DISTILLERY_HOME`, `DISTILLERY_REPLAY_DIR`, `DISTILLERY_ALLOWED_ORIGINS`, `DISTILLERY_SEED`. Gateway, router and drift settings are in [docs/API_CONTRACT.md](docs/API_CONTRACT.md).

Free, offline (fake models, fake prices, labelled "DRY RUN"; proves plumbing, not quality):

```bash
uv run python -m distillery run --pack sql --scale tiny --dry-run
uv run python -m distillery serve --port 8000        # API + replay; with `make dev`, the Vite UI too
uv run python -m pytest -q -p no:cacheprovider
uv run python -m distillery ledger status             # spend ledger
uv run python scripts/build_evidence.py --out /tmp/evidence.json   # rebuild the receipts, free
```

Paid or certified steps (all need `--i-approve-spend` / an admin token; read `--help` first): `run --scale gated`, `router build-splits | signals | fit | certify | evaluate`, `scripts/h1_endpoint.py` (`--plan-only` is free), `scripts/acme_agent.py --base-url <url>`. Container: `make docker`, then `docker run --rm -p 8000:8000 --env-file .env distillery:local` (without a key it starts in replay-only mode). Hosting notes: [docs/DEPLOY.md](docs/DEPLOY.md).

## For judges

- **Demo:** `⟨DEMO_URL⟩` (free, no sign-in; the free host sleeps when idle, so the first request can take about a minute).
- **Click, in order:** **Mission** (landing page: how it works, the proof ladder, stored runs), then **Evidence** (the featured recorded run's report: gate statistics, held-out and stress charts; the experiment tree is linked from the run); **Acme analyst** (`/playground`: ask a question, see who answered, the confidence against the threshold, SQL, rows, cost, the savings counter, the drift panel); **Receipts** (`/receipts`: every claim, its value, artifact link, hashes; pending claims show no number).
- **Live vs replay:** stored runs are recorded real runs (banner "Recorded run · <time> · real Token Factory jobs"); a dry-run sample carries the banner "DRY RUN — fake models, numbers are NOT results". The Acme analyst and gateway are live requests, capped per address and per day. Until a certificate is loaded (W-C1 pending) the gateway says "Not certified yet: all traffic goes to the teacher". New runs are locked on the public demo.
- **Public repo and license:** MIT, [LICENSE](LICENSE). Repo: `⟨REPO_URL⟩`.
- **Lineage:** an earlier AWS-based prototype (Aug 2026) never completed an end-to-end run; Distillery is a from-scratch rebuild started Sep 29, 2026 on Nebius, and no code was carried over.

## Limitations

- **The win is in-distribution.** P4 PROMOTEd on templated questions (skeleton overlap with train 0.9467, [p4_run_summary.json](docs/proofs/evidence/p4_run_summary.json)); Gate B and N1 are REJECT. The N1 and Gate B sets are agent-authored (a Claude subagent), not human-written, and their teacher accuracy is inflated by construction.
- **Toy database:** a synthetic schema with templated gold. Results say little about real-world text-to-SQL. One task pack ships (`sql`); a `toolcall` pack has only a dry-run record ([c3_toolcall_dry_run.json](docs/proofs/evidence/c3_toolcall_dry_run.json)).
- **Student serving is slow today:** sandbox CPU, with a per-day student call cap in the gateway; sandbox is unpriced, so no student cost per 1k is claimed until W-$.
- **One run, one seed** for each result; small sealed sets (n=300, 98, 86) give wide intervals.
- **Hosted on a free Render plan** (sleeps, ephemeral disk; see [docs/DEPLOY.md](docs/DEPLOY.md)); the autonomy claim is bounded to one instance lifetime.
- **Streaming and tool-calling are not supported** on the gateway (`400`).

## License

MIT. See [LICENSE](LICENSE).
