# Distillery

**Sandbox-verified distillation autopilot on Nebius Token Factory.**

Distillery turns a narrow, repetitive job that a big model does well (today: text-to-SQL over a synthetic SQLite database) into a small fine-tuned model that does it for a fraction of the cost. A planner agent (Nemotron 3 Ultra) designs the task set and reads the failures, a teacher (Nemotron 3 Super) writes training answers, a cheap triager (Nemotron 3 Nano) flags dirty output, and a small student (Qwen3-1.7B, LoRA) is fine-tuned on Token Factory. Every training example and every evaluation is verified by executing the SQL in a sandbox, and a pure-Python gate (no LLM anywhere in it) decides PROMOTE or REJECT from a sealed held-out set.

> **Status: no real end-to-end run has happened yet.** Everything below that says "verified" was verified offline with fakes or live on the inference API only. Every result number in this repo is either absent or labelled. See [STATUS.md](STATUS.md).

## How it works

Each round the agent fine-tunes the student, scores it on a dev set, has the planner cluster the failures into task families, generates targeted data for those families, verifies it by execution, and branches the sandbox so every round is a node in an experiment tree. The held-out set is opened once, for the best-on-dev candidate only.

```mermaid
flowchart TD
    A[Generate schema + questions<br/>template gold SQL] --> B[Cross-check gold:<br/>planner, teacher, template must agree on execution results]
    B --> C[Verifier self-test:<br/>gold accepted, corruptions rejected]
    C --> D[Split and SEAL held-out<br/>train / dev / held-out, family holdout]
    D --> E[Headroom check:<br/>base student on dev, refuse if too easy]
    E --> F[Teacher writes SQL for train tasks<br/>execution-verified rows only]
    F --> G[Fine-tune student LoRA<br/>Token Factory, round r]
    G --> H[Dev eval, round r]
    H --> I{dev target met<br/>or rounds used up?}
    I -- no --> J[Planner clusters dev failures<br/>into families]
    J --> K[Targeted data for those families<br/>teacher + execution check]
    K --> L[Sandbox branch:<br/>new node in the experiment tree]
    L --> G
    I -- yes --> M[Score best-dev candidate ONCE on sealed held-out<br/>base vs student vs teacher]
    M --> N{Gate<br/>pure Python}
    N -- pass --> P[PROMOTE]
    N -- fail --> R[REJECT]
```

Sandbox branching is the experiment tree. Each round forks the previous round's sandbox image, so a node is a real image with a parent and a label, and the UI draws exactly that lineage joined with that round's metrics:

```mermaid
flowchart TD
    root[base image<br/>python + sqlite] --> r1[round 1<br/>train rows, dev score]
    r1 --> r2[round 2<br/>+ targeted families, dev score]
    r2 --> r3[round 3<br/>+ targeted families, dev score]
    r2 -. selected on best DEV score .-> sel((candidate))
```

(Shape of the tree only. The number of rounds and which node is selected come from the run, never from this diagram.)

More detail, matched to the code: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

## How we use Nemotron

| Role | Model | Model ID (live-verified on `GET /v1/models`, 2026-09-30) | Job |
|---|---|---|---|
| Planner | Nemotron 3 Ultra | `nvidia/Nemotron-3-Ultra-550b-a55b` | gold cross-check, clusters dev failures into task families to target |
| Teacher | Nemotron 3 Super | `nvidia/nemotron-3-super-120b-a12b` | writes SQL for training tasks; scored as the ceiling on held-out |
| Triage | Nemotron 3 Nano | `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B` | advisory format check on teacher output; never drops verified rows |
| Student | Qwen3-1.7B + LoRA | `Qwen/Qwen3-1.7B` (fine-tunable; not on the serverless inference list) | the small model we are distilling into |

Roles map to IDs through `DISTILLERY_MODEL_PLANNER|TEACHER|TRIAGE|STUDENT`; nothing else in the code names a model. All three Nemotron models returned valid JSON with `response_format` `json_schema` (`strict: true`) on the first live try, and `usage` includes reasoning tokens.

## Where Token Factory accelerated us

Only things we actually observed. Anything not run yet says so.

- **Structured output (measured, live):** `json_schema` mode worked on Nano, Super and Ultra with the `{name, schema, strict}` wrapper. The planner and triager return parsed, schema-checked objects, which is what lets the orchestrator stay a plain state machine. Reasoning text comes back in a separate field (`reasoning` / `reasoning_content`) so `content` is the final answer only.
- **Fine-tuning API (documented, NOT yet run):** OpenAI-compatible files and jobs endpoints, LoRA on Qwen3-1.7B, checkpoint download, and cancel. We wrote the client against the docs and tested it against fakes. No fine-tune job has been created; duration and price are unmeasured.
- **Sandboxes branching (documented, NOT yet run end to end):** the docs describe each command producing a new image version that children can fork from. We connected far enough to find the project-id requirement and an SDK/docs mismatch (both in [FEEDBACK.md](FEEDBACK.md)) but have not yet run the 40-parallel or branching spike.
- **Friction we hit (real):** Sandboxes need a project id that the getting-started page does not mention; prices are not exposed in the API; the fine-tunable Qwen3-1.7B cannot be served serverless. Details in [FEEDBACK.md](FEEDBACK.md).

## Nebius services used

| Service | Use | Status |
|---|---|---|
| Token Factory inference (Nemotron 3 Ultra, Super, Nano) | planner, teacher, triage | live-verified for structured output and usage |
| Token Factory fine-tuning (LoRA, Qwen3-1.7B) | train the student | client written, not run |
| Sandboxes (Contree) | execute and verify SQL, branch experiments | partially connected, not run end to end |
| Dedicated endpoints / serverless endpoint | serve the fine-tuned student and this app | student serving unresolved (spike S4); app deploy UNVERIFIED |

The whole system is intended to run on Nebius only; the base model weights are never downloaded locally. The small LoRA adapter checkpoint that fine-tuning produces IS downloaded to the run directory (`run_dir/round*/checkpoints`) so its SHA-256 can be verified, and is then loaded inside Nebius for serving.

## Results

No real run has been made. Nothing below is measured.

| Metric | Base Qwen3-1.7B | Fine-tuned student | Teacher (Nemotron 3 Super) |
|---|---|---|---|
| Held-out execution accuracy | TBD: no real run yet | TBD: no real run yet | TBD: no real run yet |
| Student / teacher accuracy ratio, 95% lower bound | n/a | TBD: no real run yet | n/a |
| McNemar p (student vs base) | n/a | TBD: no real run yet | n/a |
| Cost per 1k tasks | TBD: no real run yet | TBD: no real run yet (serving path undecided) | TBD: no real run yet |
| Total spend for the run | TBD: no real run yet | | |

`python -m distillery run ... --dry-run` prints a report that looks like this but uses fake models and fake prices; it is labelled "DRY RUN, numbers are NOT results" everywhere it appears.

## Integrity guarantees

These are enforced in code and covered by offline tests, not by convention.

- **Sealed held-out.** Only `evaluator.py` may call `Store.load_heldout` (an AST test enforces this). The held-out set is scored once, for the candidate with the best dev score, never per round.
- **Family holdout.** Held-out includes task families and table combinations never seen in training (`unseen_family`, `unseen_table_combo`), reported separately from `seen`. Targeted data is filtered against held-out ids, families and gold hashes.
- **The planner never sees held-out** for failure analysis and every later planner call (tested). One exception, disclosed: the label cross-check at the start sends all generated questions, including future held-out ones, to the planner to catch bad gold labels ([DECISIONS.md](DECISIONS.md)).
- **Gate rules.** PROMOTE only if the paired-bootstrap lower bound of student/teacher accuracy is at least 0.85 AND the student beats base by an exact McNemar test at alpha 0.05 (student-only wins must exceed base-only wins). Thresholds are config, not code.
- **Train prompts equal eval prompts.** One message builder (`prompts.py`) is used for training rows and evaluation.
- **Artifact identity.** The adapter that is evaluated is checked by SHA-256 against the one the fine-tune job produced; a mismatch aborts.
- **No LLM in the gate.** `gate.py` is a pure function over booleans; verification is execution match, not a judge.
- **Money.** Paid resources (fine-tune jobs, dedicated endpoints) are cancelled or deleted in `finally` blocks on failure and normal exit, covered by tests for failed jobs, failed generation and failing `close()` (a failed endpoint delete is raised loudly, since it may keep billing); runs preflight against a budget cap; live runs need an admin token and explicit approval.

## Quickstart

```bash
cp .env.example .env          # then fill in secrets; .env is gitignored
python -m venv .venv && .venv/bin/pip install -e ".[dev]" fastapi uvicorn httpx
(cd frontend && npm install)
make dev                      # backend on :8000 + Vite dev server
make test                     # backend and frontend tests, all offline
```

Try the pipeline with no network and no keys:

```bash
python -m distillery run --pack sql --scale tiny --dry-run
```

That uses FAKE models and FAKE prices and takes about 9 s. It proves the plumbing (resume, cancel-on-failure, budget refusal, sealed held-out, artifact check) and says nothing about model quality. Its output is labelled as a dry run.

## Deploy

Everything in this section is **UNVERIFIED**: no image has been built or deployed by us yet.

The `Dockerfile` builds the frontend, installs the backend, and serves the API plus `frontend/dist` with uvicorn on `$PORT` (default 8000) as a non-root user, with a healthcheck on `/api/health`. No secrets are baked in.

```bash
make docker                               # docker build -t distillery:local .
docker run --rm -p 8000:8000 --env-file .env distillery:local
```

Without `NEBIUS_API_KEY` the server starts in `replay-only` mode and serves the labelled sample.

- **Generic container host:** expose the container port, set `PORT` if the host injects one, mount a volume at `/data` (`DISTILLERY_HOME`) so runs survive restarts, and set the variables from `.env.example` in the host's secret store. Health check path: `/api/health`.
- **Nebius Serverless Endpoints (first target):** we have not read a verified deploy procedure for it or run one. The expected shape is: push the image to a registry Nebius can pull from, create the endpoint with container port 8000, set the same environment variables as secrets, and point the health check at `/api/health`. Check each step against Nebius's current docs before relying on it.

## Limitations

- **No real run yet.** No Nebius fine-tune, sandbox branching run or held-out result exists. All results are TBD.
- **Student serving is unresolved.** Qwen3-1.7B is fine-tunable but not on the serverless inference API, and custom weights on dedicated endpoints are "beta, on request" per the docs. Both the base and the student must be scored on the same serving path (spike S4). Live runs refuse to start without one.
- **Toy database.** A synthetic schema with 49 templates in 12 families. Results say little about real-world text-to-SQL.
- **Small held-out.** The full scale holds 300 items and tiny holds 20, so confidence intervals are wide; the gate accounts for this by using a lower bound, but a small win can still be REJECTed.
- **Headroom unproven.** The task set may be too hard or too easy for the real models until they run.
- **Prices are hand-copied** from the console into `DISTILLERY_PRICES_FILE`; fine-tune cost is an explicit estimate you pass in.

## License

MIT. See [LICENSE](LICENSE).
