# Architecture

Describes what exists under `src/distillery` today. Statuses: "offline-tested" means unit-tested with fakes; "UNVERIFIED" means never run against the real service.

## Components

```mermaid
flowchart LR
    subgraph Client
        UI[frontend/ React SPA]
        CLI[python -m distillery<br/>cli.py]
    end
    subgraph Server["server/ (FastAPI)"]
        APP[app.py<br/>routes per docs/API_CONTRACT.md]
        WK[worker.py<br/>one background worker,<br/>runs the CLI in a subprocess]
        RD[reader.py<br/>run status from the store]
        RP[replay.py<br/>replay bundles / labelled sample]
        PG[playground.py<br/>per-IP + daily-$ limited]
        SSE[sse.py + limits.py]
    end
    subgraph Core
        ORCH[orchestrator.py<br/>staged pipeline]
        PACK[taskpacks/sql<br/>schema, questions, verifier, executor]
        PR[prompts.py<br/>single message builder]
        EV[evaluator.py<br/>only reader of held-out]
        GATE[gate.py + stats.py<br/>pure functions]
        ST[store.py<br/>SQLite + content-addressed artifacts]
        BUD[budget.py<br/>ledger, paid_resource]
        CFG[config.py<br/>env, prices, thresholds]
    end
    subgraph Nebius
        LLM[llm.py LLMClient<br/>Token Factory inference]
        FT[finetune.py<br/>files, jobs, checkpoints]
        SB[sandbox.py + sandbox_executor.py<br/>Sandboxes, branching]
        SV[student.py, sandbox_student.py,<br/>endpoint_student.py<br/>student serving options]
    end
    UI -->|/api| APP
    CLI --> ORCH
    APP --> WK --> CLI
    APP --> RD --> ST
    APP --> RP
    APP --> PG --> LLM
    ORCH --> PACK
    ORCH --> PR
    ORCH --> LLM
    ORCH --> FT
    ORCH --> SB
    ORCH --> SV
    ORCH --> EV --> GATE
    ORCH --> ST
    ORCH --> BUD
    EV --> ST
    PACK --> SB
    CFG -.-> ORCH
    CFG -.-> APP
```

Notes tied to the code:

- The API never runs the pipeline in-process. `worker.py` starts `python -m distillery run` as a subprocess (one at a time; a second live run gets 409) and streams its output to SSE.
- `pipeline_fakes.py` holds the deterministic fakes used by `--dry-run` and tests; dry runs use a separate store root and a `dry-` run id prefix.
- Serving the student has three candidate paths (spike S4, all UNVERIFIED): sandbox CPU with peft (`sandbox_student.py`), dedicated endpoint (`endpoint_student.py`), or a serverless job (not found in docs). `student.py` holds the interface.
- `LLMClient` routes by role (planner, teacher, triage, student), classifies retries, handles `json_schema` structured output, splits reasoning from content, and logs usage into the budget ledger.

## Pipeline stages

Stage names are the ones in `orchestrator.py`; `r` is the round number. Each stage is cached in the store by input hash, so a killed run resumes.

```mermaid
flowchart TD
    schema --> questions --> gold_crosscheck --> verifier_selftest
    gold_crosscheck --> split
    split --> headroom
    split --> teacher_data
    headroom --> teacher_data
    teacher_data --> finetune_r1
    finetune_r1 --> dev_eval_r1
    dev_eval_r1 --> analysis_r1 --> targeted_r1 --> sandbox_branch_r1
    sandbox_branch_r1 --> finetune_r2
    finetune_r2 --> dev_eval_r2
    dev_eval_r2 --> final_eval
    finetune_r2 --> final_eval
    split --> final_eval
    final_eval --> report[report.json + gate decision]
```

- `split` seals the held-out set; the store refuses to hand it out except to `evaluator.py`.
- `headroom` scores the base student on dev and raises `TaskTooEasyError` if accuracy is at or above the threshold (default 0.80).
- `final_eval` evaluates only the round with the best DEV accuracy, verifies the adapter SHA-256 against the fine-tune job's, scores base, student and teacher on the same held-out items in the same order, and calls the gate.
- Later rounds repeat `analysis`, `targeted`, `sandbox_branch` and `finetune` up to `max_rounds` (default 3).

## Sandbox branching tree

```mermaid
flowchart TD
    base[base image: python3 + sqlite] --> seed[image with demo DB + runner script<br/>Sandbox.branch, non-disposable]
    seed --> b1[sandbox_branch_r1<br/>label, parent uuid]
    b1 --> b2[sandbox_branch_r2<br/>label, parent uuid]
    b2 --> b3[sandbox_branch_r3<br/>label, parent uuid]
    seed -.->|disposable batch jobs| ex[SQL execution:<br/>list of statements in, list of outcomes out]
```

`Sandbox.lineage()` returns `(uuid, parent_uuid, label)` tuples. `GET /api/runs/{id}/tree` joins that lineage with per-round metrics from the report and never invents nodes. Real Contree behaviour (branching with files, stdin, output truncation, image contents) is UNVERIFIED; the fake sandbox reproduces the lineage semantics for tests.

## Data integrity boundaries

```mermaid
flowchart LR
    train[train tasks] --> teacher[teacher + planner may see]
    dev[dev tasks] --> planner[planner failure analysis]
    held[sealed held-out] -->|only| EV[evaluator.py]
    EV --> GATE[gate.py]
```

## Deploy

One container: the API plus `frontend/dist` (see `Dockerfile`). The deploy path to Nebius Serverless Endpoints is UNVERIFIED.
