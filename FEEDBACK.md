# FEEDBACK (Token Factory, Sandboxes, fine-tuning, embeddings, AI Cloud, Nemotron)

Two parts. The first is a per-tool summary of what we used, what worked and what needs work. The second is the dated log we kept while building, unchanged. Numbers below come from the repo files named next to them; where we have no measurement we say so.

## Summary

Context: Distillery is an agent that distils a text-to-SQL job from Nemotron 3 Super into a Qwen3-1.7B LoRA student. One gated run (P4) ran end to end on Token Factory for $4.26 ([p4_run_summary.json](docs/proofs/evidence/p4_run_summary.json)); the project total to date is $14.1952 ([STATUS.md](STATUS.md)).

### Token Factory inference (OpenAI-compatible)
- **Used for:** the planner, teacher and triage calls; in P4 alone Nano took 684 calls (126,538 input and 753,151 output tokens) and Super 2,638 calls (2,537,916 input, 1,079,412 output) ([p4_run_summary.json](docs/proofs/evidence/p4_run_summary.json)).
- **Worked:** the OpenAI SDK works unchanged; `usage` includes reasoning tokens; reasoning comes back in a separate field so `content` is the final answer; per-token prices are readable from `GET /v1/models?verbose=true` ([deploy/prices.json](deploy/prices.json), read 2026-10-01).
- **Needs work:** `verbose=true` is not in the quickstart, so we first believed there were no prices (2026-09-30 entry). There is no usage or billing endpoint: billed spend is readable only in the console, hours late (2026-10-04 entry). Nano spends far more output tokens than input on short answers (the P4 counts above), so "cheap model" is not "few tokens".

### Structured output (`json_schema`)
- **Used for:** planner and triage objects that the orchestrator parses.
- **Worked:** `json_schema` with `strict: true` and the `{name, schema, strict}` wrapper worked on the first try on Nano, Super and Ultra.
- **Needs work:** the fine-tuning doc's JSON-mode sample is inconsistent about the wrapper (2026-09-30 entry).

### Fine-tuning (LoRA)
- **Used for:** the student: Qwen3-1.7B at lr 1e-4 (P4: 1,488 steps, 5.84M trained tokens; N1: 1,674 steps) ([GROUND_TRUTH.md](GROUND_TRUTH.md) P4 result and row 25; learning rate from the P1.12 pilot, [DECISIONS.md](DECISIONS.md) 2026-10-04).
- **Worked:** files and jobs endpoints, resolved hyperparameters, trained tokens and per-step loss on the job object, checkpoint download, cancel. A job survived a SIGKILL of our process and was adopted again by id, not re-bought (P4 and A5; [a5_chaos.json](docs/proofs/evidence/a5_chaos.json)). The price is $0.40 per 1M trained tokens ([deploy/prices.json](deploy/prices.json), console 2026-10-04); our recomputation from `trained_tokens` matched the console within a cent in both checks (GROUND_TRUTH row 19).
- **Needs work:** defaults silently produce a near-untrained adapter (3 to 9 steps, 2026-10-03 entry); loss masking and the train-time chat template are undocumented; the tokens-per-batch minimum (32768) appears in neither docs nor schema; checkpoint order is unspecified; the job object carries no cost. A fine-tuned model can only be deployed through Dedicated endpoints ("Deployment options currently only include via Dedicated endpoints", models page) and the older "serverless LoRA, per-token billing" page returns Page Not Found ([b4_dedicated_research.md](docs/proofs/evidence/b4_dedicated_research.md), 2026-10-06).

### Sandboxes (Contree)
- **Used for:** executing every gold, training row and evaluation (10 SQL-runner jobs in flight, `EXECUTOR_CONCURRENCY` in `cli.py`, against a beta limit of 50 operations in total per the code comment); branching the experiment tree; serving base and student on CPU.
- **Worked:** branching produced real image lineage that the UI draws; Qwen3-1.7B loads on a 4 vCPU / 4 GB sandbox (peak RSS 3,784 MB, 14.55 s per sample in the batch-1 phase, [docs/spikes/s5-qwen3-1_7b.json](docs/spikes/s5-qwen3-1_7b.json)); a chaos run with injected sandbox faults plus two unplanned ones (an adapter-upload ReadError, a sandbox file-GET read timeout) recovered without human action ([a5_chaos.json](docs/proofs/evidence/a5_chaos.json)).
- **Needs work:** SDK docs and the published SDK disagree, the project id is missing from Getting Started, a job killed at our timeout is reported `SUCCESS` with its CPU accounting dropped, consumption fields are undocumented, no price (the console Usage view had no Sandboxes line on 2026-10-04, so we record sandbox at $0 and every total is a lower bound). Details in the log below.

### Embeddings (`Qwen/Qwen3-Embedding-8B`)
- **Used for:** the router's `nn_sim` signal (nearest-neighbour similarity to the training questions), truncated to the first 1024 of 4096 dimensions (it is Matryoshka-trained) and re-normalised (`embed_index.py`).
- **Worked:** it is listed with a per-token price ($0.01 per 1M input tokens, read from the verbose model listing on 2026-10-12, [deploy/prices.json](deploy/prices.json)).
- **Needs work / unknown:** no live embedding result is recorded in the repo yet, so we make no claim about quality or latency. The router that depends on it is pre-registered and pending (W-C1, W-C2 in [README.md](README.md)).

### Nebius AI Cloud endpoints (H1)
- **Used for:** a time-boxed L40S vLLM + LoRA endpoint to measure the student's serving cost (`aicloud_endpoint.py`, `scripts/h1_endpoint.py`). Not run live yet: W-$ is pending.
- **Worked:** `nebius ai endpoint create --help` (nebius 0.12.282, read 2026-10-13) documents image, platform, preset, volume, auth and on-demand/preemptible flags, enough to build a harness and an offline-tested dry plan.
- **Needs work:** hosting an app needs this separate product, its own CLI and a federated browser login, not the Token Factory key (2026-09-30 entry); the hackathon credit covers Token Factory only ([docs/DEPLOY.md](docs/DEPLOY.md)); several CLI shapes (`get`, `list`, `delete`, state field names) were unverified when the harness was written (listed in the `aicloud_endpoint.py` docstring).

### Dedicated endpoints
- **Used for:** research only; nothing was created ([b4_dedicated_research.md](docs/proofs/evidence/b4_dedicated_research.md), 2026-10-06).
- **Finding:** custom weights on a Dedicated endpoint are "in beta and available on request", so serving a fine-tuned student is not self-service. The documented endpoint template carries model, flavor, `gpu_type`, `gpu_count` and region and no price, and no Qwen3-1.7B price is published. We did not list the templates (it needs an API token), so whether a small-model template exists is unchecked.

### Nemotron 3 Ultra, Super, Nano
- **Prices per 1M tokens (input / output):** Ultra 1.0 / 3.0, Super 0.3 / 0.9, Nano 0.06 / 0.24 ([deploy/prices.json](deploy/prices.json)).
- **Super (teacher):** scores 0.900 on the 300 templated held-out questions ([p4_run_summary.json](docs/proofs/evidence/p4_run_summary.json)); with 8 examples in the prompt it scores 0.9333 on the same set, which is the fair bar the student did not beat ([b2_fair_teacher_s0_s1.json](docs/proofs/evidence/b2_fair_teacher_s0_s1.json)). It is also the model that answers every request the router defers.
- **Ultra (planner):** clusters dev failures into families and, optionally, proposes round actions under pre-registered bounds. The P4 cost table lists only Nano and Super, because the gated run used one round, so we have no Ultra usage figure from it. The Ultra 8-shot teacher setup (S2) was deferred on budget.
- **Nano (triage):** advisory format check and one-line summaries; see the token note above.

### Zero to hello world
We did not time it, so we give no number. What we recorded: inference with `json_schema` worked on the first try. Sandboxes took longer because of three gaps found the same day: `Contree(api_client)` raises `AttributeError` in the published SDK, the project id is not in Getting Started (the API answers `400 Missing "Project" header` and the SDK shows a generic `403`), and the PyPI package names are not in the docs (2026-09-30 entry).

### Would we build on it again
Yes, for inference, structured output, fine-tuning and Sandboxes: the whole agent loop ran live for $4.26 and survived injected faults. Not yet for serving the fine-tuned student: that path needs Support to enable custom weights or a separate AI Cloud account, which is why the cost-per-request claim is still pending.

### The asks that would have saved us the most time
1. A cost field on fine-tune job and sandbox operation objects (and a usage endpoint).
2. Self-service serving of a LoRA adapter (serverless LoRA or Dedicated custom weights).
3. Documented loss masking and chat template for Conversational rows, and a warning when planned steps are tiny.
4. The project id in Getting Started and in the SDK error.
5. A `TIMED_OUT` status with the CPU actually consumed.

## Dated log

Log each friction point as it happens: what I tried, what happened, what would have helped.

<!-- entries go here -->

## 2026-09-30
- **Sandboxes docs vs SDK:** the Python SDK docs say `Contree(api_client)`; the published `contree-sdk` 0.3.6 raises `AttributeError: 'ContreeAsyncClient' object has no attribute 'auth'` for that. Only `Contree(token=..., base_url=...)`/`ContreeConfig` work. Would have helped: docs pinned to a released version, or a version note.
- **Sandboxes project id is easy to miss:** the getting-started page never mentions a project id; the API answers `400 Missing "Project" header` and the SDK surfaces a generic `403 ForbiddenError`. Only the CLI `auth` page mentions `NEBIUS_AI_PROJECT`. A clear error naming the missing project (in the SDK) and a line in Getting Started would have saved time.
- **Package names not in docs:** PyPI names (`contree-sdk`, `contree-client`) are not stated anywhere in the SDK docs.
- **Prices are only partly in the API:** `GET /v1/models?verbose=true` returns a `pricing` object (per-token prompt/completion) for inference models, but only with `verbose=true` (undocumented in the quickstart), so we first thought there were none. Fine-tuning (SFT/LoRA) and Sandbox prices are in neither the API nor the docs (the pricing pages are JS shells); there is no usage/billing endpoint, so billed spend can only be read in the console. A `pricing` block on the fine-tune job and sandbox operation objects would allow exact budget guards.
- **Hosting a web service needs a different product than the key you get:** the Token Factory key cannot create a container with an inbound port (Sandboxes have none; dedicated endpoints host models). Serving an app needs Nebius AI Cloud serverless endpoints, which need the separate `nebius` CLI and a federated browser login. A single "deploy this image" path from Token Factory would help hackathon builders.
- **Fine-tunable != servable:** Qwen3-1.7B is fine-tunable but is not in the serverless model list, so there is no cheap way to run the base model or the LoRA student via the inference API; custom-weights endpoints are "on request".
- **Fine-tuning doc sample bug:** the Python poll loop's condition is inverted (`while status in ["succeeded","failed","cancelled"]`), and the JSON-mode sample is inconsistent about the `json_schema` wrapper (the `{name, schema, strict}` form works).
- **Positive:** JSON-schema mode worked on first try on all three Nemotron models; `usage` includes reasoning tokens.

## 2026-10-03
- **Fine-tune defaults silently produce a near-untrained adapter:** defaults are lr 1e-5, batch 8, packing on (8192 ctx), LoRA r8/alpha8. A 117-row conversational dataset (~960 tokens/row) packs ~8 rows per sequence, so 3 epochs = 6 optimizer steps (`trained_steps: 6`). The job succeeds and the loss falls a little (0.674 to 0.609), so nothing warns you. Would have helped: `total_steps` estimated at create time, or a warning when it is very small.
- **Loss masking for conversational data is not documented:** the datasets page explains `-100` labels only for the Pretokenized format; it does not say whether Conversational rows train on assistant tokens only, the last assistant message, or the whole sequence (system + schema prompt included). For prompt-heavy tasks (a ~900-token schema prompt, ~60-token SQL answer) this decides whether training works. We had to infer it from step-1 loss vs a local base-model loss.
- **Chat template at train time is only discoverable from checkpoint files:** the checkpoint download includes `chat_template.jinja` (stock Qwen3, renders `<think>\n\n</think>\n\n` before the final assistant turn). The docs do not say that template is the one applied to Conversational rows at train time, or which `enable_thinking` behaviour applies.
- **Checkpoint list order and `max epochs = 20` are undocumented outside the OpenAPI schema:** we pick the checkpoint with the highest `step_number`; the list order is not specified.
- **Undocumented tokens-per-batch minimum:** creating a job with `batch_size 2, context_length 8192, packing false` returns `422 'Batch size 2 and context length 8192 result in less than 32768 tokens per batch.'` The limit is in neither the docs nor the OpenAPI schema. With packing off, small batches on short rows need an inflated `context_length` (we used 16384) just to pass validation. Stating the rule (or applying it only with packing on) would help.

## 2026-10-04
- **A job killed at its timeout is reported as `SUCCESS` with its CPU accounting dropped:** in the P1.11 scoring, 2 generation jobs ran into our 210 s timeout. The operation shows `status SUCCESS`, `duration 210.6`, `consumed_cpu 0.001`, `consumed_memory ~10k`, which reads like a job that never started (we first logged it as such). A controlled re-run showed the real cause: those batches need 140-220 s of generation, i.e. our timeout was too tight (greedy decoding makes the same batch slow every time). A `TIMED_OUT` status, and the CPU actually consumed before the kill, would have made this a 1-minute diagnosis instead of an experiment. The SDK also cancels an operation on any client-side poll exception (`_operation_canceller`), which shows up as "Cancelled by user".
- **Consumption is in the API but undocumented:** `GET /v1/operations` returns `consumed_cpu`, `consumed_memory`, `duration` per operation, which is enough to meter sandbox spend, but the units (CPU-seconds? peak KiB?) and the price per unit are not documented anywhere.
- **No way to read fine-tune or sandbox cost programmatically:** the billing docs point only to the console Usage tab. `/api/public/models_info` and the model catalog list inference prices only; the only training price in the API is for spec-draft training (`SpecDraftTrainingPrice`, no unit stated).
- **Usage tab lags and has no per-job view:** at 13:57 UTC the Usage page said "Last updated 11:39 UTC", so a run's cost is visible only hours later, and only per day and service, not per fine-tune job or sandbox operation. VAT (here 18%) is added for an individual account even though the billing docs say "the rest of the world: individuals, no VAT at the moment". A cost field on the job and operation objects would make per-run costs exact.
