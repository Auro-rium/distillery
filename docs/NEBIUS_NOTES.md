# NEBIUS NOTES

Everything below is **from docs, not yet verified live** unless a line says otherwise. Sources are the
saved pages under `spikes/out/` (fetched from https://docs.tokenfactory.nebius.com/<path>.md, 2026-09-29).
Spikes S1-S5 have not been run. Discrepancies between the build spec and the docs are listed at the end.

## Inference / JSON
Source: `ai-models-inference_json.md`, `quickstart.md`
- OpenAI-compatible. `base_url="https://api.tokenfactory.nebius.com/v1/"`, key via `Authorization: Bearer`. Chat at `/v1/chat/completions`.
- `response_format={"type": "json_schema", ...}` (schema following) or `{"type": "json_object"}` (arbitrary JSON; prompt must ask for JSON).
- Docs recommend giving the schema in the prompt as well as in `response_format`, and testing several models; only models with the "JSON mode" tag support it.
- Two shapes appear: the Python sample passes the raw schema as `"json_schema": Film.model_json_schema()` (the sample also has missing commas), while the "valid JSON schema" example and JS `zodResponseFormat` use `{"name", "schema", "strict": true}`. `llm.py` uses the latter (UNVERIFIED).
- Response may carry `message.refusal`; handle before parsing `message.content`.
- The docs say nothing about reasoning/thinking output fields or a thinking toggle (grep for reasoning/thinking/enable_thinking: none in inference pages). `llm.py` splits a `reasoning_content`/`reasoning` field and inline `<think>` blocks from final content (UNVERIFIED).

## Rate limits
Source: `ai-models-inference_rate-limits.md`
- Dynamic limits in 15-minute buckets: avg usage >= 80% -> next limit x1.2; <= 50% -> divided by 1.5; hard ceiling 20x base. Example baseline 60 RPM / 400,000 TPM (defaults are shown in the console, not in docs).
- Over limit -> HTTP 429. Some over-limit requests are still served at low priority with header `x-ratelimit-over-limit: yes` (treat as a warning).
- Headers: `x-ratelimit-limit/remaining/reset-{requests,tokens}`, `x-ratelimit-dynamic-*`, and `Retry-After` (seconds) on 429. `llm.py` honours `Retry-After`, else exponential backoff with jitter.
- Batch API has "significantly higher limits" (page not saved; not used yet).

## Fine-tuning API
Sources: `post-training_how-to-fine-tune.md`, `post-training_models.md`, `post-training_datasets.md`, `api-reference_fine-tuning_cancel-fine-tuning-job.md` (OpenAPI excerpt)
- Same base URL and OpenAI SDK. Upload: `client.files.create(file=..., purpose="fine-tune")` (`POST /v1/files`, multipart). Dataset limit: 20 GB via Files API.
- Create: `POST /v1/fine_tuning/jobs` / `client.fine_tuning.jobs.create(model, training_file, validation_file?, suffix?, hyperparameters?, seed?, integrations?)`. `suffix` max length 64 (OpenAPI).
- Get: `GET /v1/fine_tuning/jobs/{id}`; events: `GET .../events?limit&after` (`data[]`, `has_more`); checkpoints: `GET .../checkpoints` (`data[]` with `id`, `step_number`, `fine_tuned_model_checkpoint`, `metrics{train_loss,valid_loss}`, `result_files[]`). Cancel: `POST /v1/fine_tuning/jobs/{job_id}/cancel` ("Immediately cancel"; returns the job).
- Status enum (OpenAPI): `validating_files`, `queued`, `running`, `succeeded`, `failed`, `cancelled`. The cURL text says terminal = succeeded/failed; the Python list includes cancelled. The Python sample loop `while job.status in ["succeeded","failed","cancelled"]` is inverted (bug in docs); `finetune.py` polls while NOT terminal.
- Poll every >= 15 s. `error` = `{code, message, param}` when failed; for transient 5xx "you can recreate the job".
- Job fields: `result_files`, `trained_tokens`, `trained_steps`, `total_steps`, `estimated_finish`.
- Hyperparameters (how-to + OpenAPI): `batch_size` 1-64 (doc "typical" 8-32, default 8); `learning_rate` >= 0 (default 1e-5); `n_epochs` 1-20 (default 3); `warmup_ratio` 0-1; `weight_decay` >= 0; `lora` bool (default false = full FT); `lora_r` 8-128 (default 8); `lora_alpha` >= 8 (default 8); `lora_dropout` 0-1; `packing` (default true); `max_grad_norm` > 0 (default 1); `context_length` 8192-131072, values 8192/16384/32768/65536/131072 (default 8192; inputs longer than this "cause errors").
- Checkpoint flow: `checkpoints.list(job_id).data` -> for each `result_files` id: `files.retrieve(id).filename` (e.g. `<checkpoint_ID>/adapter_config.json`), `files.content(id).write_to_file(path)` (saved under `basename(filename)`). Use the last checkpoint unless there is a reason not to. Exact adapter filenames beyond `adapter_config.json` are not documented (UNVERIFIED).
- Dataset formats (`post-training_datasets.md`): JSONL; conversational `{"messages":[...]}` with the last message from `assistant`; optional `tools` field; also instruction, text, pretokenized.
- Integrations: `wandb`, `hf` (HF push needs a write token; not used).
- Fine-tunable models (`post-training_models.md`). **Qwen3-1.7B is listed: "LoRA and Full Parameter fine-tuning"** (also `Qwen/Qwen3-1.7B-Base`, `-0.6B`, `-4B`, `-8B`, `-14B`, `-32B`, all LoRA+Full; MoE `Qwen3-30B-A3B*`/`235B-A22B*` LoRA+Full; Qwen2.5 0.5B-72B and Qwen2.5-Coder-32B LoRA+Full; Llama 3.1/3.2/3.3 8B-70B, 1B and 3B LoRA+Full; `unsloth/gpt-oss-20b-BF16`/`120b-BF16` LoRA+Full). **Full-only**: DeepSeek-V3-0324, DeepSeek-V4-Flash, Gemma-4 E2B/E4B/31B, Qwen3.5-27B, Qwen3.6-27B. Context lengths for every model: 8192/16384/32768/65536/131072 (default 8192). "Deployment options currently only include via Dedicated endpoints" (Meta section; the Qwen sections say nothing).

## Sandboxes SDK
Sources: `sandboxes_overview.md`, `sandboxes_sdk_python_sdk_*.md`
- Packages `contree_sdk` (`Contree`, `ContreeSync`) and `contree_client` (`contree_client.httpx.ContreeAsyncClient`). **PyPI distribution names are not stated in the saved docs**, so no dependency was added to `pyproject.toml`; `sandbox.py` imports lazily.
- Auth/build: `api_client = ContreeAsyncClient("YOUR-NEBIUS-API-KEY", base_url="https://your-instance.of.contree")`; `sdk = Contree(api_client)`. `contree_sdk` does no auth/transport; caller owns the transport lifecycle. `ContreeAsyncClient.from_profile()` reads `CONTREE_PROFILE` / `~/.config/contree/auth.ini`. The real base URL is not given (placeholder only).
- Images: `await sdk.images.use(ref)` (no API call; ref = tag, UUID, `docker://...`); `use(ref, strict=True)` verifies; `images.oci(ref)` imports if missing; `import_from` always re-imports (avoid). `tag_as`/`untag`; tags unique across images.
- Run: `await image.run(command=None, *, shell, args, env, cwd, hostname, stdin, stdout, stderr, tag, files, timeout, disposable=True, truncate_output_at, preserve_env=False)`. `command` XOR `shell` (ValueError otherwise). `files`: list of paths (land at `/<basename>`) or dict `{dest: path|bytes|UploadFileSpec}` (e.g. `"file.sh"` -> `/file.sh`). `timeout` float seconds or timedelta. Result: `stdout`, `stderr`, `exit_code`, `uuid` (new image version). `DisposableImageRunError` when running on a disposed image.
- Branching (`...branching.md`): each command yields a new image version; `disposable=False` keeps it, so children can be run again from `child.uuid`. All branches from one parent see the same parent filesystem. **Running the same command on the same image gives the same uuid** (caching); random-output commands differ across runs (example 2), so a non-disposable run is how nondeterminism gets pinned. Checkpoint images retained 180 days (untagged, unreferenced ones may be deleted afterwards).
- Beta limit: **50 simultaneously running operations**; the doc says to contact them for a higher limit.

## Dedicated endpoints
Sources: `ai-models-inference_dedicated-endpoints_{deploy-api,operating,billing-policy,custom-weights}.md`
- Control plane `https://api.tokenfactory.nebius.com`: `GET /v0/dedicated_endpoints/templates`, `POST /v0/dedicated_endpoints`, `PATCH|DELETE /v0/dedicated_endpoints/{endpoint_id}`, `GET /v0/dedicated_endpoints`. Create body: `name`, `description?`, `model_name`, `flavor_name` (`base`/`fast`), `gpu_type` (e.g. `gpu-h100-sxm`), `gpu_count`, `region` (`eu-north1`, `eu-west1`, `us-central1`; not updatable), `scaling{min_replicas,max_replicas}`; `enabled` stops/starts. Templates response is the source of truth for valid combos.
- Response gives `endpoint_id` and `routing_key`; use `routing_key` as `model` on the region-specific OpenAI-compatible data plane (e.g. `https://api.tokenfactory.us-central1.nebius.com/v1`). Provisioning takes minutes; expect 404 until routable.
- Billing: billed while >= 1 replica is running and status `ready`; not billed for provisioning, restarts, stop; DELETE releases GPUs. Endpoint deletion is permanent.
- **Custom (fine-tuned) weights: "currently in beta and available on request"; contact Support.** The `post-training/deploy-custom-model` page is linked from the fine-tune how-to but returned 404 when fetched, so LoRA deployment steps are unknown. `post-training/merge-moe-lora-weights` exists (merging LoRA into base; for MoE) and is not saved.

## Docs claims still needing live verification
- Model IDs available on `GET /v1/models` (planner/teacher/triage roles), prices, and per-model JSON-mode support. Docs examples use `Qwen/Qwen3-235B-A22B`, `openai/gpt-oss-120b`, `deepseek-ai/DeepSeek-R1-0528`, `mistralai/Mistral-Nemo-Instruct-2407`.
- Where reasoning text is returned and whether Qwen3 thinking mode has a documented toggle (nothing in docs).
- `response_format` json_schema exact shape (see above), and whether reasoning models honour it.
- Whether our dataset JSONL (`messages`, last = assistant) is accepted and how thinking content in assistant turns is trained.
- Real fine-tune job durations/costs, `n_epochs`/`batch_size` limits (OpenAPI says 64), checkpoint file names, whether `cancel` on a finished job errors.
- Whether fine-tuned Qwen3-1.7B LoRA can be served (dedicated endpoint custom weights beta; serverless job path unknown) -> spike S4.
- Sandbox: PyPI package names, real base URL, exception types on timeout, whether disposable runs return a `uuid`, output truncation defaults, actual concurrency cap (docs: 50), whether a `pip install` of `peft`/torch is feasible in the sandbox (network egress, CPU/RAM limits unknown).
- Rate-limit baselines for our account (console only).

## Model IDs (`GET /v1/models`)
_Not yet verified._ Nano / Super / Ultra / Qwen3-1.7B IDs to be pinned in `config.py` from this list. From docs only: fine-tune id is `Qwen/Qwen3-1.7B`.

## Prices
_Not yet verified._

## Spikes
- S1 Inference: not run
- S2 Sandboxes (40 parallel, branching): not run
- S3 Fine-tune (needs user approval to spend): not run
- S4 Student inference path: not run (`student.py` lists the three candidates)
- S5 Data Lab import: not run

## Discrepancies (spec vs docs)
- Build spec's `poll` note confirmed: the doc's Python loop condition is inverted. Terminal statuses are succeeded/failed/cancelled (the cURL text omits cancelled).
- `batch_size` "typical 8-32" in the how-to vs OpenAPI hard range 1-64; we enforce 1-64.
- Serving the LoRA student: the spec assumes an option exists; docs say custom weights on dedicated endpoints are beta/on request and the deploy page is missing (404).
- Doc JSON sample code is syntactically broken (missing commas) and inconsistent about the `json_schema` wrapper.
- Sandbox concurrency: spec says cap 40 with docs cap 50; consistent, but the 50 counts all simultaneous operations, so `ContreeSandbox` also holds a global 50-slot semaphore.
- Spec says `create_job` uses the paid-resource pattern; we deliberately do NOT auto-retry `create_job` on 5xx (a retry could create a second billable job), only reads/uploads/cancel.

## LIVE-VERIFIED 2026-09-30 (spikes S1, S2a; these override "from docs" statements above)
Scripts: `spikes/s1_models.py`, `spikes/s1_chat.py`, `spikes/s1_llmclient.py`, `spikes/s2_connect.py`. Raw outputs in `spikes/out/` (gitignored).

**Inference (S1) — works with the Token Factory key alone.**
- `GET /v1/models` returns 25 serverless models (fields: id, object, created, owned_by, shutdown_date; **no prices**).
  Nemotron IDs: `nvidia/Nemotron-3-Ultra-550b-a55b`, `nvidia/nemotron-3-super-120b-a12b`, `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B` (also `nvidia/Nemotron-3_5-Lightning`, untested).
- **Qwen3-1.7B / 0.6B are NOT in the serverless list** (only Qwen3-235B/30B-A3B/3.5/3.8-27B/Embedding). So the *base* model cannot be evaluated through the inference API: base must be scored on the same serving path as the student (S4). The orchestrator currently assumes an inference-API base model; this must change.
- `response_format={"type":"json_schema","json_schema":{"name","schema","strict":true}}` works on Nano, Super and Ultra (all returned valid JSON).
- Reasoning text: Nano -> `message.reasoning`; Super -> `reasoning` and `reasoning_content`; Ultra -> `reasoning_content`. `content` is the final answer only. Ultra's plain (non-JSON) answer came back in a ```sql fence.
- `usage` present: `prompt_tokens`, `completion_tokens` (includes reasoning), `completion_tokens_details.reasoning_tokens` (Super/Ultra, null on Nano), cached-token fields.
- Latency for tiny prompts: 0.85-3.8 s. Nano used 224 completion tokens for a one-line answer (heavy reasoning): "cheap" Nano is not cheap in tokens.
- `distillery.llm.LLMClient` verified live on all three roles (structured output parsed, reasoning split, usage logged).
- Prices are not exposed by the API and the public pricing page is a JS shell: **prices must be copied by hand from the console into `DISTILLERY_PRICES_FILE`**.

**Sandboxes (S2a) — blocked on project id.**
- `pip` packages: `contree-sdk` 0.3.6 and `contree-client` 0.4.0 exist on PyPI and install/import fine.
- Default base URL: `https://api.tokenfactory.nebius.com/sandboxes` (CLI docs). `contree auth` accepts `NEBIUS_API_KEY` as the token.
- **Docs vs published SDK mismatch:** docs show `Contree(api_client)`; `contree-sdk` 0.3.6 does NOT accept a client (`AttributeError: ... no attribute 'auth'`); it takes `Contree(token=..., base_url=...)` or a `ContreeConfig`. `sandbox.py`'s `ContreeSandbox` follows the docs and needs fixing.
- Raw `GET /v1/whoami` and `/v1/images` with the key return **400 `Missing "Project" header`**; the SDK call without a project returned 403 Forbidden. IAM auth needs a project id (`NEBIUS_AI_PROJECT` / `CONTREE_PROJECT` / `project=` on the client). The key is a JWT whose payload is not plain JSON, so the project id can't be read from it.
