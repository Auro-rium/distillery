# FEEDBACK (Token Factory, Sandboxes, fine-tuning, Nemotron)

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
