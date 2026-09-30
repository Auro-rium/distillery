# FEEDBACK (Token Factory, Sandboxes, fine-tuning, Nemotron)

Log each friction point as it happens: what I tried, what happened, what would have helped.

<!-- entries go here -->

## 2026-09-30
- **Sandboxes docs vs SDK:** the Python SDK docs say `Contree(api_client)`; the published `contree-sdk` 0.3.6 raises `AttributeError: 'ContreeAsyncClient' object has no attribute 'auth'` for that. Only `Contree(token=..., base_url=...)`/`ContreeConfig` work. Would have helped: docs pinned to a released version, or a version note.
- **Sandboxes project id is easy to miss:** the getting-started page never mentions a project id; the API answers `400 Missing "Project" header` and the SDK surfaces a generic `403 ForbiddenError`. Only the CLI `auth` page mentions `NEBIUS_AI_PROJECT`. A clear error naming the missing project (in the SDK) and a line in Getting Started would have saved time.
- **Package names not in docs:** PyPI names (`contree-sdk`, `contree-client`) are not stated anywhere in the SDK docs.
- **No prices in the API or a scrapeable page:** `/v1/models` has no pricing fields, and the pricing page is a JS shell, so cost tracking needs hand-copied prices. Adding price fields to `/v1/models` would allow accurate budget guards.
- **Fine-tunable != servable:** Qwen3-1.7B is fine-tunable but is not in the serverless model list, so there is no cheap way to run the base model or the LoRA student via the inference API; custom-weights endpoints are "on request".
- **Fine-tuning doc sample bug:** the Python poll loop's condition is inverted (`while status in ["succeeded","failed","cancelled"]`), and the JSON-mode sample is inconsistent about the `json_schema` wrapper (the `{name, schema, strict}` form works).
- **Positive:** JSON-schema mode worked on first try on all three Nemotron models; `usage` includes reasoning tokens.
