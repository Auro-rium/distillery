# B4 research: can a Dedicated endpoint host the student LoRA adapter, and at what hourly price?

Read-only web research, 2026-10-06. Nothing was signed up for, created, or called on Nebius.

## Student base model in this repo
- The promoted run trained **`Qwen/Qwen3-1.7B`** with LoRA (adapter sha256 `4bebc793...`): `docs/proofs/evidence/p4_run_summary.json` -> `artifact.trained_base_model`.
- Note: `.env` still sets `DISTILLERY_MODEL_STUDENT=Qwen/Qwen3-0.6B`; the run overrode it (the P5 live script pins 1.7B).
- Qwen3-1.7B is on the supported fine-tuning list (LoRA and full): https://docs.tokenfactory.nebius.com/post-training/models.md

## Findings
1. **Dedicated endpoints exist** and are priced per GPU-hour with per-minute granularity (public serverless is per token):
   https://docs.tokenfactory.nebius.com/ai-models-inference/dedicated-endpoints/overview.md
2. **Custom weights on a Dedicated endpoint are "in beta and available on request"**: the docs say to contact Support to enable access and be guided through setup; "availability, supported configurations, and onboarding steps may vary":
   https://docs.tokenfactory.nebius.com/ai-models-inference/dedicated-endpoints/custom-weights.md and
   https://docs.tokenfactory.nebius.com/ai-models-inference/dedicated-endpoints/faq.md ("I finetuned a model and want to deploy it via Dedicated endpoint").
   The fine-tuning models page states: "Deployment options currently only include via Dedicated endpoints":
   https://docs.tokenfactory.nebius.com/post-training/models.md
   So: **possible only after Support enables custom weights for our project; not self-service, and not verified for this adapter.** A LoRA adapter might need merging into the base weights first (the docs ship a merge guide for MoE LoRA only: https://docs.tokenfactory.nebius.com/post-training/merge-moe-lora-weights.md); whether a plain dense Qwen3-1.7B LoRA is accepted as an adapter is not documented, and has to be asked of Support.
3. **Serverless LoRA ("per-token billing") is mentioned but the page is gone.** The fine-tuning overview in our docs snapshot (`/home/lenovo/.claude/jobs/958b3c06/tmp/fine-tuning_overview.md`) links "Deploy Custom LoRA ... serverless LoRA adapter models ... per-token billing", but `https://docs.tokenfactory.nebius.com/fine-tuning/deploy-custom-model.md` and `/post-training/deploy-custom-model.md` both return "Page Not Found" today, and the current models page says deployment is Dedicated-only. Treat serverless LoRA as unavailable until Support confirms.
4. **No fixed Dedicated price for Qwen3-1.7B is published.** The endpoint is created from a template (`GET /v0/dedicated_endpoints/templates`, which needs an API token, so it was not called) giving model, flavor, `gpu_type`, `gpu_count`, region: https://docs.tokenfactory.nebius.com/ai-models-inference/dedicated-endpoints/deploy-api.md . Billing is while at least one replica is `ready` (provisioning and restarts are not billed): https://docs.tokenfactory.nebius.com/ai-models-inference/dedicated-endpoints/billing-policy.md . Charges may differ under a contract. Capacity is on-demand and not guaranteed across stops (https://docs.tokenfactory.nebius.com/ai-models-inference/dedicated-endpoints/capacity-and-scaling.md); no formal SLA without a contract.
5. **GPU hour prices (the only hourly numbers found)**, Nebius AI Cloud on-demand, https://nebius.com/prices : HGX H100 **$3.85/GPU-h**, HGX H200 **$4.50/GPU-h**, L40S (Intel) from **$1.55/GPU-h**, L40S (AMD) from $1.82/GPU-h; preemptible H100/H200 from $0.79, L40S from $0.74. A 1.7B model fits one GPU, so a **single-GPU replica is the floor: about $1.55/h (L40S) to $3.85/h (H100) = $37-$92/day**, **assuming** Dedicated endpoints are billed at these list GPU rates, which the docs do not state. This is an inference, not a quoted endpoint price.

## Verdict for B4
- Dedicated LoRA hosting: **not self-service; requires Support to enable custom weights (beta).** The hourly price is **unconfirmed**; the best bracket is one GPU at $1.55-$3.85/h (above).
- Per the plan's fallback, `scripts/b4_breakeven.py` therefore labels the student **"sandbox CPU, not Dedicated"** unless an hourly price is passed with `--student-label dedicated --hourly-usd X` after Support confirms. Bring a pre-flight and ask for approval before creating anything.
- Question to send Support: "Can the LoRA adapter (Qwen/Qwen3-1.7B, rank 16) from fine-tuning job ftjob-4676999d... be served on a Dedicated endpoint, which GPU template, and what is the $/GPU-hour?"
