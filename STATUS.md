# STATUS

**Date:** 2026-10-16 · **Phase:** final build (WIN PLAN H1-H6) toward the Nebius x NVIDIA Global AI Hackathon; **Devpost deadline 2026-10-30 10:00 PT** (our internal submit date is 2026-10-27; judging runs 2026-12-01 to 12-15 and the site must stay up). Baseline of proven and broken claims: `GROUND_TRUTH.md` (29 claims). Pre-registrations and results: `DECISIONS.md`. Older status detail is in git history.

## Results (all evidence in `docs/proofs/evidence/`)
| Round | Decision | Key numbers | Evidence |
|---|---|---|---|
| P4 (templated held-out) | **PROMOTE** (in-distribution only) | n=300: base 0.257, student 0.923, teacher 0.900; ratio lower bound 0.993; McNemar 202 vs 2. Stress (n=100): student 0.16 | `p4_run_summary.json` |
| B2 fair teacher (S0, S1; S2 Ultra deferred) | **no claim** | P4 reproduced exactly. Student 0.9233 vs 8-shot Super 0.9333: ratio 0.989, lower bound 0.957 (needs > 1.0) | `b2_fair_teacher_s0_s1.json` |
| Gate B (agent-authored, n=86) | **REJECT** | base 0.384, student 0.407, teacher 0.988 (inflated); ratio lower bound 0.306; McNemar p 0.84 | `gateb_agent_authored.json` |
| A5 live chaos run | **PASS** | 8 injected faults, 4 restarts, 1 admin action, 1 fine-tune job billed, verdict PROMOTE (lower bound 1.022) | `a5_chaos.json` |
| N1 natural-phrasing round (N1-eval, agent-authored, n=98) | **REJECT** | base 0.296, student 0.571, teacher 0.959; ratio lower bound **0.495 < 0.85**; McNemar p 4.6e-07 (29 vs 2). Held-out: no regression (lower bound 0.989). Stress: student 0.66, no longer unseen concepts. Gate B re-score of N1 (unpaired, fit-split context): 66/86 = 0.767 | `n1_natural_round.json`, `n1_gate_b_unpaired.json` |
| H3 certified router, W-C1 (sealed test, n=490) | **PASS** | λ̂ 0.917 certified on calibration (n=489). Student serves 240 (coverage 0.490) with 10 wrong (selective error 0.042, 95% upper 0.070, alpha 0.08); system 0.951 vs teacher-only 0.935 | `router_eval_test.json`, `router_cert.json` |
| H3 W-C2 | **PASS** (small separation) | 115/117 wrong student answers deferred (lower bound 0.947), but 113/123 right natural + stress answers deferred too (0.919) | `router_eval_test.json` |
| H6 W-D drift burst (30 credit-grant questions) | **PASS** | rollback on burst request 13 (limit 25), 0 false alarms, 0 wrong student answers during the burst; 295/300 seeds pass | `drift_burst.json`, `h3_drift_report.json`, `h3_burst_counts.json` |
| W-E2E live end to end (local live server, 2026-10-16) | **PASS** | 12/20 normal questions served by the student (all verified, all templated), 8 deferred (all 6 website examples); rollback on burst request 14 (limit 25), 0 student answers after it; student median 39 s on sandbox CPU; $0.0331 | `w_e2e.json` |

**Reading.** The agent is autonomous and the in-distribution win is real. Training on naturally phrased, teacher-verified questions moved the student from indistinguishable-from-base to clearly better than base, but not to the 0.85 bound. The two agent-authored sets differ, so that is not a measured improvement; teacher accuracy on them is inflated by construction. The plan turns this limit into the feature: a certified router (H3) serves the student only where it can guarantee the error rate and defers the rest. Certified adapter: N1 (`7d6ce813…`).

## Spend (store ledgers, from P4 onward; sandbox is recorded at $0, so every figure is a lower bound)
| Step | USD |
|---|---|
| P4 | 4.2582 |
| B2 | 1.6495 |
| Gate B (drafting 0.1743 + two scoring passes 0.0925) | 0.2668 |
| A5 | 4.1918 |
| N1 (ceiling was 5.00) | 3.8289 |
| H3 (natural-question drafting 0.6860 + fit/cal/test/burst signals 0.3433) | 1.0293 |
| W-E2E (smoke 0.0007 + run 0.0331) | 0.0338 |
| **Total** (global ledger) | **15.2583** of the 19.50 cap |
| Hosted-demo reservation (through Dec 15) | 1.50 |
| **Available to build with** | **about 2.74** (19.50 - 15.2583 - 1.50), plus sandbox seconds, which are unpriced |

## What is next (WIN PLAN)
- **H1 / W-$** Measure the student on a time-boxed Nebius AI Cloud L40S vLLM + LoRA endpoint (deleted in `finally`) to get a real cost per 1k requests. Paid, about 1.55 (preemptible about 0.74); not approved; needs the AI Cloud login and bucket.
- **H2, H4, H5** Done: gateway, Acme analyst, receipts; the site now opens on a product Overview (`/`), the distillation story is at `/mission`.
- **H3, H6** Done: W-C1, W-C2, W-D and W-E2E PASS. The certified router and the N1 checkpoint ship in the image (`deploy/router`, `deploy/student`).
- **Deploy** Render dashboard checklist in `docs/DEPLOY.md`, then Manual Deploy, `scripts/smoke_live.py`, and the hosted W-E2E check (`scripts/w_e2e.py --hosted`).

## Blockers (you)
- Render dashboard: the live service reported `DISTILLERY_PLAYGROUND_DAILY_CAP_USD` 0.5 and student `Qwen/Qwen3-0.6B`; fix both per `docs/DEPLOY.md`, then Manual Deploy.
- Console balance reading and the Sandboxes usage line, so sandbox seconds can be priced and the totals above stop being lower bounds.
- Nebius AI Cloud login and billing for H1, and whether the hackathon credits cover AI Cloud.
- Yours as agreed: the demo video (under 3:00, public on YouTube), the Devpost submit.
