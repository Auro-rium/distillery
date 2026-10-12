# STATUS

**Date:** 2026-10-12 · **Phase:** final build (WIN PLAN H1-H6) toward the Nebius x NVIDIA Global AI Hackathon; **Devpost deadline 2026-10-30 10:00 PT** (our internal submit date is 2026-10-27; judging runs 2026-12-01 to 12-15 and the site must stay up). Baseline of proven and broken claims: `GROUND_TRUTH.md` (25 claims). Pre-registrations and results: `DECISIONS.md`. Older status detail is in git history.

## Results (all evidence in `docs/proofs/evidence/`)
| Round | Decision | Key numbers | Evidence |
|---|---|---|---|
| P4 (templated held-out) | **PROMOTE** (in-distribution only) | n=300: base 0.257, student 0.923, teacher 0.900; ratio lower bound 0.993; McNemar 202 vs 2. Stress (n=100): student 0.16 | `p4_run_summary.json` |
| B2 fair teacher (S0, S1; S2 Ultra deferred) | **no claim** | P4 reproduced exactly. Student 0.9233 vs 8-shot Super 0.9333: ratio 0.989, lower bound 0.957 (needs > 1.0) | `b2_fair_teacher_s0_s1.json` |
| Gate B (agent-authored, n=86) | **REJECT** | base 0.384, student 0.407, teacher 0.988 (inflated); ratio lower bound 0.306; McNemar p 0.84 | `gateb_agent_authored.json` |
| A5 live chaos run | **PASS** | 8 injected faults, 4 restarts, 1 admin action, 1 fine-tune job billed, verdict PROMOTE (lower bound 1.022) | `a5_chaos.json` |
| N1 natural-phrasing round (N1-eval, agent-authored, n=98) | **REJECT** | base 0.296, student 0.571, teacher 0.959; ratio lower bound **0.495 < 0.85**; McNemar p 4.6e-07 (29 vs 2). Held-out: no regression (lower bound 0.989). Stress: student 0.66, no longer unseen concepts. Gate B re-score of N1: **pending** | `n1_natural_round.json` |

**Reading.** The agent is autonomous and the in-distribution win is real. Training on naturally phrased, teacher-verified questions moved the student from indistinguishable-from-base to clearly better than base, but not to the 0.85 bound. The two agent-authored sets differ, so that is not a measured improvement; teacher accuracy on them is inflated by construction. The plan turns this limit into the feature: a certified router (H3) serves the student only where it can guarantee the error rate and defers the rest. Certified adapter: N1 (`7d6ce813…`).

## Spend (store ledgers, from P4 onward; sandbox is recorded at $0, so every figure is a lower bound)
| Step | USD |
|---|---|
| P4 | 4.2582 |
| B2 | 1.6495 |
| Gate B (drafting 0.1743 + two scoring passes 0.0925) | 0.2668 |
| A5 | 4.1918 |
| N1 (ceiling was 5.00) | 3.8289 |
| **Total** (exact 14.195248) | **14.1952** of the 19.50 cap |
| Hosted-demo reservation (through Dec 15) | 1.50 |
| **Available to build with** | **about 3.80** (19.50 - 14.1952 - 1.50), plus sandbox seconds, which are unpriced |

## What is next (WIN PLAN)
- **H1** Measure the student on a time-boxed Nebius AI Cloud L40S vLLM + LoRA endpoint (deleted in `finally`) to get a real cost per 1k requests. Paid, about 1.55.
- **H2** OpenAI-compatible gateway (`/v1/chat/completions`) plus the Acme analyst agent: only the base URL changes.
- **H3** Certified router: `certify()` with alpha 0.08, delta 0.05, fit/calibration/test splits sealed and pre-registered before any scoring; serves N1 where certified, the teacher elsewhere.
- **H4** UI: the Playground becomes the Acme agent (served by, confidence, SQL and rows, cost, server-side savings counter).
- **H5** Receipts: every claim with its value, permalink, pre-registration commit and timestamp, sealed-set and adapter hashes, and ledger total; a `/receipts` route.
- **H6** Drift: a `CERTIFIED -> ROLLBACK -> TEACHER` state machine with a simulated burst, rollback, and a queued (not run) re-distill.

## Blockers (you)
- Console balance reading and the Sandboxes usage line, so sandbox seconds can be priced and the totals above stop being lower bounds.
- Render: switch **Auto-Deploy off** on service `distillery` (`srv-dauuokk1nsns73fqcv70`) and confirm. No push before that.
- Nebius AI Cloud login and billing for H1, and whether the hackathon credits cover AI Cloud.
- Approval of each paid step: the Gate B re-score of the N1 adapter (about 0.05), H3 data and scoring (about 0.87 plus sandbox time), H1 (about 1.55), the end-to-end run.
- Yours as agreed: the demo video (under 3:00, public on YouTube), the Devpost submit, making the repo public.
