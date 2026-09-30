# Demo video script (max 2:50)

Rule: every number spoken or shown must come from a real run or a measured source. Where a blank is marked `[FILL]`, do not record that line until the number exists. Any clip that is sped up carries an on-screen "SPED UP xN" label. If the recording is from a dry run, the "DRY RUN, fake models, numbers are NOT results" banner must stay visible the whole time.

## Timeline

| Time | Beat | Voice-over (edit to fit) | On screen |
|---|---|---|---|
| 0:00-0:15 | The problem | "Teams pay a big model to do the same narrow job over and over. Our teacher costs [FILL: measured cost of the teacher on N tasks, from the report] to run [FILL: N] SQL questions. A small model could do it for less, if you can trust it." | Cost number from the real report; title card |
| 0:15-1:20 | The run | "Distillery is an agent on Nebius Token Factory. Nemotron 3 Ultra plans, Super teaches, Nano triages. Watch a run." Narrate stages as they appear: gold cross-check, sealed held-out, teacher data verified by execution in a sandbox, fine-tune, dev eval, failure analysis, branch. | New run screen -> Live run screen with stage list, SSE log, live spend meter. Label "SPED UP xN" on any fast-forwarded stretch |
| 1:20-2:15 | The report | "The held-out set was opened once. Student [FILL] vs teacher [FILL] vs base [FILL]. The ratio's lower bound is [FILL] against a threshold of 0.85; McNemar p is [FILL]. The gate says [PROMOTE or REJECT], and it is plain Python, no LLM." Show examples fixed / still wrong / regressed. If the result is REJECT, say so plainly. | Report screen: accuracy with CIs, gate reasons, examples tabs, experiment tree |
| 2:15-2:50 | Architecture and Nebius usage | "Fine-tuning API for the LoRA student, Sandboxes to execute and branch every round, structured output for the agents. Cost split: [FILL from report: planner / teacher / triage / fine-tune estimate / student]." | Architecture diagram (docs/ARCHITECTURE.md), cost-split chart from the report, repo and demo URLs |

## Shot list

1. Title card and the problem cost number (from the report, not typed by hand).
2. New run form: pack `sql`, scale, budget cap visible.
3. Live run: stage list ticking, spend meter, log stream.
4. Experiment tree with the selected node highlighted.
5. Report: gate decision, CI table, the "sealed held-out" note.
6. Examples: one fixed, one still wrong, one regressed if any exist.
7. Playground: one question; show honest "unavailable" for anything not served.
8. Architecture diagram and cost-split chart.
9. End card: demo URL, repo URL.

## What to record

- A real run at scale [FILL: tiny/small/full] on the deployed app or locally against the real API, with the recorded bundle exported (`python -m distillery export-replay <run>`).
- Record at 1080p; browser at a window width that shows the layout without horizontal scroll.
- Screen-record the live run once; cut and speed up waiting periods and label them "SPED UP xN" (state N).
- Do NOT record the dry run as if it were a result. If a real run is not available, the video may show the dry run only with its banner visible and the narration must say "fake models, plumbing demo".
- Never show `.env`, tokens or the terminal env; blur the admin token field.

## Blanks to fill from the real report (report.json)

| Blank | Source |
|---|---|
| Teacher cost on N tasks | `cost.llm_by_model`, `cost.cost_per_1k_tasks` |
| Base / student / teacher held-out accuracy | `evaluation` / `decision` in the report |
| Ratio lower bound, McNemar p | `evaluation` / `decision` |
| Decision | `evaluation` / `decision` |
| Cost split by role | `cost.llm_by_model` |
| Fine-tune cost | reported as an ESTIMATE (`cost.finetune_usd`) |
| Student cost per 1k tasks | may be "unavailable: serving path undecided"; say that if so |

## Timing check

Read the voice-over aloud with a timer before recording; the total must be 2:50 or less.
