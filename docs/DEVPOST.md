# Distillery: Devpost submission (Coding and Agentic Engineering track)

Fill every `[PLACEHOLDER]` from a real run before submitting. Do not submit invented numbers; if a number was not measured, say so.

## Lineage note (required)

This concept was first explored in an earlier AWS-based prototype (Aug 2026) that never completed an end-to-end run. Distillery is a from-scratch rebuild started Sep 29, 2026 on Nebius Token Factory; no code was carried over.

## Tagline

An agent that distils a big-model task into a small model, and only ships it if execution-verified evidence says it is good enough.

## Inspiration

Teams pay a large model to do the same narrow job millions of times. Distilling that job into a small model is well understood, but doing it safely is tedious: generating data, checking labels, fine-tuning, evaluating honestly and deciding whether to ship. Most distillation demos skip the last two. We wanted an agent that does the whole loop and refuses to ship on vibes.

## What it does

Given a narrow task (today: text-to-SQL over a synthetic SQLite database), Distillery:

1. Generates tasks and gold answers, then cross-checks the gold labels with the planner (Nemotron 3 Ultra) and teacher (Nemotron 3 Super) by comparing execution results.
2. Seals a held-out set, including task families the student never trains on.
3. Has the teacher write training answers; only rows that execute correctly in a sandbox are kept. Nemotron 3 Nano flags dirty formatting.
4. Fine-tunes Qwen3-1.7B (LoRA) on Nebius Token Factory, scores it on a dev set, has the planner cluster the failures into families, generates targeted data, and branches the sandbox. Each round is a node in an experiment tree.
5. Scores the best-on-dev candidate once on the sealed held-out set and applies a pure-Python gate: PROMOTE only if the student's accuracy relative to the teacher has a bootstrap lower bound of at least 0.85 and the student beats the base model on an exact McNemar test.
6. Produces a report: accuracy with confidence intervals, cost split, examples fixed / still wrong / regressed, and the experiment tree.

## How we built it

- Python 3.12 backend: staged, resumable orchestrator; SQLite store with content-addressed artifacts; FastAPI server that streams progress over SSE; React/TypeScript frontend with Replay, New run, Live run, Experiment tree, Report and Playground screens.
- Nemotron roles: Ultra plans, Super teaches, Nano triages (IDs in the README, live-verified).
- Token Factory: structured output (`json_schema`) for planner and triage, fine-tuning API for LoRA, Sandboxes for execution and branching.
- Integrity is enforced in code and tests: only the evaluator can read held-out (AST test), train and eval prompts come from one builder, the evaluated adapter is SHA-256-checked against the trained one, paid jobs and endpoints are cancelled or deleted in `finally` blocks (tested for failed jobs, failed generation and a failing endpoint delete, which is raised loudly), and no LLM sits in the gate.

## Challenges

- The fine-tunable small model is not on the serverless inference list, so serving the base and the student is an open problem (spike S4). [UPDATE with what actually worked, or say what did not.]
- Sandboxes docs and the published SDK disagree, and a project id is required but easy to miss (see FEEDBACK.md).
- Keeping a demo honest: dry-run and sample data are labelled everywhere, and no number is shown without a label.
- [ADD real challenges from the real run.]

## Accomplishments

- [FILL after a real run: e.g. student reached X% of teacher accuracy at Y% of cost. Do not write this line unless measured.]
- A complete pipeline that runs offline end to end against fakes (`--dry-run`), so plumbing, resume, budget refusal and sealed-held-out rules are testable without spending money.
- Written feedback on Token Factory and Sandboxes friction, in FEEDBACK.md.

## What we learned

[FILL from the real run.] Already observed: Nano spends many reasoning tokens even on one-line answers, so "cheap" is not "few tokens"; `json_schema` mode worked first try on all three Nemotron models.

## What is next

Serve the student on Nebius and close the cost comparison; more task packs (a Python-function pack with generated tests); a bigger held-out set.

## Built with

Nebius Token Factory (Nemotron 3 Ultra, Super, Nano; fine-tuning; Sandboxes), Qwen3-1.7B, Python, FastAPI, SQLite, React, TypeScript, Vite, Docker.

## Results (fill from a real run only)

| Metric | Value |
|---|---|
| Held-out execution accuracy: base / student / teacher | [TBD] / [TBD] / [TBD] |
| Student/teacher ratio, 95% lower bound | [TBD] |
| McNemar p (student vs base) | [TBD] |
| Gate decision | [TBD] |
| Total spend | [TBD] |
| Cost per 1k tasks: teacher / student | [TBD] / [TBD or "unavailable: serving path undecided"] |

## Links

- Live demo: [URL]
- Repository: [URL]
- Demo video (<= 3 min): [URL]
- Recorded run bundle: [URL or path under replay/]

## Testing instructions for judges

No account or key is needed to try it.

1. Open the live demo: [URL]. The landing page is Replay. It shows either a real recorded run (banner "Recorded run, real Token Factory jobs") or the labelled dry-run sample (banner "DRY RUN, fake models, numbers are NOT results"). Check which banner you see.
2. Open the Report and Experiment tree screens of the replayed run.
3. Open the Playground. It is rate-limited and has a daily dollar cap; models that cannot be served show an honest "unavailable" reason.
4. To run locally without keys:
   ```bash
   git clone [REPO URL] && cd distillery
   python -m venv .venv && .venv/bin/pip install -e ".[dev]" fastapi uvicorn httpx
   .venv/bin/python -m distillery run --pack sql --scale tiny --dry-run
   .venv/bin/python -m pytest -q
   ```
   The dry run uses fake models and fake prices and proves plumbing only.
5. Live runs need `NEBIUS_API_KEY`, `NEBIUS_AI_PROJECT`, a prices file and the admin token; they are disabled on the public demo.

## Track fit: Coding and Agentic Engineering

An agent plans, generates, verifies by execution, fine-tunes, analyses its own failures, and decides whether to ship, with a deterministic gate as the safety boundary.
