// Dev-only mock API (VITE_MOCK=1). Serves docs/fixtures/*.json plus synthetic, LABELLED sample
// responses for every endpoint in docs/API_CONTRACT.md. Everything carries dry_run: true.
import fs from "node:fs";
import path from "node:path";
import type { IncomingMessage, ServerResponse } from "node:http";
import type { Plugin } from "vite";
import type { TreeNode } from "../src/api/types";

const FIX = path.resolve(__dirname, "../../docs/fixtures");
const NOTE = "MOCK MODE: synthetic sample data, not a result";
const STAGES = [
  "schema", "questions", "gold_crosscheck", "verifier_selftest", "split", "headroom",
  "teacher_data", "finetune_r1", "dev_eval_r1", "analysis_r1", "targeted_r1",
  "sandbox_branch_r1", "finetune_r2", "dev_eval_r2", "final_eval",
];

const report = (id: string) => ({
  ...JSON.parse(fs.readFileSync(path.join(FIX, "sample-dry-run-report.json"), "utf8")),
  run_id: id, dry_run: true, recorded: false, recorded_at: null,
});

const summary = (id: string, status: string) => ({
  run_id: id, dry_run: true, recorded: false, recorded_at: null, status,
  decision: status === "complete" ? "REJECT" : null, created_at: null,
});

function detail(id: string) {
  const r = report(id);
  const live = id.startsWith("live-");
  const cut = live ? 5 : STAGES.length;
  return {
    run_id: id, dry_run: true, recorded: false, recorded_at: null,
    status: live ? "running" : "complete", error: null,
    stages: STAGES.map((name, i) => ({
      name, status: i < cut ? "done" : i === cut ? "running" : "pending",
      // contract: started_at only while running, ended_at only when done
      started_at: i === cut ? "2026-09-29T00:00:00Z" : null,
      ended_at: i < cut ? "2026-09-29T00:00:05Z" : null,
    })),
    spend: {
      total_usd: r.cost.run_total_usd, cap_usd: r.cost.run_cap_usd,
      by_model: r.cost.llm_by_model, finetune_usd_estimate: null,
    },
    sandbox: { operations: null, concurrency_peak: null },
    verifier: {
      language: "sql",
      code: null,
      selftest: { accepted_gold: 50, rejected_corruptions: 198, failures: 0 },
    },
  };
}

function tree(id: string) {
  const r = report(id);
  const nodes: TreeNode[] = [{
    id: "root", parent_id: null, label: "base", round: null, hypothesis: null, data_delta: null,
    dev_score: r.headroom.base_dev_acc, cost_usd: null, sandbox_image: "fake-base-image", selected: false,
  }];
  for (const x of r.rounds as { round: number; dev_acc: number; sandbox_branch?: { uuid: string } }[]) {
    nodes.push({
      id: `r${x.round}`, parent_id: x.round === 1 ? "root" : "r1", label: `round-${x.round}`,
      round: x.round, hypothesis: `${NOTE}`, data_delta: null, dev_score: x.dev_acc,
      cost_usd: null, sandbox_image: x.sandbox_branch?.uuid ?? null, selected: x.round === r.candidate_round,
    });
  }
  return { nodes };
}

const examples = () =>
  [0, 1, 2].map((i) => ({
    kind: ["fixed", "fixed", "still_wrong"][i], task_id: `mock-${i}`, family: ["join", "null_handling", "date_math"][i],
    heldout_class: "seen", question: `[MOCK, not a real task] question ${i}`,
    gold_sql: "SELECT 1", base_sql: "SELECT 0", student_sql: "SELECT 1", teacher_sql: "SELECT 1",
    base_ok: false, student_ok: i !== 2, teacher_ok: true,
  }));

const unavailable = { available: false, reason: NOTE, sql: null, verified: null, rows_preview: null, error: null };

function send(res: ServerResponse, code: number, body: unknown) {
  res.statusCode = code;
  res.setHeader("Content-Type", "application/json");
  res.end(JSON.stringify(body));
}

function sse(id: string, req: IncomingMessage, res: ServerResponse) {
  res.writeHead(200, { "Content-Type": "text/event-stream", "Cache-Control": "no-cache" });
  const q = new URL(req.url ?? "/", "http://x").searchParams.get("last_event_id");
  const from = Number(q ?? req.headers["last-event-id"] ?? 0) || 0; // query wins, like the server
  const evs: [string, unknown][] = [["log", { level: "info", message: `${NOTE} (replaying ${id})` }]];
  for (const s of STAGES) {
    evs.push(["stage", { at: null, name: s, status: "running" }], ["stage", { at: null, name: s, status: "done" }]);
  }
  evs.push(["spend", detail(id).spend], ["done", { status: "complete", decision: "REJECT" }]);
  let i = from;
  const t = setInterval(() => {
    if (i >= evs.length) return void (clearInterval(t), res.end());
    const [type, data] = evs[i++];
    res.write(`id: ${i}\nevent: ${type}\ndata: ${JSON.stringify({ observed_at: new Date().toISOString(), ...(data as object) })}\n\n`);
  }, 350);
  req.on("close", () => clearInterval(t));
}

export function mockApi(): Plugin {
  return {
    name: "distillery-mock-api",
    configureServer(server) {
      server.middlewares.use("/api", (req, res, next) => {
        const url = new URL(req.url ?? "/", "http://x");
        const p = url.pathname.replace(/\/+$/, "");
        const m = /^\/runs\/([^/]+)(?:\/(\w+))?$/.exec(p);
        if (req.method === "POST") {
          if (p === "/runs") return send(res, 202, { run_id: "live-mock" });
          if (m?.[2] === "cancel") return send(res, 200, { status: "failed" });
          if (p === "/playground")
            return send(res, 200, { results: { teacher: unavailable, base: unavailable, student: unavailable }, cost_usd: 0, note: NOTE });
          return next();
        }
        if (p === "/health") return send(res, 200, { ok: true, mode: "replay-only", version: "mock" });
        if (p === "/config")
          return send(res, 200, {
            models: { planner: "fake-planner", teacher: "fake-teacher", triage: "fake-triage", student: "fake-student-base" },
            thresholds: { ratio_lower_bound_min: 0.85, mcnemar_alpha: 0.05, bootstrap_resamples: 10000 },
            run_cap_usd: 10,
            playground: { enabled: true, per_ip_per_hour: 10, daily_cap_usd: 1, spent_today_usd: 0 },
          });
        if (p === "/runs" || p === "/replay")
          return send(res, 200, [summary("dry-sql-tiny", "complete"), summary("live-mock", "running")]);
        if (p === "/evidence")
          return send(res, 200, JSON.parse(fs.readFileSync(path.resolve(__dirname, "../../deploy/evidence.json"), "utf8")));
        if (!m) return send(res, 404, { error: "not_found", message: "no such mock endpoint" });
        const [, id, sub] = m;
        if (!sub) return send(res, 200, detail(id));
        if (sub === "report") return send(res, 200, report(id));
        if (sub === "tree") return send(res, 200, tree(id));
        if (sub === "examples") {
          res.setHeader("X-Examples-Available", "true");
          res.setHeader("X-Examples-Cap-Per-Kind", "20");
          res.setHeader("X-Examples-Totals", JSON.stringify({ fixed: null, still_wrong: null, regressed: null }));
          return send(res, 200, examples());
        }
        if (sub === "events") return sse(id, req, res);
        if (sub === "experiments") return send(res, 200, { run_id: id, source: "unavailable", experiments: [] });
        return send(res, 404, { error: "not_found", message: "no such mock endpoint" });
      });
    },
  };
}
