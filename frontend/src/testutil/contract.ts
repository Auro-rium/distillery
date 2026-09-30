// Real API response shapes, generated from the running FastAPI app by
// tests/test_frontend_truth.py (`npm run contract:update`). Screens are rendered from these.
import config from "./contract/config.json";
import examples from "./contract/examples.json";
import examplesHeaders from "./contract/examples_headers.json";
import health from "./contract/health.json";
import replay from "./contract/replay.json";
import report from "./contract/report.json";
import run from "./contract/run.json";
import runs from "./contract/runs.json";
import tree from "./contract/tree.json";

export const contract = { config, examples, examplesHeaders, health, replay, report, run, runs, tree };

const json = (b: unknown, status = 200, h: Record<string, string> = {}) =>
  new Response(JSON.stringify(b), { status, headers: h });

export type Overrides = Partial<Record<keyof typeof contract, unknown>>;

/** fetch stub serving the contract fixtures (optionally overridden per endpoint). */
export function contractFetch(over: Overrides = {}): (url: string, init?: RequestInit) => Promise<Response> {
  const c = { ...contract, ...over } as typeof contract;
  return (url) => {
    const u = String(url).replace(/^\/api/, "");
    const ok = (b: unknown, h: Record<string, string> = {}) => Promise.resolve(json(b, 200, h));
    if (u === "/health") return ok(c.health);
    if (u === "/config") return ok(c.config);
    if (u === "/replay") return ok(c.replay);
    if (u === "/runs") return ok(c.runs);
    if (/^\/runs\/[^/]+$/.test(u)) return ok(c.run);
    if (/\/report$/.test(u)) return ok(c.report);
    if (/\/tree$/.test(u)) return ok(c.tree);
    if (/\/examples/.test(u)) {
      const h = c.examplesHeaders;
      return ok(c.examples, {
        "X-Examples-Available": String(h.available),
        "X-Examples-Cap-Per-Kind": String(h.cap_per_kind),
        "X-Examples-Totals": JSON.stringify(h.totals),
      });
    }
    return Promise.resolve(json({ error: "not_found", message: `no stub for ${u}` }, 404));
  };
}
