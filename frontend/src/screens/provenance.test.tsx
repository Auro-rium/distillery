import { cleanup, screen, waitFor } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { contract, contractFetch } from "../testutil/contract";
import { unexplained, visibleText, type Allow } from "../testutil/provenance";
import { renderApp, stubFetch } from "../testutil/render";

afterEach(() => { cleanup(); vi.unstubAllGlobals(); });

const id = contract.run.run_id;
const AXIS: Allow = { pattern: /(?<![\d.])(0|50|100)%(?=\s|$)/g, why: "fixed axis tick labels of the accuracy chart (0, 50, 100 percent)" };
const SHA: Allow = { pattern: /adapter [0-9a-f]{12}(?![0-9a-f])/g, why: "first 12 hex chars of adapter_sha256, shortened for display (full hash is in the title attribute)" };
const CARD: Allow = { pattern: /\bcost \/ 1k tasks/g, why: "'1k' is part of the metric name 'cost per 1k tasks'" };
const SHOWN: Allow = { pattern: /(capped list, \d+ shown|showing \d+ of \d+)/g, why: "count of items returned in the examples list (a list length, not a payload value)" };

const RUNG: Allow = { pattern: /\b[A-Z]\d\b/g, why: "two-character proof ids from the evidence payload (W2, A5, N1 ...): payload strings shorter than three characters are not matched verbatim by the checker" };

async function check(path: string, ready: RegExp | string, payloads: unknown[], allow: Allow[] = []) {
  const { container } = renderApp(path);
  await waitFor(() => expect(screen.getAllByText(ready).length).toBeGreaterThan(0), { timeout: 3000 });
  await new Promise((r) => setTimeout(r, 50)); // let secondary fetches (examples) settle
  const text = visibleText(container);
  expect(text.length).toBeGreaterThan(20);
  expect(unexplained(text, payloads, allow)).toEqual([]);
  return text;
}

describe("render provenance: numbers on screen come from the contract payloads", () => {
  it("checker itself rejects an invented number", () => {
    expect(unexplained("accuracy 71.3% n=20", [{ acc: 0.7, n: 20 }])).toEqual(["71.3%"]);
    expect(unexplained("accuracy 70.0% n=20", [{ acc: 0.7, n: 20 }])).toEqual([]);
    expect(unexplained("n=3", [{ n: 2.7 }])).toEqual(["3"]);
  });
  it("replay", async () => {
    stubFetch(contractFetch());
    await check("/mission", contract.replay[0].run_id, [contract.replay, contract.runs, contract.evidence], [RUNG]);
  });
  it("overview", async () => {
    stubFetch(contractFetch());
    await check("/", /This server, right now/, [contract.config, contract.gatewayStatus, contract.gatewayDrift, contract.evidence], [
      RUNG,
      { pattern: /\/v1\/?/g, why: "the API path segment in the base-URL example" },
      { pattern: /\/\/[a-z0-9.-]+:\d+/g, why: "the page's own origin (window.location) in the base-URL example" },
      { pattern: /(?<![\d.])[1-4](?=\s*\n|Familiar|Student|Router|Drift)/g, why: "step numbers of the request diagram" },
      { pattern: /1 − /g, why: "'1 −' in 'confidence 1 − delta', the complement of the payload's delta" },
      { pattern: /W-C\d|W-\$|W-E2E/g, why: "claim ids from the evidence payload" },
    ]);
  });
  it("report", async () => {
    stubFetch(contractFetch());
    await check(`/runs/${id}/report`, "Gate statistics", [contract.report, contract.examples, contract.examplesHeaders], [AXIS, CARD, SHOWN, SHA]);
  });
  it("live run", async () => {
    stubFetch(contractFetch());
    await check(`/runs/${id}`, "Spend", [contract.run, contract.report]);
  });
  it("tree", async () => {
    stubFetch(contractFetch());
    await check(`/runs/${id}/tree`, /round 1/, [contract.tree, contract.run]);
  });
  it("new run", async () => {
    stubFetch(contractFetch());
    await check("/new", /server default/, [contract.config]);
  });
  it("still holds when the payload lacks values (nothing invented for absent data)", async () => {
    const r = { ...contract.run, sandbox: { operations: null, concurrency_peak: null } };
    stubFetch(contractFetch({ run: r }));
    const text = await check(`/runs/${id}`, "Spend", [r]);
    expect(text).toContain("not measured");
  });
});
