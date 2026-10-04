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
    await check("/", contract.replay[0].run_id, [contract.replay, contract.runs]);
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
  it("playground", async () => {
    stubFetch(contractFetch());
    await check("/playground", /daily budget spent/, [contract.config], [
      { pattern: /\d+ \/ 500/g, why: "character counter against the 500-character input limit (input length / UI constant)" },
    ]);
  });
  it("still holds when the payload lacks values (nothing invented for absent data)", async () => {
    const r = { ...contract.run, sandbox: { operations: null, concurrency_peak: null } };
    stubFetch(contractFetch({ run: r }));
    const text = await check(`/runs/${id}`, "Spend", [r]);
    expect(text).toContain("not measured");
  });
});
