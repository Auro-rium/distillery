import { cleanup, screen, within } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { contract, contractFetch } from "../../testutil/contract";
import { json, renderApp, stubFetch } from "../../testutil/render";
import { ledgerLabel, ledgerRows } from "./LedgerPanel";

afterEach(() => { cleanup(); vi.unstubAllGlobals(); });
const rid = contract.run.run_id;

const rows = [
  { name: "finetune_job_started", data: { job_id: "ftjob-aaa", round: 1 }, created_at: "2026-10-04T16:49:05+00:00" },
  { name: "finetune_job_closed", data: { job_id: "ftjob-aaa", outcome: "aborted" }, created_at: "2026-10-04T17:06:34+00:00" },
  { name: "finetune_job_adopted", data: { job_id: "ftjob-aaa" }, created_at: "2026-10-04T17:42:10+00:00" },
  { name: "serving_image", data: { image: "img-1", kind: "deps" }, created_at: "2026-10-04T17:44:11+00:00" },
];
const withLedger = (body: unknown, status = 200) => {
  const ok = contractFetch();
  return (u: string, i?: RequestInit) => (u.endsWith("/experiments") ? Promise.resolve(json(body, status)) : ok(u, i));
};

describe("job ledger", () => {
  it("labels rows from the API's own names and carries job id and detail from the payload", () => {
    expect(ledgerLabel("finetune_job_adopted")).toBe("adopted");
    expect(ledgerLabel("serving_image")).toBe("serving image");
    const r = ledgerRows(rows);
    expect(r.map((x) => x.label)).toEqual(["started", "closed", "adopted", "serving image"]);
    expect(r[1].detail).toBe("outcome aborted");
    expect(r[0].jobId).toBe("ftjob-aaa");
  });

  it("shows the kill -> adopt story in stored order, with its source", async () => {
    stubFetch(withLedger({ run_id: rid, source: "bundle", experiments: rows }));
    renderApp(`/runs/${rid}`);
    const panel = await screen.findByRole("region", { name: "Job ledger" });
    await within(panel).findByRole("list", { name: "Fine-tune jobs" });
    const kinds = [...panel.querySelectorAll(".ledger-kind")].map((e) => e.textContent);
    expect(kinds).toEqual(["started", "closed", "adopted", "serving image"]);
    expect(panel.textContent).toContain("bundle");
    expect(panel.textContent).toContain("ftjob-aaa");
  });

  it("says plainly when a recording has no ledger, and when the request fails (never a silent empty panel)", async () => {
    stubFetch(withLedger({ run_id: rid, source: "unavailable", experiments: [] }));
    renderApp(`/runs/${rid}`);
    expect((await screen.findByRole("region", { name: "Job ledger" })).textContent).toContain("carries no job ledger");
    cleanup();
    stubFetch(withLedger({ error: "boom", message: "kaput" }, 500));
    renderApp(`/runs/${rid}`);
    const p = await screen.findByRole("region", { name: "Job ledger" });
    await vi.waitFor(() => expect(p.textContent).toContain("Job ledger unavailable"));
  });
});
