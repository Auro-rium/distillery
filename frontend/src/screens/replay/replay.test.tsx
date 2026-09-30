import { render, screen, waitFor } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";
import { afterEach, describe, expect, it, vi } from "vitest";
import Replay from ".";

const run = { run_id: "r1", dry_run: true, recorded: false, recorded_at: null, status: "complete", decision: "REJECT", created_at: "t" };
const stub = (body: unknown, status = 200) =>
  vi.stubGlobal("fetch", vi.fn(() => Promise.resolve(new Response(JSON.stringify(body), { status }))));
const view = () => render(<MemoryRouter><Replay /></MemoryRouter>);
afterEach(() => vi.unstubAllGlobals());

describe("Replay", () => {
  it("lists bundles with banner and links", async () => {
    stub([run, { ...run, run_id: "r2", dry_run: false, recorded: true, recorded_at: "2026-01-02", status: "running", decision: null }]);
    view();
    await waitFor(() => screen.getByText("r1"));
    expect(screen.getByText("DRY RUN — fake models, numbers are NOT results")).toBeTruthy();
    expect(screen.getByText("Recorded run · 2026-01-02 · real Token Factory jobs")).toBeTruthy();
    expect(screen.getByText("Report").getAttribute("href")).toBe("/runs/r1/report");
    expect(screen.getByText("Experiment tree").getAttribute("href")).toBe("/runs/r1/tree");
    expect(screen.getByText("Watch run (no report yet)").getAttribute("href")).toBe("/runs/r2");
  });
  it("empty state", async () => {
    stub([]);
    view();
    await waitFor(() => screen.getByText("No replay bundles available"));
  });
  it("error state", async () => {
    stub({ error: "boom", message: "server down" }, 500);
    view();
    await waitFor(() => screen.getByRole("alert"));
    expect(screen.getByRole("alert").textContent).toContain("boom");
    expect(screen.getByRole("alert").textContent).toContain("server down");
  });
});
