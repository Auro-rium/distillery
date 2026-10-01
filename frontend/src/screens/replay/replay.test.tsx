import { render, screen, waitFor } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";
import { afterEach, describe, expect, it, vi } from "vitest";
import Replay from ".";

const run = { run_id: "r1", dry_run: true, recorded: false, recorded_at: null, status: "complete", decision: "REJECT", created_at: "t" };
// /api/runs (the live section) answers an empty list here, so these tests only see the stored list.
const stub = (body: unknown, status = 200) =>
  vi.stubGlobal("fetch", vi.fn((url: string) =>
    Promise.resolve(new Response(JSON.stringify(String(url).endsWith("/runs") ? [] : body), { status: String(url).endsWith("/runs") ? 200 : status }))));
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
  it("each card is a list item with the run id, the label badge, the decision and the date exactly as the payload gave them", async () => {
    const at = "2026-02-03T04:05:06Z";
    stub([{ ...run, created_at: at }, { ...run, run_id: "r2", dry_run: false, recorded: true, recorded_at: "2026-01-02", decision: "PROMOTE", created_at: at }]);
    const { container } = view();
    await waitFor(() => screen.getByText("r1"));
    const items = [...container.querySelectorAll<HTMLElement>(".bundles > li")];
    expect(items).toHaveLength(2);
    const [first, second] = items;
    expect(first.querySelector("h3")!.textContent).toBe("r1");
    expect([...first.querySelectorAll(".badge")].map((b) => b.textContent)).toEqual(["dry run", "complete", "REJECT"]);
    expect([...second.querySelectorAll(".badge")].map((b) => b.textContent)).toEqual(["recorded", "complete", "PROMOTE"]);
    const t = first.querySelector("time")!;
    expect(t.textContent).toBe(at);
    expect(t.getAttribute("datetime")).toBe(at);
    expect(container.querySelector(".bundles")).not.toBeNull();
  });
  it("a card whose payload has no date or decision says so in words, and shows no decision badge it was not given", async () => {
    stub([{ ...run, created_at: null, decision: null, status: "running" }]);
    const { container } = view();
    await waitFor(() => screen.getByText("r1"));
    const item = container.querySelector<HTMLElement>(".bundles > li")!;
    expect(item.textContent).toContain("not measured");
    expect(item.querySelector("time")).toBeNull();
    expect([...item.querySelectorAll(".badge")].map((b) => b.textContent)).toEqual(["dry run", "running"]);
    expect(item.textContent).not.toContain("PROMOTE");
    expect(item.textContent).not.toContain("REJECT");
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
