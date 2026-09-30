import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";
import { afterEach, describe, expect, it, vi } from "vitest";
import Playground, { Pane, RowsPreview } from ".";

const NA = { available: false, reason: "student serving path not deployed", sql: null, verified: null, rows_preview: null, error: null };
const OK = { available: true, reason: null, sql: "SELECT 1", verified: true, rows_preview: [{ a: 1 }, { a: null }], error: null };
const cfg = { playground: { enabled: true, per_ip_per_hour: 10, daily_cap_usd: 1, spent_today_usd: 0.25 } };

function stub(post: () => Response) {
  vi.stubGlobal("fetch", vi.fn((url: string) =>
    Promise.resolve(url.endsWith("/config") ? new Response(JSON.stringify(cfg)) : post())));
}
async function ask() {
  render(<MemoryRouter><Playground /></MemoryRouter>);
  await waitFor(() => screen.getByText(/daily budget spent \$0.25 of \$1.00/));
  fireEvent.change(screen.getByLabelText("Question"), { target: { value: "how many?" } });
  fireEvent.click(screen.getByText("Ask"));
}
afterEach(() => vi.unstubAllGlobals());

describe("Playground", () => {
  it("shows three panes honestly", async () => {
    stub(() => new Response(JSON.stringify({ results: { teacher: OK, base: NA, student: NA }, cost_usd: 0.001, note: "n" })));
    await ask();
    await waitFor(() => screen.getByText("SELECT 1"));
    expect(screen.getAllByText("student serving path not deployed").length).toBe(2);
    expect(screen.getByText("verified: matches the known answer")).toBeTruthy();
    expect(screen.getByText("NULL")).toBeTruthy();
  });
  it("demo budget exhausted", async () => {
    stub(() => new Response(JSON.stringify({ error: "demo_budget_exhausted", message: "m" }), { status: 503 }));
    await ask();
    await waitFor(() => screen.getByText("Demo budget exhausted"));
    expect(screen.getByText("See replay").getAttribute("href")).toBe("/");
  });
  it("rate limit", async () => {
    stub(() => new Response(JSON.stringify({ error: "rate_limited", message: "m" }), { status: 429, headers: { "Retry-After": "42" } }));
    await ask();
    await waitFor(() => screen.getByText("Try again in 42 seconds."));
  });
  it("unverified + array rows + no rows", () => {
    render(<Pane name="Base" r={{ ...OK, verified: null, rows_preview: [] }} />);
    expect(screen.getByText(/not verified/)).toBeTruthy();
    expect(screen.getByText("Query returned no rows.")).toBeTruthy();
    render(<RowsPreview rows={[[1, "x"]]} />);
    expect(screen.getByText("col 2")).toBeTruthy();
  });
});
