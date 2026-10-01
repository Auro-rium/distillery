import { act, cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { MemoryRouter, Route, Routes } from "react-router-dom";
import { afterEach, describe, expect, it, vi } from "vitest";
import Replay from ".";

const base = { dry_run: true, recorded: false, recorded_at: null, status: "complete", decision: "PROMOTE", created_at: "2026-01-01T00:00:00Z" };
const R = (o: object) => ({ ...base, ...o });
const j = (b: unknown, status = 200) => new Response(JSON.stringify(b), { status });
const isPost = (i?: RequestInit) => i?.method === "POST";

/** GET /replay is always empty; GET /runs answers `runs()`; POST /runs answers `post(...)`. */
function stub(runs: () => Response, post: () => Promise<Response> | Response = () => j({ run_id: "demo-1" }, 202)) {
  const fn = vi.fn((url: string, init?: RequestInit) => {
    const u = String(url);
    if (isPost(init)) return Promise.resolve(post());
    return Promise.resolve(u.endsWith("/runs") ? runs() : j([]));
  });
  vi.stubGlobal("fetch", fn);
  return fn;
}
const view = () => render(
  <MemoryRouter><Routes><Route path="/" element={<Replay />} /><Route path="/runs/:id" element={<p>run page</p>} /></Routes></MemoryRouter>,
);
const runsCalls = (fn: ReturnType<typeof stub>) => fn.mock.calls.filter(([u, i]) => String(u).endsWith("/runs") && !isPost(i)).length;
const demoBtn = () => screen.getByRole("button", { name: /Watch a live demo run/ }) as HTMLButtonElement;

afterEach(() => { cleanup(); vi.useRealTimers(); vi.unstubAllGlobals(); });

describe("live runs section", () => {
  it("lists running first with a Watch live link, excludes recorded runs, and takes labels from the payload", async () => {
    stub(() => j([
      R({ run_id: "old", created_at: "2026-01-01T00:00:00Z" }),
      R({ run_id: "rec", recorded: true, recorded_at: "2026-01-02" }),
      R({ run_id: "new", status: "running", decision: null, created_at: "2025-12-01T00:00:00Z", dry_run: false }),
    ]));
    view();
    await waitFor(() => screen.getByText("new"));
    expect(screen.getByRole("heading", { name: "Live and recent runs on this server" })).toBeTruthy();
    expect(screen.queryByText("rec")).toBeNull();
    const items = [...document.querySelectorAll<HTMLElement>("ul[aria-label='Live and recent runs'] > li")];
    expect(items.map((i) => i.querySelector("h3")!.textContent)).toEqual(["new", "old"]);
    expect(screen.getByText("Watch live").getAttribute("href")).toBe("/runs/new");
    expect([...items[0].querySelectorAll(".badge")].map((b) => b.textContent)).not.toContain("dry run");
    expect([...items[1].querySelectorAll(".badge")].map((b) => b.textContent)).toEqual(["dry run", "complete", "PROMOTE"]);
    expect(screen.getByText("Report").getAttribute("href")).toBe("/runs/old/report");
  });
  it("says so plainly when nothing is active", async () => {
    stub(() => j([R({ run_id: "rec", recorded: true })]));
    view();
    await waitFor(() => screen.getByText("No runs are active right now."));
  });
  it("shows the API error instead of hiding it", async () => {
    stub(() => j({ error: "boom", message: "server down" }, 500));
    view();
    await waitFor(() => screen.getByText("Could not load live runs"));
    expect(document.body.textContent).toContain("server down");
  });
  it("polls every 3 s while a run is running, then every 15 s, and stops on unmount", async () => {
    vi.useFakeTimers();
    let running = true;
    const fn = stub(() => j([R({ run_id: "x", status: running ? "running" : "complete" })]));
    const { unmount } = view();
    await act(async () => { await vi.advanceTimersByTimeAsync(0); });
    expect(runsCalls(fn)).toBe(1);
    await act(async () => { await vi.advanceTimersByTimeAsync(3000); });
    expect(runsCalls(fn)).toBe(2);
    running = false;
    await act(async () => { await vi.advanceTimersByTimeAsync(3000); });
    expect(runsCalls(fn)).toBe(3);
    await act(async () => { await vi.advanceTimersByTimeAsync(6000); });
    expect(runsCalls(fn)).toBe(3); // idle: not every 3 s any more
    await act(async () => { await vi.advanceTimersByTimeAsync(9000); });
    expect(runsCalls(fn)).toBe(4);
    unmount();
    await act(async () => { await vi.advanceTimersByTimeAsync(60000); });
    expect(runsCalls(fn)).toBe(4);
  });
});

describe("demo run button", () => {
  it("creates a dry tiny sql run, is disabled while in flight, and navigates to it", async () => {
    let release: (r: Response) => void = () => undefined;
    const fn = stub(() => j([]), () => new Promise<Response>((r) => { release = r; }));
    view();
    expect(screen.getByText("Fake models, no spend, results are NOT real (dry run)")).toBeTruthy();
    fireEvent.click(demoBtn());
    await waitFor(() => expect(demoBtn().disabled).toBe(true));
    const post = fn.mock.calls.find(([, i]) => isPost(i))!;
    expect(String(post[0])).toMatch(/\/api\/runs$/);
    expect(JSON.parse(String(post[1]!.body))).toEqual({ pack: "sql", scale: "tiny", dry_run: true });
    await act(async () => { release(j({ run_id: "demo-1" }, 202)); });
    await waitFor(() => screen.getByText("run page"));
  });
  it.each([[401, "unauthorized"], [403, "forbidden"], [429, "rate_limited"]])("shows the server's message for HTTP %i", async (status, code) => {
    stub(() => j([]), () => j({ error: code, message: `server says no ${status}` }, status));
    view();
    fireEvent.click(demoBtn());
    await waitFor(() => screen.getByText("Could not start the demo run"));
    expect(document.body.textContent).toContain(`server says no ${status}`);
    expect(demoBtn().disabled).toBe(false);
  });
});
