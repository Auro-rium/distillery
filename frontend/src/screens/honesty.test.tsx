// Cross-screen honesty rules: errors are not empty states; labels are never silently "real".
import { act, cleanup, screen, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { DRY_RUN_TEXT } from "../components";
import { contract, contractFetch } from "../testutil/contract";
import { FakeES, json, renderApp, stubFetch } from "../testutil/render";

afterEach(() => { cleanup(); vi.unstubAllGlobals(); vi.useRealTimers(); });
const rid = contract.run.run_id;

describe("errors are not empty states", () => {
  const EMPTY = /no replay bundles|no sandbox branches|no examples|no runs|no stages/i;
  const screens: [string, string][] = [
    ["replay", "/"], ["report", `/runs/${rid}/report`], ["live", `/runs/${rid}`],
    ["tree", `/runs/${rid}/tree`], ["playground config", "/playground"], ["new run config", "/new"],
  ];
  describe.each(screens)("%s", (_n, path) => {
    it("shows the API error code and message on a 500", async () => {
      stubFetch(() => json({ error: "boom_code", message: "server exploded" }, 500));
      const { container } = renderApp(path);
      await waitFor(() => expect(screen.getAllByRole("alert").length).toBeGreaterThan(0));
      expect(container.textContent).toContain("boom_code");
      expect(container.textContent).toContain("server exploded");
      expect(container.textContent).not.toMatch(EMPTY);
      expect(container.querySelector("svg.acc-svg")).toBeNull();
    });
    it("shows an error on a network failure", async () => {
      vi.stubGlobal("fetch", vi.fn(() => Promise.reject(new TypeError("Failed to fetch"))));
      const { container } = renderApp(path);
      await waitFor(() => expect(screen.getAllByRole("alert").length).toBeGreaterThan(0));
      expect(container.textContent).toContain("network");
      expect(container.textContent).not.toMatch(EMPTY);
    });
  });
  it("examples failing inside a loaded report is an error, not 'no examples'", async () => {
    const ok = contractFetch();
    stubFetch((u, i) => (u.includes("/examples") ? json({ error: "examples_down", message: "cannot read" }, 500) : ok(u, i)));
    const { container } = renderApp(`/runs/${rid}/report`);
    await waitFor(() => screen.getByText("Examples unavailable"));
    expect(container.textContent).toContain("examples_down: cannot read");
    expect(container.textContent).not.toMatch(/no examples/i);
  });
});

type Flags = { dry_run?: boolean; recorded?: boolean; recorded_at?: string | null };
function withFlags(f: Flags) {
  const strip = <T extends object>(o: T): T => {
    const c = { ...o } as Record<string, unknown>;
    delete c.dry_run; delete c.recorded; delete c.recorded_at;
    return { ...c, ...f } as T;
  };
  return contractFetch({ run: strip(contract.run), report: strip(contract.report), replay: contract.replay.map(strip) });
}
const ROUTES: [string, string][] = [
  ["replay", "/"], ["report", `/runs/${rid}/report`], ["live", `/runs/${rid}`], ["tree", `/runs/${rid}/tree`],
];

describe.each(ROUTES)("label banner: %s", (_n, path) => {
  const banner = () => screen.getAllByRole("status").map((e) => e.textContent);
  it("shows the exact dry-run text when dry_run is true", async () => {
    stubFetch(withFlags({ dry_run: true, recorded: false, recorded_at: null }));
    renderApp(path);
    await waitFor(() => expect(banner()).toContain("DRY RUN — fake models, numbers are NOT results"));
    expect(DRY_RUN_TEXT).toBe("DRY RUN — fake models, numbers are NOT results");
  });
  it("shows the recorded label when recorded is true", async () => {
    stubFetch(withFlags({ dry_run: false, recorded: true, recorded_at: "2026-01-02T03:04:05Z" }));
    renderApp(path);
    await waitFor(() => expect(banner()).toContain("Recorded run · 2026-01-02T03:04:05Z · real Token Factory jobs"));
  });
  it("never renders missing flags as real", async () => {
    stubFetch(withFlags({}));
    renderApp(path);
    await waitFor(() => expect(banner().some((t) => /^Label unknown/.test(t ?? ""))).toBe(true));
    expect(banner().join("|")).not.toMatch(/Recorded run|Live run/);
  });
});

describe("live screen: disconnected / stale", () => {
  beforeEach(() => { FakeES.all = []; vi.stubGlobal("EventSource", FakeES); vi.useFakeTimers({ shouldAdvanceTime: true }); });
  const spend = { total_usd: 0.5, cap_usd: 10, by_model: {}, finetune_usd_estimate: null };
  const running = { ...contract.run, status: "running" };

  it("flags a silent stream after 30 s, clears on an event, and resumes with ?last_event_id=", async () => {
    stubFetch(contractFetch({ run: running }));
    renderApp(`/runs/${rid}`);
    await waitFor(() => expect(FakeES.all).toHaveLength(1));
    const a = FakeES.all[0];
    expect(a.url).not.toContain("last_event_id");
    act(() => { a.onopen!(); a.emit("spend", 1, spend); a.emit("log", 2, { level: "info", message: "hello" }); });
    expect(screen.queryByText("Disconnected / stale", { selector: "h3" })).toBeNull();
    expect(screen.getByText("Spend")).toBeTruthy();

    act(() => { vi.advanceTimersByTime(29_000); });
    expect(screen.queryByText("Disconnected / stale", { selector: "h3" })).toBeNull();
    act(() => { vi.advanceTimersByTime(1_500); });
    expect(screen.getByText("Disconnected / stale", { selector: "h3" })).toBeTruthy();
    expect(screen.getByText("Spend (last known, not current)")).toBeTruthy();

    act(() => { a.emit("log", 3, { level: "info", message: "again" }); });
    expect(screen.queryByText("Disconnected / stale", { selector: "h3" })).toBeNull();

    // disconnect: stale immediately, reconnect carries the last id in the query string
    act(() => { a.readyState = 2; a.onerror!(); });
    expect(screen.getByText("Disconnected / stale", { selector: "h3" })).toBeTruthy();
    act(() => { vi.advanceTimersByTime(600); });
    expect(FakeES.all).toHaveLength(2);
    expect(FakeES.all[1].url).toBe(`/api/runs/${rid}/events?last_event_id=3`);
    expect(screen.getByText("Disconnected / stale", { selector: "h3" })).toBeTruthy();
    act(() => { FakeES.all[1].onopen!(); });
    expect(screen.queryByText("Disconnected / stale", { selector: "h3" })).toBeNull();
  });

  it("heartbeat events keep the page live; silence still goes stale", async () => {
    stubFetch(contractFetch({ run: running }));
    renderApp(`/runs/${rid}`);
    await waitFor(() => expect(FakeES.all).toHaveLength(1));
    const a = FakeES.all[0];
    const stale = () => screen.queryByText("Disconnected / stale", { selector: "h3" });
    act(() => { a.onopen!(); a.emit("spend", 1, spend); });
    for (let i = 0; i < 6; i++) {
      act(() => { vi.advanceTimersByTime(15_000); a.emit("heartbeat", 0, { ts: i }); });
      expect(stale()).toBeNull();
    }
    expect(screen.getByText("Spend")).toBeTruthy();
    act(() => { vi.advanceTimersByTime(30_500); });
    expect(stale()).toBeTruthy();
    expect(screen.getByText("Spend (last known, not current)")).toBeTruthy();
  });

  it("shows the server's refusal (429 too_many_streams) instead of an empty page", async () => {
    const ok = contractFetch({ run: running });
    stubFetch((u, i) => (u.includes("/events") ? json({ error: "too_many_streams", message: "too many live streams" }, 429) : ok(u, i)));
    renderApp(`/runs/${rid}`);
    await waitFor(() => expect(FakeES.all).toHaveLength(1));
    act(() => { FakeES.all[0].readyState = 2; FakeES.all[0].onerror!(); });
    await waitFor(() => screen.getByText("Live stream refused"));
    expect(document.body.textContent).toContain("too_many_streams");
  });
});
