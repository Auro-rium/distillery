import { cleanup, render, screen, waitFor } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { json, stubFetch } from "../testutil/render";
import { pickFeatured, useFeaturedRun } from "./useFeaturedRun";

afterEach(() => { cleanup(); vi.unstubAllGlobals(); });

const run = (run_id: string, over: Record<string, unknown> = {}) => ({
  run_id, dry_run: false, recorded: true, recorded_at: null, status: "complete", decision: "PROMOTE", created_at: null, ...over,
});

describe("pickFeatured", () => {
  it("prefers the newest recorded non-dry PROMOTE run over a newer REJECT, whatever the API order", () => {
    const rows = [
      run("reject-newest", { decision: "REJECT", recorded_at: "2026-05-01T00:00:00Z" }),
      run("promote-old", { recorded_at: "2026-01-01T00:00:00Z" }),
      run("promote-new", { recorded_at: "2026-03-01T00:00:00Z" }),
    ];
    expect(pickFeatured(rows)?.run_id).toBe("promote-new");
    expect(pickFeatured([...rows].reverse())?.run_id).toBe("promote-new");
  });
  it("a newer dry PROMOTE never beats a recorded real PROMOTE", () => {
    const rows = [run("dry-promote", { dry_run: true, recorded_at: "2026-09-01T00:00:00Z" }), run("real-promote", { recorded_at: "2026-01-01T00:00:00Z" })];
    expect(pickFeatured(rows)?.run_id).toBe("real-promote");
  });
  it("falls back to the newest recorded run (any label) when no recorded non-dry PROMOTE exists", () => {
    const rows = [run("a", { dry_run: true, decision: null, recorded_at: "2026-01-01T00:00:00Z" }), run("b", { dry_run: true, decision: "REJECT", recorded_at: "2026-02-01T00:00:00Z" })];
    expect(pickFeatured(rows)?.run_id).toBe("b");
  });
  it("uses created_at when recorded_at is absent", () => {
    const rows = [run("x", { created_at: "2026-01-01T00:00:00Z" }), run("y", { created_at: "2026-04-01T00:00:00Z" })];
    expect(pickFeatured(rows)?.run_id).toBe("y");
  });
  it("features nothing for unrecorded runs, an empty list, or a non-array body", () => {
    expect(pickFeatured([run("live", { recorded: false })])).toBeNull();
    expect(pickFeatured([])).toBeNull();
    expect(pickFeatured({ error: "x" })).toBeNull();
    expect(pickFeatured([null, 3, { run_id: "" }])).toBeNull();
  });
});

function Probe() {
  const f = useFeaturedRun();
  return <p>{f.loading ? "loading" : f.run ? `featured ${f.run.run_id}` : "none"}</p>;
}

describe("useFeaturedRun", () => {
  it("reads /api/replay and returns the featured run", async () => {
    const f = stubFetch(() => json([run("p4", { recorded_at: "2026-02-01T00:00:00Z" })]));
    render(<Probe />);
    expect(screen.getByText("loading")).toBeTruthy();
    expect(await screen.findByText("featured p4")).toBeTruthy();
    expect(f.mock.calls.every(([u]) => String(u).endsWith("/api/replay"))).toBe(true);
  });
  it("shares one in-flight request between concurrent callers", async () => {
    const f = stubFetch(() => json([run("p4")]));
    render(<><Probe /><Probe /></>);
    await waitFor(() => expect(screen.getAllByText("featured p4")).toHaveLength(2));
    expect(f).toHaveBeenCalledTimes(1);
  });
  it("is null (not an error state) when the API fails", async () => {
    stubFetch(() => json({ error: "boom", message: "x" }, 500));
    render(<Probe />);
    expect(await screen.findByText("none")).toBeTruthy();
  });
});
