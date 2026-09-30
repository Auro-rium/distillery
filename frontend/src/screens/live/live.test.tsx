import { act, cleanup, renderHook } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { useSSE } from "../../api/useSSE";
import type { RunDetail, RunEvent } from "../../api/types";
import { applyEvents } from "./state";

class FakeES {
  static all: FakeES[] = [];
  static CLOSED = 2;
  readyState = 1;
  onopen: (() => void) | null = null;
  onerror: (() => void) | null = null;
  ls = new Map<string, (e: MessageEvent) => void>();
  constructor(public url: string) { FakeES.all.push(this); }
  addEventListener(t: string, f: (e: MessageEvent) => void) { this.ls.set(t, f); }
  close() { this.readyState = 2; }
  emit(type: string, id: number, data: object) {
    this.ls.get(type)!(new MessageEvent(type, { data: JSON.stringify({ observed_at: "t", ...data }), lastEventId: String(id) }));
  }
}

beforeEach(() => { vi.useFakeTimers(); FakeES.all = []; vi.stubGlobal("EventSource", FakeES); });
afterEach(() => { cleanup(); vi.useRealTimers(); vi.unstubAllGlobals(); });

describe("useSSE reconnect", () => {
  it("recreates a closed stream with backoff and drops replayed ids", () => {
    const { result } = renderHook(() => useSSE("/api/runs/x/events"));
    const a = FakeES.all[0];
    act(() => { a.onopen!(); a.emit("log", 1, { level: "info", message: "one" }); a.emit("log", 2, { level: "info", message: "two" }); });
    expect(result.current.state).toBe("open");
    act(() => { a.readyState = 2; a.onerror!(); });
    expect(result.current.state).toBe("reconnecting");
    expect(FakeES.all).toHaveLength(1);
    act(() => { vi.advanceTimersByTime(500); });
    expect(FakeES.all).toHaveLength(2);
    const b = FakeES.all[1];
    act(() => { b.onopen!(); b.emit("log", 1, { level: "info", message: "one" }); b.emit("log", 2, { level: "info", message: "two" }); b.emit("log", 3, { level: "info", message: "three" }); });
    expect(result.current.events.map((e) => e.id)).toEqual([1, 2, 3]);
    act(() => { b.emit("done", 4, { status: "complete", decision: "REJECT" }); });
    expect(result.current.state).toBe("closed");
  });
  it("does nothing without a url", () => {
    const { result } = renderHook(() => useSSE(null));
    expect(FakeES.all).toHaveLength(0);
    expect(result.current.state).toBe("closed");
  });
});

describe("applyEvents", () => {
  const base = {
    run_id: "x", dry_run: true, status: "running", stages: [{ name: "a", status: "running", started_at: null, ended_at: null }],
    spend: { total_usd: 0, cap_usd: 1, by_model: {}, finetune_usd_estimate: null },
  } as unknown as RunDetail;
  it("updates stages, spend, logs and done", () => {
    const evs: RunEvent[] = [
      { id: 1, type: "stage", data: { observed_at: "o1", at: "2026-01-01T00:00:01Z", name: "a", status: "done" } },
      { id: 2, type: "stage", data: { observed_at: "o2", at: null, name: "b", status: "running" } },
      { id: 3, type: "spend", data: { observed_at: "t", total_usd: 0.5, cap_usd: 1, by_model: {}, finetune_usd_estimate: null } },
      { id: 4, type: "log", data: { observed_at: "t", level: "info", message: "hi" } },
      { id: 5, type: "done", data: { observed_at: "t", status: "complete", decision: "PROMOTE" } },
    ];
    const v = applyEvents(base, evs);
    expect(v.run.stages.map((s) => [s.name, s.status])).toEqual([["a", "done"], ["b", "running"]]);
    expect(v.run.spend.total_usd).toBe(0.5);
    expect("observed_at" in v.run.spend).toBe(false);
    expect(v.run.stages[0].ended_at).toBe("2026-01-01T00:00:01Z"); // event time `at`, never observed_at
    expect(v.run.stages[1].started_at).toBeNull();
    expect(v.logs[0].message).toBe("hi");
    expect([v.done, v.status, v.decision]).toEqual([true, "complete", "PROMOTE"]);
    expect(base.stages[0].status).toBe("running"); // input not mutated
  });
});
