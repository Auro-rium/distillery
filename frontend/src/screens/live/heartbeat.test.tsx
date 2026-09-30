// The live indicator and all "live" motion must follow the real stream state, never a timer.
import { act, cleanup, render, screen, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { contract, contractFetch } from "../../testutil/contract";
import { FakeES, renderApp, stubFetch } from "../../testutil/render";
import { Heartbeat } from "./Heartbeat";
import { connectionOf } from "./state";

afterEach(() => { cleanup(); vi.unstubAllGlobals(); vi.useRealTimers(); });
const rid = contract.run.run_id;

describe("connectionOf", () => {
  it("only a healthy open stream is connected", () => {
    expect(connectionOf("open", false)).toBe("connected");
    expect(connectionOf("open", true)).toBe("stale");
    expect(connectionOf("connecting", false)).toBe("connecting");
    expect(connectionOf("connecting", true)).toBe("stale");
    expect(connectionOf("reconnecting", true)).toBe("disconnected");
    expect(connectionOf("closed", false)).toBe("disconnected");
  });
});

describe("Heartbeat", () => {
  it("shows the state as text and data-conn, and ripples only when connected with real events", () => {
    const { container, rerender } = render(<Heartbeat conn="connected" events={0} />);
    const hb = () => container.querySelector(".hb")!;
    expect(hb().getAttribute("data-conn")).toBe("connected");
    expect(hb().textContent).toContain("live");
    expect(container.querySelector(".hb-ping")).toBeNull(); // nothing arrived yet: no ripple
    rerender(<Heartbeat conn="connected" events={1} />);
    expect(container.querySelector(".hb-ping")).not.toBeNull();
    rerender(<Heartbeat conn="stale" events={1} />);
    expect(hb().getAttribute("data-conn")).toBe("stale");
    expect(hb().textContent).toContain("Disconnected / stale");
    expect(container.querySelector(".hb-ping")).toBeNull();
    rerender(<Heartbeat conn="disconnected" events={9} />);
    expect(hb().getAttribute("data-conn")).toBe("disconnected");
    expect(container.querySelector(".hb-ping")).toBeNull();
  });
});

describe("live screen heartbeat and motion follow the stream", () => {
  beforeEach(() => { FakeES.all = []; vi.stubGlobal("EventSource", FakeES); vi.useFakeTimers({ shouldAdvanceTime: true }); });
  const running = {
    ...contract.run, status: "running",
    stages: [{ name: "verifier_selftest", status: "done", started_at: null, ended_at: null }, { name: "split", status: "running", started_at: null, ended_at: null }],
  };
  const dot = (c: HTMLElement) => c.querySelector(".hb")!.getAttribute("data-conn");

  it("connecting -> connected -> ripple on event -> stale after silence -> disconnected on error -> connected", async () => {
    stubFetch(contractFetch({ run: running }));
    const { container } = renderApp(`/runs/${rid}`);
    await waitFor(() => expect(FakeES.all).toHaveLength(1));
    const es = FakeES.all[0];
    expect(dot(container)).toBe("connecting");
    expect(container.querySelector(".timeline")!.getAttribute("data-live")).toBe("false");

    act(() => { es.onopen!(); });
    expect(dot(container)).toBe("connected");
    expect(container.querySelector(".timeline")!.getAttribute("data-live")).toBe("true");
    expect(container.querySelector(".hb-ping")).toBeNull();

    act(() => { es.emit("log", 1, { level: "info", message: "hello" }); });
    expect(container.querySelector(".hb-ping")).not.toBeNull();
    expect(container.querySelector(".log-new")!.textContent).toContain("hello");

    act(() => { vi.advanceTimersByTime(30_500); });
    expect(dot(container)).toBe("stale");
    expect(container.querySelector(".hb-ping")).toBeNull();
    expect(container.querySelector(".timeline")!.getAttribute("data-live")).toBe("false"); // no pulsing over last-known data

    act(() => { es.readyState = 2; es.onerror!(); });
    expect(dot(container)).toBe("disconnected");
    expect(screen.getByText("Stages (last known, not current)")).toBeTruthy();

    act(() => { vi.advanceTimersByTime(600); });
    act(() => { FakeES.all[1].onopen!(); });
    expect(dot(container)).toBe("connected");
  });
});
