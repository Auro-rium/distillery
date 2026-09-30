import { act, cleanup, render, renderHook } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { fmtInt, fmtPercent, fmtUsd } from "../api/format";
import { CountUp } from "./CountUp";
import { frac } from "./geometry";
import { useReducedMotion } from "./useReducedMotion";

type Listener = () => void;
/** matchMedia stand-in whose `reduce` answer can be flipped at run time. */
function stubMotion(reduce: boolean) {
  const listeners = new Set<Listener>();
  const state = { reduce };
  vi.stubGlobal("matchMedia", (q: string) => ({
    matches: q.includes("reduce") ? state.reduce : false,
    addEventListener: (_: string, f: Listener) => listeners.add(f),
    removeEventListener: (_: string, f: Listener) => listeners.delete(f),
  }));
  return { set(v: boolean) { state.reduce = v; listeners.forEach((f) => f()); } };
}

/** Manual requestAnimationFrame: the test decides the clock. */
function stubRaf() {
  let queue: FrameRequestCallback[] = [];
  vi.stubGlobal("requestAnimationFrame", (f: FrameRequestCallback) => { queue.push(f); return queue.length; });
  vi.stubGlobal("cancelAnimationFrame", () => { queue = []; });
  return {
    frame(t: number) { const q = queue; queue = []; q.forEach((f) => f(t)); },
    pending: () => queue.length,
  };
}

beforeEach(() => vi.stubGlobal("IntersectionObserver", undefined));
afterEach(() => { cleanup(); vi.unstubAllGlobals(); });

describe("useReducedMotion", () => {
  it("is true when matchMedia does not exist (jsdom)", () => {
    expect(renderHook(() => useReducedMotion()).result.current).toBe(true);
  });
  it("follows the media query, including live changes", () => {
    const m = stubMotion(false);
    const { result } = renderHook(() => useReducedMotion());
    expect(result.current).toBe(false);
    act(() => m.set(true));
    expect(result.current).toBe(true);
  });
});

describe("CountUp", () => {
  it("reduced motion: renders the final value immediately, no frames scheduled", () => {
    stubMotion(true);
    const raf = stubRaf();
    const { container, rerender } = render(<CountUp value={0.1234567} format={fmtUsd} />);
    expect(container.textContent).toBe(fmtUsd(0.1234567));
    rerender(<CountUp value={2.5} format={fmtUsd} />);
    expect(container.textContent).toBe("$2.5000");
    expect(raf.pending()).toBe(0);
  });
  it("animated: eases between real values and ends on the exact payload value", () => {
    stubMotion(false);
    const raf = stubRaf();
    const { container, rerender } = render(<CountUp value={1} format={fmtInt} />);
    expect(container.textContent).toBe("1"); // first appearance shows the real value, not a count from zero
    rerender(<CountUp value={10} format={fmtInt} />);
    raf.frame(1000); // t0
    raf.frame(1350); // mid-way
    const mid = Number(container.textContent);
    expect(mid).toBeGreaterThan(1);
    expect(mid).toBeLessThan(10);
    raf.frame(5000); // past the end
    expect(container.textContent).toBe("10");
    expect(raf.pending()).toBe(0);
  });
  it("animated: the last frame is format(payload), even for values format() rounds", () => {
    stubMotion(false);
    const raf = stubRaf();
    const { container, rerender } = render(<CountUp value={0.5} format={fmtPercent} />);
    rerender(<CountUp value={0.7043} format={fmtPercent} />);
    raf.frame(0); raf.frame(100); raf.frame(99999);
    expect(container.textContent).toBe(fmtPercent(0.7043));
  });
  it("retargeting mid-flight still ends exactly on the newest value", () => {
    stubMotion(false);
    const raf = stubRaf();
    const { container, rerender } = render(<CountUp value={0} format={fmtUsd} />);
    rerender(<CountUp value={3} format={fmtUsd} />);
    raf.frame(0); raf.frame(300);
    rerender(<CountUp value={7.25} format={fmtUsd} />);
    raf.frame(400); raf.frame(90000);
    expect(container.textContent).toBe("$7.2500");
  });
  it("absent values show 'not measured' and never animate", () => {
    stubMotion(false);
    const raf = stubRaf();
    const { container, rerender } = render(<CountUp value={null} format={fmtUsd} />);
    expect(container.textContent).toBe("not measured");
    rerender(<CountUp value={Number.NaN} format={fmtUsd} />);
    expect(container.textContent).toBe("not measured");
    expect(raf.pending()).toBe(0);
  });
  it("fromZero counts up on first appearance and still finishes on the payload value", () => {
    stubMotion(false);
    const raf = stubRaf();
    const { container } = render(<CountUp value={0.7} format={fmtPercent} fromZero />);
    raf.frame(0);
    expect(container.textContent).toBe(fmtPercent(0));
    raf.frame(400);
    raf.frame(99999);
    expect(container.textContent).toBe("70.0%");
  });
});

describe("frac (bar geometry)", () => {
  it("is null for absent or unusable numbers and clamps to 0..1", () => {
    expect(frac(null, 10)).toBeNull();
    expect(frac(1, null)).toBeNull();
    expect(frac(1, 0)).toBeNull();
    expect(frac(Number.NaN, 1)).toBeNull();
    expect(frac(20, 10)).toBe(1);
    expect(frac(-1, 10)).toBe(0);
    expect(frac(2.5, 10)).toBe(0.25);
  });
});
