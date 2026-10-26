import { act, cleanup, fireEvent, render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { Flow } from "./Flow";

function stubMotion(reduce: boolean) {
  vi.stubGlobal("matchMedia", (q: string) => ({
    matches: q.includes("reduce") ? reduce : false,
    addEventListener: () => undefined,
    removeEventListener: () => undefined,
  }));
}
/** Manual requestAnimationFrame: the test decides the clock. */
function stubRaf() {
  let queue: FrameRequestCallback[] = [];
  vi.stubGlobal("requestAnimationFrame", (f: FrameRequestCallback) => { queue.push(f); return queue.length; });
  vi.stubGlobal("cancelAnimationFrame", () => { queue = []; });
  return { frame(t: number) { const q = queue; queue = []; act(() => { q.forEach((f) => f(t)); }); } };
}

afterEach(() => { cleanup(); vi.unstubAllGlobals(); });

const slots = () => [...document.querySelectorAll<HTMLElement>(".h-step-slot")];
const box = (i: number) => slots()[i].querySelector<HTMLElement>(".h-step")!;
const FIRST = "The question is embedded and compared with what the student was certified on.";

describe("How one request is served", () => {
  it("is four boxes in one list, each showing its number and title, closed at rest", () => {
    stubMotion(false);
    render(<Flow />);
    expect(slots()).toHaveLength(4);
    expect(screen.getAllByRole("heading", { level: 3 }).map((h) => h.textContent)).toEqual(["Familiar?", "Student drafts", "Router decides", "Drift guard"]);
    slots().forEach((_, i) => expect(box(i).dataset.open).toBe("false"));
    // the text is in the page for assistive technology even while it is visually closed
    expect(screen.getByText(new RegExp(FIRST))).toBeTruthy();
    expect(document.querySelectorAll(".h-bit")).toHaveLength(0);
  });

  it("opens a box on hover, decodes its text from 0s and 1s, then restores it when the pointer leaves", () => {
    stubMotion(false);
    const raf = stubRaf();
    render(<Flow />);
    fireEvent.pointerEnter(slots()[0]);
    expect(box(0).dataset.open).toBe("true");
    expect(box(1).dataset.open).toBe("false");
    raf.frame(1000); // starts the decode
    raf.frame(1100);
    const bits = [...document.querySelectorAll<HTMLElement>(".h-bit")];
    expect(bits.length).toBeGreaterThan(50);
    expect(bits.every((b) => b.dataset.b === "0" || b.dataset.b === "1")).toBe(true);
    // assistive technology reads the real text once, not the bits
    expect(document.querySelector(".h-desc-text .sr-only")!.textContent).toContain(FIRST);
    expect(document.querySelector(".h-desc-text [aria-hidden='true']")).toBeTruthy();
    raf.frame(1000 + 800); // past the decode time: every letter is real
    expect(document.querySelectorAll(".h-bit")).toHaveLength(0);
    expect(screen.getByText(new RegExp(FIRST))).toBeTruthy();
    fireEvent.pointerLeave(slots()[0]);
    expect(box(0).dataset.open).toBe("false");
    expect(document.querySelectorAll(".h-bit")).toHaveLength(0);
  });

  it("opens on keyboard focus and closes on blur", () => {
    stubMotion(false);
    stubRaf();
    render(<Flow />);
    fireEvent.focus(box(2));
    expect(box(2).dataset.open).toBe("true");
    fireEvent.blur(box(2));
    expect(box(2).dataset.open).toBe("false");
  });

  it("with reduced motion it opens at once with the real text and no bits", () => {
    stubMotion(true);
    stubRaf();
    render(<Flow />);
    fireEvent.pointerEnter(slots()[1]);
    expect(box(1).dataset.open).toBe("true");
    expect(document.querySelectorAll(".h-bit")).toHaveLength(0);
  });
});
