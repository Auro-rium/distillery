import { act, cleanup, fireEvent, render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { GlowText } from "./GlowText";

function stubMotion(reduce: boolean) {
  vi.stubGlobal("matchMedia", (q: string) => ({
    matches: q.includes("reduce") ? reduce : false,
    addEventListener: () => undefined,
    removeEventListener: () => undefined,
  }));
}
function stubRaf() {
  vi.stubGlobal("requestAnimationFrame", (f: FrameRequestCallback) => { f(0); return 1; });
  vi.stubGlobal("cancelAnimationFrame", () => undefined);
}

/** jsdom has no PointerEvent: a MouseEvent named pointermove, with the pointer type attached. */
function move(clientX: number, clientY: number, pointerType: string) {
  const e = new MouseEvent("pointermove", { clientX, clientY });
  Object.defineProperty(e, "pointerType", { value: pointerType });
  act(() => { window.dispatchEvent(e); });
}

afterEach(() => { cleanup(); vi.unstubAllGlobals(); });

describe("GlowText", () => {
  it("renders the text once for assistive technology: the copy behind it is aria-hidden", () => {
    stubMotion(false);
    render(<GlowText as="h1">Neon words</GlowText>);
    expect(screen.getByRole("heading", { level: 1, name: "Neon words" })).toBeTruthy();
    expect(document.querySelector(".glow-halo")!.getAttribute("aria-hidden")).toBe("true");
  });
  it("with reveal, each word is its own numbered span in both layers, and the heading still reads as one sentence", () => {
    stubMotion(false);
    render(<GlowText as="h1" reveal>One two three</GlowText>);
    const words = [...document.querySelectorAll<HTMLElement>(".glow-text .glow-word")];
    expect(words.map((w) => w.textContent)).toEqual(["One", "two", "three"]);
    expect(words.map((w) => w.style.getPropertyValue("--i"))).toEqual(["0", "1", "2"]);
    expect(document.querySelectorAll(".glow-halo .glow-word")).toHaveLength(3);
    expect(screen.getByRole("heading", { level: 1, name: "One two three" })).toBeTruthy();
  });
  it("without reveal the text is not split into words", () => {
    stubMotion(false);
    render(<GlowText>One two three</GlowText>);
    expect(document.querySelectorAll(".glow-word")).toHaveLength(0);
  });
  it("moves the light to the pointer, relative to the text", () => {
    stubMotion(false);
    stubRaf();
    render(<GlowText as="h1">Neon words</GlowText>);
    const el = document.querySelector<HTMLElement>(".glow")!;
    expect(el.dataset.glow).toBe("on");
    el.getBoundingClientRect = () => ({ left: 100, top: 40 }) as DOMRect;
    move(130, 55, "mouse");
    expect(el.style.getPropertyValue("--mx")).toBe("30px");
    expect(el.style.getPropertyValue("--my")).toBe("15px");
    act(() => { fireEvent(document.documentElement, new Event("pointerleave")); });
    expect(el.style.getPropertyValue("--mx")).toBe("-9999px");
  });
  it("ignores touch pointers", () => {
    stubMotion(false);
    stubRaf();
    render(<GlowText>Neon words</GlowText>);
    const el = document.querySelector<HTMLElement>(".glow")!;
    el.getBoundingClientRect = () => ({ left: 0, top: 0 }) as DOMRect;
    move(10, 10, "touch");
    expect(el.style.getPropertyValue("--mx")).toBe("");
  });
  it("with reduced motion the text stays plain and never listens to the pointer", () => {
    stubMotion(true);
    stubRaf();
    render(<GlowText>Neon words</GlowText>);
    const el = document.querySelector<HTMLElement>(".glow")!;
    expect(el.dataset.glow).toBe("off");
    el.getBoundingClientRect = () => ({ left: 0, top: 0 }) as DOMRect;
    move(10, 10, "mouse");
    expect(el.style.getPropertyValue("--mx")).toBe("");
  });
});
