import { act, cleanup, fireEvent, render, screen, within } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { Tour } from "./Tour";
import { BUILD, SERVE } from "./steps";

const motionOk = () => vi.stubGlobal("matchMedia", (q: string) => ({ matches: false, media: q, addEventListener: () => undefined, removeEventListener: () => undefined }));

afterEach(() => { cleanup(); vi.unstubAllGlobals(); vi.useRealTimers(); });

const panel = () => document.querySelector<HTMLElement>(".tour-panel")!;

describe("How it works tour", () => {
  it("starts on the first stop, with one button per stop and its plain-words panel", () => {
    motionOk();
    render(<Tour title="Act" lead="lead" steps={BUILD} />);
    expect(screen.getAllByRole("button", { name: /./ }).filter((b) => b.classList.contains("tour-dot"))).toHaveLength(BUILD.length);
    expect(within(panel()).getByRole("heading", { level: 3 }).textContent).toBe(BUILD[0].title);
    expect(panel().textContent).toContain(BUILD[0].what);
    expect(panel().textContent).toContain("Why you can trust it");
    expect(screen.getByRole("button", { name: "Back" }).hasAttribute("disabled")).toBe(true);
  });

  it("a stop, Next, Back and the arrow keys move through it", () => {
    motionOk();
    render(<Tour title="Act" lead="lead" steps={BUILD} />);
    fireEvent.click(screen.getByRole("button", { name: "Next" }));
    expect(within(panel()).getByRole("heading", { level: 3 }).textContent).toBe(BUILD[1].title);
    fireEvent.click(screen.getByRole("button", { name: BUILD[4].title }));
    expect(within(panel()).getByRole("heading", { level: 3 }).textContent).toBe(BUILD[4].title);
    expect(screen.getByRole("button", { name: BUILD[4].title }).getAttribute("aria-current")).toBe("step");
    fireEvent.keyDown(screen.getByRole("group", { name: /stops/ }), { key: "ArrowLeft" });
    expect(within(panel()).getByRole("heading", { level: 3 }).textContent).toBe(BUILD[3].title);
  });

  it("the retraining stop says it loops back, and the link jumps there", () => {
    motionOk();
    render(<Tour title="Act" lead="lead" steps={BUILD} />);
    const i = BUILD.findIndex((s) => s.loopsTo);
    expect(i).toBeGreaterThan(0);
    fireEvent.click(screen.getByRole("button", { name: BUILD[i].title }));
    fireEvent.click(within(panel()).getByRole("button", { name: BUILD.find((s) => s.id === BUILD[i].loopsTo)!.title }));
    expect(within(panel()).getByRole("heading", { level: 3 }).textContent).toBe(BUILD.find((s) => s.id === BUILD[i].loopsTo)!.title);
  });

  it("Play walks on its own, stops at the end, and a manual move pauses it", () => {
    motionOk();
    vi.useFakeTimers();
    render(<Tour title="Act" lead="lead" steps={SERVE} />);
    fireEvent.click(screen.getByRole("button", { name: "Play" }));
    expect(screen.getByRole("button", { name: "Pause" }).getAttribute("aria-pressed")).toBe("true");
    act(() => { vi.advanceTimersByTime(2800); });
    expect(within(panel()).getByRole("heading", { level: 3 }).textContent).toBe(SERVE[1].title);
    fireEvent.click(screen.getByRole("button", { name: "Next" })); // manual: pauses
    expect(screen.getByRole("button", { name: "Play" })).toBeTruthy();
    act(() => { vi.advanceTimersByTime(10000); });
    expect(within(panel()).getByRole("heading", { level: 3 }).textContent).toBe(SERVE[2].title);
    fireEvent.click(screen.getByRole("button", { name: "Play" }));
    act(() => { vi.advanceTimersByTime(2800 * 5); });
    expect(within(panel()).getByRole("heading", { level: 3 }).textContent).toBe(SERVE[SERVE.length - 1].title);
    expect(screen.getByRole("button", { name: "Replay" })).toBeTruthy();
  });

  it("offers no autoplay under reduced motion", () => {
    vi.stubGlobal("matchMedia", (q: string) => ({ matches: q.includes("reduce"), media: q, addEventListener: () => undefined, removeEventListener: () => undefined }));
    render(<Tour title="Act" lead="lead" steps={BUILD} />);
    expect(screen.queryByRole("button", { name: /Play|Pause|Replay/ })).toBeNull();
    expect(screen.getByRole("button", { name: "Next" })).toBeTruthy();
  });

  it("the words carry no figures", () => {
    for (const s of [...BUILD, ...SERVE]) expect(`${s.title} ${s.what} ${s.trust}`, s.id).not.toMatch(/\d/);
  });
});
