import { act, cleanup, fireEvent, screen, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { contractFetch } from "./testutil/contract";
import { renderApp, stubFetch } from "./testutil/render";

beforeEach(() => stubFetch(contractFetch()));
afterEach(() => {
  cleanup(); vi.unstubAllGlobals();
  delete (document as unknown as { startViewTransition?: unknown }).startViewTransition;
  delete document.documentElement.dataset.theme;
});
const noMotionPref = () => vi.stubGlobal("matchMedia", () => ({ matches: false, addEventListener() {}, removeEventListener() {} }));

describe("route transitions", () => {
  it("without the View Transitions API the new page shows at once and plays the CSS enter animation", async () => {
    const { container } = renderApp("/");
    expect(container.querySelector(".route")!.className).toContain("route-enter");
    fireEvent.click(screen.getByRole("link", { name: "New run" }));
    expect(await screen.findByRole("heading", { name: "New run" })).toBeTruthy();
  });
  it("with the API (and no reduced-motion preference) the route change is committed inside startViewTransition", async () => {
    noMotionPref();
    const calls: number[] = [];
    (document as unknown as { startViewTransition: (cb: () => void) => unknown }).startViewTransition = (cb) => {
      calls.push(1);
      queueMicrotask(cb); // browsers run the callback asynchronously
      return {};
    };
    const { container } = renderApp("/");
    expect(container.querySelector(".route")!.className).not.toContain("route-enter");
    fireEvent.click(screen.getByRole("link", { name: "Playground" }));
    expect(screen.queryByRole("heading", { name: "Playground" })).toBeNull(); // old page stays until the transition commits
    await waitFor(() => expect(screen.getByRole("heading", { name: "Playground" })).toBeTruthy());
    expect(calls).toHaveLength(1);
  });
  it("with reduced motion the API is never used", async () => {
    const spy = vi.fn();
    (document as unknown as { startViewTransition: unknown }).startViewTransition = spy; // matchMedia is absent: reduced
    renderApp("/");
    fireEvent.click(screen.getByRole("link", { name: "New run" }));
    expect(await screen.findByRole("heading", { name: "New run" })).toBeTruthy();
    expect(spy).not.toHaveBeenCalled();
  });
});

describe("keyboard shortcuts", () => {
  it("? opens and Escape closes the shortcut list; the header button opens it too", async () => {
    renderApp("/");
    expect(screen.queryByRole("dialog")).toBeNull();
    fireEvent.keyDown(window, { key: "?" });
    expect(await screen.findByRole("dialog", { name: "Keyboard shortcuts" })).toBeTruthy(); // lazy-loaded
    fireEvent.keyDown(window, { key: "Escape" });
    await waitFor(() => expect(screen.queryByRole("dialog")).toBeNull());
    fireEvent.click(screen.getByRole("button", { name: "Keyboard shortcuts" }));
    expect(await screen.findByRole("dialog")).toBeTruthy();
  });
  it("t switches the theme, g then n navigates, and typing in a field is ignored", async () => {
    renderApp("/new");
    await screen.findByRole("heading", { name: "New run" });
    const before = document.documentElement.dataset.theme;
    const input = screen.getByLabelText("Budget cap (USD)");
    fireEvent.keyDown(input, { key: "t" });
    expect(document.documentElement.dataset.theme).toBe(before);
    fireEvent.keyDown(window, { key: "t" });
    expect(document.documentElement.dataset.theme).toBeTruthy();
    expect(document.documentElement.dataset.theme).not.toBe(before);
    fireEvent.keyDown(window, { key: "g" });
    act(() => { fireEvent.keyDown(window, { key: "p" }); });
    expect(await screen.findByRole("heading", { name: "Playground" })).toBeTruthy();
  });
});

describe("loading states", () => {
  it("are skeletons with the label text, not a bare spinner", () => {
    vi.stubGlobal("fetch", vi.fn(() => new Promise(() => undefined)));
    const { container } = renderApp("/runs/x/report");
    const s = screen.getByRole("status", { name: "" });
    expect(s.textContent).toBe("Loading report");
    expect(s.getAttribute("aria-busy")).toBe("true");
    expect(container.querySelectorAll(".sk-line").length).toBeGreaterThan(0);
  });
});
