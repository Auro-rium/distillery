// App shell: landmarks, skip link, focus on route change, phone menu, footer.
import { cleanup, fireEvent, screen, waitFor, within } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { contractFetch } from "../testutil/contract";
import { renderApp, stubFetch } from "../testutil/render";

beforeEach(() => stubFetch(contractFetch()));
afterEach(() => { cleanup(); vi.unstubAllGlobals(); delete document.documentElement.dataset.theme; document.title = ""; });

const TABBABLE = 'a[href],button:not([disabled]),input:not([disabled]),select:not([disabled]),textarea:not([disabled]),[tabindex]:not([tabindex="-1"])';

describe("landmarks", () => {
  it("has exactly one main landmark, a banner, a labelled nav and a footer", () => {
    renderApp("/");
    expect(screen.getAllByRole("main")).toHaveLength(1);
    expect(screen.getAllByRole("banner")).toHaveLength(1);
    expect(screen.getByRole("navigation", { name: "Main" })).toBeTruthy();
    expect(screen.getByRole("contentinfo")).toBeTruthy();
    expect(document.querySelector("main")!.id).toBe("main");
  });
  it("footer is plain text: no links, no digits, and it never claims a real-run label", () => {
    renderApp("/");
    const footer = screen.getByRole("contentinfo");
    expect(within(footer).queryAllByRole("link")).toHaveLength(0);
    expect(footer.textContent).not.toMatch(/\d/);
    expect(footer.textContent).not.toMatch(/not measured|DRY RUN|Label unknown/);
    expect((footer.textContent ?? "").trim().length).toBeGreaterThan(20);
  });
});

describe("skip link", () => {
  it("is the first tabbable element, points at main and moves focus there", () => {
    renderApp("/");
    const first = document.body.querySelector<HTMLElement>(TABBABLE)!;
    expect(first.textContent).toBe("Skip to main content");
    expect(first.getAttribute("href")).toBe("#main");
    fireEvent.click(first);
    expect(document.activeElement).toBe(document.querySelector("main"));
  });
});

describe("focus and title on route change", () => {
  it("leaves focus alone on first render, then moves it to main and updates the title after navigating", async () => {
    renderApp("/");
    expect(document.activeElement).toBe(document.body);
    fireEvent.click(screen.getByRole("link", { name: "New run" }));
    await screen.findByRole("heading", { name: "New run" });
    await waitFor(() => expect(document.activeElement).toBe(document.querySelector("main")));
    expect(document.title).toMatch(/^New run/);
    expect(document.title).toMatch(/Distillery$/);
    fireEvent.click(screen.getByRole("link", { name: "Playground" }));
    await screen.findByRole("heading", { name: "Playground" });
    expect(document.title).toMatch(/^Playground/);
  });
  it("announces the new page politely without adding a status role", async () => {
    renderApp("/");
    fireEvent.click(screen.getByRole("link", { name: "New run" }));
    await screen.findByRole("heading", { name: "New run" });
    const live = document.querySelector('[aria-live="polite"][data-route-announcer]')!;
    await waitFor(() => expect(live.textContent).toMatch(/^New run/));
    expect(live.getAttribute("role")).toBeNull();
  });
});

describe("phone menu", () => {
  it("opens a named dialog with the same three destinations, navigates and closes", async () => {
    renderApp("/");
    const btn = screen.getByRole("button", { name: "Menu" });
    expect(btn.getAttribute("aria-haspopup")).toBe("dialog");
    fireEvent.click(btn);
    const dlg = await screen.findByRole("dialog", { name: "Menu" }); // dialog code is lazy-loaded
    expect(within(dlg).getAllByRole("link").map((l) => l.textContent)).toEqual(["Replay", "New run", "Playground"]);
    expect(within(dlg).getByRole("link", { name: "Replay" }).getAttribute("aria-current")).toBe("page");
    fireEvent.click(within(dlg).getByRole("link", { name: "New run" }));
    await screen.findByRole("heading", { name: "New run" });
    await waitFor(() => expect(screen.queryByRole("dialog")).toBeNull());
  });
  it("Escape closes it and focus goes back to the Menu button", async () => {
    renderApp("/");
    const btn = screen.getByRole("button", { name: "Menu" });
    btn.focus();
    fireEvent.click(btn);
    fireEvent.keyDown(screen.getByRole("dialog"), { key: "Escape" });
    expect(screen.queryByRole("dialog")).toBeNull();
    await waitFor(() => expect(document.activeElement).toBe(btn)); // Radix restores focus on the next tick
  });
  it("keeps a single set of main links in the DOM while the menu is closed", () => {
    renderApp("/");
    expect(screen.getAllByRole("link", { name: "New run" })).toHaveLength(1);
  });
});

describe("theme and shortcuts still work", () => {
  it("the theme button says what it will switch to and toggles the theme", () => {
    renderApp("/");
    const b = screen.getByRole("button", { name: /^Switch to (light|dark) theme$/ });
    const before = document.documentElement.dataset.theme ?? "";
    fireEvent.click(b);
    expect(document.documentElement.dataset.theme).toMatch(/^(light|dark)$/);
    expect(document.documentElement.dataset.theme).not.toBe(before);
  });
});
