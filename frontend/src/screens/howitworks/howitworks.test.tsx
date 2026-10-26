import { cleanup, fireEvent, screen, waitFor, within } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { contract, contractFetch } from "../../testutil/contract";
import { renderApp, stubFetch } from "../../testutil/render";

afterEach(() => { cleanup(); vi.unstubAllGlobals(); });

describe("How it works page", () => {
  it("is its own page with the diagram, a heading and the right document title", async () => {
    stubFetch(contractFetch());
    const { container } = renderApp("/how-it-works");
    expect(screen.getByRole("heading", { level: 1, name: "How it works" })).toBeTruthy();
    expect(container.querySelectorAll(".pipe svg")).toHaveLength(2);
    await waitFor(() => expect(document.title).toMatch(/^How it works/));
  });

  it("has a button in the nav bar that goes there, and the Mission page no longer carries the section", async () => {
    stubFetch(contractFetch());
    renderApp("/mission");
    await waitFor(() => expect(document.querySelector(".hero .m-verdict") ?? document.querySelector(".hero")).not.toBeNull());
    expect(screen.queryByRole("heading", { level: 2, name: "How it works" })).toBeNull();
    expect(document.querySelector(".pipe")).toBeNull();
    const nav = screen.getByRole("navigation", { name: /main|site/i });
    const link = within(nav).getByRole("link", { name: "How it works" });
    expect(link.getAttribute("href")).toBe("/how-it-works");
    fireEvent.click(link);
    expect(await screen.findByRole("heading", { level: 1, name: "How it works" })).toBeTruthy();
    expect(document.querySelector(".pipe")).not.toBeNull();
  });
});

describe("Mission hint box", () => {
  const rec = { ...contract.replay[0], dry_run: false, recorded: true, recorded_at: "2026-10-05T00:04:35Z", run_id: "sql-gated-20261004T155134Z-66ff347c" };

  it("is the Mission label as a button, and opens a short explanation in plain words", async () => {
    stubFetch(contractFetch({ replay: [rec] }));
    renderApp("/mission");
    const hint = await screen.findByRole("button", { name: /^Mission/ });
    expect(hint.className).toContain("m-hint");
    fireEvent.click(hint);
    const dlg = await screen.findByRole("dialog", { name: "The mission, in easy words" });
    expect(within(dlg).getByText(/taught a small AI to do a job a big AI does/)).toBeTruthy();
    expect(within(dlg).getByText(/Teacher:/)).toBeTruthy();
    expect(within(dlg).getByText(/Student:/)).toBeTruthy();
    expect(within(dlg).getByText(/Base:/)).toBeTruthy();
    expect(within(dlg).getByText("What “gate” means")).toBeTruthy();
    // the featured run's name, taken apart from the real id
    const parts = [...dlg.querySelectorAll(".m-help-parts dt")].map((d) => d.textContent);
    expect(parts).toEqual(["sql", "gated", "20261004T155134Z", "66ff347c"]);
    // the wording carries no figures of its own
    expect(dlg.querySelector(".m-help-lead")!.textContent).not.toMatch(/\d/);
    expect(dlg.textContent).not.toMatch(/\d+(\.\d+)?\s?%/);
    fireEvent.keyDown(dlg, { key: "Escape" });
    await waitFor(() => expect(screen.queryByRole("dialog")).toBeNull());
  });

  it("shows an unusual run name whole instead of guessing its parts", async () => {
    stubFetch(contractFetch({ replay: [{ ...rec, run_id: "rec-1" }] }));
    renderApp("/mission");
    fireEvent.click(await screen.findByRole("button", { name: /^Mission/ }));
    const dlg = await screen.findByRole("dialog");
    expect(dlg.querySelector(".m-help-parts")).toBeNull();
    expect(within(dlg).getByText("rec-1")).toBeTruthy();
  });
});
