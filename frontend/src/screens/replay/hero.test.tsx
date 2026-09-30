import { cleanup, screen, waitFor } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { contract, contractFetch } from "../../testutil/contract";
import { renderApp, stubFetch } from "../../testutil/render";

afterEach(() => { cleanup(); vi.unstubAllGlobals(); });

describe("replay landing", () => {
  it("the label banner comes before the hero in the document", async () => {
    stubFetch(contractFetch());
    const { container } = renderApp("/");
    await waitFor(() => screen.getByText("DRY RUN — fake models, numbers are NOT results"));
    const banner = container.querySelector(".banners")!;
    const hero = container.querySelector(".hero")!;
    expect(banner.compareDocumentPosition(hero) & Node.DOCUMENT_POSITION_FOLLOWING).toBeTruthy();
  });
  it("reserves the banner slot while loading, so nothing shifts when it arrives", () => {
    vi.stubGlobal("fetch", vi.fn(() => new Promise(() => undefined)));
    const { container } = renderApp("/");
    expect(container.querySelector(".banners .banner-skeleton")).not.toBeNull();
    expect(container.querySelector(".hero")).not.toBeNull();
  });
  it("the pipeline diagram has stage labels and no digits anywhere", () => {
    stubFetch(contractFetch());
    const { container } = renderApp("/");
    const pipe = container.querySelector(".pipe")!;
    expect(pipe.querySelectorAll("text").length).toBeGreaterThan(8);
    expect(pipe.textContent).not.toMatch(/\d/);
    expect(pipe.textContent).toContain("Gold cross-check");
    expect(pipe.querySelector("svg")!.getAttribute("aria-hidden")).toBe("true");
  });
  it("shows one banner per distinct label and tags every bundle with its own label", async () => {
    const base = contract.replay[0];
    stubFetch(contractFetch({ replay: [base, { ...base, run_id: "rec-1", dry_run: false, recorded: true, recorded_at: "2026-02-03T00:00:00Z" }] }));
    const { container } = renderApp("/");
    await waitFor(() => screen.getByText("rec-1"));
    expect(container.querySelectorAll(".banners .label-banner")).toHaveLength(2);
    const tags = [...container.querySelectorAll(".card .badge")].map((b) => b.textContent);
    expect(tags).toContain("dry run");
    expect(tags).toContain("recorded");
  });
});
