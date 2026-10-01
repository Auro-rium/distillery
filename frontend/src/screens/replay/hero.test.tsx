import { cleanup, screen, waitFor } from "@testing-library/react";
import fs from "node:fs";
import path from "node:path";
import { afterEach, describe, expect, it, vi } from "vitest";
import { contract, contractFetch } from "../../testutil/contract";
import { renderApp, stubFetch } from "../../testutil/render";

afterEach(() => { cleanup(); vi.unstubAllGlobals(); });

describe("replay landing", () => {
  it("the label banner comes before the hero in the document", async () => {
    stubFetch(contractFetch());
    const { container } = renderApp("/");
    await waitFor(() => screen.getAllByText("DRY RUN — fake models, numbers are NOT results"));
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
  it("the pipeline is drawn twice, a wide layout and a tall stepper, both decorative, neither in a horizontal scroller", () => {
    stubFetch(contractFetch());
    const { container } = renderApp("/");
    const pipe = container.querySelector(".pipe")!;
    for (const kind of ["wide", "tall"]) {
      const svg = pipe.querySelector<SVGElement>(`svg.flow-${kind}`)!;
      expect(svg, kind).not.toBeNull();
      expect(svg.getAttribute("aria-hidden")).toBe("true");
      expect(svg.querySelectorAll("text").length).toBeGreaterThan(8);
      expect(svg.getAttribute("viewBox")).toMatch(/^0 0 \d+ \d+$/);
      expect(svg.textContent).toContain("Template gold");
      expect(svg.textContent).toContain("next round");
    }
    expect(pipe.querySelector(".pipe-scroll")).toBeNull();
    expect(pipe.querySelectorAll("svg")).toHaveLength(2);
  });
  it("the diagram shows no number anywhere and its steps are also available as a list for screen readers", () => {
    stubFetch(contractFetch());
    const { container } = renderApp("/");
    const pipe = container.querySelector(".pipe")!;
    expect(pipe.textContent).not.toMatch(/\d/);
    const items = [...pipe.querySelectorAll("ol.sr-only li")].map((li) => li.textContent);
    expect(items).toHaveLength(9);
    expect(items[8]).toBe("A fixed gate decides PROMOTE or REJECT.");
  });
  it("every route is drawn once per layout, the loop is the only accent route, and only strokes and opacity animate", () => {
    stubFetch(contractFetch());
    const { container } = renderApp("/");
    for (const kind of ["wide", "tall"]) {
      const svg = container.querySelector(`svg.flow-${kind}`)!;
      expect(svg.querySelectorAll("path.flow-edge")).toHaveLength(9);
      expect(svg.querySelectorAll("path.flow-edge.loop")).toHaveLength(1);
      expect(svg.querySelectorAll("path.flow-edge").item(0).getAttribute("pathLength")).toBe("1");
    }
    const css = fs.readFileSync(path.resolve(__dirname, "replay.css"), "utf8");
    const frames = [...css.matchAll(/@keyframes\s+([\w-]+)\s*\{([\s\S]*?\})\s*\}/g)];
    expect(frames.length).toBeGreaterThan(0);
    for (const [, name, body] of frames) {
      const props = [...body.matchAll(/([a-z-]+)\s*:/g)].map((m) => m[1]);
      for (const p of props) expect(["opacity", "stroke-dashoffset"], `${name} animates ${p}`).toContain(p);
    }
    expect(css).toMatch(/prefers-reduced-motion:\s*reduce/);
  });
  it("shows one banner per distinct label and tags every bundle with its own label", async () => {
    const base = contract.replay[0];
    stubFetch(contractFetch({ runs: [], replay: [base, { ...base, run_id: "rec-1", dry_run: false, recorded: true, recorded_at: "2026-02-03T00:00:00Z" }] }));
    const { container } = renderApp("/");
    await waitFor(() => screen.getByText("rec-1"));
    expect(container.querySelectorAll(".banners .label-banner")).toHaveLength(2);
    const tags = [...container.querySelectorAll(".card .badge")].map((b) => b.textContent);
    expect(tags).toContain("dry run");
    expect(tags).toContain("recorded");
  });
});
