import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { MemoryRouter, Route, Routes } from "react-router-dom";
import { afterEach, describe, expect, it, vi } from "vitest";
import type { TreeNode } from "../../api/types";
import Tree from ".";
import { lineage } from "./path";

const n = (id: string, parent: string | null, extra: Partial<TreeNode> = {}): TreeNode => ({
  id, parent_id: parent, label: id, round: null, hypothesis: null, data_delta: null,
  dev_score: null, cost_usd: null, sandbox_image: null, selected: false, ...extra,
});
afterEach(() => { cleanup(); vi.unstubAllGlobals(); });

function show(nodes: TreeNode[]) {
  vi.stubGlobal("fetch", vi.fn((u: string) => Promise.resolve(
    new Response(JSON.stringify(u.endsWith("/tree") ? { nodes } : { run_id: "r", dry_run: true, recorded: false, recorded_at: null }), { status: 200 }))));
  return render(<MemoryRouter initialEntries={["/runs/r/tree"]}><Routes><Route path="/runs/:id/tree" element={<Tree />} /></Routes></MemoryRouter>);
}
const TREE = [n("root", null), n("a", "root", { round: 1 }), n("b", "root", { round: 1 }), n("a1", "a", { round: 2 })];

describe("lineage", () => {
  it("follows real parent links from the root, and survives orphans and cycles", () => {
    expect(lineage(TREE, "a1")).toEqual(["root", "a", "a1"]);
    expect(lineage(TREE, null)).toEqual([]);
    expect(lineage(TREE, "nope")).toEqual([]);
    expect(lineage([n("x", "y"), n("y", "x")], "x").sort()).toEqual(["x", "y"]);
    expect(lineage([n("o", "missing")], "o")).toEqual(["o"]);
  });
});

describe("tree interactions", () => {
  it("hovering or focusing a node highlights exactly its path from the root", async () => {
    const { container } = show(TREE);
    const a1 = await screen.findByRole("button", { name: /^a1/ });
    const on = () => [...container.querySelectorAll(".tnode.on-path")].map((g) => g.getAttribute("aria-label")!.split(",")[0]);
    expect(on()).toEqual([]); // nothing selected, nothing hovered
    fireEvent.pointerEnter(a1);
    expect(on().sort()).toEqual(["a", "a1", "root"]);
    expect(container.querySelectorAll(".edge.on-path")).toHaveLength(2);
    expect(container.querySelector("svg")!.getAttribute("data-hot")).toBe("true");
    fireEvent.pointerLeave(a1);
    expect(container.querySelector("svg")!.getAttribute("data-hot")).toBe("false");
    fireEvent.focus(screen.getByRole("button", { name: /^b/ }));
    expect(on().sort()).toEqual(["b", "root"]);
  });
  it("draws only real nodes and edges, with lineage depth as animation order", async () => {
    const { container } = show(TREE);
    await screen.findByRole("button", { name: /^root/ });
    expect(container.querySelectorAll(".tnode")).toHaveLength(4);
    expect(container.querySelectorAll("path.edge")).toHaveLength(3);
    const depth = (label: RegExp) => (screen.getByRole("button", { name: label }) as unknown as SVGElement).style.getPropertyValue("--depth");
    expect([depth(/^root/), depth(/^a,/), depth(/^a1/)]).toEqual(["0", "1", "2"]);
  });
  it("arrow keys move focus along the real lineage", async () => {
    show(TREE);
    const root = await screen.findByRole("button", { name: /^root/ });
    root.focus();
    fireEvent.keyDown(root, { key: "ArrowRight" });
    const a = screen.getByRole("button", { name: /^a,/ });
    expect(document.activeElement).toBe(a);
    fireEvent.keyDown(a, { key: "ArrowDown" });
    expect(document.activeElement).toBe(screen.getByRole("button", { name: /^b,/ }));
    fireEvent.keyDown(document.activeElement!, { key: "ArrowLeft" });
    expect(document.activeElement).toBe(root);
  });
  it("zoom buttons and keys change the layer transform; fit resets it", async () => {
    const { container } = show(TREE);
    await screen.findByRole("button", { name: /^root/ });
    const layer = container.querySelector(".pz-layer") as SVGGElement;
    const t0 = layer.style.transform;
    expect(t0).toContain("scale(1)");
    fireEvent.click(screen.getByRole("button", { name: "Zoom in" }));
    expect(layer.style.transform).not.toBe(t0);
    expect(layer.style.transform).toMatch(/scale\(1\.\d+\)/);
    fireEvent.click(screen.getByRole("button", { name: "Zoom out" }));
    fireEvent.click(screen.getByRole("button", { name: "Zoom out" }));
    expect(layer.style.transform).toMatch(/scale\(0\.\d+\)/);
    fireEvent.keyDown(container.querySelector(".pz-box")!, { key: "0" });
    expect(layer.style.transform).toContain("scale(1)");
    fireEvent.keyDown(container.querySelector(".pz-box")!, { key: "ArrowLeft" });
    expect(layer.style.transform).toContain("translate(60px");
  });
  it("selecting a node swaps the detail panel to that node", async () => {
    show([n("root", null, { hypothesis: "h-root" }), n("a", "root", { hypothesis: "h-a" })]);
    fireEvent.click(await screen.findByRole("button", { name: /^root/ }));
    await waitFor(() => expect(screen.getByRole("region", { name: "Node detail" }).textContent).toContain("h-root"));
    fireEvent.click(screen.getByRole("button", { name: /^a,|^a$/ }));
    expect(screen.getByRole("region", { name: "Node detail" }).textContent).toContain("h-a");
  });
});
