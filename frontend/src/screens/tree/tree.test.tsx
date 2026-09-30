import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { MemoryRouter, Route, Routes } from "react-router-dom";
import { afterEach, describe, expect, it, vi } from "vitest";
import type { TreeNode } from "../../api/types";
import Tree from ".";
import { layoutTree } from "./layout";

const n = (id: string, parent: string | null, extra: Partial<TreeNode> = {}): TreeNode => ({
  id, parent_id: parent, label: id, round: null, hypothesis: null, data_delta: null,
  dev_score: null, cost_usd: null, sandbox_image: null, selected: false, ...extra,
});
afterEach(() => { cleanup(); vi.unstubAllGlobals(); });

function show(nodes: TreeNode[]) {
  vi.stubGlobal("fetch", vi.fn((u: string) => Promise.resolve(
    new Response(JSON.stringify(u.endsWith("/tree") ? { nodes } : { run_id: "r", dry_run: true, recorded: false, recorded_at: null }), { status: 200 }))));
  render(<MemoryRouter initialEntries={["/runs/r/tree"]}><Routes><Route path="/runs/:id/tree" element={<Tree />} /></Routes></MemoryRouter>);
}

describe("layoutTree", () => {
  it("places children right of parents, handles orphans and cycles", () => {
    const l = layoutTree([n("a", null), n("b", "a"), n("c", "a"), n("o", "missing"), n("x", "y"), n("y", "x")]);
    expect(l.placed).toHaveLength(6);
    const byId = Object.fromEntries(l.placed.map((p) => [p.node.id, p]));
    expect(byId.b.x).toBeGreaterThan(byId.a.x);
    expect(byId.b.y).not.toBe(byId.c.y);
  });
});

describe("Tree screen", () => {
  it("shows an empty state and invents no nodes", async () => {
    show([]);
    expect(await screen.findByText(/no sandbox branches yet/i)).toBeTruthy();
    expect(screen.queryAllByRole("button")).toHaveLength(0);
  });
  it("renders nodes, opens detail by click and keyboard, highlights the selection", async () => {
    show([n("root", null, { label: "base", dev_score: 0.4 }), n("r1", "root", { label: "round-1", round: 1, hypothesis: "add joins", selected: true, data_delta: { added_rows: 7, families: ["join"] } })]);
    const b = await screen.findByRole("button", { name: /base/ });
    await waitFor(() => expect(screen.getByRole("region", { name: "Node detail" }).textContent).toContain("add joins"));
    fireEvent.keyDown(b, { key: "Enter" });
    expect(b.getAttribute("aria-pressed")).toBe("true");
    expect(screen.getByRole("region", { name: "Node detail" }).textContent).toContain("none recorded");
    fireEvent.click(screen.getByRole("button", { name: /round-1/ }));
    expect(screen.getByRole("region", { name: "Node detail" }).textContent).toContain("7 rows added");
  });
});
