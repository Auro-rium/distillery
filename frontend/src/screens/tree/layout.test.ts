import { describe, expect, it } from "vitest";
import type { TreeNode } from "../../api/types";
import { GX, GY, NH, NW, bestOrientation, edgePath, layoutTree } from "./layout";

const n = (id: string, parent: string | null): TreeNode => ({
  id, parent_id: parent, label: id, round: null, hypothesis: null, data_delta: null,
  dev_score: null, cost_usd: null, sandbox_image: null, selected: false,
});
const at = (l: ReturnType<typeof layoutTree>) => Object.fromEntries(l.placed.map((p) => [p.node.id, p]));

/** A chain a > b > c with a fork: b also has a second child d. */
const FORK = [n("a", null), n("b", "a"), n("c", "b"), n("d", "b")];

describe("layoutTree left to right", () => {
  it("places children right of their parent and siblings apart vertically, parent centred on its children", () => {
    const p = at(layoutTree(FORK));
    expect(p.b.x).toBe(p.a.x + NW + GX);
    expect(p.c.x).toBe(p.d.x);
    expect(p.d.y).toBe(p.c.y + NH + GY);
    expect(p.b.y).toBe((p.c.y + p.d.y) / 2);
    expect(p.a.depth).toBe(0);
    expect(p.c.depth).toBe(2);
  });
  it("handles orphans and cycles without dropping a node", () => {
    const l = layoutTree([n("a", null), n("b", "a"), n("c", "a"), n("o", "missing"), n("x", "y"), n("y", "x")]);
    expect(l.placed).toHaveLength(6);
    const p = at(l);
    expect(p.b.x).toBeGreaterThan(p.a.x);
    expect(p.b.y).not.toBe(p.c.y);
  });
  it("is empty for no nodes and a single box for one node", () => {
    expect(layoutTree([])).toMatchObject({ placed: [], edges: [], width: 0, height: 0 });
    const one = layoutTree([n("only", null)]);
    expect([one.width, one.height]).toEqual([NW, NH]);
  });
});

describe("layoutTree reading order", () => {
  it("lists every parent before its children, siblings in layout order (the DOM order screen readers follow)", () => {
    expect(layoutTree(FORK).placed.map((p) => p.node.id)).toEqual(["a", "b", "c", "d"]);
    expect(layoutTree(FORK, "tb").placed.map((p) => p.node.id)).toEqual(["a", "b", "c", "d"]);
    const two = [n("r1", null), n("x", "r1"), n("r2", null), n("y", "r2"), n("z", "x")];
    expect(layoutTree(two).placed.map((p) => p.node.id)).toEqual(["r1", "x", "z", "r2", "y"]);
  });
});

describe("layoutTree top to bottom (narrow screens)", () => {
  it("places children below their parent and siblings apart horizontally", () => {
    const p = at(layoutTree(FORK, "tb"));
    expect(p.b.y).toBeGreaterThan(p.a.y + NH);
    expect(p.c.y).toBe(p.d.y);
    expect(p.d.x).toBeGreaterThan(p.c.x + NW - 1);
    expect(p.b.x).toBe((p.c.x + p.d.x) / 2);
    expect(p.a.depth).toBe(0);
  });
  it("a plain chain is exactly one node wide", () => {
    const l = layoutTree([n("a", null), n("b", "a"), n("c", "b")], "tb");
    expect(l.width).toBe(NW);
  });
});

describe("layoutTree with many nodes", () => {
  // 30 real-shaped nodes: a chain of 6 rounds, every round branching into 4 siblings under the previous round's first node
  const many: TreeNode[] = [n("root", null)];
  for (let r = 1; r <= 6; r++) for (let k = 0; k < 4; k++) many.push(n(`r${r}-${k}`, r === 1 ? "root" : `r${r - 1}-0`));
  it.each(["lr", "tb"] as const)("never overlaps two nodes and keeps every node inside the reported size (%s)", (o) => {
    const l = layoutTree(many, o);
    expect(l.placed).toHaveLength(many.length);
    for (const a of l.placed) {
      expect(a.x).toBeGreaterThanOrEqual(0);
      expect(a.y).toBeGreaterThanOrEqual(0);
      expect(a.x + NW).toBeLessThanOrEqual(l.width);
      expect(a.y + NH).toBeLessThanOrEqual(l.height);
      for (const b of l.placed) {
        if (a === b) continue;
        const apart = a.x + NW <= b.x || b.x + NW <= a.x || a.y + NH <= b.y || b.y + NH <= a.y;
        expect(apart, `${a.node.id} overlaps ${b.node.id}`).toBe(true);
      }
    }
    expect(l.edges).toHaveLength(many.length - 1);
  });
});

describe("edgePath", () => {
  it("leaves the parent on the side that faces the child, in both orientations", () => {
    const l = layoutTree([n("a", null), n("b", "a")]);
    const [a, b] = [l.placed.find((p) => p.node.id === "a")!, l.placed.find((p) => p.node.id === "b")!];
    expect(edgePath(a, b, "lr")).toMatch(new RegExp(`^M${a.x + NW},${a.y + NH / 2} C`));
    expect(edgePath(a, b, "lr")).toMatch(new RegExp(`${b.x},${b.y + NH / 2}$`));
    const t = layoutTree([n("a", null), n("b", "a")], "tb");
    const [ta, tb] = [t.placed.find((p) => p.node.id === "a")!, t.placed.find((p) => p.node.id === "b")!];
    expect(edgePath(ta, tb, "tb")).toMatch(new RegExp(`^M${ta.x + NW / 2},${ta.y + NH} C`));
    expect(edgePath(ta, tb, "tb")).toMatch(new RegExp(`${tb.x + NW / 2},${tb.y}$`));
  });
});

describe("bestOrientation", () => {
  const chain = [n("a", null), n("b", "a"), n("c", "b")];
  it("keeps left to right when that shows the tree about as large as the alternative", () => {
    expect(bestOrientation(chain, 1000, 500)).toBe("lr");
    expect(bestOrientation([n("only", null)], 900, 500)).toBe("lr");
  });
  it("goes top to bottom when that shows the tree noticeably larger (narrow or squarish viewports)", () => {
    expect(bestOrientation(chain, 360, 500)).toBe("tb");
    expect(bestOrientation(chain, 620, 558)).toBe("tb"); // a short chain in a squarish box: full size beats a scaled-down row
  });
  it("stays left to right for a long chain in a wide, short viewport", () => {
    const long = [n("r0", null), ...Array.from({ length: 6 }, (_, i) => n(`r${i + 1}`, `r${i}`))];
    expect(bestOrientation(long, 900, 300)).toBe("lr");
  });
  it("has no opinion before there is a viewport to measure", () => {
    expect(bestOrientation(chain, 0, 0)).toBe("lr");
    expect(bestOrientation([], 800, 500)).toBe("lr");
  });
});
