import { cleanup, fireEvent, screen, waitFor, within } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { READABLE_MIN } from "./constants";
import { NH, NW } from "./layout";
import { lineage } from "./path";
import { fakeViewport, n, pos, show, view } from "./treeharness";

afterEach(() => { cleanup(); vi.unstubAllGlobals(); });

const TREE = [n("root", null), n("a", "root", { round: 1 }), n("b", "root", { round: 1 }), n("a1", "a", { round: 2 })];
const layer = (c: HTMLElement) => c.querySelector(".pz-layer") as HTMLElement;
const box = (c: HTMLElement) => c.querySelector(".pz-box") as HTMLElement;

describe("lineage", () => {
  it("follows real parent links from the root, and survives orphans and cycles", () => {
    expect(lineage(TREE, "a1")).toEqual(["root", "a", "a1"]);
    expect(lineage(TREE, null)).toEqual([]);
    expect(lineage(TREE, "nope")).toEqual([]);
    expect(lineage([n("x", "y"), n("y", "x")], "x").sort()).toEqual(["x", "y"]);
    expect(lineage([n("o", "missing")], "o")).toEqual(["o"]);
  });
});

describe("path highlighting", () => {
  const on = (c: HTMLElement) => [...c.querySelectorAll(".tnode.on-path")].map((g) => g.getAttribute("aria-label")!.split(",")[0]).sort();
  it("highlights the path from the root to the selected node from the start", async () => {
    const { container } = show([n("root", null), n("a", "root", { selected: true }), n("b", "root"), n("a1", "a")]);
    await screen.findByRole("button", { name: /^a1/ });
    expect(on(container)).toEqual(["a", "root"]);
    expect(container.querySelectorAll(".edge.on-path")).toHaveLength(1);
    expect(container.querySelector(".pz-box")!.getAttribute("data-hot")).toBe("false"); // nothing hovered: nothing dimmed
  });
  it("hovering or focusing a node highlights exactly its path from the root and dims the rest; leaving restores the selection's path", async () => {
    const { container } = show(TREE);
    const a1 = await screen.findByRole("button", { name: /^a1/ });
    expect(on(container)).toEqual(["root"]); // no candidate: the first root is selected
    fireEvent.pointerEnter(a1);
    expect(on(container)).toEqual(["a", "a1", "root"]);
    expect(container.querySelectorAll(".edge.on-path")).toHaveLength(2);
    expect(container.querySelector(".pz-box")!.getAttribute("data-hot")).toBe("true");
    fireEvent.pointerLeave(a1);
    expect(container.querySelector(".pz-box")!.getAttribute("data-hot")).toBe("false");
    expect(on(container)).toEqual(["root"]);
    fireEvent.focus(screen.getByRole("button", { name: /^b,/ }));
    expect(on(container)).toEqual(["b", "root"]);
  });
});

describe("what is drawn", () => {
  it("draws only real nodes and edges, with lineage depth as animation order", async () => {
    const { container } = show(TREE);
    await screen.findByRole("button", { name: /^root/ });
    expect(container.querySelectorAll(".tnode")).toHaveLength(4);
    expect(container.querySelectorAll("path.edge")).toHaveLength(3);
    const depth = (label: RegExp) => screen.getByRole("button", { name: label }).style.getPropertyValue("--depth");
    expect([depth(/^root/), depth(/^a,/), depth(/^a1/)]).toEqual(["0", "1", "2"]);
  });
  it("one node, thirty nodes: every node is a button and every parent link is one edge", async () => {
    const one = show([n("only", null)]);
    await screen.findByRole("button", { name: /^only/ });
    expect(one.container.querySelectorAll("path.edge")).toHaveLength(0);
    cleanup();
    const many = [n("root", null), ...Array.from({ length: 29 }, (_, i) => n(`n${i}`, i === 0 ? "root" : `n${Math.floor((i - 1) / 2)}`))];
    const { container } = show(many);
    await screen.findByRole("button", { name: /^n28/ });
    expect(container.querySelectorAll(".tnode")).toHaveLength(30);
    expect(container.querySelectorAll("path.edge")).toHaveLength(29);
  });
});

describe("keyboard", () => {
  it("arrow keys move focus along the real lineage (left to right)", async () => {
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
  it("in the top-to-bottom layout down goes to the child and left/right to siblings", async () => {
    const vp = fakeViewport({ w: 360, h: 500 });
    try {
      show(TREE);
      const root = await screen.findByRole("button", { name: /^root/ });
      root.focus();
      fireEvent.keyDown(root, { key: "ArrowDown" });
      const a = screen.getByRole("button", { name: /^a,/ });
      expect(document.activeElement).toBe(a);
      fireEvent.keyDown(a, { key: "ArrowRight" });
      expect(document.activeElement).toBe(screen.getByRole("button", { name: /^b,/ }));
      fireEvent.keyDown(document.activeElement!, { key: "ArrowUp" });
      expect(document.activeElement).toBe(root);
    } finally { vp.restore(); }
  });
});

describe("pan and zoom controls", () => {
  it("zoom buttons and keys change the layer transform; fit resets it; arrows pan when the viewport has focus", async () => {
    const { container } = show(TREE);
    await screen.findByRole("button", { name: /^root/ });
    const t0 = layer(container).style.transform;
    expect(t0).toContain("scale(1)");
    fireEvent.click(screen.getByRole("button", { name: "Zoom in" }));
    expect(view(layer(container)).k).toBeGreaterThan(1);
    fireEvent.click(screen.getByRole("button", { name: "Zoom out" }));
    fireEvent.click(screen.getByRole("button", { name: "Zoom out" }));
    expect(view(layer(container)).k).toBeLessThan(1);
    fireEvent.keyDown(box(container), { key: "0" });
    expect(layer(container).style.transform).toContain("scale(1)");
    fireEvent.keyDown(box(container), { key: "ArrowLeft" });
    expect(layer(container).style.transform).toContain("translate(60px");
    fireEvent.keyDown(box(container), { key: "+" });
    fireEvent.keyDown(box(container), { key: "-" });
    fireEvent.click(screen.getByRole("button", { name: "Fit view" }));
    expect(layer(container).style.transform).toContain("scale(1)");
  });
  it("the controls are a named group and every button is labelled", async () => {
    show(TREE);
    await screen.findByRole("button", { name: /^root/ });
    const g = screen.getByRole("group", { name: "Tree view controls" });
    expect(g.querySelectorAll("button")).toHaveLength(4);
    for (const name of ["Zoom in", "Zoom out", "Fit view", "Show path"]) expect(within(g).getByRole("button", { name })).toBeTruthy();
    expect(screen.getByRole("group", { name: "Tree viewport" }).tabIndex).toBe(0);
  });
});

describe("auto-fit", () => {
  /** Each node's box, after the layer's transform, lies inside a viewport of w x h. */
  const boxOf = (c: HTMLElement, b: HTMLElement) => {
    const v = view(layer(c));
    const p = pos(b);
    return { l: p.x * v.k + v.x, t: p.y * v.k + v.y, r: (p.x + NW) * v.k + v.x, b: (p.y + NH) * v.k + v.y };
  };
  const inView = (c: HTMLElement, b: HTMLElement, w: number, h: number) => {
    const r = boxOf(c, b);
    return r.l >= -0.5 && r.t >= -0.5 && r.r <= w + 0.5 && r.b <= h + 0.5;
  };
  /** Every node lies inside a viewport of w x h. */
  const inside = (c: HTMLElement, w: number, h: number) => {
    for (const b of c.querySelectorAll<HTMLElement>(".tnode")) expect(inView(c, b, w, h), b.getAttribute("aria-label")!).toBe(true);
    return view(layer(c));
  };
  const node = (name: RegExp) => screen.getByRole("button", { name });

  // seven nodes: fits at a readable size in both viewports used below
  const MEDIUM = [n("root", null), n("a", "root"), n("b", "root"), n("c", "root"), n("a1", "a"), n("b1", "b"), n("c1", "c")];
  // a long chain of rounds, the last one the selected candidate
  const CHAIN = Array.from({ length: 30 }, (_, i) => n(`r${i}`, i === 0 ? null : `r${i - 1}`, { label: `round ${i}`, round: i, selected: i === 29 }));
  // 90 nodes in a wide branching fan, the last leaf selected
  const HUGE = [n("root", null), ...Array.from({ length: 89 }, (_, i) => n(`n${i}`, i < 5 ? "root" : `n${Math.floor((i - 5) / 2)}`, { selected: i === 88 }))];

  it("a tree that fits at a readable size is fitted whole on load", async () => {
    const vp = fakeViewport({ w: 700, h: 480 });
    try {
      const { container } = show(MEDIUM);
      await screen.findByRole("button", { name: /^c1/ });
      const v = inside(container, 700, 480);
      expect(v.k).toBeGreaterThanOrEqual(READABLE_MIN);
    } finally { vp.restore(); }
  });

  it.each([["a chain of thirty rounds", CHAIN, /^round 29/, /^round 28/], ["ninety branching nodes", HUGE, /^n88/, /^n42/]] as const)(
    "%s: opens at a readable zoom framed on the path to the selected node, and Fit shows everything",
    async (_name, tree, selected, parent) => {
      const vp = fakeViewport({ w: 700, h: 480 });
      try {
        const { container } = show([...tree]);
        await screen.findByRole("button", { name: selected });
        const v = view(layer(container));
        expect(v.k).toBeGreaterThanOrEqual(READABLE_MIN); // labels stay legible: never shrunk to fit
        expect(inView(container, node(selected), 700, 480)).toBe(true);
        expect(inView(container, node(parent), 700, 480) || tree === HUGE).toBe(true); // the chain keeps its parent in view
        fireEvent.click(screen.getByRole("button", { name: "Fit view" }));
        const all = inside(container, 700, 480);
        expect(all.k).toBeLessThan(READABLE_MIN);
        expect(all.k).toBeGreaterThan(0);
        // Show path returns to the readable view of the selection
        fireEvent.click(screen.getByRole("button", { name: "Show path" }));
        expect(view(layer(container)).k).toBeGreaterThanOrEqual(READABLE_MIN);
        expect(inView(container, node(selected), 700, 480)).toBe(true);
        // the 0 key is Fit as well
        fireEvent.keyDown(box(container), { key: "0" });
        expect(view(layer(container)).k).toBe(all.k);
      } finally { vp.restore(); }
    },
  );

  it("keeps the selected node and its parent in view when the rest of the path runs the other way across the viewport", async () => {
    // six rounds of five branches, each round hanging off the previous round's first two nodes; the selected candidate is the
    // first node of the last round, which sits at the top of the layout while the root sits in the middle
    const SYNTH = [n("base", null)];
    for (let r = 1; r <= 6; r++) for (let k = 0; k < 5; k++) SYNTH.push(n(`r${r}-${k}`, r === 1 ? "base" : `r${r - 1}-${k % 2}`, { label: `r${r}-${k}`, round: r, selected: r === 6 && k === 0 }));
    const vp = fakeViewport({ w: 620, h: 556 });
    try {
      const { container } = show(SYNTH);
      await screen.findByRole("button", { name: /^r6-0/ });
      expect(view(layer(container)).k).toBeGreaterThanOrEqual(READABLE_MIN);
      expect(inView(container, node(/^r6-0/), 620, 556)).toBe(true);
      expect(inView(container, node(/^r5-0/), 620, 556)).toBe(true);
    } finally { vp.restore(); }
  });

  it("a small tree is shown at natural size, centred", async () => {
    const vp = fakeViewport({ w: 900, h: 500 });
    try {
      const { container } = show([n("only", null)]);
      await screen.findByRole("button", { name: /^only/ });
      const v = view(layer(container));
      expect(v.k).toBe(1);
      expect(v.x).toBeCloseTo((900 - NW) / 2);
      expect(v.y).toBeCloseTo((500 - NH) / 2);
    } finally { vp.restore(); }
  });

  it("re-fits when the viewport is resized, until the user takes control", async () => {
    const vp = fakeViewport({ w: 700, h: 480 });
    try {
      const { container } = show(MEDIUM);
      await screen.findByRole("button", { name: /^c1/ });
      const small = inside(container, 700, 480).k;
      vp.resize(1400, 900);
      const big = inside(container, 1400, 900);
      expect(big.k).toBeGreaterThan(small);
      // the user zooms: a later resize must not throw their view away
      fireEvent.click(screen.getByRole("button", { name: "Zoom in" }));
      const mine = layer(container).style.transform;
      vp.resize(900, 600);
      expect(layer(container).style.transform).toBe(mine);
      // Fit hands control back
      fireEvent.click(screen.getByRole("button", { name: "Fit view" }));
      inside(container, 900, 600);
      vp.resize(700, 480);
      inside(container, 700, 480);
    } finally { vp.restore(); }
  });

  it("re-fits when the layout flips between wide and narrow", async () => {
    const vp = fakeViewport({ w: 900, h: 500 });
    try {
      const { container } = show(TREE);
      await screen.findByRole("button", { name: /^a1/ });
      inside(container, 900, 500);
      vp.resize(360, 640);
      await waitFor(() => expect(pos(screen.getByRole("button", { name: /^a1/ })).y).toBeGreaterThan(pos(screen.getByRole("button", { name: /^a,/ })).y));
      inside(container, 360, 640);
    } finally { vp.restore(); }
  });

  it("choosing a node from the path in the detail brings it into view", async () => {
    const vp = fakeViewport({ w: 700, h: 480 });
    try {
      const { container } = show(CHAIN);
      await screen.findByRole("button", { name: /^round 29/ });
      const root = screen.getByRole("button", { name: /^round 0/ });
      expect(inView(container, root, 700, 480)).toBe(false); // far behind the readable view
      fireEvent.click(screen.getByRole("button", { name: "Go to round 0" }));
      expect(inView(container, screen.getByRole("button", { name: /^round 0/ }), 700, 480)).toBe(true);
    } finally { vp.restore(); }
  });
});
