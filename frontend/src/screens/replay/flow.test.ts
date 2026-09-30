import { describe, expect, it } from "vitest";
import { LAYOUTS, STAGE_IDS, textWidth, type FlowNode, type Layout, type Pt } from "./flow";

const names = Object.keys(LAYOUTS) as (keyof typeof LAYOUTS)[];
const inside = (p: Pt, n: FlowNode, eps: number) => p.x > n.x + eps && p.x < n.x + n.w - eps && p.y > n.y + eps && p.y < n.y + n.h - eps;
const node = (l: Layout, id: string) => l.nodes.find((n) => n.id === id)!;

describe("pipeline layouts (pure geometry, checked without a browser)", () => {
  it("has a wide layout and a tall stepper, each drawing every stage once, in reading order", () => {
    expect(names.sort()).toEqual(["tall", "wide"]);
    for (const n of names) expect(LAYOUTS[n].nodes.map((x) => x.id)).toEqual([...STAGE_IDS]);
  });

  it.each(names)("%s: every node and every route stays inside the canvas, and no two nodes overlap", (n) => {
    const l = LAYOUTS[n];
    for (const a of l.nodes) {
      expect(a.x).toBeGreaterThanOrEqual(0);
      expect(a.y).toBeGreaterThanOrEqual(0);
      expect(a.x + a.w).toBeLessThanOrEqual(l.width);
      expect(a.y + a.h).toBeLessThanOrEqual(l.height);
      for (const b of l.nodes) {
        if (a === b) continue;
        const apart = a.x + a.w <= b.x || b.x + b.w <= a.x || a.y + a.h <= b.y || b.y + b.h <= a.y;
        expect(apart, `${a.id} overlaps ${b.id}`).toBe(true);
      }
    }
    for (const e of l.edges) for (const p of e.pts) {
      expect(p.x).toBeGreaterThanOrEqual(0); expect(p.x).toBeLessThanOrEqual(l.width);
      expect(p.y).toBeGreaterThanOrEqual(0); expect(p.y).toBeLessThanOrEqual(l.height);
    }
  });

  it.each(names)("%s: no route passes through a node, so no arrow is drawn over a label", (n) => {
    const l = LAYOUTS[n];
    for (const e of l.edges) for (const nd of l.nodes) {
      const hit = e.pts.find((p) => inside(p, nd, 0.5));
      expect(hit, `edge ${e.id} enters node ${nd.id}`).toBeUndefined();
    }
  });

  it.each(names)("%s: every route starts on the edge of its source and ends on the edge of its target", (n) => {
    const l = LAYOUTS[n];
    const touches = (p: Pt, nd: FlowNode) => p.x >= nd.x - 0.5 && p.x <= nd.x + nd.w + 0.5 && p.y >= nd.y - 0.5 && p.y <= nd.y + nd.h + 0.5;
    for (const e of l.edges) {
      expect(touches(e.pts[0], node(l, e.from)), `${e.id} start`).toBe(true);
      expect(touches(e.pts[e.pts.length - 1], node(l, e.to)), `${e.id} end`).toBe(true);
    }
  });

  it.each(names)("%s: title and sub text fit inside their node with padding (conservative width estimate)", (n) => {
    const l = LAYOUTS[n];
    for (const nd of l.nodes) {
      expect(textWidth(nd.title, l.titlePx, "sans") + 2 * l.pad, `${nd.id} title`).toBeLessThanOrEqual(nd.w);
      expect(textWidth(nd.sub, l.subPx, "mono") + 2 * l.pad, `${nd.id} sub`).toBeLessThanOrEqual(nd.w);
    }
  });

  it.each(names)("%s: the loop is one curved route from Targeted data back to Fine-tune student, and its label touches nothing", (n) => {
    const l = LAYOUTS[n];
    const loops = l.edges.filter((e) => e.kind === "loop");
    expect(loops).toHaveLength(1);
    expect([loops[0].from, loops[0].to]).toEqual(["g", "e"]);
    expect(loops[0].d).toMatch(/[QC]/); // a curve, not only right angles
    const w = textWidth(l.label.text, l.labelPx, "mono");
    const box = { x0: l.label.x - w / 2 - 4, x1: l.label.x + w / 2 + 4, y0: l.label.y - l.labelPx - 2, y1: l.label.y + 5 };
    for (const e of l.edges) {
      const hit = e.pts.find((p) => p.x >= box.x0 && p.x <= box.x1 && p.y >= box.y0 && p.y <= box.y1);
      expect(hit, `label touches route ${e.id}`).toBeUndefined();
    }
    for (const nd of l.nodes) {
      const apart = box.x1 <= nd.x || nd.x + nd.w <= box.x0 || box.y1 <= nd.y || nd.y + nd.h <= box.y0;
      expect(apart, `label overlaps node ${nd.id}`).toBe(true);
    }
    expect(box.x0).toBeGreaterThanOrEqual(0);
    expect(box.x1).toBeLessThanOrEqual(l.width);
  });

  it("carries no digit in any node, label or route name (the diagram shows no numbers)", () => {
    for (const n of names) {
      const l = LAYOUTS[n];
      const words = [...l.nodes.flatMap((x) => [x.title, x.sub]), l.label.text];
      for (const w of words) expect(w).not.toMatch(/\d/);
    }
  });
});
