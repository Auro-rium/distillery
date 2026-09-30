import type { TreeNode } from "../../api/types";
import { FIT_PAD, PREFER_LR } from "./constants";

/** Node box and gaps in content pixels (layout only, never displayed). */
export const NW = 220, NH = 72, GX = 56, GY = 16;
/** Top-to-bottom gaps: siblings sit side by side close together, generations are far enough apart for the connector to read. */
const TB_GX = 16, TB_GY = 44;

export type Orientation = "lr" | "tb";
export interface Placed { node: TreeNode; x: number; y: number; depth: number }
export interface Edge { from: string; to: string }
export interface Layout { placed: Placed[]; edges: Edge[]; width: number; height: number }

/**
 * Tree layout from the real parent links. `lr`: generations run left to right (wide screens). `tb`: generations
 * run top to bottom, so a chain of rounds is one node wide and stays readable on a phone. Siblings are stacked
 * in leaf order and a parent is centred on its first and last child. Nodes whose parent is unknown become roots,
 * and a cycle never drops a node.
 */
export function layoutTree(nodes: TreeNode[], orientation: Orientation = "lr"): Layout {
  const ids = new Set(nodes.map((n) => n.id));
  const kids = new Map<string, TreeNode[]>();
  const roots: TreeNode[] = [];
  for (const n of nodes) {
    if (n.parent_id !== null && ids.has(n.parent_id) && n.parent_id !== n.id) {
      kids.set(n.parent_id, [...(kids.get(n.parent_id) ?? []), n]);
    } else roots.push(n);
  }
  const at = (depth: number, slot: number) => (orientation === "lr"
    ? { x: depth * (NW + GX), y: slot * (NH + GY) }
    : { x: slot * (NW + TB_GX), y: depth * (NH + TB_GY) });

  const placed: Placed[] = [];
  const edges: Edge[] = [];
  const seen = new Set<string>();
  const visited = new Map<string, number>(); // pre-order rank: parents before children
  let leaf = 0;
  /** Returns the node's slot: its own leaf index, or the middle of its children's. */
  const walk = (n: TreeNode, depth: number): number => {
    seen.add(n.id);
    visited.set(n.id, visited.size);
    const cs = (kids.get(n.id) ?? []).filter((c) => !seen.has(c.id));
    const slots = cs.map((c) => { edges.push({ from: n.id, to: c.id }); return walk(c, depth + 1); });
    const slot = slots.length ? (slots[0] + slots[slots.length - 1]) / 2 : leaf++;
    placed.push({ node: n, ...at(depth, slot), depth });
    return slot;
  };
  roots.forEach((r) => walk(r, 0));
  nodes.forEach((n) => { if (!seen.has(n.id)) walk(n, 0); }); // cycles: never drop a node
  placed.sort((a, b) => visited.get(a.node.id)! - visited.get(b.node.id)!); // reading order, also the DOM order
  const width = Math.max(0, ...placed.map((p) => p.x + NW));
  const height = Math.max(0, ...placed.map((p) => p.y + NH));
  return { placed, edges, width, height };
}

/** SVG path of the connector from a parent to a child: it leaves the side of the parent that faces the child. */
export function edgePath(a: Placed, b: Placed, orientation: Orientation): string {
  if (orientation === "lr") {
    const x1 = a.x + NW, y1 = a.y + NH / 2, x2 = b.x, y2 = b.y + NH / 2, mx = (x1 + x2) / 2;
    return `M${x1},${y1} C${mx},${y1} ${mx},${y2} ${x2},${y2}`;
  }
  const x1 = a.x + NW / 2, y1 = a.y + NH, x2 = b.x + NW / 2, y2 = b.y, my = (y1 + y2) / 2;
  return `M${x1},${y1} C${x1},${my} ${x2},${my} ${x2},${y2}`;
}

/** The zoom at which the whole layout fits a w x h viewport (never above natural size). */
export function fitScale(l: Layout, w: number, h: number): number {
  if (w <= 0 || h <= 0 || l.width <= 0 || l.height <= 0) return 1;
  return Math.min(1, (w - 2 * FIT_PAD) / l.width, (h - 2 * FIT_PAD) / l.height);
}

/**
 * The orientation that shows the tree larger in a w x h viewport. Left to right reads like a timeline, so it wins
 * unless top to bottom is clearly larger (`PREFER_LR`): a short chain of rounds in a squarish or narrow viewport
 * is shown at full size as a column instead of shrunk as a row. Before there is a viewport, left to right.
 */
export function bestOrientation(nodes: TreeNode[], w: number, h: number): Orientation {
  if (w <= 0 || h <= 0 || nodes.length === 0) return "lr";
  return fitScale(layoutTree(nodes, "tb"), w, h) > fitScale(layoutTree(nodes, "lr"), w, h) * PREFER_LR ? "tb" : "lr";
}
