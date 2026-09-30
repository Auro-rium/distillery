import type { TreeNode } from "../../api/types";

export const NW = 190, NH = 64, GX = 60, GY = 20;
export interface Placed { node: TreeNode; x: number; y: number; depth: number }
export interface Edge { from: string; to: string }

/** Left-to-right layout: x by depth, y by leaf order. Nodes with unknown parents become roots. */
export function layoutTree(nodes: TreeNode[]): { placed: Placed[]; edges: Edge[]; width: number; height: number } {
  const ids = new Set(nodes.map((n) => n.id));
  const kids = new Map<string, TreeNode[]>();
  const roots: TreeNode[] = [];
  for (const n of nodes) {
    if (n.parent_id !== null && ids.has(n.parent_id) && n.parent_id !== n.id) {
      kids.set(n.parent_id, [...(kids.get(n.parent_id) ?? []), n]);
    } else roots.push(n);
  }
  const placed: Placed[] = [];
  const edges: Edge[] = [];
  const seen = new Set<string>();
  let row = 0;
  const walk = (n: TreeNode, depth: number): number => {
    seen.add(n.id);
    const cs = (kids.get(n.id) ?? []).filter((c) => !seen.has(c.id));
    const ys = cs.map((c) => { edges.push({ from: n.id, to: c.id }); return walk(c, depth + 1); });
    const y = ys.length ? (ys[0] + ys[ys.length - 1]) / 2 : row++ * (NH + GY);
    placed.push({ node: n, x: depth * (NW + GX), y, depth });
    return y;
  };
  roots.forEach((r) => walk(r, 0));
  nodes.forEach((n) => { if (!seen.has(n.id)) walk(n, 0); }); // cycles: never drop a node
  const width = Math.max(0, ...placed.map((p) => p.x + NW));
  const height = Math.max(0, ...placed.map((p) => p.y + NH));
  return { placed, edges, width, height };
}
