import type { TreeNode } from "../../api/types";

/** Ids from the root down to `id`, following real parent_id links only. Cycle- and orphan-safe. */
export function lineage(nodes: TreeNode[], id: string | null): string[] {
  if (id === null) return [];
  const byId = new Map(nodes.map((n) => [n.id, n]));
  const out: string[] = [];
  const seen = new Set<string>();
  for (let cur = byId.get(id); cur && !seen.has(cur.id); cur = cur.parent_id === null ? undefined : byId.get(cur.parent_id)) {
    seen.add(cur.id);
    out.unshift(cur.id);
  }
  return out;
}
