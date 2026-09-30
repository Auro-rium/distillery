import type { Layout, Orientation } from "./layout";

// Arrow keys follow the picture: towards the parent, towards the first child, and between siblings/cousins of
// the same generation. Which physical arrow means which depends on the orientation.
const KEYS: Record<Orientation, { parent: string; child: string; before: string; after: string }> = {
  lr: { parent: "ArrowLeft", child: "ArrowRight", before: "ArrowUp", after: "ArrowDown" },
  tb: { parent: "ArrowUp", child: "ArrowDown", before: "ArrowLeft", after: "ArrowRight" },
};

/** The node an arrow key leads to from `from`, or null when there is none in that direction. */
export function neighbour(lay: Layout, from: string, key: string, o: Orientation): string | null {
  const me = lay.placed.find((p) => p.node.id === from);
  if (!me) return null;
  const k = KEYS[o];
  const along = (p: { x: number; y: number }) => (o === "lr" ? p.y : p.x);
  if (key === k.parent) {
    const pid = me.node.parent_id;
    return pid !== null && lay.placed.some((p) => p.node.id === pid) ? pid : null;
  }
  if (key === k.child) {
    return lay.placed.filter((p) => p.node.parent_id === from && lay.edges.some((e) => e.from === from && e.to === p.node.id))
      .sort((a, b) => along(a) - along(b))[0]?.node.id ?? null;
  }
  if (key === k.before || key === k.after) {
    const gen = lay.placed.filter((p) => p.depth === me.depth).sort((a, b) => along(a) - along(b));
    const at = gen.findIndex((p) => p.node.id === from);
    return gen[at + (key === k.after ? 1 : -1)]?.node.id ?? null;
  }
  return null;
}
