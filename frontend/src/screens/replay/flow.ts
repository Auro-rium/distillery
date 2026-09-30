// Geometry of the method diagram on the Replay landing page. Pure data and arithmetic, no run data and no
// displayed numbers: every number here is a coordinate in a drawing. Two layouts are drawn from the same
// stages: a wide one (four columns, three rows) and a tall stepper for phones. Each is laid out so that
// text fits its node, no route passes through a node, and the loop label touches nothing; flow.test.ts
// proves that with plain geometry, and the page is also measured in a real browser.
//
// Stage names are the pipeline's own (docs/ARCHITECTURE.md): schema, questions, gold_crosscheck, split,
// teacher_data, finetune, dev_eval, analysis, targeted, sandbox_branch, final_eval, gate.

export interface Pt { x: number; y: number }
export interface FlowNode { id: string; x: number; y: number; w: number; h: number; title: string; sub: string }
export interface FlowEdge { id: string; from: string; to: string; kind: "step" | "loop"; d: string; pts: Pt[] }
export interface Layout {
  width: number;
  height: number;
  titlePx: number;
  subPx: number;
  labelPx: number;
  /** Space between a node's edge and its text. */
  pad: number;
  nodes: FlowNode[];
  edges: FlowEdge[];
  label: { text: string; x: number; y: number };
}

export const STAGE_IDS = ["a", "b", "c", "d", "e", "f", "g", "h", "i"] as const;
type Id = (typeof STAGE_IDS)[number];

const STAGES: Record<Id, { title: string; sub: string }> = {
  a: { title: "Schema + questions", sub: "the task pack" },
  b: { title: "Gold cross-check", sub: "verified by execution" },
  c: { title: "Seal held-out set", sub: "evaluator access only" },
  d: { title: "Teacher data", sub: "verified examples" },
  e: { title: "Fine-tune student", sub: "small model" },
  f: { title: "Dev eval + failures", sub: "cluster what broke" },
  g: { title: "Targeted data", sub: "sandbox branch" },
  h: { title: "Final held-out eval", sub: "base, student, teacher" },
  i: { title: "Gate", sub: "PROMOTE or REJECT" },
};

export const NEXT_ROUND = "next round";

/**
 * A deliberately generous width for text drawn in the app's fonts (or the fallbacks a browser uses while
 * they load): the widest common glyph widths, so a node sized from it never clips.
 */
export function textWidth(text: string, px: number, kind: "sans" | "mono"): number {
  return text.length * px * (kind === "mono" ? 0.62 : 0.6);
}

type Seg = readonly ["L", number, number] | readonly ["Q", number, number, number, number];
const STEP_PX = 2;
const CURVE_STEPS = 24;

/** A route as an SVG path plus the points along it (used only to check geometry). */
function route(x0: number, y0: number, segs: readonly Seg[]): { d: string; pts: Pt[] } {
  let d = `M${x0},${y0}`;
  const pts: Pt[] = [{ x: x0, y: y0 }];
  let cx = x0, cy = y0;
  for (const s of segs) {
    if (s[0] === "L") {
      const [, x, y] = s;
      d += ` L${x},${y}`;
      const n = Math.max(1, Math.ceil(Math.hypot(x - cx, y - cy) / STEP_PX));
      for (let i = 1; i <= n; i++) pts.push({ x: cx + ((x - cx) * i) / n, y: cy + ((y - cy) * i) / n });
      cx = x; cy = y;
    } else {
      const [, qx, qy, x, y] = s;
      d += ` Q${qx},${qy} ${x},${y}`;
      for (let i = 1; i <= CURVE_STEPS; i++) {
        const t = i / CURVE_STEPS, u = 1 - t;
        pts.push({ x: u * u * cx + 2 * u * t * qx + t * t * x, y: u * u * cy + 2 * u * t * qy + t * t * y });
      }
      cx = x; cy = y;
    }
  }
  return { d, pts };
}

function build(
  spec: { width: number; height: number; w: Record<Id, number>; h: number; pos: Record<Id, Pt>; titlePx: number; subPx: number; labelPx: number; pad: number },
  edges: { id: string; from: Id; to: Id; kind: "step" | "loop"; x: number; y: number; segs: Seg[] }[],
  label: Layout["label"],
): Layout {
  return {
    width: spec.width, height: spec.height, titlePx: spec.titlePx, subPx: spec.subPx, labelPx: spec.labelPx, pad: spec.pad,
    nodes: STAGE_IDS.map((id) => ({ id, ...spec.pos[id], w: spec.w[id], h: spec.h, ...STAGES[id] })),
    edges: edges.map((e) => ({ id: e.id, from: e.from, to: e.to, kind: e.kind, ...route(e.x, e.y, e.segs) })),
    label,
  };
}

// ---------- wide: a b c d / e f h i / g under f, with the loop returning from g to e ----------
function wide(): Layout {
  const W = 190, H = 60, m = 2; // m: room for the stroke at the canvas edge
  const col = [m, m + 240, m + 480, m + 720];
  const row = [m, m + 130, m + 260];
  const pos = {
    a: { x: col[0], y: row[0] }, b: { x: col[1], y: row[0] }, c: { x: col[2], y: row[0] }, d: { x: col[3], y: row[0] },
    e: { x: col[0], y: row[1] }, f: { x: col[1], y: row[1] }, h: { x: col[2], y: row[1] }, i: { x: col[3], y: row[1] },
    g: { x: col[1], y: row[2] },
  } as Record<Id, Pt>;
  const w = Object.fromEntries(STAGE_IDS.map((k) => [k, W])) as Record<Id, number>;
  const mid = H / 2, cx = W / 2, R = 12, BIG = 28;
  const across = (a: Id, b: Id) => ({ x: pos[a].x + W, y: pos[a].y + mid, segs: [["L", pos[b].x, pos[b].y + mid]] as Seg[] });
  const dx = pos.d.x + cx, ex = pos.e.x + cx, gap = row[0] + H + (row[1] - row[0] - H) / 2;
  const fx = pos.f.x + cx, gy = pos.g.y + mid;
  return build(
    { width: col[3] + W + m, height: row[2] + H + m, w, h: H, pos, titlePx: 13, subPx: 11, labelPx: 11, pad: 12 },
    [
      { id: "a-b", from: "a", to: "b", kind: "step", ...across("a", "b") },
      { id: "b-c", from: "b", to: "c", kind: "step", ...across("b", "c") },
      { id: "c-d", from: "c", to: "d", kind: "step", ...across("c", "d") },
      // d to e: down into the gap between the rows, left along it (nothing is drawn there), down into e
      { id: "d-e", from: "d", to: "e", kind: "step", x: dx, y: row[0] + H, segs: [["L", dx, gap - R], ["Q", dx, gap, dx - R, gap], ["L", ex + R, gap], ["Q", ex, gap, ex, gap + R], ["L", ex, row[1]]] },
      { id: "e-f", from: "e", to: "f", kind: "step", ...across("e", "f") },
      { id: "f-h", from: "f", to: "h", kind: "step", ...across("f", "h") },
      { id: "h-i", from: "h", to: "i", kind: "step", ...across("h", "i") },
      { id: "f-g", from: "f", to: "g", kind: "step", x: fx, y: row[1] + H, segs: [["L", fx, row[2]]] },
      // the return path: out of g to the left, one wide curve, up into the bottom of e
      { id: "g-e", from: "g", to: "e", kind: "loop", x: pos.g.x, y: gy, segs: [["L", ex + BIG, gy], ["Q", ex, gy, ex, gy - BIG], ["L", ex, row[1] + H]] },
    ],
    { text: NEXT_ROUND, x: (ex + BIG + pos.g.x) / 2, y: gy - 12 },
  );
}

// ---------- tall: one column for phones, Targeted data in a side lane with the loop around it ----------
function tall(): Layout {
  const SW = 176, LW = 128, H = 60, m = 2, GAP = 34, LOOP_GAP = 66, R = 14;
  const ys: number[] = [];
  const order: Id[] = ["a", "b", "c", "d", "e", "f", "h", "i"];
  let y = m;
  order.forEach((id, i) => {
    ys.push(y);
    y += H + (id === "e" ? LOOP_GAP : i === order.length - 1 ? 0 : GAP);
  });
  const pos = { g: { x: 0, y: 0 } } as Record<Id, Pt>;
  order.forEach((id, i) => { pos[id] = { x: m, y: ys[i] }; });
  const laneX = m + SW + 34; // 34: room for the return path between the spine and the lane
  const eMid = pos.e.y + H / 2, fMid = pos.f.y + H / 2;
  pos.g = { x: laneX, y: Math.round((eMid + fMid) / 2 - H / 2) };
  const w = Object.fromEntries(STAGE_IDS.map((k) => [k, k === "g" ? LW : SW])) as Record<Id, number>;
  const cx = m + SW / 2, gcx = laneX + LW / 2;
  const down = (a: Id, b: Id) => ({ x: cx, y: pos[a].y + H, segs: [["L", cx, pos[b].y]] as Seg[] });
  const spineRight = m + SW;
  return build(
    { width: laneX + LW + m, height: pos.i.y + H + m, w, h: H, pos, titlePx: 13, subPx: 11, labelPx: 11, pad: 12 },
    [
      { id: "a-b", from: "a", to: "b", kind: "step", ...down("a", "b") },
      { id: "b-c", from: "b", to: "c", kind: "step", ...down("b", "c") },
      { id: "c-d", from: "c", to: "d", kind: "step", ...down("c", "d") },
      { id: "d-e", from: "d", to: "e", kind: "step", ...down("d", "e") },
      { id: "e-f", from: "e", to: "f", kind: "step", ...down("e", "f") },
      { id: "f-h", from: "f", to: "h", kind: "step", ...down("f", "h") },
      { id: "h-i", from: "h", to: "i", kind: "step", ...down("h", "i") },
      // f to g: right out of f, one curve up into the bottom of g
      { id: "f-g", from: "f", to: "g", kind: "step", x: spineRight, y: fMid, segs: [["L", gcx - R, fMid], ["Q", gcx, fMid, gcx, fMid - R], ["L", gcx, pos.g.y + H]] },
      // the return path: up out of g, one curve, left into the side of e
      { id: "g-e", from: "g", to: "e", kind: "loop", x: gcx, y: pos.g.y, segs: [["L", gcx, eMid + R], ["Q", gcx, eMid, gcx - R, eMid], ["L", spineRight, eMid]] },
    ],
    { text: NEXT_ROUND, x: Math.round((spineRight + gcx - R) / 2), y: eMid - 12 },
  );
}

export const LAYOUTS = { wide: wide(), tall: tall() } as const;
