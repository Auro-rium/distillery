import { useId, type CSSProperties } from "react";
import { LAYOUTS, NEXT_ROUND, type Layout } from "./flow";
import "./replay.css";

// Static explainer of the method. It carries no numbers and no run data; its motion (routes drawing in,
// the return path's dashes moving) is decoration, not progress, and touches only stroke and opacity.
// Two drawings of the same stages, one shown at a time by CSS: a wide flow and a tall stepper for phones.
// Both are hidden from assistive technology; the list below carries the same steps as text.

const STEPS = [
  "The task pack supplies a schema and questions.",
  "Gold answers are cross-checked by executing them.",
  "The held-out set is sealed; only the evaluator reads it.",
  "The teacher writes training data that is verified by execution.",
  "The student is fine-tuned on it.",
  "Dev evaluation finds failures and clusters them.",
  "Targeted new data is added in a sandbox branch, and the student is fine-tuned again.",
  "The best round is evaluated once on the held-out set for base, student and teacher.",
  "A fixed gate decides PROMOTE or REJECT.",
];

// Text baselines inside a node (offsets from its top edge).
const TITLE_DY = 26;
const SUB_DY = 46;
const CORNER = 8;

function Diagram({ kind, l }: { kind: "wide" | "tall"; l: Layout }) {
  const uid = useId().replace(/[^a-zA-Z0-9_-]/g, "");
  const head = `${uid}-head`;
  const headLoop = `${uid}-head-loop`;
  const loop = l.edges.find((e) => e.kind === "loop");
  return (
    <svg className={`flow flow-${kind}`} viewBox={`0 0 ${l.width} ${l.height}`} aria-hidden="true" focusable="false">
      <defs>
        <marker id={head} viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto">
          <path d="M0,0 L10,5 L0,10 z" className="flow-head" />
        </marker>
        <marker id={headLoop} viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto">
          <path d="M0,0 L10,5 L0,10 z" className="flow-head loop" />
        </marker>
      </defs>
      {l.edges.map((e, i) => (
        <path
          key={e.id}
          d={e.d}
          pathLength={1}
          className={e.kind === "loop" ? "flow-edge loop" : "flow-edge"}
          markerEnd={`url(#${e.kind === "loop" ? headLoop : head})`}
          style={{ "--i": i } as CSSProperties}
        />
      ))}
      {loop && <path d={loop.d} className="flow-march" />}
      <text x={l.label.x} y={l.label.y} textAnchor="middle" className="flow-note">{NEXT_ROUND}</text>
      {l.nodes.map((n, i) => (
        <g key={n.id} className="flow-node" style={{ "--i": i } as CSSProperties}>
          <rect x={n.x} y={n.y} width={n.w} height={n.h} rx={CORNER} />
          <text x={n.x + l.pad} y={n.y + TITLE_DY} className="flow-title">{n.title}</text>
          <text x={n.x + l.pad} y={n.y + SUB_DY} className="flow-sub">{n.sub}</text>
        </g>
      ))}
    </svg>
  );
}

export function Pipeline() {
  return (
    <figure className="pipe">
      <Diagram kind="wide" l={LAYOUTS.wide} />
      <Diagram kind="tall" l={LAYOUTS.tall} />
      <ol className="sr-only">{STEPS.map((s) => <li key={s}>{s}</li>)}</ol>
      <figcaption className="muted">The method as a diagram. It shows no run&rsquo;s progress and no results.</figcaption>
    </figure>
  );
}
