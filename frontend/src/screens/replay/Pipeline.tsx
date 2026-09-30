import type { CSSProperties } from "react";

// Static explainer of the method. Stage names are the pipeline's own (docs/ARCHITECTURE.md:
// schema, questions, gold_crosscheck, split, teacher_data, finetune, dev_eval, analysis,
// targeted, sandbox_branch, final_eval, gate). It carries no numbers and no run data, and its
// motion (marching dashes, staggered appearance) is decoration, not progress.

interface Node { id: string; x: number; y: number; title: string; sub: string }
const W = 170, H = 56;
const NODES: Node[] = [
  { id: "a", x: 10, y: 14, title: "Schema + questions", sub: "task pack" },
  { id: "b", x: 200, y: 14, title: "Gold cross-check", sub: "verified by execution" },
  { id: "c", x: 390, y: 14, title: "Seal held-out set", sub: "only the evaluator sees it" },
  { id: "d", x: 580, y: 14, title: "Teacher data", sub: "verified examples" },
  { id: "e", x: 580, y: 130, title: "Fine-tune student", sub: "small model" },
  { id: "f", x: 390, y: 130, title: "Dev eval + failures", sub: "cluster what broke" },
  { id: "g", x: 200, y: 130, title: "Targeted data", sub: "in a sandbox branch" },
  { id: "h", x: 390, y: 232, title: "Final held-out eval", sub: "base, student, teacher" },
  { id: "i", x: 200, y: 232, title: "Gate", sub: "PROMOTE or REJECT" },
];
// Straight arrows (from, to) and the one curved loop-back.
const ARROWS: [string, string][] = [["a", "b"], ["b", "c"], ["c", "d"], ["d", "e"], ["e", "f"], ["f", "g"], ["f", "h"], ["h", "i"]];
const at = (id: string) => NODES.find((n) => n.id === id)!;

function line(from: string, to: string): string {
  const a = at(from), b = at(to);
  if (a.y === b.y) { // same row: side to side
    const dir = b.x > a.x ? 1 : -1;
    return `M${a.x + (dir > 0 ? W : 0)},${a.y + H / 2} L${b.x + (dir > 0 ? 0 : W)},${b.y + H / 2}`;
  }
  return `M${a.x + W / 2},${a.y + H} L${b.x + W / 2},${b.y}`; // above to below
}
const LOOP = `M${at("g").x + W / 2},${at("g").y} C${at("g").x + W / 2},${at("g").y - 48} ${at("e").x + W * 0.8},${at("e").y - 48} ${at("e").x + W * 0.8},${at("e").y}`;

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

export function Pipeline() {
  return (
    <figure className="pipe">
      <div className="pipe-scroll">
        <svg className="pipe-svg" viewBox="0 0 760 300" aria-hidden="true" focusable="false">
          <defs>
            <marker id="pipe-arrow" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
              <path d="M0,0 L10,5 L0,10 z" className="pipe-head" />
            </marker>
          </defs>
          {ARROWS.map(([f, t], i) => (
            <path key={`${f}${t}`} d={line(f, t)} className="pipe-line" markerEnd="url(#pipe-arrow)" style={{ "--i": i } as CSSProperties} />
          ))}
          <path d={LOOP} className="pipe-line pipe-loop" markerEnd="url(#pipe-arrow)" />
          <text x={at("g").x + W + 70} y={at("g").y - 30} className="pipe-note" textAnchor="middle">next round</text>
          {NODES.map((n, i) => (
            <g key={n.id} transform={`translate(${n.x},${n.y})`}>
              <g className="pipe-node" style={{ "--i": i } as CSSProperties}>
                <rect width={W} height={H} rx={8} />
                <text x={12} y={24} className="pipe-title">{n.title}</text>
                <text x={12} y={43} className="pipe-sub">{n.sub}</text>
              </g>
            </g>
          ))}
        </svg>
      </div>
      <ol className="sr-only">{STEPS.map((s) => <li key={s}>{s}</li>)}</ol>
      <figcaption className="muted">The method as a diagram. It shows no run's progress and no results.</figcaption>
    </figure>
  );
}
