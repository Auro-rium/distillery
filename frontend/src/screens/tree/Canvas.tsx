import { useRef, type CSSProperties, type KeyboardEvent } from "react";
import { fmtInt, fmtNumber, fmtUsd } from "../../api/format";
import type { TreeNode } from "../../api/types";
import { ZOOM_STEP } from "../../motion/timing";
import { Button, IconButton, Toolbar } from "../../ui";
import { Val } from "../live/Val";
import { NH, NW, edgePath, type Layout, type Orientation } from "./layout";
import { neighbour } from "./nav";
import type { usePanZoom } from "./usePanZoom";

type PanZoom = ReturnType<typeof usePanZoom>;

/** A "round N" tag, unless the node's own label already says it. */
const roundTag = (n: TreeNode): string | null => {
  if (n.round === null) return null;
  const tag = `round ${fmtInt(n.round)}`;
  return n.label.includes(tag) ? null : tag;
};

const nodeName = (n: TreeNode) =>
  [n.label, roundTag(n), `dev score ${fmtNumber(n.dev_score)}`, n.cost_usd !== null ? `cost ${fmtUsd(n.cost_usd)}` : null, n.selected ? "selected candidate" : null]
    .filter(Boolean).join(", ");

export interface CanvasProps {
  lay: Layout;
  orientation: Orientation;
  /** The node whose detail is open. */
  open: string | null;
  onOpen: (id: string) => void;
  /** Ids from the root down to the node under the pointer or focus, else to the open node. */
  path: Set<string>;
  hot: boolean;
  onHot: (id: string | null) => void;
  pz: PanZoom;
  setBox: (el: HTMLDivElement | null) => void;
  /** "Cost per node: not measured", said once instead of on every node. */
  costNote: string | null;
}

/** Pan/zoom viewport with the real nodes as buttons and the real parent links as edges. */
export function Canvas(p: CanvasProps) {
  const { lay, orientation, pz } = p;
  const els = useRef(new Map<string, HTMLButtonElement>());
  const placed = new Map(lay.placed.map((x) => [x.node.id, x]));
  // Roving tab stop: one node is reachable by Tab (the open one), arrows move between the rest.
  const stop = p.open !== null && placed.has(p.open) ? p.open : lay.placed[0]?.node.id ?? null;

  const onNodeKey = (e: KeyboardEvent, id: string) => {
    if (e.key === "Enter" || e.key === " ") { e.preventDefault(); p.onOpen(id); return; }
    if (!e.key.startsWith("Arrow")) return;
    e.preventDefault();
    e.stopPropagation(); // arrows here move between nodes; the viewport's own arrow-panning is for when it has focus
    const to = neighbour(lay, id, e.key, orientation);
    if (to) els.current.get(to)?.focus();
  };

  return (
    <>
      <header className="tv-head">
        <h2 className="tv-title">Experiment tree</h2>
        <Toolbar label="Tree view controls" className="tv-tools">
          <IconButton label="Zoom in" onClick={() => pz.zoomBy(ZOOM_STEP)}>+</IconButton>
          <IconButton label="Zoom out" onClick={() => pz.zoomBy(1 / ZOOM_STEP)}>−</IconButton>
          <Button aria-label="Fit view" onClick={() => pz.fit()}>Fit</Button>
          <Button onClick={() => pz.showPath()}>Show path</Button>
        </Toolbar>
      </header>
      <ul className="tv-legend" aria-label="Legend">
        <li><i className="sw sw-path" aria-hidden="true" />Path from the root</li>
        <li><i className="sw sw-cand" aria-hidden="true" />Selected candidate</li>
        {p.costNote && <li className="tv-note">{p.costNote}</li>}
      </ul>
      <div
        ref={p.setBox} className="pz-box" tabIndex={0} role="group" aria-label="Tree viewport"
        data-hot={p.hot} data-orientation={orientation} onKeyDown={pz.onKeyDown}
      >
        <div ref={pz.layer} className="pz-layer" style={{ width: lay.width, height: lay.height }}>
          <svg className="tv-edges" width={lay.width} height={lay.height} aria-hidden="true" focusable="false">
            {lay.edges.map((e) => {
              const a = placed.get(e.from)!, b = placed.get(e.to)!;
              return (
                <path
                  key={`${e.from}-${e.to}`} pathLength={1} d={edgePath(a, b, orientation)}
                  className={`edge${p.path.has(e.from) && p.path.has(e.to) ? " on-path" : ""}`}
                  style={{ "--depth": a.depth } as CSSProperties}
                />
              );
            })}
          </svg>
          {lay.placed.map(({ node: n, x, y, depth }) => {
            const tag = roundTag(n);
            return (
              <button
                key={n.id} type="button" ref={(el) => { if (el) els.current.set(n.id, el); else els.current.delete(n.id); }}
                className={`tnode${n.id === p.open ? " sel" : ""}${n.selected ? " cand" : ""}${p.path.has(n.id) ? " on-path" : ""}`}
                style={{ left: x, top: y, width: NW, height: NH, "--depth": depth } as CSSProperties}
                tabIndex={n.id === stop ? 0 : -1} aria-pressed={n.id === p.open} aria-label={nodeName(n)}
                onClick={() => p.onOpen(n.id)} onKeyDown={(e) => onNodeKey(e, n.id)}
                onPointerEnter={() => p.onHot(n.id)} onPointerLeave={() => p.onHot(null)}
                onFocus={() => { p.onHot(n.id); pz.reveal(x, y, NW, NH); }}
                onBlur={() => p.onHot(null)}
              >
                <span className="tn-name" title={n.label}>{n.label}</span>
                <span className="tn-row">
                  <span className="tn-dev"><span className="tn-k">dev</span> <Val text={fmtNumber(n.dev_score)} /></span>
                  {tag && <span className="tn-tag">{tag}</span>}
                  {n.cost_usd !== null && <span className="tn-tag">{fmtUsd(n.cost_usd)}</span>}
                  {n.selected && <span className="tn-flag">candidate</span>}
                </span>
              </button>
            );
          })}
        </div>
      </div>
      <p className="tv-hint muted">
        <span className="hint-fine">Drag to pan. The buttons or + and − zoom, arrow keys move between nodes.</span>
        <span className="hint-touch">Drag to pan. Use the buttons to zoom.</span>
      </p>
    </>
  );
}
