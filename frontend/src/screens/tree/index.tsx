import { useEffect, useMemo, useRef, useState, type CSSProperties, type KeyboardEvent } from "react";
import { Link, useParams } from "react-router-dom";
import { api } from "../../api/client";
import { fmtInt, fmtNumber, fmtUsd } from "../../api/format";
import type { RunDetail, TreeNode } from "../../api/types";
import { ApiErrorState, Badge, Card, EmptyState, LabelBanner, Spinner } from "../../components";
import { ZOOM_STEP } from "../../motion/timing";
import { NH, NW, layoutTree } from "./layout";
import { lineage } from "./path";
import { usePanZoom } from "./usePanZoom";


export default function Tree() {
  const { id = "" } = useParams();
  const [nodes, setNodes] = useState<TreeNode[] | null>(null);
  const [run, setRun] = useState<RunDetail | null>(null);
  const [error, setError] = useState<unknown>(null);
  const [runError, setRunError] = useState<unknown>(null);
  const [open, setOpen] = useState<string | null>(null);
  const [hot, setHot] = useState<string | null>(null); // node under the pointer or focus
  const els = useRef(new Map<string, SVGGElement>());

  useEffect(() => {
    setNodes(null);
    setRun(null);
    setError(null);
    setRunError(null);
    api.tree(id).then((t) => { setNodes(t.nodes); setOpen(t.nodes.find((n) => n.selected)?.id ?? null); })
      .catch((e: unknown) => setError(e));
    api.run(id).then(setRun).catch((e: unknown) => setRunError(e));
  }, [id]);

  const lay = useMemo(() => layoutTree(nodes ?? []), [nodes]);
  const current = nodes?.find((n) => n.id === open) ?? null;
  const pz = usePanZoom({ w: lay.width + 4, h: lay.height + 4 }, nodes);
  // Path root -> hovered/focused node; when nothing is hovered, the path to the selected node.
  const onPath = useMemo(() => new Set(lineage(nodes ?? [], hot ?? open)), [nodes, hot, open]);
  const placed = useMemo(() => new Map(lay.placed.map((p) => [p.node.id, p])), [lay]);

  const go = (from: string, key: string): string | null => {
    const me = placed.get(from);
    if (!me) return null;
    if (key === "ArrowLeft") { const pid = me.node.parent_id; return pid !== null && placed.has(pid) ? pid : null; }
    if (key === "ArrowRight") return lay.placed.filter((p) => p.node.parent_id === from).sort((a, b) => a.y - b.y)[0]?.node.id ?? null;
    const col = lay.placed.filter((p) => p.depth === me.depth).sort((a, b) => a.y - b.y);
    const at = col.findIndex((p) => p.node.id === from);
    return col[at + (key === "ArrowDown" ? 1 : -1)]?.node.id ?? null;
  };
  const onKey = (e: KeyboardEvent, nid: string) => {
    if (e.key === "Enter" || e.key === " ") { e.preventDefault(); setOpen(nid); }
    else if (e.key.startsWith("Arrow")) {
      const to = go(nid, e.key);
      e.preventDefault();
      e.stopPropagation();
      if (to) els.current.get(to)?.focus();
    }
  };

  return (
    <div className="stack">
      {/* The tree payload carries no dry_run flag; the label comes from the run detail. While it is
          loading nothing is shown; if it failed, the banner says the label is unknown. */}
      {(run || runError !== null) && <LabelBanner dry_run={run?.dry_run} recorded={run?.recorded} recorded_at={run?.recorded_at} />}
      {runError !== null && <ApiErrorState title="Could not load the run label" error={runError} />}
      <h1>Experiment tree</h1>
      <p><Link to={`/runs/${encodeURIComponent(id)}`}>Back to run</Link> · <Link to={`/runs/${encodeURIComponent(id)}/report`}>Report</Link></p>
      {error !== null && <ApiErrorState error={error} />}
      {error === null && (nodes === null || (run === null && runError === null)) && <Spinner label="Loading tree" />}
      {nodes && nodes.length === 0 && <EmptyState title="No sandbox branches yet">Nodes appear once the run records real sandbox lineage.</EmptyState>}
      {nodes && nodes.length > 0 && (run || runError !== null) && (
        <Card>
          <div className="pz-bar row">
            <button type="button" className="btn icon-btn" aria-label="Zoom in" onClick={() => pz.zoomBy(ZOOM_STEP)}>+</button>
            <button type="button" className="btn icon-btn" aria-label="Zoom out" onClick={() => pz.zoomBy(1 / ZOOM_STEP)}>−</button>
            <button type="button" className="btn" aria-label="Fit view" onClick={pz.fit}>Fit</button>
            <span className="muted pz-hint">Drag to pan. Keys + / − zoom (the scroll wheel too, on large trees). Arrow keys move between nodes.</span>
          </div>
          <div
            ref={pz.box} className="pz-box" tabIndex={0} role="group" aria-label="Tree viewport"
            style={{ "--h": `${lay.height + 8}px` } as CSSProperties} onKeyDown={pz.onKeyDown}
          >
            <svg className="pz-svg" role="group" aria-label="Sandbox branch tree" data-hot={hot !== null}>
              <g ref={pz.layer} className="pz-layer">
                {lay.edges.map((e) => {
                  const a = placed.get(e.from)!;
                  const b = placed.get(e.to)!;
                  const x1 = a.x + NW, y1 = a.y + NH / 2 + 2, x2 = b.x, y2 = b.y + NH / 2 + 2, mx = (x1 + x2) / 2;
                  return (
                    <path
                      key={`${e.from}-${e.to}`} pathLength={1} className={`edge${onPath.has(e.from) && onPath.has(e.to) ? " on-path" : ""}`}
                      style={{ "--depth": a.depth } as CSSProperties}
                      d={`M${x1},${y1} C${mx},${y1} ${mx},${y2} ${x2},${y2}`} fill="none" strokeWidth={2}
                    />
                  );
                })}
                {lay.placed.map(({ node: n, x, y, depth }) => {
                  const sel = n.id === open;
                  return (
                    <g key={n.id} transform={`translate(${x},${y + 2})`}>
                      <g
                        ref={(el) => { if (el) els.current.set(n.id, el); else els.current.delete(n.id); }}
                        className={`tnode${sel ? " sel" : ""}${n.selected ? " cand" : ""}${onPath.has(n.id) ? " on-path" : ""}`}
                        style={{ "--depth": depth } as CSSProperties}
                        role="button" tabIndex={0} aria-pressed={sel}
                        aria-label={`${n.label}${n.round !== null ? `, round ${n.round}` : ""}, dev score ${fmtNumber(n.dev_score)}${n.selected ? ", selected candidate" : ""}`}
                        onClick={() => setOpen(n.id)} onKeyDown={(e) => onKey(e, n.id)}
                        onPointerEnter={() => setHot(n.id)} onPointerLeave={() => setHot(null)}
                        onFocus={() => { setHot(n.id); pz.reveal(x, y, NW, NH); }} onBlur={() => setHot(null)}
                      >
                        <rect width={NW} height={NH} rx={6} />
                        <text x={10} y={22} className="t-label">{n.label.slice(0, 22)}</text>
                        <text x={10} y={40} className="t-sub">{n.round === null ? "base" : `round ${fmtInt(n.round)}`} · dev {fmtNumber(n.dev_score)}</text>
                        <text x={10} y={56} className="t-meta">{n.cost_usd === null ? "cost not measured" : fmtUsd(n.cost_usd)}{n.selected ? " · candidate" : ""}</text>
                      </g>
                    </g>
                  );
                })}
              </g>
            </svg>
          </div>
        </Card>
      )}
      {current && (
        <Card key={current.id} title={`Node: ${current.label}`} className="node-detail swap">
          <div role="region" aria-label="Node detail">
            {current.selected && <Badge tone="ok">selected candidate</Badge>}
            <dl>
              <dt>Round</dt><dd>{current.round === null ? "base (no round)" : fmtInt(current.round)}</dd>
              <dt>Hypothesis</dt><dd>{current.hypothesis ?? "none recorded"}</dd>
              <dt>Data delta</dt>
              <dd>{current.data_delta ? `${fmtInt(current.data_delta.added_rows)} rows added; families: ${current.data_delta.families.join(", ") || "none"}` : "none recorded"}</dd>
              <dt>Dev score</dt><dd>{fmtNumber(current.dev_score)}</dd>
              <dt>Cost</dt><dd>{fmtUsd(current.cost_usd)}</dd>
              <dt>Sandbox image</dt><dd className="mono">{current.sandbox_image ?? "none"}</dd>
            </dl>
          </div>
        </Card>
      )}
    </div>
  );
}
