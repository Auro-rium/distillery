import { useEffect, useMemo, useState } from "react";
import { useParams } from "react-router-dom";
import { api } from "../../api/client";
import { fmtUsd } from "../../api/format";
import type { RunDetail, TreeNode } from "../../api/types";
import { ApiErrorState, EmptyState, LabelBanner, Spinner } from "../../components";
import { RunShell } from "../../components/RunShell";
import { Canvas } from "./Canvas";
import { NH, NW, bestOrientation, layoutTree } from "./layout";
import { NodeDetail } from "./NodeDetail";
import { lineage } from "./path";
import { useElementSize, usePanZoom } from "./usePanZoom";
import "./tree.css";

/** The node to show first: the candidate the API selected, else the first root, else the first node. */
function firstOpen(nodes: TreeNode[]): string | null {
  const ids = new Set(nodes.map((n) => n.id));
  return (nodes.find((n) => n.selected) ?? nodes.find((n) => n.parent_id === null || !ids.has(n.parent_id)) ?? nodes[0])?.id ?? null;
}

/**
 * Experiment tree inside the shared RunShell: the tree (master) beside the selected node's detail, below it on
 * phones. Nodes and edges come only from GET /api/runs/{id}/tree; nothing is invented.
 */
export default function Tree() {
  const { id = "" } = useParams();
  const [nodes, setNodes] = useState<TreeNode[] | null>(null);
  const [run, setRun] = useState<RunDetail | null>(null);
  const [error, setError] = useState<unknown>(null);
  const [runError, setRunError] = useState<unknown>(null);
  const [open, setOpen] = useState<string | null>(null);
  const [hot, setHot] = useState<string | null>(null); // node under the pointer or focus
  const [box, setBox] = useState<HTMLDivElement | null>(null);

  useEffect(() => {
    setNodes(null);
    setRun(null);
    setError(null);
    setRunError(null);
    api.tree(id).then((t) => { setNodes(t.nodes); setOpen(firstOpen(t.nodes)); }).catch((e: unknown) => setError(e));
    api.run(id).then(setRun).catch((e: unknown) => setRunError(e));
  }, [id]);

  const size = useElementSize(box);
  const orientation = useMemo(() => bestOrientation(nodes ?? [], size.w, size.h), [nodes, size.w, size.h]);
  const lay = useMemo(() => layoutTree(nodes ?? [], orientation), [nodes, orientation]);
  const byId = useMemo(() => new Map((nodes ?? []).map((n) => [n.id, n])), [nodes]);
  const current = open !== null ? byId.get(open) ?? null : null;
  // Path root -> hovered/focused node; when nothing is hovered, the path to the open node.
  const trail = useMemo(() => lineage(nodes ?? [], hot ?? open), [nodes, hot, open]);
  const onPath = useMemo(() => new Set(trail), [trail]);
  const crumbs = useMemo(() => lineage(nodes ?? [], open).map((i) => byId.get(i)!), [nodes, open, byId]);
  // Boxes of the path root -> open node: what the view is framed on when the whole tree is too big to read.
  const focus = useMemo(() => {
    const at = new Map(lay.placed.map((p) => [p.node.id, p]));
    return crumbs.flatMap((c) => { const p = at.get(c.id); return p ? [{ x: p.x, y: p.y, w: NW, h: NH }] : []; });
  }, [lay, crumbs]);
  const pz = usePanZoom(box, { w: lay.width, h: lay.height }, nodes, focus);
  const select = (id: string) => {
    setOpen(id);
    const p = lay.placed.find((x) => x.node.id === id);
    if (p) pz.reveal(p.x, p.y, NW, NH); // chosen from the detail: bring it into view if it is off screen
  };
  const costNote = nodes && nodes.length > 0 && nodes.every((n) => n.cost_usd === null) ? `Cost per node: ${fmtUsd(null)}` : null;

  // The tree payload carries no dry_run flag; the label comes from the run detail (shown by RunShell). No tree
  // number is shown before it is known; if it could not be loaded, the banner says the label is unknown.
  const labelKnown = run !== null || runError !== null;
  const ready = nodes !== null && labelKnown;

  return (
    <RunShell runId={id} tab="tree" run={run ?? undefined}>
      <div className="treev">
        {runError !== null && (
          <>
            <LabelBanner />
            <ApiErrorState title="Could not load the run label" error={runError} />
          </>
        )}
        {error !== null && <ApiErrorState error={error} />}
        {error === null && !ready && <Spinner label="Loading tree" />}
        {error === null && ready && nodes.length === 0 && (
          <EmptyState title="No sandbox branches yet">Nodes appear once the run records real sandbox lineage.</EmptyState>
        )}
        {error === null && ready && nodes.length > 0 && (
          <div className="tv-layout">
            <section className="card tv-canvas" aria-label="Experiment tree" data-orientation={orientation}>
              <Canvas
                lay={lay} orientation={orientation} open={open} onOpen={setOpen} path={onPath} hot={hot !== null}
                onHot={setHot} pz={pz} setBox={setBox} costNote={costNote}
              />
            </section>
            {current && <NodeDetail key={current.id} node={current} path={crumbs} onSelect={select} />}
          </div>
        )}
      </div>
    </RunShell>
  );
}
