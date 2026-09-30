import { useEffect, useMemo, useState, type KeyboardEvent } from "react";
import { Link, useParams } from "react-router-dom";
import { api } from "../../api/client";
import { fmtInt, fmtNumber, fmtUsd } from "../../api/format";
import type { RunDetail, TreeNode } from "../../api/types";
import { ApiErrorState, Badge, Card, EmptyState, LabelBanner, Spinner } from "../../components";
import { NH, NW, layoutTree } from "./layout";


export default function Tree() {
  const { id = "" } = useParams();
  const [nodes, setNodes] = useState<TreeNode[] | null>(null);
  const [run, setRun] = useState<RunDetail | null>(null);
  const [error, setError] = useState<unknown>(null);
  const [runError, setRunError] = useState<unknown>(null);
  const [open, setOpen] = useState<string | null>(null);

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

  const onKey = (e: KeyboardEvent, nid: string) => {
    if (e.key === "Enter" || e.key === " ") { e.preventDefault(); setOpen(nid); }
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
          <div style={{ overflowX: "auto" }}>
            <svg width={lay.width + 4} height={lay.height + 4} role="group" aria-label="Sandbox branch tree" style={{ display: "block", minWidth: lay.width }}>
              {lay.edges.map((e) => {
                const a = lay.placed.find((p) => p.node.id === e.from)!;
                const b = lay.placed.find((p) => p.node.id === e.to)!;
                const x1 = a.x + NW, y1 = a.y + NH / 2 + 2, x2 = b.x, y2 = b.y + NH / 2 + 2, mx = (x1 + x2) / 2;
                return <path key={`${e.from}-${e.to}`} d={`M${x1},${y1} C${mx},${y1} ${mx},${y2} ${x2},${y2}`} fill="none" stroke="var(--line)" strokeWidth={2} />;
              })}
              {lay.placed.map(({ node: n, x, y }) => {
                const sel = n.id === open;
                return (
                  <g key={n.id} transform={`translate(${x},${y + 2})`} role="button" tabIndex={0} aria-pressed={sel}
                    aria-label={`${n.label}${n.round !== null ? `, round ${n.round}` : ""}, dev score ${fmtNumber(n.dev_score)}${n.selected ? ", selected candidate" : ""}`}
                    onClick={() => setOpen(n.id)} onKeyDown={(e) => onKey(e, n.id)} style={{ cursor: "pointer" }}>
                    <rect width={NW} height={NH} rx={6} fill={sel ? "var(--surface-2)" : "var(--surface)"} stroke={sel ? "var(--accent)" : n.selected ? "var(--ok)" : "var(--line)"} strokeWidth={sel || n.selected ? 3 : 1.5} />
                    <text x={10} y={22} fill="var(--ink)" fontWeight={600} fontSize={13}>{n.label.slice(0, 22)}</text>
                    <text x={10} y={40} fill="var(--ink-2)" fontSize={12}>{n.round === null ? "base" : `round ${fmtInt(n.round)}`} · dev {fmtNumber(n.dev_score)}</text>
                    <text x={10} y={56} fill="var(--ink-2)" fontSize={11}>{n.cost_usd === null ? "cost not measured" : fmtUsd(n.cost_usd)}{n.selected ? " · candidate" : ""}</text>
                  </g>
                );
              })}
            </svg>
          </div>
        </Card>
      )}
      {current && (
        <Card title={`Node: ${current.label}`} className="node-detail">
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
