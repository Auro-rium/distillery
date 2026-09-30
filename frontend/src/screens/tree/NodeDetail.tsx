import { fmtInt, fmtNumber, fmtText, fmtUsd } from "../../api/format";
import type { TreeNode } from "../../api/types";
import { Badge } from "../../components";
import { Val } from "../live/Val";

/**
 * Everything the API recorded about one node. Null fields read "not measured" through the format helpers.
 * `path` is the chain of real ancestors from the root to `node` (following parent_id only); a crumb selects it.
 */
export function NodeDetail(props: { node: TreeNode; path: TreeNode[]; onSelect: (id: string) => void }) {
  const { node: n, path } = props;
  const delta = n.data_delta;
  return (
    <section className="card tv-detail swap" aria-label="Node detail">
      <header className="nd-head">
        <h2 className="nd-title">{n.label}</h2>
        {n.selected && <Badge tone="ok">selected candidate</Badge>}
      </header>
      <nav aria-label="Path from root">
        <ol className="crumbs">
          {path.map((a) => (
            <li key={a.id}>
              {a.id === n.id
                ? <span aria-current="location">{a.label}</span>
                : <button type="button" className="crumb" aria-label={`Go to ${a.label}`} onClick={() => props.onSelect(a.id)}>{a.label}</button>}
            </li>
          ))}
        </ol>
      </nav>
      <dl className="nd-list">
        <dt>Dev score</dt><dd className="nd-big"><Val text={fmtNumber(n.dev_score)} /></dd>
        <dt>Round</dt><dd><Val text={fmtInt(n.round)} /></dd>
        <dt>Cost</dt><dd><Val text={fmtUsd(n.cost_usd)} /></dd>
        <dt>Hypothesis</dt><dd><Val text={fmtText(n.hypothesis)} /></dd>
        <dt>Data delta</dt>
        <dd>
          {delta ? (
            <>
              <span>{fmtInt(delta.added_rows)} rows added</span>
              {delta.families.length > 0 && (
                <ul className="chips" aria-label="Families">{delta.families.map((f) => <li key={f} className="mono">{f}</li>)}</ul>
              )}
            </>
          ) : <Val text={fmtText(null)} />}
        </dd>
        <dt>Sandbox image</dt><dd className="mono nd-id"><Val text={fmtText(n.sandbox_image)} /></dd>
      </dl>
    </section>
  );
}
