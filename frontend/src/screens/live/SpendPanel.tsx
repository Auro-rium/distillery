import type { CSSProperties } from "react";
import { fmtInt, fmtUsd } from "../../api/format";
import type { Spend } from "../../api/types";
import { frac } from "../../motion/geometry";
import { Panel } from "./Panel";
import { Metric, Val } from "./Val";

// Bar lengths are geometry only (value / cap as a CSS scale factor); every number shown is the payload value,
// formatted, and always its exact final value. The bars ease to a new length because the CSS transition runs
// when `--f` changes: that transform is the only thing that moves.
const scale = (f: number | null) => ({ "--f": f ?? 0 }) as CSSProperties;

/** `spend` is undefined when the payload carries none: every figure then reads "not measured". */
export function SpendPanel({ spend, title }: { spend: Spend | null | undefined; title: string }) {
  const total = frac(spend?.total_usd, spend?.cap_usd);
  const models = Object.entries(spend?.by_model ?? {});
  return (
    <Panel title={title} className="lp-spend">
      <dl className="metrics">
        <Metric label="Total" value={<Val text={fmtUsd(spend?.total_usd)} />} hint={`cap ${fmtUsd(spend?.cap_usd)}`} />
        <Metric label="Fine-tune (estimate)" value={<Val text={fmtUsd(spend?.finetune_usd_estimate)} />} />
      </dl>
      {total !== null && spend && (
        <div className="meter" role="meter" aria-label="Spend against cap" aria-valuemin={0} aria-valuemax={spend.cap_usd} aria-valuenow={spend.total_usd}>
          <span className="meter-fill" style={scale(total)} />
        </div>
      )}
      <h3 className="lp-sub">By model</h3>
      {models.length === 0 ? (
        <p className="muted lp-none">No model calls recorded yet.</p>
      ) : (
        <ul className="models">
          {models.map(([name, v]) => {
            const f = frac(v.usd, spend?.cap_usd);
            return (
              <li key={name} className="model">
                <div className="model-head">
                  <span className="mono model-name">{name}</span>
                  <span className="model-usd"><Val text={fmtUsd(v.usd)} /></span>
                </div>
                {f !== null && <span className="mbar" aria-hidden="true"><span className="mbar-fill" style={scale(f)} /></span>}
                <dl className="model-meta">
                  <div><dt>Calls</dt><dd>{fmtInt(v.calls)}</dd></div>
                  <div><dt>In tok</dt><dd>{fmtInt(v.input_tokens)}</dd></div>
                  <div><dt>Out tok</dt><dd>{fmtInt(v.output_tokens)}</dd></div>
                </dl>
              </li>
            );
          })}
        </ul>
      )}
      {total !== null && <p className="muted lp-note">Bars show spend against the cap.</p>}
    </Panel>
  );
}
