import type { CSSProperties } from "react";
import { fmtInt, fmtUsd } from "../../api/format";
import type { Spend } from "../../api/types";
import { CountUp } from "../../motion/CountUp";
import { frac } from "../../motion/geometry";

// Bar lengths are geometry only (value / cap as a CSS scale factor); the numbers shown are the
// payload values. The bars ease to each new value because the CSS transition runs on change.
const scale = (f: number | null) => ({ "--f": f ?? 0 }) as CSSProperties;

export function SpendMeter({ spend }: { spend: Spend }) {
  const total = frac(spend.total_usd, spend.cap_usd);
  const models = Object.entries(spend.by_model);
  return (
    <>
      {total !== null && (
        <div className="meter" role="meter" aria-label="Spend against cap" aria-valuemin={0} aria-valuemax={spend.cap_usd} aria-valuenow={spend.total_usd}>
          <span className="meter-fill" style={scale(total)} />
        </div>
      )}
      <div className="tbl-wrap">
      <table className="tbl spend-tbl">
        <thead>
          <tr>
            <th align="left">Model</th><th align="left">Against cap</th><th align="right">USD</th>
            <th align="right">Calls</th><th align="right">In tok</th><th align="right">Out tok</th>
          </tr>
        </thead>
        <tbody>
          {models.map(([m, v]) => (
            <tr key={m}>
              <td className="mono">{m}</td>
              <td className="mbar-cell">
                {frac(v.usd, spend.cap_usd) !== null && <span className="mbar"><span className="mbar-fill" style={scale(frac(v.usd, spend.cap_usd))} /></span>}
              </td>
              <td align="right"><CountUp value={v.usd} format={fmtUsd} /></td>
              <td align="right">{fmtInt(v.calls)}</td>
              <td align="right">{fmtInt(v.input_tokens)}</td>
              <td align="right">{fmtInt(v.output_tokens)}</td>
            </tr>
          ))}
        </tbody>
      </table>
      </div>
    </>
  );
}
