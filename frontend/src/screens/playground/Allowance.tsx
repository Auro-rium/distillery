import type { CSSProperties } from "react";
import { fmtInt, fmtUsd } from "../../api/format";
import type { Config } from "../../api/types";
import { frac } from "../../motion/geometry";
import { Card } from "../../ui";

// Bar length is geometry only (spent / cap as a CSS scale factor). Every figure shown is the API's own,
// formatted; nothing is subtracted, so no "remaining" is claimed: the API does not return one.
const scale = (f: number) => ({ "--f": f }) as CSSProperties;

export function Allowance({ pg }: { pg: Config["playground"] }) {
  const f = frac(pg.spent_today_usd, pg.daily_cap_usd);
  return (
    <Card className="pg-allow">
      <h2 className="pg-h">Allowance</h2>
      <p className="pg-limit">Limit {fmtInt(pg.per_ip_per_hour)} requests per hour per IP</p>
      <p className="pg-spent">daily budget spent {fmtUsd(pg.spent_today_usd, 2)} of {fmtUsd(pg.daily_cap_usd, 2)}</p>
      {f !== null && (
        <div className="meter pg-meter" role="meter" aria-label="Daily budget spent, against the cap" aria-valuemin={0} aria-valuemax={pg.daily_cap_usd} aria-valuenow={pg.spent_today_usd}>
          <span className="meter-fill" style={scale(f)} />
        </div>
      )}
      <p className="muted pg-note">As reported by the server.</p>
    </Card>
  );
}
