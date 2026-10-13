import { fmtNumber, fmtPercent, fmtText } from "../../api/format";
import type { GatewayStatus } from "../../api/types";
import { Badge, Card } from "../../ui";

/** First characters of a commit for display; the full value stays in the title attribute. */
const SHORT = 7;

function Item({ label, value, title }: { label: string; value: string; title?: string }) {
  return (
    <div className="pg-st-item">
      <dt className="eyebrow">{label}</dt>
      <dd className="num" title={title}>{value}</dd>
    </div>
  );
}

/**
 * The gateway's certificate exactly as GET /api/gateway/status reports it: certified or not, alpha and delta,
 * coverage, the confidence threshold lambda and the pre-registration commit. When nothing is certified the
 * server's own reason is shown and the page says plainly where the traffic goes.
 */
export function StatusStrip({ s }: { s: GatewayStatus }) {
  return (
    <Card className="pg-status" tone={s.certified ? "ok" : "warn"} aria-labelledby="pg-status-h">
      <div className="pg-st-head">
        <h2 id="pg-status-h" className="pg-h">Gateway</h2>
        {s.certified ? <Badge tone="ok">certified</Badge> : <Badge tone="warn">not certified</Badge>}
      </div>
      {!s.certified && (
        <div className="pg-uncert" role="status">
          <p className="pg-uncert-h">Not certified yet: all traffic goes to the teacher.</p>
          <p className="muted">{fmtText(s.reason)}</p>
        </div>
      )}
      <dl className="pg-st-list">
        <Item label="alpha" value={fmtNumber(s.alpha, 2)} />
        <Item label="delta" value={fmtNumber(s.delta, 2)} />
        <Item label="coverage" value={fmtPercent(s.coverage, 1)} />
        <Item label="selective risk" value={fmtPercent(s.selective_risk, 1)} />
        <Item label="lambda" value={fmtNumber(s.lambda, 3)} />
        <Item
          label="pre-registration"
          value={s.prereg_commit ? s.prereg_commit.slice(0, SHORT) : "not recorded"}
          title={s.prereg_commit ?? undefined}
        />
        <Item label="student server" value={s.student} />
      </dl>
    </Card>
  );
}
