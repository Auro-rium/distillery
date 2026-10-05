import type { ProofItem } from "../../api/types";
import { ProofCard, Sparkline } from "../../charts";
import { ApiErrorState, Spinner } from "../../components";
import type { Async } from "../report/useAsync";
import { fmtSig, proofsOf, proofStatus, rungOf } from "./mission";

function headline(p: ProofItem): string {
  const h = p.headline;
  if (!h || typeof h !== "object") return fmtSig(null);
  const v = fmtSig(h.value);
  return h.unit ? `${v} ${h.unit}` : v;
}

function Proof({ p }: { p: ProofItem }) {
  const status = proofStatus(p.verdict);
  const key = Array.isArray(p.metrics) ? p.metrics[0] : undefined;
  const series = Array.isArray(p.series) ? p.series : [];
  return (
    <ProofCard
      rung={rungOf(p)}
      title={p.title}
      status={status}
      metric={headline(p)}
      metricLabel={p.headline?.label}
      visual={series.length > 0 ? (
        <Sparkline values={series.map((s) => s.train_loss)} label="Train loss by checkpoint" format={fmtSig} showLast />
      ) : undefined}
      detail={key ? <>{key.label}: <span className="num">{fmtSig(key.value)}</span></> : undefined}
      href={p.evidence_url}
      linkLabel="Evidence"
    />
  );
}

export function ProofLadder({ res, onRetry }: { res: Async<unknown>; onRetry: () => void }) {
  const proofs = res.state === "ok" ? proofsOf(res.data) : null;
  return (
    <section className="m-section" aria-labelledby="proof-h">
      <div className="m-section-head">
        <h2 id="proof-h">Proof ladder</h2>
        <p className="muted">Each rung is a recorded experiment with its own evidence file in the repository.</p>
      </div>
      {res.state === "loading" && <Spinner label="Loading the proof ladder" lines={3} />}
      {res.state === "error" && <ApiErrorState title="Could not load the proof ladder" error={res.error} onRetry={onRetry} />}
      {res.state === "ok" && proofs === null && <p className="muted">The evidence payload has no list of proofs.</p>}
      {res.state === "ok" && proofs !== null && proofs.length === 0 && <p className="muted">The evidence file lists no proofs yet.</p>}
      {proofs && proofs.length > 0 && (
        <ul className="m-proofs" role="list" aria-label="Proof ladder">
          {proofs.map((p) => <li key={p.id}><Proof p={p} /></li>)}
        </ul>
      )}
    </section>
  );
}
