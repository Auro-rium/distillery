import { api } from "../../api/client";
import { fmtText } from "../../api/format";
import type { Evidence } from "../../api/types";
import { ApiErrorState, EmptyState } from "../../components";
import { Spinner } from "../../ui";
import { useAsync } from "../report/useAsync";
import { Receipt } from "./Receipt";
import "./receipts.css";

/**
 * Receipts: every claim the project makes, one row each, with the file that backs it so a reader can check it
 * without trusting this page. The rows are the `receipts` of GET /api/evidence (built from the evidence files by
 * scripts/build_evidence.py); nothing is added up, rounded away or filled in here. A claim whose evidence file does
 * not exist yet is shown as pending, never with a number.
 */
export default function Receipts() {
  const [res, retry] = useAsync<Evidence>(() => api.evidence(), "evidence");
  const rows = res.state === "ok" && Array.isArray(res.data.receipts) ? res.data.receipts : null;
  return (
    <div className="rc-page">
      <header className="rc-head">
        <h1>Receipts</h1>
        <p className="muted rc-lead">
          Each claim, the status it earned, and the file in the repository that backs it. Open the artifact and check
          it yourself. A claim marked pending is pre-registered and has not been scored: it shows no number until its
          evidence file exists.
        </p>
        {res.state === "ok" && (
          <p className="muted rc-meta">
            Built from commit <code className="rc-hash num">{res.data.commit}</code> at <span className="num">{fmtText(res.data.generated_at)}</span>.
          </p>
        )}
      </header>
      {res.state === "loading" && <Spinner label="Loading the receipts" lines={4} />}
      {res.state === "error" && <ApiErrorState title="Could not load the receipts" error={res.error} onRetry={retry} />}
      {res.state === "ok" && rows === null && (
        <EmptyState title="No receipts in the evidence file">The evidence file predates receipts; rebuild it with scripts/build_evidence.py.</EmptyState>
      )}
      {rows && rows.length === 0 && <p className="muted">The evidence file lists no receipts.</p>}
      {rows && rows.length > 0 && (
        <ul className="rc-list" role="list" aria-label="Receipts">
          {rows.map((r) => <li key={r.id}><Receipt r={r} /></li>)}
        </ul>
      )}
    </div>
  );
}
