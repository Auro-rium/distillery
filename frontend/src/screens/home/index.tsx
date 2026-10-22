// Home (`/`): what Distillery is for a visitor who has a minute. The product first (one base-URL change, a certified
// specialist, the teacher for the rest, a drift guard), then this server's live gateway state, then the pre-registered
// claims as receipts. Every number comes from /api/gateway/status, /api/gateway/drift, /api/config or /api/evidence;
// the copy on this page carries no figures of its own.
import { Link } from "react-router-dom";
import { api } from "../../api/client";
import { fmtNumber, fmtPercent, fmtText } from "../../api/format";
import type { Config, Evidence, GatewayDrift, GatewayStatus, Receipt } from "../../api/types";
import { ApiErrorState, Badge, Card } from "../../components";
import { GITHUB_URL } from "../../components/Shell";
import { GlowText } from "../../motion/GlowText";
import { Skeleton } from "../../ui";
import { useAsync, type Async } from "../report/useAsync";
import "./home.css";

/** The pre-registered serving claims, in reading order. Others (the distillation rungs) live on the Mission page. */
export const FEATURED = ["W-C1", "W-C2", "W-D", "W-E2E", "W-G", "W-$"] as const;

const STATUS_TEXT: Record<Receipt["status"], string> = { verified: "Verified", failed: "Not met", pending: "Pending" };
const STATUS_TONE = { verified: "ok", failed: "bad", pending: "warn" } as const;

/** What the drift state means for a visitor; chosen by the state only, no figures. */
export function driftText(state: string | null | undefined): string {
  if (state === "CERTIFIED") return "the student may serve what the certificate covers";
  if (state === "ROLLBACK" || state === "TEACHER")
    return "recent traffic did not look like what the certificate covers, so the teacher answers everything until the router is re-certified";
  return "not configured on this server";
}

const short = (sha: string | null | undefined) => (sha ? `${sha.slice(0, 8)}…` : fmtText(null));

function Hero({ config }: { config: Async<Config> }) {
  const models = config.state === "ok" ? config.data.models : null;
  return (
    <section className="h-hero" aria-labelledby="h-title">
      <p className="eyebrow">Nebius Token Factory · NVIDIA Nemotron</p>
      <GlowText as="h1" id="h-title">
        An autonomous agent that distils a big model’s narrow skill into a small, certified student, served behind an OpenAI-compatible gateway with automatic rollback.
      </GlowText>
      <GlowText className="h-flow">
        Write → Test in Sandboxes → Fine-tune on Token Factory → Gate → Certify the router → Serve → Guard
      </GlowText>
      {models && models.teacher && models.student && (
        <dl className="h-models" aria-label="Models on this server">
          <div><dt>Teacher</dt><dd className="mono">{fmtText(models.teacher)}</dd></div>
          <div><dt>Student</dt><dd className="mono">{fmtText(models.student)}</dd></div>
        </dl>
      )}
      <div className="h-cta">
        <Link className="btn primary" to="/receipts">Check the receipts</Link>
        <a className="btn ghost" href={GITHUB_URL} target="_blank" rel="noreferrer noopener">Source on GitHub</a>
      </div>
    </section>
  );
}

function Swap() {
  const origin = typeof window !== "undefined" ? window.location.origin : "";
  return (
    <section className="h-swap" aria-labelledby="h-swap-title">
      <h2 id="h-swap-title">The only change in your agent</h2>
      <pre className="h-code" aria-label="Before and after">
        <code>
          <span className="h-del">- client = OpenAI(base_url="https://api.tokenfactory.nebius.com/v1/")</span>{"\n"}
          <span className="h-add">+ client = OpenAI(base_url="{origin}/v1")</span>
        </code>
      </pre>
      <p className="muted">
        Same request, same response format. Each answer carries who served it, the confidence, the threshold and the
        reason in a <code>distillery</code> field and in response headers.
      </p>
    </section>
  );
}

const STEPS: { title: string; body: string }[] = [
  { title: "Familiar?", body: "The question is embedded and compared with what the student was certified on. Unfamiliar questions go straight to the teacher, before the student spends anything." },
  { title: "Student drafts", body: "The fine-tuned student writes the SQL on a Nebius sandbox, and the SQL is run read-only against the database." },
  { title: "Router decides", body: "A small model scores confidence from whether the SQL ran, the student's token log-probabilities and the similarity. Only a score at or above the certified threshold is served." },
  { title: "Drift guard", body: "The share of deferred requests is watched over a sliding window and a sample of student answers is audited by the teacher. A burst of unfamiliar traffic rolls the gateway back." },
];

function Flow() {
  return (
    <section aria-labelledby="h-flow-title">
      <h2 id="h-flow-title">How one request is served</h2>
      <ol className="h-flow">
        {STEPS.map((s, i) => (
          <li key={s.title} className="h-step">
            <span className="h-step-n mono" aria-hidden="true">{i + 1}</span>
            <h3>{s.title}</h3>
            <p className="muted">{s.body}</p>
          </li>
        ))}
      </ol>
    </section>
  );
}

function Live({ status, drift }: { status: Async<GatewayStatus>; drift: Async<GatewayDrift> }) {
  if (status.state === "loading") return <Skeleton className="h-skel" />;
  if (status.state === "error") return <ApiErrorState title="Could not read the gateway" error={status.error} />;
  const s = status.data;
  const d = drift.state === "ok" ? drift.data : null;
  const pinned = s.certified_adapter_sha256 != null && s.student_adapter_sha256 === s.certified_adapter_sha256;
  return (
    <section aria-labelledby="h-live-title">
      <div className="h-sec-head">
        <h2 id="h-live-title">This server, right now</h2>
        <Badge tone={s.certified ? "ok" : "warn"}>{s.certified ? "Certified" : "Not certified"}</Badge>
      </div>
      {!s.certified && <p className="muted">{fmtText(s.reason)}</p>}
      <div className="h-live">
        <Card className="h-tile"><div className="eyebrow">Threshold λ̂</div><div className="num h-big">{fmtNumber(s.lambda, 3)}</div><p className="muted">served only at or above this confidence</p></Card>
        <Card className="h-tile"><div className="eyebrow">Error bound α</div><div className="num h-big">{fmtPercent(s.alpha, 0)}</div><p className="muted">certified at confidence 1 − {fmtNumber(s.delta, 2)}</p></Card>
        <Card className="h-tile"><div className="eyebrow">Calibration coverage</div><div className="num h-big">{fmtPercent(s.coverage, 1)}</div><p className="muted">share of calibration questions the student would take</p></Card>
        <Card className="h-tile">
          <div className="eyebrow">Drift guard</div>
          <div className="num h-big">{d ? fmtText(d.state) : fmtText(s.drift.state)}</div>
          <p className="muted">{driftText(d && d.enabled === false ? null : d ? d.state : s.drift.state)}</p>
        </Card>
      </div>
      <p className="muted h-pin">
        Student server: <span className="mono">{fmtText(s.student)}</span>
        {s.student_adapter_sha256 ? (
          <> · checkpoint <span className="mono" title={s.student_adapter_sha256}>{short(s.student_adapter_sha256)}</span>{" "}
            {pinned ? "is the one the certificate was measured on" : s.certified_adapter_sha256 ? "is not the certified checkpoint, so the teacher answers" : ""}</>
        ) : s.certified_adapter_sha256 ? " · no shipped checkpoint is served here, so the teacher answers" : null}
        {s.prereg_commit && <> · pre-registered at <span className="mono">{s.prereg_commit.slice(0, 7)}</span></>}
      </p>
    </section>
  );
}

function Claims({ evidence }: { evidence: Async<Evidence> }) {
  if (evidence.state === "loading") return <Skeleton className="h-skel" />;
  if (evidence.state === "error") return <ApiErrorState title="Could not load the receipts" error={evidence.error} />;
  const rows = (evidence.data.receipts ?? []).filter((r) => (FEATURED as readonly string[]).includes(r.id));
  rows.sort((a, b) => FEATURED.indexOf(a.id as (typeof FEATURED)[number]) - FEATURED.indexOf(b.id as (typeof FEATURED)[number]));
  return (
    <section aria-labelledby="h-claims-title">
      <div className="h-sec-head">
        <h2 id="h-claims-title">What has been shown, with the file that shows it</h2>
        <Link to="/receipts">All receipts →</Link>
      </div>
      <ul className="h-claims" role="list">
        {rows.map((r) => (
          <li key={r.id}>
            <Card className="h-claim lift">
              <div className="h-claim-head">
                <span className="mono h-claim-id">{r.id}</span>
                <Badge tone={STATUS_TONE[r.status]}>{STATUS_TEXT[r.status]}</Badge>
              </div>
              <p className="h-claim-text">{r.claim}</p>
              {r.status === "pending"
                ? <p className="muted">Pre-registered, not yet scored.</p>
                : <p className="h-claim-value">{fmtText(r.value_display)}</p>}
              {r.status !== "pending" && <a href={r.permalink} target="_blank" rel="noreferrer noopener">Evidence file →</a>}
            </Card>
          </li>
        ))}
      </ul>
    </section>
  );
}

function Limits() {
  return (
    <section aria-labelledby="h-limits-title" className="h-limits">
      <h2 id="h-limits-title">What it does not claim</h2>
      <ul>
        <li>The guarantee holds for traffic like the calibration mix. On naturally phrased questions the router hands almost everything to the teacher, which is safe but saves little.</li>
        <li>The student runs on a CPU sandbox, so its answers take tens of seconds; the teacher is faster.</li>
        <li>Spend is read from the ledger; sandbox seconds have no published price yet, so totals are lower bounds.</li>
      </ul>
      <p className="muted">How the student was trained and gated: <Link to="/mission">the distillation run</Link>.</p>
    </section>
  );
}

export default function Home() {
  const [config] = useAsync<Config>(() => api.config(), "config");
  const [status] = useAsync<GatewayStatus>(() => api.gatewayStatus(), "gateway-status");
  const [drift] = useAsync<GatewayDrift>(() => api.gatewayDrift(), "gateway-drift");
  const [evidence] = useAsync<Evidence>(() => api.evidence(), "evidence");
  return (
    <div className="home">
      <Hero config={config} />
      <Swap />
      <Live status={status} drift={drift} />
      <Flow />
      <Claims evidence={evidence} />
      <Limits />
    </div>
  );
}
