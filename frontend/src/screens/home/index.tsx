// Home (`/`): what Distillery is for a visitor who has a minute. The product first (one base-URL change, a certified
// specialist, the teacher for the rest, a drift guard), then how one request is served and what it does not claim.
// The model ids come from /api/config; the copy on this page carries no figures of its own.
import { Link } from "react-router-dom";
import { api } from "../../api/client";
import { fmtText } from "../../api/format";
import type { Config } from "../../api/types";
import { GITHUB_URL } from "../../components/Shell";
import { GlowText } from "../../motion/GlowText";
import { Edges } from "./Edges";
import { Flow } from "./Flow";
import { Robot } from "./Robot";
import { useAsync, type Async } from "../report/useAsync";
import "./home.css";

function Hero({ config }: { config: Async<Config> }) {
  const models = config.state === "ok" ? config.data.models : null;
  return (
    <section className="h-hero" aria-labelledby="h-title">
      <p className="eyebrow">Nebius Token Factory · NVIDIA Nemotron</p>
      <GlowText as="h1" id="h-title" reveal>
        An autonomous agent that distils a big model’s narrow skill into a small, certified student, served behind an OpenAI-compatible gateway with automatic rollback.
      </GlowText>
      <GlowText className="h-pipeline">
        Write → Test in Sandboxes → Fine-tune on Token Factory → Gate → Certify the router → Serve → Guard
      </GlowText>
      {models && models.teacher && models.student && (
        <dl className="h-models" aria-label="Models on this server">
          <div><dt>Teacher</dt><dd className="mono">{fmtText(models.teacher)}</dd></div>
          <div><dt>Student</dt><dd className="mono">{fmtText(models.student)}</dd></div>
        </dl>
      )}
      <div className="h-cta">
        <Link className="btn primary" to="/mission">Go to mission</Link>
        <a className="btn primary bulge" href={GITHUB_URL} target="_blank" rel="noreferrer noopener">Source on GitHub</a>
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
  return (
    <div className="home">
      <Edges />
      <div className="h-top">
        <Hero config={config} />
        <Robot />
      </div>
      <Swap />
      <Flow />
      <Limits />
    </div>
  );
}
