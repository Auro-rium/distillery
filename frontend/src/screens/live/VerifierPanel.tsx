import { fmtInt, fmtText } from "../../api/format";
import type { RunDetail } from "../../api/types";
import { Badge, CodeBlock } from "../../components";
import { Panel } from "./Panel";
import { Metric, Val } from "./Val";

/** Verifier self-test result and code. Null (not yet reported / not stored) reads "not measured". */
export function VerifierPanel({ verifier }: { verifier: RunDetail["verifier"] | null | undefined }) {
  const st = verifier?.selftest ?? null;
  const code = verifier?.code ?? null;
  return (
    <Panel title="Verifier" aside={verifier?.language ? <Badge>{verifier.language}</Badge> : null}>
      <h3 className="lp-sub">Self-test</h3>
      {st ? (
        <dl className="metrics">
          <Metric label="Accepted gold" value={fmtInt(st.accepted_gold)} />
          <Metric label="Rejected corruptions" value={fmtInt(st.rejected_corruptions)} />
          <Metric label="Failures" value={<span className={st.failures === 0 ? "tone-ok" : "tone-bad"}>{fmtInt(st.failures)}</span>} />
        </dl>
      ) : (
        <p className="lp-none">Result: <Val text={fmtInt(null)} /></p>
      )}
      <h3 className="lp-sub">Code</h3>
      {code ? (
        <CodeBlock code={code} caption={`${verifier?.language ?? ""} verifier`.trim()} />
      ) : (
        <p className="lp-none">Verifier code: <Val text={fmtText(null)} /></p>
      )}
    </Panel>
  );
}
