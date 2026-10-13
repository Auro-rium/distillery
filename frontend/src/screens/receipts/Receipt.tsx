import type { ReactNode } from "react";
import { fmtText, fmtUsd } from "../../api/format";
import type { Receipt as ReceiptRow } from "../../api/types";
import { Badge } from "../../ui";
import type { Tone } from "../../ui/Card";
import { fmtSig } from "../replay/mission";

/** Characters of a commit / sha256 shown in the row; the whole value is in the title attribute. */
export const SHORT_COMMIT = 7;
export const SHORT_SHA = 12;

interface Look { text: string; tone: Tone; cls: string }

/**
 * How a receipt is badged. `failed` is never softened: a claim that was not met says so, and a PARTIAL proof
 * (part of the claim held) says "partly met" instead of a bare FAIL.
 */
export function look(r: ReceiptRow): Look {
  if (r.status === "verified") return { text: "VERIFIED", tone: "ok", cls: "is-pass" };
  if (r.status === "pending") return { text: "PENDING", tone: "warn", cls: "is-pending" };
  if (r.verdict === "PARTIAL") return { text: "PARTLY MET", tone: "warn", cls: "is-partial" };
  return { text: "NOT MET", tone: "bad", cls: "is-fail" };
}

function Hash({ value, len }: { value: string; len: number }) {
  return <code className="rc-hash num" title={value}>{value.slice(0, len)}</code>;
}

function Fact({ label, children }: { label: string; children: ReactNode }) {
  return (
    <div className="rc-fact">
      <dt className="eyebrow">{label}</dt>
      <dd>{children}</dd>
    </div>
  );
}

function gateValue(v: number | string | boolean): string {
  return typeof v === "boolean" ? (v ? "yes" : "no") : fmtSig(v);
}

/**
 * One claim and the file that backs it. Everything shown is a field of the evidence payload: the status the
 * builder gave it, the value as the evidence states it, the commit-pinned link to the artifact, the
 * pre-registration commit, the sealed-set and adapter hashes, the gate's inputs and the ledger total. A pending
 * claim shows no number at all, only that it is pre-registered and not yet scored.
 */
export function Receipt({ r }: { r: ReceiptRow }) {
  const l = look(r);
  const pending = r.status === "pending";
  const sealed = Object.entries(r.sealed_sha256);
  const gate = r.gate_inputs ? Object.entries(r.gate_inputs) : [];
  return (
    <article className={`proof-card rc ${l.cls}`} id={r.id} aria-labelledby={`rc-${r.id}-h`}>
      <header className="proof-head">
        <span className="proof-rung num">{r.id}</span>
        <Badge tone={l.tone}>{l.text}</Badge>
      </header>
      <h3 className="proof-title rc-claim" id={`rc-${r.id}-h`}>{r.claim}</h3>
      {pending ? (
        <p className="rc-pending" role="note">Pending: pre-registered, not yet scored</p>
      ) : (
        <p className="rc-value num">{fmtText(r.value_display)}</p>
      )}
      {!pending && r.status === "failed" && <p className="rc-notmet muted">This claim was not met; the value above is what the evidence recorded.</p>}
      <dl className="rc-facts">
        <Fact label={pending ? "will be written to" : "artifact"}>
          {pending ? (
            <span className="rc-path num">{r.artifact}</span>
          ) : (
            <a className="rc-path num" href={r.permalink} target="_blank" rel="noreferrer noopener">
              {r.artifact}
              <span className="sr-only"> (opens in a new tab)</span>
            </a>
          )}
        </Fact>
        {!pending && (
          <Fact label="pre-registration">
            {r.prereg ? (
              <>
                <Hash value={r.prereg.commit} len={SHORT_COMMIT} /> <span className="rc-when num">{r.prereg.committed_at}</span>
              </>
            ) : (
              <span className="muted">no pre-registration commit recorded in this evidence</span>
            )}
          </Fact>
        )}
        {sealed.length > 0 && (
          <Fact label="sealed sets (sha256)">
            <ul className="rc-sealed">
              {sealed.map(([name, sha]) => (
                <li key={name}><span className="rc-sname">{name}</span> <Hash value={sha} len={SHORT_SHA} /></li>
              ))}
            </ul>
          </Fact>
        )}
        {r.adapter_sha256 && <Fact label="adapter (sha256)"><Hash value={r.adapter_sha256} len={SHORT_SHA} /></Fact>}
        {r.ledger_total_usd !== null && <Fact label="ledger total"><span className="num">{fmtUsd(r.ledger_total_usd, 4)}</span></Fact>}
      </dl>
      {gate.length > 0 && (
        <details className="rc-gate">
          <summary>Gate inputs</summary>
          <dl className="rc-gate-list">
            {gate.map(([k, v]) => (
              <div key={k}>
                <dt>{k}</dt>
                <dd className="num">{gateValue(v)}</dd>
              </div>
            ))}
          </dl>
        </details>
      )}
    </article>
  );
}
