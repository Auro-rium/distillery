import type { CSSProperties, ReactNode } from "react";
import { NOT_MEASURED, fmtInt, fmtNumber, fmtPercent, fmtUsd } from "../../api/format";
import type { AccTriple, Report, TeacherCostPer1k } from "../../api/types";
import { Badge, Card, EmptyState, Stat } from "../../components";
import { AccuracyChart, AccuracyTable, Legend, MODELS, type BarRow, type TableRow } from "./AccuracyChart";
import { Breakable } from "./Breakable";

function modelRows(acc: Partial<AccTriple> | undefined, studentCi: [number, number] | null | undefined): BarRow[] {
  return MODELS.map((m) => ({ ...m, value: acc?.[m.key] ?? null, ci: m.key === "student" ? studentCi ?? null : null }));
}

export function Accuracy({ r }: { r: Report }) {
  const ev = r.evaluation;
  const rows = modelRows(ev.accuracy, ev.gate.student_ci);
  const u = ev.unparseable;
  const table: TableRow[] = [{ name: "All held-out", n: ev.n, values: rows }];
  return (
    <Card title="Held-out accuracy" className="rp-acc">
      <Legend />
      <AccuracyChart title="Held-out accuracy by model" rows={rows} />
      <p className="muted rp-note">
        Held-out n={fmtInt(ev.n)}. Whiskers are the report&rsquo;s confidence interval; a row without one has no CI in the report.
      </p>
      <p className="muted rp-note">
        Unparseable outputs: base {fmtInt(u?.base)}, student {fmtInt(u?.student)}, teacher {fmtInt(u?.teacher)}
      </p>
      <AccuracyTable caption="Held-out accuracy" columns={MODELS} rows={table} showN nLabel="n" />
    </Card>
  );
}

export function ByClass({ r }: { r: Report }) {
  const ev = r.evaluation;
  const classes = Object.keys(ev.accuracy_by_class ?? {});
  const table: TableRow[] = classes.map((c) => ({ name: c, n: ev.class_counts?.[c] ?? null, values: modelRows(ev.accuracy_by_class[c], null) }));
  return (
    <Card title="Accuracy by held-out class" className="rp-cls">
      {classes.length === 0 ? (
        <EmptyState title="No class split in this report" />
      ) : (
        <>
          <Legend />
          <div className="rp-multiples">
            {classes.map((c, i) => (
              <div className="rp-mult" role="group" aria-label={c} key={c} style={{ "--i": i } as CSSProperties}>
                <div className="rp-mult-head">
                  <strong>{c}</strong>
                  <Badge>n={fmtInt(ev.class_counts?.[c])}</Badge>
                  {c.startsWith("unseen") && <Badge tone="info">not in training</Badge>}
                </div>
                <AccuracyChart title={`Accuracy, class ${c}`} rows={modelRows(ev.accuracy_by_class[c], null)} showCi={false} />
              </div>
            ))}
          </div>
          <AccuracyTable caption="Accuracy by held-out class" columns={MODELS} rows={table} showN nLabel="n" />
        </>
      )}
    </Card>
  );
}

export function GateDetails({ r }: { r: Report }) {
  const g = r.evaluation.gate;
  const t = g.thresholds;
  return (
    <Card title="Gate statistics" className="rp-gate">
      <div className="rp-stats">
        <Stat label="Student / teacher ratio" value={fmtNumber(g.ratio_point)} hint={`interval ${fmtNumber(g.ratio_lo)} to ${fmtNumber(g.ratio_hi)}; needs lower bound ≥ ${fmtNumber(t.ratio_lower_bound_min, 2)}`} />
        <Stat label="McNemar exact p" value={fmtNumber(g.mcnemar_p, 4)} hint={`alpha ${fmtNumber(t.mcnemar_alpha, 2)}`} />
        <Stat label="Student right, base wrong" value={fmtInt(g.student_only_vs_base)} hint="held-out items" />
        <Stat label="Base right, student wrong" value={fmtInt(g.base_only_vs_student)} hint="held-out items" />
        <Stat label="Bootstrap resamples" value={fmtInt(g.bootstrap_resamples_used)} hint={`skipped ${fmtInt(g.bootstrap_skipped)}`} />
        <Stat label="Held-out n" value={fmtInt(g.n)} />
      </div>
    </Card>
  );
}

function CostCell({ label, v }: { label: string; v: string | TeacherCostPer1k | undefined }) {
  if (v === undefined) return <Stat label={label} value={NOT_MEASURED} hint="not in report" />;
  // A string here is the backend's own explanation (for example "unavailable: ..."); it is shown verbatim.
  if (typeof v === "string") return <Stat label={label} value={NOT_MEASURED} hint={v} />;
  return (
    <Stat
      label={label}
      value={fmtUsd(v.usd_per_1k_tasks, 5)}
      hint={`per 1k tasks · ${v.basis} · ${fmtInt(v.held_out_tasks)} tasks, ${fmtInt(v.input_tokens)} in / ${fmtInt(v.output_tokens)} out tokens, ${fmtUsd(v.usd, 6)}`}
    />
  );
}

export function CostLatency({ r }: { r: Report }) {
  const c = r.cost;
  const per = c.cost_per_1k_tasks as Record<string, string | TeacherCostPer1k | undefined>;
  const latency = (r as unknown as { latency?: unknown }).latency;
  const models = Object.entries(c.llm_by_model ?? {});
  return (
    <Card title="Cost and latency" className="rp-cost">
      <div className="rp-stats">
        <CostCell label="Student cost / 1k tasks" v={per.student} />
        <CostCell label="Teacher cost / 1k tasks" v={per.teacher} />
        <CostCell label="Base cost / 1k tasks" v={per.base} />
        <Stat label="Latency" value={NOT_MEASURED} hint={latency === undefined ? "the report has no latency data" : "latency present in report but not rendered"} />
      </div>
      <p className="muted rp-note">
        Run total {fmtUsd(c.run_total_usd)} of {fmtUsd(c.run_cap_usd)} cap. Fine-tune: {fmtUsd(c.finetune_usd)}
      </p>
      {models.length > 0 && (
        <div className="tbl-wrap">
          <table className="tbl rp-tbl">
            <caption className="sr-only">Spend by model</caption>
            <thead>
              <tr><th scope="col">Model</th><th scope="col" className="num">Calls</th><th scope="col" className="num">In tokens</th><th scope="col" className="num">Out tokens</th><th scope="col" className="num">Spent</th></tr>
            </thead>
            <tbody>
              {models.map(([name, m]) => (
                <tr key={name}>
                  <th scope="row" className="mono">{name}</th>
                  <td className="num">{fmtInt(m.calls)}</td>
                  <td className="num">{fmtInt(m.input_tokens)}</td>
                  <td className="num">{fmtInt(m.output_tokens)}</td>
                  <td className="num">{fmtUsd(m.usd)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </Card>
  );
}

const ISSUE = /drop|error|fail|discard|unparseable|skipped|shortfall|suspect/;

/** `purpose:metric` keys of llm_attempt_counters, pivoted to one row per purpose. */
function pivotAttempts(a: Record<string, number>): { metrics: string[]; purposes: [string, Record<string, number>][] } {
  const metrics: string[] = [];
  const by = new Map<string, Record<string, number>>();
  for (const [key, v] of Object.entries(a)) {
    const cut = key.lastIndexOf(":");
    const purpose = cut < 0 ? key : key.slice(0, cut);
    const metric = cut < 0 ? "count" : key.slice(cut + 1);
    if (!metrics.includes(metric)) metrics.push(metric);
    by.set(purpose, { ...(by.get(purpose) ?? {}), [metric]: v });
  }
  return { metrics, purposes: [...by.entries()] };
}

const num = (v: number): ReactNode => <span className={v === 0 ? "zero" : "nonzero"}>{fmtInt(v)}</span>;

export function Counters({ r }: { r: Report }) {
  const groups = Object.entries(r.counters ?? {})
    .map(([stage, vals]) => [stage, Object.entries(vals).filter(([k]) => ISSUE.test(k))] as const)
    .filter(([, rows]) => rows.length > 0);
  const attempts = pivotAttempts(r.llm_attempt_counters ?? {});
  const llm = Object.entries(r.llm_errors_by_purpose ?? {});
  return (
    <Card title="Drops, errors and retries" className="rp-cnt">
      {groups.length === 0 ? (
        <EmptyState title="No drop or error counters in this report" />
      ) : (
        <div className="rp-ctrs">
          {groups.map(([stage, rows]) => (
            <div className="tbl-wrap" key={stage}>
              <table className="tbl rp-tbl ctr">
                <caption>{stage}</caption>
                <tbody>
                  {rows.map(([k, v]) => <tr key={k}><th scope="row"><Breakable text={k} /></th><td className="num">{num(v)}</td></tr>)}
                </tbody>
              </table>
            </div>
          ))}
        </div>
      )}
      {attempts.purposes.length > 0 && (
        <div className="tbl-wrap rp-attempts">
          <table className="tbl rp-tbl">
            <caption>LLM attempts by purpose</caption>
            <thead>
              <tr>
                <th scope="col">Purpose</th>
                {attempts.metrics.map((m) => <th scope="col" className="num" key={m}>{m.replace(/_/g, " ")}</th>)}
              </tr>
            </thead>
            <tbody>
              {attempts.purposes.map(([purpose, vals]) => (
                <tr key={purpose}>
                  <th scope="row" className="mono"><Breakable text={purpose} /></th>
                  {attempts.metrics.map((m) => (
                    <td className="num" key={m}>{vals[m] === undefined ? NOT_MEASURED : num(vals[m])}</td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
      <p className="muted rp-note">
        LLM errors by purpose: {llm.length === 0 ? "none recorded" : llm.map(([k, v]) => `${k} ${fmtInt(v)}`).join(", ")}
      </p>
    </Card>
  );
}

export function Clusters({ r }: { r: Report }) {
  const withClusters = (r.rounds ?? []).filter((x) => x.clusters.length > 0);
  return (
    <Card title="Failure clusters" className="rp-clu">
      {withClusters.length === 0 ? (
        <EmptyState title="No failure clusters in this report" />
      ) : (
        withClusters.map((x) => (
          <section className="rp-round" key={x.round}>
            <div className="eyebrow">
              Round {fmtInt(x.round)} · dev accuracy {fmtPercent(x.dev_acc)}{x.round === r.candidate_round ? " · selected candidate" : ""}
            </div>
            <ul className="rp-clusters" role="list">
              {x.clusters.map((c) => (
                <li key={c.name}>
                  <strong>{c.name}</strong>
                  <p>{c.description}</p>
                  <span className="rp-chips">{c.target_families.map((f) => <Badge key={f}>{f}</Badge>)}</span>
                </li>
              ))}
            </ul>
          </section>
        ))
      )}
      <dl className="rp-why">
        <div><dt>Rounds stopped</dt><dd>{r.rounds_stop_reason ?? NOT_MEASURED}</dd></div>
        <div><dt>Candidate selection</dt><dd>{r.candidate_selection ?? NOT_MEASURED}</dd></div>
      </dl>
    </Card>
  );
}
