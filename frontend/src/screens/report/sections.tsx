import type { CSSProperties, ReactNode } from "react";
import { NOT_MEASURED, fmtInt, fmtNumber, fmtPercent, fmtText, fmtUsd } from "../../api/format";
import type { AccTriple, FinetuneRecord, Report, SandboxCostPer1k, TeacherCostPer1k } from "../../api/types";
import { AccuracyBars, ChartLegend, CiNumberLine, DiscordantMatrix, LossChart, MODEL_LABEL, StressBars, type ModelKey } from "../../charts";
import { Badge, Card, CopyButton, EmptyState, Stat } from "../../components";
import { AccuracyTable, MODELS, ciOf, ciText, type BarRow, type TableRow } from "./AccuracyTable";
import { Breakable } from "./Breakable";

function modelRows(acc: Partial<AccTriple> | undefined, studentCi: [number, number] | null | undefined): BarRow[] {
  return MODELS.map((m) => ({ ...m, value: acc?.[m.key] ?? null, ci: m.key === "student" ? studentCi ?? null : null }));
}

const MODEL_KEY = MODELS.map((m) => ({ key: m.key, label: MODEL_LABEL[m.key], className: `m-${m.key}` }));

/** One sentence that says, in text, which rows have a whisker: the report carries a CI for the student only. */
function CiNote({ rows }: { rows: BarRow[] }) {
  const withCi = rows.filter((r) => ciOf(r));
  const without = rows.filter((r) => !ciOf(r)).map((r) => r.label.toLowerCase());
  return (
    <p className="muted rp-note rp-ci-note">
      {withCi.map((r) => <span key={r.key}>Whisker: {r.label.toLowerCase()} confidence interval CI {ciText(ciOf(r)!)} (from the report). </span>)}
      {without.length > 0 && <>No CI in report for {without.join(", ")}.</>}
    </p>
  );
}

// ---------------------------------------------------------------- Verdict tab

/** The single "why PROMOTE" graphic plus the McNemar evidence: ratio CI vs threshold, discordant pairs. */
export function GateDetails({ r }: { r: Report }) {
  const g = r.evaluation.gate;
  const t = g.thresholds;
  return (
    <Card title="Gate statistics" className="rp-gate">
      <CiNumberLine
        title="Student / teacher accuracy ratio"
        point={g.ratio_point}
        lo={g.ratio_lo}
        hi={g.ratio_hi}
        threshold={t.ratio_lower_bound_min}
        thresholdLabel="lower-bound threshold"
        caption="Point estimate and bootstrap interval of student accuracy over teacher accuracy on the held-out set; the lower bound must clear the threshold."
      />
      <DiscordantMatrix
        title="Discordant pairs, student vs base"
        studentOnly={g.student_only_vs_base}
        baseOnly={g.base_only_vs_student}
        caption={`Held-out items where exactly one of the two models is right; McNemar's exact test compares these two counts (p ${fmtNumber(g.mcnemar_p, 4)}, alpha ${fmtNumber(t.mcnemar_alpha, 2)}).`}
      />
      <div className="rp-stats">
        <Stat label="McNemar exact p" value={fmtNumber(g.mcnemar_p, 4)} hint={`alpha ${fmtNumber(t.mcnemar_alpha, 2)}`} />
        <Stat label="Bootstrap resamples" value={fmtInt(g.bootstrap_resamples_used)} hint={`skipped ${fmtInt(g.bootstrap_skipped)}`} />
        <Stat label="Held-out n" value={fmtInt(g.n)} />
      </div>
    </Card>
  );
}

export function Accuracy({ r }: { r: Report }) {
  const ev = r.evaluation;
  const rows = modelRows(ev.accuracy, ev.gate.student_ci);
  const u = ev.unparseable;
  const table: TableRow[] = [{ name: "All held-out", n: ev.n, values: rows }];
  return (
    <Card title="Held-out accuracy" className="rp-acc">
      <AccuracyBars
        title="Held-out accuracy by model"
        rows={rows}
        legend
        caption={`Execution accuracy on the sealed held-out set, n=${fmtInt(ev.n)}.`}
      />
      <CiNote rows={rows} />
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
          <ChartLegend items={MODEL_KEY} />
          <div className="rp-multiples">
            {classes.map((c, i) => (
              <div className="rp-mult" role="group" aria-label={c} key={c} style={{ "--i": i } as CSSProperties}>
                <div className="rp-mult-head">
                  <strong>{c}</strong>
                  <Badge>n={fmtInt(ev.class_counts?.[c])}</Badge>
                  {c.startsWith("unseen") && <Badge tone="info">not in training</Badge>}
                </div>
                <AccuracyBars title={`Accuracy, class ${c}`} rows={modelRows(ev.accuracy_by_class[c], null)} showCi={false} />
              </div>
            ))}
          </div>
          <p className="muted rp-note">The report has no per-class confidence interval, so these bars carry no whiskers.</p>
          <AccuracyTable caption="Accuracy by held-out class" columns={MODELS} rows={table} showN nLabel="n" />
        </>
      )}
    </Card>
  );
}

/** Shown only when the payload carries the overlap rate: the gate set shares question shapes with training. */
export function GateCaveat({ r }: { r: Report }) {
  const rate = r.data?.heldout_skeleton_overlap_rate;
  if (typeof rate !== "number") return null;
  return (
    <Card title="Gate set is in-distribution" className="rp-caveat" tone="warn">
      <p role="note" className="rp-caveat-text">
        Gate set is in-distribution: {fmtPercent(rate, 0)} of gate questions share a question skeleton with a training question
        {typeof r.data.train_distinct_skeletons === "number" ? ` (${fmtInt(r.data.train_distinct_skeletons)} distinct training skeletons)` : ""}.
        The gate measures generalisation across literals within known question shapes, not novel query structure.
      </p>
    </Card>
  );
}

/** Gate B. Rendered only when the report has a human block; every figure comes from the payload. */
export function HumanSet({ r }: { r: Report }) {
  const h = r.evaluation.human;
  if (!h) return null;
  const d = r.data.human;
  const rows = modelRows(h.accuracy, h.gate.student_ci);
  const tone = r.decision_human === "PROMOTE" ? "ok" : r.decision_human === "REJECT" ? "bad" : undefined;
  return (
    <Card
      title="Human held-out set (Gate B)"
      className="rp-human"
      meta={r.decision_human ? <Badge tone={tone}>{r.decision_human}</Badge> : undefined}
    >
      <p className="muted rp-note">
        Same thresholds as the gate above; reported separately and never mixed with it or the stress set.
      </p>
      <dl className="rp-kv">
        <div><dt>Gate B decision</dt><dd>{r.decision_human ?? NOT_MEASURED}</dd></div>
        <div><dt>Human n</dt><dd>{fmtInt(h.n)}</dd></div>
        {d && (
          <>
            <div><dt>Questions in file</dt><dd>{fmtInt(d.counts.questions)}</dd></div>
            <div><dt>Drafts kept</dt><dd>{fmtInt(d.counts.kept)}</dd></div>
            <div><dt>Discarded</dt><dd>{fmtInt(d.counts.discarded)} ({Object.entries(d.counts.discarded_by_reason).map(([k, v]) => `${k} ${fmtInt(v)}`).join(", ")})</dd></div>
            <div><dt>Rejected by the human</dt><dd>{fmtInt(d.counts.rejected)}</dd></div>
            <div><dt>Dropped as exact overlap</dt><dd>{fmtInt(d.dropped_exact_overlap.total)}</dd></div>
            <div><dt>Skeleton shared with training</dt><dd>{fmtInt(d.skeleton_in_train)}</dd></div>
          </>
        )}
      </dl>
      <ul className="rp-reasons">{h.gate.reasons.map((x) => <li key={x}>{x}</li>)}</ul>
      <ul className="rp-reasons" aria-label="Human set accuracy">
        {rows.map((m) => <li key={m.key}>{m.label} accuracy {m.value === null ? NOT_MEASURED : fmtPercent(m.value, 1)}</li>)}
      </ul>
      {(h.note ?? d?.note) && <p className="muted rp-note">{h.note ?? d?.note}</p>}
    </Card>
  );
}

// ---------------------------------------------------------------- Training tab

/** The fine-tune record of the evaluated candidate round; reports with one record per round otherwise fall back to the first. */
function candidateRecord(r: Report): FinetuneRecord | undefined {
  const all = r.finetune ?? [];
  return all.find((f) => f.round === r.candidate_round) ?? all[0];
}

/** A hyperparameter value as the payload has it: integers via fmtInt, other numbers verbatim (fmtNumber would round a learning rate to zero). */
function hpValue(v: number | boolean | string | null): string {
  if (typeof v === "number") return Number.isInteger(v) ? fmtInt(v) : fmtText(String(v));
  if (typeof v === "boolean") return v ? "true" : "false";
  return fmtText(v);
}

const short = (h: string | undefined) => (h ? h.slice(0, 12) : NOT_MEASURED);

function IdRow({ label, id, display, title }: { label: string; id: string | undefined; display?: string; title?: string }) {
  return (
    <div>
      <dt>{label}</dt>
      <dd>
        <span className="mono" title={title}>{display ?? fmtText(id)}</span>
        {id ? <CopyButton text={id} /> : null}
      </dd>
    </div>
  );
}

export function FineTune({ r }: { r: Report }) {
  const rec = candidateRecord(r);
  const art = r.evaluation.artifact as Partial<Report["evaluation"]["artifact"]> | undefined;
  const jobs = (r.rounds ?? []).map((x) => x.job_id).filter(Boolean);
  const cand = (r.rounds ?? []).find((x) => x.round === r.candidate_round);
  const hp = Object.entries(rec?.hyperparameters ?? {});
  return (
    <Card
      title="Fine-tune"
      className="rp-ft"
      meta={rec ? <Badge>round {fmtInt(rec.round)}{rec.round === r.candidate_round ? " · selected candidate" : ""}</Badge> : undefined}
    >
      {!rec ? (
        <EmptyState title="This report has no fine-tune record">Reports recorded before the fine-tune diagnostics existed do not carry a loss curve.</EmptyState>
      ) : (
        <>
          <LossChart
            title={`Train and validation loss by step, fine-tune round ${fmtInt(rec.round)}`}
            points={rec.loss_curve ?? []}
            caption="Loss reported by the fine-tune job at each recorded checkpoint; straight segments, no smoothing."
          />
          <p className="rp-interp">
            <strong>Interpretation.</strong> Validation loss is token-level against the gold SQL text, while the student was trained on teacher
            SQL, so the two can drift apart without the queries getting worse. Execution accuracy is the gate metric: dev{" "}
            {fmtPercent(cand?.dev_acc)} at this round, held-out {fmtPercent(r.evaluation.accuracy.student)}.
          </p>
          <ul className="rp-chips rp-ft-chips" aria-label="Training facts">
            <li><Badge>trained tokens {fmtInt(rec.trained_tokens)}</Badge></li>
            <li><Badge>steps {fmtInt(rec.trained_steps)} of {fmtInt(rec.total_steps)}</Badge></li>
            <li><Badge>base model {fmtText(rec.base_model)}</Badge></li>
          </ul>
          {hp.length > 0 && (
            <details className="rp-hp">
              <summary>Hyperparameters</summary>
              <ul className="rp-chips" aria-label="Hyperparameters">
                {hp.map(([k, v]) => <li key={k}><span className="rp-chip mono">{k} <strong>{hpValue(v)}</strong></span></li>)}
              </ul>
            </details>
          )}
          {rec.diagnostics_error && <p className="muted rp-note">Fine-tune diagnostics error: {rec.diagnostics_error}</p>}
        </>
      )}
      {art && (
        <div className="ft-artifact">
          <div className="eyebrow">Fine-tune artifact</div>
          <dl className="rp-ids">
            <IdRow label="job" id={art.job_id} />
            <IdRow label="checkpoint" id={art.checkpoint_id} />
            <IdRow label="adapter" id={art.adapter_sha256} display={short(art.adapter_sha256)} title={art.adapter_sha256} />
          </dl>
          {jobs.length > 0 && (
            <p className="muted rp-note">
              Round jobs {jobs.map((j, i) => <span key={`${j}-${i}`}>{i > 0 && ", "}<span className="mono">{j}</span></span>)}
            </p>
          )}
        </div>
      )}
    </Card>
  );
}

type CostV = string | TeacherCostPer1k | SandboxCostPer1k | undefined;

function CostCell({ label, v }: { label: string; v: CostV }) {
  if (v === undefined) return <Stat label={label} value={NOT_MEASURED} hint="not in report" />;
  // A string here is the backend's own explanation (for example "unavailable: ..."); it is shown verbatim.
  if (typeof v === "string") return <Stat label={label} value={NOT_MEASURED} hint={v} />;
  // The student shape with a sandbox price has no token fields: seconds x price, over `samples`.
  if (!("input_tokens" in v)) {
    return (
      <Stat
        label={label}
        value={fmtUsd(v.usd_per_1k_tasks, 5)}
        hint={`per 1k tasks · ${v.basis} · ${fmtInt(v.samples)} samples, ${fmtNumber(v.sandbox_seconds, 1)} sandbox s`}
      />
    );
  }
  const tasks = v.items_scored ?? v.held_out_tasks;
  return (
    <Stat
      label={label}
      value={fmtUsd(v.usd_per_1k_tasks, 5)}
      hint={`per 1k tasks · ${v.basis} · ${fmtInt(tasks)} tasks, ${fmtInt(v.input_tokens)} in / ${fmtInt(v.output_tokens)} out tokens, ${fmtUsd(v.usd, 6)}`}
    />
  );
}

/** `cost.finetune_lines`: written by newer reports, not in the typed contract yet, so read defensively. */
interface CostLine { kind?: string; model?: string; units?: number | null; usd?: number | null; basis?: string }
function costLines(c: Report["cost"]): CostLine[] {
  const v = (c as unknown as { finetune_lines?: unknown }).finetune_lines;
  return Array.isArray(v) ? (v.filter((x) => x && typeof x === "object") as CostLine[]) : [];
}

export function CostLatency({ r }: { r: Report }) {
  const c = r.cost;
  const per = c.cost_per_1k_tasks as Record<string, CostV>;
  const latency = (r as unknown as { latency?: unknown }).latency;
  const models = Object.entries(c.llm_by_model ?? {});
  const lines = costLines(c);
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
      {c.basis && <p className="muted rp-note cost-basis">{c.basis}</p>}
      {lines.length > 0 && (
        <ul className="rp-lines" aria-label="Fine-tune cost lines">
          {lines.map((l, i) => (
            <li key={i}>
              <span className="mono">{fmtText(l.model)}</span> {fmtText(l.kind)}: {fmtInt(l.units)} units, {fmtUsd(l.usd)} <span className="muted">({fmtText(l.basis)})</span>
            </li>
          ))}
        </ul>
      )}
      {models.length > 0 && (
        <div className="tbl-wrap" tabIndex={0} role="region" aria-label="Spend by model, scrollable">
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

const STRESS_MODELS: readonly ModelKey[] = ["student", "teacher"];

/** Rendered only when the report has a stress block; nothing is invented otherwise. */
export function Stress({ r }: { r: Report }) {
  const s = r.evaluation.stress;
  if (!s) return null;
  const fams = Object.keys(s.accuracy_by_family ?? {});
  const rows = modelRows(s.accuracy, null);
  const table: TableRow[] = [
    { name: "All stress", n: s.n, values: rows },
    ...fams.map((f) => ({ name: f, n: null, values: modelRows(s.accuracy_by_family[f], null) })),
  ];
  const bars = [
    { name: "All stress", n: s.n, values: s.accuracy },
    ...fams.map((f) => ({ name: f, values: s.accuracy_by_family[f] })),
  ];
  return (
    <Card title="Stress set" className="rp-stress">
      <p className="muted rp-note">Reserved families, never in train/dev/gate, NOT a gate input.</p>
      <StressBars
        title="Stress accuracy, student vs teacher"
        rows={bars}
        models={STRESS_MODELS}
        caption={`Execution accuracy on task families the student never trained on. Stress n=${fmtInt(s.n)}${fams.length > 0 ? `; families ${fams.join(", ")}` : ""}. Base accuracy is in the table view.`}
      />
      <AccuracyTable caption="Stress accuracy overall and per family" columns={MODELS} rows={table} showN nLabel="n" />
    </Card>
  );
}

// ---------------------------------------------------------------- Examples & errors tab

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
