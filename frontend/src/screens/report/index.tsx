import { useId, useMemo, useState } from "react";
import { Link, useParams } from "react-router-dom";
import { ApiError, api } from "../../api/client";
import { NOT_MEASURED, fmtInt, fmtNumber, fmtPercent, fmtUsd } from "../../api/format";
import type { Example, ExamplesResult, Report, TeacherCostPer1k } from "../../api/types";
import { ApiErrorState, Badge, Card, CodeBlock, EmptyState, LabelBanner, Spinner, Stat } from "../../components";
import { diffAgainstGold, type SideBySide } from "../../lib/diff";
import { AccuracyChart, type BarRow } from "./AccuracyChart";
import { useAsync } from "./useAsync";
import "./report.css";

const NAMES = { base: "Base", student: "Student", teacher: "Teacher" } as const;
const KEYS = ["base", "student", "teacher"] as const;

export function Verdict({ r }: { r: Report }) {
  const reasons = r.decision_reasons;
  return (
    <div className={`verdict ${r.decision}`} role="region" aria-label="Gate decision">
      <div className="eyebrow">Gate decision</div>
      <div className="big">{r.decision}</div>
      {reasons.length > 0 ? (
        <ul>{reasons.map((x) => <li key={x}>{x}</li>)}</ul>
      ) : (
        <p style={{ margin: "8px 0 0" }}>No reasons were reported by the gate.</p>
      )}
    </div>
  );
}

function rows(acc: Partial<Report["evaluation"]["accuracy"]>, n: number | null, studentCi: [number, number] | null): BarRow[] {
  return KEYS.map((k) => ({
    key: k, label: NAMES[k], value: acc[k] ?? null, n, ci: k === "student" ? studentCi : null,
  }));
}

export function Accuracy({ r }: { r: Report }) {
  const ev = r.evaluation;
  const classes = Object.keys(ev.accuracy_by_class);
  return (
    <Card title="Held-out accuracy">
      <AccuracyChart title="Held-out accuracy by model" rows={rows(ev.accuracy, ev.n, ev.gate.student_ci)} />
      <p className="muted" style={{ marginTop: 10 }}>
        Held-out n={fmtInt(ev.n)}. The report gives a confidence interval for the student only; base and teacher
        have none. Unparseable outputs: base {fmtInt(ev.unparseable.base)}, student {fmtInt(ev.unparseable.student)},
        teacher {fmtInt(ev.unparseable.teacher)}.
      </p>
      <h3 style={{ marginTop: 20 }}>By held-out class</h3>
      {classes.length === 0 ? (
        <EmptyState title="No class split in this report" />
      ) : (
        <div className="stack">
          {classes.map((c) => (
            <div key={c}>
              <div className="row" style={{ gap: 8 }}>
                <strong>{c}</strong>
                <Badge>n={fmtInt(ev.class_counts[c])}</Badge>
                {c.startsWith("unseen") && <Badge tone="info">not in training</Badge>}
              </div>
              <AccuracyChart title={`Accuracy, class ${c}`} rows={rows(ev.accuracy_by_class[c], ev.class_counts[c] ?? null, null)} />
            </div>
          ))}
        </div>
      )}
    </Card>
  );
}

export function GateDetails({ r }: { r: Report }) {
  const g = r.evaluation.gate;
  const t = g.thresholds;
  return (
    <Card title="Gate statistics">
      <div className="grid">
        <Stat label="Student / teacher ratio" value={fmtNumber(g.ratio_point)} hint={`interval ${fmtNumber(g.ratio_lo)} to ${fmtNumber(g.ratio_hi)}; needs lower bound ≥ ${fmtNumber(t.ratio_lower_bound_min, 2)}`} />
        <Stat label="McNemar exact p" value={fmtNumber(g.mcnemar_p, 4)} hint={`alpha ${fmtNumber(t.mcnemar_alpha, 2)}; student-only ${fmtInt(g.student_only_vs_base)}, base-only ${fmtInt(g.base_only_vs_student)}`} />
        <Stat label="Bootstrap resamples" value={fmtInt(g.bootstrap_resamples_used)} hint={`skipped ${fmtInt(g.bootstrap_skipped)}`} />
        <Stat label="Held-out n" value={fmtInt(g.n)} />
      </div>
    </Card>
  );
}

function CostCell({ label, v }: { label: string; v: string | TeacherCostPer1k | undefined }) {
  if (v === undefined) return <Stat label={label} value={NOT_MEASURED} hint="not in report" />;
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
  return (
    <Card title="Cost and latency">
      <div className="grid">
        <CostCell label="Student cost / 1k tasks" v={per.student} />
        <CostCell label="Teacher cost / 1k tasks" v={per.teacher} />
        <CostCell label="Base cost / 1k tasks" v={per.base} />
        <Stat label="Latency" value={NOT_MEASURED} hint={latency === undefined ? "the report has no latency data" : "latency present in report but not rendered"} />
      </div>
      <p className="muted" style={{ marginTop: 10 }}>
        Run total {fmtUsd(c.run_total_usd)} of {fmtUsd(c.run_cap_usd)} cap. Fine-tune: {fmtUsd(c.finetune_usd)}
      </p>
    </Card>
  );
}

const ISSUE = /drop|error|fail|discard|unparseable|skipped|shortfall|suspect/;
export function Counters({ r }: { r: Report }) {
  const list: [string, string, number][] = [];
  for (const [g, vals] of Object.entries(r.counters))
    for (const [k, v] of Object.entries(vals)) if (ISSUE.test(k)) list.push([g, k, v]);
  const llm = Object.entries(r.llm_errors_by_purpose);
  return (
    <Card title="Drop and error counters">
      {list.length === 0 ? <EmptyState title="No drop or error counters in this report" /> : (
        <div className="tbl-wrap">
          <table className="tbl">
            <thead><tr><th>Stage</th><th>Counter</th><th>Value</th></tr></thead>
            <tbody>{list.map(([g, k, v]) => (
              <tr key={`${g}.${k}`}><td>{g}</td><td>{k}</td><td>{fmtInt(v)}</td></tr>
            ))}</tbody>
          </table>
        </div>
      )}
      <p className="muted" style={{ marginTop: 10 }}>
        LLM errors by purpose: {llm.length === 0 ? "none recorded" : llm.map(([k, v]) => `${k} ${fmtInt(v)}`).join(", ")}
      </p>
    </Card>
  );
}

export function Clusters({ r }: { r: Report }) {
  const withClusters = r.rounds.filter((x) => x.clusters.length > 0);
  return (
    <Card title="Failure clusters">
      {withClusters.length === 0 ? <EmptyState title="No failure clusters in this report" /> : withClusters.map((x) => (
        <div key={x.round} style={{ marginBottom: 12 }}>
          <div className="eyebrow">Round {x.round} · dev accuracy {fmtPercent(x.dev_acc)}{x.round === r.candidate_round ? " · selected candidate" : ""}</div>
          <ul>{x.clusters.map((c) => (
            <li key={c.name}><strong>{c.name}</strong>: {c.description} <span className="muted">[{c.target_families.join(", ")}]</span></li>
          ))}</ul>
        </div>
      ))}
    </Card>
  );
}

type Model = "base" | "student" | "teacher";
const MODELS: [Model, string][] = [["base", "Base"], ["student", "Student"], ["teacher", "Teacher"]];

function Sql({ label, sql, ok, d }: { label: string; sql: string; ok: boolean; d: SideBySide | null }) {
  return (
    <div>
      <div className="row" style={{ gap: 6 }}><span className="eyebrow">{label}</span><Badge tone={ok ? "ok" : "bad"}>{ok ? "correct" : "wrong"}</Badge></div>
      <CodeBlock code={sql} segments={d?.other} side="other" />
      <p className="diff-note muted">{d === null ? "diff skipped: SQL too long" : d.identical ? "same tokens as gold" : "tokens not in gold are highlighted"}</p>
    </div>
  );
}

/** One example. Correctness flags come from the API; highlighting is a plain token diff of the returned SQL strings. */
function ExampleCard({ e, open0 }: { e: Example; open0: boolean }) {
  const [open, setOpen] = useState(open0);
  const [against, setAgainst] = useState<Model>("student");
  const body = useId();
  const diffs = useMemo(
    () => (open ? { base: diffAgainstGold(e.gold_sql, e.base_sql), student: diffAgainstGold(e.gold_sql, e.student_sql), teacher: diffAgainstGold(e.gold_sql, e.teacher_sql) } : null),
    [open, e.gold_sql, e.base_sql, e.student_sql, e.teacher_sql],
  );
  return (
    <div className={`ex${open ? " open" : ""}`}>
      <button type="button" className="ex-head" aria-expanded={open} aria-controls={open ? body : undefined} onClick={() => setOpen((o) => !o)}>
        <strong>{e.question}</strong>
        <span className="chev" aria-hidden="true" />
      </button>
      <div className="muted mono">{e.task_id} · {e.family} · {e.heldout_class}</div>
      {open && diffs && (
        <div id={body} className="ex-body">
          <div className="row seg" role="group" aria-label="Highlight gold tokens missing from">
            <span className="eyebrow">Gold vs</span>
            {MODELS.map(([k, name]) => (
              <button key={k} type="button" className="seg-btn" aria-pressed={against === k} onClick={() => setAgainst(k)}>{name}</button>
            ))}
          </div>
          <div className="cmp">
            <div>
              <span className="eyebrow">Gold</span>
              <CodeBlock code={e.gold_sql} segments={diffs[against]?.gold} side="gold" />
              <p className="diff-note muted">tokens missing from {against} SQL are highlighted</p>
            </div>
            <Sql label="Base" sql={e.base_sql} ok={e.base_ok} d={diffs.base} />
            <Sql label="Student" sql={e.student_sql} ok={e.student_ok} d={diffs.student} />
            <Sql label="Teacher" sql={e.teacher_sql} ok={e.teacher_ok} d={diffs.teacher} />
          </div>
        </div>
      )}
    </div>
  );
}

export function ExampleList(
  { title, items, total }: { title: string; items: Example[]; total: number | null | undefined },
) {
  // "showing N of M" only when the server gave M (whole held-out set); otherwise it is only a capped list.
  const count = total === null || total === undefined ? `capped list, ${items.length} shown` : `showing ${items.length} of ${fmtInt(total)}`;
  return (
    <div>
      <h3>{title} ({count})</h3>
      {items.length === 0 ? <p className="muted">None returned.</p> : items.map((e, i) => (
        <ExampleCard key={`${title}-${e.task_id}`} e={e} open0={i === 0} />
      ))}
    </div>
  );
}

export function ExamplesCard({ id }: { id: string }) {
  const [res, retry] = useAsync(
    () => Promise.all([api.examples(id, "fixed", 5), api.examples(id, "still_wrong", 5)]),
    id,
  );
  const [fixed, wrong]: (ExamplesResult | undefined)[] = res.state === "ok" ? res.data : [];
  const none = fixed && wrong && fixed.items.length + wrong.items.length === 0;
  return (
    <Card title="Examples: fixed vs still wrong">
      {res.state === "loading" && <Spinner label="Loading examples" />}
      {res.state === "error" && <ApiErrorState title="Examples unavailable" error={res.error} onRetry={retry} />}
      {none && fixed.available === false && (
        <EmptyState title="This report has no examples">The report was written without a held-out examples field.</EmptyState>
      )}
      {none && fixed.available !== false && <EmptyState title="No examples returned for this run" />}
      {fixed && wrong && !none && (
        <div className="stack">
          <ExampleList title="Fixed (student right, base wrong)" items={fixed.items} total={fixed.totals?.fixed} />
          <ExampleList title="Still wrong (base and student wrong)" items={wrong.items} total={wrong.totals?.still_wrong} />
        </div>
      )}
    </Card>
  );
}

export function ReportView({ r }: { r: Report }) {
  return (
    <div className="rp stack">
      <LabelBanner dry_run={r.dry_run} recorded={r.recorded} recorded_at={r.recorded_at} />
      <div>
        <h1>Report <span className="mono muted" style={{ fontSize: ".5em" }}>{r.run_id}</span></h1>
        <div className="row">
          <Link to={`/runs/${encodeURIComponent(r.run_id)}/tree`}>Experiment tree</Link>
          <Link to={`/runs/${encodeURIComponent(r.run_id)}`}>Run detail</Link>
          <span className="muted">pack {r.pack} · candidate round {fmtInt(r.candidate_round)}</span>
        </div>
      </div>
      <Verdict r={r} />
      <Accuracy r={r} />
      <GateDetails r={r} />
      <CostLatency r={r} />
      <Counters r={r} />
      <Clusters r={r} />
      <ExamplesCard id={r.run_id} />
    </div>
  );
}

export default function ReportScreen() {
  const { id = "" } = useParams();
  const [res, retry] = useAsync(() => api.report(id), id);
  if (res.state === "loading") return <Spinner label="Loading report" />;
  if (res.state === "error") {
    const e = res.error;
    if (e instanceof ApiError && e.code === "report_not_ready")
      return <EmptyState title="Report not ready">{e.code}: {e.message} <Link to={`/runs/${encodeURIComponent(id)}`}>Watch it live</Link></EmptyState>;
    return <ApiErrorState error={e} onRetry={retry} />;
  }
  return <ReportView r={res.data} />;
}
