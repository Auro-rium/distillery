import { fmtInt, fmtPercent, fmtText, fmtUsd } from "../../api/format";
import { StressBars, type StressRow } from "../../charts";
import { costOf, finetuneLines, perMillion, reportData, stressOf } from "./mission";

/** Where the student is weak (unseen task families) and what the run cost, each from the featured report. */
export function Limits({ report }: { report: unknown }) {
  const s = stressOf(report);
  const d = reportData(report);
  const c = costOf(report);
  const lines = finetuneLines(report);
  const rows: StressRow[] = s
    ? [
        { name: "all unseen families", n: s.n, values: { student: s.accuracy?.student, teacher: s.accuracy?.teacher } },
        ...Object.entries(s.accuracy_by_family).map(([name, a]) => ({ name, values: { student: a?.student, teacher: a?.teacher } })),
      ]
    : [];
  return (
    <div className="m-limits">
      <div className="m-limit-chart">
        {s ? (
          <StressBars
            rows={rows}
            title="Stress set: task families never seen in training"
            caption={s.note ?? "Accuracy on task families held out of train, dev and gate; reported separately, not a gate input."}
          />
        ) : (
          <p className="muted">This run&rsquo;s report has no stress-set evaluation.</p>
        )}
        <p className="m-overlap">
          <span className="eyebrow">In-distribution</span>{" "}
          <span className="num">{fmtPercent(d?.heldout_skeleton_overlap_rate, 1)}</span> of held-out tasks share a query
          skeleton with a training task{d?.heldout_kind ? <> ({d.heldout_kind})</> : null}. The gate measures this task
          distribution; the stress set above is the out-of-distribution check.
        </p>
      </div>
      <div className="m-cost">
        <div className="stat m-total">
          <div className="v num">{fmtUsd(c?.run_total_usd, 2)}</div>
          <div className="l eyebrow">run total</div>
          <div className="h">cap {fmtUsd(c?.run_cap_usd, 2)}</div>
        </div>
        {lines.length > 0 && (
          <ul className="m-ftlines" role="list" aria-label="Fine-tune cost">
            {lines.map((l, i) => (
              <li key={i}>
                <span className="eyebrow">fine-tune{l.kind && l.kind !== "finetune" ? ` (${l.kind})` : ""}</span>{" "}
                <span className="num">{fmtUsd(l.usd, 4)}</span> for <span className="num">{fmtInt(l.units)}</span> tokens
                {" · "}<span className="num">{perMillion(l)}</span> per million tokens (derived)
                {l.basis && <span className="muted"> · {l.basis}</span>}
              </li>
            ))}
          </ul>
        )}
        <details className="m-basis">
          <summary>Cost basis</summary>
          <p className="muted">{fmtText(c?.basis)}</p>
        </details>
      </div>
    </div>
  );
}
