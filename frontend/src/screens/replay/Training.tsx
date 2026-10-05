import { fmtInt, fmtPercent, fmtText } from "../../api/format";
import { LossChart } from "../../charts";
import { candidateDevAcc, finetuneOf, gateOf, hpChips } from "./mission";

/** Fine-tune record of the featured run: loss curve, trained tokens and steps, config chips. */
export function Training({ report }: { report: unknown }) {
  const f = finetuneOf(report);
  const g = gateOf(report);
  if (!f) return <p className="muted">This run&rsquo;s report has no fine-tune record (it was recorded before that field existed).</p>;
  const points = Array.isArray(f.loss_curve) ? f.loss_curve : [];
  return (
    <div className="m-training">
      <LossChart
        points={points}
        title="Train and validation loss by checkpoint"
        caption="Loss at each recorded checkpoint of the first fine-tune round, no smoothing."
      />
      <div className="m-train-side">
        <ul className="m-chips" role="list" aria-label="Fine-tune record">
          <li><span className="eyebrow">trained tokens</span> <span className="num">{fmtInt(f.trained_tokens)}</span></li>
          <li><span className="eyebrow">steps</span> <span className="num">{fmtInt(f.trained_steps)} / {fmtInt(f.total_steps)}</span></li>
          <li><span className="eyebrow">base model</span> <span className="mono">{fmtText(f.base_model)}</span></li>
          {hpChips(f).map((c) => (
            <li key={c.key}><span className="eyebrow">{c.label}</span> <span className="num">{c.value}</span></li>
          ))}
        </ul>
        <p className="m-interp">
          <span className="eyebrow">Interpretation</span>{" "}
          Validation loss is token-level loss against the gold SQL text, while the student was trained on SQL the
          teacher wrote, so a rising validation loss does not by itself mean worse answers. The gate metric is
          execution accuracy: the query is run and its result compared.
        </p>
        <p className="muted m-acc">
          Execution accuracy: dev <span className="num">{fmtPercent(candidateDevAcc(report), 1)}</span> · sealed held-out{" "}
          <span className="num">{fmtPercent(g?.student_acc, 1)}</span>
        </p>
      </div>
    </div>
  );
}
