import { fmtInt, fmtPercent, fmtText } from "../../api/format";
import { LossChart } from "../../charts";
import { DataMenu, type DataRow } from "./DataMenu";
import { candidateDevAcc, finetuneOf, fmtSig, gateOf, hpChips } from "./mission";

const TRAIN_LOSS_HELP = "How wrong the model is on the practice questions it learns from. Lower means it is fitting them better.";
const VALID_LOSS_HELP = "How wrong it is on questions kept out of training. If it rises while train loss falls, the model may be memorising.";

/** Fine-tune record of the featured run: the loss curve, and everything else recorded in a DATA menu. */
export function Training({ report }: { report: unknown }) {
  const f = finetuneOf(report);
  const g = gateOf(report);
  if (!f) return <p className="muted">This run&rsquo;s report has no fine-tune record (it was recorded before that field existed).</p>;
  const points = Array.isArray(f.loss_curve) ? f.loss_curve : [];
  const last = points.length > 0 ? points[points.length - 1] : null;
  const rows: DataRow[] = [
    { key: "tokens", label: "trained tokens", value: <span className="num">{fmtInt(f.trained_tokens)}</span> },
    { key: "steps", label: "steps", value: <span className="num">{fmtInt(f.trained_steps)} / {fmtInt(f.total_steps)}</span> },
    { key: "base", label: "base model", value: <span className="mono">{fmtText(f.base_model)}</span> },
    ...hpChips(f).map((c) => ({ key: c.key, label: c.label, value: <span className="num">{c.value}</span> })),
    { key: "train_loss", label: "train loss", value: <span className="num">{fmtSig(last?.train_loss)}</span>, help: TRAIN_LOSS_HELP, valueTitle: "At the last recorded checkpoint" },
    { key: "valid_loss", label: "validation loss", value: <span className="num">{fmtSig(last?.valid_loss)}</span>, help: VALID_LOSS_HELP, valueTitle: "At the last recorded checkpoint" },
    { key: "dev_acc", label: "dev accuracy", value: <span className="num">{fmtPercent(candidateDevAcc(report), 1)}</span> },
    { key: "heldout_acc", label: "sealed held-out accuracy", value: <span className="num">{fmtPercent(g?.student_acc, 1)}</span> },
  ];
  return (
    <div className="m-training">
      <LossChart
        points={points}
        title="Train and validation loss by checkpoint"
        caption="Loss at each recorded checkpoint of the first fine-tune round, no smoothing."
      />
      <div className="m-train-side">
        <DataMenu rows={rows} />
      </div>
    </div>
  );
}
