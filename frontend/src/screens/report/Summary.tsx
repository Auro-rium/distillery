import { useId, useState } from "react";
import { fmtInt, fmtNumber } from "../../api/format";
import type { Report } from "../../api/types";
import { LabelTag } from "./label";
import { useStickyTop } from "./useStickyTop";

/**
 * Compact, sticky summary: the verdict, which label the numbers carry, n and the thresholds the gate
 * used (all from the payload). The gate's reasons stay collapsed until asked for.
 */
export function Summary({ r }: { r: Report }) {
  const g = r.evaluation.gate;
  const t = g.thresholds;
  const [open, setOpen] = useState(false);
  const panel = useId();
  const ref = useStickyTop<HTMLElement>();
  const reasons = r.decision_reasons;
  return (
    <section className="rp-summary" aria-label="Gate decision" ref={ref}>
      <div className="rp-sum-row">
        <span className={`rp-verdict ${r.decision}`}>
          <svg viewBox="0 0 16 16" aria-hidden="true" focusable="false" className="rp-verdict-icon">
            {r.decision === "PROMOTE"
              ? <path d="M3 8.5l3.2 3.2L13 4.8" />
              : <path d="M4 4l8 8M12 4l-8 8" />}
          </svg>
          <strong>{r.decision}</strong>
        </span>
        <LabelTag b={r} />
        <dl className="rp-facts">
          <div><dt>Held-out</dt><dd>n={fmtInt(g.n)}</dd></div>
          <div><dt>Ratio lower bound</dt><dd>{fmtNumber(g.ratio_lo)} <span className="rp-need">needs ≥ {fmtNumber(t.ratio_lower_bound_min, 2)}</span></dd></div>
          <div><dt>McNemar p</dt><dd>{fmtNumber(g.mcnemar_p, 4)} <span className="rp-need">alpha {fmtNumber(t.mcnemar_alpha, 2)}</span></dd></div>
        </dl>
        {reasons.length > 0 ? (
          <button type="button" className="rp-toggle" aria-expanded={open} aria-controls={panel} onClick={() => setOpen((o) => !o)}>
            Gate reasons<span className="chev" aria-hidden="true" />
          </button>
        ) : (
          <span className="rp-noreasons muted">No reasons were reported by the gate.</span>
        )}
      </div>
      {open && reasons.length > 0 && (
        <div id={panel} className="rp-reasons">
          <ul>{reasons.map((x) => <li key={x}>{x}</li>)}</ul>
          <p className="muted">
            Thresholds used: ratio lower bound ≥ {fmtNumber(t.ratio_lower_bound_min, 2)}, McNemar alpha {fmtNumber(t.mcnemar_alpha, 2)},
            bootstrap resamples {fmtInt(t.bootstrap_resamples)}, seed {fmtInt(t.seed)}.
          </p>
        </div>
      )}
    </section>
  );
}
