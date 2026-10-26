import { useState } from "react";
import { LazyDialog } from "../../ui/LazyDialog";
import "./hint.css";

/** A run name is `{pack}-{scale}-{start time}` plus, on stored runs, a short tag. Anything else is shown whole. */
const NAME = /^([^-]+)-([^-]+)-(\d{8}T\d{6}Z)(?:-(.+))?$/;

function RunName({ id }: { id: string }) {
  const m = NAME.exec(id);
  if (!m) return <p>The run is called <span className="mono">{id}</span>.</p>;
  const [, pack, scale, stamp, tag] = m;
  const parts: [string, string][] = [
    [pack, "the job it was taught"],
    [scale, "the kind of run. A “gated” run ends at a strict pass/fail gate"],
    [stamp, "when the run started (date and time, UTC)"],
  ];
  if (tag) parts.push([tag, "a short tag, so no two runs share a name"]);
  return (
    <dl className="m-help-parts">
      {parts.map(([k, v]) => (
        <div key={k}><dt className="mono">{k}</dt><dd>{v}</dd></div>
      ))}
    </dl>
  );
}

/**
 * The "Mission" label at the top of the page, as a box that gently pulses outward. Clicking it explains the page in
 * plain words. The wording carries no figures; the numbers are the ones in the verdict box, read from the report.
 */
export function MissionHint({ runId }: { runId: string | null }) {
  const [open, setOpen] = useState(false);
  return (
    <>
      <button type="button" className="m-hint" aria-haspopup="dialog" onClick={() => setOpen(true)}>
        <span>Mission</span>
        <span className="m-hint-q" aria-hidden="true">?</span>
        <span className="sr-only"> (what is this page about?)</span>
      </button>
      <LazyDialog open={open} onOpenChange={setOpen} title="The mission, in easy words" className="m-help-dialog">
        <div className="m-help">
          <p className="m-help-lead">We taught a small AI to do a job a big AI does, then checked honestly whether it worked.</p>
          <ul>
            <li><strong>Teacher:</strong> the big AI. Smart, but slow and costly.</li>
            <li><strong>Student:</strong> the small AI. Cheaper and faster. We trained it to copy the teacher on one job: turning questions into SQL (database search commands).</li>
            <li><strong>Base:</strong> the same small AI before any training, to compare against.</li>
          </ul>

          <h3>The top box</h3>
          <p>
            It shows the latest recorded run. PROMOTE means the small AI passed the test; REJECT means it did not. The three
            big numbers are the scores of the student, the teacher and the untrained base on test questions none of them had seen.
          </p>

          {runId && (
            <>
              <h3>The name next to it</h3>
              <p>It is like a name tag on an experiment. Its parts:</p>
              <RunName id={runId} />
            </>
          )}

          <h3>What “gate” means</h3>
          <p>
            A strict pass/fail check. Plain software does it, not another AI, so nobody can talk it into a yes. The student is
            promoted only if it scores close enough to the teacher and clearly beats the untrained base, by more than luck could explain.
          </p>
        </div>
      </LazyDialog>
    </>
  );
}
