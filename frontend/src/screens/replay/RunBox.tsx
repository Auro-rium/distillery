import { useEffect, useId, useRef, useState } from "react";
import { api } from "../../api/client";
import { fmtDateTimeUtc } from "../../api/format";
import type { RunDetail, RunSummary } from "../../api/types";
import { ApiErrorState } from "../../components";
import { useAsync } from "../report/useAsync";

/** The stage that is the test: the sealed held-out evaluation, run once, whose result is the PROMOTE / REJECT decision. */
export const TEST_STAGE = "final_eval";

/** What kind of run it is, in words, for a recorded run (the DRY RUN and unknown labels stay as banners above). */
function recordedLine(run: Pick<RunSummary, "dry_run" | "recorded">): string | null {
  return run.dry_run === false && run.recorded === true ? "Recorded run · real Token Factory jobs" : null;
}

/** When the test was done: the end of the run's own test stage, as the server recorded it. */
function When({ run }: { run: RunSummary }) {
  const [detail, retry] = useAsync<RunDetail>(() => api.run(run.run_id), `run-when:${run.run_id}`);
  if (detail.state === "loading") return <p className="muted m-run-when">Loading when the test ran</p>;
  if (detail.state === "error") return <ApiErrorState title="Could not load when the test ran" error={detail.error} onRetry={retry} />;
  const stage = (detail.data.stages ?? []).find((s) => s.name === TEST_STAGE && s.status === "done");
  return (
    <p className="m-run-when">
      <span className="eyebrow">Test done</span>
      <span className="num">{fmtDateTimeUtc(stage?.ended_at)}</span>
    </p>
  );
}

/**
 * The run's name in a box themed like its PROMOTE / REJECT chip. The arrow opens a small panel with when the test was
 * done (read from the run's own stage record, fetched when opened) and what kind of run it is.
 */
export function RunBox({ run, tone }: { run: RunSummary; tone: "ok" | "bad" | "neutral" }) {
  const [open, setOpen] = useState(false);
  const panel = useId();
  const root = useRef<HTMLDivElement>(null);
  useEffect(() => {
    if (!open) return;
    const away = (e: PointerEvent) => { if (root.current && !root.current.contains(e.target as Node)) setOpen(false); };
    const esc = (e: KeyboardEvent) => { if (e.key === "Escape") setOpen(false); };
    document.addEventListener("pointerdown", away);
    document.addEventListener("keydown", esc);
    return () => { document.removeEventListener("pointerdown", away); document.removeEventListener("keydown", esc); };
  }, [open]);
  const kind = recordedLine(run);
  return (
    <div className="m-runbox" ref={root}>
      <button
        type="button"
        className={`badge lg m-runbtn${tone === "neutral" ? "" : ` ${tone}`}`}
        aria-expanded={open}
        aria-controls={panel}
        onClick={() => setOpen((o) => !o)}
      >
        <span className="m-runid">{run.run_id}</span>
        <span className="m-caret" aria-hidden="true">▾</span>
        <span className="sr-only">{open ? "Hide when the test ran" : "Show when the test ran"}</span>
      </button>
      <div id={panel} className="m-runpanel" role="region" aria-label="When the test ran" hidden={!open}>
        {open && <When run={run} />}
        {open && kind && <p className="muted m-run-kind">{kind}</p>}
      </div>
    </div>
  );
}
