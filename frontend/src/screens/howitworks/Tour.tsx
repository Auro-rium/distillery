import { useCallback, useEffect, useId, useState, type CSSProperties, type KeyboardEvent } from "react";
import { useReducedMotion } from "../../motion/useReducedMotion";
import { ACTOR, type Step } from "./steps";

const STEP_MS = 2800;

/**
 * One act of the tour: a track of stops and a panel for the current one. Click a stop, use Back / Next or the arrow
 * keys, or press Play to walk through on its own (it stops at the end, and any manual move pauses it). Under reduced
 * motion Play is not offered and nothing animates.
 */
export function Tour({ title, lead, steps }: { title: string; lead: string; steps: Step[] }) {
  const [at, setAt] = useState(0);
  const [playing, setPlaying] = useState(false);
  const reduced = useReducedMotion();
  const uid = useId();
  const last = steps.length - 1;
  const step = steps[at];

  const go = useCallback((i: number) => setAt(Math.max(0, Math.min(last, i))), [last]);
  const manual = (i: number) => { setPlaying(false); go(i); };

  useEffect(() => {
    if (!playing) return;
    if (at >= last) { setPlaying(false); return; }
    const t = setTimeout(() => setAt((i) => i + 1), STEP_MS);
    return () => clearTimeout(t);
  }, [playing, at, last]);

  const onKey = (e: KeyboardEvent<HTMLDivElement>) => {
    if (e.key === "ArrowRight") { e.preventDefault(); manual(at + 1); }
    else if (e.key === "ArrowLeft") { e.preventDefault(); manual(at - 1); }
  };
  const loopTarget = step.loopsTo ? steps.find((s) => s.id === step.loopsTo) : undefined;

  return (
    <section className="tour" aria-labelledby={`${uid}-h`} data-playing={playing}>
      <div className="tour-head">
        <div>
          <h2 id={`${uid}-h`}>{title}</h2>
          <p className="muted">{lead}</p>
        </div>
        <div className="tour-controls">
          <button type="button" className="tour-btn" onClick={() => manual(at - 1)} disabled={at === 0}>Back</button>
          {!reduced && (
            <button
              type="button"
              className="tour-btn tour-play"
              aria-pressed={playing}
              onClick={() => { if (playing) setPlaying(false); else { if (at >= last) setAt(0); setPlaying(true); } }}
            >
              {playing ? "Pause" : at >= last ? "Replay" : "Play"}
            </button>
          )}
          <button type="button" className="tour-btn" onClick={() => manual(at + 1)} disabled={at === last}>Next</button>
        </div>
      </div>

      <div className="tour-track" role="group" aria-label={`${title}: stops`} onKeyDown={onKey} style={{ "--n": steps.length } as CSSProperties}>
        <div className="tour-rail" aria-hidden="true">
          <div className="tour-rail-fill" style={{ transform: `scaleX(${last === 0 ? 1 : at / last})` }} />
        </div>
        <ol className="tour-stops">
          {steps.map((s, i) => (
            <li key={s.id} className="tour-stop" data-state={i < at ? "done" : i === at ? "now" : "next"}>
              <button type="button" className="tour-dot" aria-current={i === at ? "step" : undefined} onClick={() => manual(i)}>
                <span className="tour-dot-core" aria-hidden="true" />
                <span className="tour-stop-name">{s.title}</span>
              </button>
              {s.loopsTo && <span className="tour-loop" aria-hidden="true">↺</span>}
            </li>
          ))}
        </ol>
      </div>

      <article className="tour-panel" key={step.id} aria-live={playing ? "off" : "polite"}>
        <div className="tour-panel-top">
          <h3>{step.title}</h3>
          <ul className="tour-who" aria-label="Who does it">
            {step.who.map((w) => <li key={w} className={`who-${w}`}>{ACTOR[w]}</li>)}
          </ul>
        </div>
        <p className="tour-what">{step.what}</p>
        <p className="tour-trust"><span className="tour-trust-tag">Why you can trust it</span> {step.trust}</p>
        {loopTarget && (
          <p className="tour-loop-note">
            Then back to <button type="button" className="tour-link" onClick={() => manual(steps.indexOf(loopTarget))}>{loopTarget.title}</button>, for as many rounds as the run allows.
          </p>
        )}
        {playing && <div className="tour-timer" aria-hidden="true" style={{ animationDuration: `${STEP_MS}ms` }} />}
      </article>
    </section>
  );
}
