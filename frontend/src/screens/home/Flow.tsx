import { useEffect, useState, type CSSProperties } from "react";
import { prefersReducedMotion } from "../../motion/useReducedMotion";
import "./flow.css";

const STEPS: { title: string; body: string }[] = [
  { title: "Familiar?", body: "The question is embedded and compared with what the student was certified on. Unfamiliar questions go straight to the teacher, before the student spends anything." },
  { title: "Student drafts", body: "The fine-tuned student writes the SQL on a Nebius sandbox, and the SQL is run read-only against the database." },
  { title: "Router decides", body: "A small model scores confidence from whether the SQL ran, the student's token log-probabilities and the similarity. Only a score at or above the certified threshold is served." },
  { title: "Drift guard", body: "The share of deferred requests is watched over a sliding window and a sample of student answers is audited by the teacher. A burst of unfamiliar traffic rolls the gateway back." },
];

const DECODE_MS = 750;
const FRAME_MS = 55;

/** A 0 or 1 for a character position and animation frame. A hash, not a random number: the same input gives the same bit. */
const bit = (i: number, frame: number): string => (((Math.imul(i + 1, 2654435761) ^ Math.imul(frame + 7, 40503)) >>> 9) & 1) === 1 ? "1" : "0";

/** Columns of binary for the rain behind an open box, as a CSS string (so it is never part of the page text). It is
 *  written twice so the scroll loops without a seam. */
const RAIN_LINES = Array.from({ length: 22 }, (_, r) => Array.from({ length: 20 }, (_, c) => bit(c * 7 + r * 31, r)).join(" "));
const RAIN = `"${[...RAIN_LINES, ...RAIN_LINES].join("\\A ")}"`;

/**
 * Progress of the 0/1 decode while `active`: 0 when it starts, 1 when the real text is fully shown (and whenever the
 * box is closed or motion is reduced). `frame` ticks every few tens of ms so the bits flicker.
 */
function useDecode(active: boolean): { progress: number; frame: number } {
  const [state, setState] = useState({ progress: 1, frame: 0 });
  useEffect(() => {
    if (!active || prefersReducedMotion()) { setState({ progress: 1, frame: 0 }); return; }
    let raf = 0;
    let t0 = -1;
    let last = -1;
    const tick = (now: number) => {
      if (t0 < 0) t0 = now;
      const progress = Math.min(1, (now - t0) / DECODE_MS);
      const frame = Math.floor((now - t0) / FRAME_MS);
      if (frame !== last || progress === 1) { last = frame; setState({ progress, frame }); }
      if (progress < 1) raf = requestAnimationFrame(tick);
    };
    setState({ progress: 0, frame: 0 });
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [active]);
  return state;
}

/**
 * The step text. While decoding, every letter keeps its place (it is only made invisible) and a 0 or 1 is drawn on top
 * of it, so nothing reflows; letters turn real from the start of the text to the end. Assistive technology gets the
 * real text throughout.
 */
function Body({ text, progress, frame }: { text: string; progress: number; frame: number }) {
  if (progress >= 1) return <p className="h-desc-text">{text}</p>;
  let i = 0;
  return (
    <p className="h-desc-text">
      <span className="sr-only">{text}</span>
      <span aria-hidden="true">
        {text.split(" ").map((word, w) => (
          <span key={w} className="h-word">
            {[...word].map((ch) => {
              const idx = i++;
              const done = idx / text.length < progress;
              return done
                ? <span key={idx}>{ch}</span>
                : <span key={idx} className="h-bit" data-b={bit(idx, frame)}>{ch}</span>;
            })}
            {" "}
          </span>
        ))}
      </span>
    </p>
  );
}

function Step({ n, title, body }: { n: number; title: string; body: string }) {
  const [open, setOpen] = useState(false);
  const { progress, frame } = useDecode(open);
  return (
    <li
      className="h-step-slot"
      onPointerEnter={() => setOpen(true)}
      onPointerLeave={() => setOpen(false)}
      onFocus={() => setOpen(true)}
      onBlur={() => setOpen(false)}
    >
      <div className="h-step" data-open={open} tabIndex={0} style={{ "--rain": RAIN } as CSSProperties}>
        <span className="h-step-n mono" aria-hidden="true">{n}</span>
        <h3>{title}</h3>
        <div className="h-desc">
          <div className="h-desc-in">
            <Body text={body} progress={progress} frame={frame} />
          </div>
        </div>
      </div>
    </li>
  );
}

export function Flow() {
  return (
    <section aria-labelledby="h-flow-title">
      <h2 id="h-flow-title">How one request is served</h2>
      <ol className="h-steps">
        {STEPS.map((s, i) => <Step key={s.title} n={i + 1} title={s.title} body={s.body} />)}
      </ol>
    </section>
  );
}
