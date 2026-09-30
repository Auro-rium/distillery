import { useLayoutEffect, useRef } from "react";
import { useReducedMotion } from "./useReducedMotion";
import { COUNT_MS, easeOut } from "./timing";

/**
 * Shows `format(value)`. When `value` changes it eases from the number on screen to the new real
 * value and ALWAYS finishes by writing `format(value)` itself, so the last frame is the exact
 * payload value and never an interpolated one. Absent values (null, NaN) go through `format` and
 * show "not measured" without animating.
 *
 * With reduced motion (or no matchMedia, as in tests) the final text is written at once.
 * The text is written straight to the DOM node from requestAnimationFrame, so there is no
 * per-frame React state. React owns no children of this span.
 *
 * `fromZero`: on first appearance count up from zero, but only when the element scrolls into
 * view (or at once without IntersectionObserver); until then the final value is in the DOM (find-in-page, print and screen readers see it).
 */
export function CountUp(props: {
  value: number | null | undefined;
  format: (v: number | null | undefined) => string;
  fromZero?: boolean;
  className?: string;
}) {
  const { value, format, fromZero = false, className } = props;
  const ref = useRef<HTMLSpanElement>(null);
  const shown = useRef<number | null>(null); // number currently on screen, if any
  const reduced = useReducedMotion();
  const fmt = useRef(format);
  fmt.current = format;

  const intro = useRef(fromZero); // the from-zero intro still has to play

  useLayoutEffect(() => {
    const el = ref.current;
    if (!el) return;
    const finite = typeof value === "number" && Number.isFinite(value);
    const finalText = fmt.current(value);
    const settle = () => { el.textContent = finalText; shown.current = finite ? (value as number) : null; };
    if (reduced || !finite || typeof requestAnimationFrame !== "function") return settle();
    const playIntro = intro.current;
    if (!playIntro && shown.current === null) return settle(); // first appearance: no count from zero
    const from = playIntro ? 0 : (shown.current as number);
    if (from === value) return settle();
    const target = value as number;

    let raf = 0;
    let io: IntersectionObserver | undefined;
    const run = () => {
      intro.current = false;
      let t0: number | null = null;
      const step = (t: number) => {
        if (t0 === null) t0 = t;
        const p = Math.min(1, (t - t0) / COUNT_MS);
        if (p >= 1) return settle();
        const cur = from + (target - from) * easeOut(p);
        shown.current = cur;
        el.textContent = fmt.current(cur);
        raf = requestAnimationFrame(step);
      };
      el.textContent = fmt.current(from);
      shown.current = from;
      raf = requestAnimationFrame(step);
    };
    if (playIntro && typeof IntersectionObserver !== "undefined") {
      settle(); // the final value stays in the DOM until the element is seen
      io = new IntersectionObserver((es) => {
        if (es.some((e) => e.isIntersecting)) { io?.disconnect(); run(); }
      }, { threshold: 0.2 });
      io.observe(el);
    } else run();
    return () => {
      cancelAnimationFrame(raf);
      io?.disconnect();
      settle(); // never leave a half-way number behind
    };
  }, [value, reduced]);

  return <span ref={ref} className={className} />;
}
