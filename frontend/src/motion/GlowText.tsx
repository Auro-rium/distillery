import { useEffect, useRef, type ElementType } from "react";
import "./glow.css";
import { useReducedMotion } from "./useReducedMotion";

interface Props {
  /** The element to render: a heading or a paragraph. */
  as?: ElementType;
  id?: string;
  className?: string;
  children: string;
}

const OFF = "-9999px";

/**
 * Text with a neon light that follows the pointer. The light is a gradient clipped to the glyphs, plus a blurred
 * copy of the text behind them, so it shows only on the text and nowhere else. Skipped (plain text) when the user
 * prefers reduced motion. The copy behind is `aria-hidden`; assistive technology reads the text once.
 */
export function GlowText({ as: Tag = "p", id, className, children }: Props) {
  const ref = useRef<HTMLElement>(null);
  const reduced = useReducedMotion();
  useEffect(() => {
    const el = ref.current;
    if (reduced || !el) return;
    let raf = 0;
    let x = 0;
    let y = 0;
    const paint = () => {
      raf = 0;
      const r = el.getBoundingClientRect();
      el.style.setProperty("--mx", `${x - r.left}px`);
      el.style.setProperty("--my", `${y - r.top}px`);
    };
    const move = (e: PointerEvent) => {
      if (e.pointerType === "touch") return;
      x = e.clientX;
      y = e.clientY;
      if (!raf) raf = requestAnimationFrame(paint);
    };
    const leave = () => {
      if (raf) cancelAnimationFrame(raf);
      raf = 0;
      el.style.setProperty("--mx", OFF);
      el.style.setProperty("--my", OFF);
    };
    window.addEventListener("pointermove", move, { passive: true });
    document.documentElement.addEventListener("pointerleave", leave);
    return () => {
      window.removeEventListener("pointermove", move);
      document.documentElement.removeEventListener("pointerleave", leave);
      if (raf) cancelAnimationFrame(raf);
    };
  }, [reduced]);
  return (
    <Tag ref={ref} id={id} className={["glow", className].filter(Boolean).join(" ")} data-glow={reduced ? "off" : "on"}>
      <span className="glow-halo" aria-hidden="true">{children}</span>
      <span className="glow-text">{children}</span>
    </Tag>
  );
}
