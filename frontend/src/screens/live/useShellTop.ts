import { useLayoutEffect, useState } from "react";

/**
 * The run shell's header is sticky on wide screens and its height changes (the label banner arrives, text wraps).
 * Sticky pieces of the Live screen must sit just below it, so this measures `.run-sticky` and publishes its height as
 * `--live-shell` on the element the ref is attached to; when the header is not sticky (phones) the value is zero.
 * Layout only: the value is never displayed. Outside a run shell it does nothing.
 * Returns a callback ref, so it also works when the element mounts later than the component (after loading).
 */
export function useShellTop<T extends HTMLElement>(): (el: T | null) => void {
  const [el, setEl] = useState<T | null>(null);
  useLayoutEffect(() => {
    const bar = el?.closest(".run-shell")?.querySelector<HTMLElement>(".run-sticky");
    if (!el || !bar) return;
    const measure = () => {
      const sticky = getComputedStyle(bar).position === "sticky";
      el.style.setProperty("--live-shell", sticky ? `${bar.offsetHeight}px` : "0px");
    };
    measure();
    window.addEventListener("resize", measure);
    const ro = typeof ResizeObserver === "function" ? new ResizeObserver(measure) : null;
    ro?.observe(bar);
    return () => {
      window.removeEventListener("resize", measure);
      ro?.disconnect();
    };
  }, [el]);
  return setEl;
}
