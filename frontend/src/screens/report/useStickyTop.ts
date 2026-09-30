import { useLayoutEffect, useRef, type RefObject } from "react";

/**
 * The run shell's header is sticky and its height changes (label banner arrives, text wraps). Sticky
 * pieces of the page must sit just below it, so this measures `.run-sticky` (and the summary itself)
 * and publishes both heights as `--rp-top` and `--rp-sum` on the report root. When the shell header is
 * not sticky (phones) the offset is zero. Layout only: the values are never displayed.
 */
export function useStickyTop<T extends HTMLElement>(): RefObject<T> {
  const ref = useRef<T>(null);
  useLayoutEffect(() => {
    const el = ref.current;
    const root = el?.closest<HTMLElement>(".rp") ?? el;
    const bar = el?.closest(".run-shell")?.querySelector<HTMLElement>(".run-sticky");
    if (!el || !root || !bar) return;
    const measure = () => {
      const sticky = getComputedStyle(bar).position === "sticky";
      root.style.setProperty("--rp-top", sticky ? `${bar.offsetHeight}px` : "0px");
      root.style.setProperty("--rp-sum", `${el.offsetHeight}px`);
    };
    measure();
    window.addEventListener("resize", measure);
    const ro = typeof ResizeObserver === "function" ? new ResizeObserver(measure) : null;
    ro?.observe(bar);
    ro?.observe(el);
    return () => {
      window.removeEventListener("resize", measure);
      ro?.disconnect();
    };
  }, []);
  return ref;
}
