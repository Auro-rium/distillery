import { useEffect, useRef, useState } from "react";
import { useReducedMotion } from "./useReducedMotion";

/**
 * Becomes true once, the first time the element is on screen. With reduced motion, or without
 * IntersectionObserver, it is true from the first render so final values are shown immediately.
 */
export function useInView<T extends Element>(): [React.RefObject<T>, boolean] {
  const ref = useRef<T>(null);
  const reduced = useReducedMotion();
  const noObserver = typeof IntersectionObserver === "undefined";
  const [seen, setSeen] = useState(false);
  useEffect(() => {
    const el = ref.current;
    if (seen || reduced || noObserver || !el) return;
    const io = new IntersectionObserver((entries) => {
      if (entries.some((e) => e.isIntersecting)) {
        setSeen(true);
        io.disconnect();
      }
    }, { threshold: 0.2 });
    io.observe(el);
    return () => io.disconnect();
  }, [seen, reduced, noObserver]);
  return [ref as React.RefObject<T>, seen || reduced || noObserver];
}
