import { useEffect, useRef } from "react";

export interface Shortcuts {
  help: () => void;
  theme: () => void;
  go: (target: "replay" | "new" | "playground") => void;
}

const GO_WINDOW_MS = 1200;
const editable = (t: EventTarget | null): boolean => {
  const el = t as HTMLElement | null;
  return !!el && (el.isContentEditable || /^(INPUT|TEXTAREA|SELECT)$/.test(el.tagName));
};

/** Global keys: ? help, t theme, g then r / n / p to go somewhere. Ignored while typing or with modifiers. */
export function useShortcuts(s: Shortcuts): void {
  const cb = useRef(s);
  cb.current = s;
  useEffect(() => {
    let armed = 0;
    const onKey = (e: KeyboardEvent) => {
      if (e.ctrlKey || e.metaKey || e.altKey || editable(e.target)) return;
      const now = Date.now();
      if (armed && now - armed < GO_WINDOW_MS) {
        armed = 0;
        const to = { r: "replay", n: "new", p: "playground" } as const;
        const k = e.key.toLowerCase();
        if (k in to) { e.preventDefault(); cb.current.go(to[k as keyof typeof to]); return; }
      }
      if (e.key === "?") { e.preventDefault(); cb.current.help(); }
      else if (e.key === "t") cb.current.theme();
      else if (e.key === "g") armed = now;
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, []);
}
