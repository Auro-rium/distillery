import { useSyncExternalStore } from "react";

/** Where the examples browser switches from a stacked list to a list beside the open example. Keep in step with report.css. */
const QUERY = "(min-width: 900px)";

function subscribe(onChange: () => void): () => void {
  if (typeof window === "undefined" || typeof window.matchMedia !== "function") return () => undefined;
  const mq = window.matchMedia(QUERY);
  mq.addEventListener?.("change", onChange);
  return () => mq.removeEventListener?.("change", onChange);
}
const read = (): boolean => typeof window !== "undefined" && typeof window.matchMedia === "function" && window.matchMedia(QUERY).matches;

/** True when the window is wide enough for two columns. Without matchMedia (tests, very old browsers) it is false: the stacked layout. */
export function useWide(): boolean {
  return useSyncExternalStore(subscribe, read, () => false);
}
