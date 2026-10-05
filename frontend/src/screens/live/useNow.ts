import { useEffect, useState } from "react";

const TICK_MS = 1000;

/** Wall-clock ms, re-read every second while `on`; null when off (nothing is extrapolated then). */
export function useNow(on: boolean): number | null {
  const [now, setNow] = useState(() => Date.now());
  useEffect(() => {
    if (!on) return;
    setNow(Date.now());
    const t = setInterval(() => setNow(Date.now()), TICK_MS);
    return () => clearInterval(t);
  }, [on]);
  return on ? now : null;
}
