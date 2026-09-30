import { useCallback, useEffect, useState } from "react";
import { api } from "../../api/client";
import type { Config } from "../../api/types";

export type Limits =
  | { state: "loading" }
  | { state: "error"; error: unknown }
  | { state: "ok"; pg: Config["playground"] };

/**
 * The playground's caps and today's spend, as GET /api/config returns them. `refresh` re-reads them (after a
 * request the spent figure has moved) without blanking what is on screen; if that re-read fails the last
 * figures stay, and they are still the API's, only older.
 */
export function useLimits(): [Limits, () => void] {
  const [limits, setLimits] = useState<Limits>({ state: "loading" });
  const load = useCallback((quiet: boolean) => {
    let live = true;
    api.config().then(
      (c) => { if (live) setLimits({ state: "ok", pg: c.playground }); },
      (error: unknown) => { if (live && !quiet) setLimits({ state: "error", error }); },
    );
    return () => { live = false; };
  }, []);
  useEffect(() => load(false), [load]);
  return [limits, useCallback(() => { load(true); }, [load])];
}
