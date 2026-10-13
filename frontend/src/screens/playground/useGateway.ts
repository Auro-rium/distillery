import { useCallback, useEffect, useRef, useState } from "react";
import { api } from "../../api/client";
import type { GatewaySavings, GatewayStatus } from "../../api/types";

export type Loaded<T> = { state: "loading" } | { state: "error"; error: unknown } | { state: "ok"; data: T };

/**
 * Loads `fn` on mount. `refresh` re-reads it without blanking what is on screen (the figures have moved after a
 * question): if that re-read fails the last figures stay, and they are still the server's, only older.
 */
function useReread<T>(fn: () => Promise<T>): [Loaded<T>, () => void] {
  const [res, setRes] = useState<Loaded<T>>({ state: "loading" });
  const live = useRef(true);
  const load = useCallback(
    (quiet: boolean) => {
      fn().then(
        (data) => { if (live.current) setRes({ state: "ok", data }); },
        (error: unknown) => { if (live.current && !quiet) setRes({ state: "error", error }); },
      );
    },
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [],
  );
  useEffect(() => {
    live.current = true;
    load(false);
    return () => { live.current = false; };
  }, [load]);
  return [res, useCallback(() => load(true), [load])];
}

/** GET /api/gateway/status and GET /api/gateway/savings; `refresh` re-reads both after a question was asked. */
export function useGateway(): { status: Loaded<GatewayStatus>; savings: Loaded<GatewaySavings>; refresh: () => void } {
  const [status, refreshStatus] = useReread(api.gatewayStatus);
  const [savings, refreshSavings] = useReread(api.gatewaySavings);
  const refresh = useCallback(() => { refreshStatus(); refreshSavings(); }, [refreshStatus, refreshSavings]);
  return { status, savings, refresh };
}
