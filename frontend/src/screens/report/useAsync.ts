import { useCallback, useEffect, useState } from "react";
import { ApiError } from "../../api/client";

export type Async<T> =
  | { state: "loading" }
  | { state: "error"; error: ApiError | Error }
  | { state: "ok"; data: T };

/** Runs `fn` on mount / when `key` changes. `retry` reruns it. */
export function useAsync<T>(fn: () => Promise<T>, key: string): [Async<T>, () => void] {
  const [res, setRes] = useState<Async<T>>({ state: "loading" });
  const [tick, setTick] = useState(0);
  useEffect(() => {
    let live = true;
    setRes({ state: "loading" });
    fn().then(
      (data) => live && setRes({ state: "ok", data }),
      (error: unknown) =>
        live && setRes({ state: "error", error: error instanceof Error ? error : new Error(String(error)) }),
    );
    return () => {
      live = false;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [key, tick]);
  return [res, useCallback(() => setTick((t) => t + 1), [])];
}
