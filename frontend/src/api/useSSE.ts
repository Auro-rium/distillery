import { useEffect, useRef, useState } from "react";
import type { RunEvent } from "./types";

export type SSEState = "connecting" | "open" | "reconnecting" | "closed";
const TYPES = ["stage", "spend", "log", "done"] as const;
export const STALE_MS = 30_000;

export interface StreamRefusal { status: number; code: string; message: string }

/** Reconnect URL. The server supports `?last_event_id=` (header alone is unreliable). */
export function resumeUrl(url: string, lastId: number): string {
  if (lastId <= 0) return url;
  return `${url}${url.includes("?") ? "&" : "?"}last_event_id=${lastId}`;
}

/**
 * Subscribe to a run's event stream.
 * - Every reconnect is done by us (never the browser's silent retry) so that it carries
 *   `?last_event_id=`; ids <= the last seen are still dropped client-side.
 * - `stale` is true while disconnected, and when no event arrived for `staleMs` (30 s). Callers
 *   must then stop presenting the last values as current. It clears on reconnect or on any event.
 * - `refusal` is set when the server answers the stream with an HTTP error (e.g. 429
 *   too_many_streams), which EventSource itself cannot report.
 * Closes on `done`, on unmount, or when url is null.
 */
export function useSSE(url: string | null, maxDelayMs = 15000, staleMs = STALE_MS) {
  const [events, setEvents] = useState<RunEvent[]>([]);
  const [state, setState] = useState<SSEState>("closed");
  const [quiet, setQuiet] = useState(false);
  const [refusal, setRefusal] = useState<StreamRefusal | null>(null);
  const lastId = useRef(0);

  useEffect(() => {
    setEvents([]);
    setQuiet(false);
    setRefusal(null);
    lastId.current = 0;
    if (!url) {
      setState("closed");
      return;
    }
    let es: EventSource | null = null;
    let timer: ReturnType<typeof setTimeout> | undefined;
    let staleTimer: ReturnType<typeof setTimeout> | undefined;
    let attempt = 0;
    let everOpened = false;
    let stopped = false;
    const probe = new AbortController();

    const touch = () => {
      setQuiet(false);
      clearTimeout(staleTimer);
      staleTimer = setTimeout(() => setQuiet(true), staleMs);
    };

    // EventSource hides HTTP status codes; ask once, and abort right after the headers.
    const explain = () => {
      if (everOpened || typeof fetch !== "function") return;
      fetch(resumeUrl(url, lastId.current), { signal: probe.signal, headers: { Accept: "text/event-stream" } })
        .then(async (res) => {
          if (res.ok) return probe.abort();
          const b = (await res.json().catch(() => ({}))) as Partial<{ error: string; message: string }>;
          if (!stopped) setRefusal({ status: res.status, code: b.error ?? `http_${res.status}`, message: b.message ?? res.statusText });
        })
        .catch(() => undefined);
    };

    const open = () => {
      es = new EventSource(resumeUrl(url, lastId.current));
      setState(attempt === 0 ? "connecting" : "reconnecting");
      es.onopen = () => {
        attempt = 0;
        everOpened = true;
        setRefusal(null);
        setState("open");
        touch();
      };
      for (const type of TYPES) {
        es.addEventListener(type, (ev) => {
          const m = ev as MessageEvent<string>;
          const id = Number(m.lastEventId);
          touch();
          if (Number.isFinite(id) && id > 0) {
            if (id <= lastId.current) return;
            lastId.current = id;
          }
          let data: unknown;
          try {
            data = JSON.parse(m.data);
          } catch {
            return;
          }
          setEvents((prev) => [...prev, { id, type, data } as RunEvent]);
          if (type === "done") {
            stopped = true;
            clearTimeout(staleTimer);
            setQuiet(false);
            es?.close();
            setState("closed");
          }
        });
      }
      // Liveness only: resets the stale timer; never becomes a stage/spend/log event.
      es.addEventListener("heartbeat", () => { if (!stopped) touch(); });
      es.onerror = () => {
        if (stopped) return;
        setState("reconnecting");
        es?.close();
        if (attempt === 0) explain();
        const delay = Math.min(maxDelayMs, 500 * 2 ** attempt++);
        timer = setTimeout(open, delay);
      };
    };
    open();

    return () => {
      stopped = true;
      probe.abort();
      clearTimeout(timer);
      clearTimeout(staleTimer);
      es?.close();
    };
  }, [url, maxDelayMs, staleMs]);

  const stale = state === "reconnecting" || (state !== "closed" && quiet);
  return { events, state, stale, refusal };
}
