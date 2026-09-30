import type { Connection } from "./state";

// The words are the README's wording. `title` tells the two "not live" cases apart.
const LABEL: Record<Connection, string> = {
  connected: "live",
  connecting: "connecting",
  stale: "Disconnected / stale",
  disconnected: "Disconnected / stale",
};
const WHY: Record<Connection, string> = {
  connected: "Event stream connected",
  connecting: "Connecting to the event stream",
  stale: "Stream connected but silent, or the last refresh failed: values are last known",
  disconnected: "Event stream disconnected, retrying: values are last known",
};

/**
 * Live indicator driven only by the real stream state. The one-shot ripple restarts when a real
 * event has arrived (`events` is the count of received events); nothing animates while the
 * stream is stale or disconnected, and no timer here pretends to be a heartbeat.
 */
export function Heartbeat({ conn, events }: { conn: Connection; events: number }) {
  return (
    <span className="hb" data-conn={conn} title={WHY[conn]}>
      <span className="hb-dot" aria-hidden="true">
        {conn === "connected" && events > 0 && <span key={events} className="hb-ping" />}
      </span>
      <span className="hb-label">{LABEL[conn]}</span>
      <span className="sr-only">{WHY[conn]}</span>
    </span>
  );
}
