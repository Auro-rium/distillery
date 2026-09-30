import { fmtUsd } from "../../api/format";
import { Heartbeat } from "./Heartbeat";
import type { Connection } from "./state";
import { Val } from "./Val";

/**
 * At-a-glance status of an active run: the real connection state, the stage that is running now and the total
 * spend so far. Sticky under the section tabs on phones. Only shown while the run is active.
 */
export function LiveStatus(props: { conn: Connection; events: number; stage: string | null; total: number | null | undefined }) {
  return (
    <div className="live-status" role="group" aria-label="Run status">
      <Heartbeat conn={props.conn} events={props.events} />
      {props.stage && (
        <span className="ls-item"><span className="eyebrow">Now</span><span className="mono ls-stage">{props.stage}</span></span>
      )}
      <span className="ls-item"><span className="eyebrow">Spent</span><Val text={fmtUsd(props.total)} /></span>
    </div>
  );
}
