import { fmtInt } from "../../api/format";
import type { RunDetail } from "../../api/types";
import { Panel } from "./Panel";
import { Metric, Val } from "./Val";

/** Sandbox counters. The core records none today, so these normally read "not measured". */
export function SandboxPanel({ sandbox }: { sandbox: RunDetail["sandbox"] | null | undefined }) {
  return (
    <Panel title="Sandbox">
      <dl className="metrics">
        <Metric label="Operations" value={<Val text={fmtInt(sandbox?.operations)} />} />
        <Metric label="Peak concurrency" value={<Val text={fmtInt(sandbox?.concurrency_peak)} />} />
      </dl>
    </Panel>
  );
}
