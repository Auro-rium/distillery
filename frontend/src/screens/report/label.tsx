import type { RunSummary } from "../../api/types";
import { Badge } from "../../components";

export type Flags = Pick<RunSummary, "dry_run" | "recorded" | "recorded_at">;

/** What the label says, by the same rule as LabelBanner: a missing flag is never read as real. */
export function labelKey(b: Partial<Flags>): string {
  if (b.dry_run === true) return "dry";
  if (b.dry_run === false && b.recorded === true) return `recorded:${b.recorded_at ?? ""}`;
  if (b.dry_run === false && b.recorded === false) return "live";
  return "unknown";
}

/** Short label badge for a card or a sticky summary, from the payload's own flags. */
export function LabelTag({ b }: { b: Partial<Flags> }) {
  const k = labelKey(b);
  if (k === "dry") return <Badge tone="warn">dry run</Badge>;
  if (k.startsWith("recorded")) return <Badge tone="info">recorded</Badge>;
  if (k === "live") return <Badge tone="info">live run</Badge>;
  return <Badge tone="warn">label unknown</Badge>;
}
