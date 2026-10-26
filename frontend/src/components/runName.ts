// Plain-words names for a run, taken from its id (`{pack}-{scale}-{start}`, with a `dry-` prefix on fake-model runs).
// Only packs the app knows by name are named; any other id gets the neutral wording, never a guess.

const PACK_LABEL: Record<string, string> = { sql: "SQL", toolcall: "Tool-call" };

/** The task the run was about ("SQL"), or null when the id does not start with a known pack. */
export function packLabel(runId: string): string | null {
  const parts = runId.split("-");
  const first = parts[0] === "dry" ? parts[1] : parts[0];
  return (first && PACK_LABEL[first]) || null;
}

/** The heading, e.g. "SQL run"; "Run" when the pack is not known. */
export function runTitle(runId: string): { pack: string | null; rest: string } {
  const pack = packLabel(runId);
  return pack ? { pack, rest: " run" } : { pack: null, rest: "Run" };
}

/** One easy sentence about what this page is, for people who never saw a run id. */
export function runSentence(runId: string, latest: boolean): string {
  const pack = packLabel(runId);
  const what = pack ? `the small model on ${pack} questions` : "the small model";
  return latest ? `The newest recorded test of ${what}.` : `One test of ${what}.`;
}
