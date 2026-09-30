// Token diff of two strings the API returned. It only marks which whitespace-delimited tokens of
// each string are not part of the longest common token sequence; it adds no text of its own, and
// joining the segments of a side gives back that side's input exactly.
export interface Seg { text: string; changed: boolean }
export interface SideBySide { gold: Seg[]; other: Seg[]; identical: boolean }

/** Above this many table cells the diff is skipped rather than freezing the tab. */
const MAX_CELLS = 250_000;

interface Tok { text: string; ws: boolean }
const tokenize = (s: string): Tok[] => s.split(/(\s+)/).filter((t) => t !== "").map((text) => ({ text, ws: /^\s+$/.test(text) }));

/** Marks (per token index) which non-whitespace tokens fall outside the LCS. Null when too large. */
function lcsMarks(a: Tok[], b: Tok[]): { a: boolean[]; b: boolean[] } | null {
  const x = a.map((t, i) => [t, i] as const).filter(([t]) => !t.ws);
  const y = b.map((t, i) => [t, i] as const).filter(([t]) => !t.ws);
  const n = x.length, m = y.length;
  if ((n + 1) * (m + 1) > MAX_CELLS) return null;
  const w = m + 1;
  const dp = new Uint32Array((n + 1) * w);
  for (let i = n - 1; i >= 0; i--)
    for (let j = m - 1; j >= 0; j--)
      dp[i * w + j] = x[i][0].text === y[j][0].text ? dp[(i + 1) * w + j + 1] + 1 : Math.max(dp[(i + 1) * w + j], dp[i * w + j + 1]);
  const ca = new Array<boolean>(a.length).fill(false);
  const cb = new Array<boolean>(b.length).fill(false);
  for (const [, i] of x) ca[i] = true;
  for (const [, j] of y) cb[j] = true;
  let i = 0, j = 0;
  while (i < n && j < m) {
    if (x[i][0].text === y[j][0].text) { ca[x[i][1]] = false; cb[y[j][1]] = false; i++; j++; }
    else if (dp[(i + 1) * w + j] >= dp[i * w + j + 1]) i++;
    else j++;
  }
  return { a: ca, b: cb };
}

function segments(toks: Tok[], changed: boolean[]): Seg[] {
  // whitespace between two changed tokens belongs to the highlighted run
  const flags = toks.map((t, i) => {
    if (!t.ws) return changed[i];
    return i > 0 && i < toks.length - 1 && changed[i - 1] && changed[i + 1];
  });
  const out: Seg[] = [];
  toks.forEach((t, i) => {
    const last = out[out.length - 1];
    if (last && last.changed === flags[i]) last.text += t.text;
    else out.push({ text: t.text, changed: flags[i] });
  });
  return out;
}

/** Diff `other` against `gold`. Null when the strings are too large to diff cheaply. */
export function diffAgainstGold(gold: string, other: string): SideBySide | null {
  const a = tokenize(gold), b = tokenize(other);
  const marks = lcsMarks(a, b);
  if (!marks) return null;
  const g = segments(a, marks.a), o = segments(b, marks.b);
  return { gold: g, other: o, identical: !g.some((s) => s.changed) && !o.some((s) => s.changed) };
}
