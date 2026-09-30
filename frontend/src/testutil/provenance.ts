// Render-provenance checker: every number a screen shows must come from the payload it was given.
//
// visibleText()   text of the rendered DOM, text nodes joined by spaces
// unexplained()   numeric tokens on screen that no payload value explains
//
// Method: (1) collapse whitespace; (2) delete every payload string of length >= 3 that appears
// verbatim (ids, timestamps, SQL, labels: these are shown as-is, digits and all); (3) delete the
// caller's explicit allow-list entries; (4) tokenise what is left into numbers and require each to
// equal some payload number after the same rounding the UI applies (token decimals; x100 for a
// token followed by %). Integer tokens (no decimals) must match EXACTLY, so "3" is never
// explained by 2.7.

export interface Allow {
  /** Removed from the text before tokenising. */
  pattern: RegExp;
  /** Mandatory: why this number is legitimately not in the payload. */
  why: string;
}

const TOKEN = /(\d+(?:\.\d+)?)(%?)/g;

export function visibleText(root: Node): string {
  const parts: string[] = [];
  const walker = document.createTreeWalker(root, NodeFilter.SHOW_TEXT);
  for (let n = walker.nextNode(); n; n = walker.nextNode()) parts.push(n.nodeValue ?? "");
  return parts.join(" ").replace(/\s+/g, " ").trim();
}

function collect(v: unknown, nums: number[], strs: Set<string>): void {
  if (typeof v === "number") nums.push(v);
  else if (typeof v === "string") strs.add(v.replace(/\s+/g, " ").trim());
  else if (Array.isArray(v)) for (const x of v) collect(x, nums, strs);
  else if (v && typeof v === "object")
    for (const [k, x] of Object.entries(v)) {
      strs.add(k);
      collect(x, nums, strs);
    }
}

const decimals = (t: string): number => (t.includes(".") ? t.length - t.indexOf(".") - 1 : 0);

function explained(token: string, pct: boolean, nums: number[]): boolean {
  const d = decimals(token);
  const v = Number(token);
  return nums.some((n) => {
    const x = pct ? n * 100 : n;
    return d === 0 ? Math.abs(x - v) < 1e-9 : Number(x.toFixed(d)) === v;
  });
}

/** Numeric tokens in `text` that no payload explains. Empty array = every number has provenance. */
export function unexplained(text: string, payloads: unknown[], allow: Allow[] = []): string[] {
  const nums: number[] = [];
  const strs = new Set<string>();
  for (const p of payloads) collect(p, nums, strs);
  let rest = ` ${text.replace(/\s+/g, " ")} `;
  const known = [...strs].filter((s) => s.length >= 3).sort((a, b) => b.length - a.length);
  for (const s of known) rest = rest.split(s).join(" ");
  for (const a of allow) {
    if (!a.why) throw new Error("allow-list entries need a reason");
    rest = rest.replace(a.pattern, " ");
  }
  rest = rest.replace(/(\d),(?=\d{3})/g, "$1");
  const bad: string[] = [];
  for (const m of rest.matchAll(TOKEN)) if (!explained(m[1], m[2] === "%", nums)) bad.push(m[0]);
  return bad;
}
