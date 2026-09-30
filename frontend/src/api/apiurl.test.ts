// Every request goes through apiUrl() so VITE_API_BASE (frontend hosted apart from the API) is honoured.
import fs from "node:fs";
import path from "node:path";
import { describe, expect, it } from "vitest";

const SRC = path.resolve(__dirname, "..");
const walk = (d: string): string[] => fs.readdirSync(d, { withFileTypes: true }).flatMap((e) => (e.isDirectory() ? walk(path.join(d, e.name)) : [path.join(d, e.name)]));
const shipped = walk(SRC).filter((f) => /\.tsx?$/.test(f) && !/\.test\.tsx?$/.test(f) && !f.includes(`${path.sep}testutil${path.sep}`));
const stripComments = (s: string) => s.replace(/\/\*[\s\S]*?\*\//g, "").replace(/(^|[^:])\/\/.*$/gm, "$1");

/** Literals containing /api/ that are not the direct argument of apiUrl(). Relative import paths (../api/x) are exempt. */
export function bareApiLiterals(code: string): string[] {
  const out: string[] = [];
  for (const m of stripComments(code).matchAll(/(apiUrl\(\s*)?(["'`])((?:(?!\2).)*?(?<!\.)\/api\/(?:(?!\2).)*)\2/g)) if (!m[1]) out.push(m[0]);
  return out;
}

describe("API base URL", () => {
  it("the checker flags bare literals anywhere in a string and accepts apiUrl() and import paths", () => {
    expect(bareApiLiterals('fetch("/api/x")')).toHaveLength(1);
    expect(bareApiLiterals("fetch(`${base}/api/runs`)")).toHaveLength(1);
    expect(bareApiLiterals('new EventSource("https://h.example/api/e")')).toHaveLength(1);
    expect(bareApiLiterals("useSSE(apiUrl(`/api/runs/${id}/events`))")).toEqual([]);
    expect(bareApiLiterals('import x from "../../api/types"; import y from "./api/z";')).toEqual([]);
  });
  it("no string literal containing /api/ appears outside client.ts unless it is the argument of apiUrl()", () => {
    const hits: string[] = [];
    for (const f of shipped.filter((x) => !x.endsWith(`api${path.sep}client.ts`)))
      for (const h of bareApiLiterals(fs.readFileSync(f, "utf8"))) hits.push(`${path.relative(SRC, f)}: ${h}`);
    expect(hits).toEqual([]);
  });
  it("the live screen builds its EventSource URL with apiUrl()", () => {
    const src = fs.readFileSync(path.join(SRC, "screens/live/index.tsx"), "utf8");
    expect(src).toMatch(/useSSE\(\s*active \? apiUrl\(/);
  });
});
