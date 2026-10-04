// The frontend must not contain invented content, and the production bundle must not ship the mock.
import { execFileSync } from "node:child_process";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { describe, expect, it } from "vitest";
import fixture from "../../docs/fixtures/sample-dry-run-report.json";

const SRC = path.resolve(__dirname);
const ROOT = path.resolve(__dirname, "..");

function walk(dir: string): string[] {
  return fs.readdirSync(dir, { withFileTypes: true }).flatMap((e) => {
    const p = path.join(dir, e.name);
    return e.isDirectory() ? walk(p) : [p];
  });
}
const isTest = (f: string) => /\.test\.[tj]sx?$/.test(f);
// Shipped source: everything in src/ except tests and src/testutil (test-only helpers and the
// generated contract fixtures, which are never imported by app code; checked below).
const shipped = walk(SRC).filter((f) => /\.(tsx?|css)$/.test(f) && !isTest(f) && !f.includes(`${path.sep}testutil${path.sep}`));
const read = (f: string) => fs.readFileSync(f, "utf8");
const rel = (f: string) => path.relative(SRC, f);

describe("no invented content in src/", () => {
  it("scans a non-trivial set of files", () => {
    expect(shipped.length).toBeGreaterThan(10);
  });
  it("has no Math.random, lorem, sample text or fake TODOs", () => {
    const bad = /Math\.random|lorem|\bsample (?:text|data|output)\b|TODO fake/i;
    const hits = shipped.filter((f) => bad.test(read(f))).map(rel);
    expect(hits).toEqual([]);
  });
  it("has no hardcoded percentages, dollar amounts or decimals in JSX text", () => {
    const hits: string[] = [];
    for (const f of shipped.filter((x) => x.endsWith(".tsx"))) {
      // JSX text nodes: text between > and < on the same statement, excluding {expressions}
      for (const m of read(f).matchAll(/>([^<>{}=]+)</g)) {
        if (/\d+(\.\d+)?\s?%|\$\s?\d|\d+\.\d+/.test(m[1])) hits.push(`${rel(f)}: ${m[1].trim()}`);
      }
    }
    expect(hits).toEqual([]);
  });
  it("has no numeric literal used as a displayed value (toFixed on payloads goes through format.ts)", () => {
    const hits = shipped.filter((f) => !f.endsWith(`api${path.sep}format.ts`) && /\.toFixed\(/.test(read(f))).map(rel);
    expect(hits).toEqual([]);
  });
});

describe("no mock or fixture imports in src/", () => {
  const importRe = /(?:from|import)\s*\(?\s*["']([^"']+)["']/g;
  it("never imports mock/ or docs/fixtures, and app code never imports testutil", () => {
    const hits: string[] = [];
    for (const f of walk(SRC).filter((x) => /\.tsx?$/.test(x) && !isTest(x))) {
      const inTestutil = f.includes(`${path.sep}testutil${path.sep}`);
      for (const m of read(f).matchAll(importRe)) {
        const spec = m[1];
        if (/(^|\/)mock(\/|$)/.test(spec) || spec.includes("docs/fixtures")) hits.push(`${rel(f)} -> ${spec}`);
        if (!inTestutil && /testutil/.test(spec)) hits.push(`${rel(f)} -> ${spec}`);
      }
    }
    expect(hits).toEqual([]);
  });
});

describe("production bundle", () => {
  it("built without VITE_MOCK contains no mock, synthetic data or fixture values", { timeout: 60_000 }, () => {
    const out = fs.mkdtempSync(path.join(os.tmpdir(), "distillery-build-"));
    const env = { ...process.env };
    delete env.VITE_MOCK;
    try {
      execFileSync(
        process.execPath,
        [path.join(ROOT, "node_modules/vite/bin/vite.js"), "build", "--outDir", out, "--emptyOutDir", "--logLevel", "error"],
        { cwd: ROOT, env: { ...env, NODE_ENV: "production" }, stdio: "pipe", timeout: 55_000 },
      );
      const files = walk(out).filter((f) => /\.(js|html|css)$/.test(f));
      const js = files.filter((f) => f.endsWith(".js")).map(read).join("\n");
      expect(js.length).toBeGreaterThan(10_000);
      const all = files.map(read).join("\n");
      const f = fixture as unknown as {
        run_id: string;
        data: { heldout_sealed_sha256: string };
        evaluation: { heldout_sha256: string; artifact: { adapter_sha256: string; checkpoint_id: string; job_id: string } };
        cost: { cost_per_1k_tasks: { teacher: { input_tokens: number; usd: number; usd_per_1k_tasks: number } } };
      };
      const t = f.cost.cost_per_1k_tasks.teacher;
      // strings/numbers that exist only in the fixture: its run id, hashes, ids and teacher cost figures
      const unique = [
        f.run_id, f.data.heldout_sealed_sha256, f.evaluation.heldout_sha256, f.evaluation.artifact.adapter_sha256,
        f.evaluation.artifact.checkpoint_id, f.evaluation.artifact.job_id,
        String(t.input_tokens), String(t.usd), String(t.usd_per_1k_tasks),
      ].filter((s) => s.length >= 5);
      expect(unique.length).toBeGreaterThan(4);
      for (const needle of ["VITE_MOCK", "synthetic", "docs/fixtures", "dry-sql-tiny", ...unique]) {
        expect(all.includes(needle), `bundle contains ${needle}`).toBe(false);
      }
    } finally {
      fs.rmSync(out, { recursive: true, force: true });
    }
  });
});
