// Truthfulness (README rules 1-5): the UI shows what the API sent. The API is intercepted with a known payload whose
// numbers appear nowhere in the fixtures, and the numbers on screen are compared with the payload.
import fs from "node:fs";
import path from "node:path";
import { expect, open, test } from "./fixtures";

const API = /^https?:\/\/[^/]+\/api\//;
const FIXTURE = JSON.parse(fs.readFileSync(path.resolve(process.cwd(), "../docs/fixtures/sample-dry-run-report.json"), "utf8"));

const K = { base: 0.137, student: 0.642, teacher: 0.981, ciLo: 0.411, ciHi: 0.873, n: 37, p: 0.0123, ratioLo: 0.777, ratioPt: 0.654 };
const pct = (v: number) => `${(v * 100).toFixed(1)}%`;

function payload(id: string) {
  const r = JSON.parse(JSON.stringify(FIXTURE));
  r.run_id = id; r.dry_run = true; r.recorded = false; r.recorded_at = null;
  const ev = r.evaluation;
  ev.n = K.n;
  ev.accuracy = { base: K.base, student: K.student, teacher: K.teacher };
  Object.assign(ev.gate, {
    n: K.n, base_acc: K.base, student_acc: K.student, teacher_acc: K.teacher, student_ci: [K.ciLo, K.ciHi],
    mcnemar_p: K.p, ratio_lo: K.ratioLo, ratio_point: K.ratioPt,
  });
  return r;
}

/** Every number in the payload, as the display helpers would format it. */
function allowed(value: unknown, out = new Set<string>()): Set<string> {
  if (typeof value === "number" && Number.isFinite(value)) {
    out.add(pct(value)); out.add(`${(value * 100).toFixed(0)}%`);
    for (const d of [0, 1, 2, 3, 4, 5, 6]) out.add(value.toFixed(d));
  } else if (Array.isArray(value)) value.forEach((v) => allowed(v, out));
  else if (value && typeof value === "object") Object.values(value).forEach((v) => allowed(v, out));
  return out;
}

test("report: accuracy numbers on screen equal the intercepted payload", async ({ page }) => {
  const body = payload("truth-run");
  await page.route(API, (route) => {
    const u = new URL(route.request().url());
    if (u.pathname === "/api/runs/truth-run/report") return route.fulfill({ json: body });
    return route.fallback();
  });
  await open(page, "/runs/truth-run/report");
  const main = page.locator("main");
  for (const v of [K.base, K.student, K.teacher, K.ciLo, K.ciHi]) await expect(main).toContainText(pct(v));
  await expect(main).toContainText(`n=${K.n}`);
  await expect(main).toContainText(K.p.toFixed(4));
  await expect(main).toContainText(K.ratioLo.toFixed(3));

  // The fixture's own headline values must be gone (they were replaced in the payload).
  const text = await main.innerText();
  for (const stale of ["30.0%", "70.0%", "n=20"]) expect(text, `stale fixture value ${stale}`).not.toContain(stale);

  // Provenance: every percentage or decimal shown is derivable from the payload (axis ticks 0/50/100 % are geometry).
  const ok = allowed(body);
  for (const t of ["0%", "50%", "100%"]) ok.add(t);
  const shown = text.match(/\d+(?:\.\d+)?%|\b\d+\.\d+\b/g) ?? [];
  const unknown = [...new Set(shown)].filter((s) => !ok.has(s));
  expect(unknown, "numbers on screen that are not in the payload").toEqual([]);
});

test("replay: labels come from each payload, and a missing flag is never shown as a real run", async ({ page }) => {
  const item = (run_id: string, extra: object) => ({ run_id, status: "complete", decision: "PROMOTE", created_at: null, ...extra });
  await page.route(API, (route) => {
    const u = new URL(route.request().url());
    if (u.pathname === "/api/replay")
      return route.fulfill({
        json: [
          item("truth-recorded", { dry_run: false, recorded: true, recorded_at: "2026-01-02T03:04:05Z" }),
          item("truth-unflagged", { recorded: false, recorded_at: null }),
        ],
      });
    return route.fallback();
  });
  await open(page, "/");
  const main = page.locator("main");
  await expect(main).toContainText("Recorded run");
  await expect(main).toContainText("2026-01-02T03:04:05Z");
  await expect(main).toContainText(/Label unknown/);
  await expect(main.getByText(/DRY RUN — fake models/)).toHaveCount(0);
});

test("mock mode: the served data is labelled dry_run, so the DRY RUN banner is on every data route", async ({ page }) => {
  await open(page, "/runs/dry-sql-tiny/report");
  const r = await page.evaluate(() => fetch("/api/runs/dry-sql-tiny/report").then((x) => x.json()));
  expect(r.dry_run).toBe(true);
  await expect(page.getByRole("status").filter({ hasText: /DRY RUN/ }).first()).toBeVisible();
});
