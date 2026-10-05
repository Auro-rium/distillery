import AxeBuilder from "@axe-core/playwright";
import fs from "node:fs";
import path from "node:path";
import { known } from "./known";
import { expect, open, test } from "./fixtures";
import { SHOTS } from "./global-setup";
import { ROUTES } from "./routes";

/**
 * Run one named check. Failures are collected (soft) so one page load reports every broken check. A check listed in
 * known.ts (a documented pre-existing bug) must still fail; if it starts passing the test fails and asks for removal.
 */
async function check(route: string, name: string, fn: () => Promise<void>) {
  const k = known(route, name, test.info().project.name);
  try {
    await fn();
  } catch (e) {
    if (k) return void test.info().annotations.push({ type: "known-finding", description: `${k.id} ${name}: ${k.note}` });
    expect.soft(false, `${name}: ${(e as Error).message}`).toBe(true);
    return;
  }
  if (k) expect.soft(false, `${name} now passes: remove ${k.id} from e2e/known.ts and FINDINGS.md`).toBe(true);
}

for (const route of ROUTES) {
  test.describe(route.name, () => {
    test("page checks: clean console, no overflow, landmarks, label, focus ring; screenshot", async ({ page, watch }, info) => {
      await open(page, route.path);
      const { width, scheme } = info.project.metadata as { width: number; scheme: string };
      expect(await page.evaluate(() => document.documentElement.dataset.theme), "applied theme matches the project scheme").toBe(scheme);
      fs.mkdirSync(SHOTS, { recursive: true });
      await page.screenshot({ path: path.join(SHOTS, `${route.name}-${width}-${scheme}.png`), fullPage: true, animations: "disabled" });

      await check(route.name, "console", async () => {
        expect(watch.problems).toEqual([]);
      });

      await check(route.name, "overflow", async () => {
        const m = await page.evaluate(() => ({ sw: document.documentElement.scrollWidth, iw: window.innerWidth, bw: document.body.scrollWidth }));
        expect(m.sw, `documentElement.scrollWidth ${m.sw} > innerWidth ${m.iw}`).toBeLessThanOrEqual(m.iw);
        expect(m.bw, `body.scrollWidth ${m.bw} > innerWidth ${m.iw}`).toBeLessThanOrEqual(m.iw);
      });

      await check(route.name, "landmarks", async () => {
        await expect(page.getByRole("main")).toHaveCount(1, { timeout: 1000 });
        await expect(page.getByRole("heading", { level: 1 })).toHaveCount(1, { timeout: 1000 });
        await expect(page.getByRole("heading", { level: 1 })).toBeVisible({ timeout: 1000 });
      });

      if (route.banner) {
        await check(route.name, "banner", async () => {
          await expect(page.getByRole("status").filter({ hasText: /DRY RUN/ }).first()).toBeVisible({ timeout: 2000 });
        });
      }

      await check(route.name, "focus", async () => {
        await page.evaluate(() => (document.activeElement as HTMLElement | null)?.blur());
        await page.keyboard.press("Tab");
        const f = await page.evaluate(() => {
          const el = document.activeElement as HTMLElement | null;
          if (!el || el === document.body) return null;
          const cs = getComputedStyle(el);
          const r = el.getBoundingClientRect();
          return {
            tag: el.tagName, text: (el.textContent ?? "").trim().slice(0, 40),
            w: r.width, h: r.height, inView: r.bottom > 0 && r.right > 0 && r.top < innerHeight && r.left < innerWidth,
            outline: cs.outlineStyle !== "none" && parseFloat(cs.outlineWidth) > 0,
            shadow: cs.boxShadow !== "none",
          };
        });
        expect(f, "Tab focused nothing").not.toBeNull();
        expect(f!.w * f!.h, `focused ${f!.tag} "${f!.text}" has no size`).toBeGreaterThan(0);
        expect(f!.inView, `focused ${f!.tag} "${f!.text}" is off screen`).toBe(true);
        expect(f!.outline || f!.shadow, `focused ${f!.tag} "${f!.text}" has no outline or shadow`).toBe(true);
      });
    });

    test("axe: no serious or critical violations", async ({ page }) => {
      await open(page, route.path);
      await check(route.name, "axe", async () => {
        const res = await new AxeBuilder({ page })
          .withTags(["wcag2a", "wcag2aa", "wcag21a", "wcag21aa", "wcag22aa", "best-practice"])
          .analyze();
        const bad = res.violations.filter((v) => v.impact === "serious" || v.impact === "critical");
        const summary = bad.map((v) => `${v.id} (${v.impact}): ${v.nodes.map((n) => n.target.join(" ")).slice(0, 6).join(" | ")}`);
        expect(summary).toEqual([]);
      });
    });
  });
}
