import { test as base, expect, type Page } from "@playwright/test";

export { expect };

export interface Watch {
  problems: string[];
  /** Same-origin requests seen (path only). */
  requests: string[];
}

const sameOrigin = (url: string, base: string) => {
  try { return new URL(url).origin === new URL(base).origin; } catch { return false; }
};

/**
 * `watch` records everything that should be absent on a healthy page: console errors, uncaught exceptions,
 * failed and 4xx/5xx same-origin requests. Cross-origin failures (Google Fonts when offline) are not the app's
 * fault and are ignored; ERR_ABORTED (an SSE stream closed by navigation or teardown) is a cancel, not a failure.
 */
export const test = base.extend<{ watch: Watch }>({
  watch: async ({ page, baseURL }, use) => {
    const w: Watch = { problems: [], requests: [] };
    const origin = baseURL!;
    page.on("console", (m) => {
      if (m.type() !== "error") return;
      const u = m.location().url;
      if (u && !sameOrigin(u, origin)) return;
      w.problems.push(`console.error: ${m.text()}`);
    });
    page.on("pageerror", (e) => w.problems.push(`pageerror: ${e.message}`));
    page.on("requestfailed", (r) => {
      if (!sameOrigin(r.url(), origin)) return;
      const t = r.failure()?.errorText ?? "";
      if (t.includes("ERR_ABORTED")) return;
      w.problems.push(`request failed: ${r.method()} ${new URL(r.url()).pathname} (${t})`);
    });
    page.on("response", (r) => {
      if (!sameOrigin(r.url(), origin)) return;
      w.requests.push(new URL(r.url()).pathname);
      if (r.status() >= 400) w.problems.push(`HTTP ${r.status()}: ${r.request().method()} ${new URL(r.url()).pathname}`);
    });
    await use(w);
  },
});

/** Open a route and wait until the app has rendered and no loading skeleton is left. */
export async function open(page: Page, path: string) {
  await page.goto(path, { waitUntil: "load" });
  await page.locator("main").waitFor();
  await page.waitForFunction(() => !document.querySelector('main [aria-busy="true"]'), undefined, { timeout: 15_000 });
  // Web fonts (production loads them from Google) change text widths: measure after they settled, bounded.
  await page.evaluate(() => Promise.race([document.fonts.ready, new Promise((r) => setTimeout(r, 4000))]));
  await page.waitForTimeout(400);
}
