// Before the suite: clear old screenshots and warm the Vite dev server (dependency pre-bundling triggers a full
// page reload the first time a lazy chunk such as the dialog is discovered, which would flake the first tests).
import fs from "node:fs";
import path from "node:path";
import { chromium, type FullConfig } from "@playwright/test";
import { ROUTES } from "./routes";

export const SHOTS = path.resolve(process.cwd(), "e2e-artifacts/screenshots");

export default async function globalSetup(config: FullConfig) {
  fs.rmSync(SHOTS, { recursive: true, force: true });
  fs.mkdirSync(SHOTS, { recursive: true });
  const base = config.projects[0].use.baseURL!;
  const browser = await chromium.launch({ channel: "chrome" });
  try {
    const page = await browser.newPage({ viewport: { width: 400, height: 800 } });
    for (const r of ROUTES) {
      await page.goto(base + r.path, { waitUntil: "load" });
      await page.locator("main").waitFor();
      await page.waitForTimeout(1200);
    }
    // Lazy dialogs: phone menu and the shortcut list.
    await page.goto(base + "/", { waitUntil: "load" });
    await page.getByRole("button", { name: "Menu" }).click();
    await page.waitForTimeout(800);
    await page.keyboard.press("Escape");
    await page.keyboard.press("?");
    await page.waitForTimeout(800);
    await page.keyboard.press("Escape");
    await page.waitForTimeout(500);
  } finally {
    await browser.close();
  }
}
