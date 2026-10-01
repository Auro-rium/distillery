// Errors are not empty states (README rule 4): with the API unreachable or failing, every data screen must show
// an error, never the empty-state copy and never numbers.
import { expect, open, test } from "./fixtures";

// Only the backend: "/api/..." at the origin root ("/src/api/client.ts" is the app's own source module).
const API = /^https?:\/\/[^/]+\/api\//;

const DATA_ROUTES = [
  { name: "replay", path: "/" },
  { name: "live", path: "/runs/dry-sql-tiny" },
  { name: "report", path: "/runs/dry-sql-tiny/report" },
  { name: "tree", path: "/runs/dry-sql-tiny/tree" },
];

async function assertErrorNotEmptyNorNumbers(page: import("@playwright/test").Page) {
  const alert = page.getByRole("alert").first();
  await expect(alert).toBeVisible();
  // No empty-state block (".state" without ".error") and no chart/number output next to the error.
  await expect(page.locator("main .state:not(.error)")).toHaveCount(0);
  const text = await page.locator("main").innerText();
  expect(text, "a percentage or dollar amount is shown while the API is down").not.toMatch(/\d\s?%|\$\s?\d/);
  expect(text).not.toMatch(/No replay bundles|Nothing has been exported/i);
  await expect(page.locator("main figure.acc, main table")).toHaveCount(0);
}

for (const r of DATA_ROUTES) {
  test(`API unreachable: ${r.name} shows an error state`, async ({ page }) => {
    await page.route(API, (route) => route.abort("connectionrefused"));
    await open(page, r.path);
    await assertErrorNotEmptyNorNumbers(page);
    await expect(page.getByRole("alert").first()).toContainText(/cannot reach|network|unreachable|failed/i);
  });
}

test("API answers 500 with a JSON error: the code and message are shown, not an empty state", async ({ page }) => {
  await page.route(API, (route) =>
    route.fulfill({ status: 500, contentType: "application/json", body: JSON.stringify({ error: "boom_code", message: "boom message from the server" }) }),
  );
  await open(page, "/");
  await assertErrorNotEmptyNorNumbers(page);
  await expect(page.getByRole("alert").first()).toContainText("boom_code");
  await expect(page.getByRole("alert").first()).toContainText("boom message from the server");
});

test("a static host answering /api with HTML is an error, not data", async ({ page }) => {
  await page.route(API, (route) => route.fulfill({ status: 200, contentType: "text/html", body: "<!doctype html><title>x</title>" }));
  await open(page, "/");
  await assertErrorNotEmptyNorNumbers(page);
});
