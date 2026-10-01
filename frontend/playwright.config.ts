// Browser safety net (spec D6). Runs the Vite dev server with the dev-only mock API (VITE_MOCK=1) and drives the
// installed Google Chrome (channel "chrome"), so no browser download is needed.
// Spec files are named *.e2e.ts so that vitest (npm test) does not pick them up.
import { defineConfig } from "@playwright/test";

export const PORT = Number(process.env.E2E_PORT ?? 5171);
const BASE = `http://127.0.0.1:${PORT}`;

const SIZES = [
  { width: 400, height: 800 },
  { width: 768, height: 1024 },
  { width: 1440, height: 900 },
];
const SCHEMES = ["light", "dark"] as const;

export default defineConfig({
  testDir: "./e2e",
  testMatch: "**/*.e2e.ts",
  outputDir: "./e2e-artifacts/test-results",
  globalSetup: "./e2e/global-setup.ts",
  fullyParallel: true,
  workers: process.env.CI ? 2 : 4,
  retries: 0,
  timeout: 45_000,
  expect: { timeout: 8_000 },
  forbidOnly: !!process.env.CI,
  reporter: process.env.CI
    ? [["list"], ["github"], ["html", { open: "never", outputFolder: "e2e-artifacts/report" }]]
    : [["list"], ["html", { open: "never", outputFolder: "e2e-artifacts/report" }]],
  use: { baseURL: BASE, channel: "chrome", trace: "retain-on-failure" },
  webServer: {
    command: `npx vite --host 127.0.0.1 --port ${PORT} --strictPort`,
    // /api/health only answers 200 from the mock plugin; a plain dev server would proxy to a dead :8000.
    url: `${BASE}/api/health`,
    env: { VITE_MOCK: "1" },
    reuseExistingServer: process.env.E2E_REUSE_SERVER === "1",
    timeout: 60_000,
  },
  projects: [
    // Matrix: every route x 3 widths x 2 colour schemes (structure, axe, screenshots).
    ...SCHEMES.flatMap((scheme) =>
      SIZES.map((s) => ({
        name: `w${s.width}-${scheme}`,
        testMatch: "**/routes.e2e.ts",
        metadata: { width: s.width, scheme },
        use: { viewport: s, colorScheme: scheme, reducedMotion: "reduce" as const, deviceScaleFactor: 1 },
      })),
    ),
    // Behaviour that does not depend on the viewport (API errors, truthfulness): run once.
    {
      name: "behaviour",
      testIgnore: "**/routes.e2e.ts",
      use: { viewport: SIZES[2], colorScheme: "light" as const, reducedMotion: "reduce" as const },
    },
  ],
});
