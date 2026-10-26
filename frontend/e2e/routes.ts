// Every route of the app, with the mock-API ids that exist in frontend/mock/plugin.ts.
export interface RouteSpec {
  name: string;
  path: string;
  /** Page shows run/report data, which the mock labels dry_run: true, so the DRY RUN banner must be visible. */
  banner: boolean;
}

export const ROUTES: RouteSpec[] = [
  { name: "overview", path: "/", banner: false },
  { name: "replay", path: "/mission", banner: true },
  { name: "new-run", path: "/new", banner: false },
  { name: "live-running", path: "/runs/live-mock", banner: true },
  { name: "live-complete", path: "/runs/dry-sql-tiny", banner: true },
  { name: "report", path: "/runs/dry-sql-tiny/report", banner: true },
  { name: "tree", path: "/runs/dry-sql-tiny/tree", banner: true },
  { name: "not-found", path: "/no/such/page", banner: false },
];
