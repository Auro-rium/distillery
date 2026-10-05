// Pre-existing UI bugs the harness found, documented in e2e/FINDINGS.md. A test listed here is marked
// test.fail(): it must keep failing until the bug is fixed, and Playwright then reports "expected to fail but
// passed", which forces this entry (and the finding) to be removed. Nothing is silently skipped.
export interface Known { id: string; route: string; check: string; note: string; project?: RegExp }

export const KNOWN: Known[] = [
  { id: "F1", route: "not-found", check: "landmarks",
    note: "the not-found page has no h1 (EmptyState renders an h3 'Page not found' in App.tsx)" },
];

export function known(route: string, check: string, project: string): Known | undefined {
  return KNOWN.find((k) => k.route === route && k.check === check && (!k.project || k.project.test(project)));
}
