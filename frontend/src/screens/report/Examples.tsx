import { Suspense, lazy } from "react";
import { api } from "../../api/client";
import { ApiErrorState, Card, EmptyState, Spinner } from "../../components";
import { useAsync } from "./useAsync";

// The browser (tabs, list, diff, SQL blocks) is only needed once examples exist, so it is its own chunk.
const ExampleBrowser = lazy(() => import("./ExampleBrowser"));

/** The API keeps at most this many per kind; asking for the maximum returns every stored one. */
const LIMIT = 200;

export function ExamplesCard({ id, language }: { id: string; language?: string }) {
  const [res, retry] = useAsync(() => api.examples(id, "all", LIMIT), id);
  return (
    <Card title="Examples: fixed, still wrong, regressed" className="rp-ex">
      {res.state === "loading" && <Spinner label="Loading examples" />}
      {res.state === "error" && <ApiErrorState title="Examples unavailable" error={res.error} onRetry={retry} />}
      {res.state === "ok" && res.data.items.length === 0 && (
        res.data.available === false ? (
          <EmptyState title="This report has no examples">The report was written without a held-out examples field.</EmptyState>
        ) : (
          <EmptyState title="No examples returned for this run" />
        )
      )}
      {res.state === "ok" && res.data.items.length > 0 && (
        <Suspense fallback={<Spinner label="Loading examples" />}>
          <ExampleBrowser result={res.data} language={language} />
        </Suspense>
      )}
    </Card>
  );
}
