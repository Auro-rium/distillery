import { LazyDialog } from "../ui/LazyDialog";

/**
 * Shows the evidence record for one claim on the page itself, as JSON, instead of sending the reader to another site.
 * `record` is exactly what the evidence endpoint returned for it (nothing added or computed). The repository file is
 * one optional click away in the footer. Closed when `title` is null.
 */
export function EvidenceViewer({ title, record, file, onClose }: {
  title: string | null;
  record: unknown;
  file: { path: string; url: string } | null;
  onClose: () => void;
}) {
  return (
    <LazyDialog
      open={title !== null}
      onOpenChange={(o) => { if (!o) onClose(); }}
      title={title ?? "Evidence"}
      description="The record the server returned for this claim."
      className="json-dialog"
    >
      {title !== null && (
        <>
          <pre className="json-view" tabIndex={0} aria-label="Evidence record (JSON)"><code>{JSON.stringify(record, null, 2)}</code></pre>
          {file && file.url && (
            <p className="json-foot muted">
              The file itself lives in the repository:{" "}
              <a href={file.url} target="_blank" rel="noreferrer noopener">{file.path}<span className="sr-only"> (opens in a new tab)</span></a>
            </p>
          )}
        </>
      )}
    </LazyDialog>
  );
}
