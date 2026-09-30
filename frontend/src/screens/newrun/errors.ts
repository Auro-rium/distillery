import { ApiError } from "../../api/client";

/** Plain-language message for a failed POST /api/runs. */
export function describeCreateError(e: unknown): string {
  if (!(e instanceof ApiError)) return "Unexpected error while starting the run.";
  // Codes the server explains itself: show its exact message, not ours.
  const retry = e.retryAfter ? ` (retry in ${e.retryAfter} seconds)` : "";
  if (e.code === "admin_token_required" || e.code === "queue_full") return `${e.code}: ${e.message}${retry}`;
  switch (e.status) {
    case 0: return "Cannot reach the API server. Check that the backend is running.";
    case 401: return "Admin token missing. Enter the admin token to start a live run.";
    case 403: return "Admin token rejected. Check the token and try again.";
    case 409: return "Another live run is already active. Wait for it to finish or cancel it, then retry.";
    case 429: return e.retryAfter
      ? `Too many dry runs from your address. Retry in ${e.retryAfter} seconds.`
      : "Too many dry runs from your address. Retry later.";
    default: return `${e.message} (HTTP ${e.status})`;
  }
}
