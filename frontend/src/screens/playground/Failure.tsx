import { Link } from "react-router-dom";
import { ApiError } from "../../api/client";
import { ApiErrorState } from "../../components";

/**
 * A failed request, told apart by what the API said: the daily demo budget being used up, this client
 * being rate limited, or any other failure (shown with its code and message). None of them is an empty result.
 */
export function Failure({ e }: { e: unknown }) {
  if (e instanceof ApiError && e.code === "demo_budget_exhausted") {
    return (
      <div className="state" role="alert">
        <h3>Demo budget exhausted</h3>
        <p><span className="mono">{e.code}</span>: {e.message}</p>
        <p><Link to="/">See replay</Link> for stored runs, which cost nothing to open.</p>
      </div>
    );
  }
  if (e instanceof ApiError && e.status === 429) {
    return (
      <div className="state" role="alert">
        <h3>Rate limit reached</h3>
        <p><span className="mono">{e.code}</span>: {e.message}</p>
        <p>{e.retryAfter ? `Try again in ${e.retryAfter} seconds.` : "Try again later."}</p>
      </div>
    );
  }
  return <ApiErrorState title="The request failed" error={e} />;
}
