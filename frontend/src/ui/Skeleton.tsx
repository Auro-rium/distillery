import { cx } from "./cx";
import { SKELETON_LINES } from "./constants";

/** One decorative placeholder bar. It carries no data-shaped content and is hidden from assistive tech. */
export function Skeleton({ className }: { className?: string }) {
  return <i className={cx("sk-line", className)} aria-hidden="true" />;
}

/**
 * Loading state. By default a shimmering skeleton block with the label as text; `inline` is the small dot
 * pulse used next to a button. Distinct from the error and empty states, which have their own components.
 */
export function Spinner(props: { label?: string; inline?: boolean; lines?: number }) {
  const label = <span className={props.label ? "muted skeleton-label" : "sr-only"}>{props.label ?? "Loading"}</span>;
  if (props.inline) {
    return (
      <span role="status" aria-live="polite" className="dots-wrap">
        <span className="dots" aria-hidden="true"><i /><i /><i /></span>
        {label}
      </span>
    );
  }
  return (
    <div role="status" aria-live="polite" aria-busy="true" className="skeleton-wrap">
      {label}
      <div className="skeleton" aria-hidden="true">
        {Array.from({ length: props.lines ?? SKELETON_LINES }, (_, i) => <Skeleton key={i} className={`sk-${i % SKELETON_LINES}`} />)}
      </div>
    </div>
  );
}
