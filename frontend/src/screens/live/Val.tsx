import type { ReactNode } from "react";
import { NOT_MEASURED } from "../../api/format";

/**
 * A formatted value (the string comes from api/format.ts). Absent data renders as the plain words
 * "not measured", styled quieter than a real figure so it cannot be read as one.
 */
export function Val({ text }: { text: string }) {
  return <span className={text === NOT_MEASURED ? "absent" : undefined}>{text}</span>;
}

/** One labelled figure in a `<dl className="metrics">`. */
export function Metric({ label, value, hint }: { label: string; value: ReactNode; hint?: ReactNode }) {
  return (
    <div className="metric">
      <dt>{label}</dt>
      <dd>{value}{hint ? <span className="hint">{hint}</span> : null}</dd>
    </div>
  );
}
