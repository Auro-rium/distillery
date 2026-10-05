import type { ReactNode } from "react";
import { CopyButton } from "../../components";
import type { Seg } from "../../lib/diff";
import "./sql.css";

/**
 * One code string the API returned (SQL, JSON tool calls, ...), with its copy button in a header row above the code, so the button
 * can never sit on top of a long first line. `segments` (from lib/diff) highlight tokens; their text
 * joins back to `code`, so the block always shows exactly the string it was given.
 */
export function CodeBlock(props: {
  label: string;
  code: string;
  /** The pack's answer language ("sql", "json", ...): exposed as data-language and a language-* class for styling. */
  language?: string;
  /** Sits next to the label, e.g. a "correct" / "wrong" badge taken from the payload. */
  status?: ReactNode;
  segments?: Seg[];
  side?: "gold" | "other";
  /** Small print under the code, e.g. why a diff was or was not drawn. */
  note?: ReactNode;
}) {
  return (
    <div className="sql-block code-block" data-language={props.language ?? "text"}>
      <div className="sql-head">
        <span className="code-cap">{props.label}</span>
        {props.status}
        <span className="sql-copy"><CopyButton text={props.code} /></span>
      </div>
      <pre className="code">
        <code className={props.language ? `language-${props.language}` : undefined}>
          {props.segments
            ? props.segments.map((g, i) => (g.changed ? <mark key={i} className={`diff-${props.side ?? "other"}`}>{g.text}</mark> : g.text))
            : props.code}
        </code>
      </pre>
      {props.note ? <p className="diff-note muted">{props.note}</p> : null}
    </div>
  );
}
