import type { ReactNode } from "react";
import { cx } from "./cx";

/** A labelled row of related controls that wraps on narrow screens. */
export function Toolbar(props: { label: string; children: ReactNode; className?: string }) {
  return <div role="group" aria-label={props.label} className={cx("toolbar", props.className)}>{props.children}</div>;
}
