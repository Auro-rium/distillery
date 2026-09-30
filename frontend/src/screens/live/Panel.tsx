import { useId, type ReactNode } from "react";
import { cx } from "../../ui";

/** A titled region of the Live screen. The title names the region for assistive tech. */
export function Panel(props: { title: string; children: ReactNode; className?: string; aside?: ReactNode }) {
  const id = useId();
  return (
    <section className={cx("card", "lp", props.className)} aria-labelledby={id}>
      <header className="lp-head">
        <h2 id={id} className="lp-title">{props.title}</h2>
        {props.aside}
      </header>
      {props.children}
    </section>
  );
}
