import { useEffect, useId, useRef, useState, type ReactNode } from "react";

export interface DataRow {
  key: string;
  label: string;
  value: ReactNode;
  /** A short plain-words definition. A row that has one opens it beside the row when clicked. */
  help?: string;
  /** What the value is exactly, for people who hover it. */
  valueTitle?: string;
}

/** Width reserved for a definition beside its row (see `.m-data-pop`); when it does not fit on the right it goes below. */
const POP_WIDTH = 250;
const POP_GAP = 12;
const EDGE = 16;

/**
 * A green "DATA" box that opens downward into a list of everything recorded for the fine-tune. Rows that have a
 * definition (train loss, validation loss) open it in a small note to their right when clicked, or just under the
 * row when there is no room on the right. Esc closes the note first, then the list; so does a click elsewhere.
 */
export function DataMenu({ rows, label = "Fine-tune record" }: { rows: DataRow[]; label?: string }) {
  const [open, setOpen] = useState(false);
  const [help, setHelp] = useState<{ key: string; side: "right" | "below" } | null>(null);
  const root = useRef<HTMLDivElement>(null);
  const panel = useId();

  useEffect(() => {
    if (!open) return;
    const away = (e: PointerEvent) => {
      if (root.current && !root.current.contains(e.target as Node)) { setOpen(false); setHelp(null); }
    };
    const esc = (e: KeyboardEvent) => {
      if (e.key !== "Escape") return;
      if (help) setHelp(null);
      else setOpen(false);
    };
    document.addEventListener("pointerdown", away);
    document.addEventListener("keydown", esc);
    return () => { document.removeEventListener("pointerdown", away); document.removeEventListener("keydown", esc); };
  }, [open, help]);

  function toggleHelp(key: string, el: HTMLElement) {
    if (help?.key === key) { setHelp(null); return; }
    const r = el.getBoundingClientRect();
    const fits = r.right + POP_GAP + POP_WIDTH + EDGE <= window.innerWidth;
    setHelp({ key, side: fits ? "right" : "below" });
  }

  return (
    <div className="m-data" ref={root}>
      <button
        type="button"
        className="badge lg ok m-data-btn"
        aria-expanded={open}
        aria-controls={panel}
        onClick={() => { setOpen((o) => !o); setHelp(null); }}
      >
        <span>Data</span>
        <span className="m-caret" aria-hidden="true">▾</span>
      </button>
      <ul id={panel} className="m-data-list" role="list" aria-label={label} hidden={!open}>
        {rows.map((row) => {
          const showing = help?.key === row.key;
          const body = (
            <>
              <span className="eyebrow m-data-label">{row.label}</span>{" "}
              <span className="m-data-value" title={row.valueTitle}>{row.value}</span>
            </>
          );
          return (
            <li key={row.key} className="m-data-item">
              {row.help ? (
                <button
                  type="button"
                  className="m-data-row has-help"
                  aria-expanded={showing}
                  aria-describedby={showing ? `${panel}-${row.key}` : undefined}
                  onClick={(e) => toggleHelp(row.key, e.currentTarget)}
                >
                  {body}
                  <span className="m-data-i" aria-hidden="true">i</span>
                </button>
              ) : (
                <div className="m-data-row">{body}</div>
              )}
              {showing && row.help && (
                <div id={`${panel}-${row.key}`} role="note" className={`m-data-pop ${help!.side}`}>{row.help}</div>
              )}
            </li>
          );
        })}
      </ul>
    </div>
  );
}
