import { Fragment, useId, useMemo, useRef, useState } from "react";
import { fmtInt } from "../../api/format";
import type { Example, ExamplesResult } from "../../api/types";
import { Badge } from "../../components";
import { diffAgainstGold, type SideBySide } from "../../lib/diff";
import { ChoiceGroup, Tabs, TabsContent, TabsList, TabsTrigger } from "../../ui";
import { SqlBlock } from "./SqlBlock";
import { useWide } from "./useWide";

type Kind = Example["kind"];
type Model = "base" | "student" | "teacher";

const KINDS: readonly { kind: Kind; label: string; meaning: string }[] = [
  { kind: "fixed", label: "Fixed", meaning: "The student is right and the base model is wrong." },
  { kind: "still_wrong", label: "Still wrong", meaning: "Both the base model and the student are wrong." },
  { kind: "regressed", label: "Regressed", meaning: "The base model is right and the student is wrong." },
];
const MODELS: readonly { key: Model; label: string }[] = [
  { key: "base", label: "Base" },
  { key: "student", label: "Student" },
  { key: "teacher", label: "Teacher" },
];
const SQL_OF: Record<Model, (e: Example) => string> = { base: (e) => e.base_sql, student: (e) => e.student_sql, teacher: (e) => e.teacher_sql };
const OK_OF: Record<Model, (e: Example) => boolean> = { base: (e) => e.base_ok, student: (e) => e.student_ok, teacher: (e) => e.teacher_ok };

function diffNote(d: SideBySide | null): string {
  return d === null ? "diff skipped: SQL too long" : d.identical ? "same tokens as gold" : "tokens not in gold are highlighted";
}

/** One example, side by side. Correctness flags come from the API; the highlighting is a plain token diff of the returned SQL strings. */
function ExampleDetail({ e, id }: { e: Example; id: string }) {
  const [against, setAgainst] = useState<Model>("student");
  const group = useId();
  const diffs = useMemo(
    () => ({
      base: diffAgainstGold(e.gold_sql, e.base_sql),
      student: diffAgainstGold(e.gold_sql, e.student_sql),
      teacher: diffAgainstGold(e.gold_sql, e.teacher_sql),
    }),
    [e.gold_sql, e.base_sql, e.student_sql, e.teacher_sql],
  );
  return (
    <div className="exd" id={id} role="region" aria-label={`Side-by-side SQL for ${e.question}`}>
      <h4 className="exd-q">{e.question}</h4>
      <p className="mono muted exd-meta">{e.task_id} · {e.family} · {e.heldout_class}</p>
      <ChoiceGroup
        legend="Highlight gold tokens missing from"
        name={`against-${group}`}
        value={against}
        onChange={setAgainst}
        options={MODELS.map((m) => ({ value: m.key, label: m.label }))}
      />
      <div className="cmp">
        <div className="exd-sql" data-model="gold">
          <SqlBlock
            label="Gold"
            code={e.gold_sql}
            segments={diffs[against]?.gold}
            side="gold"
            note={diffs[against] === null ? "diff skipped: SQL too long" : `tokens missing from ${against} SQL are highlighted`}
          />
        </div>
        {MODELS.map((m) => (
          <div className="exd-sql" data-model={m.key} key={m.key}>
            <SqlBlock
              label={m.label}
              status={<Badge tone={OK_OF[m.key](e) ? "ok" : "bad"}>{OK_OF[m.key](e) ? "correct" : "wrong"}</Badge>}
              code={SQL_OF[m.key](e)}
              segments={diffs[m.key]?.other}
              side="other"
              note={diffNote(diffs[m.key])}
            />
          </div>
        ))}
      </div>
    </div>
  );
}

/** Which of the three models got this example right, from the payload's own flags: a glyph, the name and, for screen readers, the word. */
function Marks({ e }: { e: Example }) {
  return (
    <span className="exb-marks">
      {MODELS.map((m) => {
        const ok = OK_OF[m.key](e);
        return (
          <span key={m.key} className={`mk ${ok ? "ok" : "bad"}`}>
            <span aria-hidden="true" className="mk-glyph">{ok ? "\u2713" : "\u2717"}</span>
            <span className="mk-name">{m.label}</span>
            <span className="sr-only">{ok ? " correct" : " wrong"}</span>
          </span>
        );
      })}
    </span>
  );
}

/**
 * One example's row. Question (clamped to a few lines here, in full in the detail), where it came from, and who got it right.
 */
function Row(props: { e: Example; open: boolean; panel: string; onPick: () => void; refCb: (el: HTMLButtonElement | null) => void }) {
  const { e, open, panel, onPick, refCb } = props;
  return (
    <button ref={refCb} type="button" className="exb-item" aria-expanded={open} aria-controls={open ? panel : undefined} onClick={onPick}>
      <strong>{e.question}</strong>
      <span className="mono muted exb-meta">{e.task_id} · {e.family} · {e.heldout_class}</span>
      <Marks e={e} />
    </button>
  );
}

/**
 * The examples of one kind. Narrow windows stack them: the open example opens directly under its own row.
 * Wide windows put the rows in a bounded, scrolling list and the open example beside it at its natural height,
 * so it is never clipped or scrolled inside itself. Exactly one example is open at a time; the first opens by itself.
 */
function KindList({ items, total, meaning }: { items: Example[]; total: number | null | undefined; meaning: string }) {
  const uid = useId();
  const wide = useWide();
  const [picked, setPicked] = useState<string | null | undefined>(undefined); // undefined: the first one is open
  const rows = useRef<Map<string, HTMLElement>>(new Map());
  const open = picked === undefined ? items[0]?.task_id ?? null : picked;
  const openItem = items.find((e) => e.task_id === open) ?? null;
  // "showing N of M" only when the server gave M (the whole held-out set); otherwise it is just a capped list.
  const count = total === null || total === undefined ? `capped list, ${items.length} shown` : `showing ${items.length} of ${fmtInt(total)}`;
  const pick = (id: string, isOpen: boolean) => {
    setPicked(isOpen ? null : id);
    // Stacked: the example that was open above collapses and moves the row being opened; keep that row in view.
    if (!wide && !isOpen) requestAnimationFrame(() => rows.current.get(id)?.scrollIntoView?.({ block: "nearest" }));
  };
  const row = (e: Example, i: number) => (
    <Row
      e={e} open={e.task_id === open} panel={`${uid}-${i}`} onPick={() => pick(e.task_id, e.task_id === open)}
      refCb={(el) => { if (el) rows.current.set(e.task_id, el); else rows.current.delete(e.task_id); }}
    />
  );
  return (
    <div className="exb-kind">
      <p className="exb-count"><span>{count}</span> <span className="muted">{meaning}</span></p>
      {items.length === 0 ? (
        <p className="muted">None returned for this kind.</p>
      ) : wide ? (
        <div className="exb-split">
          <ul className="exb-list" role="list" aria-label="Examples">
            {items.map((e, i) => <li key={e.task_id}>{row(e, i)}</li>)}
          </ul>
          {openItem ? (
            <ExampleDetail e={openItem} id={`${uid}-${items.indexOf(openItem)}`} />
          ) : (
            <p className="exb-pick muted">Select an example to compare its SQL.</p>
          )}
        </div>
      ) : (
        <div className="exb-stack">
          {items.map((e, i) => (
            <Fragment key={e.task_id}>
              {row(e, i)}
              {e.task_id === open && <ExampleDetail e={e} id={`${uid}-${i}`} />}
            </Fragment>
          ))}
          {open === null && <p className="exb-pick muted">Select an example to compare its SQL.</p>}
        </div>
      )}
    </div>
  );
}

export default function ExampleBrowser({ result }: { result: ExamplesResult }) {
  const byKind = useMemo(() => {
    const m = new Map<Kind, Example[]>(KINDS.map((k) => [k.kind, []]));
    for (const e of result.items) m.get(e.kind)?.push(e);
    return m;
  }, [result.items]);
  const first = KINDS.find((k) => (byKind.get(k.kind)?.length ?? 0) > 0)?.kind ?? "fixed";
  const [kind, setKind] = useState<Kind>(first);
  return (
    <Tabs value={kind} onValueChange={(v) => setKind(v as Kind)} className="exb">
      <TabsList aria-label="Example kind" className="exb-tabs">
        {KINDS.map((k) => <TabsTrigger key={k.kind} value={k.kind}>{k.label}</TabsTrigger>)}
      </TabsList>
      {KINDS.map((k) => (
        <TabsContent key={k.kind} value={k.kind}>
          <KindList items={byKind.get(k.kind) ?? []} total={result.totals?.[k.kind]} meaning={k.meaning} />
        </TabsContent>
      ))}
    </Tabs>
  );
}
