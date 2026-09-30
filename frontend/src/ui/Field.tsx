import {
  createContext, forwardRef, useContext, useId,
  type InputHTMLAttributes, type LabelHTMLAttributes, type ReactNode, type SelectHTMLAttributes, type TextareaHTMLAttributes,
} from "react";
import { cx } from "./cx";

interface FieldCtx {
  controlId: string;
  describedBy: string | undefined;
  invalid: boolean;
}
const Ctx = createContext<FieldCtx | null>(null);

type A11y = { id?: string; "aria-describedby"?: string; "aria-invalid"?: boolean | "true" | "false" | "grammar" | "spelling" };

/**
 * The id / aria-describedby / aria-invalid a control must carry when it sits inside a Field. Explicit
 * props on the control win; a control outside any Field is left exactly as the caller wrote it.
 */
export function useFieldControl(props: A11y): A11y {
  const f = useContext(Ctx);
  if (!f) return {};
  const described = [props["aria-describedby"], f.describedBy].filter(Boolean).join(" ") || undefined;
  return {
    id: props.id ?? f.controlId,
    "aria-describedby": described,
    "aria-invalid": props["aria-invalid"] ?? (f.invalid ? true : undefined),
  };
}

export function Label({ className, htmlFor, ...rest }: LabelHTMLAttributes<HTMLLabelElement>) {
  const f = useContext(Ctx);
  return <label className={cx("field-label", className)} htmlFor={htmlFor ?? f?.controlId} {...rest} />;
}

export function HelpText({ id, className, children }: { id?: string; className?: string; children: ReactNode }) {
  return <p id={id} className={cx("field-help", className)}>{children}</p>;
}

export interface FieldProps {
  label: ReactNode;
  /** Muted text inside the label, after a space, e.g. "(optional)". It is part of the control's accessible name. */
  note?: ReactNode;
  /** Explanatory text below the control; the control's aria-describedby points at it. */
  help?: ReactNode;
  /** Validation message; also sets aria-invalid on the control and is part of its description. */
  error?: ReactNode;
  /** Sets aria-invalid without a message of its own (the explanation is shown elsewhere, e.g. in a page alert). */
  invalid?: boolean;
  /** "inline" puts the control (a switch or checkbox) before the label text. */
  layout?: "stack" | "inline";
  className?: string;
  children: ReactNode;
}

/** Wires one control to its label, help text and error text with matching ids. */
export function Field({ label, note, help, error, invalid, layout = "stack", className, children }: FieldProps) {
  const uid = useId();
  const controlId = `${uid}-control`;
  const helpId = `${uid}-help`;
  const errorId = `${uid}-error`;
  const describedBy = [help ? helpId : null, error ? errorId : null].filter(Boolean).join(" ") || undefined;
  const labelEl = <Label htmlFor={controlId}>{label}{note ? <> <span className="field-note">{note}</span></> : null}</Label>;
  const helpEl = help ? <HelpText id={helpId}>{help}</HelpText> : null;
  const errorEl = error ? <p id={errorId} className="field-error">{error}</p> : null;
  return (
    <Ctx.Provider value={{ controlId, describedBy, invalid: Boolean(error) || Boolean(invalid) }}>
      {layout === "inline" ? (
        <div className={cx("field", "inline", className)}>
          <div className="field-control">{children}</div>
          <div className="field-text">{labelEl}{helpEl}{errorEl}</div>
        </div>
      ) : (
        <div className={cx("field", className)}>
          {labelEl}{children}{helpEl}{errorEl}
        </div>
      )}
    </Ctx.Provider>
  );
}

export interface InputProps extends InputHTMLAttributes<HTMLInputElement> {
  /** A unit or adornment shown inside the field after the text. Decorative: put the unit in the label too. */
  suffix?: ReactNode;
}

export const Input = forwardRef<HTMLInputElement, InputProps>(function Input({ suffix, className, ...rest }, ref) {
  const a = useFieldControl(rest);
  const el = <input ref={ref} className={cx("input", className)} {...rest} {...a} />;
  if (!suffix) return el;
  return (
    <span className="input-group">
      {el}
      <span className="input-suffix" aria-hidden="true">{suffix}</span>
    </span>
  );
});

export const Select = forwardRef<HTMLSelectElement, SelectHTMLAttributes<HTMLSelectElement>>(function Select({ className, ...rest }, ref) {
  const a = useFieldControl(rest);
  return <select ref={ref} className={cx("select", className)} {...rest} {...a} />;
});

export const Textarea = forwardRef<HTMLTextAreaElement, TextareaHTMLAttributes<HTMLTextAreaElement>>(function Textarea({ className, ...rest }, ref) {
  const a = useFieldControl(rest);
  return <textarea ref={ref} className={cx("textarea", className)} {...rest} {...a} />;
});
