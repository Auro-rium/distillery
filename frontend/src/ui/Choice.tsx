import { forwardRef, useId, type InputHTMLAttributes, type ReactNode } from "react";
import { cx } from "./cx";
import { useFieldControl } from "./Field";

/** A native checkbox (for agreeing to something, as opposed to a Switch, which changes a setting). */
export const Checkbox = forwardRef<HTMLInputElement, Omit<InputHTMLAttributes<HTMLInputElement>, "type">>(function Checkbox({ className, ...rest }, ref) {
  const a = useFieldControl(rest);
  return <input ref={ref} type="checkbox" className={cx("check", className)} {...rest} {...a} />;
});

export interface ChoiceOption<T extends string> { value: T; label: ReactNode }

/** One-of-many choice: a fieldset with a legend and native radios, drawn as a segmented control. */
export function ChoiceGroup<T extends string>(props: {
  legend: ReactNode;
  name: string;
  value: T;
  onChange: (value: T) => void;
  options: readonly ChoiceOption<T>[];
  help?: ReactNode;
}) {
  const helpId = useId();
  return (
    <fieldset className="choice-group" aria-describedby={props.help ? helpId : undefined}>
      <legend className="field-label">{props.legend}</legend>
      <div className="choice-row">
        {props.options.map((o) => (
          <label key={o.value} className="choice">
            <input type="radio" name={props.name} value={o.value} checked={props.value === o.value} onChange={() => props.onChange(o.value)} />
            <span>{o.label}</span>
          </label>
        ))}
      </div>
      {props.help ? <p id={helpId} className="field-help">{props.help}</p> : null}
    </fieldset>
  );
}
