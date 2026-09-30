import { forwardRef, type InputHTMLAttributes } from "react";
import { cx } from "./cx";
import { useFieldControl } from "./Field";

export interface SwitchProps extends Omit<InputHTMLAttributes<HTMLInputElement>, "type" | "role" | "checked" | "onChange" | "defaultChecked"> {
  checked: boolean;
  onCheckedChange: (checked: boolean) => void;
}

/**
 * On/off setting that takes effect immediately. A native checkbox with role="switch" (Space toggles it,
 * a click on its label toggles it, it takes part in form data) drawn as a track and thumb. Inside a
 * Field it picks up the label, help text and error wiring automatically.
 */
export const Switch = forwardRef<HTMLInputElement, SwitchProps>(function Switch({ checked, onCheckedChange, className, ...rest }, ref) {
  const a = useFieldControl(rest);
  return (
    <span className={cx("switch", className)}>
      <input ref={ref} type="checkbox" role="switch" checked={checked} onChange={(e) => onCheckedChange(e.target.checked)} {...rest} {...a} />
      <span className="switch-track" aria-hidden="true"><span className="switch-thumb" /></span>
    </span>
  );
});
