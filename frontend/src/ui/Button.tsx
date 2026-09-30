import { forwardRef, type ButtonHTMLAttributes } from "react";
import { cx } from "./cx";

export type ButtonVariant = "default" | "primary" | "ghost" | "danger";

export interface ButtonProps extends ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: ButtonVariant;
  size?: "md" | "sm";
  /** Disables the button and marks it busy while an action is running. */
  loading?: boolean;
}

/** `type` defaults to "button" so a Button inside a form never submits by accident. */
export const Button = forwardRef<HTMLButtonElement, ButtonProps>(function Button(
  { variant = "default", size = "md", loading, disabled, className, type = "button", ...rest },
  ref,
) {
  return (
    <button
      ref={ref}
      type={type}
      className={cx("btn", variant !== "default" && variant, size === "sm" && "sm", className)}
      disabled={disabled || loading}
      aria-busy={loading ? true : undefined}
      {...rest}
    />
  );
});

export interface IconButtonProps extends Omit<ButtonProps, "aria-label"> {
  /** The accessible name. */
  label: string;
}

/**
 * A compact button whose visible content is a glyph. The name comes from `label`, never from the glyph.
 * To add a hover/focus hint, wrap it in `<Tooltip>`; that is left to the caller so the tooltip code
 * (popper positioning) is only bundled where it is used.
 */
export const IconButton = forwardRef<HTMLButtonElement, IconButtonProps>(function IconButton({ label, className, ...rest }, ref) {
  return <Button ref={ref} className={cx("icon-btn", className)} aria-label={label} {...rest} />;
});
