import * as RDialog from "@radix-ui/react-dialog";
import { useRef, type ReactNode } from "react";
import { Button } from "./Button";
import { cx } from "./cx";

export interface DialogProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  title: ReactNode;
  description?: ReactNode;
  children?: ReactNode;
  /** "sheet" slides in from the top edge (used for the phone menu). */
  variant?: "modal" | "sheet";
  className?: string;
}

/**
 * Modal dialog: focus is trapped, Escape and the Close button dismiss it, the page behind is inert.
 * Focus returns to whatever had it when the dialog opened, unless something else took focus meanwhile
 * (for example the app moving focus to the new page after a navigation).
 */
export function Dialog(props: DialogProps) {
  const opener = useRef<HTMLElement | null>(null);
  return (
    <RDialog.Root open={props.open} onOpenChange={props.onOpenChange}>
      <RDialog.Portal>
        <RDialog.Overlay className="dialog-overlay" />
        <RDialog.Content
          className={cx("dialog-content", props.variant === "sheet" && "sheet", props.className)}
          {...(props.description ? {} : { "aria-describedby": undefined })}
          onOpenAutoFocus={() => { opener.current = document.activeElement as HTMLElement | null; }}
          onCloseAutoFocus={(e) => {
            const a = document.activeElement;
            if (a && a !== document.body) { e.preventDefault(); return; }
            const back = opener.current;
            if (back && back.isConnected) { e.preventDefault(); back.focus(); }
          }}
        >
          <div className="dialog-head">
            <RDialog.Title className="dialog-title">{props.title}</RDialog.Title>
            <RDialog.Close asChild><Button size="sm">Close</Button></RDialog.Close>
          </div>
          {props.description && <RDialog.Description className="dialog-desc muted">{props.description}</RDialog.Description>}
          {props.children}
        </RDialog.Content>
      </RDialog.Portal>
    </RDialog.Root>
  );
}
