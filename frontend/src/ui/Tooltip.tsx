import * as RTooltip from "@radix-ui/react-tooltip";
import type { ReactElement, ReactNode } from "react";
import { TOOLTIP_DELAY_MS, TOOLTIP_OFFSET } from "./constants";

/**
 * Supplementary label for a control that already has an accessible name. The trigger keeps its own
 * name; the tooltip only repeats or expands it for sighted pointer and keyboard users.
 * Self-contained: it brings its own provider, so it works anywhere in the tree.
 */
export function Tooltip(props: { content: ReactNode; children: ReactElement; side?: "top" | "bottom" | "left" | "right" }) {
  return (
    <RTooltip.Provider delayDuration={TOOLTIP_DELAY_MS} skipDelayDuration={TOOLTIP_DELAY_MS}>
      <RTooltip.Root>
        <RTooltip.Trigger asChild>{props.children}</RTooltip.Trigger>
        <RTooltip.Portal>
          <RTooltip.Content className="tooltip" side={props.side ?? "bottom"} sideOffset={TOOLTIP_OFFSET} collisionPadding={TOOLTIP_OFFSET}>
            {props.content}
          </RTooltip.Content>
        </RTooltip.Portal>
      </RTooltip.Root>
    </RTooltip.Provider>
  );
}
