import { lazy, Suspense, type ComponentProps } from "react";
import type { Dialog as DialogType } from "./Dialog";

// Radix Dialog (focus trap, portal, scroll lock) is ~20 kB gzip and only needed once a dialog is opened.
// Loading it lazily keeps it out of the initial bundle; the wrapper renders nothing while closed.
const Inner = lazy(() => import("./Dialog").then((m) => ({ default: m.Dialog })));

export function LazyDialog(props: ComponentProps<typeof DialogType>) {
  if (!props.open) return null;
  return (
    <Suspense fallback={null}>
      <Inner {...props} />
    </Suspense>
  );
}
