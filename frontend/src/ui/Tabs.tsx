import * as RTabs from "@radix-ui/react-tabs";
import { forwardRef, type ComponentPropsWithoutRef, type ElementRef } from "react";
import { cx } from "./cx";

export const Tabs = RTabs.Root;

export const TabsList = forwardRef<ElementRef<typeof RTabs.List>, ComponentPropsWithoutRef<typeof RTabs.List>>(
  function TabsList({ className, ...rest }, ref) {
    return <RTabs.List ref={ref} className={cx("tabs-list", className)} {...rest} />;
  },
);

export const TabsTrigger = forwardRef<ElementRef<typeof RTabs.Trigger>, ComponentPropsWithoutRef<typeof RTabs.Trigger>>(
  function TabsTrigger({ className, ...rest }, ref) {
    return <RTabs.Trigger ref={ref} className={cx("tabs-trigger", className)} {...rest} />;
  },
);

export const TabsContent = forwardRef<ElementRef<typeof RTabs.Content>, ComponentPropsWithoutRef<typeof RTabs.Content>>(
  function TabsContent({ className, ...rest }, ref) {
    return <RTabs.Content ref={ref} className={cx("tabs-content", className)} {...rest} />;
  },
);
