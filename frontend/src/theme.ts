// Theme: dark ("mission control") by default, light when the user toggled to it. The choice is a UI
// preference only; index.html applies it before first paint so there is no flash.
import { prefersReducedMotion } from "./motion/useReducedMotion";

export type Theme = "light" | "dark";
const KEY = "distillery.theme";
const FADE_MS = 500;

export const DEFAULT_THEME: Theme = "dark";

/** The OS preference. Informational only: the site defaults to dark regardless (see DEFAULT_THEME). */
export function systemTheme(): Theme {
  return window.matchMedia?.("(prefers-color-scheme: dark)").matches ? "dark" : "light";
}

export function currentTheme(): Theme {
  const t = document.documentElement.dataset.theme;
  return t === "light" || t === "dark" ? t : DEFAULT_THEME;
}

let fadeTimer: ReturnType<typeof setTimeout> | undefined;

export function setTheme(t: Theme): void {
  const root = document.documentElement;
  // Colours cross-fade (CSS `theme-fade`) unless the user asked for reduced motion.
  if (!prefersReducedMotion()) {
    root.classList.add("theme-fade");
    clearTimeout(fadeTimer);
    fadeTimer = setTimeout(() => root.classList.remove("theme-fade"), FADE_MS);
  }
  root.dataset.theme = t;
  try {
    localStorage.setItem(KEY, t);
  } catch {
    /* storage unavailable: preference just isn't remembered */
  }
}
