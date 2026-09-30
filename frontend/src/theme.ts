// Theme: follows prefers-color-scheme unless the user toggled. The choice is a UI preference only.
import { prefersReducedMotion } from "./motion/useReducedMotion";

export type Theme = "light" | "dark";
const KEY = "distillery.theme";
const FADE_MS = 500;

export function systemTheme(): Theme {
  return window.matchMedia?.("(prefers-color-scheme: dark)").matches ? "dark" : "light";
}

export function currentTheme(): Theme {
  const t = document.documentElement.dataset.theme;
  return t === "light" || t === "dark" ? t : systemTheme();
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
