// Theme: follows prefers-color-scheme unless the user toggled. The choice is a UI preference only.
export type Theme = "light" | "dark";
const KEY = "distillery.theme";

export function systemTheme(): Theme {
  return window.matchMedia?.("(prefers-color-scheme: dark)").matches ? "dark" : "light";
}

export function currentTheme(): Theme {
  const t = document.documentElement.dataset.theme;
  return t === "light" || t === "dark" ? t : systemTheme();
}

export function setTheme(t: Theme): void {
  document.documentElement.dataset.theme = t;
  try {
    localStorage.setItem(KEY, t);
  } catch {
    /* storage unavailable: preference just isn't remembered */
  }
}
