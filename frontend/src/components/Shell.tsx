// App chrome: skip link, header with nav and controls, phone menu, shortcut list, footer.
import { useEffect, useState } from "react";
import { Link, NavLink, useLocation } from "react-router-dom";
import { useFeaturedRun } from "../api/useFeaturedRun";
import type { Theme } from "../theme";
import { Button, IconButton } from "../ui";
import { LazyDialog as Dialog } from "../ui/LazyDialog";

export const SITE_NAME = "Distillery";
export const GITHUB_URL = "https://github.com/Auro-rium/distillery";

interface NavItem { to: string; label: string; end: boolean }

/**
 * Mission · Evidence · Live · Playground · New run. Evidence (the featured run's report) and Live (its
 * run page) exist only once /api/replay names a featured run (see useFeaturedRun); with none they are
 * left out rather than pointed at a guessed target. Live uses `end` so it is not also active on the report.
 */
function useNav(): NavItem[] {
  const { run } = useFeaturedRun();
  const id = run ? encodeURIComponent(run.run_id) : null;
  return [
    { to: "/", label: "Mission", end: true },
    ...(id ? [{ to: `/runs/${id}/report`, label: "Evidence", end: true }, { to: `/runs/${id}`, label: "Live", end: true }] : []),
    { to: "/playground", label: "Playground", end: false },
    { to: "/new", label: "New run", end: false },
  ];
}

function GitHubLink({ className }: { className: string }) {
  return (
    <a className={className} href={GITHUB_URL} target="_blank" rel="noreferrer noopener" aria-label="GitHub repository (opens in a new tab)">
      <svg className="gh-ico" viewBox="0 0 16 16" aria-hidden="true" focusable="false">
        <path d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.013 8.013 0 0016 8c0-4.42-3.58-8-8-8z" />
      </svg>
      GitHub
    </a>
  );
}

/** Human name of a route, used for the document title and the page announcement. */
export function pageTitle(pathname: string): string {
  if (pathname === "/") return "Mission";
  if (pathname === "/new") return "New run";
  if (pathname === "/playground") return "Playground";
  if (pathname === "/humanset") return "Human set";
  const m = /^\/runs\/([^/]+)(?:\/(report|tree))?\/?$/.exec(pathname);
  if (m) {
    let id = m[1];
    try { id = decodeURIComponent(id); } catch { /* keep the raw segment */ }
    return m[2] === "report" ? `Report ${id}` : m[2] === "tree" ? `Tree ${id}` : `Run ${id}`;
  }
  return "Page not found";
}

export function SkipLink({ onSkip }: { onSkip: () => void }) {
  return (
    <a className="skip-link" href="#main" onClick={(e) => { e.preventDefault(); onSkip(); }}>
      Skip to main content
    </a>
  );
}

export function ThemeToggle({ theme, onToggle }: { theme: Theme; onToggle: () => void }) {
  const next: Theme = theme === "dark" ? "light" : "dark";
  return (
    <Button className="theme-btn" aria-label={`Switch to ${next} theme`} onClick={onToggle}>
      <span className="theme-ico" data-t={theme} aria-hidden="true" />
      {theme === "dark" ? "Light" : "Dark"}
    </Button>
  );
}

/** Narrow-screen navigation (900 px and below): the same destinations plus GitHub in a modal sheet. */
function MobileMenu({ nav }: { nav: NavItem[] }) {
  const [open, setOpen] = useState(false);
  const { pathname } = useLocation();
  useEffect(() => { setOpen(false); }, [pathname]);
  return (
    <>
      <Button className="menu-btn" aria-haspopup="dialog" aria-expanded={open} onClick={() => setOpen(true)}>Menu</Button>
      <Dialog open={open} onOpenChange={setOpen} title="Menu" variant="sheet">
        <nav className="menu-nav" aria-label="Menu">
          {nav.map((n) => (
            <NavLink key={n.to} to={n.to} end={n.end} onClick={() => setOpen(false)}>{n.label}</NavLink>
          ))}
          <GitHubLink className="gh-menu" />
        </nav>
      </Dialog>
    </>
  );
}

export function SiteHeader(props: { theme: Theme; onToggleTheme: () => void; onShowHelp: () => void }) {
  const nav = useNav();
  return (
    <header className="site-header">
      <div className="container site-header-row">
        <Link to="/" className="brand">{SITE_NAME}</Link>
        <nav className="site-nav" aria-label="Main">
          {nav.map((n) => <NavLink key={n.to} to={n.to} end={n.end}>{n.label}</NavLink>)}
        </nav>
        <div className="header-actions">
          <IconButton className="kbd-btn" label="Keyboard shortcuts" aria-haspopup="dialog" onClick={props.onShowHelp}>?</IconButton>
          <GitHubLink className="gh-link" />
          <ThemeToggle theme={props.theme} onToggle={props.onToggleTheme} />
          <MobileMenu nav={nav} />
        </div>
      </div>
    </header>
  );
}

export function SiteFooter() {
  return (
    <footer className="site-footer">
      <div className="container">
        <p className="muted">
          {SITE_NAME}. Every figure on these pages is read from the API response, and runs on fake models are labelled as such.
        </p>
      </div>
    </footer>
  );
}

const HELP: [string, string][] = [
  ["?", "Show or hide this list"],
  ["t", "Switch theme"],
  ["g then r", "Go to Mission"],
  ["g then n", "Go to New run"],
  ["g then p", "Go to Playground"],
  ["+ and −", "Zoom the experiment tree"],
  ["Arrow keys", "Move between tree nodes, or pan the tree"],
  ["Esc", "Close"],
];

export function ShortcutHelp({ open, onClose }: { open: boolean; onClose: () => void }) {
  // Radix closes on Escape pressed inside the dialog; this also closes it when the key arrives at the window.
  useEffect(() => {
    if (!open) return;
    const esc = (e: KeyboardEvent) => { if (e.key === "Escape") onClose(); };
    window.addEventListener("keydown", esc);
    return () => window.removeEventListener("keydown", esc);
  }, [open, onClose]);
  return (
    <Dialog open={open} onOpenChange={(o) => { if (!o) onClose(); }} title="Keyboard shortcuts">
      <dl className="kbd-list">
        {HELP.map(([k, d]) => (<div key={k}><dt><kbd>{k}</kbd></dt><dd>{d}</dd></div>))}
      </dl>
    </Dialog>
  );
}
