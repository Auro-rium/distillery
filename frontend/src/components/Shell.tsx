// App chrome: skip link, header with nav and controls, phone menu, shortcut list, footer.
import { useEffect, useState } from "react";
import { Link, NavLink, useLocation } from "react-router-dom";
import type { Theme } from "../theme";
import { Button, IconButton } from "../ui";
import { LazyDialog as Dialog } from "../ui/LazyDialog";

export const SITE_NAME = "Distillery";

const NAV = [
  { to: "/", label: "Replay", end: true },
  { to: "/new", label: "New run", end: false },
  { to: "/playground", label: "Playground", end: false },
] as const;

/** Human name of a route, used for the document title and the page announcement. */
export function pageTitle(pathname: string): string {
  if (pathname === "/") return "Replay";
  if (pathname === "/new") return "New run";
  if (pathname === "/playground") return "Playground";
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

/** Phone navigation (shown at 560 px and below): the same three destinations in a modal sheet. */
function MobileMenu() {
  const [open, setOpen] = useState(false);
  const { pathname } = useLocation();
  useEffect(() => { setOpen(false); }, [pathname]);
  return (
    <>
      <Button className="menu-btn" aria-haspopup="dialog" aria-expanded={open} onClick={() => setOpen(true)}>Menu</Button>
      <Dialog open={open} onOpenChange={setOpen} title="Menu" variant="sheet">
        <nav className="menu-nav" aria-label="Menu">
          {NAV.map((n) => (
            <NavLink key={n.to} to={n.to} end={n.end} onClick={() => setOpen(false)}>{n.label}</NavLink>
          ))}
        </nav>
      </Dialog>
    </>
  );
}

export function SiteHeader(props: { theme: Theme; onToggleTheme: () => void; onShowHelp: () => void }) {
  return (
    <header className="site-header">
      <div className="container site-header-row">
        <Link to="/" className="brand">{SITE_NAME}</Link>
        <nav className="site-nav" aria-label="Main">
          {NAV.map((n) => <NavLink key={n.to} to={n.to} end={n.end}>{n.label}</NavLink>)}
        </nav>
        <div className="header-actions">
          <IconButton className="kbd-btn" label="Keyboard shortcuts" aria-haspopup="dialog" onClick={props.onShowHelp}>?</IconButton>
          <ThemeToggle theme={props.theme} onToggle={props.onToggleTheme} />
          <MobileMenu />
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
  ["g then r", "Go to Replay"],
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
