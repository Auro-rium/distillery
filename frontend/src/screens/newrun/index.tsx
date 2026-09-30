import { useEffect, useState, type FormEvent } from "react";
import { useNavigate } from "react-router-dom";
import { ApiError, api, setAdminToken } from "../../api/client";
import type { Config, NewRunBody } from "../../api/types";
import { fmtUsd } from "../../api/format";
import { ApiErrorState, Card, ErrorState } from "../../components";
import { describeCreateError } from "./errors";

const SCALES = ["tiny", "small", "full"] as const;

export default function NewRun() {
  const nav = useNavigate();
  const [cfg, setCfg] = useState<Config | null>(null);
  const [scale, setScale] = useState<(typeof SCALES)[number]>("tiny");
  const [dry, setDry] = useState(true);
  const [budget, setBudget] = useState("");
  const [token, setToken] = useState(""); // component state only, never persisted
  const [approve, setApprove] = useState(false);
  const [err, setErr] = useState<string | null>(null);
  const [cfgErr, setCfgErr] = useState<unknown>(null);
  const [busy, setBusy] = useState(false);

  useEffect(() => {
    api.config().then((c) => { setCfg(c); setBudget((b) => b || String(c.run_cap_usd)); }).catch((e: unknown) => setCfgErr(e));
  }, []);

  async function submit(ev: FormEvent) {
    ev.preventDefault();
    setErr(null);
    const cap = budget.trim() === "" ? undefined : Number(budget);
    if (cap !== undefined && (!Number.isFinite(cap) || cap <= 0)) return setErr("Budget cap must be a positive number of USD.");
    if (!dry && !token) return setErr(describeCreateError(new ApiError(401, "unauthorized", "")));
    if (!dry && !approve) return setErr("Tick the spend approval box to start a live run.");
    const body: NewRunBody = { pack: "sql", scale, dry_run: dry };
    if (cap !== undefined) body.budget_usd = cap;
    if (!dry) body.approve_spend = true;
    setAdminToken(token);
    setBusy(true);
    try {
      const r = await api.createRun(body);
      nav(`/runs/${encodeURIComponent(r.run_id)}`);
    } catch (e) {
      setErr(describeCreateError(e));
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="stack">
      <h1>New run</h1>
      {cfgErr !== null && <ApiErrorState title="Could not load the server defaults" error={cfgErr} />}
      <Card>
        <form className="stack" onSubmit={submit} noValidate>
          <label>Pack<br /><select value="sql" disabled aria-label="Pack"><option value="sql">sql (text-to-SQL)</option></select></label>
          <fieldset>
            <legend>Scale</legend>
            {SCALES.map((s) => (
              <label key={s} style={{ marginRight: 16 }}>
                <input type="radio" name="scale" value={s} checked={scale === s} onChange={() => setScale(s)} /> {s}
              </label>
            ))}
          </fieldset>
          <label>
            <input type="checkbox" checked={dry} onChange={(e) => setDry(e.target.checked)} /> Dry run (fake models, no spend, results are NOT real)
          </label>
          <label>Budget cap (USD){cfg ? ` — server default ${fmtUsd(cfg.run_cap_usd, 2)}` : ""}<br />
            <input inputMode="decimal" value={budget} onChange={(e) => setBudget(e.target.value)} aria-label="Budget cap (USD)" />
          </label>
          <label>Admin token{dry ? " (optional for dry runs)" : ""}<br />
            <input type="password" autoComplete="off" value={token} onChange={(e) => setToken(e.target.value)} aria-label="Admin token" />
            <span className="muted"> Held in memory only; lost on reload.</span>
          </label>
          {!dry && (
            <div className="state" role="note" style={{ textAlign: "left" }}>
              <strong>This will spend real money.</strong> A live run calls paid Nebius Token Factory models and starts fine-tuning
              jobs. Spending stops at the budget cap, but every dollar up to it is billed to the account.
              <p style={{ margin: "8px 0 0" }}>
                <label><input type="checkbox" checked={approve} onChange={(e) => setApprove(e.target.checked)} /> I approve spending up to the budget cap on this run.</label>
              </p>
            </div>
          )}
          {err && <ErrorState title="Could not start run" message={err} />}
          <button className="btn primary" type="submit" disabled={busy}>{busy ? "Starting…" : dry ? "Start dry run" : "Start live run"}</button>
        </form>
      </Card>
    </div>
  );
}
