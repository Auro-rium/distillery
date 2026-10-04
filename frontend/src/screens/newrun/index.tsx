import { useEffect, useState, type FormEvent } from "react";
import { useNavigate } from "react-router-dom";
import { ApiError, api, setAdminToken } from "../../api/client";
import type { Config, NewRunBody } from "../../api/types";
import { fmtUsd } from "../../api/format";
import { ApiErrorState, ErrorState } from "../../components";
import { Button, Card, Checkbox, ChoiceGroup, Field, Input, Select, Switch } from "../../ui";
import { describeCreateError } from "./errors";

const SCALES = ["tiny", "small", "full", "gated"] as const;
type Scale = (typeof SCALES)[number];
const SCALE_OPTIONS = SCALES.map((s) => ({ value: s, label: s }));
const MEMORY_ONLY = "Held in memory only; lost on reload.";

export default function NewRun() {
  const nav = useNavigate();
  const [cfg, setCfg] = useState<Config | null>(null);
  const [scale, setScale] = useState<Scale>("tiny");
  const [dry, setDry] = useState(true);
  const [budget, setBudget] = useState("");
  const [token, setToken] = useState(""); // component state only, never persisted
  const [approve, setApprove] = useState(false);
  const [err, setErr] = useState<string | null>(null);
  const [badField, setBadField] = useState<"budget" | "token" | "approve" | null>(null);
  const [cfgErr, setCfgErr] = useState<unknown>(null);
  const [busy, setBusy] = useState(false);

  useEffect(() => {
    api.config().then((c) => { setCfg(c); setBudget((b) => b || String(c.run_cap_usd)); }).catch((e: unknown) => setCfgErr(e));
  }, []);

  function fail(field: typeof badField, message: string) {
    setBadField(field);
    setErr(message);
  }

  async function submit(ev: FormEvent) {
    ev.preventDefault();
    setErr(null);
    setBadField(null);
    const cap = budget.trim() === "" ? undefined : Number(budget);
    if (cap !== undefined && (!Number.isFinite(cap) || cap <= 0)) return fail("budget", "Budget cap must be a positive number of USD.");
    if (!dry && !token) return fail("token", describeCreateError(new ApiError(401, "unauthorized", "")));
    if (!dry && !approve) return fail("approve", "Tick the spend approval box to start a live run.");
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

  const budgetHelp = cfg
    ? `Blank uses the server default of ${fmtUsd(cfg.run_cap_usd, 2)}. Spending stops at this cap.`
    : cfgErr !== null
      ? "The server default could not be loaded. Blank uses it anyway."
      : "Blank uses the server default. Spending stops at this cap.";

  return (
    <div className="stack newrun">
      <h1>New run</h1>
      {cfgErr !== null && <ApiErrorState title="Could not load the server defaults" error={cfgErr} />}
      <Card className="form-card">
        <form className="form" onSubmit={submit} noValidate>
          <div className="form-grid">
            <Field label="Pack" help="Text-to-SQL is the only pack this server offers.">
              <Select value="sql" disabled><option value="sql">sql (text-to-SQL)</option></Select>
            </Field>
            <ChoiceGroup legend="Scale" name="scale" value={scale} onChange={setScale} options={SCALE_OPTIONS} />
            <Field label="Budget cap (USD)" help={budgetHelp}>
              <Input
                inputMode="decimal"
                suffix="USD"
                value={budget}
                onChange={(e) => { setBudget(e.target.value); if (badField === "budget") setBadField(null); }}
                aria-invalid={badField === "budget" || undefined}
              />
            </Field>
            <Field label="Admin token" help={`${dry ? "Optional for dry runs." : "Required for live runs."} ${MEMORY_ONLY}`}>
              <Input
                type="password"
                autoComplete="off"
                value={token}
                onChange={(e) => { setToken(e.target.value); if (badField === "token") setBadField(null); }}
                aria-invalid={badField === "token" || undefined}
              />
            </Field>
            <Field
              className="span-2"
              layout="inline"
              label="Dry run (fake models, no spend, results are NOT real)"
              help="Turn this off to call paid Token Factory models. A live run also needs the admin token and your approval below."
            >
              <Switch checked={dry} onCheckedChange={setDry} />
            </Field>
          </div>
          {!dry && (
            <div className="state warn spend-warning" role="note">
              <p><strong>This will spend real money.</strong> A live run calls paid Nebius Token Factory models and starts fine-tuning
                jobs. Spending stops at the budget cap, but every dollar up to it is billed to the account.</p>
              <Field layout="inline" label="I approve spending up to the budget cap on this run.">
                <Checkbox checked={approve} onChange={(e) => { setApprove(e.target.checked); if (badField === "approve") setBadField(null); }} aria-invalid={badField === "approve" || undefined} />
              </Field>
            </div>
          )}
          {err && <ErrorState title="Could not start run" message={err} />}
          <div className="form-actions">
            <Button variant="primary" type="submit" loading={busy}>{busy ? "Starting…" : dry ? "Start dry run" : "Start live run"}</Button>
          </div>
        </form>
      </Card>
    </div>
  );
}
