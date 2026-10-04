# ruff: noqa: E501  # long evidence strings in a proof harness
"""P5 / W5 proof harness: start a run FROM THE UI, watch it in the UI, kill and resume it.

Dry (default, zero spend):
    PY=/home/lenovo/.claude/jobs/958b3c06/tmp/pwv/bin/python   # any python with `playwright` installed
    $PY docs/proofs/p5_live.py                       # headless, starts its own API on a free port
    $PY docs/proofs/p5_live.py --headed              # watch it

Live (PAID, needs operator approval; NOT exercised by the dry proof):
    Full documented command (one run: UI submit, SIGKILL mid fine-tune, restart, resubmit, finish):
    $PY docs/proofs/p5_live.py --live --scale gated --admin-token $TOKEN --kill-stage finetune_r1 \
        --headed --video --out docs/proofs/evidence/p5_live.json
    Add --stop-before-start for a zero-spend check of the form + server env (POST aborted in browser).
    The live path starts its OWN local API on .distillery/p5live with .env + student Qwen3-1.7B,
    DISTILLERY_PRICES_FILE=.distillery/prices.json and DISTILLERY_RUN_CAP_USD=<budget>.

Checks (JSON summary written to --out):
  A  start from the UI; each stage transition visible in the UI <= 5 s after the store shows it; the spend
     meter is present and moves; the experiment tree is rendered; the report screen matches report.json.
  B  kill the worker (and the API) mid-stage, restart, resubmit the same run id, assert it resumes without
     re-running completed stages and without duplicate charge rows.
  C  forced failure, no product code: `--fail-image TAG` starts a SECOND local API whose live-run child
     is configured with fake credentials, DISTILLERY_SANDBOX_URL pointing at a closed port and
     DISTILLERY_SANDBOX_IMAGE=TAG. A dry run CANNOT exercise this (build_dry_run uses FakeSandbox and never
     reads DISTILLERY_SANDBOX_IMAGE), so the failure is induced on the live-run code path
     (make_live_deps -> ContreeSandbox.ensure_image) with fake credentials: zero spend, the real sandbox
     client really fails (connection refused) before any stage or paid job exists.
     Asserts: run ends failed, UI shows the reason, SSE event log ends with done/failed, no spend rows.
     The cancel/orphan_cancelled assertion needs an open fine-tune job, so it is live-only (N/A here).
     Result is merged into --out under "C_forced_failure" (`--only-c` runs just this check).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import signal
import socket
import sqlite3
import subprocess  # noqa: S404
import sys
import threading
import time
from pathlib import Path
from typing import Any

from playwright.sync_api import Page, sync_playwright

REPO = Path(__file__).resolve().parents[2]
LAT_LIMIT_S = 5.0
FINAL_STAGE = "final_eval"


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


class Server:
    def __init__(
        self,
        root: Path,
        log: Path,
        extra_env: dict[str, str] | None = None,
        keep_secrets: bool = False,
    ) -> None:
        self.root, self.log, self.port = root, log, free_port()
        self.extra_env = extra_env or {}
        self.keep_secrets = keep_secrets  # live only: the real credentials stay in the child env
        self.proc: subprocess.Popen[bytes] | None = None

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def start(self) -> None:
        env = {**os.environ, "DISTILLERY_HOME": str(self.root), "PYTHONUNBUFFERED": "1"}
        if not self.keep_secrets:
            for k in ("NEBIUS_API_KEY", "DISTILLERY_ADMIN_TOKEN"):
                env.pop(k, None)  # zero spend: dry only
        env.update(self.extra_env)
        self.proc = subprocess.Popen(  # noqa: S603
            [str(REPO / ".venv/bin/python"), "-m", "distillery", "--root", str(self.root), "serve",
             "--port", str(self.port)],
            cwd=REPO, env=env, stdout=self.log.open("ab"), stderr=subprocess.STDOUT,
            start_new_session=True,
        )  # fmt: skip
        import urllib.request

        for _ in range(100):
            try:
                urllib.request.urlopen(self.url + "/api/health", timeout=1)  # noqa: S310
                return
            except OSError:
                time.sleep(0.2)
        raise RuntimeError("server did not start")

    def kill(self) -> None:
        if self.proc and self.proc.poll() is None:
            os.killpg(self.proc.pid, signal.SIGKILL)
            self.proc.wait()


class StoreWatch:
    """Polls the run store (sqlite, read-only) and stamps the first time each (stage,status) is seen."""

    def __init__(self, db: Path, run_id: str) -> None:
        self.db, self.run_id = db, run_id
        self.first: dict[tuple[str, str], float] = {}
        self.stop = threading.Event()
        self.t = threading.Thread(target=self._loop, daemon=True)
        self.t.start()

    def rows(self) -> list[tuple[str, str]]:
        return stage_rows(self.db, self.run_id)

    def _loop(self) -> None:
        while not self.stop.is_set():
            now = time.time()
            for stage, status in self.rows():
                self.first.setdefault((stage, "done" if status == "complete" else status), now)
            time.sleep(0.03)


def q(db: Path, sql: str, args: tuple[Any, ...] = ()) -> list[tuple[Any, ...]]:
    if not db.exists():
        return []
    try:
        con = sqlite3.connect(f"file:{db}?mode=ro", uri=True, timeout=2)
        try:
            return con.execute(sql, args).fetchall()
        finally:
            con.close()
    except sqlite3.Error:
        return []


def stage_rows(db: Path, run_id: str) -> list[tuple[str, str]]:
    return [
        (r[0], r[1])
        for r in q(
            db, "SELECT stage,status FROM stages WHERE run_id=? ORDER BY updated_at", (run_id,)
        )
    ]


def ui_stages(page: Page) -> dict[str, str]:
    return dict(
        page.evaluate(
            "[...document.querySelectorAll('li.stg-step')].map(li=>"
            "[li.querySelector('.stg-name').textContent, li.dataset.status])"
        )
    )


def ui_meter(page: Page) -> float | None:
    v = page.evaluate(
        "(()=>{const m=document.querySelector('[role=meter]');return m?m.getAttribute('aria-valuenow'):null})()"
    )
    return None if v is None else float(v)


def start_from_ui(page: Page, base: str, live: bool, token: str | None) -> str:
    page.goto(f"{base}/new")
    page.get_by_role("button", name=re.compile("Start (dry|live) run")).wait_for()
    if live:
        page.get_by_label(re.compile("Dry run")).uncheck()
        page.get_by_label(re.compile("Admin token")).fill(token or "")
        page.get_by_label(re.compile("I approve spending")).check()
    page.get_by_role("button", name=re.compile("Start (dry|live) run")).click()
    page.wait_for_url(re.compile(r"/runs/[^/]+$"), timeout=15000)
    return page.url.rsplit("/", 1)[1]


def watch_ui(page: Page, db: Path, run_id: str, timeout_s: float) -> dict[str, Any]:
    ui_first: dict[tuple[str, str], float] = {}
    meters: list[float] = []
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        now = time.time()
        snap = ui_stages(page)
        for name, st in snap.items():
            ui_first.setdefault((name, st), now)
        m = ui_meter(page)
        if m is not None and (not meters or meters[-1] != m):
            meters.append(m)
        if (db.parent / "runs" / run_id / "report.json").exists() and ui_stages(page).get(
            FINAL_STAGE
        ) == "done":
            break
        page.wait_for_timeout(60)
    return {"ui_first": ui_first, "meters": meters}


def check_report(page: Page, base: str, run_dir: Path, run_id: str) -> dict[str, Any]:
    rep = json.loads((run_dir / "report.json").read_text())
    ev = rep["evaluation"]
    g = ev["gate"]
    page.goto(f"{base}/runs/{run_id}/report")
    page.wait_for_selector("text=Held-out accuracy", timeout=15000)
    text = page.inner_text("main")
    api = page.request.get(f"{base}/api/runs/{run_id}/report").json()
    want = {
        "decision": rep["decision"],
        "base": f"{ev['accuracy']['base'] * 100:.1f}%",
        "student": f"{ev['accuracy']['student'] * 100:.1f}%",
        "teacher": f"{ev['accuracy']['teacher'] * 100:.1f}%",
        "mcnemar_p": f"{g['mcnemar_p']:.4f}",
        "ratio_lo": f"{g['ratio_lo']:.3f}",
        "n": f"n={ev['n']}",
    }
    found = {k: (v in text) for k, v in want.items()}
    same_api = all(api.get(k) == rep[k] for k in ("decision", "run_id", "evaluation", "cost"))
    return {"want": want, "found_in_ui": found, "api_matches_report_json": same_api,
            "ok": all(found.values()) and same_api}  # fmt: skip


def check_tree(page: Page, base: str, run_id: str) -> dict[str, Any]:
    page.goto(f"{base}/runs/{run_id}/tree")
    page.wait_for_selector("text=Experiment tree", timeout=15000)
    page.wait_for_selector("text=base model", timeout=15000)
    text = page.inner_text("main")
    nodes = page.request.get(f"{base}/api/runs/{run_id}/tree").json()["nodes"]
    labels = [n["label"] for n in nodes]
    found = {lab: lab in text for lab in labels}
    return {"api_nodes": labels, "found_in_ui": found, "ok": len(nodes) > 1 and all(found.values())}


def dup_spend(db: Path, run_id: str) -> dict[str, Any]:
    rows = q(
        db,
        "SELECT kind,model,usd,input_tokens,output_tokens,created_at FROM spend WHERE run_id=?",
        (run_id,),
    )
    by_kind: dict[str, int] = {}
    for r in rows:
        by_kind[r[0]] = by_kind.get(r[0], 0) + 1
    return {
        "rows": len(rows),
        "by_kind": by_kind,
        "identical_row_dupes": len(rows) - len(set(rows)),
    }


def experiments(db: Path, run_id: str) -> list[dict[str, Any]]:
    out = []
    for name, data in q(
        db, "SELECT name,data_json FROM experiments WHERE run_id=? ORDER BY id", (run_id,)
    ):
        if name.startswith("finetune_job"):
            out.append({"name": name, **json.loads(data)})
    return out


def sse_events(base: str, run_id: str, timeout_s: float = 20.0) -> list[tuple[str, dict[str, Any]]]:
    """Read the run's SSE stream until its `done` event (or timeout); returns (event, data) pairs."""
    import urllib.request

    out: list[tuple[str, dict[str, Any]]] = []
    ev = None
    t0 = time.time()
    with urllib.request.urlopen(f"{base}/api/runs/{run_id}/events", timeout=timeout_s) as r:  # noqa: S310
        for raw in r:
            line = raw.decode().rstrip("\n")
            if line.startswith("event:"):
                ev = line[6:].strip()
            elif line.startswith("data:") and ev:
                out.append((ev, json.loads(line[5:])))
                if ev == "done" or time.time() - t0 > timeout_s:
                    break
    return out


def forced_failure(page: Page, proof: Path, fail_image: str) -> dict[str, Any]:
    """Check C: see module docstring. Returns evidence including per-assertion booleans."""
    token = "p5-fake-admin-token"  # noqa: S105  (throwaway, local server only)
    root = proof / "home_fail"
    if root.exists():
        import shutil

        shutil.rmtree(root)
    closed = free_port()  # nothing listens here
    env = {
        "NEBIUS_API_KEY": "fake-key-not-a-secret", "NEBIUS_BASE_URL": "http://127.0.0.1:9/v1",
        "NEBIUS_AI_PROJECT": "fake-project", "DISTILLERY_ADMIN_TOKEN": token,
        "DISTILLERY_SANDBOX_URL": f"http://127.0.0.1:{closed}", "DISTILLERY_SANDBOX_IMAGE": fail_image,
    }  # fmt: skip
    srv = Server(root, proof / "server_fail.log", env)
    srv.start()
    ev: dict[str, Any] = {
        "induced": {
            "DISTILLERY_SANDBOX_IMAGE": fail_image,
            "DISTILLERY_SANDBOX_URL": env["DISTILLERY_SANDBOX_URL"],
            "credentials": "fake (zero spend)",
            "path": "live run child -> make_live_deps -> ContreeSandbox.ensure_image (real client, closed port)",
        }
    }
    try:
        run_id = start_from_ui(page, srv.url, True, token)
        ev["run_id"] = run_id
        t0, detail = time.time(), {}
        while time.time() - t0 < 60:
            detail = page.request.get(f"{srv.url}/api/runs/{run_id}").json()
            if detail.get("status") == "failed":
                break
            time.sleep(0.3)
        reason = str((detail.get("run") or detail).get("error") or "")
        ev["api_status"], ev["api_error"] = detail.get("status"), reason
        page.goto(f"{srv.url}/runs/{run_id}")
        page.wait_for_selector("text=Run error", timeout=15000)
        ui_text = " ".join(page.inner_text("main").split())
        ev["ui_text"] = ui_text[:500]
        events = sse_events(srv.url, run_id)
        ev["sse_events_tail"] = events[-5:]
        done = [d for n, d in events if n == "done"]
        db = root / "index.sqlite"
        spend = q(db, "SELECT count(*) FROM spend WHERE run_id=?", (run_id,))
        jobs = [
            e for e in experiments(db, run_id)
            if e["name"] in ("finetune_job_started", "finetune_job_closed")
        ]  # fmt: skip
        ev["spend_rows"], ev["finetune_events"] = spend, jobs
        ev["assertions"] = {
            "run_ends_failed": detail.get("status") == "failed",
            "api_has_reason": "exit code" in reason,
            "ui_shows_reason": bool(reason) and reason[:60] in ui_text,
            "event_log_has_failure": bool(done) and done[-1].get("status") == "failed",
            "no_spend_rows": not spend or spend[0][0] == 0,
            "no_open_paid_job": not jobs,
        }
        ev["live_only"] = {
            "cancel_event_for_open_finetune_job": "N/A here: the failure happens before any fine-tune job "
            "exists. Needs a real live run failing after finetune_job_started; expect an experiments row "
            "finetune_job_closed outcome=orphan_cancelled (orchestrator._cancel_orphans) and a ceiling-only ledger",
            "ledger_correct_after_paid_failure": "needs paid spend rows; live only",
        }
    finally:
        srv.kill()
    return ev


def fault_hooks() -> list[str]:
    pat = re.compile(r"(FAULT|INJECT_|FAIL_STAGE|DISTILLERY_FAIL|CHAOS)", re.I)
    hits = []
    for p in (REPO / "src").rglob("*.py"):
        for i, line in enumerate(p.read_text().splitlines(), 1):
            if pat.search(line):
                hits.append(f"{p.relative_to(REPO)}:{i}: {line.strip()}")
    return hits


def only_c(a: argparse.Namespace, proof: Path) -> int:
    tag = a.fail_image or "docker://distillery-nonexistent-image:fail"
    with sync_playwright() as pw:
        browser = pw.chromium.launch(channel=a.chrome or None, headless=not a.headed)
        ctx = browser.new_context(viewport={"width": 1280, "height": 900})
        try:
            ev = forced_failure(ctx.new_page(), proof, tag)
        finally:
            ctx.close()
            browser.close()
    asserts = ev["assertions"]
    ev["result"] = "PASS" if all(asserts.values()) else "FAIL"
    for k, v in asserts.items():
        print(f"{'PASS' if v else 'FAIL':5} C.{k}")
    out = Path(a.out)
    data = json.loads(out.read_text()) if out.exists() else {}
    data["C_forced_failure"] = ev
    out.write_text(json.dumps(data, indent=2, default=str))
    print(f"merged C_forced_failure into {out}")
    return 0 if ev["result"] == "PASS" else 1


# ======================= LIVE (paid) path =======================
LIVE_BUDGET_USD = 8.0
LIVE_STUDENT = "Qwen/Qwen3-1.7B"
TERMINAL = ("failed", "cancelled", "canceled", "error")


def load_dotenv(path: Path) -> dict[str, str]:
    """Minimal KEY=VALUE parser (the app itself does not read .env). Values are never printed."""
    out: dict[str, str] = {}
    if not path.exists():
        return out
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        k = k.removeprefix("export ").strip()
        v = v.strip()
        if len(v) >= 2 and v[0] == v[-1] and v[0] in "\"'":
            v = v[1:-1]
        out[k] = v
    return out


def live_env(token: str, budget: float) -> dict[str, str]:
    env = load_dotenv(REPO / ".env")
    env.update({
        "DISTILLERY_ADMIN_TOKEN": token,
        "DISTILLERY_MODEL_STUDENT": LIVE_STUDENT,
        # the price file is ONLY read through this variable (config.load_config); without it every
        # LLM price is missing and the run refuses before spending anything
        "DISTILLERY_PRICES_FILE": str(REPO / ".distillery/prices.json"),
        # POST /api/runs rejects budget_usd > the server's run cap (default from .env was 5)
        "DISTILLERY_RUN_CAP_USD": str(budget),
    })  # fmt: skip
    return env


def secret_values(env: dict[str, str]) -> list[str]:
    keys = [
        k for k in env if any(t in k for t in ("KEY", "TOKEN", "SECRET", "PASSWORD", "PROJECT"))
    ]
    return [env[k] for k in keys if len(env[k]) >= 6]


def scrub(text: str, secrets: list[str]) -> str:
    for sec in secrets:
        text = text.replace(sec, "[REDACTED]")
    return text


def preflight_config(env: dict[str, str], budget: float) -> dict[str, Any]:
    """Load the exact config the live server and its CLI child will see, in the repo venv, and report
    booleans only (never values)."""
    code = (
        "import json,os;from distillery.config import load_config;c=load_config();"
        "print(json.dumps({'api_key_set':c.nebius_api_key is not None,'admin_token_set':c.admin_token is not None,"
        "'run_cap_usd':c.run_cap_usd,'project_cap_usd':c.project_cap_usd,'student':c.model_ids.get('student'),"
        "'prices_for_roles':{r:(c.model_ids.get(r) in c.prices) for r in ('planner','teacher','triage')},"
        "'finetune_price_for_student':c.model_ids.get('student') in c.finetune_prices,"
        "'finetune_usd_per_mtok':(c.finetune_prices[c.model_ids['student']].usd_per_mtok_trained_tokens if c.model_ids.get('student') in c.finetune_prices else None),"
        "'sandbox_usd_per_s':c.sandbox_price.usd_per_second if c.sandbox_price else None}))"
    )
    r = subprocess.run(  # noqa: S603
        [str(REPO / ".venv/bin/python"), "-c", code], cwd=REPO, env={**os.environ, **env},
        capture_output=True, text=True, timeout=60, check=False,
    )  # fmt: skip
    if r.returncode != 0:
        return {"ok": False, "error": r.stderr.strip().splitlines()[-1:] or ["?"]}
    d = json.loads(r.stdout)
    d["ok"] = bool(
        d["api_key_set"] and d["admin_token_set"] and d["run_cap_usd"] >= budget
        and d["student"] == LIVE_STUDENT and d["finetune_price_for_student"] and all(d["prices_for_roles"].values())
    )  # fmt: skip
    return d


def fill_live_form(page: Page, base: str, token: str, scale: str, budget: float) -> None:
    page.goto(f"{base}/new")
    page.get_by_role("button", name=re.compile("Start (dry|live) run")).wait_for()
    page.get_by_label(re.compile("Dry run")).uncheck()
    page.get_by_role("radio", name=scale, exact=True).check()
    page.get_by_label("Budget cap (USD)").fill(str(budget))
    page.get_by_label(re.compile("Admin token")).fill(token)
    page.get_by_label(re.compile("I approve spending")).check()
    page.get_by_role("button", name="Start live run").wait_for()


def check_form_only(
    page: Page, base: str, token: str, scale: str, budget: float, secrets: list[str]
) -> dict[str, Any]:
    """--stop-before-start: fill the real form, press Start, but ABORT the POST in the browser so it
    never reaches the server (no run, no worker, no spend). Asserts what would have been sent."""
    seen: list[dict[str, Any]] = []

    def trap(route: Any) -> None:
        req = route.request
        if req.method == "POST":
            seen.append({"url": req.url, "body": req.post_data_json,
                         "has_admin_header": bool(req.headers.get("x-admin-token")),
                         "header_matches_token": req.headers.get("x-admin-token") == token})  # fmt: skip
            route.abort()
        else:
            route.continue_()

    page.route(re.compile(r"/api/runs$"), trap)
    fill_live_form(page, base, token, scale, budget)
    pw_type = page.get_by_label(re.compile("Admin token")).get_attribute("type")
    page.get_by_role("button", name="Start live run").click()
    page.wait_for_timeout(1500)
    want = {
        "pack": "sql",
        "scale": scale,
        "dry_run": False,
        "budget_usd": budget,
        "approve_spend": True,
    }
    ok = (
        len(seen) == 1
        and seen[0]["body"] == want
        and seen[0]["header_matches_token"]
        and pw_type == "password"
    )
    ev = {"intercepted_post": [{k: v for k, v in x.items() if k != "header_matches_token"} for x in seen],
          "expected_body": want, "token_input_type": pw_type, "header_matches_token": bool(seen) and seen[0]["header_matches_token"],
          "url_after_click": page.url, "ok": ok and "/runs/" not in page.url}  # fmt: skip
    return json.loads(scrub(json.dumps(ev), secrets))


def run_status(page: Page, base: str, run_id: str) -> str:
    try:
        return str(page.request.get(f"{base}/api/runs/{run_id}").json().get("status"))
    except Exception:  # noqa: BLE001
        return "unreachable"


def run_live(a: argparse.Namespace, proof: Path) -> int:  # noqa: C901, PLR0912, PLR0915
    token = a.admin_token or os.environ.get("P5_ADMIN_TOKEN") or ""
    if not token:
        print("--live needs --admin-token (or $P5_ADMIN_TOKEN)")
        return 2
    scale = a.scale or "gated"
    budget = a.budget
    kill_stage = a.kill_stage or "finetune_r1"
    env = live_env(token, budget)
    secrets = secret_values(env)
    root = Path(a.root) if a.root else REPO / ".distillery/p5live"
    root.mkdir(parents=True, exist_ok=True)
    db = root / "index.sqlite"
    summary: dict[str, Any] = {"mode": "live", "scale": scale, "budget_usd": budget, "kill_stage": kill_stage,
                               "started": time.strftime("%FT%TZ", time.gmtime()), "checks": {}, "notes": [],
                               "stop_before_start": a.stop_before_start}  # fmt: skip
    checks = summary["checks"]
    pre = preflight_config(env, budget)
    summary["preflight_config"] = pre
    print("preflight:", json.dumps(pre), flush=True)
    if not pre.get("ok"):
        print("FAIL preflight config; refusing to start")
        return 2
    srv = Server(root, proof / "server_live.log", env, keep_secrets=True)
    srv.start()
    out = Path(a.out)

    def write() -> None:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(scrub(json.dumps(summary, indent=2, default=str), secrets))

    def verdict(name: str, ok: bool | None, **evidence: Any) -> None:
        checks[name] = {"result": "PASS" if ok else ("N/A" if ok is None else "FAIL"), **evidence}
        print(f"{checks[name]['result']:5} {name}", flush=True)
        write()

    video_kw: dict[str, Any] = {}
    if a.video:
        video_kw = {
            "record_video_dir": str(proof / "video"),
            "record_video_size": {"width": 960, "height": 720},
        }
    t_start = time.time()
    try:
        with sync_playwright() as pw:
            browser = pw.chromium.launch(channel=a.chrome or None, headless=not a.headed)
            ctx = browser.new_context(viewport={"width": 960, "height": 720}, **video_kw)
            try:
                page = ctx.new_page()
                if a.stop_before_start:
                    f = check_form_only(page, srv.url, token, scale, budget, secrets)
                    verdict("FORM_submit_body_and_header", f.pop("ok"), **f)
                    summary["all_checked_pass"] = all(
                        c["result"] != "FAIL" for c in checks.values()
                    )
                    write()
                    return 0 if summary["all_checked_pass"] else 1
                fill_live_form(page, srv.url, token, scale, budget)
                page.get_by_role("button", name="Start live run").click()
                page.wait_for_url(re.compile(r"/runs/[^/]+$"), timeout=30000)
                run_id = page.url.rsplit("/", 1)[1]
                summary["run_id"] = run_id
                print("LIVE RUN STARTED:", run_id, flush=True)
                write()
                run_dir = root / "runs" / run_id
                first: dict[tuple[str, str], float] = {}
                ui_first: dict[tuple[str, str], float] = {}
                meters: list[float] = []
                restarted_at: float | None = None
                killed = False
                kill_info: dict[str, Any] = {}
                cur_page, cur_base = page, srv.url
                deadline = time.time() + a.timeout_h * 3600
                end_state = "timeout"
                last_log = 0.0
                while time.time() < deadline:
                    now = time.time()
                    for stage, st in stage_rows(db, run_id):
                        first.setdefault((stage, "done" if st == "complete" else st), now)
                    try:
                        for name, st in ui_stages(cur_page).items():
                            ui_first.setdefault((name, st), now)
                        m = ui_meter(cur_page)
                        if m is not None and (not meters or meters[-1] != m):
                            meters.append(m)
                    except Exception:  # noqa: BLE001, S110  (page mid-navigation)
                        pass
                    rows = stage_rows(db, run_id)
                    evs = experiments(db, run_id)
                    started = [e for e in evs if e["name"] == "finetune_job_started"]
                    if (
                        not killed
                        and any(s == kill_stage and st != "complete" for s, st in rows)
                        and started
                    ):
                        before_upd = dict(
                            q(db, "SELECT stage,updated_at FROM stages WHERE run_id=?", (run_id,))
                        )
                        child = None
                        pidf = run_dir / "driver.pid"
                        if pidf.exists():
                            child = int(pidf.read_text().split()[0])
                            try:
                                os.kill(child, signal.SIGKILL)
                            except ProcessLookupError:
                                child = None
                        srv.kill()
                        killed = True
                        kill_info = {
                            "killed_driver_pid": child, "stage_rows_at_kill": rows,
                            "finetune_events_at_kill": evs, "spend_at_kill": dup_spend(db, run_id),
                            "t_kill": time.strftime("%FT%TZ", time.gmtime()),
                        }  # fmt: skip
                        print(f"KILLED CLI child {child} and API during {kill_stage}", flush=True)
                        write()
                        time.sleep(1.0)
                        srv.start()
                        restarted_at = time.time()
                        cur_base = srv.url
                        cur_page = ctx.new_page()
                        cur_page.goto(f"{cur_base}/runs/{run_id}")
                        cur_page.wait_for_selector("text=Stages", timeout=30000)
                        r = cur_page.request.post(
                            f"{cur_base}/api/runs", headers={"x-admin-token": token},
                            data={"pack": "sql", "scale": scale, "dry_run": False, "run_id": run_id,
                                  "approve_spend": True, "budget_usd": budget},
                        )  # fmt: skip
                        kill_info["resubmit_status"] = r.status
                        kill_info["resubmit_body"] = scrub(r.text(), secrets)[:300]
                        kill_info["stage_updated_at_before_kill"] = before_upd
                        print("resubmitted, status", r.status, flush=True)
                        write()
                        cur_page.goto(f"{cur_base}/runs/{run_id}")
                    if (run_dir / "report.json").exists():
                        end_state = "report"
                        break
                    if killed and now - (restarted_at or now) > 20:
                        stt = run_status(cur_page, cur_base, run_id)
                        if stt in TERMINAL:
                            end_state = stt
                            break
                    if (
                        not killed
                        and now - t_start > 120
                        and run_status(cur_page, cur_base, run_id) in TERMINAL
                    ):
                        end_state = "failed_before_kill"
                        break
                    if now - last_log > 600:
                        last_log = now
                        print(
                            time.strftime("%T"),
                            "stages:",
                            rows[-3:],
                            "ft_started:",
                            len(started),
                            flush=True,
                        )
                    cur_page.wait_for_timeout(250)
                time.sleep(2)
                summary["end_state"] = end_state
                lat: dict[str, float | None] = {}
                for (stage, st), t_store in sorted(first.items(), key=lambda kv: kv[1]):
                    t_ui = ui_first.get((stage, st))
                    lat[f"{stage}:{st}"] = None if t_ui is None else round(t_ui - t_store, 3)
                done_lat = {k: v for k, v in lat.items() if k.endswith(":done")}
                if restarted_at is not None:
                    # stages that were already done when the first poll on the NEW page ran carry no latency info
                    done_lat = {
                        k: v
                        for k, v in done_lat.items()
                        if first[(k.split(":")[0], "done")] > restarted_at
                    }
                    done_pre = {
                        k: v for k, v in lat.items() if k.endswith(":done") and k not in done_lat
                    }
                else:
                    done_pre = {}
                verdict(
                    "A1_stage_transitions_visible_within_5s",
                    bool(done_lat) and all(v is not None and v <= LAT_LIMIT_S for v in done_lat.values()),
                    done_latency_s=done_lat, pre_restart_done_latency_s=done_pre,
                    running_latency_s={k: v for k, v in lat.items() if k.endswith(":running")},
                    note="pre-restart stages were watched on the first page; post-restart on a new page",
                )  # fmt: skip
                if end_state != "report":
                    verdict(
                        "RUN_COMPLETED",
                        False,
                        end_state=end_state,
                        status=run_status(cur_page, cur_base, run_id),
                    )
                # ---- kill / resume invariants ----
                fte = experiments(db, run_id)
                st_ev = [e for e in fte if e["name"] == "finetune_job_started"]
                ad_ev = [e for e in fte if e["name"] == "finetune_job_adopted"]
                cl_ev = [e for e in fte if e["name"] == "finetune_job_closed"]
                jobs = sorted({e["job_id"] for e in st_ev})
                after = dup_spend(db, run_id)
                kill_info["finetune_events_final"] = fte
                kill_info["distinct_finetune_job_ids_started"] = len(jobs)
                kill_info["finetune_job_started_events"] = len(st_ev)
                summary["kill_resume"] = kill_info
                if not killed:
                    verdict(
                        "B_kill_resume",
                        False,
                        note=f"kill condition never met ({kill_stage} running + finetune_job_started)",
                    )
                else:
                    verdict("B1_resumes_to_completion", end_state == "report" and kill_info.get("resubmit_status") == 202,
                            resubmit_status=kill_info.get("resubmit_status"))  # fmt: skip
                    cd = {s for s, st in kill_info["stage_rows_at_kill"] if st == "complete"}
                    after_upd = dict(
                        q(db, "SELECT stage,updated_at FROM stages WHERE run_id=?", (run_id,))
                    )
                    rerun = [
                        s
                        for s in cd
                        if after_upd.get(s) != kill_info["stage_updated_at_before_kill"].get(s)
                    ]
                    verdict(
                        "B2_completed_stages_not_rerun",
                        end_state == "report" and not rerun,
                        completed_stages_rerun=rerun,
                    )
                    verdict(
                        "B3_job_adopted_not_recreated",
                        bool(ad_ev) and len(jobs) == 1 and len(st_ev) == 1,
                        adopted_events=ad_ev, finetune_job_started_events=len(st_ev), distinct_job_ids=len(jobs),
                        closed_events=cl_ev,
                        note="STRICT per the task: adoption + exactly one job. Code reality (orchestrator._cancel_orphans runs BEFORE _adoptable_job): after a SIGKILL the unclosed job is cancelled (orphan_cancelled) and a second job is created; adoption only fires for a job closed as 'aborted' whose provider status is still active/succeeded.",
                    )  # fmt: skip
                    verdict(
                        "B3b_no_two_jobs_running_at_once",
                        len(st_ev) <= 2 and all(e["job_id"] in {c["job_id"] for c in cl_ev} for e in st_ev[:-1]),
                        note="weaker: every job started before the last was explicitly closed (cancelled) before another was created",
                    )  # fmt: skip
                    ceil_rows = after["by_kind"].get("finetune_ceiling", 0) + after["by_kind"].get(
                        "finetune", 0
                    )
                    verdict("B4_no_duplicate_charge_rows", end_state == "report" and after["identical_row_dupes"] == 0 and ceil_rows <= len(jobs),
                            spend=after, finetune_charge_rows=ceil_rows, distinct_jobs=len(jobs))  # fmt: skip
                # ---- A2-A4 on real spend ----
                rows = q(db, "SELECT kind,model,usd FROM spend WHERE run_id=?", (run_id,))
                total = round(sum(r[2] for r in rows), 6)
                by_kind: dict[str, float] = {}
                for k, _m, usd in rows:
                    by_kind[k] = round(by_kind.get(k, 0.0) + usd, 6)
                cur_page.goto(f"{cur_base}/runs/{run_id}")
                cur_page.wait_for_selector("text=Spend", timeout=30000)
                cur_page.wait_for_timeout(1500)
                final_meter = ui_meter(cur_page)
                by_model = cur_page.inner_text("main")
                verdict(
                    "A2_spend_meter_matches_real_spend_rows",
                    bool(rows) and total > 0 and total <= budget and final_meter is not None
                    and abs(final_meter - total) <= max(0.01, 0.01 * total) and "BY MODEL" in by_model.upper(),
                    spend_rows=len(rows), spend_total_usd=total, spend_by_kind_usd=by_kind, final_meter_usd=final_meter,
                    meter_values_seen=meters[:200], budget_cap_usd=budget,
                )  # fmt: skip
                tree = check_tree(cur_page, cur_base, run_id)
                verdict("A3_experiment_tree_rendered", tree.pop("ok"), **tree)
                if (run_dir / "report.json").exists():
                    rep = check_report(cur_page, cur_base, run_dir, run_id)
                    verdict("A4_report_matches_report_json", rep.pop("ok"), **rep)
                else:
                    verdict(
                        "A4_report_matches_report_json",
                        None,
                        note="no report.json (run did not complete)",
                    )
            finally:
                ctx.close()
                browser.close()
    finally:
        srv.kill()
    summary["videos"] = sorted(
        str(p) for p in (proof / "video").glob("*.webm") if p.stat().st_mtime >= t_start
    )
    summary["all_checked_pass"] = all(c["result"] != "FAIL" for c in checks.values())
    write()
    print(f"wrote {out}")
    return 0 if summary["all_checked_pass"] else 1


def main() -> int:  # noqa: C901, PLR0912, PLR0915
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--base-url", default=None, help="connect to a running API instead of starting one"
    )
    ap.add_argument("--root", default=None, help="store root of that API (needed for store checks)")
    ap.add_argument(
        "--live",
        action="store_true",
        help="PAID run; needs --admin-token (own local API, own store; see run_live)",
    )
    ap.add_argument("--admin-token", default=None)
    ap.add_argument(
        "--kill-stage",
        default=None,
        help="stage to kill in (default: paraphrase in dry, finetune_r1 live)",
    )
    ap.add_argument(
        "--fail-image", default=None, metavar="TAG",
        help="run check C: 2nd local API whose live-run child gets DISTILLERY_SANDBOX_IMAGE=TAG (+ fake "
        "credentials, unreachable DISTILLERY_SANDBOX_URL) so the sandbox step really fails; zero spend",
    )  # fmt: skip
    ap.add_argument(
        "--only-c", action="store_true", help="run only check C and merge it into --out"
    )
    ap.add_argument("--scale", default=None, help="live: scale radio to pick (default gated)")
    ap.add_argument("--budget", type=float, default=LIVE_BUDGET_USD, help="live: budget cap USD")
    ap.add_argument("--video", action="store_true", help="live: record video (dry always records)")
    ap.add_argument(
        "--timeout-h", type=float, default=6.0, help="live: give up waiting after N hours"
    )
    ap.add_argument(
        "--stop-before-start", action="store_true",
        help="live: fill the real form and press Start but abort the POST in the browser (zero spend)",
    )  # fmt: skip
    ap.add_argument("--headed", action="store_true")
    ap.add_argument("--chrome", default="chrome", help="playwright channel; '' = bundled chromium")
    ap.add_argument("--proof-dir", default=str(REPO / ".distillery/proofs/p5"))
    ap.add_argument("--out", default=None, help="default p5_dry.json, or p5_live.json with --live")
    a = ap.parse_args()
    a.out = a.out or str(REPO / f"docs/proofs/evidence/p5_{'live' if a.live else 'dry'}.json")
    proof = Path(a.proof_dir)
    proof.mkdir(parents=True, exist_ok=True)
    if a.only_c:
        return only_c(a, proof)
    if a.live:
        return run_live(a, proof)
    live = a.live
    summary: dict[str, Any] = {"mode": "live" if live else "dry", "started": time.strftime("%FT%TZ", time.gmtime()),
                               "checks": {}, "notes": []}  # fmt: skip
    checks = summary["checks"]

    root = Path(a.root) if a.root else proof / "home"
    if a.base_url is None:
        if root.exists() and not a.root:
            import shutil

            shutil.rmtree(root)
        server: Server | None = Server(root, proof / "server.log")
        server.start()
        base = server.url
    else:
        server, base = None, a.base_url.rstrip("/")
    store_dir = root if live else root / "dry-runs"
    db = store_dir / "index.sqlite"
    if a.base_url and not a.root:
        summary["notes"].append("no --root: store-based checks (latency, kill, duplicates) skipped")

    def verdict(name: str, ok: bool | None, **evidence: Any) -> None:
        checks[name] = {"result": "PASS" if ok else ("N/A" if ok is None else "FAIL"), **evidence}
        print(f"{checks[name]['result']:5} {name}", flush=True)

    with sync_playwright() as pw:
        browser = pw.chromium.launch(channel=a.chrome or None, headless=not a.headed)
        ctx = browser.new_context(
            viewport={"width": 1280, "height": 900}, record_video_dir=str(proof / "video")
        )
        try:
            # ---------------- A: acceptance run ----------------
            page = ctx.new_page()
            run_id = start_from_ui(page, base, live, a.admin_token)
            summary["acceptance_run_id"] = run_id
            watch = StoreWatch(db, run_id) if db.parent.exists() else None
            w = watch_ui(page, db, run_id, 300 if live else 120)
            if watch:
                time.sleep(0.3)
                watch.stop.set()
            ui_first = w["ui_first"]
            lat: dict[str, float | None] = {}
            if watch:
                for (stage, st), t_store in sorted(watch.first.items(), key=lambda kv: kv[1]):
                    t_ui = ui_first.get((stage, st))
                    lat[f"{stage}:{st}"] = None if t_ui is None else round(t_ui - t_store, 3)
            done_lat = {k: v for k, v in lat.items() if k.endswith(":done")}
            run_lat = {k: v for k, v in lat.items() if k.endswith(":running")}
            verdict(
                "A1_stage_transitions_visible_within_5s",
                bool(done_lat) and all(v is not None and v <= LAT_LIMIT_S for v in done_lat.values()),
                done_latency_s=done_lat, running_latency_s=run_lat,
                note="'done' transitions are the strict check; a 'running' state shorter than the UI poll can legitimately be skipped (None)",
                max_done_latency_s=max((v for v in done_lat.values() if v is not None), default=None),
            )  # fmt: skip
            meters = w["meters"]
            page.goto(f"{base}/runs/{run_id}")
            page.wait_for_selector("text=Spend", timeout=15000)
            by_model = page.inner_text("main")
            verdict(
                "A2_spend_meter_present",
                len(meters) >= 1 and "BY MODEL" in by_model.upper(),
                meter_values_seen=meters,
                note="UI has ONE run-level spend meter plus a by-model breakdown; there is no per-STAGE spend in the UI, API or spend table (no stage column)",
            )  # fmt: skip
            verdict(
                "A2b_spend_meter_per_stage",
                None,
                note="not available: spend rows carry kind/model only, not stage; would need code",
            )
            run_dir = store_dir / "runs" / run_id
            tree = check_tree(page, base, run_id)
            verdict("A3_experiment_tree_rendered", tree.pop("ok"), **tree)
            if run_dir.exists():
                rep = check_report(page, base, run_dir, run_id)
                verdict("A4_report_matches_report_json", rep.pop("ok"), **rep)
            else:
                verdict(
                    "A4_report_matches_report_json", None, note="run dir not reachable (no --root)"
                )
            ctrl = dup_spend(db, run_id)
            ctrl_ft = experiments(db, run_id)
            summary["control_run"] = {"spend": ctrl, "finetune_events": ctrl_ft}

            # ---------------- B: kill + resume ----------------
            if server is None or not db.parent.exists():
                verdict(
                    "B_kill_resume",
                    None,
                    note="needs the local server (no --base-url) or --root + a way to kill the worker",
                )
            else:
                page2 = ctx.new_page()
                kid = start_from_ui(page2, base, live, a.admin_token)
                run_dir2 = store_dir / "runs" / kid
                kill_stage = a.kill_stage or ("finetune_r1" if live else "paraphrase")
                t0 = time.time()
                killed_at = None
                while time.time() - t0 < 120:
                    rows = stage_rows(db, kid)
                    if any(s == kill_stage and st != "complete" for s, st in rows):
                        killed_at = rows
                        break
                    time.sleep(0.01)
                if killed_at is None:
                    verdict(
                        "B_kill_resume",
                        False,
                        note=f"never saw stage {kill_stage} running before the run ended",
                    )
                else:
                    before = {(s, st) for s, st in killed_at}
                    before_upd = dict(
                        q(db, "SELECT stage,updated_at FROM stages WHERE run_id=?", (kid,))
                    )
                    before_spend = dup_spend(db, kid)
                    child = None
                    pidf = run_dir2 / "driver.pid"
                    if pidf.exists():
                        child = int(pidf.read_text().split()[0])
                        try:
                            os.kill(child, signal.SIGKILL)
                        except ProcessLookupError:
                            child = None
                    server.kill()  # the API (and its worker thread) dies too
                    time.sleep(0.5)
                    report_before = (run_dir2 / "report.json").exists()
                    server.start()  # restart (new port; the UI is re-opened on it)
                    base2 = server.url
                    page3 = ctx.new_page()
                    page3.goto(f"{base2}/runs/{kid}")
                    page3.wait_for_selector("text=Stages", timeout=15000)
                    page3.wait_for_timeout(1500)
                    ui_after_kill = page3.inner_text("main")[:300]
                    # resume: same run id through the API (the UI has no run-id field)
                    r = page3.request.post(
                        f"{base2}/api/runs",
                        data={"pack": "sql", "scale": "tiny", "dry_run": True, "run_id": kid},
                    )
                    page3.goto(f"{base2}/runs/{kid}")
                    t1 = time.time()
                    while time.time() - t1 < (300 if live else 120):
                        if (run_dir2 / "report.json").exists() and ui_stages(page3).get(
                            FINAL_STAGE
                        ) == "done":
                            break
                        page3.wait_for_timeout(200)
                    resumed = (run_dir2 / "report.json").exists()
                    after_upd = dict(
                        q(db, "SELECT stage,updated_at FROM stages WHERE run_id=?", (kid,))
                    )
                    complete_before = {s for s, st in before if st == "complete"}
                    # a stage that was complete before the kill must keep its original row (not re-run)
                    rerun = [s for s in complete_before if after_upd.get(s) != before_upd.get(s)]
                    after = dup_spend(db, kid)
                    fte = experiments(db, kid)
                    started = [e for e in fte if e["name"] == "finetune_job_started"]
                    adopted = [e for e in fte if e["name"] == "finetune_job_adopted"]
                    rounds_started: dict[int, int] = {}
                    for e in started:
                        rounds_started[e["round"]] = rounds_started.get(e["round"], 0) + 1
                    ceil_ctrl, ceil_now = (
                        ctrl["by_kind"].get("finetune_ceiling", 0),
                        after["by_kind"].get("finetune_ceiling", 0),
                    )
                    ev = {
                        "run_id": kid, "killed_driver_pid": child, "kill_snapshot_stages": sorted(before),
                        "report_existed_at_kill": report_before, "ui_text_after_kill_and_restart": ui_after_kill,
                        "resubmit_status": r.status, "resumed_to_report": resumed,
                        "completed_stages_rerun": rerun, "finetune_events": fte,
                        "spend_before_kill": before_spend, "spend_after_resume": after,
                        "control_finetune_ceiling_rows": ceil_ctrl, "resumed_finetune_ceiling_rows": ceil_now,
                    }  # fmt: skip
                    verdict("B1_resumes_to_completion", resumed and r.status == 202, **ev)
                    verdict(
                        "B2_completed_stages_not_rerun",
                        resumed and not rerun,
                        completed_stages_rerun=rerun,
                    )
                    mid_ft = kill_stage.startswith("finetune")
                    verdict(
                        "B3_job_adopted_not_recreated",
                        (bool(adopted) and all(n == 1 for n in rounds_started.values()))
                        if mid_ft
                        else None,
                        adopted_events=adopted, jobs_started_per_round=rounds_started,
                        note="only meaningful when killed mid fine-tune (live, or --kill-stage finetune_r1). Dry: the fake fine-tune stage lasts milliseconds and its job lives inside the killed process, so adoption cannot be exercised; code path: unclosed jobs are cancelled by _cancel_orphans, finetune_job_adopted only fires for a job closed as 'aborted'",
                    )  # fmt: skip
                    verdict(
                        "B4_no_duplicate_charge_rows",
                        resumed
                        and after["identical_row_dupes"] == 0
                        and ceil_now <= len({e["job_id"] for e in started}),
                        identical_row_dupes=after["identical_row_dupes"], ceiling_rows_resumed=ceil_now, distinct_jobs_started=len({e["job_id"] for e in started}),
                    )  # fmt: skip

            # ---------------- C: forced failure ----------------
            hooks = fault_hooks()
            if a.fail_image:
                ev_c = forced_failure(ctx.new_page(), proof, a.fail_image)
                summary["C_forced_failure"] = ev_c
                verdict(
                    "C_failure_ui_reason_and_cancel_event", all(ev_c["assertions"].values()), **ev_c
                )
            elif hooks:
                verdict(
                    "C_failure_ui_reason_and_cancel_event",
                    None,
                    note="hook(s) found but harness has no driver for them",
                    hooks=hooks,
                )
            else:
                verdict(
                    "C_failure_ui_reason_and_cancel_event", None,
                    note="not possible without code: no fault-injection hook/env var exists in src/ (grep for FAULT|INJECT_|FAIL_STAGE|DISTILLERY_FAIL|CHAOS found nothing)",
                    grep_hits=hooks,
                )  # fmt: skip
        finally:
            ctx.close()  # flushes the videos
            browser.close()
            if server:
                server.kill()
    vids = sorted(str(p) for p in (proof / "video").glob("*.webm"))
    summary["videos"] = vids
    summary["all_checked_pass"] = all(c["result"] != "FAIL" for c in checks.values())
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(summary, indent=2, default=str))
    print(f"wrote {a.out}")
    return 0 if summary["all_checked_pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
