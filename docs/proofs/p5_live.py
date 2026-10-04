# ruff: noqa: E501  # long evidence strings in a proof harness
"""P5 / W5 proof harness: start a run FROM THE UI, watch it in the UI, kill and resume it.

Dry (default, zero spend):
    PY=/home/lenovo/.claude/jobs/958b3c06/tmp/pwv/bin/python   # any python with `playwright` installed
    $PY docs/proofs/p5_live.py                       # headless, starts its own API on a free port
    $PY docs/proofs/p5_live.py --headed              # watch it

Live (PAID, needs operator approval; NOT exercised by the dry proof):
    $PY docs/proofs/p5_live.py --live --base-url http://127.0.0.1:8000 --root .distillery/live \
        --admin-token ... --kill-stage finetune_r1 --out docs/proofs/evidence/p5_live.json
    (the live path is untested; it needs the live server's store reachable via --root, see README notes
    in the JSON `notes`).

Checks (JSON summary written to --out):
  A  start from the UI; each stage transition visible in the UI <= 5 s after the store shows it; the spend
     meter is present and moves; the experiment tree is rendered; the report screen matches report.json.
  B  kill the worker (and the API) mid-stage, restart, resubmit the same run id, assert it resumes without
     re-running completed stages and without duplicate charge rows.
  C  forced stage failure: only if a fault-injection hook exists in src/ (none is added here).
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
    def __init__(self, root: Path, log: Path) -> None:
        self.root, self.log, self.port = root, log, free_port()
        self.proc: subprocess.Popen[bytes] | None = None

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def start(self) -> None:
        env = {**os.environ, "DISTILLERY_HOME": str(self.root), "PYTHONUNBUFFERED": "1"}
        for k in ("NEBIUS_API_KEY", "DISTILLERY_ADMIN_TOKEN"):
            env.pop(k, None)  # zero spend: dry only
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


def fault_hooks() -> list[str]:
    pat = re.compile(r"(FAULT|INJECT_|FAIL_STAGE|DISTILLERY_FAIL|CHAOS)", re.I)
    hits = []
    for p in (REPO / "src").rglob("*.py"):
        for i, line in enumerate(p.read_text().splitlines(), 1):
            if pat.search(line):
                hits.append(f"{p.relative_to(REPO)}:{i}: {line.strip()}")
    return hits


def main() -> int:  # noqa: C901, PLR0912, PLR0915
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--base-url", default=None, help="connect to a running API instead of starting one"
    )
    ap.add_argument("--root", default=None, help="store root of that API (needed for store checks)")
    ap.add_argument("--live", action="store_true", help="PAID run; needs --admin-token. Untested.")
    ap.add_argument("--admin-token", default=None)
    ap.add_argument(
        "--kill-stage",
        default=None,
        help="stage to kill in (default: paraphrase in dry, finetune_r1 live)",
    )
    ap.add_argument("--headed", action="store_true")
    ap.add_argument("--chrome", default="chrome", help="playwright channel; '' = bundled chromium")
    ap.add_argument("--proof-dir", default=str(REPO / ".distillery/proofs/p5"))
    ap.add_argument("--out", default=str(REPO / "docs/proofs/evidence/p5_dry.json"))
    a = ap.parse_args()
    proof = Path(a.proof_dir)
    proof.mkdir(parents=True, exist_ok=True)
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
            if hooks:
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
