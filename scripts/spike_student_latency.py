"""Latency / memory spike: base Qwen3-1.7B on sandbox CPU (SPENDS sandbox compute; the user runs
it, not the builder). No adapter, no fine-tune.

Procedure (stops at the first failure and records it as the result):
  1. build the deps image (pip + weights + script), record build time;
  2. ONE load at batch size 1 (one prompt): load_s, gen_s, per-sample latency, peak_rss_mb;
  3. ONLY if 2 succeeded: batch size 2 (one job, two prompts);
  4. ONLY if 3 succeeded: throughput at concurrency 1/5/10/20 (batch size 1 per job, at most 40
     samples each, bounded by the total generation cap).
Sandboxes are fixed at 4 vCPU / 4 GB: a 1.7B bf16 model (~3.4 GB of weights) may not fit, and an
OOM / non-zero exit / timeout is a valid, recorded result, not an error. Servers are always closed.

The 40 prompts come from ``prompts.build_messages`` over template dev tasks of the synthetic DB
(``generate_tasks``); nothing sealed is touched.

Usage:
  python scripts/spike_student_latency.py [--model Qwen/Qwen3-1.7B] [--out PATH]
  python scripts/spike_student_latency.py --dry-run     # fakes only, no network, FAKE table

Environment for a live run (never printed): NEBIUS_API_KEY, NEBIUS_PROJECT_ID / NEBIUS_AI_PROJECT,
optional DISTILLERY_SANDBOX_URL / DISTILLERY_SANDBOX_IMAGE.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from distillery.prompts import build_messages
from distillery.sandbox import SandboxError
from distillery.student import ChatMessages, StudentServingError
from distillery.taskpacks.sql import schema as sql_schema
from distillery.taskpacks.sql.questions import generate_tasks

DEFAULT_MODEL = "Qwen/Qwen3-1.7B"
DEFAULT_OUT = Path("docs/spikes/s5-qwen3-1_7b.json")
N_PROMPTS = 40
LEVELS = (1, 5, 10, 20)  # below the 50-operation cap and student_concurrency=20
MAX_PER_LEVEL = 40
BASELINE = {  # Qwen3-0.6B, measured live 2026-09-30 (STATUS.md)
    "model": "Qwen/Qwen3-0.6B",
    "s_per_sample": 7.8,
    "peak_rss_mb": 2140,  # 2.09 GB
}
_OOM_HINTS = (
    "oom",
    "out of memory",
    "killed",
    "memoryerror",
    "exit=137",
    "exit=-9",
    "cannot allocate",
)
_RECOVERABLE = (StudentServingError, SandboxError, RuntimeError, OSError, TimeoutError)


class Server(Protocol):
    timings: list[dict[str, object]]

    def generate(self, messages_batch: Sequence[ChatMessages]) -> list[str]: ...

    def close(self) -> None: ...


MakeServer = Callable[[int, int, float], Server]  # batch_size, concurrency, timeout_s


@dataclass
class Budget:
    max_generations: int
    deadline: float  # monotonic
    clock: Callable[[], float]
    used: int = 0

    @property
    def remaining(self) -> int:
        return self.max_generations - self.used

    def seconds_left(self) -> float:
        return self.deadline - self.clock()


def classify(exc: BaseException) -> str:
    text = f"{type(exc).__name__}: {exc}".lower()
    if any(h in text for h in _OOM_HINTS):
        return "oom"
    if "timed_out=true" in text or "timeout" in text or isinstance(exc, TimeoutError):
        return "timeout"
    return "error"


def build_prompts(n: int = N_PROMPTS, seed: int = 0) -> list[ChatMessages]:
    db = sql_schema.build_database(seed)
    ddl = sql_schema.schema_ddl(seed)
    tasks = generate_tasks(db, n, seed).tasks[:n]
    if len(tasks) < n:
        raise SystemExit(f"only {len(tasks)} template tasks generated, need {n}")
    return [build_messages(t.question, ddl, role="eval_student") for t in tasks]


def _phase(
    name: str,
    server: Server,
    prompts: Sequence[ChatMessages],
    *,
    batch_size: int,
    concurrency: int,
    budget: Budget,
    clock: Callable[[], float],
) -> dict[str, Any]:
    """Run one generate call and summarise it. Raises on failure (the caller records it)."""
    seen = len(server.timings)
    budget.used += len(prompts)  # counted before the call: a failed job still spent compute
    t0 = clock()
    server.generate(prompts)
    wall = clock() - t0
    new = server.timings[seen:]
    per_sample = [
        float(t["gen_s"]) / float(t["n"])  # type: ignore[arg-type]
        for t in new
        if t.get("gen_s") is not None and t.get("n")
    ]
    loads = [float(t["load_s"]) for t in new if t.get("load_s") is not None]  # type: ignore[arg-type]
    rss = [float(t["peak_rss_mb"]) for t in new if t.get("peak_rss_mb") is not None]  # type: ignore[arg-type]
    return {
        "name": name,
        "batch_size": batch_size,
        "concurrency": concurrency,
        "n": len(prompts),
        "jobs": len(new),
        "wall_s": round(wall, 2),
        "samples_per_s": round(len(prompts) / wall, 4) if wall > 0 else None,
        "load_s": round(sum(loads) / len(loads), 2) if loads else None,
        "per_sample_s": round(sum(per_sample) / len(per_sample), 2) if per_sample else None,
        "peak_rss_mb": max(rss) if rss else None,
        "status": "ok",
    }


def run_spike(
    *,
    model: str,
    prompts: Sequence[ChatMessages],
    build_deps: Callable[[], str],
    make_server: MakeServer,
    max_generations: int = 60,
    max_wall_s: float = 3600.0,
    levels: Sequence[int] = LEVELS,
    clock: Callable[[], float] = time.monotonic,
) -> dict[str, Any]:
    """Run the spike procedure; never raises on OOM/exit/timeout, records it as the outcome."""
    budget = Budget(max_generations, clock() + max_wall_s, clock)
    result: dict[str, Any] = {
        "model": model,
        "adapter": None,
        "sandbox": {"vcpu": 4, "ram_gb": 4},
        "n_prompts": len(prompts),
        "max_generations": max_generations,
        "max_wall_s": max_wall_s,
        "build_s": None,
        "phases": [],
        "outcome": "completed",
        "failure": None,
        "baseline": BASELINE,
    }
    servers: list[Server] = []

    def attempt(name: str, bs: int, conc: int, n: int) -> bool:
        if budget.seconds_left() <= 0:
            result["outcome"] = "aborted_wall_clock"
            result["failure"] = {"phase": name, "reason": "wall-clock budget exhausted"}
            return False
        n = min(n, budget.remaining, len(prompts))
        if n <= 0:
            result["phases"].append({"name": name, "status": "skipped_generation_cap"})
            result["outcome"] = "stopped_generation_cap"
            return False
        server: Server | None = None
        try:
            server = make_server(bs, conc, min(1800.0, max(1.0, budget.seconds_left())))
            servers.append(server)
            result["phases"].append(
                _phase(name, server, prompts[:n], batch_size=bs, concurrency=conc,
                       budget=budget, clock=clock)
            )  # fmt: skip
            return True
        except _RECOVERABLE as exc:
            kind = classify(exc)
            result["outcome"] = f"aborted_{kind}"
            result["failure"] = {"phase": name, "reason": str(exc)[:600]}
            result["phases"].append({"name": name, "batch_size": bs, "concurrency": conc,
                                     "n": n, "status": f"failed_{kind}"})  # fmt: skip
            return False
        finally:
            if server is not None:
                server.close()

    try:
        t0 = clock()
        try:
            build_deps()
        except _RECOVERABLE as exc:
            result["build_s"] = round(clock() - t0, 2)
            result["outcome"] = f"aborted_{classify(exc)}"
            result["failure"] = {"phase": "build", "reason": str(exc)[:600]}
        else:
            result["build_s"] = round(clock() - t0, 2)
            ok = attempt("batch1_load", 1, 1, 1) and attempt("batch2", 2, 1, 2)
            for conc in levels:
                if not ok:
                    break
                ok = attempt(f"concurrency_{conc}", 1, conc, min(MAX_PER_LEVEL, max(2, 2 * conc)))
    finally:
        for s in servers:  # idempotent; belt and braces on any unexpected exception
            s.close()
    result["total_generations"] = budget.used
    return result


# ---------------------------------------------------------------- reporting


def format_table(result: dict[str, Any], *, fake: bool = False) -> str:
    def f(v: object, unit: str = "") -> str:
        return "-" if v is None else f"{v}{unit}"

    head = ["run", "batch", "conc", "n", "s/sample", "load_s", "peak RSS MB", "samples/s", "status"]
    b = BASELINE
    rows = [["0.6B baseline", "-", "-", "-", f(b["s_per_sample"]), "-", f(b["peak_rss_mb"]), "-",
             "measured (STATUS.md)"]]  # fmt: skip
    for p in result["phases"]:
        rows.append([
            f"{result['model'].split('/')[-1]} {p['name']}", f(p.get("batch_size")),
            f(p.get("concurrency")), f(p.get("n")), f(p.get("per_sample_s")), f(p.get("load_s")),
            f(p.get("peak_rss_mb")), f(p.get("samples_per_s")), p["status"],
        ])  # fmt: skip
    widths = [max(len(r[i]) for r in [head, *rows]) for i in range(len(head))]
    lines = ["  ".join(c.ljust(w) for c, w in zip(r, widths, strict=True)) for r in [head, *rows]]
    banner = ["*** FAKE DATA (dry run, no sandbox was used): NOT a measurement ***"] if fake else []
    foot = [
        f"outcome: {result['outcome']}   build_s: {f(result['build_s'])}   "
        f"generations: {result['total_generations']}/{result['max_generations']}"
    ]
    if result["failure"]:
        foot.append(f"failure: {result['failure']}")
    return "\n".join([*banner, *lines, *foot, *banner])


# ---------------------------------------------------------------- fakes (dry run and tests)


class FakeSpikeServer:
    """In-memory server with the ``SandboxCpuStudent`` timing shape; no sandbox, no network."""

    def __init__(self, *, fail: Exception | None = None, peak_rss_mb: int = 3100) -> None:
        self.timings: list[dict[str, object]] = []
        self.closed = False
        self.calls = 0
        self._fail = fail
        self._rss = peak_rss_mb

    def generate(self, messages_batch: Sequence[ChatMessages]) -> list[str]:
        self.calls += 1
        if self._fail is not None:
            raise self._fail
        self.timings.append({"load_s": 1.0, "gen_s": 2.0 * len(messages_batch),
                             "n": len(messages_batch), "peak_rss_mb": self._rss})  # fmt: skip
        return ["SELECT 1"] * len(messages_batch)

    def close(self) -> None:
        self.closed = True


def fake_prompts(n: int = N_PROMPTS) -> list[ChatMessages]:
    return [[{"role": "user", "content": f"q{i}"}] for i in range(n)]


# ---------------------------------------------------------------- live wiring


def live_wiring(
    sandbox: Any, image: str, bridge: Any, model: str
) -> tuple[Callable[[], str], MakeServer]:
    from distillery.sandbox_student import SandboxCpuStudent, ServingImages

    images = ServingImages(sandbox, image, bridge, base_model=model)

    def make_server(batch_size: int, concurrency: int, timeout_s: float) -> Server:
        return SandboxCpuStudent(
            sandbox, image, bridge, base_model=model, batch_size=batch_size,
            concurrency=concurrency, timeout_s=timeout_s, images=images,
        )  # fmt: skip

    return images.deps_image, make_server


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--out", default=None, help=f"results JSON (live default: {DEFAULT_OUT})")
    ap.add_argument("--dry-run", action="store_true", help="fakes only; no network, FAKE table")
    ap.add_argument("--max-generations", type=int, default=60)
    ap.add_argument("--max-wall-s", type=float, default=3600.0)
    args = ap.parse_args(argv)

    if args.dry_run:
        prompts = fake_prompts()
        result = run_spike(
            model=args.model, prompts=prompts, build_deps=lambda: "fake-image",
            make_server=lambda bs, c, t: FakeSpikeServer(),
            max_generations=args.max_generations, max_wall_s=args.max_wall_s,
        )  # fmt: skip
        result["fake"] = True
        print(format_table(result, fake=True))
        if args.out:
            _write(Path(args.out), result)
        return 0

    from distillery.cli import SANDBOX_BASE_IMAGE
    from distillery.config import load_config
    from distillery.sandbox import ContreeSandbox
    from distillery.sandbox_executor import AsyncBridge

    config = load_config()
    if config.nebius_api_key is None or config.nebius_project_id is None:
        raise SystemExit("NEBIUS_API_KEY and NEBIUS_PROJECT_ID must be set in the environment")
    secret = config.nebius_api_key.get_secret_value()
    prompts = build_prompts()
    with AsyncBridge() as bridge:
        sandbox = ContreeSandbox(
            lambda: secret,
            os.environ.get("DISTILLERY_SANDBOX_URL") or None,
            project_id=config.nebius_project_id.get_secret_value(),
        )
        image = bridge.run(
            sandbox.ensure_image(os.environ.get("DISTILLERY_SANDBOX_IMAGE") or SANDBOX_BASE_IMAGE)
        )
        build_deps, make_server = live_wiring(sandbox, image, bridge, args.model)
        result = run_spike(
            model=args.model, prompts=prompts, build_deps=build_deps, make_server=make_server,
            max_generations=args.max_generations, max_wall_s=args.max_wall_s,
        )  # fmt: skip
    result["fake"] = False
    out = Path(args.out) if args.out else DEFAULT_OUT
    _write(out, result)
    print(format_table(result))
    print(f"wrote {out}")
    return 0


def _write(path: Path, result: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    sys.exit(main())
