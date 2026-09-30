"""Batched sandbox diagnosis of the student vs the base model (SPENDS sandbox compute; the
orchestrator runs it, not the builder).

For a run directory, generates with (i) the base model and (ii) each named adapter over the
adapter's OWN training rows, the dev tasks and the sealed held-out, in one job per model (each
model loads once). Writes raw and extracted SQL per task plus per-set accuracy and the
identical-string rate base vs student (raw text and extracted SQL) to a JSON file.

Usage:
  python scripts/diagnose_student.py RUN_ID --adapter round1=1:/path/to/checkpoint_dir \\
      [--adapter round2=2:/path/to/other] [--root .distillery] [--out diag.json]

Environment (never printed): NEBIUS_API_KEY, NEBIUS_PROJECT_ID / NEBIUS_AI_PROJECT,
DISTILLERY_MODEL_STUDENT, optional DISTILLERY_SANDBOX_URL / DISTILLERY_SANDBOX_IMAGE.
"""

import argparse
import json
import os
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from distillery import diagnostics as d
from distillery.config import load_config
from distillery.sandbox_executor import AsyncBridge
from distillery.store import Store
from distillery.taskpacks.sql import schema as sql_schema
from distillery.taskpacks.sql.executor import LocalExecutor

ADAPTER_FILES = ("adapter_config.json", "adapter_model.safetensors", "adapter_model.bin")


def parse_adapter(spec: str) -> tuple[str, tuple[int, list[Path]]]:
    """NAME=ROUND:CHECKPOINT_DIR -> (name, (round, adapter files))."""
    name, _, rest = spec.partition("=")
    rnd, _, path = rest.partition(":")
    if not (name and rnd.isdigit() and path):
        raise SystemExit(f"--adapter must look like NAME=ROUND:DIR, got {spec!r}")
    files = [Path(path) / f for f in ADAPTER_FILES if (Path(path) / f).exists()]
    if not files:
        raise SystemExit(f"no adapter files in {path}")
    return name, (int(rnd), files)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("run_id")
    ap.add_argument("--adapter", action="append", required=True)
    ap.add_argument("--root", default=".distillery")
    ap.add_argument("--out", default=None)
    ap.add_argument("--db-seed", type=int, default=0)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--concurrency", type=int, default=10)
    ap.add_argument("--max-new-tokens", type=int, default=160)
    args = ap.parse_args(argv)

    from distillery.cli import SANDBOX_BASE_IMAGE
    from distillery.sandbox import ContreeSandbox
    from distillery.sandbox_student import SandboxCpuStudent, ServingImages

    config = load_config()
    if config.nebius_api_key is None or config.nebius_project_id is None:
        raise SystemExit("NEBIUS_API_KEY and NEBIUS_PROJECT_ID must be set in the environment")
    base_model = config.require_model("student")
    secret = config.nebius_api_key.get_secret_value()
    store = Store(Path(args.root))
    out_path = Path(args.out or store.run_dir(args.run_id) / "diagnose_student.json")
    adapters = dict(parse_adapter(a) for a in args.adapter)
    with AsyncBridge() as bridge:
        sandbox = ContreeSandbox(
            lambda: secret,
            os.environ.get("DISTILLERY_SANDBOX_URL") or None,
            project_id=config.nebius_project_id.get_secret_value(),
        )
        image = bridge.run(
            sandbox.ensure_image(os.environ.get("DISTILLERY_SANDBOX_IMAGE") or SANDBOX_BASE_IMAGE)
        )
        images = ServingImages(sandbox, image, bridge, base_model=base_model)

        def make_server(files: Sequence[Path]) -> Any:
            return SandboxCpuStudent(
                sandbox, image, bridge, base_model=base_model, adapter_files=files,
                batch_size=args.batch_size, concurrency=args.concurrency,
                max_new_tokens=args.max_new_tokens, images=images,
            )  # fmt: skip

        db_path = store.run_dir(args.run_id) / "db.sqlite"
        result = d.run_student_diagnosis(
            store, args.run_id, adapters, make_server=make_server, executor=LocalExecutor(),
            db_ref=str(db_path), schema_ddl=sql_schema.schema_ddl(args.db_seed), out_path=out_path,
        )  # fmt: skip
    store.close()
    print(json.dumps({"out": str(out_path), "accuracy": result["accuracy"],
                      "identical_to_base": result["identical_to_base"]}, indent=2))  # fmt: skip
    return 0


if __name__ == "__main__":
    sys.exit(main())
