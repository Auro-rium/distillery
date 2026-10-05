"""C3 proof: the tool-call pack runs end to end in a DRY run (fake models, offline) and the report
carries the integrity evidence (sealed held-out, headroom check, execution-verified teacher data).

usage: python c3_toolcall_dry_run.py OUT.json
Runs ``python -m distillery --root <tmp> run --dry-run --pack toolcall --scale tiny`` and writes a
summary of the resulting report.json. Every number is from FAKE models: structure, not results.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path

CMD = ["run", "--dry-run", "--pack", "toolcall", "--scale", "tiny"]


def main(out: str) -> int:
    with tempfile.TemporaryDirectory() as tmp:
        subprocess.run(
            [sys.executable, "-m", "distillery", "--root", tmp, *CMD],
            check=True, capture_output=True, text=True,
        )  # fmt: skip
        run = Path(tmp) / "dry-runs" / "runs" / "dry-toolcall-tiny"
        rep = json.loads((run / "report.json").read_text())
        sealed = {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (run / "heldout").iterdir()
        }
    d, ev = rep["data"], rep["evaluation"]
    stage = {s["stage"] for s in rep["stages"]}
    summary = {
        "command": "python -m distillery --root <tmp> " + " ".join(CMD),
        "label": rep["label"],
        "pack": rep["pack"],
        "run_id": rep["run_id"],
        "stages_completed": sorted(stage),
        "gate": {
            "decision": rep["decision"],
            "reasons": rep["decision_reasons"],
            "accuracy": ev["accuracy"],
            "n_heldout": d["heldout_tasks"],
        },
        "heldout_sealed_sha256": d["heldout_sealed_sha256"],
        "stress_sealed_sha256": d["stress_sealed_sha256"],
        "sealed_file_sha256_on_disk": sealed,
        "stress_families": d["stress_families"],
        "heldout_class_counts": d["heldout_class_counts"],
        "headroom": rep["headroom"],
        "teacher_data": {
            "verified_rows_round1": d["teacher_verified_rows_round1"],
            "rule": "kept only if replaying the teacher's calls reaches the template gold's final state",
            "paraphrase_verified": rep["paraphrase"]["verified"],
            "paraphrase_failed_verification": rep["paraphrase"]["discards"]["failed_verification"],
        },
        "rounds": rep["rounds"],
        "caveat": "fake models and a local replay executor: proves the wiring and integrity "
        "rules, not model quality or live behaviour (no sandbox executor for tool calls yet)",
    }
    Path(out).write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print("decision", rep["decision"], "->", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1]))
