"""Compare two dry-run report.json files, ignoring timestamps and durations.

usage: python c0_regression_diff.py BASELINE NEW OUT.json
Exit 0 and ``identical: true`` when the reports differ in nothing but time-valued fields.
A field is time-valued when its key looks like a timestamp/duration (``*_at``, ``*seconds*``,
``*duration*``, ``elapsed*``, ``started``, ``finished``, ``time``) or its string value is an
ISO-8601 timestamp. Everything else is compared exactly (floats included).
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any

TIME_KEY = re.compile(r"(_at$|seconds|duration|elapsed|^started|^finished|^time$)", re.I)
ISO = re.compile(r"^\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}")


def diff(a: Any, b: Any, path: str, out: list[dict[str, Any]], ignored: list[str]) -> None:
    if isinstance(a, dict) and isinstance(b, dict):
        for k in sorted(set(a) | set(b)):
            p = f"{path}.{k}"
            if k not in a or k not in b:
                out.append(
                    {"path": p, "baseline": a.get(k, "<missing>"), "new": b.get(k, "<missing>")}
                )
            elif TIME_KEY.search(k) and not isinstance(a[k], (dict, list)):
                ignored.append(p)
            else:
                diff(a[k], b[k], p, out, ignored)
    elif isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            out.append({"path": path, "baseline": f"len {len(a)}", "new": f"len {len(b)}"})
            return
        for i, (x, y) in enumerate(zip(a, b, strict=True)):
            diff(x, y, f"{path}[{i}]", out, ignored)
    elif isinstance(a, str) and isinstance(b, str) and ISO.match(a) and ISO.match(b):
        ignored.append(path)
    elif a != b or type(a) is not type(b):
        out.append({"path": path, "baseline": a, "new": b})


def main(argv: list[str]) -> int:
    base_p, new_p, out_p = argv
    base, new = json.loads(Path(base_p).read_text()), json.loads(Path(new_p).read_text())
    diffs: list[dict[str, Any]] = []
    ignored: list[str] = []
    diff(base, new, "$", diffs, ignored)
    result = {
        "command": "python -m distillery --root <tmp> run --dry-run --run-id dry-base --scale tiny",
        "baseline_sha256": hashlib.sha256(Path(base_p).read_bytes()).hexdigest(),
        "new_sha256": hashlib.sha256(Path(new_p).read_bytes()).hexdigest(),
        "identical": not diffs,
        "differences": diffs,
        "ignored_time_fields": sorted(ignored),
        "n_ignored": len(ignored),
    }
    Path(out_p).write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(
        f"identical={result['identical']} differences={len(diffs)} ignored_time_fields={len(ignored)}"
    )
    return 0 if not diffs else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
