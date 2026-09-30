"""Who is driving a run right now.

A run can be driven by the CLI (its own process) while the API server only reads its files. The
API cannot tell that from a crashed run by looking at the database, so the CLI records its process
in ``driver.pid`` for the duration of the run, and the API asks whether that process still exists.
The start time guards against a recycled pid. Single host only, which is where both run.
"""

import os
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

FILE = "driver.pid"


def _start_time(pid: int) -> str:
    """Process start time in clock ticks on Linux; the pid itself elsewhere (weaker guard)."""
    try:
        stat = Path(f"/proc/{pid}/stat").read_text()
    except OSError:
        return "-"
    return stat.rsplit(")", 1)[1].split()[19]  # field 22 (starttime), after the ')' of comm


@contextmanager
def driving(run_dir: Path) -> Iterator[None]:
    run_dir.mkdir(parents=True, exist_ok=True)
    path = run_dir / FILE
    path.write_text(f"{os.getpid()} {_start_time(os.getpid())}")
    try:
        yield
    finally:
        path.unlink(missing_ok=True)


def driver_alive(run_dir: Path) -> bool:
    try:
        pid_s, started = (run_dir / FILE).read_text().split()
        pid = int(pid_s)
    except (OSError, ValueError):
        return False
    if _start_time(pid) != started or started == "-":
        return False
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True
