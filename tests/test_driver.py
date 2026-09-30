"""A run driven by the CLI (another process) must read as running while that process lives."""

import os
import subprocess
import sys
from pathlib import Path

from distillery.driver import driver_alive, driving


def test_alive_only_while_the_claiming_process_runs(tmp_path: Path) -> None:
    run_dir = tmp_path / "runs" / "r1"
    assert not driver_alive(run_dir)  # nothing claimed
    with driving(run_dir):
        assert driver_alive(run_dir)  # this process holds it
    assert not driver_alive(run_dir)  # released on exit


def test_released_even_when_the_run_raises(tmp_path: Path) -> None:
    run_dir = tmp_path / "runs" / "r1"
    try:
        with driving(run_dir):
            raise RuntimeError("boom")
    except RuntimeError:
        pass
    assert not driver_alive(run_dir)


def test_a_dead_process_is_not_alive(tmp_path: Path) -> None:
    run_dir = tmp_path / "runs" / "r1"
    run_dir.mkdir(parents=True)
    child = subprocess.Popen([sys.executable, "-c", "pass"])
    child.wait()
    (run_dir / "driver.pid").write_text(f"{child.pid} 1")
    assert not driver_alive(run_dir)


def test_a_reused_pid_with_another_start_time_is_not_alive(tmp_path: Path) -> None:
    run_dir = tmp_path / "runs" / "r1"
    run_dir.mkdir(parents=True)
    (run_dir / "driver.pid").write_text(f"{os.getpid()} 1")  # our pid, wrong start time
    assert not driver_alive(run_dir)


def test_garbage_file_is_not_alive(tmp_path: Path) -> None:
    run_dir = tmp_path / "runs" / "r1"
    run_dir.mkdir(parents=True)
    (run_dir / "driver.pid").write_text("not a pid")
    assert not driver_alive(run_dir)
