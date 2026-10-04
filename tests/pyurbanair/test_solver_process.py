"""The shared solver launcher: errors carry the log, no process outlives its owner."""

import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

from pyurbanair.utils.solver_process import (
    SolverProcessError,
    kill_process_group,
    run_solver,
)


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    # A zombie still answers kill(0); it is dead for our purposes.
    status = subprocess.run(
        ["ps", "-o", "stat=", "-p", str(pid)], capture_output=True, text=True
    ).stdout.strip()
    return bool(status) and not status.startswith("Z")


def _wait_for(path: Path, timeout: float = 30.0) -> int:
    deadline = time.monotonic() + timeout
    while not (path.exists() and path.read_text().strip()):
        assert time.monotonic() < deadline, f"{path} was never written"
        time.sleep(0.05)
    return int(path.read_text())


def _wait_dead(pid: int, timeout: float = 15.0) -> None:
    deadline = time.monotonic() + timeout
    while _alive(pid):
        assert time.monotonic() < deadline, f"process {pid} outlived its owner"
        time.sleep(0.1)


def test_failure_carries_the_log_tail(tmp_path: Path) -> None:
    log = tmp_path / "run.log"
    log.write_text("".join(f"step {i}\n" for i in range(100)) + "dt collapsed\n")
    with pytest.raises(subprocess.CalledProcessError) as error:
        run_solver(["sh", "-c", "exit 3"], env=os.environ, log_path=log)
    assert isinstance(error.value, SolverProcessError)
    assert error.value.returncode == 3
    assert "dt collapsed" in str(error.value)
    assert "step 0\n" not in str(error.value)


def test_group_of_zombies_counts_as_gone() -> None:
    # macOS: killpg on a group holding only unreaped zombies raises EPERM.
    proc = subprocess.Popen(["true"], start_new_session=True)
    _wait_dead(proc.pid)  # exited, not yet reaped
    kill_process_group(proc)
    proc.wait()


def test_poll_stops_the_whole_tree(tmp_path: Path) -> None:
    pidfile = tmp_path / "grandchild"
    command = ["sh", "-c", f"sleep 300 & echo $! > {pidfile}; wait"]
    with pytest.raises(SolverProcessError):
        run_solver(command, env=os.environ, poll=pidfile.exists, poll_interval_s=0.05)
    _wait_dead(int(pidfile.read_text()))


@pytest.mark.parametrize("signum", [signal.SIGTERM, signal.SIGKILL])
def test_solver_dies_with_its_owner(tmp_path: Path, signum: int) -> None:
    pidfile = tmp_path / "solver"
    owner = subprocess.Popen(
        [
            sys.executable,
            "-c",
            "import os, sys\n"
            "from pyurbanair.utils.solver_process import run_solver\n"
            "run_solver(['sh', '-c', 'echo $$ > ' + sys.argv[1] + '; exec sleep 300'],"
            " env=os.environ)\n",
            str(pidfile),
        ]
    )
    solver = _wait_for(pidfile)
    owner.send_signal(signum)
    owner.wait()
    _wait_dead(solver)
