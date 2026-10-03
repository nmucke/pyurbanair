"""Run a solver's process tree so that it never outlives its owner.

Every backend (uDALES, LBM, PALM) launches its solver through ``run_solver``.
The command starts in its own process group, so the whole tree (shell ->
mpiexec/prterun -> ranks) can be signalled at once:

- on a normal exit, an exception or a ``poll`` request, the group is killed
  here before returning;
- if this Python process dies without running that cleanup (SIGKILL, an
  unhandled SIGTERM, a broken worker pool), a small ``sh`` lifeline notices
  that the pipe it reads from has closed and kills the group itself.

The MCP job supervisor (``mcp_server.jobs.processes``) adds the job-level layer
on top: it signals every process carrying the job's token.
"""

from __future__ import annotations

import os
import signal
import subprocess
import time
from pathlib import Path
from typing import IO, Any, Callable, Mapping, Optional, Sequence

# Seconds between SIGTERM and SIGKILL when stopping a process group.
GRACE_SECONDS = 5

# Waits for EOF on stdin (the owner exiting), then stops the solver's group.
# `kill -s SIG -- -PGID` is the POSIX form: dash (Ubuntu's sh) rejects
# `kill -SIG -- -PGID`. The first kill fails only when the group is already
# gone (its error still reaches stderr); the group exiting on SIGTERM before
# the SIGKILL is the normal case, so that one is quiet.
_LIFELINE = (
    'read _; kill -s TERM -- "-$1" || exit 0; '
    f'sleep {GRACE_SECONDS}; kill -s KILL -- "-$1" 2>/dev/null; exit 0'
)


class SolverProcessError(subprocess.CalledProcessError):
    """A non-zero solver exit, with the end of the solver's log in the message.

    Still a ``CalledProcessError``, so the ensemble treats it as a member
    failure (``MEMBER_FAILURES``) exactly as before.
    """

    def __str__(self) -> str:
        text = super().__str__()
        return f"{text}\n{self.output}" if self.output else text


def log_tail(path: Optional[Path], lines: int = 40) -> str:
    """Return the last ``lines`` lines of ``path``, or "" when it is missing."""
    if path is None or not Path(path).is_file():
        return ""
    tail = Path(path).read_text(errors="replace").splitlines()[-lines:]
    return f"Last {len(tail)} lines of {path}:\n" + "\n".join(tail)


def kill_process_group(proc: subprocess.Popen) -> None:
    """Stop the process group led by ``proc``: SIGTERM, then SIGKILL."""
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    deadline = time.monotonic() + GRACE_SECONDS
    while proc.poll() is None and time.monotonic() < deadline:
        time.sleep(0.1)
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass


def run_solver(
    command: Sequence[str],
    *,
    env: Mapping[str, str],
    cwd: Optional[Path] = None,
    stdin: int | IO[Any] | None = subprocess.DEVNULL,
    stdout: int | IO[Any] | None = None,
    stderr: int | IO[Any] | None = None,
    log_path: Optional[Path] = None,
    poll: Optional[Callable[[], bool]] = None,
    poll_interval_s: float = 1.0,
) -> None:
    """Run ``command`` to completion like ``subprocess.run(check=True)``.

    Args:
        command: Argument vector to execute.
        env: Environment for the solver.
        cwd: Working directory for the solver (default: the current one).
        stdin / stdout / stderr: Passed to ``subprocess.Popen``.
        log_path: The solver's log; its tail goes into the error on failure.
        poll: Called every ``poll_interval_s`` while the solver runs; returning
            True stops the run, which is then reported as failed.
        poll_interval_s: Seconds between ``poll`` calls.

    Raises:
        SolverProcessError: On a non-zero exit or when ``poll`` stopped it.
    """
    proc = subprocess.Popen(
        list(command),
        env=dict(env),
        cwd=cwd,
        stdin=stdin,
        stdout=stdout,
        stderr=stderr,
        start_new_session=True,
    )
    # Only this process holds the write end (os.pipe fds are not inherited),
    # so it closes exactly when this process exits, however it exits.
    read_end, write_end = os.pipe()
    lifeline = subprocess.Popen(
        ["sh", "-c", _LIFELINE, "lifeline", str(proc.pid)],
        stdin=read_end,
        stdout=subprocess.DEVNULL,
    )
    os.close(read_end)
    stopped = False
    try:
        if poll is None:
            proc.wait()
        else:
            while proc.poll() is None:
                if poll():
                    stopped = True
                    break
                time.sleep(poll_interval_s)
    finally:
        # Also reaps ranks left behind by a launcher that exited first.
        kill_process_group(proc)
        proc.wait()
        lifeline.kill()
        lifeline.wait()
        os.close(write_end)
    if stopped or proc.returncode != 0:
        raise SolverProcessError(
            proc.returncode, list(command), output=log_tail(log_path)
        )
