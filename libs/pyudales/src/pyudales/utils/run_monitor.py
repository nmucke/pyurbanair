"""Watchdog that kills a uDALES run early when its timestep collapses.

uDALES occasionally goes numerically unstable.  When it does, the adaptive
timestep ``dt`` shrinks toward zero and the run crawls for a long time before
finally crashing (or reaching its end time at a glacial pace).  In an ensemble
that wastes a large amount of wall-clock, because the doomed member holds up the
batch long before the existing failure handling (resample-from-successful-donor)
gets a chance to kick in.

This module runs the executable under a watchdog that tails the ``run.<exp>.log``
file, parses the printed ``dt`` values, and -- if ``dt`` stays below an absolute
floor for a sustained number of steps -- kills the whole process tree (through
the shared ``pyurbanair.utils.solver_process.run_solver``) and raises
``subprocess.CalledProcessError``.  That is exactly the signal the ensemble layer
already treats as a member failure, so the resample path fires unchanged, just
much sooner.
"""

import logging
import re
from dataclasses import dataclass, fields
from pathlib import Path
from typing import IO, Any, Optional

from pyurbanair.utils.solver_process import run_solver

logger = logging.getLogger(__name__)

# Matches the "... dt:  0.242654880" (or "dt:  1.2E-07") field of each uDALES
# step line in run.<exp>.log.
_DT_RE = re.compile(r"\bdt:\s*([0-9.]+(?:[eE][+-]?[0-9]+)?)")


@dataclass
class InstabilityCheck:
    """Configuration for the dt-collapse watchdog.

    Attributes:
        enabled: When False, the run executes with no monitoring.
        min_dt: Absolute timestep floor.  A step whose ``dt`` is below this is
            counted as "low".
        patience: Number of *consecutive* low steps that must occur before the
            run is declared unstable and killed.
        warmup_steps: Ignore the first N logged steps; uDALES legitimately uses
            a small ``dt`` while ramping up at the start of a run.
        poll_interval_s: How often to re-read the log file while the run is live.
    """

    enabled: bool = True
    min_dt: float = 1.0e-6
    patience: int = 40
    warmup_steps: int = 20
    poll_interval_s: float = 2.0

    @classmethod
    def from_config(cls, config: Optional[dict]) -> "InstabilityCheck":
        """Build from an optional config dict, ignoring unknown keys."""
        if config is None:
            return cls()
        known = {f.name for f in fields(cls)}
        unknown = set(config) - known
        if unknown:
            logger.warning(
                "Ignoring unknown instability_check keys: %s", sorted(unknown)
            )
        return cls(**{k: v for k, v in config.items() if k in known})


class _DtWatch:
    """Stateful tracker that trips after ``patience`` consecutive low steps."""

    def __init__(self, check: InstabilityCheck) -> None:
        self.check = check
        self.step_count = 0
        self.consecutive_low = 0

    def update(self, dt: float) -> bool:
        """Feed one ``dt`` value; return True if the collapse criterion trips."""
        self.step_count += 1
        if self.step_count <= self.check.warmup_steps:
            return False
        if dt < self.check.min_dt:
            self.consecutive_low += 1
        else:
            self.consecutive_low = 0
        return self.consecutive_low >= self.check.patience


def _consume_log(
    log_path: Path, offset: int, buffer: bytes, watch: _DtWatch
) -> tuple[int, bytes, bool]:
    """Read new bytes appended to ``log_path`` and feed any dt values to ``watch``.

    Reads in binary and keeps a ``buffer`` of the trailing partial line so a
    half-written step line is not parsed until it is complete.  Returns the new
    ``(offset, buffer, tripped)``.
    """
    if not log_path.exists():
        return offset, buffer, False
    size = log_path.stat().st_size
    if size <= offset:
        return offset, buffer, False

    with open(log_path, "rb") as f:
        f.seek(offset)
        chunk = f.read(size - offset)
    offset = size

    buffer += chunk
    *lines, buffer = buffer.split(b"\n")
    for raw in lines:
        match = _DT_RE.search(raw.decode("utf-8", "replace"))
        if match and watch.update(float(match.group(1))):
            return offset, buffer, True
    return offset, buffer, False


def run_with_dt_watchdog(
    command: list[str],
    env: dict[str, str],
    log_path: Path,
    check: InstabilityCheck,
    stdout: int | IO[Any] | None = None,
    stderr: int | IO[Any] | None = None,
) -> None:
    """Run ``command`` through ``run_solver``, plus the dt watchdog when enabled.

    Raises ``SolverProcessError`` (a ``subprocess.CalledProcessError`` carrying
    the tail of ``log_path``) on a non-zero exit *and* when the timestep
    collapse criterion in ``check`` trips, after killing the process tree.

    Args:
        command: Argument vector to execute.
        env: Environment for the subprocess.
        log_path: Path to the run's ``run.<exp>.log`` (written via ``tee -a``).
        check: Watchdog configuration.
        stdout / stderr: Passed straight through to the subprocess, so the
            caller's ``verbose`` behavior is preserved (the watchdog reads the
            log file, not the pipe).
    """
    if not check.enabled:
        run_solver(command, env=env, stdout=stdout, stderr=stderr, log_path=log_path)
        return

    # The log is reused across warm-start windows (tee -a appends), so only
    # parse lines written by *this* run: start from the current end-of-file.
    offset = log_path.stat().st_size if log_path.exists() else 0
    buffer = b""
    watch = _DtWatch(check)

    def collapsed() -> bool:
        nonlocal offset, buffer
        offset, buffer, tripped = _consume_log(log_path, offset, buffer, watch)
        if tripped:
            logger.warning(
                "uDALES instability detected (%s): dt < %g for %d consecutive "
                "steps. Killing run early so it can be resampled.",
                log_path.name,
                check.min_dt,
                check.patience,
            )
        return tripped

    run_solver(
        command,
        env=env,
        stdout=stdout,
        stderr=stderr,
        log_path=log_path,
        poll=collapsed,
        poll_interval_s=check.poll_interval_s,
    )
