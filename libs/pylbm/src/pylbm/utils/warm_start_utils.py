"""Utilities for handling LBM warmstart/restart files."""

import pathlib
import re
from typing import Optional

from .dir_utils import DirectoryPaths

RESTART_FILE_PATTERN = re.compile(
    r"^(?P<prefix>restart|turbulence|theta|pottemp|tracer)_(?P<tile>\d{4})_(?P<iteration>\d{6})\.uf$"
)
MAIN_RESTART_PATTERN = re.compile(r"^restart_\d{4}_(?P<iteration>\d{6})\.uf$")


def _restart_dir(dirs: DirectoryPaths) -> pathlib.Path:
    """Return the restart directory path."""
    return dirs.experiment_dir / "restart"


def identify_latest_restart_iteration(dirs: DirectoryPaths) -> Optional[int]:
    """
    Find the latest available main restart iteration.

    Returns:
        Latest iteration number if any main restart file exists, otherwise None.
    """
    restart_dir = _restart_dir(dirs)
    if not restart_dir.exists():
        return None

    iterations: list[int] = []
    for path in restart_dir.iterdir():
        if not path.is_file():
            continue
        match = MAIN_RESTART_PATTERN.match(path.name)
        if match is None:
            continue
        iterations.append(int(match.group("iteration")))

    return max(iterations) if iterations else None


def remove_old_restart_files(
    dirs: DirectoryPaths,
    keep_iteration: Optional[int] = None,
) -> None:
    """
    Remove restart files from older iterations.

    If keep_iteration is None, the newest available main restart iteration is kept.
    """
    restart_dir = _restart_dir(dirs)
    if not restart_dir.exists():
        return

    if keep_iteration is None:
        keep_iteration = identify_latest_restart_iteration(dirs)

    if keep_iteration is None:
        return

    for path in restart_dir.iterdir():
        if not path.is_file():
            continue
        match = RESTART_FILE_PATTERN.match(path.name)
        if match is None:
            continue
        iteration = int(match.group("iteration"))
        if iteration < keep_iteration:
            path.unlink(missing_ok=True)


def clean_output_files(dirs: DirectoryPaths) -> None:
    """Remove LBM netCDF output files from the output directory."""
    for output_file in dirs.output_dir.glob("out_*.nc"):
        output_file.unlink(missing_ok=True)
