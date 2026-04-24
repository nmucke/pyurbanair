"""Clean-up utilities for the pypalm ForwardModel."""

import pathlib
import shutil

from .dir_utils import PALMDirectoryPaths


def clean_palm_output_dir(dirs: PALMDirectoryPaths) -> None:
    """Remove PALM run artefacts so the next run starts clean.

    Clears ``OUTPUT`` (NetCDF + ASCII diagnostics), ``MONITORING`` (RUN_CONTROL
    etc.) and ``RESTART`` (binary restart files). Leaves ``INPUT`` intact so
    the _p3d / _topo files produced at __init__ are reused.
    """
    for root in (dirs.output_dir, dirs.monitoring_dir, dirs.restart_dir):
        if not root.exists():
            continue
        for item in root.iterdir():
            if item.is_file() or item.is_symlink():
                item.unlink(missing_ok=True)
            elif item.is_dir():
                shutil.rmtree(item)


def clean_palm_input_dir(dirs: PALMDirectoryPaths, keep_suffixes: tuple[str, ...] = ("_p3d", "_topo")) -> None:
    """Remove generated input files, keeping the canonical _p3d / _topo pair."""
    if not dirs.input_dir.exists():
        return
    for item in dirs.input_dir.iterdir():
        if not item.is_file():
            continue
        if any(item.name.endswith(suffix) for suffix in keep_suffixes):
            continue
        item.unlink(missing_ok=True)
