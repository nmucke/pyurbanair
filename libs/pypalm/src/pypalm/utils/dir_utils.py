"""Directory utilities for the pypalm ForwardModel."""

import os
import pathlib
from dataclasses import dataclass
from typing import Optional


def get_project_root(start_path: pathlib.Path | None = None) -> pathlib.Path:
    """Walk up from ``start_path`` (or this file) to find the workspace root.

    The workspace root is identified by the presence of both ``pyproject.toml``
    and a ``libs/`` directory.
    """
    current = pathlib.Path(start_path) if start_path else pathlib.Path(__file__).parent

    max_depth = 10
    depth = 0
    while current != current.parent and depth < max_depth:
        if (current / "pyproject.toml").exists() and (current / "libs").exists():
            return current
        current = current.parent
        depth += 1

    raise RuntimeError(f"Could not find project root. Searched up from {current}")


def create_dir(dir_path: pathlib.Path) -> pathlib.Path:
    os.makedirs(pathlib.Path(dir_path), exist_ok=True)
    return pathlib.Path(dir_path)


@dataclass
class PALMDirectoryPaths:
    """All directories referenced by the PALM ForwardModel.

    PALM expects a JOBS directory layout of the form::

        experiment_dir/<experiment_name>/INPUT/<experiment_name>_p3d
        experiment_dir/<experiment_name>/INPUT/<experiment_name>_topo
        experiment_dir/<experiment_name>/OUTPUT/<experiment_name>_3d.nc
        experiment_dir/<experiment_name>/MONITORING/
        experiment_dir/<experiment_name>/RESTART/

    We stage that layout under ``experiment_base_dir / experiment_name``; the
    PALM run is invoked with that path as the JOBS root.
    """

    cwd: pathlib.Path
    temp_dir: pathlib.Path
    experiment_base_dir: pathlib.Path
    experiment_dir: pathlib.Path
    input_dir: pathlib.Path
    output_dir: pathlib.Path
    monitoring_dir: pathlib.Path
    restart_dir: pathlib.Path
    case_dir: pathlib.Path
    experiment_name: str
    results_dir: Optional[pathlib.Path] = None


def get_palm_directory_paths(
    case_dir: pathlib.Path,
    experiment_name: str,
    temp_dir: pathlib.Path | None = None,
    experiment_base_dir: pathlib.Path | None = None,
    cwd: pathlib.Path | None = None,
    results_dir: pathlib.Path | None = None,
) -> PALMDirectoryPaths:
    if cwd is None:
        cwd = get_project_root()

    temp = create_dir(temp_dir) if temp_dir is not None else create_dir(cwd / ".temp")

    if experiment_base_dir is None:
        experiment_base = create_dir(temp / "palm_experiment")
    else:
        experiment_base = create_dir(experiment_base_dir)

    experiment = create_dir(experiment_base / experiment_name)
    input_dir = create_dir(experiment / "INPUT")
    output_dir = create_dir(experiment / "OUTPUT")
    monitoring_dir = create_dir(experiment / "MONITORING")
    restart_dir = create_dir(experiment / "RESTART")

    results = create_dir(results_dir) if results_dir is not None else None

    return PALMDirectoryPaths(
        cwd=cwd,
        temp_dir=temp,
        experiment_base_dir=experiment_base,
        experiment_dir=experiment,
        input_dir=input_dir,
        output_dir=output_dir,
        monitoring_dir=monitoring_dir,
        restart_dir=restart_dir,
        case_dir=pathlib.Path(case_dir),
        experiment_name=experiment_name,
        results_dir=results,
    )
