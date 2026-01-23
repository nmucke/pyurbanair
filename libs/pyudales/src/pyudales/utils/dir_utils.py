"""Directory utilities for ForwardModel."""

import os
import pathlib
from dataclasses import dataclass


def get_project_root(start_path: pathlib.Path | None = None) -> pathlib.Path:
    """
    Get the project root directory by looking for the root pyproject.toml file.

    The project structure is:
        root/
            pyproject.toml
            libs/
                pyudales/
                    pyproject.toml
                    src/
                        pyudales/
                            forward_model.py

    Args:
        start_path: Optional starting path to search from. If None, uses the location
                    of this file as the starting point.

    Returns:
        Path to the project root directory.

    Raises:
        RuntimeError: If the project root cannot be found.
    """
    if start_path is None:
        current = pathlib.Path(__file__).parent
    else:
        current = pathlib.Path(start_path)

    # Walk up the directory tree looking for the root pyproject.toml
    # We want to find the root one, not the pyudales one
    max_depth = 10  # Safety limit to avoid infinite loops
    depth = 0

    while current != current.parent and depth < max_depth:
        # Check if this is the root (has pyproject.toml and libs/ directory)
        if (current / "pyproject.toml").exists() and (current / "libs").exists():
            return current
        current = current.parent
        depth += 1

    raise RuntimeError(f"Could not find project root. Searched up from {current}")


def create_dir(
    dir_path: pathlib.Path,
) -> pathlib.Path:
    """Create a temporary directory in the given directory."""
    os.makedirs(pathlib.Path(dir_path), exist_ok=True)
    return pathlib.Path(dir_path)


@dataclass
class DirectoryPaths:
    """Holds all relevant directory paths for ForwardModel."""

    udales_root_path: pathlib.Path
    cwd: pathlib.Path
    temp_dir: pathlib.Path  # Base temp directory (e.g., {cwd}/.temp)
    experiment_base_dir: pathlib.Path  # Base directory for experiments (e.g., {temp_dir}/experiment)
    experiment_dir: pathlib.Path  # Specific experiment directory (e.g., {experiment_base_dir}/{experiment_name})
    output_dir: pathlib.Path
    case_dir: pathlib.Path  # Original case directory provided by user
    experiment_name: str


def get_udales_directory_paths(
    case_dir: pathlib.Path,
    experiment_name: str,
    udales_root_path: pathlib.Path,
    temp_dir: pathlib.Path | None = None,
    experiment_base_dir: pathlib.Path | None = None,
    cwd: pathlib.Path | None = None,
    output_dir: pathlib.Path | None = None,
) -> DirectoryPaths:
    """
    Create DirectoryPaths instance with provided paths or defaults.

    This function centralizes the logic for determining directory paths for uDALES
    forward model execution. If optional paths are not provided, it uses sensible
    defaults based on the project structure.

    Directory structure:
    - temp_dir: Base temp directory (defaults to {cwd}/.temp)
    - experiment_base_dir: Base directory for experiments (defaults to {temp_dir}/experiment)
    - experiment_dir: Specific experiment directory ({experiment_base_dir}/{experiment_name})

    Args:
        case_dir: The directory containing the original case files (required).
        experiment_name: The name of the experiment (required).
        udales_root_path: The root path to the uDALES code (required).
        temp_dir: Optional base temp directory. If None, uses {cwd}/.temp.
        experiment_base_dir: Optional base directory for experiments. If None, uses {temp_dir}/experiment.
        cwd: Optional current working directory (project root). If None, uses get_project_root().
        output_dir: Optional output directory. If None, uses {cwd}/.temp/outputs.

    Returns:
        DirectoryPaths instance with all paths configured.
    """
    # Determine base directory
    if cwd is None:
        cwd = get_project_root()

    # Base temp directory (e.g., {cwd}/.temp)
    if temp_dir is None:
        temp_dir_path = create_dir(cwd / ".temp")
    else:
        temp_dir_path = create_dir(temp_dir)

    # Base directory for experiments (e.g., {temp_dir}/experiment)
    if experiment_base_dir is None:
        experiment_base_dir_path = create_dir(temp_dir_path / "experiment")
    else:
        experiment_base_dir_path = create_dir(experiment_base_dir)

    # Specific experiment directory (e.g., {experiment_base_dir}/{experiment_name})
    experiment_dir_path = create_dir(experiment_base_dir_path / experiment_name)

    # Output directory where the intermediate udales outputs will be saved
    if output_dir is None:
        output_dir_path = create_dir(cwd / ".temp" / "outputs")
    else:
        output_dir_path = create_dir(output_dir)

    return DirectoryPaths(
        udales_root_path=udales_root_path,
        cwd=cwd,
        temp_dir=temp_dir_path,
        experiment_base_dir=experiment_base_dir_path,
        experiment_dir=experiment_dir_path,
        output_dir=output_dir_path,
        case_dir=case_dir,
        experiment_name=experiment_name,
    )
