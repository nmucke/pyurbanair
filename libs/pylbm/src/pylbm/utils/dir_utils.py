import os
import pathlib
from dataclasses import dataclass
from typing import Optional

from pylbm import LBM_PATH
from pylbm.utils.environment_utils import identify_environment


def create_dir(
    dir_path: pathlib.Path,
) -> pathlib.Path:
    """Create a temporary directory in the given directory."""
    os.makedirs(pathlib.Path(dir_path), exist_ok=True)
    return pathlib.Path(dir_path)


@dataclass
class DirectoryPaths:
    """Holds all relevant directory paths for ForwardModel."""

    lbm_src_path: pathlib.Path
    cwd: pathlib.Path
    temp_dir: pathlib.Path  # Base temp directory (e.g., {cwd}/.temp)
    experiment_base_dir: (
        pathlib.Path
    )  # Base directory for experiments (e.g., {temp_dir}/experiment)
    experiment_dir: (
        pathlib.Path
    )  # Specific experiment directory (e.g., {experiment_base_dir}/{experiment_name})
    output_dir: pathlib.Path
    case_dir: pathlib.Path  # Original case directory provided by user
    experiment_name: str
    infile_path: pathlib.Path
    main_f90_path: pathlib.Path
    mod_dimensions_path: pathlib.Path
    executable_path: pathlib.Path
    makefile_path: pathlib.Path
    pixi_env_path: pathlib.Path
    results_dir: Optional[pathlib.Path] = None


def get_lbm_directory_paths(
    temp_dir: pathlib.Path,
    case_dir: pathlib.Path,
    experiment_name: str,
    results_dir: Optional[pathlib.Path] = None,
) -> DirectoryPaths:
    """
    Build DirectoryPaths for the LBM forward model.

    Args:
        lbm_src_path: Path to the LBM src directory (contains makefile, mod_dimensions.F90, main.F90).
        cwd: Current working directory.
        temp_dir: Base temp directory (e.g. {cwd}/.temp).
        experiment_base_dir: Base directory for experiments.
        experiment_dir: Specific experiment directory (e.g. {experiment_base_dir}/{experiment_name}).
        output_dir: Output directory.
        case_dir: Original case directory provided by the user.
        experiment_name: Name of the experiment.
        pixi_env_path: Path to the pixi/conda environment (HOME for build and run).
        results_dir: Optional results directory.

    Returns:
        DirectoryPaths with all paths set (infile, makefile, mod_dimensions, executable, etc.).
    """
    lbm_src_path = LBM_PATH / "src"  # type: ignore[operator]
    main_f90_path = lbm_src_path / "main.F90"
    mod_dimensions_path = lbm_src_path / "mod_dimensions.F90"
    makefile_path = lbm_src_path / "makefile"

    experiment_base_dir = create_dir(temp_dir / "experiment")
    experiment_dir = create_dir(experiment_base_dir / experiment_name)

    output_dir = create_dir(experiment_dir / "output")

    pixi_env_path = identify_environment(pathlib.Path.cwd())

    infile_path = experiment_dir / "infile.in"
    executable_path = pixi_env_path / "bin" / "boltzmann"

    if results_dir is not None:
        results_dir = create_dir(results_dir)

    return DirectoryPaths(
        lbm_src_path=lbm_src_path,
        cwd=pathlib.Path.cwd(),
        temp_dir=temp_dir,
        experiment_base_dir=experiment_base_dir,
        experiment_dir=experiment_dir,
        output_dir=output_dir,
        case_dir=case_dir,
        experiment_name=experiment_name,
        results_dir=results_dir,
        infile_path=infile_path,
        main_f90_path=main_f90_path,
        mod_dimensions_path=mod_dimensions_path,
        executable_path=executable_path,
        makefile_path=makefile_path,
        pixi_env_path=pixi_env_path,
    )
