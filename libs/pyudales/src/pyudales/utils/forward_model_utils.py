"""Utilities for creating and managing ForwardModel instances."""

import copy
import logging
import pathlib

from ..forward_model import ForwardModel
from .config_utils import create_config_sh
from .dir_utils import DirectoryPaths, create_dir
from .file_utils import change_file_extensions, copy_files
from .namoptions_utils import rename_namoptions_file

logger = logging.getLogger(__name__)


def create_new_forward_model(
    forward_model: ForwardModel,
    experiment_base_dir: pathlib.Path,
    experiment_name: str,
) -> ForwardModel:
    """
    Create a new ForwardModel instance with new directories.

    Copies files from the original forward model's experiment_dir to the new experiment_dir,
    updates file names to reflect the new experiment name, and creates a new
    ForwardModel instance with the updated directories.

    Args:
        forward_model: The original ForwardModel instance to copy from.
        experiment_base_dir: The new experiment base directory path.
        experiment_name: The new experiment name.

    Returns:
        A new ForwardModel instance with the updated directories.
    """
    old_experiment_dir = forward_model.dirs.experiment_dir
    old_experiment_name = forward_model.dirs.experiment_name

    cwd = forward_model.dirs.cwd

    # Create new directories
    new_experiment_base_dir = create_dir(cwd / experiment_base_dir)
    new_experiment_dir = create_dir(new_experiment_base_dir / experiment_name)

    # Copy all files from old experiment_dir to new experiment_dir
    if old_experiment_dir.exists():
        copy_files(old_experiment_dir, new_experiment_dir)

    # Create a deep copy of the forward model
    new_forward_model = copy.deepcopy(forward_model)

    # Update directory paths dataclass
    new_forward_model.dirs = DirectoryPaths(
        udales_root_path=forward_model.dirs.udales_root_path,
        cwd=forward_model.dirs.cwd,
        temp_dir=forward_model.dirs.temp_dir,
        experiment_base_dir=new_experiment_base_dir,
        experiment_dir=new_experiment_dir,
        output_dir=forward_model.dirs.output_dir,
        case_dir=forward_model.dirs.case_dir,
        experiment_name=experiment_name,
        results_dir=forward_model.dirs.results_dir,
    )

    # Rename files that reference the old experiment name
    if old_experiment_name != experiment_name:
        change_file_extensions(
            new_forward_model.dirs.experiment_dir, old_experiment_name, experiment_name
        )

    # Update namoptions file to reflect new experiment name
    rename_namoptions_file(new_forward_model.dirs.experiment_dir, experiment_name)

    # Update config.sh file to reflect new directories
    create_config_sh(
        dirs=new_forward_model.dirs,
        matlab_bin=new_forward_model.matlab_bin,
        ncpu=new_forward_model.ncpu,
    )

    logger.info(
        f"Created new ForwardModel: experiment_dir={new_forward_model.dirs.experiment_dir}, "
        f"output_dir={new_forward_model.dirs.output_dir}, "
        f"experiment_name={new_forward_model.dirs.experiment_name}"
    )

    return new_forward_model
