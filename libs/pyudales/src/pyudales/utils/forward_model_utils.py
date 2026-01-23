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
    temp_dir: pathlib.Path,
    output_dir: pathlib.Path,
    experiment_name: str,
) -> ForwardModel:
    """
    Create a new ForwardModel instance with new directories.

    Copies files from the original forward model's temp_dir to the new temp_dir,
    updates file names to reflect the new experiment name, and creates a new
    ForwardModel instance with the updated directories.

    Args:
        forward_model: The original ForwardModel instance to copy from.
        temp_dir: The new temporary directory path.
        output_dir: The new output directory path.
        experiment_name: The new experiment name.

    Returns:
        A new ForwardModel instance with the updated directories.
    """
    old_temp_dir = forward_model.dirs.temp_dir
    old_experiment_name = forward_model.dirs.experiment_name

    # Create new directories
    new_temp_dir = create_dir(temp_dir)
    new_output_dir = create_dir(output_dir)

    # Copy all files from old temp_dir to new temp_dir
    if old_temp_dir.exists():
        copy_files(old_temp_dir, new_temp_dir)

    # Create a deep copy of the forward model
    new_forward_model = copy.deepcopy(forward_model)

    # Update directory paths dataclass
    new_forward_model.dirs = DirectoryPaths(
        udales_root_path=forward_model.dirs.udales_root_path,
        cwd=forward_model.dirs.cwd,
        temp_dir=new_temp_dir,
        output_dir=new_output_dir,
        experiment_dir=forward_model.dirs.experiment_dir,
        experiment_name=experiment_name,
    )

    # Rename files that reference the old experiment name
    if old_experiment_name != experiment_name:
        change_file_extensions(
            new_forward_model.dirs.temp_dir, old_experiment_name, experiment_name
        )

    # Update namoptions file to reflect new experiment name
    rename_namoptions_file(new_forward_model.dirs.temp_dir, experiment_name)

    # Update config.sh file to reflect new directories
    create_config_sh(
        dirs=new_forward_model.dirs,
        matlab_bin=new_forward_model.matlab_bin,
        ncpu=new_forward_model.ncpu,
    )

    logger.info(
        f"Created new ForwardModel: temp_dir={new_forward_model.dirs.temp_dir}, "
        f"output_dir={new_forward_model.dirs.output_dir}, "
        f"experiment_name={new_forward_model.dirs.experiment_name}"
    )

    return new_forward_model
