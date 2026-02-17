"""Utilities for creating and managing ForwardModel instances."""

import copy
import logging
import pathlib
import shutil
from typing import TYPE_CHECKING

from pylbm.utils.dir_utils import create_dir, get_lbm_directory_paths
from pylbm.utils.infile_utils import Infile

if TYPE_CHECKING:
    from pylbm.forward_model import ForwardModel

logger = logging.getLogger(__name__)


def create_new_forward_model(
    forward_model: "ForwardModel",
    experiment_base_dir: pathlib.Path,
    experiment_name: str,
) -> "ForwardModel":
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
    # Import here to avoid circular import
    from pylbm.forward_model import ForwardModel  # noqa: F811

    old_experiment_dir = forward_model.dirs.experiment_dir
    old_experiment_name = forward_model.dirs.experiment_name

    cwd = forward_model.dirs.cwd

    # Create new directories
    new_experiment_base_dir = create_dir(cwd / experiment_base_dir)
    new_experiment_dir = create_dir(new_experiment_base_dir / experiment_name)

    # Copy all files from old experiment_dir to new experiment_dir
    if old_experiment_dir.exists():
        # Copy all files and subdirectories
        for item in old_experiment_dir.iterdir():
            dest = new_experiment_dir / item.name
            if item.is_dir():
                shutil.copytree(item, dest, dirs_exist_ok=True)
            else:
                shutil.copy2(item, dest)

    # Create a deep copy of the forward model
    new_forward_model = copy.deepcopy(forward_model)

    # Update directory paths
    new_forward_model.dirs = get_lbm_directory_paths(
        temp_dir=forward_model.dirs.temp_dir,
        case_dir=forward_model.dirs.case_dir,
        experiment_name=experiment_name,
        experiment_base_dir=new_experiment_base_dir,
        results_dir=forward_model.dirs.results_dir,
    )

    logger.info(
        f"Created new ForwardModel: experiment_dir={new_forward_model.dirs.experiment_dir}, "
        f"output_dir={new_forward_model.dirs.output_dir}, "
        f"experiment_name={new_forward_model.dirs.experiment_name}"
    )

    return new_forward_model
