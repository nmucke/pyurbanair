"""Utilities for handling save frequency settings for uDALES."""

import logging

from .dir_utils import DirectoryPaths
from .namoptions_utils import NamoptionsFile

logger = logging.getLogger(__name__)


def apply_save_only_last_timestep(dirs: DirectoryPaths) -> None:
    """
    Apply the save_only_last_timestep flag.

    Sets tfielddump = runtime in the namoptions file.

    Args:
        dirs: DirectoryPaths instance containing temp_dir and experiment_name.
    """
    namoptions_path = dirs.temp_dir / f"namoptions.{dirs.experiment_name}"

    namoptions = NamoptionsFile(namoptions_path)

    # Get runtime value from &RUN section
    runtime_value = namoptions.get_value("RUN", "runtime")

    if runtime_value is not None:
        # Clean up the runtime value (remove trailing period if present)
        runtime_clean = runtime_value.rstrip(".")
        if "." in runtime_value:
            runtime_clean = runtime_value

        # Set tfielddump to runtime in &OUTPUT section
        namoptions.set_value("OUTPUT", "tfielddump", runtime_clean)
        namoptions.write()


def apply_output_frequency(
    dirs: DirectoryPaths,
    output_frequency: float,
) -> None:
    """
    Apply the output frequency.

    Sets tfielddump to the specified output frequency in the namoptions file.

    Args:
        dirs: DirectoryPaths instance containing temp_dir and experiment_name.
        output_frequency: The frequency at which the output will be saved.
    """
    namoptions_path = dirs.temp_dir / f"namoptions.{dirs.experiment_name}"

    if not namoptions_path.exists():
        logger.warning(
            f"namoptions.{dirs.experiment_name} not found, skipping output frequency update"
        )
        return

    # Update namoptions file using NamoptionsFile
    namoptions = NamoptionsFile(namoptions_path)
    namoptions.set_value("OUTPUT", "tfielddump", output_frequency)
    namoptions.write()

    logger.info(
        f"Updated tfielddump to {output_frequency} in namoptions.{dirs.experiment_name}"
    )
