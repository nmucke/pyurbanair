"""Utilities for handling random initial condition settings for uDALES."""

import logging

from .dir_utils import DirectoryPaths
from .namoptions_utils import NamoptionsFile

logger = logging.getLogger(__name__)


def apply_random_initial_condition(
    dirs: DirectoryPaths,
    random_initial_condition_args: dict,
) -> None:
    """
    Apply random initial condition settings to the namoptions file.

    Sets the random initial condition parameters in the &RUN section of the
    namoptions file. The parameters irandom, randqt, randthl, and randu are
    provided by the user in the random_initial_condition_args dict.
    Additionally, lrandomize is always set to .true.

    Args:
        dirs: DirectoryPaths instance containing experiment_dir and experiment_name.
        random_initial_condition_args: Dictionary containing the random initial
            condition parameters:
            - irandom: Random seed (e.g., 43)
            - randqt: Random perturbation for specific humidity (e.g., 2.5e-5)
            - randthl: Random perturbation for liquid water potential temperature (e.g., 0.001)
            - randu: Random perturbation for velocity (e.g., 0.01)
    """
    namoptions_path = dirs.experiment_dir / f"namoptions.{dirs.experiment_name}"

    if not namoptions_path.exists():
        logger.warning(
            f"namoptions.{dirs.experiment_name} not found, "
            "skipping random initial condition update"
        )
        return

    namoptions = NamoptionsFile(namoptions_path)

    # Set the random initial condition parameters from the dict
    if "irandom" in random_initial_condition_args:
        namoptions.set_value("RUN", "irandom", random_initial_condition_args["irandom"])

    if "randqt" in random_initial_condition_args:
        namoptions.set_value("RUN", "randqt", random_initial_condition_args["randqt"])

    if "randthl" in random_initial_condition_args:
        namoptions.set_value("RUN", "randthl", random_initial_condition_args["randthl"])

    if "randu" in random_initial_condition_args:
        namoptions.set_value("RUN", "randu", random_initial_condition_args["randu"])

    # Always set lrandomize to .true.
    namoptions.set_value("RUN", "lrandomize", ".true.")

    namoptions.write()

    logger.info(
        f"Applied random initial condition settings to namoptions.{dirs.experiment_name}"
    )

