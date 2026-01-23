"""Utilities for validating and synchronizing NCPU with namoptions settings."""

import logging

from .dir_utils import DirectoryPaths
from .namoptions_utils import NamoptionsFile

logger = logging.getLogger(__name__)


def validate_and_sync_ncpu(
    dirs: DirectoryPaths,
    ncpu: int,
) -> int:
    """
    Validate and synchronize NCPU with nprocx * nprocy from namoptions.

    uDALES requires: nprocx * nprocy = NCPU
    Also validates divisibility constraints:
    - itot must be divisible by nprocx
    - jtot must be divisible by nprocy
    - ktot must be divisible by nprocy

    Args:
        temp_dir: The temporary directory containing the namoptions file.
        experiment_name: The experiment name (used in namoptions filename).
        ncpu: The current NCPU value.

    Returns:
        The validated/synchronized NCPU value (may be updated to match nprocx * nprocy).

    Raises:
        ValueError: If divisibility constraints are not met.
    """
    namoptions_path = dirs.temp_dir / f"namoptions.{dirs.experiment_name}"

    if not namoptions_path.exists():
        logger.warning(
            f"namoptions.{dirs.experiment_name} not found, skipping NCPU validation"
        )
        return ncpu

    # Read namoptions to get nprocx, nprocy, itot, jtot, ktot
    namoptions = NamoptionsFile(namoptions_path)

    nprocx = namoptions.get_value_as_int("RUN", "nprocx")
    nprocy = namoptions.get_value_as_int("RUN", "nprocy")
    itot = namoptions.get_value_as_int("DOMAIN", "itot")
    jtot = namoptions.get_value_as_int("DOMAIN", "jtot")
    ktot = namoptions.get_value_as_int("DOMAIN", "ktot")

    # Validate and sync
    if nprocx is not None and nprocy is not None:
        expected_ncpu = nprocx * nprocy

        # Check divisibility constraints
        if itot is not None and itot % nprocx != 0:
            raise ValueError(
                f"itot ({itot}) must be divisible by nprocx ({nprocx}). "
                f"Please adjust nprocx or itot in namoptions.{dirs.experiment_name}"
            )

        if jtot is not None and jtot % nprocy != 0:
            raise ValueError(
                f"jtot ({jtot}) must be divisible by nprocy ({nprocy}). "
                f"Please adjust nprocy or jtot in namoptions.{dirs.experiment_name}"
            )

        if ktot is not None and ktot % nprocy != 0:
            raise ValueError(
                f"ktot ({ktot}) must be divisible by nprocy ({nprocy}). "
                f"Please adjust nprocy or ktot in namoptions.{dirs.experiment_name}"
            )

        # Check if NCPU matches nprocx * nprocy
        if ncpu != expected_ncpu:
            logger.warning(
                f"NCPU ({ncpu}) does not match nprocx * nprocy ({nprocx} * {nprocy} = {expected_ncpu}). "
                f"Updating NCPU to {expected_ncpu} to match namoptions."
            )
            return expected_ncpu
    else:
        logger.warning(
            f"Could not read nprocx and/or nprocy from namoptions.{dirs.experiment_name}. "
            f"Using NCPU={ncpu} as specified."
        )

    return ncpu
