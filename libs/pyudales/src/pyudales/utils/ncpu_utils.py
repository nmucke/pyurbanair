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
    Derive nprocx/nprocy from NCPU and write them into namoptions.

    uDALES requires: nprocx * nprocy = NCPU. To let callers pick only NCPU,
    this sets nprocx = ncpu and nprocy = 1 (so nprocx * nprocy = ncpu) and
    writes the values back to the namoptions file.

    Validates divisibility constraints before writing:
    - itot must be divisible by nprocx
    - jtot must be divisible by nprocy
    - ktot must be divisible by nprocy

    Args:
        dirs: DirectoryPaths instance containing experiment_dir and experiment_name.
        ncpu: The number of CPUs to use.

    Returns:
        The (unchanged) NCPU value.

    Raises:
        ValueError: If divisibility constraints are not met.
    """
    namoptions_path = dirs.experiment_dir / f"namoptions.{dirs.experiment_name}"

    if not namoptions_path.exists():
        logger.warning(
            f"namoptions.{dirs.experiment_name} not found, skipping NCPU sync"
        )
        return ncpu

    # Derive domain decomposition from NCPU.
    nprocx = ncpu
    nprocy = 1

    # Read namoptions to get itot, jtot, ktot for the divisibility checks.
    namoptions = NamoptionsFile(namoptions_path)

    itot = namoptions.get_value_as_int("DOMAIN", "itot")
    jtot = namoptions.get_value_as_int("DOMAIN", "jtot")
    ktot = namoptions.get_value_as_int("DOMAIN", "ktot")

    # Check divisibility constraints.
    if itot is not None and itot % nprocx != 0:
        raise ValueError(
            f"itot ({itot}) must be divisible by nprocx ({nprocx}). "
            f"Choose an NCPU that divides itot evenly."
        )

    if jtot is not None and jtot % nprocy != 0:
        raise ValueError(
            f"jtot ({jtot}) must be divisible by nprocy ({nprocy})."
        )

    if ktot is not None and ktot % nprocy != 0:
        raise ValueError(
            f"ktot ({ktot}) must be divisible by nprocy ({nprocy})."
        )

    # Write the derived decomposition back to namoptions.
    namoptions.set_value("RUN", "nprocx", nprocx)
    namoptions.set_value("RUN", "nprocy", nprocy)
    namoptions.write()

    logger.info(
        f"Set nprocx={nprocx}, nprocy={nprocy} in namoptions."
        f"{dirs.experiment_name} (NCPU={ncpu})"
    )

    return ncpu
