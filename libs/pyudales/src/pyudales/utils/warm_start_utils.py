"""Utilities for handling warm start settings for uDALES."""

import logging
import pathlib
import re
import shutil

from .dir_utils import DirectoryPaths
from .namoptions_utils import NamoptionsFile

logger = logging.getLogger(__name__)


def set_trestart(dirs: DirectoryPaths) -> None:
    """
    Set trestart to runtime in the namoptions file.

    Sets trestart = runtime in the &RUN section of the namoptions file.
    This is used for warm restart functionality where the restart time
    should match the runtime of the previous simulation.

    Args:
        dirs: DirectoryPaths instance containing experiment_dir and experiment_name.
    """
    namoptions_path = dirs.experiment_dir / f"namoptions.{dirs.experiment_name}"

    if not namoptions_path.exists():
        logger.warning(
            f"namoptions.{dirs.experiment_name} not found, skipping trestart update"
        )
        return

    namoptions = NamoptionsFile(namoptions_path)

    # Get runtime value from &RUN section
    runtime_value = namoptions.get_value("RUN", "runtime")

    if runtime_value is not None:
        # Clean up the runtime value (remove trailing period if present, but preserve decimals)
        runtime_clean = runtime_value.rstrip(".")
        if "." in runtime_value:
            runtime_clean = runtime_value

        # Set trestart to runtime in &RUN section
        namoptions.set_value("RUN", "trestart", runtime_clean)
        namoptions.write()

        logger.info(
            f"Updated trestart to {runtime_clean} (runtime) in namoptions.{dirs.experiment_name}"
        )
    else:
        logger.warning(
            f"runtime not found in &RUN section of namoptions.{dirs.experiment_name}, "
            "cannot set trestart"
        )


def identify_warmstart_file(
    dirs: DirectoryPaths,
) -> str:
    """
    Identify the warmstart file and return it in x-format.

    Returns the warmstart filename in the format 'initd{timestamp}_xxx_xxx.{experiment_name}'
    where xxx_xxx is a wildcard pattern that matches processor numbers.

    Args:
        dirs: DirectoryPaths instance containing output_dir and experiment_name.

    Returns:
        Warmstart filename in x-format (e.g., 'initd00000440_xxx_xxx.300').

    Raises:
        ValueError: If no warmstart file is found in output_dir/{experiment_name}.
    """
    # Pattern to match and extract timestamp: initd{timestamp}_{proc1}_{proc2}.{experiment_name}
    pattern = re.compile(rf"^initd(\d+)_\d+_\d+\.{re.escape(dirs.experiment_name)}$")
    output_experiment_dir = dirs.output_dir.joinpath(dirs.experiment_name)

    if not output_experiment_dir.exists():
        raise ValueError(
            f"Output experiment directory {output_experiment_dir} does not exist"
        )

    for item in output_experiment_dir.iterdir():
        if item.is_file():
            match = pattern.match(item.name)
            if match:
                timestamp = match.group(1)
                # Return in x-format: initd{timestamp}_xxx_xxx.{experiment_name}
                return f"initd{timestamp}_xxx_xxx.{dirs.experiment_name}"

    raise ValueError(f"No warmstart file found in {output_experiment_dir}")


def set_warm_start(
    dirs: DirectoryPaths,
) -> None:
    """
    Set warm start settings in the namoptions file.

    Looks for warmstart files in output_dir/{experiment_name} (actual files with processor numbers),
    extracts the timestamp, and writes the x-format to namoptions.

    Sets lwarmstart to .true. and startfile to the pattern matching warmstart files.
    The startfile format is 'initd{timestamp}_xxx_xxx.<experiment_name>' where
    xxx_xxx is a wildcard pattern that matches processor numbers.

    Args:
        dirs: DirectoryPaths instance containing experiment_dir, output_dir, and experiment_name.
    """
    namoptions_path = dirs.experiment_dir / f"namoptions.{dirs.experiment_name}"

    if not namoptions_path.exists():
        logger.warning(
            f"namoptions.{dirs.experiment_name} not found, skipping warm start setup"
        )
        return

    namoptions = NamoptionsFile(namoptions_path)

    # Look for actual warmstart files in output_dir/{experiment_name}
    # Pattern to match actual files: initd{timestamp}_{proc1}_{proc2}.{experiment_name}
    pattern = re.compile(rf"^initd(\d+)_\d+_\d+\.{re.escape(dirs.experiment_name)}$")

    output_experiment_dir = dirs.output_dir / dirs.experiment_name

    if not output_experiment_dir.exists():
        logger.warning(
            f"Output experiment directory {output_experiment_dir} does not exist, "
            "cannot find warmstart files"
        )
        return

    # Find a warmstart file to extract the timestamp
    warmstart_file = None
    for item in output_experiment_dir.iterdir():
        if item.is_file():
            match = pattern.match(item.name)
            if match:
                warmstart_file = item.name
                timestamp = match.group(1)
                break

    if warmstart_file is None:
        logger.warning(
            f"No warmstart files found in {output_experiment_dir}, "
            "cannot set startfile"
        )
        return

    # Create x-format startfile value
    startfile_value = f"initd{timestamp}_xxx_xxx.{dirs.experiment_name}"

    # Set lwarmstart to .true.
    namoptions.set_value("RUN", "lwarmstart", ".true.")

    # Set startfile (with quotes as it's a string value)
    namoptions.set_value("RUN", "startfile", f"'{startfile_value}'")

    namoptions.write()

    logger.info(
        f"Set lwarmstart = .true. and startfile = '{startfile_value}' "
        f"in namoptions.{dirs.experiment_name}"
    )


def move_warmstart_files(
    dirs: DirectoryPaths,
    warmstart_dir: pathlib.Path,
) -> None:
    """
    Move warmstart files to the warmstart directory.

    When trestart is enabled, the model generates files in the format:
    'initd00000440_000_000.<experiment_name>'

    This function finds all files matching this pattern in the output_dir
    and moves them to the warmstart_dir.

    Args:
        dirs: DirectoryPaths instance containing output_dir and experiment_name.
        warmstart_dir: Directory where warmstart files should be moved.
    """
    if not dirs.output_dir.exists():
        logger.warning(
            f"Output directory {dirs.output_dir} does not exist, "
            "cannot move warmstart files"
        )
        return

    # Create warmstart directory if it doesn't exist
    warmstart_dir.mkdir(parents=True, exist_ok=True)

    # Pattern: initd followed by digits, then _digits_digits., then experiment_name
    # Example: initd00000440_000_000.300
    pattern = re.compile(rf"^initd\d+_\d+_\d+\.{re.escape(dirs.experiment_name)}$")

    output_experiment_dir = dirs.output_dir.joinpath(dirs.experiment_name)

    moved_files = []
    for item in output_experiment_dir.iterdir():
        if item.is_file() and pattern.match(item.name):
            target_path = warmstart_dir / item.name
            # Remove target if it exists
            if target_path.exists():
                target_path.unlink()
            shutil.move(str(item), str(target_path))
            moved_files.append(item.name)

    if moved_files:
        logger.info(
            f"Moved {len(moved_files)} warmstart file(s) to {warmstart_dir}: "
            f"{', '.join(moved_files)}"
        )
    else:
        logger.debug(
            f"No warmstart files found matching pattern 'initd*_*_*.{dirs.experiment_name}' "
            f"in {dirs.output_dir}"
        )


def clean_output_except_warmstart_files(
    dirs: DirectoryPaths,
) -> None:
    """
    Remove all files in output_dir/{experiment_name} except for warmstart files.

    This function keeps only files matching the warmstart pattern:
    'initd{timestamp}_{proc1}_{proc2}.{experiment_name}'

    All other files and directories in the output experiment directory are removed.

    Args:
        dirs: DirectoryPaths instance containing output_dir and experiment_name.
    """
    output_experiment_dir = dirs.output_dir / dirs.experiment_name

    if not output_experiment_dir.exists():
        logger.debug(
            f"Output experiment directory {output_experiment_dir} does not exist, "
            "nothing to clean"
        )
        return

    # Pattern to match warmstart files: initd{timestamp}_{proc1}_{proc2}.{experiment_name}
    warmstart_pattern = re.compile(
        rf"^initd\d+_\d+_\d+\.{re.escape(dirs.experiment_name)}$"
    )

    removed_files = []
    kept_files = []

    for item in output_experiment_dir.iterdir():
        if item.is_file():
            if warmstart_pattern.match(item.name):
                kept_files.append(item.name)
            else:
                item.unlink(missing_ok=True)
                removed_files.append(item.name)
        elif item.is_dir():
            shutil.rmtree(item)
            removed_files.append(f"{item.name}/ (directory)")

    if removed_files:
        logger.info(
            f"Removed {len(removed_files)} file(s) from {output_experiment_dir}, "
            f"kept {len(kept_files)} warmstart file(s): {', '.join(kept_files) if kept_files else 'none'}"
        )
    else:
        logger.debug(
            f"No files to remove in {output_experiment_dir}, "
            f"kept {len(kept_files)} warmstart file(s)"
        )


def remove_old_warmstart_files(
    dirs: DirectoryPaths,
) -> None:
    """
    Remove old warmstart files, keeping only the newest ones.

    The timestamp following 'initd' in warmstart filenames is always increasing.
    This function finds all warmstart files, identifies the maximum timestamp,
    and removes all files with timestamps less than the maximum.

    Args:
        dirs: DirectoryPaths instance containing output_dir and experiment_name.
    """
    output_experiment_dir = dirs.output_dir / dirs.experiment_name

    if not output_experiment_dir.exists():
        logger.debug(
            f"Output experiment directory {output_experiment_dir} does not exist, "
            "nothing to clean"
        )
        return

    # Pattern to match warmstart files and extract timestamp: initd{timestamp}_{proc1}_{proc2}.{experiment_name}
    warmstart_pattern = re.compile(
        rf"^initd(\d+)_\d+_\d+\.{re.escape(dirs.experiment_name)}$"
    )

    # Find all warmstart files and extract their timestamps
    warmstart_files = []
    for item in output_experiment_dir.iterdir():
        if item.is_file():
            match = warmstart_pattern.match(item.name)
            if match:
                timestamp = int(match.group(1))
                warmstart_files.append((timestamp, item))

    if not warmstart_files:
        logger.debug(f"No warmstart files found in {output_experiment_dir}")
        return

    # Find the maximum timestamp
    max_timestamp = max(timestamp for timestamp, _ in warmstart_files)

    # Remove files with timestamps less than the maximum
    removed_files = []
    kept_files = []

    for timestamp, item in warmstart_files:
        if timestamp < max_timestamp:
            item.unlink(missing_ok=True)
            removed_files.append(item.name)
        else:
            kept_files.append(item.name)

    if removed_files:
        logger.info(
            f"Removed {len(removed_files)} old warmstart file(s) from {output_experiment_dir}, "
            f"kept {len(kept_files)} newest file(s) with timestamp {max_timestamp}"
        )
    else:
        logger.debug(
            f"All warmstart files in {output_experiment_dir} are already the newest "
            f"(timestamp {max_timestamp}), nothing to remove"
        )
