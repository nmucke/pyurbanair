"""Clean up utilities for ForwardModel."""

import pathlib
import shutil

from .dir_utils import DirectoryPaths


def clean_output_dir(dirs: DirectoryPaths, except_for_files: list[str] = []) -> None:
    """Delete the output directory contents (files and subdirectories)."""
    output_experiment_dir = dirs.output_dir.joinpath(dirs.experiment_name)
    if not output_experiment_dir.exists():
        return

    for item in output_experiment_dir.iterdir():
        if item.name in except_for_files:
            continue
        elif item.is_file():
            item.unlink(missing_ok=True)
        elif item.is_dir():
            shutil.rmtree(item)


def clean_temp_dir(dirs: DirectoryPaths) -> None:
    """Clean the temp directory."""
    # Empty the temp directory by removing all its contents
    for item in pathlib.Path(dirs.temp_dir).iterdir():
        name = item.name
        lower_name = name.lower()
        # Exclude config.sh, any namoptions*, and any *.stl (case-insensitive)
        if lower_name == "config.sh":
            continue
        if lower_name.startswith("namoptions"):
            continue
        if lower_name.endswith(".stl"):
            continue
        if item.is_file():
            item.unlink(missing_ok=True)
        elif item.is_dir():
            shutil.rmtree(item)
