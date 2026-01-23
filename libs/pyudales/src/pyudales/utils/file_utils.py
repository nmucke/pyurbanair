import os
import pathlib
import shutil


def change_file_extensions(
    directory: pathlib.Path, old_experiment_name: str, new_experiment_name: str
) -> None:
    """Change file extensions from old experiment name to new experiment name.

    Renames all files in the directory that end with .{old_experiment_name}
    to end with .{new_experiment_name}.

    Args:
        directory: Directory containing files to rename.
        old_experiment_name: Old experiment name used as file extension.
        new_experiment_name: New experiment name to use as file extension.
    """
    if old_experiment_name == new_experiment_name:
        return

    directory_path = pathlib.Path(directory)
    if not directory_path.exists() or not directory_path.is_dir():
        return

    # Find and rename files with pattern *.{old_experiment_name}
    for item in directory_path.iterdir():
        if item.is_file() and item.name.endswith(f".{old_experiment_name}"):
            new_name = item.name.replace(
                f".{old_experiment_name}", f".{new_experiment_name}"
            )
            new_path = directory_path / new_name
            item.rename(new_path)


def copy_files(source_dir: pathlib.Path, target_dir: pathlib.Path) -> None:
    """Copy files from source_dir to target_dir."""
    source_path = pathlib.Path(source_dir)
    target_path = pathlib.Path(target_dir)
    for item in source_path.iterdir():
        target = target_path / item.name
        if item.is_file():
            # Remove target if it exists, then copy
            if target.exists():
                target.unlink()
            shutil.copy2(str(item), str(target))
        elif item.is_dir():
            # Remove target if it exists, then copy
            if target.exists():
                shutil.rmtree(target)
            shutil.copytree(item, target)
