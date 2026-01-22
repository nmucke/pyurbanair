import os
import pathlib
import shutil


def rename_namoptions_file(temp_dir: pathlib.Path, experiment_name: str) -> None:
    """Rename the namoptions file to have the experiment_name as its extension and update iexpnr."""
    namoptions_files = list(temp_dir.glob("namoptions*"))

    # If no namoptions file found, return early
    if not namoptions_files:
        return

    old_namoptions_path = namoptions_files[0]
    new_namoptions_path = temp_dir / f"namoptions.{experiment_name}"

    # Only rename if the file doesn't already have the correct name
    if old_namoptions_path != new_namoptions_path:
        old_namoptions_path.rename(new_namoptions_path)
        namoptions_file_path = new_namoptions_path
    else:
        namoptions_file_path = old_namoptions_path

    # Update iexpnr in the namoptions file to match experiment_name
    lines = []
    with open(namoptions_file_path, "r") as f:
        lines = f.readlines()

    output_lines = []
    in_run = False
    iexpnr_updated = False

    for line in lines:
        stripped = line.strip()

        # Check if we're entering or leaving the &RUN section
        if stripped.startswith("&RUN"):
            in_run = True
            output_lines.append(line)
        elif stripped.startswith("/") and in_run:
            # We're leaving the &RUN section, add iexpnr if it wasn't found
            if not iexpnr_updated:
                output_lines.append(f"iexpnr       = {experiment_name}\n")
            output_lines.append(line)
            in_run = False
        elif in_run and "iexpnr" in stripped and "=" in stripped:
            # Update the iexpnr line
            output_lines.append(f"iexpnr       = {experiment_name}\n")
            iexpnr_updated = True
        else:
            # Keep the original line
            output_lines.append(line)

    # If &RUN section was not found, add it at the beginning with iexpnr
    if not any(
        stripped.startswith("&RUN") for stripped in [line.strip() for line in lines]
    ):
        output_lines.insert(0, "&RUN\n")
        output_lines.insert(1, f"iexpnr       = {experiment_name}\n")
        output_lines.insert(2, "/\n")

    # Write the updated file
    with open(namoptions_file_path, "w") as f:
        f.writelines(output_lines)


def create_dir(
    dir_path: pathlib.Path,
) -> pathlib.Path:
    """Create a temporary directory in the given directory."""
    os.makedirs(pathlib.Path(dir_path), exist_ok=True)
    return pathlib.Path(dir_path)


def copy_files(source: pathlib.Path, destination: pathlib.Path) -> None:
    """Copy files from source to destination.

    If source is a file, it will be copied to destination (or into destination if destination is a directory).
    If source is a directory:
    - If destination exists and is a directory, all contents of source will be copied into destination.
    - If destination doesn't exist, source will be copied/renamed to destination.

    Args:
        source: Source file or directory to copy from.
        destination: Destination directory or path to copy to.
    """
    source_path = pathlib.Path(source)
    dest_path = pathlib.Path(destination)

    if not source_path.exists():
        return

    # If source is a file, copy it directly
    if source_path.is_file():
        # If destination is an existing directory, copy file into it
        if dest_path.exists() and dest_path.is_dir():
            target = dest_path / source_path.name
        else:
            # Destination is a file path
            target = dest_path

        if target.exists():
            target.unlink()
        shutil.copy2(str(source_path), str(target))
        return

    # If source is a directory
    if source_path.is_dir():
        # If destination exists and is a directory, copy all contents into it
        if dest_path.exists() and dest_path.is_dir():
            for item in source_path.iterdir():
                target = dest_path / item.name
                if target.exists():
                    if target.is_file():
                        target.unlink()
                    elif target.is_dir():
                        shutil.rmtree(target)
                if item.is_file():
                    shutil.copy2(str(item), str(target))
                elif item.is_dir():
                    shutil.copytree(str(item), str(target))
        else:
            # Destination doesn't exist or is a file path - copy the directory itself
            if dest_path.exists():
                if dest_path.is_file():
                    dest_path.unlink()
                elif dest_path.is_dir():
                    shutil.rmtree(dest_path)
            # Ensure parent directory exists
            dest_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copytree(str(source_path), str(dest_path))


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


def move_files_to_temp_dir(
    experiment_dir: pathlib.Path, temp_dir: pathlib.Path
) -> None:
    """Move files from experiment_dir to temp_dir."""
    experiment_path = pathlib.Path(experiment_dir)
    temp_dir_path = temp_dir
    for item in experiment_path.iterdir():
        target = temp_dir_path / item.name
        if item.is_file():
            target.write_bytes(item.read_bytes())
        elif item.is_dir():
            # Remove target if it exists, then copy
            if target.exists():
                shutil.rmtree(target)
            shutil.copytree(item, target)
