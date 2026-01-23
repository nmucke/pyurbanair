"""Utilities for parsing and editing namoptions files."""

import logging
import pathlib
from typing import Optional

logger = logging.getLogger(__name__)


class NamoptionsFile:
    """
    A parser and editor for namoptions files.

    This class reads a namoptions file, parses it into sections, and allows
    updating values while preserving formatting, comments, and structure.
    """

    def __init__(self, file_path: pathlib.Path) -> None:
        """
        Initialize the NamoptionsFile by reading and parsing the file.

        Args:
            file_path: Path to the namoptions file.
        """
        self.file_path = pathlib.Path(file_path)
        self.sections: dict[str, dict[str, str]] = {}
        self.section_order: list[str] = []
        self.raw_lines: list[str] = []
        self._parse_file()

    def _parse_file(self) -> None:
        """Parse the namoptions file into sections."""
        if not self.file_path.exists():
            logger.warning(f"Namoptions file {self.file_path} does not exist")
            return

        with open(self.file_path, "r") as f:
            self.raw_lines = f.readlines()

        current_section: Optional[str] = None
        current_section_data: dict[str, str] = {}

        for line in self.raw_lines:
            stripped = line.strip()

            # Check if we're entering a new section
            if stripped.startswith("&") and not stripped.startswith("&/"):
                # Save previous section if it exists
                if current_section is not None:
                    self.sections[current_section] = current_section_data
                    if current_section not in self.section_order:
                        self.section_order.append(current_section)

                # Start new section
                current_section = stripped.lstrip("&")
                current_section_data = {}
            elif stripped.startswith("/") and current_section is not None:
                # End of current section
                self.sections[current_section] = current_section_data
                if current_section not in self.section_order:
                    self.section_order.append(current_section)
                current_section = None
                current_section_data = {}
            elif (
                current_section is not None
                and "=" in stripped
                and not stripped.startswith("#")
            ):
                # Parse key-value pair
                parts = stripped.split("=", 1)
                if len(parts) == 2:
                    key = parts[0].strip()
                    value = parts[1].strip()
                    current_section_data[key] = value

        # Handle case where file ends without closing section
        if current_section is not None:
            self.sections[current_section] = current_section_data
            if current_section not in self.section_order:
                self.section_order.append(current_section)

    def get_value(self, section: str, key: str) -> Optional[str]:
        """
        Get a value from a section.

        Args:
            section: Section name (without & prefix).
            key: Key name.

        Returns:
            The value as a string, or None if not found.
        """
        if section not in self.sections:
            return None
        return self.sections[section].get(key)

    def get_value_as_float(self, section: str, key: str) -> Optional[float]:
        """
        Get a value from a section as a float.

        Args:
            section: Section name (without & prefix).
            key: Key name.

        Returns:
            The value as a float, or None if not found or cannot be converted.
        """
        value = self.get_value(section, key)
        if value is None:
            return None
        try:
            # Remove trailing period if present (e.g., "5." -> "5")
            cleaned = value.rstrip(".")
            return float(cleaned)
        except ValueError:
            return None

    def get_value_as_int(self, section: str, key: str) -> Optional[int]:
        """
        Get a value from a section as an int.

        Args:
            section: Section name (without & prefix).
            key: Key name.

        Returns:
            The value as an int, or None if not found or cannot be converted.
        """
        value = self.get_value(section, key)
        if value is None:
            return None
        try:
            # Remove trailing period if present (e.g., "5." -> "5")
            cleaned = value.rstrip(".")
            return int(cleaned)
        except ValueError:
            return None

    def set_value(self, section: str, key: str, value: str | float | int) -> None:
        """
        Set a value in a section.

        Args:
            section: Section name (without & prefix).
            key: Key name.
            value: Value to set (will be converted to string).
        """
        if section not in self.sections:
            self.sections[section] = {}
            if section not in self.section_order:
                self.section_order.append(section)

        self.sections[section][key] = str(value)

    def has_section(self, section: str) -> bool:
        """
        Check if a section exists.

        Args:
            section: Section name (without & prefix).

        Returns:
            True if section exists, False otherwise.
        """
        return section in self.sections

    def get_section_keys(self, section: str) -> list[str]:
        """
        Get all keys in a section.

        Args:
            section: Section name (without & prefix).

        Returns:
            List of keys in the section, or empty list if section doesn't exist.
        """
        if section not in self.sections:
            return []
        return list(self.sections[section].keys())

    def write(self, file_path: Optional[pathlib.Path] = None) -> None:
        """
        Write the namoptions file back to disk.

        Args:
            file_path: Optional path to write to. If None, writes to original path.
        """
        output_path = file_path if file_path is not None else self.file_path

        # Rebuild the file content
        output_lines = []
        current_section: Optional[str] = None
        in_section = False
        keys_written_in_section: set[str] = set()

        for i, line in enumerate(self.raw_lines):
            stripped = line.strip()

            # Check if we're entering a section
            if stripped.startswith("&") and not stripped.startswith("&/"):
                section_name = stripped.lstrip("&")
                current_section = section_name
                in_section = True
                keys_written_in_section = set()
                output_lines.append(line)
            elif (
                stripped.startswith("/") and in_section and current_section is not None
            ):
                # End of section - add any new keys before closing
                if current_section in self.sections:
                    for key, value in self.sections[current_section].items():
                        if key not in keys_written_in_section:
                            # Format: key padded to ~12 chars, then =, then value
                            output_lines.append(f"{key:<12} = {value}\n")

                output_lines.append(line)
                in_section = False
                current_section = None
                keys_written_in_section = set()
            elif (
                in_section
                and current_section is not None
                and "=" in stripped
                and not stripped.startswith("#")
            ):
                # Update existing key-value pair
                parts = stripped.split("=", 1)
                if len(parts) == 2:
                    key = parts[0].strip()
                    if (
                        current_section in self.sections
                        and key in self.sections[current_section]
                    ):
                        # Replace with updated value
                        value = self.sections[current_section][key]
                        # Preserve original formatting style (key padded to ~12 chars)
                        output_lines.append(f"{key:<12} = {value}\n")
                        keys_written_in_section.add(key)
                    else:
                        # Keep original line (key not in our updates)
                        output_lines.append(line)
                        keys_written_in_section.add(key)
                else:
                    output_lines.append(line)
            else:
                # Keep original line (comments, empty lines, etc.)
                output_lines.append(line)

        # Add any sections that weren't in the original file
        existing_sections = {
            line.strip().lstrip("&")
            for line in self.raw_lines
            if line.strip().startswith("&") and not line.strip().startswith("&/")
        }
        for section in self.section_order:
            if section not in existing_sections:
                output_lines.append(f"\n&{section}\n")
                for key, value in self.sections[section].items():
                    output_lines.append(f"{key:<12} = {value}\n")
                output_lines.append("/\n")

        # Write to file
        with open(output_path, "w") as f:
            f.writelines(output_lines)

        # Update internal path if we wrote to a different location
        if file_path is not None:
            self.file_path = file_path
            self._parse_file()


def update_namoptions_section(
    file_path: pathlib.Path,
    section: str,
    updates: dict[str, str | float | int],
    create_if_missing: bool = True,
) -> None:
    """
    Convenience function to update multiple values in a namoptions file section.

    Args:
        file_path: Path to the namoptions file.
        section: Section name (without & prefix).
        updates: Dictionary of key-value pairs to update.
        create_if_missing: If True, create the section if it doesn't exist.
    """
    namoptions = NamoptionsFile(file_path)

    if not namoptions.has_section(section) and not create_if_missing:
        logger.warning(f"Section {section} not found in {file_path}")
        return

    for key, value in updates.items():
        namoptions.set_value(section, key, value)

    namoptions.write()


def read_namoptions_value(
    file_path: pathlib.Path,
    section: str,
    key: str,
    value_type: type = str,
) -> Optional[str | float | int]:
    """
    Convenience function to read a single value from a namoptions file.

    Args:
        file_path: Path to the namoptions file.
        section: Section name (without & prefix).
        key: Key name.
        value_type: Type to convert to (str, int, or float).

    Returns:
        The value, converted to the requested type, or None if not found.
    """
    namoptions = NamoptionsFile(file_path)

    if value_type == int:
        return namoptions.get_value_as_int(section, key)
    elif value_type == float:
        return namoptions.get_value_as_float(section, key)
    else:
        return namoptions.get_value(section, key)


def rename_namoptions_file(experiment_dir: pathlib.Path, experiment_name: str) -> None:
    """
    Rename the namoptions file to have the experiment_name as its extension and update iexpnr.

    This function finds any namoptions file in the experiment_dir, renames it to match the
    experiment_name format (namoptions.{experiment_name}), and updates the iexpnr
    value in the &RUN section to match the experiment_name.

    Args:
        experiment_dir: Directory containing the namoptions file(s).
        experiment_name: The experiment name to use for the file extension and iexpnr value.
    """
    namoptions_files = list(experiment_dir.glob("namoptions*"))

    # If no namoptions file found, return early
    if not namoptions_files:
        logger.warning(f"No namoptions file found in {experiment_dir}")
        return

    old_namoptions_path = namoptions_files[0]
    new_namoptions_path = experiment_dir / f"namoptions.{experiment_name}"

    # Only rename if the file doesn't already have the correct name
    if old_namoptions_path != new_namoptions_path:
        old_namoptions_path.rename(new_namoptions_path)
        namoptions_file_path = new_namoptions_path
    else:
        namoptions_file_path = old_namoptions_path

    # Update iexpnr in the namoptions file to match experiment_name
    namoptions = NamoptionsFile(namoptions_file_path)
    namoptions.set_value("RUN", "iexpnr", experiment_name)
    namoptions.write()

    logger.info(
        f"Renamed namoptions file to namoptions.{experiment_name} and updated iexpnr={experiment_name}"
    )
