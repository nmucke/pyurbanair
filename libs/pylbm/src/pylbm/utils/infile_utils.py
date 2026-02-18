"""Utilities for parsing and editing LBM infile.in files."""

import logging
import os
import pathlib
import re
import subprocess
import sys
from typing import TYPE_CHECKING, Optional, Union

if TYPE_CHECKING:
    from .dir_utils import DirectoryPaths

logger = logging.getLogger(__name__)


class Infile:
    """
    A parser and editor for LBM infile.in files.

    Each line is typically: "value(s)  ! key : description".
    Values can be single (T/F, number) or multiple tokens (e.g. "256 1 1").
    This class allows reading and updating values by key while preserving
    comments and structure.
    """

    def __init__(self, file_path: pathlib.Path) -> None:
        """
        Initialize the Infile by reading and parsing the file.

        Args:
            file_path: Path to the infile.in file.
        """
        self.file_path = pathlib.Path(file_path)
        self._key_to_value: dict[str, str] = {}
        self._key_to_line_index: dict[str, int] = {}
        self.raw_lines: list[str] = []

        if self.file_path.exists():
            self._parse_file()

    def _parse_file(self) -> None:
        """Parse the infile into key-value pairs (key = first word after !, before :)."""
        if not self.file_path.exists():
            return

        with open(self.file_path, "r") as f:
            self.raw_lines = f.readlines()

        # Pattern: optional whitespace, value part, then "! key :" or "! key:"
        # Key is the first token after "!" until ":" (or end of line).
        for i, line in enumerate(self.raw_lines):
            if "!" not in line:
                continue
            before_excl, _, after_excl = line.partition("!")
            if ":" in after_excl:
                key_part, _, _ = after_excl.partition(":")
                key = key_part.strip().split()[0] if key_part.strip() else None
            else:
                key = after_excl.strip().split()[0] if after_excl.strip() else None

            if key:
                value = before_excl.strip()
                self._key_to_value[key] = value
                self._key_to_line_index[key] = i

    def get_value(self, key: str) -> Optional[str]:
        """
        Get the value for a key (the token(s) before the "! key :" comment).

        Args:
            key: Key name (e.g. "ltiming", "nt1", "uini," for "uini, udir").

        Returns:
            The value as a string, or None if not found.
        """
        return self._key_to_value.get(key)

    def get_value_as_float(self, key: str) -> Optional[float]:
        """Get the first token of the value as a float, or None."""
        value = self.get_value(key)
        if value is None:
            return None
        tokens = value.split()
        if not tokens:
            return None
        try:
            return float(tokens[0])
        except ValueError:
            return None

    def get_value_as_int(self, key: str) -> Optional[int]:
        """Get the first token of the value as an int, or None."""
        value = self.get_value(key)
        if value is None:
            return None
        tokens = value.split()
        if not tokens:
            return None
        try:
            return int(tokens[0])
        except ValueError:
            return None

    def get_value_as_bool(self, key: str) -> Optional[bool]:
        """Interpret T/F as True/False."""
        value = self.get_value(key)
        if value is None:
            return None
        v = value.strip().upper()
        if v == "T":
            return True
        if v == "F":
            return False
        return None

    def set_value(self, key: str, value: Union[str, float, int, bool]) -> None:
        """
        Set the value for a key. Replaces the value part before "!" on that line.
        If the key is new, appends a new line (caller should ensure key/comment format).

        Args:
            key: Key name.
            value: Value (converted to string; T/F for bool).
        """
        if isinstance(value, bool):
            value_str = "T" if value else "F"
        else:
            value_str = str(value)

        if key in self._key_to_line_index:
            idx = self._key_to_line_index[key]
            line = self.raw_lines[idx]
            if "!" in line:
                before_excl, excl, after_excl = line.partition("!")
                # Preserve width and leading space pattern of value field
                # Check if original had a leading space
                has_leading_space = before_excl.startswith(" ")
                # Calculate the original width (including trailing spaces)
                original_width = len(before_excl)
                # Add leading space to new value if original had one
                if has_leading_space and not value_str.startswith(" "):
                    value_str = " " + value_str
                # Pad the new value to match the original width (left-aligned)
                # Use max to ensure we don't truncate if new value is longer
                target_width = max(original_width, len(value_str))
                new_before = value_str.ljust(target_width)
                # Reconstruct the line with proper spacing
                self.raw_lines[idx] = new_before + excl + after_excl
            else:
                self.raw_lines[idx] = value_str + "\n"
        else:
            self.raw_lines.append(f"{value_str}                ! {key}\n")
            self._key_to_line_index[key] = len(self.raw_lines) - 1

        self._key_to_value[key] = value_str

    def has_key(self, key: str) -> bool:
        """Return True if the key exists."""
        return key in self._key_to_value

    def get_keys(self) -> list[str]:
        """Return all keys in parse order (line order)."""
        order = sorted(self._key_to_line_index.items(), key=lambda x: x[1])
        return [k for k, _ in order]

    def write(self, file_path: Optional[pathlib.Path] = None) -> None:
        """
        Write the infile back to disk.

        Args:
            file_path: Optional path to write to. If None, writes to original path.
        """
        output_path = file_path if file_path is not None else self.file_path
        with open(output_path, "w") as f:
            f.writelines(self.raw_lines)
        if file_path is not None:
            self.file_path = output_path
            self._parse_file()


def create_infile(dirs: "DirectoryPaths", verbose: bool = True) -> None:
    """
    Create infile.in by running the boltzmann executable.

    This function:
    1. Changes to the experiment directory
    2. Runs the boltzmann executable which generates infile.in

    Args:
        dirs: DirectoryPaths object containing all relevant paths (including experiment_dir
              and executable_path).
        verbose: If True, print output. If False, suppress output.

    Raises:
        FileNotFoundError: If executable doesn't exist.
        RuntimeError: If executable fails to create infile.in.
    """
    if not dirs.executable_path.exists():
        raise FileNotFoundError(
            f"Executable not found at {dirs.executable_path}. "
            f"Make sure the program has been compiled successfully."
        )

    # Ensure experiment directory exists
    dirs.experiment_dir.mkdir(parents=True, exist_ok=True)

    if verbose:
        logger.info("Creating infile.in by running boltzmann executable...")
        logger.info("Executable: %s", dirs.executable_path)
        logger.info("Working directory: %s", dirs.experiment_dir)

    # Set up environment variables
    env = os.environ.copy()
    env["HOME"] = str(dirs.pixi_env_path)
    if "PIXI_ENVIRONMENT" not in env:
        env["PIXI_ENVIRONMENT"] = str(dirs.pixi_env_path)

    # Change to experiment directory and run executable
    original_cwd = pathlib.Path.cwd()
    stdout = sys.stdout if verbose else subprocess.DEVNULL
    stderr = sys.stderr if verbose else subprocess.DEVNULL

    try:
        os.chdir(dirs.experiment_dir)

        result = subprocess.run(  # type: ignore[call-overload]
            [str(dirs.executable_path)],
            env=env,
            stdout=stdout,
            stderr=stderr,
            text=True,
        )

        if result.returncode != 0:
            raise RuntimeError(
                f"Failed to create infile.in. Executable exited with code {result.returncode}. "
                f"See output above for details."
            )

        # Verify that infile.in was created
        if not pathlib.Path(dirs.infile_path.name).exists():
            raise RuntimeError(
                f"Executable completed but infile.in was not created at {dirs.infile_path}"
            )

        if verbose:
            logger.info("Successfully created infile.in at %s", dirs.infile_path)

    finally:
        # Always return to original directory
        os.chdir(original_cwd)
