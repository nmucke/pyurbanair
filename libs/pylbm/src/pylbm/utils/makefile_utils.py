"""Utilities for parsing and editing the LBM makefile (path section at top)."""

import logging
import pathlib
import re
from typing import Optional, Union

logger = logging.getLogger(__name__)

# Only these variables at the top of the makefile are edited by this class.
MAKEFILE_PATH_VARS = ("SRC_DIR", "BUILD", "VPATH", "HOME", "BINDIR", "NCFDIR")


class Makefile:
    """
    A parser and editor for the LBM makefile path section (first few lines).

    Only the top path variables are managed: SRC_DIR, BUILD, VPATH, HOME, BINDIR, NCFDIR.
    The rest of the makefile is preserved unchanged when writing.
    """

    def __init__(self, file_path: pathlib.Path) -> None:
        """
        Initialize the Makefile by reading and parsing the path section.

        Args:
            file_path: Path to the makefile.
        """
        self.file_path = pathlib.Path(file_path)
        self._path_vars: dict[str, str] = {}
        self._var_to_line_index: dict[str, int] = {}
        self.raw_lines: list[str] = []
        self._path_section_end: int = 0  # index of first line after path section

        if self.file_path.exists():
            self._parse_file()

    def _parse_file(self) -> None:
        """Parse the makefile and extract path variable assignments (only top section)."""
        if not self.file_path.exists():
            return

        with open(self.file_path, "r") as f:
            self.raw_lines = f.readlines()

        # Match "VAR := value" or "VAR = value" (with optional spaces)
        pattern = re.compile(r"^\s*([A-Za-z_][A-Za-z0-9_]*)\s*(?::?=)\s*(.*)$")

        for i, line in enumerate(self.raw_lines):
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                if not self._path_vars:
                    continue
                # First non-assignment after we saw vars ends the path section
                if self._path_vars and i not in self._var_to_line_index.values():
                    self._path_section_end = i
                    break
                continue

            m = pattern.match(stripped)
            if m:
                name, value = m.group(1), m.group(2).strip()
                if name in MAKEFILE_PATH_VARS:
                    self._path_vars[name] = value
                    self._var_to_line_index[name] = i
                    self._path_section_end = i + 1
            else:
                # Non-assignment line: if we already have path vars, stop
                if self._path_vars:
                    self._path_section_end = i
                    break

    def get_path(self, var: str) -> Optional[str]:
        """
        Get the value of a path variable.

        Args:
            var: Variable name (e.g. "SRC_DIR", "HOME", "BINDIR").

        Returns:
            The value as a string, or None if not found.
        """
        return self._path_vars.get(var)

    def set_path(self, var: str, value: Union[str, pathlib.Path]) -> None:
        """
        Set a path variable. Only variables in MAKEFILE_PATH_VARS are stored.
        If the variable already exists in the file, its line is updated; otherwise
        a new line is appended to the path section.

        Args:
            var: Variable name (e.g. "HOME", "NCFDIR").
            value: Value (string or path).
        """
        if var not in MAKEFILE_PATH_VARS:
            logger.warning(
                f"Makefile path variable '{var}' is not in {MAKEFILE_PATH_VARS}; setting anyway."
            )
        value_str = (
            str(pathlib.Path(value)) if isinstance(value, pathlib.Path) else str(value)
        )

        if var in self._var_to_line_index:
            idx = self._var_to_line_index[var]
            line = self.raw_lines[idx]
            # Preserve "VAR := " or "VAR = " style
            if ":=" in line:
                self.raw_lines[idx] = re.sub(
                    r"^(\s*[A-Za-z_][A-Za-z0-9_]*\s*:=).*",
                    r"\g<1> " + value_str + "\n",
                    line,
                    count=1,
                )
            else:
                self.raw_lines[idx] = re.sub(
                    r"^(\s*[A-Za-z_][A-Za-z0-9_]*\s*=).*",
                    r"\g<1> " + value_str + "\n",
                    line,
                    count=1,
                )
        else:
            # Append new line at end of path section
            self.raw_lines.insert(self._path_section_end, f"{var} := {value_str}\n")
            self._path_section_end += 1
            # Re-index
            self._var_to_line_index = {}
            for i, line in enumerate(self.raw_lines):
                for v in MAKEFILE_PATH_VARS:
                    if re.match(rf"^\s*{re.escape(v)}\s*(?::?=)", line.strip()):
                        self._var_to_line_index[v] = i
                        break

        self._path_vars[var] = value_str

    def set_paths(
        self,
        src_dir: Optional[Union[str, pathlib.Path]] = None,
        build: Optional[Union[str, pathlib.Path]] = None,
        vpath: Optional[str] = None,
        home: Optional[Union[str, pathlib.Path]] = None,
        bindir: Optional[Union[str, pathlib.Path]] = None,
        ncfdir: Optional[Union[str, pathlib.Path]] = None,
    ) -> None:
        """
        Set multiple path variables at once. Omitted arguments are left unchanged.

        Args:
            src_dir: SRC_DIR value.
            build: BUILD value.
            vpath: VPATH value.
            home: HOME value.
            bindir: BINDIR value.
            ncfdir: NCFDIR value.
        """
        if src_dir is not None:
            self.set_path("SRC_DIR", src_dir)
        if build is not None:
            self.set_path("BUILD", build)
        if vpath is not None:
            self.set_path("VPATH", vpath)
        if home is not None:
            self.set_path("HOME", home)
        if bindir is not None:
            self.set_path("BINDIR", bindir)
        if ncfdir is not None:
            self.set_path("NCFDIR", ncfdir)

    def write(self, file_path: Optional[pathlib.Path] = None) -> None:
        """
        Write the makefile back to disk.

        Args:
            file_path: Optional path to write to. If None, writes to original path.
        """
        output_path = file_path if file_path is not None else self.file_path
        with open(output_path, "w") as f:
            f.writelines(self.raw_lines)
        if file_path is not None:
            self.file_path = output_path
            self._parse_file()
