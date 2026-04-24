"""Read / edit PALM ``_p3d`` parameter files.

PALM's ``_p3d`` is a Fortran namelist with multiple sections such as::

    &initialization_parameters
        nx = 39, ny = 39, nz = 40,
        dx = 2.0, dy = 2.0, dz = 2.0,
        initializing_actions = 'set_constant_profiles',
        bc_lr = 'cyclic', bc_ns = 'cyclic',
        ug_surface = 3.0, vg_surface = 0.0,
    /
    &runtime_parameters
        end_time = 30.0,
        dt_data_output = 1.0,
        data_output = 'u', 'v', 'w',
    /

This module provides a structurally simple editor modelled after
``pyudales.utils.namoptions_utils.NamoptionsFile``: parse into a nested
``{section: {key: value}}`` dict preserving comments / order in the raw line
list, then rewrite with in-place edits. Array values are stored as whole
strings (e.g. ``"'u', 'v', 'w'"``) so callers can pass preformatted arrays.
"""

import logging
import pathlib
import re
from typing import Iterable, Optional

logger = logging.getLogger(__name__)


_SECTION_START_RE = re.compile(r"^\s*&(\w+)")
_SECTION_END_RE = re.compile(r"^\s*/\s*$")


def _format_array(values: Iterable[float]) -> str:
    return ", ".join(f"{float(v):.7f}" for v in values)


class P3DFile:
    """Parse + edit PALM ``_p3d`` namelist files.

    Only whole-line edits are supported. Multi-line array values are parsed as
    the concatenated RHS text up to the next key=value line or section end;
    ``set_array`` always writes a single-line replacement.
    """

    def __init__(self, file_path: pathlib.Path) -> None:
        self.file_path = pathlib.Path(file_path)
        self.sections: dict[str, dict[str, str]] = {}
        self.section_order: list[str] = []
        self.raw_lines: list[str] = []
        self._parse_file()

    def _parse_file(self) -> None:
        if not self.file_path.exists():
            logger.warning("p3d file %s does not exist", self.file_path)
            return

        with open(self.file_path, "r") as f:
            self.raw_lines = f.readlines()

        current_section: Optional[str] = None
        current: dict[str, str] = {}

        for line in self.raw_lines:
            stripped = line.strip()
            start = _SECTION_START_RE.match(stripped)
            if start:
                if current_section is not None:
                    self.sections[current_section] = current
                    if current_section not in self.section_order:
                        self.section_order.append(current_section)
                current_section = start.group(1)
                current = {}
                continue
            if _SECTION_END_RE.match(stripped) and current_section is not None:
                self.sections[current_section] = current
                if current_section not in self.section_order:
                    self.section_order.append(current_section)
                current_section = None
                current = {}
                continue
            if current_section is None or stripped.startswith("!") or not stripped:
                continue
            if "=" in stripped:
                key, _, value = stripped.partition("=")
                key = key.strip()
                value = value.strip().rstrip(",").strip()
                current[key] = value

        if current_section is not None:
            self.sections[current_section] = current
            if current_section not in self.section_order:
                self.section_order.append(current_section)

    def has_section(self, section: str) -> bool:
        return section in self.sections

    def get_value(self, section: str, key: str) -> Optional[str]:
        return self.sections.get(section, {}).get(key)

    def set_value(self, section: str, key: str, value: str | float | int) -> None:
        """Set ``section/key`` to ``value``. Scalar values get appropriate formatting."""
        if section not in self.sections:
            self.sections[section] = {}
            if section not in self.section_order:
                self.section_order.append(section)

        if isinstance(value, bool):
            formatted = ".true." if value else ".false."
        elif isinstance(value, (int, float)):
            formatted = repr(value)
        else:
            formatted = str(value)

        self.sections[section][key] = formatted

    def set_array(self, section: str, key: str, values: Iterable[float]) -> None:
        """Store a numeric array as a preformatted comma-separated string."""
        self.set_value(section, key, _format_array(values))

    def set_string(self, section: str, key: str, value: str) -> None:
        """Store a Fortran string, adding single quotes if missing."""
        if not (value.startswith("'") and value.endswith("'")):
            value = f"'{value}'"
        self.set_value(section, key, value)

    def write(self, file_path: Optional[pathlib.Path] = None) -> None:
        """Rewrite the file with current section data.

        Existing lines are preserved where possible; edited keys are updated in
        place; new keys are appended before the section terminator.
        """
        output_path = file_path if file_path is not None else self.file_path

        output_lines: list[str] = []
        current_section: Optional[str] = None
        keys_written: set[str] = set()

        for line in self.raw_lines:
            stripped = line.strip()
            start = _SECTION_START_RE.match(stripped)
            if start:
                current_section = start.group(1)
                keys_written = set()
                output_lines.append(line)
                continue
            if _SECTION_END_RE.match(stripped) and current_section is not None:
                remaining = self.sections.get(current_section, {})
                for key, value in remaining.items():
                    if key not in keys_written:
                        output_lines.append(f"    {key:<26} = {value},\n")
                        keys_written.add(key)
                output_lines.append(line)
                current_section = None
                continue
            if current_section is not None and "=" in stripped and not stripped.startswith("!"):
                key = stripped.split("=", 1)[0].strip()
                if key in self.sections.get(current_section, {}):
                    value = self.sections[current_section][key]
                    output_lines.append(f"    {key:<26} = {value},\n")
                    keys_written.add(key)
                    continue
            output_lines.append(line)

        existing_sections = {
            _SECTION_START_RE.match(line.strip()).group(1)
            for line in self.raw_lines
            if _SECTION_START_RE.match(line.strip() or "")
        }
        for section in self.section_order:
            if section not in existing_sections:
                output_lines.append(f"\n&{section}\n")
                for key, value in self.sections[section].items():
                    output_lines.append(f"    {key:<26} = {value},\n")
                output_lines.append("/\n")

        with open(output_path, "w") as f:
            f.writelines(output_lines)

        if file_path is not None:
            self.file_path = file_path
            self._parse_file()
