"""Utilities for parsing and editing the LBM makefile (path section at top)."""

import logging
import pathlib
import re
from typing import Optional, Union

logger = logging.getLogger(__name__)

# Only these variables at the top of the makefile are edited by this class.
MAKEFILE_PATH_VARS = ("SRC_DIR", "BUILD", "VPATH", "HOME", "BINDIR", "NCFDIR")

# "VAR := value" or "VAR = value"
_ASSIGNMENT = re.compile(r"^\s*([A-Za-z_][A-Za-z0-9_]*)\s*(?::?=)\s*(.*)$")
_CONDITIONAL_OPEN = re.compile(r"^\s*(ifdef|ifndef|ifeq|ifneq)\b")
_CONDITIONAL_CLOSE = re.compile(r"^\s*endif\b")


def _unconditional_assignments(lines: list[str]) -> list[tuple[int, str, str]]:
    """
    Find managed variable assignments at the makefile's top level.

    Assignments inside ``ifdef``/``ifeq`` blocks are skipped deliberately. This
    class rewrites a variable by replacing its line and deleting every other
    assignment of it; applied to a conditional override (say a ``BINDIR`` set only
    under ``ifdef MPI``) that would silently delete a branch that was never meant
    to be collapsed. Today's upstream makefile declares all six on lines 1-6 with
    no conditionals, so this changes nothing -- it bounds the blast radius if that
    ever stops being true.

    Args:
        lines: The makefile's raw lines.

    Returns:
        ``(line_index, name, value)`` for each top-level managed assignment.
    """
    found: list[tuple[int, str, str]] = []
    depth = 0
    for i, line in enumerate(lines):
        if _CONDITIONAL_OPEN.match(line):
            depth += 1
            continue
        if _CONDITIONAL_CLOSE.match(line):
            depth = max(0, depth - 1)
            continue
        if depth:
            continue
        m = _ASSIGNMENT.match(line.strip())
        if m and m.group(1) in MAKEFILE_PATH_VARS:
            found.append((i, m.group(1), m.group(2).strip()))
    return found


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
        """
        Parse the makefile and index every managed path-variable assignment.

        The whole file is scanned rather than stopping at the first blank line.
        Stopping early made the class non-idempotent: any variable that fell below
        the cut was reported as absent and re-appended by ``set_path`` on every
        construction, which is how a 240-line makefile grew 1275 duplicate
        ``NCFDIR`` lines in the checked-out submodule.
        """
        if not self.file_path.exists():
            return

        with open(self.file_path, "r") as f:
            self.raw_lines = f.readlines()

        self._reindex()

    def _reindex(self) -> None:
        """Rebuild the variable index from ``raw_lines`` (first assignment wins)."""
        self._path_vars = {}
        self._var_to_line_index = {}
        self._path_section_end = 0
        for i, name, value in _unconditional_assignments(self.raw_lines):
            if name not in self._var_to_line_index:
                self._path_vars[name] = value
                self._var_to_line_index[name] = i
            self._path_section_end = max(self._path_section_end, i + 1)

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
            # Preserve "VAR := " or "VAR = " style. The assignment operator is
            # kept and everything after it replaced; `.*\n?` must consume the
            # line's own newline, otherwise the replacement's newline is appended
            # to it and each write injects a blank line into the makefile.
            operator = ":=" if ":=" in line else "="
            self.raw_lines[idx] = re.sub(
                rf"^(\s*[A-Za-z_][A-Za-z0-9_]*\s*{re.escape(operator)}).*\n?",
                r"\g<1> " + value_str.replace("\\", r"\\") + "\n",
                line,
                count=1,
            )
        else:
            # Append new line at end of path section
            self.raw_lines.insert(self._path_section_end, f"{var} := {value_str}\n")
            self._path_section_end += 1

        # Drop any duplicate top-level assignment of this variable left by an
        # earlier (non-idempotent) write, so repeated edits converge on one line.
        # Conditional overrides are not candidates -- see _unconditional_assignments.
        duplicate_lines = [
            i
            for i, name, _ in _unconditional_assignments(self.raw_lines)
            if name == var
        ][1:]
        if duplicate_lines:
            logger.info(
                "Removed %d duplicate '%s' assignment(s) from %s",
                len(duplicate_lines),
                var,
                self.file_path,
            )
            drop = set(duplicate_lines)
            self.raw_lines = [
                line for i, line in enumerate(self.raw_lines) if i not in drop
            ]

        # Re-index: line numbers shift on insert and on dedup.
        self._reindex()
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

    def set_gpu_arch(self, cc: str) -> bool:
        """
        Retarget every ``-gpu=cc<NN>`` compute-capability token to ``cc<cc>``.

        The upstream LBM makefile hardcodes a single GPU architecture (e.g.
        ``cc120``) in CCFLAG/LINKFLAGS. On a host with a different GPU the binary
        compiles but its device kernels are missing at run time ("Could not find
        symbol ... Rebuild with -gpu=ccXX"). Rewriting the token to the host's
        actual compute capability fixes that without touching anything else.

        Args:
            cc: Compute capability without the dot (e.g. "86" for an RTX 3090).

        Returns:
            True if at least one token was rewritten.
        """
        pattern = re.compile(r"-gpu=cc\d+")
        replacement = f"-gpu=cc{cc}"
        changed = False
        for i, line in enumerate(self.raw_lines):
            if "-gpu=cc" in line:
                new_line = pattern.sub(replacement, line)
                if new_line != line:
                    self.raw_lines[i] = new_line
                    changed = True
        return changed

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
