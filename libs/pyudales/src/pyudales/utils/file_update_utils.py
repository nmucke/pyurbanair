"""Utilities for updating column-based input files."""

import logging
import pathlib
from typing import Optional

logger = logging.getLogger(__name__)


class ColumnBasedFileUpdater:
    """
    A utility class for updating column-based input files.

    This class handles files with:
    - Header lines (typically starting with #)
    - Data lines with space-separated columns
    - Specific formatting requirements for each column
    """

    def __init__(
        self,
        file_path: pathlib.Path,
        column_formats: list[str],
        header_prefix: str = "#",
        min_columns: Optional[int] = None,
    ) -> None:
        """
        Initialize the ColumnBasedFileUpdater.

        Args:
            file_path: Path to the file to update.
            column_formats: List of format strings for each column (e.g., ["20.15f", "12.6f"]).
            header_prefix: Prefix that identifies header lines (default: "#").
            min_columns: Minimum number of columns required for a valid data line.
                If None, uses len(column_formats).
        """
        self.file_path = pathlib.Path(file_path)
        self.column_formats = column_formats
        self.header_prefix = header_prefix
        self.min_columns = (
            min_columns if min_columns is not None else len(column_formats)
        )

    def update_columns(
        self,
        updates: dict[int, float],
        log_message: Optional[str] = None,
    ) -> None:
        """
        Update specific columns in the file.

        Args:
            updates: Dictionary mapping column indices (0-based) to new float values.
                Only columns in this dictionary will be updated; others keep their original values.
            log_message: Optional log message to output after updating.
        """
        if not self.file_path.exists():
            logger.warning(f"File {self.file_path} not found, skipping update")
            return

        if not updates:
            return

        # Read the file
        lines = []
        with open(self.file_path, "r") as f:
            lines = f.readlines()

        # Update data lines (skip header lines and empty lines)
        output_lines = []
        for line in lines:
            stripped = line.strip()
            if stripped.startswith(self.header_prefix) or not stripped:
                # Keep header and empty lines as is
                output_lines.append(line)
            else:
                # Parse the data line
                parts = stripped.split()
                if len(parts) >= self.min_columns:
                    # Parse all columns as floats
                    try:
                        values = [
                            float(part) for part in parts[: len(self.column_formats)]
                        ]

                        # Update values for columns specified in updates
                        for col_idx, new_value in updates.items():
                            if col_idx < len(values):
                                values[col_idx] = new_value

                        # Format the line according to column formats
                        formatted_parts = []
                        for i, fmt in enumerate(self.column_formats):
                            if i < len(values):
                                formatted_parts.append(f"{values[i]:{fmt}}")
                            elif i < len(parts):
                                # Keep original value if we don't have a format for it
                                formatted_parts.append(parts[i])

                        # Add any remaining columns that weren't formatted
                        if len(parts) > len(self.column_formats):
                            formatted_parts.extend(parts[len(self.column_formats) :])

                        output_lines.append(" ".join(formatted_parts) + "\n")
                    except (ValueError, IndexError):
                        # Keep line as is if parsing fails
                        output_lines.append(line)
                else:
                    # Keep line as is if format is unexpected
                    output_lines.append(line)

        # Write the updated file
        with open(self.file_path, "w") as f:
            f.writelines(output_lines)

        if log_message:
            logger.info(log_message)


def update_prof_file(
    file_path: pathlib.Path,
    u0: Optional[float] = None,
    v0: Optional[float] = None,
) -> None:
    """
    Update prof.inp.* file with new u0 and v0 values.

    Args:
        file_path: Path to the prof.inp.* file.
        u0: New u0 value (column 3, 0-based index). If None, keeps original value.
        v0: New v0 value (column 4, 0-based index). If None, keeps original value.
    """
    if u0 is None and v0 is None:
        return

    # Format: z thl qt u v tke
    # Format strings: %-20.15f %-12.6f %-12.6f %-12.6f %-12.6f %-12.6f
    column_formats = ["20.15f", "12.6f", "12.6f", "12.6f", "12.6f", "12.6f"]

    updates = {}
    if u0 is not None:
        updates[3] = u0
    if v0 is not None:
        updates[4] = v0

    log_message = f"Updated {file_path.name} with u0={u0}, v0={v0}" if updates else None

    updater = ColumnBasedFileUpdater(file_path, column_formats)
    updater.update_columns(updates, log_message=log_message)


def update_lscale_file(
    file_path: pathlib.Path,
    u0: Optional[float] = None,
    v0: Optional[float] = None,
    dpdx: Optional[float] = None,
    dpdy: Optional[float] = None,
) -> None:
    """
    Update lscale.inp.* file with new u0, v0, dpdx, and dpdy values.

    Args:
        file_path: Path to the lscale.inp.* file.
        u0: New u0 value (column 1, 0-based index). If None, keeps original value.
        v0: New v0 value (column 2, 0-based index). If None, keeps original value.
        dpdx: New dpdx value (column 3, 0-based index). If None, keeps original value.
        dpdy: New dpdy value (column 4, 0-based index). If None, keeps original value.
    """
    if u0 is None and v0 is None and dpdx is None and dpdy is None:
        return

    # Format: z uq vq pqx pqy wfls dqtdxls dqtdyls dqtdtls dthlrad
    # Format strings match preprocessing (write_lscale in preprocessing.py)
    column_formats = [
        "20.15f",  # z
        "12.6f",  # uq
        "12.6f",  # vq
        "12.9f",  # pqx
        "12.9f",  # pqy (was 12.6f, fixed to match preprocessing 12.9f)
        "15.9f",  # wfls
        "12.6f",  # dqtdxls
        "12.6f",  # dqtdyls
        "12.6f",  # dqtdtls
        "17.12f",  # dthlrad
    ]

    updates = {}
    if u0 is not None:
        updates[1] = u0
    if v0 is not None:
        updates[2] = v0
    if dpdx is not None:
        updates[3] = dpdx
    if dpdy is not None:
        updates[4] = dpdy

    log_message = (
        f"Updated {file_path.name} with u0={u0}, v0={v0}, dpdx={dpdx}, dpdy={dpdy}"
        if updates
        else None
    )

    updater = ColumnBasedFileUpdater(file_path, column_formats)
    updater.update_columns(updates, log_message=log_message)
