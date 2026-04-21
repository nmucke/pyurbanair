"""Utilities for updating column-based input files."""

import logging
import pathlib
from typing import Callable, Optional

import numpy as np

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

    def _rewrite(
        self,
        row_updates: Callable[[int], dict[int, float]],
        log_message: Optional[str] = None,
    ) -> int:
        """Apply ``row_updates(row_idx)`` to each non-header data row.

        Returns the number of data rows rewritten.
        """
        if not self.file_path.exists():
            logger.warning(f"File {self.file_path} not found, skipping update")
            return 0

        with open(self.file_path, "r") as f:
            lines = f.readlines()

        output_lines: list[str] = []
        data_row_idx = 0
        for line in lines:
            stripped = line.strip()
            if stripped.startswith(self.header_prefix) or not stripped:
                output_lines.append(line)
                continue

            parts = stripped.split()
            if len(parts) < self.min_columns:
                output_lines.append(line)
                continue

            try:
                values = [float(p) for p in parts[: len(self.column_formats)]]
                updates = row_updates(data_row_idx)
                for col_idx, new_value in updates.items():
                    if col_idx < len(values):
                        values[col_idx] = new_value

                formatted_parts = []
                for i, fmt in enumerate(self.column_formats):
                    if i < len(values):
                        formatted_parts.append(f"{values[i]:{fmt}}")
                    elif i < len(parts):
                        formatted_parts.append(parts[i])

                if len(parts) > len(self.column_formats):
                    formatted_parts.extend(parts[len(self.column_formats) :])

                output_lines.append(" ".join(formatted_parts) + "\n")
                data_row_idx += 1
            except (ValueError, IndexError):
                output_lines.append(line)

        with open(self.file_path, "w") as f:
            f.writelines(output_lines)

        if log_message:
            logger.info(log_message)

        return data_row_idx

    def update_columns(
        self,
        updates: dict[int, float],
        log_message: Optional[str] = None,
    ) -> None:
        """
        Update specific columns in the file with the same scalar on every row.

        Args:
            updates: Dictionary mapping column indices (0-based) to new float values.
                Only columns in this dictionary will be updated; others keep their original values.
            log_message: Optional log message to output after updating.
        """
        if not updates:
            return
        self._rewrite(lambda _idx: updates, log_message=log_message)

    def update_columns_per_row(
        self,
        updates: dict[int, np.ndarray],
        log_message: Optional[str] = None,
    ) -> None:
        """
        Update specific columns in the file with per-row values.

        Args:
            updates: Dictionary mapping column indices (0-based) to 1-D arrays
                whose length matches the number of non-header data rows.
            log_message: Optional log message to output after updating.

        Raises:
            ValueError: If any array length does not equal the data-row count.
        """
        if not updates:
            return

        def _row_updates(idx: int) -> dict[int, float]:
            return {col: float(arr[idx]) for col, arr in updates.items()}

        n_rows = self._rewrite(_row_updates, log_message=log_message)
        for col, arr in updates.items():
            if len(arr) != n_rows:
                raise ValueError(
                    f"Column {col} profile length {len(arr)} does not match "
                    f"{n_rows} data rows in {self.file_path.name}"
                )


def _as_profile(value: Optional[float], length: int) -> Optional[np.ndarray]:
    if value is None:
        return None
    return np.full(length, float(value))


def update_prof_file(
    file_path: pathlib.Path,
    u0: Optional[float] = None,
    v0: Optional[float] = None,
) -> None:
    """
    Update prof.inp.* file with new u0 and v0 scalars (uniform in z).

    Args:
        file_path: Path to the prof.inp.* file.
        u0: New u0 value (column 3). If None, keeps original value.
        v0: New v0 value (column 4). If None, keeps original value.
    """
    if u0 is None and v0 is None:
        return

    column_formats = ["20.15f", "12.6f", "12.6f", "12.6f", "12.6f", "12.6f"]
    updates: dict[int, float] = {}
    if u0 is not None:
        updates[3] = u0
    if v0 is not None:
        updates[4] = v0

    log_message = f"Updated {file_path.name} with u0={u0}, v0={v0}"
    ColumnBasedFileUpdater(file_path, column_formats).update_columns(
        updates, log_message=log_message
    )


def update_prof_file_profile(
    file_path: pathlib.Path,
    u_profile: Optional[np.ndarray] = None,
    v_profile: Optional[np.ndarray] = None,
) -> None:
    """
    Update prof.inp.* file with per-height u and v profiles.

    Args:
        file_path: Path to the prof.inp.* file.
        u_profile: Per-row u values (column 3), length must equal ktot.
        v_profile: Per-row v values (column 4), length must equal ktot.
    """
    if u_profile is None and v_profile is None:
        return

    column_formats = ["20.15f", "12.6f", "12.6f", "12.6f", "12.6f", "12.6f"]
    updates: dict[int, np.ndarray] = {}
    if u_profile is not None:
        updates[3] = np.asarray(u_profile, dtype=float)
    if v_profile is not None:
        updates[4] = np.asarray(v_profile, dtype=float)

    log_message = f"Updated {file_path.name} with u/v profiles"
    ColumnBasedFileUpdater(file_path, column_formats).update_columns_per_row(
        updates, log_message=log_message
    )


def update_lscale_file(
    file_path: pathlib.Path,
    u0: Optional[float] = None,
    v0: Optional[float] = None,
    dpdx: Optional[float] = None,
    dpdy: Optional[float] = None,
) -> None:
    """
    Update lscale.inp.* file with new u0, v0, dpdx, and dpdy scalars (uniform in z).

    Args:
        file_path: Path to the lscale.inp.* file.
        u0: New u0 value (column 1). If None, keeps original value.
        v0: New v0 value (column 2). If None, keeps original value.
        dpdx: New dpdx value (column 3). If None, keeps original value.
        dpdy: New dpdy value (column 4). If None, keeps original value.
    """
    if u0 is None and v0 is None and dpdx is None and dpdy is None:
        return

    column_formats = [
        "20.15f",  # z
        "12.6f",  # uq
        "12.6f",  # vq
        "12.9f",  # pqx
        "12.9f",  # pqy
        "15.9f",  # wfls
        "12.6f",  # dqtdxls
        "12.6f",  # dqtdyls
        "12.6f",  # dqtdtls
        "17.12f",  # dthlrad
    ]
    updates: dict[int, float] = {}
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
    )
    ColumnBasedFileUpdater(file_path, column_formats).update_columns(
        updates, log_message=log_message
    )


def update_lscale_file_profile(
    file_path: pathlib.Path,
    u_profile: Optional[np.ndarray] = None,
    v_profile: Optional[np.ndarray] = None,
    dpdx_profile: Optional[np.ndarray] = None,
    dpdy_profile: Optional[np.ndarray] = None,
) -> None:
    """
    Update lscale.inp.* file with per-height profiles.

    Args:
        file_path: Path to the lscale.inp.* file.
        u_profile: Per-row uq values (column 1).
        v_profile: Per-row vq values (column 2).
        dpdx_profile: Per-row pqx values (column 3).
        dpdy_profile: Per-row pqy values (column 4).
    """
    if all(p is None for p in (u_profile, v_profile, dpdx_profile, dpdy_profile)):
        return

    column_formats = [
        "20.15f",
        "12.6f",
        "12.6f",
        "12.9f",
        "12.9f",
        "15.9f",
        "12.6f",
        "12.6f",
        "12.6f",
        "17.12f",
    ]
    updates: dict[int, np.ndarray] = {}
    if u_profile is not None:
        updates[1] = np.asarray(u_profile, dtype=float)
    if v_profile is not None:
        updates[2] = np.asarray(v_profile, dtype=float)
    if dpdx_profile is not None:
        updates[3] = np.asarray(dpdx_profile, dtype=float)
    if dpdy_profile is not None:
        updates[4] = np.asarray(dpdy_profile, dtype=float)

    log_message = f"Updated {file_path.name} with lscale profiles"
    ColumnBasedFileUpdater(file_path, column_formats).update_columns_per_row(
        updates, log_message=log_message
    )
