"""Unit tests for pypalm's periodic nudging driver (NUDGING_DATA + LSF_DATA).

These pin the exact on-disk format PALM's ``nudge_init`` / ``lsf_init`` readers
parse — block markers, the u/v-only sentinel columns, the tnudge near-wall
cutoff, the terminal bracketing snapshot, and the inert-LSF layout.
"""

import pathlib
from typing import Any

import numpy as np
import pytest
import xarray
from pypalm.utils.nudging_utils import (
    NUDGE_SENTINEL,
    _nudging_heights,
    _tnudge_column,
    apply_nudging_driver,
    write_inert_lsf_data,
    write_nudging_data,
)

BOUNDS = ((0.0, 40.0), (0.0, 40.0), (0.0, 40.0))


def _static_params(angle: float = 30.0, speed: float = 5.0) -> xarray.Dataset:
    return xarray.Dataset(
        data_vars={"inflow_angle": float(angle), "velocity_magnitude": float(speed)}
    )


def _time_varying_params(times: Any, angles: Any, speeds: Any) -> xarray.Dataset:
    def _var(v: Any) -> Any:
        return ("time", np.asarray(v, dtype=float)) if np.ndim(v) else float(v)

    return xarray.Dataset(
        data_vars={"inflow_angle": _var(angles), "velocity_magnitude": _var(speeds)},
        coords={"time": np.asarray(times, dtype=float)},
    )


# --- _nudging_heights / _tnudge_column --------------------------------------


def test_nudging_heights_span_grid_with_anchor_and_top() -> None:
    nz, dz = 10, 4.0
    heights = _nudging_heights(nz, dz, nnudge_meters=4.0)
    # z=0 anchor present, monotonically increasing, no zmin offset (0-based).
    assert heights[0] == 0.0
    assert np.all(np.diff(heights) > 0)
    # Top row is comfortably above the last cell centre (zu(nzt+1)).
    last_centre = (nz - 0.5) * dz
    assert heights[-1] >= last_centre + 2.0 * dz - 1e-9
    # Cutoff pair straddles nnudge_meters for a sharp tnudge step.
    assert np.any(heights < 4.0) and np.any(np.isclose(heights, 4.0, atol=0.5))


def test_nudging_heights_no_cutoff_pair_when_disabled() -> None:
    heights = _nudging_heights(nz=10, dz=4.0, nnudge_meters=0.0)
    # No negative straddle rows when the cutoff is disabled.
    assert np.all(heights >= 0.0)


def test_tnudge_column_huge_below_cutoff() -> None:
    heights = np.array([0.0, 2.0, 4.0, 6.0])
    col = _tnudge_column(heights, tnudge=15.0, nnudge_meters=4.0)
    assert col[0] > 1e6 and col[1] > 1e6  # below cutoff -> disabled
    assert col[2] == 15.0 and col[3] == 15.0  # at/above cutoff -> active


# --- write_nudging_data ------------------------------------------------------


def test_write_nudging_data_block_and_column_shape(tmp_path: pathlib.Path) -> None:
    times = np.array([0.0, 50.0])
    heights = np.array([0.0, 2.0, 4.0])
    tnudge_column = np.array([1e9, 1e9, 15.0])
    u = np.array([[0.0, 1.0, 2.0], [0.0, 1.5, 3.0]])
    v = np.array([[0.0, 0.5, 1.0], [0.0, 0.7, 1.4]])
    path = tmp_path / "run_nudge"
    write_nudging_data(path, times, heights, tnudge_column, u, v)

    lines = path.read_text().splitlines()
    markers = [ln for ln in lines if ln.startswith("#")]
    assert len(markers) == 2  # one block marker per time
    assert markers[0].split()[1].startswith("0")
    # Each block: one marker + one row per height.
    assert len(lines) == 2 * (1 + len(heights))

    # Every data row carries the w/pt/q sentinel in its last three columns and
    # its own tnudge / u / v.
    first_row = lines[1].split()
    assert len(first_row) == 7  # height tnudge u v w pt q
    assert [float(x) for x in first_row[-3:]] == [NUDGE_SENTINEL] * 3
    assert float(first_row[0]) == 0.0
    assert float(first_row[2]) == 0.0  # u(z0)


def test_write_nudging_data_rejects_shape_mismatch(tmp_path: pathlib.Path) -> None:
    times = np.array([0.0, 50.0])
    heights = np.array([0.0, 2.0, 4.0])
    tnudge_column = np.array([1e9, 1e9, 15.0])
    bad_u = np.zeros((2, 2))  # wrong Nz
    v = np.zeros((2, 3))
    with pytest.raises(ValueError, match="profile shape mismatch"):
        write_nudging_data(tmp_path / "x", times, heights, tnudge_column, bad_u, v)


# --- write_inert_lsf_data ----------------------------------------------------


def test_write_inert_lsf_data_layout(tmp_path: pathlib.Path) -> None:
    end_time = 300.0
    path = tmp_path / "run_lsf"
    write_inert_lsf_data(path, end_time)
    lines = path.read_text().splitlines()

    # 3 header comment lines, one surface row, a bare '#', a '# <time>' marker.
    assert len(lines) == 6
    assert all(lines[i].startswith("#") for i in (0, 1, 2))
    surface = lines[3].split()
    assert len(surface) == 6  # time shf qsws pt q p
    assert float(surface[0]) > end_time  # surface time beyond end_time -> lsf_surf off
    assert lines[4] == "#"  # sacrificial separator consumed by the skip loop
    assert lines[5].startswith("# ")
    assert float(lines[5].split()[1]) > end_time  # profile time beyond end_time


# --- apply_nudging_driver ----------------------------------------------------


def test_apply_nudging_driver_static_two_blocks(tmp_path: pathlib.Path) -> None:
    nudge = tmp_path / "run_nudge"
    lsf = tmp_path / "run_lsf"
    init = apply_nudging_driver(
        params=_static_params(angle=0.0, speed=3.0),
        nudge_path=nudge,
        lsf_path=lsf,
        bounds=BOUNDS,
        nz=10,
        profile_config={"type": "power_law", "alpha": 0.25},
        tnudge=15.0,
        nnudge_meters=4.0,
        spinup_time=0.0,
        simulation_time=100.0,
    )
    markers = [ln for ln in nudge.read_text().splitlines() if ln.startswith("#")]
    assert len(markers) == 2  # static -> exactly two blocks
    assert lsf.exists()
    # Returns the t=0 scalar values for the init writes.
    assert float(init["inflow_angle"]) == 0.0
    assert float(init["velocity_magnitude"]) == 3.0


def test_apply_nudging_driver_time_varying_block_count(tmp_path: pathlib.Path) -> None:
    nudge = tmp_path / "run_nudge"
    lsf = tmp_path / "run_lsf"
    times = [0.0, 50.0, 100.0]
    apply_nudging_driver(
        params=_time_varying_params(times, [0.0, 30.0, 60.0], 5.0),
        nudge_path=nudge,
        lsf_path=lsf,
        bounds=BOUNDS,
        nz=8,
        profile_config={"type": "power_law", "alpha": 0.25},
        tnudge=15.0,
        nnudge_meters=4.0,
        spinup_time=20.0,
        simulation_time=100.0,
    )
    markers = [ln for ln in nudge.read_text().splitlines() if ln.startswith("#")]
    # one block per param time (3) + spinup plateau (1) + terminal pad (1).
    assert len(markers) == len(times) + 2


def test_apply_nudging_driver_terminal_snapshot_past_end_time(
    tmp_path: pathlib.Path,
) -> None:
    nudge = tmp_path / "run_nudge"
    lsf = tmp_path / "run_lsf"
    apply_nudging_driver(
        params=_static_params(),
        nudge_path=nudge,
        lsf_path=lsf,
        bounds=BOUNDS,
        nz=6,
        profile_config=None,
        tnudge=15.0,
        nnudge_meters=4.0,
        spinup_time=0.0,
        simulation_time=100.0,
    )
    markers = [
        float(ln.split()[1])
        for ln in nudge.read_text().splitlines()
        if ln.startswith("#")
    ]
    assert markers[-1] > 100.0  # bracketing snapshot past end_time
