"""Unit tests for the per-axis extrapolation margin in interpolate_dataarray_at_points."""

import numpy as np
import pytest
import xarray
from data_assimilation.interpolation import interpolate_dataarray_at_points


def _make_field(x_coords: np.ndarray) -> xarray.DataArray:
    """A (z, y, x) field, constant along y/z, whose value equals the x coordinate.

    y and z each get a trivial 2-point strictly-increasing coordinate (the
    helper requires >=2 points per axis); observations are kept at y=z=0.25,
    comfortably inside both bounds, so every test below is really only
    exercising the x-axis extrapolation margin.
    """
    nz = ny = 2
    values = np.broadcast_to(x_coords.astype(float), (nz, ny, x_coords.size))
    return xarray.DataArray(
        values.copy(),
        dims=("z", "y", "x"),
        coords={
            "x": x_coords.astype(float),
            "y": np.array([0.0, 1.0]),
            "z": np.array([0.0, 1.0]),
        },
    )


def _yz(num_points: int) -> tuple[np.ndarray, np.ndarray]:
    """obs_y/obs_z arrays of the given length, held fixed at 0.25 (in-bounds)."""
    return np.full(num_points, 0.25), np.full(num_points, 0.25)


def test_resolves_cross_solver_staggered_dim_names() -> None:
    # A udales observation operator asks for ``x_dim="xm"``, but a PALM-generated
    # truth state carries ``u`` on ``xu`` (PALM's staggered convention). The
    # requested name must resolve to the state's actual staggered variant so
    # cross-solver truth/assim pairings interpolate instead of raising.
    x_coords = np.array([0.0, 1.0, 2.0, 3.0])
    values = np.broadcast_to(x_coords.astype(float), (2, 2, x_coords.size))
    palm_u = xarray.DataArray(
        values.copy(),
        dims=("z", "y", "xu"),  # PALM: u lives on xu
        coords={
            "xu": x_coords.astype(float),
            "y": np.array([0.0, 1.0]),
            "z": np.array([0.0, 1.0]),
        },
    )

    result = interpolate_dataarray_at_points(
        palm_u,
        x_dim="xm",  # udales operator's requested name
        y_dim="yt",
        z_dim="zt",
        obs_x=np.array([1.5]),
        obs_y=np.array([0.5]),
        obs_z=np.array([0.5]),
    )
    # The field equals its x coordinate, so interpolating at x=1.5 gives 1.5.
    np.testing.assert_allclose(result.values, [1.5])


def test_unresolvable_dim_still_raises() -> None:
    # A genuinely absent axis (no known alias present) must still raise, so a real
    # convention mismatch is not silently swallowed.
    field = _make_field(np.array([0.0, 1.0, 2.0]))
    with pytest.raises(ValueError, match="not present"):
        interpolate_dataarray_at_points(
            field.rename({"x": "longitude"}),
            x_dim="x",
            y_dim="y",
            z_dim="z",
            obs_x=np.array([1.0]),
            obs_y=np.array([0.5]),
            obs_z=np.array([0.5]),
        )


def test_uniform_grid_extrapolation_margin_matches_half_spacing() -> None:
    # Uniform grid: spacing is 2.0 everywhere, so median == edge spacing and
    # behavior must match the pre-fix single symmetric margin (1.0 each side).
    x_coords = np.array([0.0, 2.0, 4.0, 6.0, 8.0])
    field = _make_field(x_coords)
    obs_y, obs_z = _yz(2)

    # Just within the margin on both sides: accepted.
    result = interpolate_dataarray_at_points(
        field,
        x_dim="x",
        y_dim="y",
        z_dim="z",
        obs_x=np.array([-0.9, 8.9]),
        obs_y=obs_y,
        obs_z=obs_z,
    )
    assert result.shape == (2,)

    # Just beyond the margin: rejected.
    obs_y1, obs_z1 = _yz(1)
    with pytest.raises(ValueError, match="outside the grid bounds"):
        interpolate_dataarray_at_points(
            field,
            x_dim="x",
            y_dim="y",
            z_dim="z",
            obs_x=np.array([-1.1]),
            obs_y=obs_y1,
            obs_z=obs_z1,
        )


def test_stretched_grid_uses_local_edge_spacing_not_median() -> None:
    # Stretched grid: fine spacing of 0.1 at the lower (e.g. wall-refined) edge,
    # coarse spacing of 10.0 at the upper edge, with a large median spacing in
    # between. The accepted margin must come from each edge's OWN cell, not the
    # grid-wide median -- otherwise the fine edge would (wrongly) accept an
    # extrapolation far beyond half its own cell.
    x_coords = np.array([0.0, 0.1, 5.0, 10.0, 20.0])
    field = _make_field(x_coords)
    spacing = np.diff(x_coords)
    median_spacing = float(np.median(spacing))
    assert median_spacing > 0.5 * spacing[0]  # median is not the local edge spacing

    lower_margin = 0.5 * float(spacing[0])  # 0.05
    upper_margin = 0.5 * float(spacing[-1])  # 5.0
    obs_y1, obs_z1 = _yz(1)

    # Within half the local fine cell below the lower edge: accepted.
    result = interpolate_dataarray_at_points(
        field,
        x_dim="x",
        y_dim="y",
        z_dim="z",
        obs_x=np.array([0.0 - 0.9 * lower_margin]),
        obs_y=obs_y1,
        obs_z=obs_z1,
    )
    assert result.shape == (1,)

    # Beyond half the local fine cell below the lower edge: rejected, even
    # though it would have been accepted under the old median-based margin.
    assert 1.5 * lower_margin < median_spacing  # old code would have accepted this
    with pytest.raises(ValueError, match="outside the grid bounds"):
        interpolate_dataarray_at_points(
            field,
            x_dim="x",
            y_dim="y",
            z_dim="z",
            obs_x=np.array([0.0 - 1.5 * lower_margin]),
            obs_y=obs_y1,
            obs_z=obs_z1,
        )

    # Within half the local coarse cell above the upper edge: accepted.
    result = interpolate_dataarray_at_points(
        field,
        x_dim="x",
        y_dim="y",
        z_dim="z",
        obs_x=np.array([20.0 + 0.9 * upper_margin]),
        obs_y=obs_y1,
        obs_z=obs_z1,
    )
    assert result.shape == (1,)

    # Beyond half the local coarse cell above the upper edge: rejected.
    with pytest.raises(ValueError, match="outside the grid bounds"):
        interpolate_dataarray_at_points(
            field,
            x_dim="x",
            y_dim="y",
            z_dim="z",
            obs_x=np.array([20.0 + 1.1 * upper_margin]),
            obs_y=obs_y1,
            obs_z=obs_z1,
        )
