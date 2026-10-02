"""Unit tests for the pypalm warm-start driver writer.

These exercise ``warm_start_utils.write_warmstart_driver`` without running PALM:
warm-start is plain NetCDF bookkeeping (write ``init_atmosphere_*`` LOD=2 fields
into the PIDS_DYNAMIC driver). The behaviour pinned down here is the grid
contract PALM enforces — 0-based value-checked z/zw, correct staggered dim
lengths, ``lod=2`` attributes, and no fill values — plus coexistence with a
time-varying ``inflow_plane_*`` driver in the same file.
"""

import os

os.environ.setdefault("PYPALM_SKIP_AUTOINSTALL", "1")

import numpy as np
import pytest
import xarray

from pypalm.utils.warm_start_utils import _FILL, write_warmstart_driver

# pypalm grid-point counts (PALM namelist nx=NX-1, ny=NY-1, nz=NZ).
NX, NY, NZ = 6, 5, 4
# Non-zero offsets so the 0-based file axes differ from the physical state.
BOUNDS = ((10.0, 22.0), (5.0, 15.0), (4.0, 12.0))
DX = (BOUNDS[0][1] - BOUNDS[0][0]) / NX  # 2.0
DY = (BOUNDS[1][1] - BOUNDS[1][0]) / NY  # 2.0
DZ = (BOUNDS[2][1] - BOUNDS[2][0]) / NZ  # 2.0

U_CONST, V_CONST, W_CONST = 3.0, -1.0, 0.2


def _make_state() -> xarray.Dataset:
    """A postprocessed-style PALM state.

    Mirrors ``_load_and_postprocess_state``: only the vertical axis is unified
    (``w`` on ``z``), so the horizontal staggers survive — ``u`` on ``xu``,
    ``v`` on ``yv``, ``w``/scalars on ``x``/``y`` — all in the physical (offset)
    frame.
    """
    (xmin, _), (ymin, _), (zmin, _) = BOUNDS
    x = (np.arange(NX) + 0.5) * DX + xmin       # scalar x (NX)
    xu = np.arange(NX - 1) * DX + xmin          # u-staggered x (NX-1)
    y = (np.arange(NY) + 0.5) * DY + ymin       # scalar y (NY)
    yv = np.arange(NY - 1) * DY + ymin          # v-staggered y (NY-1)
    z = (np.arange(NZ) + 0.5) * DZ + zmin       # scalar z (NZ)

    def _arr(value: float, *lengths: int) -> np.ndarray:
        return np.full((1, *lengths), value, dtype=np.float32)

    return xarray.Dataset(
        data_vars={
            "u": (("time", "z", "y", "xu"), _arr(U_CONST, NZ, NY, NX - 1)),
            "v": (("time", "z", "yv", "x"), _arr(V_CONST, NZ, NY - 1, NX)),
            "w": (("time", "z", "y", "x"), _arr(W_CONST, NZ, NY, NX)),
        },
        coords={"time": [0], "z": z, "y": y, "yv": yv, "x": x, "xu": xu},
    )


def test_write_warmstart_driver_static_case(tmp_path):
    """No pre-existing driver: a fresh file with only init_atmosphere_* fields."""
    driver = tmp_path / "urban_run_dynamic"
    write_warmstart_driver(driver, _make_state(), BOUNDS, NX, NY, NZ, pt_surface=300.0)

    ds = xarray.open_dataset(driver)

    # Staggered dim lengths PALM enforces (DRV0004).
    assert ds["init_atmosphere_u"].shape == (NZ, NY, NX - 1)      # (z, y, xu)
    assert ds["init_atmosphere_v"].shape == (NZ, NY - 1, NX)      # (z, yv, x)
    assert ds["init_atmosphere_w"].shape == (NZ - 1, NY, NX)      # (zw, y, x)
    assert ds["init_atmosphere_pt"].shape == (NZ, NY, NX)         # (z, y, x)
    assert ds["init_atmosphere_u"].dims == ("z", "y", "xu")
    assert ds["init_atmosphere_v"].dims == ("z", "yv", "x")
    assert ds["init_atmosphere_w"].dims == ("zw", "y", "x")

    # lod=2 attribute on every init field (the reader gates LOD=2 on this).
    for name in ("u", "v", "w", "pt"):
        assert int(ds[f"init_atmosphere_{name}"].attrs["lod"]) == 2

    # Constant input fields round-trip through interpolation unchanged.
    assert np.allclose(ds["init_atmosphere_u"].values, U_CONST, atol=1e-4)
    assert np.allclose(ds["init_atmosphere_v"].values, V_CONST, atol=1e-4)
    assert np.allclose(ds["init_atmosphere_w"].values, W_CONST, atol=1e-4)
    assert np.allclose(ds["init_atmosphere_pt"].values, 300.0)


def test_vertical_axes_are_zero_based(tmp_path):
    """z/zw are value-checked by PALM (DRV0005): must be 0-based zu/zw."""
    driver = tmp_path / "urban_run_dynamic"
    write_warmstart_driver(driver, _make_state(), BOUNDS, NX, NY, NZ, pt_surface=300.0)
    ds = xarray.open_dataset(driver)

    expected_z = (np.arange(NZ) + 0.5) * DZ      # zu(k)=(k-0.5)*dz, 0-based
    expected_zw = np.arange(1, NZ) * DZ          # zw(k)=k*dz, 0-based
    assert np.allclose(ds["z"].values, expected_z, atol=0.1 * DZ)
    assert np.allclose(ds["zw"].values, expected_zw, atol=0.1 * DZ)
    # Crucially NOT offset by zmin (=4.0): the state is physical, the file 0-based.
    assert ds["z"].values[0] < BOUNDS[2][0]


def test_no_fill_or_nan_values(tmp_path):
    """PALM rejects any field containing its _FillValue (DRV0008)."""
    driver = tmp_path / "urban_run_dynamic"
    write_warmstart_driver(driver, _make_state(), BOUNDS, NX, NY, NZ, pt_surface=300.0)
    ds = xarray.open_dataset(driver, mask_and_scale=False)
    for name in ("u", "v", "w", "pt"):
        vals = ds[f"init_atmosphere_{name}"].values
        assert np.all(np.isfinite(vals))
        assert not np.any(vals == _FILL)


def test_merges_with_existing_inflow_driver(tmp_path):
    """Warm-start augments a time-varying inflow driver, keeping both var sets."""
    driver = tmp_path / "urban_run_dynamic"

    # A minimal inflow driver as the post-change writer produces it: 0-based
    # z/zw, plus an inflow_plane_u plane and a time_inflow axis.
    z = (np.arange(NZ) + 0.5) * DZ
    zw = np.arange(1, NZ) * DZ
    y = (np.arange(NY) + 0.5) * DY
    inflow = xarray.Dataset(
        data_vars={
            "inflow_plane_u": (
                ("time_inflow", "z", "y"),
                np.zeros((2, NZ, NY), dtype=np.float32),
            ),
        },
        coords={
            "time_inflow": ("time_inflow", np.array([0.0, 5.0], dtype=np.float32)),
            "z": ("z", z.astype(np.float32)),
            "zw": ("zw", zw.astype(np.float32)),
            "y": ("y", y.astype(np.float32)),
        },
    )
    inflow.to_netcdf(driver)

    write_warmstart_driver(driver, _make_state(), BOUNDS, NX, NY, NZ, pt_surface=300.0)
    ds = xarray.open_dataset(driver)

    # Both the inflow plane and the warm-start init fields survive.
    assert "inflow_plane_u" in ds.data_vars
    assert "init_atmosphere_u" in ds.data_vars
    assert "time_inflow" in ds.coords
    # Shared vertical axis stayed 0-based.
    assert np.allclose(ds["z"].values, z, atol=1e-5)
    assert ds["init_atmosphere_u"].shape == (NZ, NY, NX - 1)


def test_drops_time_and_ensemble_dims(tmp_path):
    """A multi-frame, ensembled state is reduced to one frame before writing."""
    state = _make_state().expand_dims(ensemble=3)
    driver = tmp_path / "urban_run_dynamic"
    write_warmstart_driver(driver, state, BOUNDS, NX, NY, NZ, pt_surface=300.0)
    ds = xarray.open_dataset(driver)
    assert "ensemble" not in ds.dims
    assert "time" not in ds.dims
    assert ds["init_atmosphere_u"].shape == (NZ, NY, NX - 1)
