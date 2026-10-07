"""Forward-model domain-check relaxation for a domain-flexible surrogate.

A :class:`~neural_surrogates.DomainDecomposed` model advertises
``domain_flexible = True`` and decomposes the global grid internally, so the
forward model may run it on a grid whose size/extent differs from the trained
one -- as long as the cell spacing ``(dx, dy, dz)`` matches. These tests cover:

1. a global grid LARGER than the trained grid but with the SAME spacing
   constructs and rolls out to a trajectory of the right shape,
2. a grid with a DIFFERENT spacing raises a spacing-related error, and
3. a non-flexible model (``UNetConvNeXt``) on a mismatched grid still raises
   the original domain-mismatch error (regression guard).
"""

from __future__ import annotations

import pathlib
from typing import Optional

import numpy as np
import pytest
import xarray as xr

torch = pytest.importorskip("torch")

from neural_surrogates import (
    DomainDecomposed,
    NeuralSurrogateForwardModel,
    UNetConvNeXt,
)

from pyurbanair.base_forward_model import BaseForwardModel

# Trained grid: spacing is 1.0 on every axis (bounds == dims).
T_NZ, T_NY, T_NX = 4, 8, 8
STATE_VARS = ("u", "v", "w")
PARAM_VARS = ("inflow_angle", "velocity_magnitude")


class _GridStubSpinup(BaseForwardModel):
    """Spin-up backend returning a developed pylbm-style field of any size."""

    def __init__(
        self, nz: int, ny: int, nx: int, results_dir: Optional[pathlib.Path] = None
    ) -> None:
        super().__init__(results_dir=results_dir)
        self.nz, self.ny, self.nx = nz, ny, nx
        self.spinup_time = 0.0
        self.dirs = None
        self.calls = 0
        self._rng = np.random.default_rng(0)

    def _apply_inflow_settings(self, params: xr.Dataset) -> None:
        pass

    def save_results(self, state: xr.Dataset, sim_name: str = "state") -> None:
        self._save_results(state, sim_name)

    def _clean_output(self) -> None:
        pass

    def disable_spinup(self) -> None:
        self.spinup_time = 0.0

    def run_single(self, state=None, params=None, sim_name="state") -> xr.Dataset:
        self.calls += 1
        coords = {
            "z": np.arange(self.nz) + 0.5,
            "y": np.arange(self.ny) + 0.5,
            "x": np.arange(self.nx) + 0.5,
            "time": [0],
        }
        data = {
            v: (
                ("time", "z", "y", "x"),
                self._rng.standard_normal((1, self.nz, self.ny, self.nx)),
            )
            for v in STATE_VARS
        }
        return xr.Dataset(data, coords=coords)


def _domain_decomposed() -> DomainDecomposed:
    """A tiny domain-decomposed model so the rollout stays fast."""
    return DomainDecomposed(
        n_state_channels=len(STATE_VARS),
        n_params=len(PARAM_VARS),
        decomposition=dict(
            interior_size=8,
            halo=2,
            taper=2,
            coarsen_factor=2,
            n_pos=3,
        ),
        fine_net=dict(
            _target_="neural_surrogates.UNetConvNeXt",
            base_channels=4, channel_mults=[1, 2], depths=[1, 1], kernel_size=3,
            residual=True,
        ),
        coarse_net=dict(
            _target_="neural_surrogates.UNetConvNeXt",
            base_channels=4, channel_mults=[1, 2], depths=[1, 1], kernel_size=3,
            residual=True,
        ),
    )


def _unet() -> UNetConvNeXt:
    return UNetConvNeXt(
        n_state_channels=len(STATE_VARS),
        n_params=len(PARAM_VARS),
        base_channels=4,
        channel_mults=(1, 2),
        depths=(1, 1),
        kernel_size=3,
        expansion=2,
    )


def _trained_domain() -> dict:
    # bounds ordered (x, y, z); spacing 1.0 on every axis.
    return {
        "nx": T_NX,
        "ny": T_NY,
        "nz": T_NZ,
        "bounds": [[0.0, T_NX], [0.0, T_NY], [0.0, T_NZ]],
    }


def _make_model(architecture, *, nx, ny, nz, bounds, **overrides):
    kwargs = dict(
        architecture=architecture,
        spinup_forward_model=_GridStubSpinup(nz, ny, nx),
        nx=nx,
        ny=ny,
        nz=nz,
        bounds=bounds,
        simulation_time=3.0,
        output_frequency=1.0,
        trained_output_frequency=1.0,
        trained_domain=_trained_domain(),
        state_vars=STATE_VARS,
        param_vars=PARAM_VARS,
        spinup_time=0.0,
        allow_uninitialized_weights=True,
    )
    kwargs.update(overrides)
    return NeuralSurrogateForwardModel(**kwargs)


def _params() -> xr.Dataset:
    t = np.linspace(0.0, 3.0, 4)
    return xr.Dataset(
        {
            "inflow_angle": ("time", np.linspace(10.0, 20.0, t.size)),
            "velocity_magnitude": ("time", np.linspace(3.0, 4.0, t.size)),
        },
        coords={"time": t},
    )


def test_flexible_larger_grid_same_spacing_runs() -> None:
    """A LARGER global grid at the trained spacing is allowed and rolls out."""
    # Double every axis, keep spacing == 1.0 (bounds == dims).
    g_nz, g_ny, g_nx = 2 * T_NZ, 2 * T_NY, 2 * T_NX
    model = _make_model(
        _domain_decomposed(),
        nx=g_nx,
        ny=g_ny,
        nz=g_nz,
        bounds=[[0.0, g_nx], [0.0, g_ny], [0.0, g_nz]],
    )

    out = model.run_single(params=_params())
    # 3 predicted frames at the larger global grid.
    assert out.sizes["time"] == 3
    for v in STATE_VARS:
        assert out[v].dims == ("time", "z", "y", "x")
        assert out[v].shape == (3, g_nz, g_ny, g_nx)


def test_flexible_different_spacing_raises() -> None:
    """A grid with a different cell spacing raises a spacing-related error."""
    # Same extent as trained but more cells -> finer spacing.
    g_nz, g_ny, g_nx = T_NZ, T_NY, 2 * T_NX
    with pytest.raises(ValueError, match="spacing"):
        _make_model(
            _domain_decomposed(),
            nx=g_nx,
            ny=g_ny,
            nz=g_nz,
            bounds=[[0.0, T_NX], [0.0, T_NY], [0.0, T_NZ]],
        )


def test_non_flexible_grid_mismatch_still_raises() -> None:
    """A non-flexible UNet on a mismatched grid keeps the strict check."""
    with pytest.raises(ValueError, match="does not match the domain"):
        _make_model(
            _unet(),
            nx=T_NX + 1,
            ny=T_NY,
            nz=T_NZ,
            bounds=[[0.0, T_NX + 1], [0.0, T_NY], [0.0, T_NZ]],
        )
