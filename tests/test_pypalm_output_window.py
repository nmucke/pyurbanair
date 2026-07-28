"""PALM's short-3D-file handling must fail loudly, not fabricate the window.

PALM signals its own numerical divergence by terminating with **exit 0** (see
``ForwardModel._locate_3d_output``). When it diverges *after* the first 3D dump
the only evidence is a short 3D file, so the wrapper must not quietly pad the
missing frames by repeating the last one — that produces a state that "runs a
few steps and then stays constant" while reporting success.
"""

from __future__ import annotations

import pathlib
import subprocess
import types

import numpy as np
import pytest
import xarray


def _state(n_time: int) -> xarray.Dataset:
    """A minimal PALM-shaped state with distinguishable frames."""
    return xarray.Dataset(
        {"u": (("time", "z", "y", "x"), np.arange(n_time * 8).reshape(n_time, 2, 2, 2))}
    )


def _fit(state: xarray.Dataset, *, sim: float, freq: float, spinup: float = 0.0):
    from pypalm.forward_model import ForwardModel

    stub = types.SimpleNamespace(
        spinup_time=spinup,
        output_frequency=freq,
        simulation_time=sim,
        experiment_name="urban_run",
    )
    return ForwardModel._fit_output_window(stub, state)  # type: ignore[arg-type]


def test_exact_count_is_untouched() -> None:
    out = _fit(_state(60), sim=300.0, freq=5.0)
    assert out.sizes["time"] == 60
    assert np.array_equal(out["u"].values, _state(60)["u"].values)


def test_extra_outputs_are_trimmed_from_the_front() -> None:
    out = _fit(_state(62), sim=300.0, freq=5.0)
    assert out.sizes["time"] == 60
    # keeps the LAST 60
    assert np.array_equal(out["u"].values, _state(62)["u"].values[2:])


def test_one_missing_frame_is_padded() -> None:
    """The documented adaptive-timestep case stays tolerated."""
    out = _fit(_state(59), sim=300.0, freq=5.0)
    assert out.sizes["time"] == 60
    # last frame repeated
    assert np.array_equal(out["u"].values[-1], out["u"].values[-2])


def test_many_missing_frames_raise_instead_of_padding() -> None:
    """The divergence case: refuse to fabricate 54 of 60 frames."""
    with pytest.raises(subprocess.CalledProcessError) as exc:
        _fit(_state(6), sim=300.0, freq=5.0)
    assert "6 of 60" in str(exc.value.output)
    assert "diverged" in str(exc.value.output)


def test_spinup_frames_are_dropped_before_the_check() -> None:
    """spinup frames are stripped first, so the count is of the window only."""
    out = _fit(_state(70), sim=300.0, freq=5.0, spinup=50.0)
    assert out.sizes["time"] == 60


# ---------------------------------------------------------------------------
# Multigrid coarsening predictor — 1 level means PALM cannot coarsen at all,
# which is what crippled the pressure solver on the 75x60x25 grid.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "nx,ny,nz,npex,expect_degenerate",
    [
        (75, 60, 25, 5, True),  # odd nx -> PALM reported "number of grid levels: 1"
        (75, 60, 64, 5, True),  # finer dz does NOT rescue an odd nx
        (64, 64, 32, 1, False),
        (64, 64, 32, 4, False),
        (64, 64, 64, 4, False),  # the verified-good recipe
    ],
)
def test_multigrid_levels_flags_uncoarsenable_grids(
    nx: int, ny: int, nz: int, npex: int, expect_degenerate: bool
) -> None:
    from pypalm.utils.ncpu_utils import multigrid_levels

    assert (multigrid_levels(nx, ny, nz, npex=npex) <= 1) is expect_degenerate


def test_multigrid_levels_accounts_for_the_x_subdomain() -> None:
    """Coarsening happens on the decomposed grid, so npex matters."""
    from pypalm.utils.ncpu_utils import multigrid_levels

    # 80 points coarsens fine undecomposed (80 = 16*5), and still does at 5
    # ranks (16 per rank) -- but 16 ranks leave 5 points per rank, which is odd,
    # so x cannot be halved even once.
    assert multigrid_levels(80, 64, 64, npex=1) > 1
    assert multigrid_levels(80, 64, 64, npex=5) > 1
    assert multigrid_levels(80, 64, 64, npex=16) <= 1
