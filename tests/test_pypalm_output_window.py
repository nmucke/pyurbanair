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
