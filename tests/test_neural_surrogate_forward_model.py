"""Unit tests for :class:`NeuralSurrogateForwardModel`.

These cover the four behaviours the surrogate must get right without
needing a trained checkpoint or a real CFD solver:

* the requested domain must match the trained domain,
* the requested output cadence must be a multiple of the trained step size,
* a geometry channel can be voxelised from an ``.stl`` file, and
* the autoregressive rollout produces a well-formed state trajectory for
  both cold (spin-up) and warm starts.

A lightweight stub stands in for the spin-up CFD backend so the rollout
loop and bookkeeping are exercised in milliseconds.
"""

from __future__ import annotations

import pathlib
from typing import Optional

import numpy as np
import pytest
import xarray as xr

torch = pytest.importorskip("torch")
trimesh = pytest.importorskip("trimesh")

from neural_surrogates import NeuralSurrogateForwardModel, UNetConvNeXt
from neural_surrogates.geometry import stl_to_fluid_mask
from pyurbanair.base_forward_model import BaseForwardModel

NZ, NY, NX = 4, 8, 8
STATE_VARS = ("u", "v", "w")
PARAM_VARS = ("inflow_angle", "velocity_magnitude")


class _StubSpinup(BaseForwardModel):
    """Synthetic spin-up backend returning a developed pylbm-style field."""

    def __init__(self, results_dir: Optional[pathlib.Path] = None) -> None:
        super().__init__(results_dir=results_dir)
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
            "z": np.arange(NZ) + 0.5,
            "y": np.arange(NY) + 0.5,
            "x": np.arange(NX) + 0.5,
            "time": [0],
        }
        data = {
            v: (("time", "z", "y", "x"), self._rng.standard_normal((1, NZ, NY, NX)))
            for v in STATE_VARS
        }
        return xr.Dataset(data, coords=coords)


def _architecture() -> UNetConvNeXt:
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
    return {
        "nx": NX,
        "ny": NY,
        "nz": NZ,
        "bounds": [[0.0, NX], [0.0, NY], [0.0, NZ]],
    }


def _make_model(**overrides) -> NeuralSurrogateForwardModel:
    kwargs = dict(
        architecture=_architecture(),
        spinup_forward_model=_StubSpinup(),
        nx=NX,
        ny=NY,
        nz=NZ,
        bounds=[[0.0, NX], [0.0, NY], [0.0, NZ]],
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


def test_resolves_everything_from_model_dir(
    tmp_path, surrogate_model_dir_factory
) -> None:
    """Architecture, vars, trained grid + cadence all come from the folder."""
    model_dir = surrogate_model_dir_factory(
        tmp_path,
        domain={
            "nx": NX,
            "ny": NY,
            "nz": NZ,
            "bounds": [[0.0, NX], [0.0, NY], [0.0, NZ]],
        },
        time={"simulation_time": 5.0, "output_frequency": 2.0, "spinup_time": 0.0},
        state_vars=STATE_VARS,
        param_vars=PARAM_VARS,
    )

    model = NeuralSurrogateForwardModel(
        spinup_forward_model=_StubSpinup(),
        nx=NX,
        ny=NY,
        nz=NZ,
        bounds=[[0.0, NX], [0.0, NY], [0.0, NZ]],
        simulation_time=4.0,
        output_frequency=2.0,  # equals trained cadence -> substeps == 1
        model_dir=model_dir,
    )

    assert model.state_vars == STATE_VARS
    assert model.param_vars == PARAM_VARS
    assert model.trained_output_frequency == 2.0
    assert model.substeps == 1
    assert isinstance(model.model, UNetConvNeXt)

    out = model(params=_params())
    assert out.sizes["time"] == 2  # 4.0 / 2.0


def test_model_dir_domain_mismatch_raises(
    tmp_path, surrogate_model_dir_factory
) -> None:
    """The trained domain read from the folder is enforced."""
    model_dir = surrogate_model_dir_factory(
        tmp_path,
        domain={
            "nx": NX + 4,  # trained on a wider grid than requested
            "ny": NY,
            "nz": NZ,
            "bounds": [[0.0, NX + 4], [0.0, NY], [0.0, NZ]],
        },
        time={"simulation_time": 5.0, "output_frequency": 1.0, "spinup_time": 0.0},
        state_vars=STATE_VARS,
        param_vars=PARAM_VARS,
    )
    with pytest.raises(ValueError, match="does not match the domain"):
        NeuralSurrogateForwardModel(
            spinup_forward_model=_StubSpinup(),
            nx=NX,
            ny=NY,
            nz=NZ,
            bounds=[[0.0, NX], [0.0, NY], [0.0, NZ]],
            simulation_time=4.0,
            output_frequency=1.0,
            model_dir=model_dir,
        )


def test_default_params_fill_missing_trained_param(tmp_path) -> None:
    """A trained param the caller omits is filled from default_params."""
    model = _make_model(
        param_vars=("inflow_angle", "velocity_magnitude", "pressure_gradient_magnitude"),
        architecture=UNetConvNeXt(
            n_state_channels=len(STATE_VARS),
            n_params=3,
            base_channels=4,
            channel_mults=(1, 2),
            depths=(1, 1),
            kernel_size=3,
            expansion=2,
        ),
        default_params={"pressure_gradient_magnitude": 0.004},
    )
    # _params() has no pressure_gradient_magnitude; the default fills it in.
    out = model(params=_params())
    assert out.sizes["time"] == 3


def test_missing_resolution_raises() -> None:
    """Without model_dir or explicit metadata, construction fails loudly."""
    with pytest.raises(ValueError, match="could not resolve"):
        NeuralSurrogateForwardModel(
            spinup_forward_model=_StubSpinup(),
            nx=NX,
            ny=NY,
            nz=NZ,
            bounds=[[0.0, NX], [0.0, NY], [0.0, NZ]],
            simulation_time=4.0,
            output_frequency=1.0,
        )


def test_domain_mismatch_raises() -> None:
    with pytest.raises(ValueError, match="does not match the domain"):
        _make_model(nx=NX + 1)


def test_bounds_mismatch_raises() -> None:
    with pytest.raises(ValueError, match="bounds"):
        _make_model(bounds=[[0.0, NX + 5], [0.0, NY], [0.0, NZ]])


def test_output_frequency_must_be_multiple_of_trained_step() -> None:
    with pytest.raises(ValueError, match="integer multiple"):
        _make_model(output_frequency=0.7, trained_output_frequency=0.5)


def test_substeps_resolution() -> None:
    model = _make_model(output_frequency=1.0, trained_output_frequency=0.5)
    assert model.substeps == 2


def test_cold_start_rollout_shape_and_spinup() -> None:
    model = _make_model()
    out = model(params=_params())
    assert out is not None
    assert out.sizes["time"] == 3  # simulation_time / output_frequency
    for v in STATE_VARS:
        assert out[v].dims == ("time", "z", "y", "x")
    # Cold start must consult the spin-up backend exactly once.
    assert model.spinup_forward_model.calls == 1


def test_substeps_emit_correct_number_of_frames() -> None:
    model = _make_model(output_frequency=1.0, trained_output_frequency=0.5)
    out = model(params=_params())
    # 3 output frames even though the network takes 6 internal steps.
    assert out.sizes["time"] == 3


def test_warm_start_skips_spinup() -> None:
    model = _make_model()
    cold = model(params=_params())
    model.spinup_forward_model.calls = 0
    warm = model(state=cold, params=_params())
    assert warm.sizes["time"] == 3
    assert model.spinup_forward_model.calls == 0


def test_stl_geometry_voxelisation(tmp_path: pathlib.Path) -> None:
    # A unit cube centred in the domain; its interior cells become obstacles.
    box = trimesh.creation.box(extents=(4.0, 4.0, 4.0))
    box.apply_translation((NX / 2, NY / 2, NZ / 2))
    stl_path = tmp_path / "box.stl"
    box.export(stl_path)

    template_var = xr.DataArray(
        np.zeros((NZ, NY, NX)),
        dims=("z", "y", "x"),
        coords={
            "z": np.arange(NZ) + 0.5,
            "y": np.arange(NY) + 0.5,
            "x": np.arange(NX) + 0.5,
        },
    )
    mask = stl_to_fluid_mask(stl_path, template_var)
    assert mask.shape == (NZ, NY, NX)
    # Domain corner is fluid; the central cell sits inside the box.
    assert mask[0, 0, 0] == 1.0
    assert mask[NZ // 2, NY // 2, NX // 2] == 0.0


def test_stl_geometry_used_in_rollout(tmp_path: pathlib.Path) -> None:
    box = trimesh.creation.box(extents=(4.0, 4.0, 4.0))
    box.apply_translation((NX / 2, NY / 2, NZ / 2))
    stl_path = tmp_path / "box.stl"
    box.export(stl_path)

    model = _make_model(stl_path=stl_path)
    out = model(params=_params())
    assert out.sizes["time"] == 3
