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

import copy
import pathlib
from types import SimpleNamespace
from typing import Optional

import numpy as np
import pytest
import xarray as xr

torch = pytest.importorskip("torch")
trimesh = pytest.importorskip("trimesh")

from neural_surrogates import (
    NeuralSurrogateEnsembleForwardModel,
    NeuralSurrogateForwardModel,
    UNetConvNeXt,
)
from neural_surrogates import ensemble_forward_model as ens_mod
from neural_surrogates import forward_model as fm_mod
from neural_surrogates.geometry import stl_to_fluid_mask

from pyurbanair.base_ensemble_forward_model import BaseEnsembleForwardModel
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
    assert out.sizes["time"] == 3  # 4.0 / 2.0 outputs, plus the t=0 frame


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
        param_vars=(
            "inflow_angle",
            "velocity_magnitude",
            "pressure_gradient_magnitude",
        ),
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
    assert out.sizes["time"] == 4  # 3 output frames plus the t=0 frame


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


def test_output_frequency_finer_than_trained_raises() -> None:
    # The surrogate cannot emit between trained network steps.
    with pytest.raises(ValueError, match="finer than the trained step"):
        _make_model(output_frequency=0.3, trained_output_frequency=0.5)


def test_substeps_resolution() -> None:
    model = _make_model(output_frequency=1.0, trained_output_frequency=0.5)
    assert model.substeps == 2


def test_mismatched_step_size_emits_only_at_requested_frequency() -> None:
    # Requested cadence is not an integer multiple of the trained step; the
    # network still steps at its trained cadence but emits at the requested one.
    model = _make_model(
        simulation_time=2.1,
        output_frequency=0.7,
        trained_output_frequency=0.5,
    )
    # 4 internal steps (round(2.1/0.5)); 3 emitted frames (round(2.1/0.7)).
    n_internal, emit_steps = model._output_schedule()
    assert n_internal == 4
    assert len(emit_steps) == 3
    out = model(params=_params())
    assert out.sizes["time"] == 4  # 3 emitted frames plus the t=0 frame


def test_cold_start_rollout_shape_and_spinup() -> None:
    model = _make_model()
    out = model(params=_params())
    assert out is not None
    # simulation_time / output_frequency output frames, plus the t=0 frame.
    assert out.sizes["time"] == 4
    for v in STATE_VARS:
        assert out[v].dims == ("time", "z", "y", "x")
    # Cold start must consult the spin-up backend exactly once.
    assert model.spinup_forward_model.calls == 1


def test_substeps_emit_correct_number_of_frames() -> None:
    model = _make_model(output_frequency=1.0, trained_output_frequency=0.5)
    out = model(params=_params())
    # 3 output frames (plus the t=0 frame) even though the network takes 6
    # internal steps.
    assert out.sizes["time"] == 4


def _staggered_udales_state() -> xr.Dataset:
    """Synthetic pyudales-style staggered C-grid snapshot."""
    rng = np.random.default_rng(1)
    coords = {
        "time": [0],
        "xt": np.arange(NX) + 0.5,
        "yt": np.arange(NY) + 0.5,
        "zt": np.arange(NZ) + 0.5,
        "xm": np.arange(NX).astype(float),
        "ym": np.arange(NY).astype(float),
        "zm": np.arange(NZ).astype(float),
    }
    return xr.Dataset(
        {
            "u": (("time", "zt", "yt", "xm"), rng.standard_normal((1, NZ, NY, NX))),
            "v": (("time", "zt", "ym", "xt"), rng.standard_normal((1, NZ, NY, NX))),
            "w": (("time", "zm", "yt", "xt"), rng.standard_normal((1, NZ, NY, NX))),
        },
        coords=coords,
    )


def test_staggered_spinup_state_is_collocated_before_network() -> None:
    """A staggered spin-up field is interpolated to the regular (z,y,x) grid."""
    regular = NeuralSurrogateForwardModel._to_regular_grid(_staggered_udales_state())
    for v in STATE_VARS:
        assert regular[v].dims == ("time", "z", "y", "x")
    assert {"xm", "ym", "zm"}.isdisjoint(regular.dims)
    assert {"x", "y", "z"}.issubset(regular.coords)


class _StaggeredStubSpinup(_StubSpinup):
    """Spin-up stub that returns a staggered C-grid field, like pyudales."""

    def run_single(self, state=None, params=None, sim_name="state") -> xr.Dataset:
        self.calls += 1
        return _staggered_udales_state()


def test_cold_start_collocates_spinup_output() -> None:
    model = _make_model(spinup_forward_model=_StaggeredStubSpinup())
    out = model(params=_params())
    # Output lands on the regular grid even though spin-up was staggered.
    for v in STATE_VARS:
        assert out[v].dims == ("time", "z", "y", "x")


def test_warm_start_skips_spinup() -> None:
    model = _make_model()
    cold = model(params=_params())
    model.spinup_forward_model.calls = 0
    warm = model(state=cold, params=_params())
    assert warm.sizes["time"] == 4  # 3 output frames plus the t=0 frame
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
    assert out.sizes["time"] == 4  # 3 output frames plus the t=0 frame


# -- batched rollout + ensemble (parallel spin-up) --------------------------


def _regular_snapshot(seed: int) -> xr.Dataset:
    """A single-snapshot regular-grid field, like a collocated spin-up state."""
    rng = np.random.default_rng(seed)
    coords = {
        "z": np.arange(NZ) + 0.5,
        "y": np.arange(NY) + 0.5,
        "x": np.arange(NX) + 0.5,
    }
    return xr.Dataset(
        {v: (("z", "y", "x"), rng.standard_normal((NZ, NY, NX))) for v in STATE_VARS},
        coords=coords,
    )


def _member_params(velocity: float) -> xr.Dataset:
    t = np.linspace(0.0, 3.0, 4)
    return xr.Dataset(
        {
            "inflow_angle": ("time", np.linspace(10.0, 20.0, t.size)),
            "velocity_magnitude": ("time", np.full(t.size, velocity)),
        },
        coords={"time": t},
    )


def test_rollout_batched_matches_per_member() -> None:
    """A batched rollout equals rolling each member out on its own.

    Guards against cross-member leakage in the batched forward pass: with the
    network shared and the maths batched over dim 0, member b's trajectory
    must depend only on member b's initial field and parameters.
    """
    model = _make_model()
    templates = [_regular_snapshot(i) for i in range(3)]
    params = [_member_params(v) for v in (2.0, 3.0, 4.0)]

    batched = model.rollout_batched(templates, params)
    assert len(batched) == 3
    for i in range(3):
        single = model.rollout_batched([templates[i]], [params[i]])[0]
        for v in STATE_VARS:
            np.testing.assert_allclose(
                batched[i][v].values, single[v].values, rtol=1e-5, atol=1e-5
            )


def _ensemble_stub_factory(tmp_path: pathlib.Path):
    """Spin-up stub carrying a ``dirs.temp_dir`` so the ensemble can build."""

    def make_stub() -> _StubSpinup:
        stub = _StubSpinup()
        stub.dirs = SimpleNamespace(temp_dir=tmp_path)
        return stub

    return make_stub


class _FakeBackendEnsemble(BaseEnsembleForwardModel):
    """Stand-in for pyudales/pylbm ``EnsembleForwardModel`` (no CFD, no MATLAB)."""

    def _create_new_forward_model(
        self, forward_model, experiment_base_dir, experiment_name
    ):
        return copy.copy(forward_model)

    def _pre_run_ensemble(self, *args, **kwargs) -> None:
        pass

    def _post_run_ensemble(self, *args, **kwargs) -> None:
        pass


def _patch_backend(monkeypatch, make_stub) -> None:
    """Detach the surrogate ensemble from the real backend packages."""
    # clone_for_member would import the backend's create_new_forward_model.
    monkeypatch.setattr(
        fm_mod, "_clone_backend_forward_model", lambda fm, base, name: make_stub()
    )
    # _get_spinup_ensemble imports "{backend}.ensemble_forward_model".
    fake_module = SimpleNamespace(EnsembleForwardModel=_FakeBackendEnsemble)
    monkeypatch.setattr(ens_mod, "import_module", lambda name: fake_module)


def test_ensemble_parallel_spinup_and_batched_rollout(tmp_path, monkeypatch) -> None:
    """Cold start: every member is spun up (in parallel) then rolled out batched."""
    make_stub = _ensemble_stub_factory(tmp_path)
    _patch_backend(monkeypatch, make_stub)

    template = _make_model(spinup_forward_model=make_stub())
    ensemble = NeuralSurrogateEnsembleForwardModel(template, ensemble_size=3)

    out = ensemble.run_ensemble(params=_params())
    assert out.sizes["ensemble"] == 3
    assert out.sizes["time"] == 4  # 3 output frames plus the t=0 frame
    for v in STATE_VARS:
        assert out[v].dims == ("ensemble", "time", "z", "y", "x")

    # Each member's spin-up backend ran exactly once.
    for member in ensemble.ensemble_forward_models:
        assert member.spinup_forward_model.calls == 1


def test_ensemble_warm_start_skips_spinup(tmp_path, monkeypatch) -> None:
    """Warm start: provided per-member states roll forward, no spin-up."""
    make_stub = _ensemble_stub_factory(tmp_path)
    _patch_backend(monkeypatch, make_stub)

    template = _make_model(spinup_forward_model=make_stub())
    ensemble = NeuralSurrogateEnsembleForwardModel(template, ensemble_size=3)

    warm = xr.concat(
        [_regular_snapshot(i) for i in range(3)], dim="ensemble", join="override"
    )
    out = ensemble.run_ensemble(state=warm, params=_params())
    assert out.sizes["ensemble"] == 3
    assert out.sizes["time"] == 4
    for member in ensemble.ensemble_forward_models:
        assert member.spinup_forward_model.calls == 0
