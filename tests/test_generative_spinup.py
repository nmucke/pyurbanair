"""Tests for the generative spin-up (plan 07 phase 4, §5.4–5.5).

Covers :class:`neural_surrogates.generative_spinup.GenerativeSpinup` and its
integration into the surrogate forward model, the surrogate ensemble, the
ESMDA smoother lifecycle and ``scripts/esmda/run_esmda.py``.

Almost everything runs against an **instrumented stub generator** -- a tiny
``nn.Module`` with the real ``sample(...)`` signature that records every call
(conditioning + noise) and returns a deterministic function of the first
history knot and the noise -- injected through the ``GenerativeSpinup._load_model``
seam. The real :class:`TadpoleLatentGenerator` (which needs diffusers/timm) is
only exercised by one strict-reload sampling smoke test.
"""

from __future__ import annotations

import pathlib
from types import SimpleNamespace
from typing import Any, Optional

import numpy as np
import pytest
import xarray as xr

torch = pytest.importorskip("torch")

from neural_surrogates import (
    GenerativeSpinup,
    NeuralSurrogateEnsembleForwardModel,
    NeuralSurrogateForwardModel,
    UNetConvNeXt,
)
from neural_surrogates import ensemble_forward_model as ens_mod
from neural_surrogates.generative_spinup import _member_seed, constant_history
from omegaconf import OmegaConf

from pyurbanair.base_forward_model import BaseForwardModel

NZ, NY, NX = 4, 8, 8
GRID = (NZ, NY, NX)
BOUNDS = [[0.0, NX], [0.0, NY], [0.0, NZ]]
STATE_VARS = ("u", "v", "w")
PARAM_VARS = ("inflow_angle", "velocity_magnitude")
HP = 3
LATENT_DIM = 4
LATENT_STRIDE = 4


# ---------------------------------------------------------------------------
# Instrumented stub generator + fake artifact + template
# ---------------------------------------------------------------------------


class _StubGenerator(torch.nn.Module):
    """Records every ``sample`` call; output = first knot + upsampled noise.

    Exposes exactly the attributes :class:`GenerativeSpinup` reads off a real
    :class:`TadpoleLatentGenerator`. ``ae`` advertises no SDF features so no
    ``geom_features`` are built.
    """

    def __init__(
        self,
        n_state_channels: int = len(STATE_VARS),
        n_params: int = len(PARAM_VARS),
        param_history_steps: int = HP,
        num_sampling_steps: int = 50,
    ) -> None:
        super().__init__()
        self.n_state_channels = int(n_state_channels)
        self.n_params = int(n_params)
        self.param_history_steps = int(param_history_steps)
        self.num_sampling_steps = int(num_sampling_steps)
        self.state_latent_dim = LATENT_DIM
        self.ae = SimpleNamespace(n_geom_feature_channels=0, sdf_features_enabled=False)
        # A parameter so weights.pt / strict load are non-trivial.
        self.scale = torch.nn.Parameter(torch.ones(()))
        self.calls: list[dict[str, Any]] = []

    def latent_grid_for(self, grid):
        return tuple(-(-int(s) // LATENT_STRIDE) for s in grid)

    def sample(
        self,
        params_hist,
        geometry,
        geom_features=None,
        *,
        initial_noise=None,
        generator=None,
        num_steps=None,
    ):
        assert initial_noise is not None, "GenerativeSpinup must pass per-member noise"
        assert generator is None
        b = params_hist.shape[0]
        assert params_hist.shape == (b, self.param_history_steps, self.n_params)
        assert torch.isfinite(params_hist).all()
        grid = tuple(geometry.shape[1:])
        assert tuple(initial_noise.shape) == (
            b,
            LATENT_DIM,
            *self.latent_grid_for(grid),
        )
        self.calls.append(
            {
                "params_hist": params_hist.detach().cpu().clone(),
                "noise": initial_noise.detach().cpu().clone(),
                "geometry": geometry.detach().cpu().clone(),
                "geom_features": geom_features,
                "num_steps": num_steps,
                "batch": b,
            }
        )
        c = self.n_state_channels
        first = params_hist[:, 0, :]  # (B, P)
        cond = torch.stack([first[:, i % self.n_params] for i in range(c)], dim=1)
        noise = torch.nn.functional.interpolate(
            initial_noise[:, :c].to(torch.float32), size=grid, mode="nearest"
        )
        out = (cond[:, :, None, None, None] + noise) * self.scale
        return out * geometry.unsqueeze(1).to(out.dtype)


def _never_instantiate(*args, **kwargs):
    raise AssertionError("the CFD spin-up backend must not be instantiated")


class _NoopBackend(BaseForwardModel):
    """A built (never run) CFD backend stand-in for the non-generative modes."""

    def __init__(self) -> None:
        super().__init__(results_dir=None)
        self.dirs = None

    def _apply_inflow_settings(self, params) -> None:
        pass

    def save_results(self, state, sim_name="state") -> None:
        pass

    def _clean_output(self) -> None:
        pass

    def run_single(self, state=None, params=None, sim_name="state"):
        raise AssertionError("the CFD backend must not run")


def _dummy_backend_node():
    """A config node that explodes if the surrogate ever instantiates it."""
    return OmegaConf.create(
        {"_target_": "tests.test_generative_spinup._never_instantiate"}
    )


def _blanking(grid=GRID) -> np.ndarray:
    """Ground layer + a small block are obstacles."""
    nz, ny, nx = grid
    b = np.zeros(grid, dtype=np.float64)
    b[0] = 1.0
    b[: max(1, nz // 2), ny // 2 : ny // 2 + 2, nx // 2 : nx // 2 + 2] = 1.0
    return b


def _write_template(
    path: pathlib.Path,
    grid=GRID,
    bounds=BOUNDS,
    with_blanking: bool = True,
    with_time: bool = False,
    seed: int = 123,
) -> pathlib.Path:
    nz, ny, nx = grid
    dx = (bounds[0][1] - bounds[0][0]) / nx
    dy = (bounds[1][1] - bounds[1][0]) / ny
    dz = (bounds[2][1] - bounds[2][0]) / nz
    coords = {
        "z": bounds[2][0] + (np.arange(nz) + 0.5) * dz,
        "y": bounds[1][0] + (np.arange(ny) + 0.5) * dy,
        "x": bounds[0][0] + (np.arange(nx) + 0.5) * dx,
    }
    rng = np.random.default_rng(seed)
    dims = ("z", "y", "x")
    data: dict[str, Any] = {
        v: (dims, 100.0 + rng.standard_normal(grid)) for v in STATE_VARS
    }
    if with_blanking:
        data["blanking"] = (dims, _blanking(grid))
    ds = xr.Dataset(data, coords=coords)
    if with_time:
        ds = xr.concat([ds, ds], dim="time").assign_coords(time=[0.0, 1.0])
    path.parent.mkdir(parents=True, exist_ok=True)
    ds.to_netcdf(path)
    return path


def _write_artifact(
    model_dir: pathlib.Path,
    *,
    grid=GRID,
    bounds=BOUNDS,
    fluid_cells: Optional[int] = None,
    hp: int = HP,
    param_vars=PARAM_VARS,
    state_vars=STATE_VARS,
    saved_num_steps: int = 7,
    spacing_override: Optional[dict] = None,
    stub: Optional[_StubGenerator] = None,
) -> pathlib.Path:
    """A ``train_latent_generator.py``-shaped artifact around the stub."""
    nz, ny, nx = grid
    if fluid_cells is None:
        fluid_cells = int((1.0 - _blanking(grid)).sum())
    grid_block = {
        "nz": nz,
        "ny": ny,
        "nx": nx,
        "dz": (bounds[2][1] - bounds[2][0]) / nz,
        "dy": (bounds[1][1] - bounds[1][0]) / ny,
        "dx": (bounds[0][1] - bounds[0][0]) / nx,
        "bounds": [list(b) for b in bounds],
    }
    if spacing_override:
        grid_block.update(spacing_override)
    cfg = {
        "architecture": {
            "_target_": "tests.test_generative_spinup._StubGenerator",
            "param_history_steps": hp,
            "num_sampling_steps": saved_num_steps,
        },
        "dataset": {"root_dir": "/nonexistent", "state_vars": list(state_vars)},
        "generator": {
            "physical_schema": {
                "state_vars": list(state_vars),
                "param_vars": list(param_vars),
                "param_history_steps": hp,
                "history_dt_seconds": 5.0,
                "units": {"u": "m/s", "inflow_angle": "deg"},
                "geometry_mask_convention": "blanking: 1 = obstacle; fluid = 1 - blanking",
                "coordinate_order": ["z", "y", "x"],
                "grid": grid_block,
                "supported_geometries": [
                    {"shape": [nz, ny, nx], "fluid_cells": fluid_cells}
                ],
            },
            "ae_fingerprint": "deadbeef",
            "sampling": {"num_steps": saved_num_steps},
        },
    }
    model_dir.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(OmegaConf.create(cfg), model_dir / "config.yaml")
    stub = stub or _StubGenerator(
        param_history_steps=hp, num_sampling_steps=saved_num_steps
    )
    torch.save(stub.state_dict(), model_dir / "weights.pt")
    return model_dir


@pytest.fixture
def stub() -> _StubGenerator:
    return _StubGenerator()


@pytest.fixture
def inject_stub(monkeypatch, stub):
    """Route every GenerativeSpinup at the shared instrumented stub."""
    monkeypatch.setattr(GenerativeSpinup, "_load_model", lambda self, cfg, schema: stub)
    return stub


@pytest.fixture
def artifact(tmp_path) -> tuple[pathlib.Path, pathlib.Path]:
    return (
        _write_artifact(tmp_path / "model_dir"),
        _write_template(tmp_path / "template.nc"),
    )


def _params(velocity: float = 3.0, angle0: float = 10.0) -> xr.Dataset:
    """Time-varying member params; the first knot is (angle0, velocity)."""
    t = np.linspace(0.0, 3.0, 4)
    return xr.Dataset(
        {
            "inflow_angle": ("time", angle0 + np.linspace(0.0, 10.0, t.size)),
            "velocity_magnitude": ("time", np.full(t.size, velocity)),
        },
        coords={"time": t},
    )


def _first_knots(params: xr.Dataset) -> np.ndarray:
    """(N_e, P) first-knot values in PARAM_VARS order."""
    cols = []
    for name in PARAM_VARS:
        da = params[name]
        if "time" in da.dims:
            da = da.isel(time=0)
        cols.append(np.asarray(da.values, dtype=float))
    return np.stack(cols, axis=-1)


def _gs(artifact, **kw) -> GenerativeSpinup:
    model_dir, template = artifact
    return GenerativeSpinup(model_dir, template, **kw)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def test_constant_history_repeats_current_values() -> None:
    hist = constant_history(np.array([1.0, 2.0]), 4)
    assert hist.shape == (4, 2)
    assert np.all(hist == [1.0, 2.0])
    with pytest.raises(ValueError):
        constant_history(np.array([1.0]), 0)


def test_member_seed_is_injective_and_stable() -> None:
    seeds = {_member_seed(s, m) for s in range(3) for m in range(100)}
    assert len(seeds) == 300
    assert _member_seed(2, 5) == 2 * 1_000_003 + 5
    with pytest.raises(ValueError):
        _member_seed(0, -1)


# ---------------------------------------------------------------------------
# §5.4 sampling / noise policy
# ---------------------------------------------------------------------------


def test_noise_identical_across_batch_sizes_and_calls(artifact, inject_stub) -> None:
    """Per-member noise depends on (seed, member) only -- not on batching."""
    params = [_params(3.0), _params(4.0), _params(5.0)]
    one = _gs(artifact, sample_batch_size=1).generate(params, [0, 1, 2])
    n_calls_one = len(inject_stub.calls)
    assert n_calls_one == 3  # batched in chunks of 1
    three = _gs(artifact, sample_batch_size=3).generate(params, [0, 1, 2])
    assert len(inject_stub.calls) == n_calls_one + 1  # one batched call
    for a, b in zip(one, three):
        for v in STATE_VARS:
            np.testing.assert_allclose(a[v].values, b[v].values, rtol=1e-6, atol=1e-6)

    # The noise handed to the stub is the same tensor per member either way.
    per_member_single = [c["noise"][0] for c in inject_stub.calls[:3]]
    batched = inject_stub.calls[3]["noise"]
    for i in range(3):
        assert torch.equal(per_member_single[i], batched[i])

    # A second call regenerates (new stub call) with identical noise + output.
    again = _gs(artifact, sample_batch_size=3).generate(params, [0, 1, 2])
    assert len(inject_stub.calls) == n_calls_one + 2
    assert torch.equal(inject_stub.calls[-1]["noise"], batched)
    for a, b in zip(three, again):
        np.testing.assert_array_equal(a["u"].values, b["u"].values)


def test_distinct_members_get_distinct_noise(artifact, inject_stub) -> None:
    gs = _gs(artifact)
    gs.generate([_params(3.0)] * 3, [0, 1, 2])
    noise = inject_stub.calls[-1]["noise"]
    assert not torch.equal(noise[0], noise[1])
    assert not torch.equal(noise[1], noise[2])
    # Noise follows the member index, not the position in the batch.
    gs.generate([_params(3.0)], [2])
    assert torch.equal(inject_stub.calls[-1]["noise"][0], noise[2])
    # A different seed gives different noise for the same member.
    _gs(artifact, seed=1).generate([_params(3.0)], [0])
    assert not torch.equal(inject_stub.calls[-1]["noise"][0], noise[0])


def test_changed_params_rerun_generator_only_changing_that_member(
    artifact, inject_stub
) -> None:
    gs = _gs(artifact)
    before = gs.generate([_params(3.0), _params(4.0)], [0, 1])
    after = gs.generate([_params(3.0), _params(9.0)], [0, 1])
    assert len(inject_stub.calls) == 2  # regenerated, no result caching
    for v in STATE_VARS:
        np.testing.assert_array_equal(before[0][v].values, after[0][v].values)
    # velocity_magnitude is the stub's second conditioning channel -> ``v``.
    np.testing.assert_array_equal(before[1]["u"].values, after[1]["u"].values)
    assert not np.array_equal(before[1]["v"].values, after[1]["v"].values)
    # Same noise for member 1 across the two calls: only the conditioning moved.
    assert torch.equal(
        inject_stub.calls[0]["noise"][1], inject_stub.calls[1]["noise"][1]
    )
    np.testing.assert_allclose(
        inject_stub.calls[1]["params_hist"][1].numpy(),
        np.repeat([[10.0, 9.0]], HP, axis=0),
    )


def test_static_default_and_missing_params(artifact, inject_stub) -> None:
    # Static (no time dim) params are used as-is.
    static = xr.Dataset({"inflow_angle": 12.0, "velocity_magnitude": 2.5})
    _gs(artifact).generate([static], [0])
    np.testing.assert_allclose(
        inject_stub.calls[-1]["params_hist"][0].numpy(),
        np.repeat([[12.0, 2.5]], HP, axis=0),
    )
    # A missing trained parameter falls back to default_params.
    partial = xr.Dataset(
        {"inflow_angle": ("time", [1.0, 2.0])}, coords={"time": [0.0, 1.0]}
    )
    _gs(artifact, default_params={"velocity_magnitude": 7.0}).generate([partial], [0])
    np.testing.assert_allclose(
        inject_stub.calls[-1]["params_hist"][0].numpy(),
        np.repeat([[1.0, 7.0]], HP, axis=0),
    )
    # ... and raises, naming the member and the variable, without a default.
    with pytest.raises(ValueError, match="member 4.*velocity_magnitude"):
        _gs(artifact).generate([partial], [4])
    # Non-finite values are rejected.
    bad = xr.Dataset({"inflow_angle": np.nan, "velocity_magnitude": 1.0})
    with pytest.raises(ValueError, match="not finite"):
        _gs(artifact).generate([bad], [0])


def test_time_varying_params_condition_on_first_knot(artifact, inject_stub) -> None:
    params = _params(velocity=4.5, angle0=-20.0)
    _gs(artifact).generate([params], [0])
    hist = inject_stub.calls[-1]["params_hist"][0].numpy()
    assert hist.shape == (HP, len(PARAM_VARS))
    np.testing.assert_allclose(hist, np.repeat([[-20.0, 4.5]], HP, axis=0))
    # The later knots (angle ramps to -10) never enter the conditioning.
    assert not np.any(np.isclose(hist[:, 0], -10.0))


def test_output_on_canonical_coords_with_mask_and_zero_obstacles(
    artifact, inject_stub
) -> None:
    model_dir, template_path = artifact
    with xr.open_dataset(template_path) as ds:
        template = ds.load()
    out = _gs(artifact).generate([_params(3.0)], [0])[0]
    assert set(out.data_vars) == {*STATE_VARS, "blanking"}
    for v in STATE_VARS:
        assert out[v].dims == ("z", "y", "x")
        assert out[v].shape == GRID
        # The template's own velocity values are never used.
        assert not np.allclose(out[v].values, template[v].values)
    for axis in ("z", "y", "x"):
        np.testing.assert_array_equal(out[axis].values, template[axis].values)
    np.testing.assert_array_equal(out["blanking"].values, template["blanking"].values)
    obstacle = out["blanking"].values == 1.0
    for v in STATE_VARS:
        assert np.all(out[v].values[obstacle] == 0.0)
        assert np.all(np.isfinite(out[v].values))
    assert "time" not in out.dims and "time" not in out.coords


def test_template_with_time_axis_is_reduced_to_one_frame(tmp_path, inject_stub) -> None:
    model_dir = _write_artifact(tmp_path / "m")
    template = _write_template(tmp_path / "t.nc", with_time=True)
    out = GenerativeSpinup(model_dir, template).generate([_params()], [0])[0]
    assert "time" not in out.dims
    assert out["u"].dims == ("z", "y", "x")


def test_template_without_blanking_raises(tmp_path, inject_stub) -> None:
    model_dir = _write_artifact(tmp_path / "m")
    template = _write_template(tmp_path / "t.nc", with_blanking=False)
    with pytest.raises(ValueError, match="no 'blanking'"):
        GenerativeSpinup(model_dir, template).generate([_params()], [0])


def test_grid_shape_mismatch_raises(tmp_path, inject_stub) -> None:
    model_dir = _write_artifact(tmp_path / "m")
    wide = (NZ, NY, NX + 4)
    template = _write_template(
        tmp_path / "t.nc", grid=wide, bounds=[[0.0, NX + 4], [0.0, NY], [0.0, NZ]]
    )
    with pytest.raises(ValueError, match="does not match the generator's trained grid"):
        GenerativeSpinup(model_dir, template).generate([_params()], [0])


def test_grid_spacing_mismatch_raises(tmp_path, inject_stub) -> None:
    model_dir = _write_artifact(tmp_path / "m", spacing_override={"dx": 2.0})
    template = _write_template(tmp_path / "t.nc")
    with pytest.raises(ValueError, match="spacing"):
        GenerativeSpinup(model_dir, template).generate([_params()], [0])


def test_bounds_mismatch_raises(tmp_path, inject_stub) -> None:
    model_dir = _write_artifact(tmp_path / "m")
    shifted = [[10.0, 10.0 + NX], [0.0, NY], [0.0, NZ]]
    template = _write_template(tmp_path / "t.nc", bounds=shifted)
    with pytest.raises(ValueError, match="trained bounds"):
        GenerativeSpinup(model_dir, template).generate([_params()], [0])


def test_unsupported_geometry_raises(tmp_path, inject_stub) -> None:
    model_dir = _write_artifact(tmp_path / "m", fluid_cells=1)
    template = _write_template(tmp_path / "t.nc")
    with pytest.raises(ValueError, match="supported_geometries"):
        GenerativeSpinup(model_dir, template).generate([_params()], [0])


def test_state_var_dims_mismatch_raises(tmp_path, inject_stub) -> None:
    model_dir = _write_artifact(tmp_path / "m")
    template = _write_template(tmp_path / "t.nc")
    with xr.open_dataset(template) as ds:
        broken = ds.load()
    broken["v"] = broken["v"].transpose("x", "y", "z")
    broken.to_netcdf(template.with_name("broken.nc"))
    with pytest.raises(ValueError, match="coordinate order"):
        GenerativeSpinup(model_dir, template.with_name("broken.nc")).generate(
            [_params()], [0]
        )


def test_constructor_validation() -> None:
    with pytest.raises(ValueError, match="model_dir"):
        GenerativeSpinup(None, "t.nc")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="sample_batch_size"):
        GenerativeSpinup("m", "t.nc", sample_batch_size=0)
    with pytest.raises(ValueError, match="num_sampling_steps"):
        GenerativeSpinup("m", "t.nc", num_sampling_steps=0)
    with pytest.raises(FileNotFoundError):
        GenerativeSpinup("/nonexistent/model", "t.nc").hp


def test_real_loader_strict_load_and_sampling_steps(artifact) -> None:
    """The un-patched loader rebuilds ``architecture`` and loads strictly."""
    gs = _gs(artifact)
    assert gs.hp == HP and gs.param_vars == PARAM_VARS
    assert gs.history_dt_seconds == 5.0
    assert gs.num_sampling_steps == 7  # the artifact's validated default
    assert "Hp=3" in gs.describe() and "seed=0" in gs.describe()
    out = gs.generate([_params()], [0])
    assert isinstance(gs._model, _StubGenerator)
    assert gs._model.calls[-1]["num_steps"] == 7
    assert out[0]["u"].shape == GRID
    # An explicit override wins over the saved default.
    gs2 = _gs(artifact, num_sampling_steps=3)
    gs2.generate([_params()], [0])
    assert gs2._model.calls[-1]["num_steps"] == 3


def test_strict_load_rejects_mismatched_weights(tmp_path) -> None:
    model_dir = _write_artifact(tmp_path / "m")
    torch.save({"unexpected": torch.zeros(1)}, model_dir / "weights.pt")
    template = _write_template(tmp_path / "t.nc")
    with pytest.raises(RuntimeError):
        GenerativeSpinup(model_dir, template).generate([_params()], [0])


def test_generator_failure_names_member_and_conditioning(
    artifact, inject_stub, monkeypatch
) -> None:
    def boom(*args, **kwargs):
        raise RuntimeError("kaboom")

    monkeypatch.setattr(inject_stub, "sample", boom)
    with pytest.raises(RuntimeError, match=r"members \[5\].*inflow_angle.*kaboom"):
        _gs(artifact).generate([_params()], [5])


def test_diagnostics_dir_records_every_call(artifact, inject_stub, tmp_path) -> None:
    gs = _gs(artifact)
    gs.diagnostics_dir = tmp_path / "diag"
    gs.generate([_params(), _params()], [0, 1])
    gs.generate([_params()], [3])
    assert (tmp_path / "diag" / "call_0" / "member_0.nc").exists()
    assert (tmp_path / "diag" / "call_0" / "member_1.nc").exists()
    assert (tmp_path / "diag" / "call_1" / "member_3.nc").exists()
    # Re-pointing the directory restarts the call counter.
    gs.diagnostics_dir = tmp_path / "diag2"
    gs.generate([_params()], [0])
    assert (tmp_path / "diag2" / "call_0" / "member_0.nc").exists()
    gs.diagnostics_dir = None
    gs.generate([_params()], [0])
    assert not (tmp_path / "diag2" / "call_1").exists()


# ---------------------------------------------------------------------------
# Forward-model integration
# ---------------------------------------------------------------------------


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


def _make_model(artifact, **overrides) -> NeuralSurrogateForwardModel:
    model_dir, template = artifact
    kwargs: dict[str, Any] = dict(
        architecture=_architecture(),
        spinup_forward_model=_dummy_backend_node(),
        nx=NX,
        ny=NY,
        nz=NZ,
        bounds=BOUNDS,
        simulation_time=3.0,
        output_frequency=1.0,
        trained_output_frequency=1.0,
        trained_domain={"nx": NX, "ny": NY, "nz": NZ, "bounds": BOUNDS},
        state_vars=STATE_VARS,
        param_vars=PARAM_VARS,
        spinup_time=0.0,
        allow_uninitialized_weights=True,
        spinup_source="generative",
        generative_spinup={
            "model_dir": str(model_dir),
            "template_path": str(template),
            "seed": 0,
            "sample_batch_size": 8,
            "num_sampling_steps": None,
            "save_diagnostics": False,
        },
    )
    kwargs.update(overrides)
    return NeuralSurrogateForwardModel(**kwargs)


def test_forward_model_generative_requires_block(artifact) -> None:
    with pytest.raises(ValueError, match="generative_spinup"):
        _make_model(artifact, generative_spinup=None)
    with pytest.raises(ValueError, match="template_path"):
        _make_model(
            artifact, generative_spinup={"model_dir": "m", "template_path": None}
        )
    # The block is ignored (and the backend still built) in the other modes.
    model = _make_model(
        artifact,
        spinup_source="training_data",
        spinup_forward_model=_NoopBackend(),
    )
    assert model._generative_spinup is None


def test_forward_model_cold_start_samples_once_and_never_builds_backend(
    artifact, inject_stub
) -> None:
    model = _make_model(artifact)
    assert model.spinup_forward_model is None  # the config node was not built
    assert not hasattr(model, "dirs")
    model.disable_spinup()  # no backend to propagate to; must not raise
    assert model._generative_spinup.default_params == {}

    params = _params(velocity=6.0, angle0=15.0)
    out = model.run_single(state=None, params=params)
    assert len(inject_stub.calls) == 1
    assert inject_stub.calls[0]["batch"] == 1
    np.testing.assert_allclose(
        inject_stub.calls[0]["params_hist"][0].numpy(),
        np.repeat([[15.0, 6.0]], HP, axis=0),
    )
    assert out.sizes["time"] == 3
    for v in STATE_VARS:
        assert out[v].dims == ("time", "z", "y", "x")
    # The trajectory carries the template's mask (geometry source 0).
    assert "blanking" in out.data_vars

    # member_index seeds the noise: a second member gets different noise.
    model.run_single(state=None, params=params, member_index=1)
    assert not torch.equal(
        inject_stub.calls[0]["noise"][0], inject_stub.calls[1]["noise"][0]
    )


def test_forward_model_warm_start_skips_generator(artifact, inject_stub) -> None:
    model = _make_model(artifact)
    snap = model._generative_spinup.generate([_params()], [0])[0]
    n = len(inject_stub.calls)
    out = model.run_single(state=snap, params=_params())
    assert len(inject_stub.calls) == n
    assert out.sizes["time"] == 3


def test_forward_model_default_params_reach_generator(artifact, inject_stub) -> None:
    model = _make_model(artifact, default_params={"velocity_magnitude": 2.0})
    partial = xr.Dataset(
        {"inflow_angle": ("time", [5.0, 6.0])}, coords={"time": [0.0, 1.0]}
    )
    model.run_single(state=None, params=partial)
    np.testing.assert_allclose(
        inject_stub.calls[-1]["params_hist"][0].numpy(),
        np.repeat([[5.0, 2.0]], HP, axis=0),
    )


def test_forward_model_geometry_requires_mask_in_generative_mode(
    artifact, inject_stub
) -> None:
    model = _make_model(artifact)
    snap = model._generative_spinup.generate([_params()], [0])[0]
    geom = model._build_geometry(snap).cpu().numpy()
    np.testing.assert_array_equal(geom, 1.0 - snap["blanking"].values)
    with pytest.raises(RuntimeError, match="never inferred"):
        model._build_geometry(snap.drop_vars("blanking"))


def test_forward_model_rejects_generated_snapshot_off_its_domain(
    artifact, inject_stub, monkeypatch
) -> None:
    model = _make_model(artifact)
    good = model._generative_spinup.generate([_params()], [0])[0]
    bad = good.isel(x=slice(0, NX - 1))
    monkeypatch.setattr(model._generative_spinup, "generate", lambda p, i: [bad])
    with pytest.raises(ValueError, match="cells along 'x'"):
        model.run_single(state=None, params=_params())


def test_clone_for_member_shares_generator_and_no_backend(
    artifact, inject_stub, tmp_path
) -> None:
    model = _make_model(artifact)
    clone = model.clone_for_member(tmp_path / "exp", "000")
    assert clone._generative_spinup is model._generative_spinup
    assert clone.spinup_forward_model is None
    assert clone.model is model.model


def test_prepare_neural_surrogate_skips_generative(artifact, inject_stub) -> None:
    from pyurbanair.config.hydra_helpers import clean_outputs, prepare_neural_surrogate

    model = _make_model(artifact)
    prepare_neural_surrogate(model, spinup_backend="pyudales")  # no backend touched
    clean_outputs("neural_surrogate", model)


# ---------------------------------------------------------------------------
# Ensemble integration
# ---------------------------------------------------------------------------


def _ensemble(
    artifact, tmp_path, monkeypatch, n=3
) -> NeuralSurrogateEnsembleForwardModel:
    # Building the CFD spin-up ensemble imports "{backend}.ensemble_forward_model";
    # the generative path must never get there.
    monkeypatch.setattr(ens_mod, "import_module", _never_instantiate)
    template = _make_model(artifact)
    return NeuralSurrogateEnsembleForwardModel(
        template, ensemble_size=n, temp_dir=tmp_path
    )


def _ensemble_params(velocities=(3.0, 4.0, 5.0)) -> xr.Dataset:
    return xr.concat([_params(v) for v in velocities], dim="ensemble", join="override")


def test_ensemble_cold_start_generates_all_members_without_spinup_ensemble(
    artifact, inject_stub, tmp_path, monkeypatch
) -> None:
    ensemble = _ensemble(artifact, tmp_path, monkeypatch)
    params = _ensemble_params()
    out = ensemble.run_ensemble(state=None, params=params)
    assert ensemble._spinup_ensemble is None
    assert ensemble._last_failure_substitutions == {}
    assert out.sizes["ensemble"] == 3 and out.sizes["time"] == 3
    assert len(inject_stub.calls) == 1  # one bounded batch of all members
    call = inject_stub.calls[0]
    assert call["batch"] == 3
    np.testing.assert_allclose(
        call["params_hist"][:, 0, :].numpy(), _first_knots(params)
    )
    # Member indices 0..N-1 seed the noise: matches the single-member draws.
    gs = ensemble.forward_model._generative_spinup
    for i in range(3):
        assert torch.equal(call["noise"][i], gs._member_noise(i))
    # Every member's backend is the shared "no backend".
    for member in ensemble.ensemble_forward_models:
        assert member.spinup_forward_model is None
        assert member._generative_spinup is gs


def test_ensemble_batches_by_sample_batch_size(
    artifact, inject_stub, tmp_path, monkeypatch
) -> None:
    monkeypatch.setattr(ens_mod, "import_module", _never_instantiate)
    template = _make_model(artifact)
    template._generative_spinup.sample_batch_size = 2
    ensemble = NeuralSurrogateEnsembleForwardModel(
        template, ensemble_size=3, temp_dir=tmp_path
    )
    ensemble.run_ensemble(state=None, params=_ensemble_params())
    assert [c["batch"] for c in inject_stub.calls] == [2, 1]


def test_ensemble_warm_start_never_calls_generator(
    artifact, inject_stub, tmp_path, monkeypatch
) -> None:
    ensemble = _ensemble(artifact, tmp_path, monkeypatch)
    params = _ensemble_params()
    gs = ensemble.forward_model._generative_spinup
    warm = xr.concat(
        gs.generate([_params()] * 3, [0, 1, 2]), dim="ensemble", join="override"
    )
    n = len(inject_stub.calls)
    out = ensemble.run_ensemble(state=warm, params=params)
    assert len(inject_stub.calls) == n
    assert out.sizes["ensemble"] == 3

    # A per-member state directory (the ESMDA disk path) is warm too.
    state_dir = tmp_path / "states"
    state_dir.mkdir()
    for i in range(3):
        warm.isel(ensemble=i).to_netcdf(state_dir / f"state_{i}.nc")
    ensemble.run_ensemble(state=state_dir, params=params, sim_name="state")
    assert len(inject_stub.calls) == n


# ---------------------------------------------------------------------------
# ESMDA lifecycle on a REAL parameter smoother
# ---------------------------------------------------------------------------


def _obs_op():
    from data_assimilation.observation_operator import ObservationOperator

    return ObservationOperator(
        obs_ids_x=[2, 6],
        obs_ids_y=[2, 6],
        obs_ids_z=[3, 3],
        obs_states=["u"],
        solver_name="pylbm",
    )


@pytest.mark.parametrize("smoother", ["static", "dynamic"])
def test_esmda_regenerates_on_every_cold_forecast(
    artifact, inject_stub, tmp_path, monkeypatch, smoother
) -> None:
    """One generator call per cold forecast (num_steps + final), each conditioned
    on that iteration's CURRENT first-knot parameters; none on a warm window."""
    import jax
    import jax.numpy as jnp
    from data_assimilation.smoothing.esmda import (
        ParameterESMDA,
        TimeVaryingParameterESMDA,
    )

    ensemble = _ensemble(artifact, tmp_path, monkeypatch)
    n_e = ensemble.ensemble_size
    if smoother == "static":
        prior = xr.Dataset(
            {
                "inflow_angle": ("ensemble", np.array([10.0, 20.0, 30.0])),
                "velocity_magnitude": ("ensemble", np.array([3.0, 4.0, 5.0])),
            }
        )
        esmda = ParameterESMDA(
            observation_operator=_obs_op(),
            forward_model=ensemble,
            C_D=jnp.eye(2) * 0.05,
            num_steps=2,
            rng_key=jax.random.PRNGKey(0),
        )
    else:
        prior = _ensemble_params()
        esmda = TimeVaryingParameterESMDA(
            observation_operator=_obs_op(),
            forward_model=ensemble,
            C_D=jnp.eye(2) * 0.05,
            num_time_points=int(prior.sizes["time"]),
            num_steps=2,
            rng_key=jax.random.PRNGKey(0),
        )
        assert esmda.pin_initial_time_point is False

    obs = jnp.array([12.0, 13.0])
    params_history, state = esmda(
        state=None, params=prior, observations=obs, return_params_history=True
    )
    assert params_history.sizes["esmda_step"] == 3  # prior + 2 updates
    assert len(inject_stub.calls) == 3  # 2 iterations + the final forecast
    for k, call in enumerate(inject_stub.calls):
        assert call["batch"] == n_e
        expected = _first_knots(params_history.isel(esmda_step=k))
        np.testing.assert_allclose(
            call["params_hist"][:, 0, :].numpy(), expected, rtol=1e-6
        )
        # Constant history: every knot of the conditioning is the current value.
        for h in range(1, HP):
            np.testing.assert_array_equal(
                call["params_hist"][:, h, :], call["params_hist"][:, 0, :]
            )
        # Common random numbers across iterations.
        assert torch.equal(call["noise"], inject_stub.calls[0]["noise"])
    # The update actually moved the parameters, so the re-conditioning is real.
    assert not np.allclose(
        _first_knots(params_history.isel(esmda_step=0)),
        _first_knots(params_history.isel(esmda_step=-1)),
    )
    if smoother == "dynamic":
        assert esmda.pin_initial_time_point is False  # the t=0 knot stayed free

    # A later (warm) window carries the posterior state forward: no generation.
    posterior = params_history.isel(esmda_step=-1)
    esmda(state=state.isel(time=-1), params=posterior, observations=obs)
    assert len(inject_stub.calls) == 3
    assert ensemble._spinup_ensemble is None


# ---------------------------------------------------------------------------
# run_esmda.py: config branch + end-to-end smoke run with the stub generator
# ---------------------------------------------------------------------------

SMOKE_GRID = (4, 20, 20)
SMOKE_BOUNDS = [[0.0, 20.0], [0.0, 20.0], [0.0, 10.0]]


def _generative_overrides(model_dir, template, gen_dir, smoother, truth_dir):
    return [
        "model@truth_model=pylbm",
        "model@assim_model=neural_surrogate",
        f"esmda/smoother={smoother}",
        "params@prior_params=dynamic",
        "params@truth_params=dynamic_truth",
        "esmda.localization=null",
        "ensemble.ensemble_size=2",
        "ensemble.num_parallel_processes=1",
        "esmda.num_steps=2",
        "esmda.num_assimilation_windows=2",
        "run.skip_viz=true",
        "run.ensemble_save_on_disk=false",
        f"run.truth_dir={truth_dir}",
        "obs.x_points=[2.5,2.5,18.0,18.0]",
        "obs.y_points=[5.0,15.0,5.0,15.0]",
        "obs.z_points=[3.0,3.0,3.0,3.0]",
        "esmda.interval_seconds=3.0",
        "time.seconds_per_knot=1.5",
        f"assim_model.forward_model.model_dir={model_dir}",
        "assim_model.forward_model.device=cpu",
        "assim_model.forward_model.spinup_source=generative",
        f"assim_model.forward_model.generative_spinup.model_dir={gen_dir}",
        f"assim_model.forward_model.generative_spinup.template_path={template}",
        "assim_model.forward_model.generative_spinup.save_diagnostics=true",
    ]


def _write_truth(truth_dir: pathlib.Path, n_frames: int = 6) -> pathlib.Path:
    """A synthetic ``run_forward_model``-shaped truth (state.nc + params.nc)."""
    nz, ny, nx = SMOKE_GRID
    rng = np.random.default_rng(0)
    times = (np.arange(n_frames) + 1) * 1.0
    coords = {
        "time": times,
        "z": (np.arange(nz) + 0.5) * 2.5,
        "y": np.arange(ny) + 0.5,
        # Face-aligned x so the truth needs no x shift onto the domain frame.
        "x": np.arange(nx, dtype=float),
    }
    state = xr.Dataset(
        {
            v: (
                ("time", "z", "y", "x"),
                2.0 + rng.standard_normal((n_frames, nz, ny, nx)),
            )
            for v in STATE_VARS
        },
        coords=coords,
    )
    knots = np.arange(0.0, 6.0 + 1e-9, 1.5)
    params = xr.Dataset(
        {
            "inflow_angle": ("time", np.linspace(0.0, 20.0, knots.size)),
            "velocity_magnitude": ("time", np.linspace(2.0, 3.0, knots.size)),
        },
        coords={"time": knots},
    )
    truth_dir.mkdir(parents=True, exist_ok=True)
    state.to_netcdf(truth_dir / "state.nc")
    params.to_netcdf(truth_dir / "params.nc")
    return truth_dir


@pytest.fixture
def smoke_artifacts(tmp_path, surrogate_model_dir_factory):
    model_dir = surrogate_model_dir_factory(
        tmp_path,
        domain={"nx": 20, "ny": 20, "nz": 4, "bounds": SMOKE_BOUNDS},
        time={"simulation_time": 3.0, "output_frequency": 1.0, "spinup_time": 0.0},
        state_vars=STATE_VARS,
        param_vars=PARAM_VARS,
    )
    gen_dir = _write_artifact(
        tmp_path / "generator", grid=SMOKE_GRID, bounds=SMOKE_BOUNDS
    )
    template = _write_template(
        tmp_path / "template.nc", grid=SMOKE_GRID, bounds=SMOKE_BOUNDS
    )
    truth_dir = _write_truth(tmp_path / "truth")
    return model_dir, gen_dir, template, truth_dir


def test_run_esmda_generative_config_branch(compose_test_cfg, smoke_artifacts) -> None:
    from data_assimilation.smoothing.esmda import (
        ParameterESMDA,
        StateAndParameterESMDA,
        StateESMDA,
    )

    from scripts.esmda.run_esmda import (
        _check_generative_smoother,
        _generative_spinup_block,
    )

    model_dir, gen_dir, template, truth_dir = smoke_artifacts
    cfg = compose_test_cfg(
        _generative_overrides(model_dir, template, gen_dir, "dynamic", truth_dir),
        config_name="run_esmda",
    )
    block = _generative_spinup_block(cfg)
    assert block is not None
    assert str(block.model_dir) == str(gen_dir)
    assert block.save_diagnostics is True
    # The default config composes with the block present but inert.
    default = compose_test_cfg(
        ["model@assim_model=neural_surrogate"], config_name="run_esmda"
    )
    assert default.assim_model.forward_model.spinup_source == "training_data"
    assert default.assim_model.forward_model.generative_spinup.model_dir is None
    assert _generative_spinup_block(default) is None

    _check_generative_smoother(ParameterESMDA.__new__(ParameterESMDA), True)
    for cls in (StateAndParameterESMDA, StateESMDA):
        with pytest.raises(ValueError, match="not supported"):
            _check_generative_smoother(cls.__new__(cls), True)
        _check_generative_smoother(cls.__new__(cls), False)  # other modes unaffected


def test_run_esmda_generative_lifecycle(
    compose_test_cfg, smoke_artifacts, inject_stub, monkeypatch
) -> None:
    """Full run: cold window 0 regenerates per forecast, warm window 1 does not,
    no _initial_states, no t=0 pinning in window 0, no CFD backend built."""
    import pyudales.forward_model as udales_fm
    from data_assimilation.smoothing.esmda import TimeVaryingParameterESMDA

    from scripts.esmda.run_esmda import run

    monkeypatch.setattr(udales_fm.ForwardModel, "__init__", _never_instantiate)
    pins: list[bool] = []
    original = TimeVaryingParameterESMDA._analysis

    def recording_analysis(self, *args, **kwargs):
        pins.append(bool(self.pin_initial_time_point))
        return original(self, *args, **kwargs)

    monkeypatch.setattr(TimeVaryingParameterESMDA, "_analysis", recording_analysis)

    model_dir, gen_dir, template, truth_dir = smoke_artifacts
    cfg = compose_test_cfg(
        _generative_overrides(model_dir, template, gen_dir, "dynamic", truth_dir),
        config_name="run_esmda",
    )
    run(cfg)

    out_dir = pathlib.Path(cfg.paths.results_dir)
    assert not (out_dir / "_initial_states").exists()
    assert pins == [False, True]  # window 0 free, window 1 boundary-pinned
    # Window 0: num_steps + 1 cold forecasts, each regenerated; window 1: none.
    assert len(inject_stub.calls) == 3
    prior = xr.load_dataset(out_dir / "windows" / "window_0_prior_params.nc")
    posterior = xr.load_dataset(out_dir / "windows" / "window_0_posterior_params.nc")
    np.testing.assert_allclose(
        inject_stub.calls[0]["params_hist"][:, 0, :].numpy(),
        _first_knots(prior),
        rtol=1e-6,
    )
    np.testing.assert_allclose(
        inject_stub.calls[-1]["params_hist"][:, 0, :].numpy(),
        _first_knots(posterior),
        rtol=1e-6,
    )
    assert torch.equal(inject_stub.calls[0]["noise"], inject_stub.calls[-1]["noise"])
    # Diagnostics: one call_<k> folder per cold forecast of window 0 only.
    diag = out_dir / "_generated_states"
    assert sorted(p.name for p in (diag / "window_0").iterdir()) == [
        "call_0",
        "call_1",
        "call_2",
    ]
    assert (diag / "window_0" / "call_2" / "member_1.nc").exists()
    assert not (diag / "window_1").exists()
    assert (out_dir / "posterior_params.nc").exists()


def test_run_esmda_rejects_joint_state_smoother_before_any_forecast(
    compose_test_cfg, smoke_artifacts, inject_stub
) -> None:
    from scripts.esmda.run_esmda import run

    model_dir, gen_dir, template, truth_dir = smoke_artifacts
    cfg = compose_test_cfg(
        _generative_overrides(
            model_dir, template, gen_dir, "state_and_dynamic", truth_dir
        ),
        config_name="run_esmda",
    )
    with pytest.raises(ValueError, match="not supported with the state-bearing"):
        run(cfg)
    assert inject_stub.calls == []
    assert not (pathlib.Path(cfg.paths.results_dir) / "windows").exists()


# ---------------------------------------------------------------------------
# The real generator through the real loader (strict reload + sampling)
# ---------------------------------------------------------------------------


def test_real_tadpole_generator_artifact_samples(tmp_path) -> None:
    pytest.importorskip("diffusers")
    pytest.importorskip("timm")
    pytest.importorskip("einops")
    from neural_surrogates import TadpoleAE, TadpoleLatentGenerator

    grid = (16, 16, 32)
    bounds = [[0.0, 32.0], [0.0, 16.0], [0.0, 16.0]]
    ae_kwargs: dict[str, Any] = {
        "size": "S",
        "encoder_crop_size": 16,
        "latent_type": "mode",
        "normalize": True,
        "sdf_clamp_cells": 8.0,
        "spatial_mode": "global",
        "halo_size": 16,
        "encode_geometry": False,
        "sdf_features": "sdf",
        "geometry_branch": {"width": 8},
    }
    ae_dir = tmp_path / "ae"
    ae_dir.mkdir()
    ae = TadpoleAE(n_state_channels=len(STATE_VARS), **ae_kwargs)
    ae.set_normalization([0.1, -0.2, 0.3], [1.0, 1.5, 0.7])
    # The AE decoder head and the velocity-net output projection are zero-init;
    # randomise them so the samples actually depend on the latents/noise.
    with torch.no_grad():
        torch.nn.init.normal_(
            ae.ae.decoder.transformer_decoder.final_layer.out_proj.weight, std=0.05
        )
    torch.save(ae.state_dict(), ae_dir / "weights.pt")
    OmegaConf.save(
        OmegaConf.create(
            {
                "architecture": {
                    "_target_": "neural_surrogates.TadpoleAE",
                    **ae_kwargs,
                },
                "dataset": {"root_dir": str(tmp_path), "state_vars": list(STATE_VARS)},
            }
        ),
        ae_dir / "config.yaml",
    )
    net = dict(n_layers=1, num_heads=2, film_hidden=8, time_embed_dim=8)
    m = TadpoleLatentGenerator(
        n_state_channels=len(STATE_VARS),
        n_params=len(PARAM_VARS),
        param_history_steps=HP,
        pretrained_ae_dir=str(ae_dir),
        num_sampling_steps=2,
        **net,
    ).eval()
    with torch.no_grad():
        torch.nn.init.normal_(m.velocity_net.seqmodel.out_proj.weight, std=0.05)
    g = torch.Generator().manual_seed(11)
    m.set_latent_normalization(
        torch.randn(m.working_latent_dim, generator=g),
        0.5 + torch.rand(m.working_latent_dim, generator=g),
    )
    m.set_normalization(None, None, [10.0, 3.0], [3.0, 1.0])

    template = _write_template(tmp_path / "t.nc", grid=grid, bounds=bounds)
    fluid_cells = int((1.0 - _blanking(grid)).sum())
    gen_dir = _write_artifact(
        tmp_path / "gen",
        grid=grid,
        bounds=bounds,
        fluid_cells=fluid_cells,
        saved_num_steps=2,
    )
    cfg = OmegaConf.load(gen_dir / "config.yaml")
    cfg.architecture = {
        "_target_": "neural_surrogates.TadpoleLatentGenerator",
        "param_history_steps": HP,
        "skip_pretrained_load": True,
        "pretrained_ae_dir": None,
        "ae_kwargs": m.ae_kwargs,
        "num_sampling_steps": 2,
        **net,
    }
    OmegaConf.save(cfg, gen_dir / "config.yaml")
    torch.save(m.state_dict(), gen_dir / "weights.pt")

    gs = GenerativeSpinup(gen_dir, template, sample_batch_size=2)
    out = gs.generate([_params(3.0), _params(4.0)], [0, 1])
    assert isinstance(gs._model, TadpoleLatentGenerator)
    assert gs._geom_features is not None  # the AE's SDF features were cached
    assert len(out) == 2
    for snap in out:
        assert snap["u"].shape == grid
        assert np.all(np.isfinite(snap["u"].values))
        assert np.all(snap["u"].values[snap["blanking"].values == 1.0] == 0.0)
    assert not np.allclose(out[0]["u"].values, out[1]["u"].values)
    # Batch-size independence holds for the real sampler too.
    single = GenerativeSpinup(gen_dir, template, sample_batch_size=1).generate(
        [_params(4.0)], [1]
    )[0]
    np.testing.assert_allclose(
        single["u"].values, out[1]["u"].values, rtol=1e-4, atol=1e-4
    )
