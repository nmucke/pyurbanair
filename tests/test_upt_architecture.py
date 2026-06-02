"""Unit tests for the Universal Physics Transformer (:class:`UPT`).

These cover the UPT module in isolation -- its ``forward(state, params,
geometry) -> next_state`` contract on a regular grid -- plus its drop-in
behaviour through :class:`NeuralSurrogateForwardModel`, mirroring the
UNetConvNeXt coverage in ``test_neural_surrogate_forward_model.py``.

Everything runs on a tiny ``4x8x8`` grid with a ``tiny``-sized UPT so the whole
file stays sub-second on CPU. ``torch`` / ``kappamodules`` are imported via
``importorskip`` so the suite skips cleanly when the UPT deps are absent.
"""

from __future__ import annotations

import pathlib
from typing import Optional

import numpy as np
import pytest
import xarray as xr

torch = pytest.importorskip("torch")
pytest.importorskip("kappamodules")
trimesh = pytest.importorskip("trimesh")

from hydra.utils import instantiate
from omegaconf import OmegaConf

from neural_surrogates import NeuralSurrogateForwardModel, UPT
from neural_surrogates.architectures import UPT as UPT_from_architectures

from pyurbanair.base_forward_model import BaseForwardModel

NZ, NY, NX = 4, 8, 8
N_STATE = 3
N_PARAMS = 2
STATE_VARS = ("u", "v", "w")
PARAM_VARS = ("inflow_angle", "velocity_magnitude")

# ``tiny``-preset sizing (mirrors conf/.../upt/tiny.yaml); small enough for CPU.
TINY = dict(
    dim=32,
    num_latent_tokens=16,
    num_supernodes=16,
    gnn_dim=16,
    enc_depth=1,
    approx_depth=1,
    dec_depth=1,
    num_heads=2,
    radius=2.5,
    max_degree=8,
)

PRESET_DIR = (
    pathlib.Path(__file__).resolve().parents[1]
    / "conf"
    / "neural_surrogate_architectures"
    / "upt"
)


# -- helper builders --------------------------------------------------------


def _tiny_upt(**overrides) -> UPT:
    """Build a ``tiny``-sized UPT with the test channel/param counts."""
    kwargs = dict(n_state_channels=N_STATE, n_params=N_PARAMS, **TINY)
    kwargs.update(overrides)
    return UPT(**kwargs)


def _inputs(batch: int = 2, *, seed: int = 0, obstacle: bool = True):
    """Random ``(state, params, geometry)`` on the tiny grid.

    A central column of cells is marked obstacle when ``obstacle`` so the
    obstacle-zeroing behaviour can be checked. Geometry is shared across the
    batch (the load-bearing contract assumption).
    """
    g = torch.Generator().manual_seed(seed)
    state = torch.randn(batch, N_STATE, NZ, NY, NX, generator=g)
    params = torch.randn(batch, N_PARAMS, generator=g)
    mask = torch.ones(NZ, NY, NX)
    if obstacle:
        mask[:, NY // 2, NX // 2] = 0.0
    geometry = mask.unsqueeze(0).expand(batch, -1, -1, -1).contiguous()
    return state, params, geometry, mask


# -- 1. forward shape -------------------------------------------------------


def test_forward_shape_and_finite() -> None:
    torch.manual_seed(0)
    model = _tiny_upt().eval()
    state, params, geometry, _ = _inputs(batch=2)
    with torch.no_grad():
        out = model(state, params, geometry)
    assert out.shape == (2, N_STATE, NZ, NY, NX)
    assert torch.isfinite(out).all()


# -- 2. geometry shapes (B,D,H,W) == (B,1,D,H,W) ----------------------------


def test_geometry_shape_variants_match() -> None:
    torch.manual_seed(0)
    model = _tiny_upt().eval()
    state, params, geometry, _ = _inputs(batch=2)
    with torch.no_grad():
        out_4d = model(state, params, geometry)
        out_5d = model(state, params, geometry.unsqueeze(1))
    assert out_5d.shape == out_4d.shape
    torch.testing.assert_close(out_4d, out_5d, rtol=0.0, atol=0.0)


# -- 3. obstacle cells are exactly zero -------------------------------------


def test_obstacle_cells_are_exactly_zero() -> None:
    torch.manual_seed(0)
    model = _tiny_upt().eval()
    state, params, geometry, mask = _inputs(batch=2)
    with torch.no_grad():
        out = model(state, params, geometry)
    obstacle = mask == 0
    assert obstacle.any(), "test geometry must contain obstacle cells"
    # every channel/batch entry at an obstacle cell must be identically zero
    assert torch.all(out[:, :, obstacle] == 0.0)


# -- 4. batched == per-member (determinism / no cross-sample leakage) -------


def test_batched_matches_per_member() -> None:
    """A batched forward equals running each member alone.

    Guards against cross-sample attention leakage and nondeterministic
    supernode selection: member ``b`` of a batched call must byte-match a
    single-member call, given shared geometry but distinct state/params.
    """
    torch.manual_seed(0)
    model = _tiny_upt().eval()
    state, params, geometry, _ = _inputs(batch=3, seed=1)
    with torch.no_grad():
        batched = model(state, params, geometry)
        for b in range(3):
            single = model(
                state[b : b + 1], params[b : b + 1], geometry[b : b + 1]
            )
            torch.testing.assert_close(
                batched[b : b + 1], single, rtol=1e-5, atol=1e-5
            )


# -- 5. differentiable w.r.t. state and params ------------------------------


def test_gradients_flow_to_state_and_params() -> None:
    torch.manual_seed(0)
    model = _tiny_upt()
    state, params, geometry, _ = _inputs(batch=2)
    state = state.clone().requires_grad_(True)
    params = params.clone().requires_grad_(True)
    out = model(state, params, geometry)
    out.sum().backward()
    assert state.grad is not None
    assert params.grad is not None
    assert torch.isfinite(state.grad).all()
    assert torch.isfinite(params.grad).all()
    # gradient should actually be non-trivial (model is not a constant map)
    assert state.grad.abs().sum() > 0
    assert params.grad.abs().sum() > 0


# -- 6. repeated autoregressive calls under no_grad -------------------------


def test_autoregressive_rollout_stays_finite() -> None:
    """Feed the output back in as the next input, as ``rollout_batched`` does."""
    torch.manual_seed(0)
    model = _tiny_upt().eval()
    state, params, geometry, _ = _inputs(batch=2)
    with torch.no_grad():
        cur = state
        for _ in range(5):
            cur = model(cur, params, geometry)
            assert cur.shape == state.shape
            assert torch.isfinite(cur).all()


# -- 7. all five Hydra presets instantiate; tiny forwards -------------------


@pytest.mark.parametrize(
    "preset", ["tiny", "small", "medium", "large", "xlarge"]
)
def test_presets_instantiate(preset: str) -> None:
    cfg = OmegaConf.load(PRESET_DIR / f"{preset}.yaml")
    model = instantiate(cfg, n_state_channels=N_STATE, n_params=N_PARAMS)
    assert isinstance(model, UPT)


def test_tiny_preset_forward_shape() -> None:
    """Only forward-pass the tiny preset to keep the suite fast."""
    torch.manual_seed(0)
    cfg = OmegaConf.load(PRESET_DIR / "tiny.yaml")
    model = instantiate(cfg, n_state_channels=N_STATE, n_params=N_PARAMS).eval()
    state, params, geometry, _ = _inputs(batch=2)
    with torch.no_grad():
        out = model(state, params, geometry)
    assert out.shape == (2, N_STATE, NZ, NY, NX)
    assert torch.isfinite(out).all()


# -- 9. cond_dim DiT conditioning path --------------------------------------


def test_cond_dim_dit_path_forward() -> None:
    """A UPT with ``cond_dim`` set runs the DiT-conditioning branch."""
    torch.manual_seed(0)
    model = _tiny_upt(cond_dim=32).eval()
    assert model.param_proj is not None
    state, params, geometry, _ = _inputs(batch=2)
    with torch.no_grad():
        out = model(state, params, geometry)
    assert out.shape == (2, N_STATE, NZ, NY, NX)
    assert torch.isfinite(out).all()


def test_upt_exported_from_both_namespaces() -> None:
    assert UPT is UPT_from_architectures


# -- 8. integration through NeuralSurrogateForwardModel ---------------------


class _StubSpinup(BaseForwardModel):
    """Synthetic spin-up backend returning a developed pylbm-style field.

    Replicated from ``test_neural_surrogate_forward_model.py`` so this file is
    self-contained.
    """

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


def _trained_domain() -> dict:
    return {
        "nx": NX,
        "ny": NY,
        "nz": NZ,
        "bounds": [[0.0, NX], [0.0, NY], [0.0, NZ]],
    }


def _params() -> xr.Dataset:
    t = np.linspace(0.0, 3.0, 4)
    return xr.Dataset(
        {
            "inflow_angle": ("time", np.linspace(10.0, 20.0, t.size)),
            "velocity_magnitude": ("time", np.linspace(3.0, 4.0, t.size)),
        },
        coords={"time": t},
    )


def _make_upt_model(**overrides) -> NeuralSurrogateForwardModel:
    kwargs = dict(
        architecture=_tiny_upt(),
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


def test_upt_cold_start_rollout_shape_and_spinup() -> None:
    """UPT is a true drop-in: cold-start rollout yields a well-formed traj."""
    torch.manual_seed(0)
    model = _make_upt_model()
    out = model(params=_params())
    assert out is not None
    # simulation_time / output_frequency output frames, plus the t=0 frame.
    assert out.sizes["time"] == 4
    for v in STATE_VARS:
        assert out[v].dims == ("time", "z", "y", "x")
    # Cold start must consult the spin-up backend exactly once.
    assert model.spinup_forward_model.calls == 1
