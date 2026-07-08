"""Unit tests for the P3D surrogate (:class:`P3D`).

These cover the P3D wrapper in isolation -- its ``forward(state, params,
geometry) -> next_state`` contract on a regular grid -- plus its drop-in
behaviour through :class:`NeuralSurrogateForwardModel`, mirroring the UPT
coverage in ``test_upt_architecture.py``.

P3D's U-structure downsamples by 16x total, so the grid is ``16x16x16`` here
(the wrapper pads non-periodic axes up to a multiple of 16, but a degenerate
sub-16 grid pads to the least-tested 1^3-latent regime, so the tests use a clean
multiple of 16). ``p3d_surrogate`` is imported via ``importorskip`` so the suite
skips cleanly when the (heavy) P3D deps are absent.
"""

from __future__ import annotations

import pathlib
from typing import Optional

import numpy as np
import pytest
import xarray as xr

torch = pytest.importorskip("torch")
pytest.importorskip("p3d_surrogate")
trimesh = pytest.importorskip("trimesh")

from hydra.utils import instantiate
from neural_surrogates import P3D, NeuralSurrogateForwardModel, sdf_features
from neural_surrogates.architectures import P3D as P3D_from_architectures
from omegaconf import OmegaConf

from pyurbanair.base_forward_model import BaseForwardModel

NZ, NY, NX = 16, 16, 16
N_STATE = 3
N_PARAMS = 2
STATE_VARS = ("u", "v", "w")
PARAM_VARS = ("inflow_angle", "velocity_magnitude")

# ``tiny``-preset sizing (mirrors conf/.../p3d/tiny.yaml); smallest upstream
# size (S) with a small window so the suite stays runnable on CPU.
TINY = dict(size="S", window_size=2, partition_size=1)

PRESET_DIR = (
    pathlib.Path(__file__).resolve().parents[1]
    / "conf"
    / "neural_surrogate"
    / "architectures"
    / "p3d"
)


# -- helper builders --------------------------------------------------------


def _tiny_p3d(**overrides) -> P3D:
    """Build a ``tiny``-sized P3D with the test channel/param counts."""
    kwargs = dict(n_state_channels=N_STATE, n_params=N_PARAMS, **TINY)
    kwargs.update(overrides)
    return P3D(**kwargs)


def _inputs(batch: int = 2, *, seed: int = 0, obstacle: bool = True):
    """Random ``(state, params, geometry)`` on the test grid.

    A central column of cells is marked obstacle when ``obstacle`` so the
    obstacle-zeroing behaviour can be checked. Geometry is shared across the
    batch (the contract assumption from ``training.py``).
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
    model = _tiny_p3d().eval()
    state, params, geometry, _ = _inputs(batch=2)
    with torch.no_grad():
        out = model(state, params, geometry)
    assert out.shape == (2, N_STATE, NZ, NY, NX)
    assert torch.isfinite(out).all()


# -- 2. geometry shapes (B,D,H,W) == (B,1,D,H,W) ----------------------------


def test_geometry_shape_variants_match() -> None:
    torch.manual_seed(0)
    model = _tiny_p3d().eval()
    state, params, geometry, _ = _inputs(batch=2)
    with torch.no_grad():
        out_4d = model(state, params, geometry)
        out_5d = model(state, params, geometry.unsqueeze(1))
    assert out_5d.shape == out_4d.shape
    torch.testing.assert_close(out_4d, out_5d, rtol=0.0, atol=0.0)


# -- 3. obstacle cells are exactly zero -------------------------------------


def test_obstacle_cells_are_exactly_zero() -> None:
    torch.manual_seed(0)
    model = _tiny_p3d().eval()
    state, params, geometry, mask = _inputs(batch=2)
    with torch.no_grad():
        out = model(state, params, geometry)
    obstacle = mask == 0
    assert obstacle.any(), "test geometry must contain obstacle cells"
    assert torch.all(out[:, :, obstacle] == 0.0)


# -- 4. batched == per-member (determinism / no cross-sample leakage) -------


def test_batched_matches_per_member() -> None:
    """A batched forward equals running each member alone.

    Guards against cross-sample leakage and nondeterminism (upstream label
    dropout is disabled in the wrapper). Runs in ``eval()``.
    """
    torch.manual_seed(0)
    model = _tiny_p3d().eval()
    state, params, geometry, _ = _inputs(batch=3, seed=1)
    with torch.no_grad():
        batched = model(state, params, geometry)
        for b in range(3):
            single = model(state[b : b + 1], params[b : b + 1], geometry[b : b + 1])
            torch.testing.assert_close(batched[b : b + 1], single, rtol=1e-4, atol=1e-4)


# -- 5. differentiable w.r.t. state and params ------------------------------


def test_gradients_flow_to_state_and_params() -> None:
    """Default (channels) conditioning: params enter through the input conv so
    their gradient is non-trivial even at init."""
    torch.manual_seed(0)
    model = _tiny_p3d()
    state, params, geometry, _ = _inputs(batch=2)
    state = state.clone().requires_grad_(True)
    params = params.clone().requires_grad_(True)
    out = model(state, params, geometry)
    out.sum().backward()
    assert state.grad is not None
    assert params.grad is not None
    assert torch.isfinite(state.grad).all()
    assert torch.isfinite(params.grad).all()
    assert state.grad.abs().sum() > 0
    assert params.grad.abs().sum() > 0


# -- 6. repeated autoregressive calls under no_grad -------------------------


def test_autoregressive_rollout_stays_finite() -> None:
    """Feed the output back in as the next input, as ``rollout_batched`` does."""
    torch.manual_seed(0)
    model = _tiny_p3d().eval()
    state, params, geometry, _ = _inputs(batch=2)
    with torch.no_grad():
        cur = state
        for _ in range(3):
            cur = model(cur, params, geometry)
            assert cur.shape == state.shape
            assert torch.isfinite(cur).all()


# -- 7. presets instantiate; tiny forwards ----------------------------------


@pytest.mark.parametrize("preset", ["tiny", "small", "medium", "large", "xlarge"])
def test_presets_instantiate(preset: str) -> None:
    cfg = OmegaConf.load(PRESET_DIR / f"{preset}.yaml")
    model = instantiate(cfg, n_state_channels=N_STATE, n_params=N_PARAMS)
    assert isinstance(model, P3D)


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


# -- 8. native pde_parameters conditioning path -----------------------------


def test_native_param_conditioning_path() -> None:
    """``param_conditioning='native'`` routes params through P3D's scalar
    pde_parameters adaLN path; output is well-formed (the param gradient is not
    asserted: P3D zero-inits its adaLN modulation, so it can be 0 at init)."""
    torch.manual_seed(0)
    model = _tiny_p3d(param_conditioning="native").eval()
    assert model.in_channels == N_STATE + 1
    state, params, geometry, _ = _inputs(batch=2)
    with torch.no_grad():
        out = model(state, params, geometry)
    assert out.shape == (2, N_STATE, NZ, NY, NX)
    assert torch.isfinite(out).all()


def test_channels_mode_input_channels() -> None:
    model = _tiny_p3d(param_conditioning="channels")
    assert model.in_channels == N_STATE + 1 + N_PARAMS


# -- 8b. extra_in_channels contract (domain-decomposition fine net) ---------


def test_extra_in_channels_zero_is_byte_identical() -> None:
    """The default extra_in_channels=0 leaves the stem (and the whole state
    dict) byte-identical to a standard P3D, so existing checkpoints load."""
    a = _tiny_p3d()
    b = _tiny_p3d(extra_in_channels=0)
    # Default param_conditioning is "native" -> no per-param input channels.
    assert a.in_channels == N_STATE + 1
    assert set(a.state_dict()) == set(b.state_dict())
    # state-dict tensor shapes must match exactly (load_state_dict succeeds).
    b.load_state_dict(a.state_dict())


def test_extra_in_channels_guard() -> None:
    """extra must be provided iff extra_in_channels > 0, with matching width."""
    torch.manual_seed(0)
    state, params, geometry, _ = _inputs(batch=2)
    # extra_in_channels=0 net rejects a stray extra tensor.
    plain = _tiny_p3d().eval()
    with pytest.raises(ValueError):
        plain(state, params, geometry, extra=torch.randn(2, 4, NZ, NY, NX))
    # extra_in_channels>0 net requires extra, and validates its channel count.
    widened = _tiny_p3d(extra_in_channels=4).eval()
    # native default -> N_STATE + geometry(1) + extra(4), no param channels.
    assert widened.in_channels == N_STATE + 1 + 4
    with pytest.raises(ValueError):
        widened(state, params, geometry)  # missing extra
    with pytest.raises(ValueError):
        widened(state, params, geometry, extra=torch.randn(2, 3, NZ, NY, NX))


def test_extra_forward_runs_and_is_differentiable() -> None:
    """With extra_in_channels>0, the raw extra channels flow through the stem
    and gradients reach them."""
    torch.manual_seed(0)
    model = _tiny_p3d(extra_in_channels=4)
    state, params, geometry, _ = _inputs(batch=2)
    extra = torch.randn(2, 4, NZ, NY, NX, requires_grad=True)
    out = model(state, params, geometry, extra=extra)
    assert out.shape == (2, N_STATE, NZ, NY, NX)
    assert torch.isfinite(out).all()
    out.sum().backward()
    assert extra.grad is not None and torch.isfinite(extra.grad).all()


def test_p3d_as_domain_decomposed_fine_and_coarse_net() -> None:
    """P3D works as BOTH the fine and coarse sub-net of DomainDecomposed: the
    wrapper injects extra_in_channels on the fine net and the model runs the
    drop-in forward plus the Eq (9) return_intermediates path end-to-end."""
    from neural_surrogates import DomainDecomposed

    torch.manual_seed(0)

    def p3d_node() -> dict:
        return dict(
            _target_="neural_surrogates.P3D",
            **TINY,
            normalize=True,
            predict_residual=True,
        )

    # interior 8 + 2*halo 4 -> 16-cell fine blocks (no P3D padding); coarse grid
    # = grid / 2. y is periodic (16 % 8 == 0).
    dd = DomainDecomposed(
        n_state_channels=N_STATE,
        n_params=N_PARAMS,
        decomposition=dict(
            interior_size=8,
            halo=4,
            taper=2,
            coarsen_factor=2,
            n_pos=3,
            periodic_axes=(False, True, False),
        ),
        fine_net=p3d_node(),
        coarse_net=p3d_node(),
    )
    # The wrapper widened the fine net's stem; the coarse net stays at extra=0.
    assert dd.fine_net.extra_in_channels == N_STATE + dd.dd.n_pos
    assert dd.coarse_net.extra_in_channels == 0

    state, params, geometry, _ = _inputs(batch=2)

    merged = dd(state, params, geometry)
    assert merged.shape == (2, N_STATE, NZ, NY, NX)
    assert torch.isfinite(merged).all()

    merged2, info = dd(state, params, geometry, return_intermediates=True)
    assert merged2.shape == merged.shape
    assert {"coarse_pred", "patch_pred", "context", "num_patches", "dd"} <= set(info)
    merged2.sum().backward()  # gradients flow through both sub-nets


# -- 8c. SDF geometry features (sdf_features) -------------------------------


def _batched_sdf(geometry: torch.Tensor, clamp: float = 32.0) -> torch.Tensor:
    """(B, 4, *grid) features for a (B, *grid) mask, via the shared helper."""
    return torch.stack(
        [
            sdf_features(geometry[b], clamp_cells=clamp)
            for b in range(geometry.shape[0])
        ],
        dim=0,
    )


def test_sdf_features_false_is_byte_identical() -> None:
    """Default sdf_features=False leaves in_channels and the state dict
    byte-identical to a model built without the flag (repo no-op rule)."""
    a = _tiny_p3d()  # flag absent -> default False
    b = _tiny_p3d(sdf_features=False)
    assert a.n_geom_feature_channels == 0
    assert a.in_channels == b.in_channels == N_STATE + 1
    assert set(a.state_dict()) == set(b.state_dict())
    b.load_state_dict(a.state_dict())


def test_sdf_off_forward_matches_pre_change_golden() -> None:
    """The sdf-off forward reproduces a golden output captured from the parent
    P3D (commit cc19a62, before this branch touched p3d.py) -- a regression
    guard on the shared, unchanged code path (plan §7.3a). Weight init is
    deterministic under the seed because the sdf-off ctor draws no extra RNG, so
    the same seed yields the same weights in the current and parent P3D.
    Regenerate the fixture from the parent code only if the disabled path ever
    legitimately changes."""
    torch.manual_seed(1234)
    model = _tiny_p3d().eval()  # sdf off
    state, params, geometry, _ = _inputs(batch=2)  # generator-seeded, seed=0
    with torch.no_grad():
        out = model(state, params, geometry)
    golden = np.load(
        pathlib.Path(__file__).resolve().parent / "fixtures" / "p3d_sdf_off_golden.npz"
    )["out"]
    torch.testing.assert_close(out, torch.from_numpy(golden), rtol=1e-5, atol=1e-5)


def test_sdf_features_true_widens_stem_and_accepts_provided() -> None:
    """sdf_features=True adds 4 stem channels and consumes provided features."""
    model = _tiny_p3d(sdf_features=True).eval()
    assert model.n_geom_feature_channels == 4
    assert model.in_channels == N_STATE + 1 + 4  # native default: no param chans
    state, params, geometry, mask = _inputs(batch=2)
    feat = _batched_sdf(geometry)
    assert feat.shape == (2, 4, NZ, NY, NX)
    with torch.no_grad():
        out = model(state, params, geometry, geom_features=feat)
    assert out.shape == (2, N_STATE, NZ, NY, NX)
    assert torch.isfinite(out).all()
    # Output hard-masking still holds with the extra channels.
    assert torch.all(out[:, :, mask == 0] == 0.0)
    # Wrong channel count is rejected.
    with pytest.raises(ValueError):
        model(state, params, geometry, geom_features=feat[:, :3])
    # A model without the flag rejects stray features.
    with pytest.raises(ValueError):
        _tiny_p3d().eval()(state, params, geometry, geom_features=feat)


@pytest.mark.parametrize("mode, n_feat", [("sdf", 1), ("grad", 3), ("both", 4)])
def test_sdf_feature_mode_widens_stem_by_selected_channels(
    mode: str, n_feat: int
) -> None:
    """Each mode widens the stem by exactly the channels it selects, and the
    model consumes matching precomputed features and self-computes them the same
    way at inference."""
    model = _tiny_p3d(sdf_features=mode).eval()
    assert model.sdf_feature_mode == mode
    assert model.n_geom_feature_channels == n_feat
    assert model.in_channels == N_STATE + 1 + n_feat
    state, params, geometry, mask = _inputs(batch=2)
    feat = torch.stack(
        [sdf_features(geometry[b], mode=mode) for b in range(geometry.shape[0])],
        dim=0,
    )
    assert feat.shape == (2, n_feat, NZ, NY, NX)
    with torch.no_grad():
        provided = model(state, params, geometry, geom_features=feat)
        self_computed = model(state, params, geometry)  # inference self-compute
    torch.testing.assert_close(provided, self_computed)
    assert torch.all(provided[:, :, mask == 0] == 0.0)


def test_sdf_bool_true_is_both_mode() -> None:
    """The legacy boolean flag still works: True == 'both' (4 channels), so
    already-trained config.yaml files (which stored a bool) load unchanged."""
    model = _tiny_p3d(sdf_features=True)
    assert model.sdf_feature_mode == "both"
    assert model.n_geom_feature_channels == 4


def test_sdf_self_compute_caches_once_and_matches_provided(monkeypatch) -> None:
    """Inference self-compute: the EDT runs once for a stable geometry object
    (steps 2..T hit the cache), a new geometry tensor recomputes, and the
    self-computed output equals the explicitly-provided-features path."""
    import neural_surrogates.architectures.p3d as p3d_mod

    calls = {"n": 0}
    real = p3d_mod.compute_sdf_features

    def counting(*args, **kwargs):
        calls["n"] += 1
        return real(*args, **kwargs)

    monkeypatch.setattr(p3d_mod, "compute_sdf_features", counting)

    model = _tiny_p3d(sdf_features=True).eval()
    state, params, geometry, _ = _inputs(batch=2)  # shared mask across batch
    with torch.no_grad():
        out1 = model(state, params, geometry)  # computes (member 0 only)
        out2 = model(state, params, geometry)  # same tensor object -> cache hit
    # Shared mask -> one EDT for member 0, reused for member 1; second call is a
    # pure cache hit. So exactly one compute across both forwards.
    assert calls["n"] == 1
    torch.testing.assert_close(out1, out2, rtol=0.0, atol=0.0)

    # A distinct object with IDENTICAL content reuses the cache (content
    # revalidation via torch.equal) -- no recompute.
    with torch.no_grad():
        model(state, params, geometry.clone())
    assert calls["n"] == 1

    # A DIFFERENT-content geometry forces a recompute.
    other = geometry.clone()
    other[:, 0, 0, 0] = 0.0
    with torch.no_grad():
        model(state, params, other)
    assert calls["n"] == 2

    # Self-computed output equals the explicitly-provided-features path.
    feat = _batched_sdf(geometry, clamp=model.sdf_clamp_cells)
    with torch.no_grad():
        out_provided = model(state, params, geometry.clone(), geom_features=feat)
    torch.testing.assert_close(out1, out_provided, rtol=1e-5, atol=1e-5)


def test_sdf_cache_no_stale_alias_on_content_change() -> None:
    """The cache holds a reference to its keyed geometry (so a freed address
    can't be recycled into a silent stale hit), and a distinct geometry with
    the SAME shape+device but DIFFERENT mask content returns *its* features,
    never the previously cached ones -- the exact aliasing failure a bare
    data_ptr key would hit on a fixed domain."""
    model = _tiny_p3d(sdf_features=True).eval()
    state = torch.randn(2, N_STATE, NZ, NY, NX)
    params = torch.randn(2, N_PARAMS)

    m1 = torch.ones(NZ, NY, NX)
    m1[:, 3, 3] = 0.0
    g1 = m1.unsqueeze(0).expand(2, -1, -1, -1).contiguous()
    with torch.no_grad():
        out1 = model(state, params, g1)
    # The cache keeps g1 alive -> its storage/address cannot be recycled.
    assert model._sdf_cache[0] is g1

    m2 = torch.ones(NZ, NY, NX)
    m2[:, 6, 6] = 0.0  # different obstacle cell -> different features
    g2 = m2.unsqueeze(0).expand(2, -1, -1, -1).contiguous()
    del g1
    with torch.no_grad():
        out2 = model(state, params, g2)

    # out2 must be g2's features (compared against the independent helper path),
    # not g1's stale cached ones.
    feat2 = _batched_sdf(g2, clamp=model.sdf_clamp_cells)
    with torch.no_grad():
        out2_provided = model(state, params, g2.clone(), geom_features=feat2)
    torch.testing.assert_close(out2, out2_provided, rtol=1e-5, atol=1e-5)
    assert not torch.allclose(out1, out2)


def test_sdf_shared_mask_shortcut_vs_distinct_masks() -> None:
    """The batched self-compute reuses member 0 for identical masks and falls
    back to a per-member EDT when they differ."""
    model = _tiny_p3d(sdf_features=True).eval()
    state = torch.randn(2, N_STATE, NZ, NY, NX)
    params = torch.randn(2, N_PARAMS)
    # Two DIFFERENT masks in the batch -> per-member features.
    m0 = torch.ones(NZ, NY, NX)
    m0[:, 4, 4] = 0.0
    m1 = torch.ones(NZ, NY, NX)
    m1[:, 2, 6] = 0.0
    geometry = torch.stack([m0, m1], dim=0)
    with torch.no_grad():
        out = model(state, params, geometry)
    assert out.shape == (2, N_STATE, NZ, NY, NX)
    assert torch.isfinite(out).all()
    # Each member's obstacle column is zeroed by its own mask.
    assert torch.all(out[0, :, m0 == 0] == 0.0)
    assert torch.all(out[1, :, m1 == 0] == 0.0)


def test_sdf_preset_comment_knob_instantiates() -> None:
    """A p3d preset with sdf_features enabled builds a widened model."""
    cfg = OmegaConf.load(PRESET_DIR / "tiny.yaml")
    cfg.sdf_features = True
    model = instantiate(cfg, n_state_channels=N_STATE, n_params=N_PARAMS)
    assert model.n_geom_feature_channels == 4


def test_p3d_sdf_features_cold_start_rollout() -> None:
    """The forward-model / rollout plumbing is UNCHANGED: it still passes only
    (state, params, geometry) and the sdf-enabled model self-computes."""
    torch.manual_seed(0)
    model = _make_p3d_model(architecture=_tiny_p3d(sdf_features=True))
    out = model(params=_params())
    assert out is not None
    assert out.sizes["time"] == 3
    for v in STATE_VARS:
        assert out[v].dims == ("time", "z", "y", "x")


# -- 9. periodic-axis divisibility guard ------------------------------------


def test_periodic_axis_requires_divisibility() -> None:
    """A periodic axis cannot be zero-padded (it would break the circular
    convs), so a non-divisible periodic size must raise; a divisible one runs."""
    torch.manual_seed(0)
    model = _tiny_p3d(periodic_axes=["y"]).eval()
    # NY = 16 is divisible -> runs fine.
    state, params, geometry, _ = _inputs(batch=1)
    with torch.no_grad():
        out = model(state, params, geometry)
    assert torch.isfinite(out).all()
    # NY = 20 is not a multiple of 16 -> must raise.
    bad = torch.randn(1, N_STATE, 16, 20, 16)
    bad_geom = torch.ones(1, 16, 20, 16)
    with pytest.raises(ValueError):
        with torch.no_grad():
            model(bad, params, bad_geom)


def test_non_periodic_grid_is_padded_and_cropped() -> None:
    """Non-periodic sub-16 / non-multiple grids are padded to 16 and cropped
    back to the original shape."""
    torch.manual_seed(0)
    model = _tiny_p3d().eval()
    state = torch.randn(1, N_STATE, 8, 12, 16)
    geometry = torch.ones(1, 8, 12, 16)
    params = torch.randn(1, N_PARAMS)
    with torch.no_grad():
        out = model(state, params, geometry)
    assert out.shape == (1, N_STATE, 8, 12, 16)
    assert torch.isfinite(out).all()


def test_p3d_exported_from_both_namespaces() -> None:
    assert P3D is P3D_from_architectures


# -- 10. integration through NeuralSurrogateForwardModel --------------------


class _StubSpinup(BaseForwardModel):
    """Synthetic spin-up backend returning a developed pylbm-style field.

    Replicated from ``test_upt_architecture.py`` so this file is self-contained.
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


def _make_p3d_model(**overrides) -> NeuralSurrogateForwardModel:
    kwargs = dict(
        architecture=_tiny_p3d(),
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


def test_p3d_cold_start_rollout_shape_and_spinup() -> None:
    """P3D is a true drop-in: cold-start rollout yields a well-formed traj."""
    torch.manual_seed(0)
    model = _make_p3d_model()
    out = model(params=_params())
    assert out is not None
    # round(simulation_time / output_frequency) = round(3.0 / 1.0) = 3 frames.
    assert out.sizes["time"] == 3
    for v in STATE_VARS:
        assert out[v].dims == ("time", "z", "y", "x")
    assert model.spinup_forward_model.calls == 1
