"""AE -> time-stepper (plan 03): unit tests on ``TadpoleTimeStepper``.

* **Unit** (``TadpoleTimeStepper``): the identity-at-init parity invariant (the
  single most informative test of the DFT wiring -- a fresh stepper's forward is
  bit-identical to the pure-AE reconstruction), the param-conditioning no-op
  (``param_conditioning="none"`` and ``n_params=0`` build the same module tree),
  ``encode_geometry`` on/off shapes, padding round-trip on a non-divisible grid,
  weight round-trip parity, and the ``skip_pretrained_load`` (no-AE-dir) build.

Gated with ``importorskip`` on the vendored Tadpole runtime deps
(``diffusers`` / ``timm`` / ``einops``) + ``peft``, so envs without them skip.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
import xarray as xr
from omegaconf import OmegaConf

torch = pytest.importorskip("torch")
pytest.importorskip("diffusers")
pytest.importorskip("timm")
pytest.importorskip("einops")
pytest.importorskip("peft")

from neural_surrogates import TadpoleTimeStepper
from neural_surrogates.architectures._tadpole.architecture.downstream import (
    SequentialModel,
)

from pyurbanair.base_forward_model import BaseForwardModel

STATE_VARS = ("u", "v", "w")
PARAM_VARS = ("inflow_angle", "velocity_magnitude")

CROP = 16  # encoder_crop_size must be a multiple of 16 (encoder downsamples /16)


# --------------------------------------------------------------------------- #
# Unit tests on TadpoleTimeStepper (Phase 1 wrapper; no Phase 2A dependency).
# --------------------------------------------------------------------------- #


def _stepper(
    *,
    n_state_channels: Any = 3,
    n_params: Any = 2,
    encode_geometry: Any = True,
    sdf_features: Any = "none",
    param_conditioning: Any = "film",
    latent_type: Any = "mode",
    **kw: Any,
) -> TadpoleTimeStepper:
    """A fresh (random-init, no AE dir) stepper on CPU smoke shapes."""
    kw.setdefault("encoder_crop_size", CROP)
    kw.setdefault("sdf_clamp_cells", 8)
    return TadpoleTimeStepper(
        n_state_channels=n_state_channels,
        n_params=n_params,
        size="S",
        pretrained_ae_dir=None,
        skip_pretrained_load=True,
        encode_geometry=encode_geometry,
        sdf_features=sdf_features,
        param_conditioning=param_conditioning,
        latent_type=latent_type,
        **kw,
    )


def _inputs(b: Any = 1, grid: Any = (16, 16, 16), n_params: Any = 2) -> Any:
    state = torch.randn(b, 3, *grid)
    params = torch.randn(b, n_params) if n_params else None
    geom = (torch.rand(b, *grid) > 0.2).float()
    return state, params, geom


def _nonzero_sequential(**kwargs: Any) -> SequentialModel:
    model = SequentialModel(
        in_dim=4,
        hidden_size=8,
        num_heads=2,
        n_layers=2,
        attention_method="naive",
        init_zero_proj=False,
        **kwargs,
    )
    return model


def test_sequential_checkpoint_matches_eager_forward_and_backward() -> None:
    eager = _nonzero_sequential(use_checkpoint=False)
    checkpointed = _nonzero_sequential(use_checkpoint=True)
    checkpointed.load_state_dict(eager.state_dict())
    x_eager = torch.randn(2, 4, 1, 2, 3, requires_grad=True)
    x_checkpointed = x_eager.detach().clone().requires_grad_(True)

    eager(x_eager).square().mean().backward()
    checkpointed(x_checkpointed).square().mean().backward()

    assert torch.allclose(eager(x_eager.detach()), checkpointed(x_eager.detach()))
    assert torch.allclose(x_eager.grad, x_checkpointed.grad)
    for p_eager, p_checkpointed in zip(eager.parameters(), checkpointed.parameters()):
        assert torch.allclose(p_eager.grad, p_checkpointed.grad)


def test_sequential_context_windows_preserve_grid_and_gradients() -> None:
    model = _nonzero_sequential(in_context_patches=3)
    x = torch.randn(2, 4, 1, 2, 4, requires_grad=True)
    out = model(x)
    assert out.shape == x.shape
    assert torch.isfinite(out).all()
    out.square().mean().backward()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()
    assert torch.count_nonzero(x.grad) > 0


def test_identity_at_init_parity() -> None:
    """THE DFT-wiring invariant: at init the zero-init subnetwork/gamma skips
    leave the DFT output *identical* to the plain-AE reconstruction, so
    ``stepper(state) == stepper._ae_reference_recon(state, geom)`` exactly.

    This is NOT ``state_next == state`` (that needs a perfectly-reconstructing
    AE -- a training outcome, not a wiring invariant)."""
    m = _stepper(latent_type="mode").eval()
    m.set_normalization([0, 0, 0], [1, 1, 1], [0, 0], [1, 1])
    state, params, geom = _inputs()
    with torch.no_grad():
        out = m(state, params, geom)
        ref = m._ae_reference_recon(state, geom)
    assert out.shape == state.shape
    assert torch.isfinite(out).all()
    assert torch.equal(out, ref)  # exact, bitwise: the load-bearing invariant
    # obstacle cells are exactly zeroed in the physical prediction
    mask = geom.unsqueeze(1)
    assert torch.allclose(out * (1 - mask), torch.zeros_like(out))


def test_identity_at_init_parity_non_divisible_grid() -> None:
    """The parity invariant also holds on a grid not divisible by the crop size
    (internal zero-pad -> crop-back path)."""
    m = _stepper(latent_type="mode").eval()
    m.set_normalization([0, 0, 0], [1, 1, 1], [0, 0], [1, 1])
    state, params, geom = _inputs(grid=(18, 16, 16))
    with torch.no_grad():
        out = m(state, params, geom)
        ref = m._ae_reference_recon(state, geom)
    assert out.shape == state.shape
    assert torch.equal(out, ref)  # exact, bitwise


def test_param_conditioning_none_matches_zero_params_module_tree() -> None:
    """``param_conditioning="none"`` (with params) and ``n_params=0`` build the
    *same* module tree (no param machinery) -- the repo no-op rule -- and both
    forward cleanly."""
    none_build = _stepper(n_params=2, param_conditioning="none")
    zero_build = _stepper(n_params=0)
    assert set(none_build.state_dict().keys()) == set(zero_build.state_dict().keys())
    # ... and a param-conditioned build has strictly more keys (the param MLP).
    film_build = _stepper(n_params=2, param_conditioning="film")
    assert set(film_build.state_dict().keys()) > set(none_build.state_dict().keys())

    state, params, geom = _inputs()
    none_build.eval()
    zero_build.eval()
    with torch.no_grad():
        out_none = none_build(state, params, geom)  # params ignored
        out_zero = zero_build(state, None, geom)
    assert out_none.shape == state.shape
    assert torch.isfinite(out_none).all()
    assert out_zero.shape == state.shape
    assert torch.isfinite(out_zero).all()


def test_encode_geometry_on_off_shapes() -> None:
    """encode_geometry toggles the folded geometry channels but the public
    forward always returns the physical state shape (obstacles zeroed)."""
    state, params, geom = _inputs()
    for eg, extra in [(False, 0), (True, 1)]:
        m = _stepper(encode_geometry=eg).eval()
        m.set_normalization([0, 0, 0], [1, 1, 1], [0, 0], [1, 1])
        assert m.n_geometry_channels == extra
        with torch.no_grad():
            out = m(state, params, geom)
        assert out.shape == state.shape
        assert torch.isfinite(out).all()


def test_padding_round_trip_non_divisible_grid() -> None:
    """A grid not divisible by the crop size is padded internally and cropped
    back, so the prediction matches the (odd) input shape."""
    m = _stepper(latent_type="mode").eval()
    m.set_normalization([0, 0, 0], [1, 1, 1], [0, 0], [1, 1])
    grid = (20, 24, 18)  # none divisible by 16
    state, params, geom = _inputs(grid=grid)
    with torch.no_grad():
        out = m(state, params, geom)
    assert out.shape[2:] == grid


def test_weight_round_trip_output_parity() -> None:
    """weights.pt (state_dict) reload reproduces outputs, incl. the installed
    normalization buffers. ``latent_type="mode"`` makes the forward
    deterministic so a broken buffer save/load would change the output."""
    m = _stepper(latent_type="mode")
    m.set_normalization([1.0, -2.0, 0.5], [3.0, 0.5, 2.0], [0.1, 0.2], [1.5, 0.5])
    m.eval()
    state, params, geom = _inputs()
    with torch.no_grad():
        out = m(state, params, geom)

    fresh = _stepper(latent_type="mode")
    fresh.load_state_dict(m.state_dict())  # carries the (random) net + norm buffers
    fresh.eval()
    with torch.no_grad():
        out_reloaded = fresh(state, params, geom)
    assert torch.allclose(out, out_reloaded, atol=1e-6)
    # and the normalization buffers actually travelled (not left at identity)
    assert torch.allclose(fresh.state_mean, torch.tensor([1.0, -2.0, 0.5]))
    assert torch.allclose(fresh.param_mean, torch.tensor([0.1, 0.2]))


def test_skip_pretrained_load_builds_without_ae_dir() -> None:
    """``skip_pretrained_load=True`` (the ESMDA-deploy build) needs no AE dir --
    the merged weights.pt carries every weight -- and forwards finitely."""
    m = _stepper().eval()  # _stepper builds with skip_pretrained_load=True
    m.set_normalization([0, 0, 0], [1, 1, 1], [0, 0], [1, 1])
    state, params, geom = _inputs()
    with torch.no_grad():
        out = m(state, params, geom)
    assert out.shape == state.shape


def test_forward_requires_params_when_conditioned() -> None:
    """A conditioned build called without params errors loudly rather than
    silently degrading to unconditioned (see the no-op-rule guard in forward)."""
    m = _stepper(n_params=2, param_conditioning="film").eval()
    m.set_normalization([0, 0, 0], [1, 1, 1], [0, 0], [1, 1])
    state, _, geom = _inputs()
    with pytest.raises(ValueError, match="requires params"):
        m(state, None, geom)


def test_predict_residual_false_rejected() -> None:
    """``predict_residual`` is a dead knob (the residual is intrinsic); False is
    rejected rather than silently ignored."""
    with pytest.raises(ValueError, match="predict_residual=True"):
        _stepper(predict_residual=False)


def test_identity_at_init_parity_after_lora_injection() -> None:
    """The case that actually ships: after standard-LoRA injection (B zero-init),
    the parity invariant STILL holds -- injection adds no delta at init."""
    inject_lora = pytest.importorskip("neural_surrogates.finetuning").inject_lora
    resolve_target_modules = pytest.importorskip(
        "neural_surrogates.finetuning"
    ).resolve_target_modules

    m = _stepper(latent_type="mode")
    m.set_normalization([0, 0, 0], [1, 1, 1], [0, 0], [1, 1])
    targets = resolve_target_modules(m, preset="tadpole_encdec")
    peft_model = inject_lora(m, rank=4, alpha=8, target_modules=targets)
    peft_model.eval()
    state, params, geom = _inputs()
    with torch.no_grad():
        out = peft_model(state, params, geom)
        # the base model's reference recon is reached through the PEFT wrapper
        ref = peft_model.base_model.model._ae_reference_recon(state, geom)
    assert torch.equal(out, ref)  # exact, bitwise: injection adds no delta at init


def test_max_internal_batchsize_chunked_path_parity() -> None:
    """The vendored DFT chunks the folded crops when max_internal_batchsize is
    set; toggling it on the SAME weights must not change the output (covers the
    otherwise-untested chunk / res-chunk decode path in dft.py)."""
    m = _stepper(latent_type="mode").eval()
    m.set_normalization([0, 0, 0], [1, 1, 1], [0, 0], [1, 1])
    state, params, geom = _inputs()  # 16^3 -> 4 folded crops (channels folded)
    with torch.no_grad():
        unchunked = m(state, params, geom)
        m.dft.max_internal_batchsize = 2  # force the chunk path
        chunked = m(state, params, geom)
    assert torch.allclose(unchunked, chunked, atol=1e-5)
    assert torch.isfinite(chunked).all()


def test_encoder_crop_size_must_be_multiple_of_16() -> None:
    with pytest.raises(ValueError, match="multiple of 16"):
        _stepper(encoder_crop_size=8)


def test_anisotropic_encoder_tiles_run_through_dft() -> None:
    model = _stepper(encoder_crop_size=(16, 16, 32)).eval()
    model.set_normalization([0, 0, 0], [1, 1, 1], [0, 0], [1, 1])
    state, params, geom = _inputs(grid=(16, 16, 32))
    encoder_inputs = []
    hook = model.dft.encoder.register_forward_pre_hook(
        lambda _, args: encoder_inputs.append(tuple(args[0].shape))
    )
    with torch.no_grad():
        result = model(state, params, geom)
    hook.remove()

    # Three state channels plus one folded geometry channel, one tile each.
    assert encoder_inputs == [(4, 1, 16, 16, 32)]
    assert result.shape == state.shape


def test_sdf_geom_features_precompute_matches_recompute() -> None:
    """M6 correctness: passing precomputed ``geom_features`` yields output
    byte-identical to letting the stepper recompute the SDF transform inside
    ``forward`` -- the substitution the ESMDA rollout cache relies on."""
    m = _stepper(sdf_features="sdf", encode_geometry=True, latent_type="mode").eval()
    m.set_normalization([0, 0, 0], [1, 1, 1], [0, 0], [1, 1])
    state, params, geom = _inputs(b=2)  # (n_members, ...) like a rollout chunk
    with torch.no_grad():
        feats = m._sdf_features(geom)  # what _rollout_chunk caches once
        out_cached = m(state, params, geom, feats)
        out_recompute = m(state, params, geom)  # geom_features=None -> recompute
    assert torch.equal(out_cached, out_recompute)


def test_rollout_computes_sdf_features_once(tmp_path: Any) -> None:
    """M6 optimisation: an SDF-enabled stepper computes its (static) SDF
    features exactly once per rollout, not once per internal step; the rollout
    still produces a finite trajectory over multiple emitted frames."""
    from neural_surrogates import NeuralSurrogateForwardModel

    stepper = _stepper(sdf_features="sdf", encode_geometry=True, latent_type="mode")
    stepper.set_normalization([0, 0, 0], [1, 1, 1], [0, 0], [1, 1])

    # Minimal trained-model dir: the surrogate reads state/param vars from it but
    # every trained field is overridden below, so no real training artifacts.
    model_dir = tmp_path / "model_weights" / "sdf_stepper"
    model_dir.mkdir(parents=True)
    OmegaConf.save(
        OmegaConf.create(
            {
                "architecture": {"_target_": "neural_surrogates.TadpoleTimeStepper"},
                "dataset": {
                    "root_dir": str(tmp_path / "unused_data"),
                    "state_vars": list(STATE_VARS),
                    "param_vars": list(PARAM_VARS),
                },
            }
        ),
        model_dir / "config.yaml",
    )
    model = NeuralSurrogateForwardModel(
        spinup_forward_model=_StubSpinup(),
        nx=NX,
        ny=NY,
        nz=NZ,
        bounds=[[0.0, NX], [0.0, NY], [0.0, NZ]],
        simulation_time=float(T),
        output_frequency=1.0,
        model_dir=model_dir,
        architecture=stepper,
        state_vars=STATE_VARS,
        param_vars=PARAM_VARS,
        trained_output_frequency=1.0,
        trained_domain={
            "nx": NX,
            "ny": NY,
            "nz": NZ,
            "bounds": [[0.0, NX], [0.0, NY], [0.0, NZ]],
        },
        weights_path=None,
        allow_uninitialized_weights=True,
    )

    calls = {"n": 0}
    original = stepper._sdf_features

    def _counting(geometry: Any) -> Any:
        calls["n"] += 1
        return original(geometry)

    stepper._sdf_features = _counting

    result = model(params=_params())

    # T > 1 emitted frames => >= T internal steps, yet the SDF transform ran once.
    assert result.sizes["time"] == T
    assert calls["n"] == 1
    for v in STATE_VARS:
        assert np.isfinite(result[v].values).all()


# --------------------------------------------------------------------------- #
# Rollout scaffolding (smoke shapes + a synthetic spin-up backend).
# --------------------------------------------------------------------------- #

NZ, NY, NX, T = 16, 16, 16, 4


class _StubSpinup(BaseForwardModel):
    """Synthetic spin-up backend returning a developed pylbm-style field.

    A trimmed copy of the plan-01 e2e stub kept local so this test does not
    import a module whose top-level ``importorskip`` would raise mid-test."""

    def __init__(self, results_dir: Any = None) -> None:
        super().__init__(results_dir=results_dir)
        self.spinup_time = 0.0
        self._rng = np.random.default_rng(0)

    def _apply_inflow_settings(self, params: Any) -> None:
        pass

    def save_results(self, state: Any, sim_name: str = "state") -> None:
        self._save_results(state, sim_name)

    def _clean_output(self) -> None:
        pass

    def disable_spinup(self) -> None:
        self.spinup_time = 0.0

    def run_single(
        self, state: Any = None, params: Any = None, sim_name: Any = "state"
    ) -> xr.Dataset:
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


def _params() -> xr.Dataset:
    t = np.linspace(0.0, 3.0, T)
    return xr.Dataset(
        {
            "inflow_angle": ("time", np.linspace(10.0, 20.0, t.size)),
            "velocity_magnitude": ("time", np.linspace(3.0, 4.0, t.size)),
        },
        coords={"time": t},
    )
