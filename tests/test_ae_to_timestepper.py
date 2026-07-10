"""AE -> time-stepper (plan 03): unit + end-to-end smoke tests.

Two layers:

* **Unit** (``TadpoleTimeStepper``): the identity-at-init parity invariant (the
  single most informative test of the DFT wiring -- a fresh stepper's forward is
  bit-identical to the pure-AE reconstruction), the param-conditioning no-op
  (``param_conditioning="none"`` and ``n_params=0`` build the same module tree),
  ``encode_geometry`` on/off shapes, padding round-trip on a non-divisible grid,
  weight round-trip parity, and the ``skip_pretrained_load`` (no-AE-dir) build.
* **End-to-end** (``finetune_neural_surrogate.run`` in ``finetune_mode=dft``):
  fabricate a tiny pre-trained ``TadpoleAE`` ``model_dir`` cheaply (no real
  pre-train), compose ``finetuning.yaml`` with ``finetune_mode=dft`` + smoke
  overrides, fine-tune 2 epochs, and assert the exported dir has
  ``weights.pt``/``adapter/``/``config.yaml`` (with ``skip_pretrained_load: true``
  stamped in), loads into ``NeuralSurrogateForwardModel`` and rolls out. Mirrors
  the plan-01 e2e in ``test_lora_finetuning.py`` (reuses its forward-model +
  stub-spinup scaffolding).

The e2e and the finetune-script cross-check tests exercise the DFT
``finetune_mode`` (``finetune_mode/dft.yaml`` + the script's DFT dispatch + the
``tadpole_encdec`` LoRA preset); the unit tests depend only on the wrapper.

Gated with ``importorskip`` on the vendored Tadpole runtime deps
(``diffusers`` / ``timm`` / ``einops``) + ``peft``, so envs without them skip.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest
import xarray as xr
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

torch = pytest.importorskip("torch")
pytest.importorskip("diffusers")
pytest.importorskip("timm")
pytest.importorskip("einops")
pytest.importorskip("peft")

from neural_surrogates import TadpoleAE, TadpoleTimeStepper

from pyurbanair.base_forward_model import BaseForwardModel

_WORKTREE = Path(__file__).resolve().parents[1]
_CONF = _WORKTREE / "conf"
_SCRIPT = _WORKTREE / "scripts" / "neural_surrogate" / "finetune_neural_surrogate.py"

STATE_VARS = ("u", "v", "w")
PARAM_VARS = ("inflow_angle", "velocity_magnitude")

CROP = 16  # encoder_crop_size must be a multiple of 16 (encoder downsamples /16)


def _load_finetune_run():
    spec = importlib.util.spec_from_file_location("finetune_ns_under_test", _SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.run


# --------------------------------------------------------------------------- #
# Unit tests on TadpoleTimeStepper (Phase 1 wrapper; no Phase 2A dependency).
# --------------------------------------------------------------------------- #


def _stepper(
    *,
    n_state_channels=3,
    n_params=2,
    encode_geometry=True,
    sdf_features="none",
    param_conditioning="film",
    latent_type="mode",
    **kw,
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


def _inputs(b=1, grid=(16, 16, 16), n_params=2):
    state = torch.randn(b, 3, *grid)
    params = torch.randn(b, n_params) if n_params else None
    geom = (torch.rand(b, *grid) > 0.2).float()
    return state, params, geom


def test_identity_at_init_parity():
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


def test_identity_at_init_parity_non_divisible_grid():
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


def test_param_conditioning_none_matches_zero_params_module_tree():
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


def test_encode_geometry_on_off_shapes():
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


def test_padding_round_trip_non_divisible_grid():
    """A grid not divisible by the crop size is padded internally and cropped
    back, so the prediction matches the (odd) input shape."""
    m = _stepper(latent_type="mode").eval()
    m.set_normalization([0, 0, 0], [1, 1, 1], [0, 0], [1, 1])
    grid = (20, 24, 18)  # none divisible by 16
    state, params, geom = _inputs(grid=grid)
    with torch.no_grad():
        out = m(state, params, geom)
    assert out.shape[2:] == grid


def test_weight_round_trip_output_parity():
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


def test_skip_pretrained_load_builds_without_ae_dir():
    """``skip_pretrained_load=True`` (the ESMDA-deploy build) needs no AE dir --
    the merged weights.pt carries every weight -- and forwards finitely."""
    m = _stepper().eval()  # _stepper builds with skip_pretrained_load=True
    m.set_normalization([0, 0, 0], [1, 1, 1], [0, 0], [1, 1])
    state, params, geom = _inputs()
    with torch.no_grad():
        out = m(state, params, geom)
    assert out.shape == state.shape


def test_forward_requires_params_when_conditioned():
    """A conditioned build called without params errors loudly rather than
    silently degrading to unconditioned (see the no-op-rule guard in forward)."""
    m = _stepper(n_params=2, param_conditioning="film").eval()
    m.set_normalization([0, 0, 0], [1, 1, 1], [0, 0], [1, 1])
    state, _, geom = _inputs()
    with pytest.raises(ValueError, match="requires params"):
        m(state, None, geom)


def test_predict_residual_false_rejected():
    """``predict_residual`` is a dead knob (the residual is intrinsic); False is
    rejected rather than silently ignored."""
    with pytest.raises(ValueError, match="predict_residual=True"):
        _stepper(predict_residual=False)


def test_identity_at_init_parity_after_lora_injection():
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


def test_max_internal_batchsize_chunked_path_parity():
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


def test_encoder_crop_size_must_be_multiple_of_16():
    with pytest.raises(ValueError, match="multiple of 16"):
        _stepper(encoder_crop_size=8)


def test_sdf_geom_features_precompute_matches_recompute():
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


def test_rollout_computes_sdf_features_once(tmp_path):
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

    def _counting(geometry):
        calls["n"] += 1
        return original(geometry)

    stepper._sdf_features = _counting  # type: ignore[method-assign]

    result = model(params=_params())

    # T > 1 emitted frames => >= T internal steps, yet the SDF transform ran once.
    assert result.sizes["time"] == T
    assert calls["n"] == 1
    for v in STATE_VARS:
        assert np.isfinite(result[v].values).all()


# --------------------------------------------------------------------------- #
# AE model_dir fabrication (shared by the cross-check + e2e tests). Cheap: build
# a TadpoleAE on CPU smoke shapes and save the artifacts pretrain_autoencoder.py
# would write -- NO real pre-training.
# --------------------------------------------------------------------------- #

NZ, NY, NX, T = 16, 16, 16, 4


def _make_ae_model_dir(
    root: Path,
    *,
    size: str = "S",
    encode_geometry: bool = True,
    sdf_features: str = "none",
    sdf_clamp_cells: float = 32.0,
    config_overrides: dict | None = None,
) -> Path:
    """Fabricate a pre-trained AE ``model_dir`` (weights.pt + encoder.pt +
    decoder.pt + config.yaml) matching what ``pretrain_autoencoder.py`` writes.

    ``config_overrides`` patches the saved ``architecture`` node (used by the
    cross-check tests to force an AE<->stepper size / encode_geometry mismatch
    without rebuilding a heavier AE)."""
    ae = TadpoleAE(
        n_state_channels=len(STATE_VARS),
        size=size,
        encoder_crop_size=CROP,
        latent_type="mode",
        encode_geometry=encode_geometry,
        sdf_features=sdf_features,
        sdf_clamp_cells=sdf_clamp_cells,
        normalize=True,
    )
    ae.set_normalization([0.1, 0.2, 0.3], [1.0, 1.1, 1.2])

    model_dir = root / "model_weights" / "tadpole_ae_base"
    model_dir.mkdir(parents=True)
    torch.save(ae.state_dict(), model_dir / "weights.pt")
    ae.ae.save_separate_weights(
        str(model_dir / "encoder.pt"), str(model_dir / "decoder.pt")
    )

    arch = {
        "_target_": "neural_surrogates.TadpoleAE",
        "size": size,
        "encoder_crop_size": CROP,
        "latent_type": "mode",
        "encode_geometry": encode_geometry,
        "sdf_features": sdf_features,
        "sdf_clamp_cells": sdf_clamp_cells,
        "normalize": True,
    }
    arch.update(config_overrides or {})
    OmegaConf.save(
        OmegaConf.create(
            {
                "architecture": arch,
                "dataset": {
                    "root_dir": str(root / "ae_data"),
                    "state_vars": list(STATE_VARS),
                    "sdf_features": sdf_features,
                    "sdf_clamp_cells": sdf_clamp_cells,
                },
            }
        ),
        model_dir / "config.yaml",
    )
    return model_dir


# --------------------------------------------------------------------------- #
# Cross-check tests (Phase 2A: the DFT-mode script fails loud on an AE<->stepper
# mismatch). Substrings are best-effort -- adjust if 2A's message drifts.
# --------------------------------------------------------------------------- #


def _compose_dft(pretrained_dir, data_dir, extra_overrides=None):
    with initialize_config_dir(version_base=None, config_dir=str(_CONF)):
        cfg = compose(
            config_name="neural_surrogate/finetuning",
            overrides=[
                "neural_surrogate/finetune_mode@_global_=dft",
                f"pretrained_model_dir={pretrained_dir}",
                "model_name=tadpole_stepper_test",
                f"dataset.root_dir={data_dir}",
                *(extra_overrides or []),
            ],
        )
    OmegaConf.set_struct(cfg, False)
    return cfg


def test_cross_check_size_mismatch_raises(tmp_path):
    """AE config size != stepper size -> the DFT script fails loud (ValueError)."""
    data_dir = tmp_path / "data"
    _write_transition_dataset(data_dir)
    # A real size-S AE dir whose config *claims* size B -> the stepper (dft.yaml
    # default size S) mismatches the AE config the script cross-checks.
    ae_dir = _make_ae_model_dir(tmp_path, size="S", config_overrides={"size": "B"})
    cfg = _compose_dft(ae_dir, data_dir, ["architecture.encoder_crop_size=16"])
    _shrink_dft_for_cpu(cfg)
    with pytest.raises(ValueError, match="(?i)size"):
        _load_finetune_run()(cfg)


def test_cross_check_encode_geometry_mismatch_raises(tmp_path):
    """AE config encode_geometry != stepper encode_geometry -> ValueError."""
    data_dir = tmp_path / "data"
    _write_transition_dataset(data_dir)
    ae_dir = _make_ae_model_dir(
        tmp_path, encode_geometry=True, config_overrides={"encode_geometry": False}
    )
    cfg = _compose_dft(
        ae_dir,
        data_dir,
        ["architecture.encoder_crop_size=16", "architecture.encode_geometry=true"],
    )
    _shrink_dft_for_cpu(cfg)
    with pytest.raises(ValueError, match="(?i)encode_geometry|geometry"):
        _load_finetune_run()(cfg)


# --------------------------------------------------------------------------- #
# End-to-end: compose finetuning.yaml (finetune_mode=dft) -> run -> exported dir
# loads into NeuralSurrogateForwardModel + rolls out. Mirrors the plan-01 e2e.
# --------------------------------------------------------------------------- #


class _StubSpinup(BaseForwardModel):
    """Synthetic spin-up backend returning a developed pylbm-style field.

    A trimmed copy of the plan-01 e2e stub kept local so this test does not
    import a module whose top-level ``importorskip`` would raise mid-test."""

    def __init__(self, results_dir=None) -> None:
        super().__init__(results_dir=results_dir)
        self.spinup_time = 0.0
        self._rng = np.random.default_rng(0)

    def _apply_inflow_settings(self, params) -> None:
        pass

    def save_results(self, state, sim_name: str = "state") -> None:
        self._save_results(state, sim_name)

    def _clean_output(self) -> None:
        pass

    def disable_spinup(self) -> None:
        self.spinup_time = 0.0

    def run_single(self, state=None, params=None, sim_name="state") -> xr.Dataset:
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


def _write_transition_dataset(root: Path, *, nz=NZ, ny=NY, nx=NX, t=T) -> None:
    """TransitionDataset fixture (state + param nc files), plan-01 shape."""
    rng = np.random.default_rng(0)
    for split, n in {"train": 2, "val": 1}.items():
        (root / "state" / split).mkdir(parents=True, exist_ok=True)
        (root / "param" / split).mkdir(parents=True, exist_ok=True)
        for i in range(n):
            blank = np.zeros((nz, ny, nx), "f4")
            blank[0] = 1.0
            state = {
                v: (
                    ("time", "z", "y", "x"),
                    rng.standard_normal((t, nz, ny, nx)).astype("f4"),
                )
                for v in STATE_VARS
            }
            state["blanking"] = (("z", "y", "x"), blank)
            xr.Dataset(
                state,
                coords=dict(
                    time=np.arange(t) * 1.0,
                    z=np.arange(nz),
                    y=np.arange(ny),
                    x=np.arange(nx),
                ),
            ).to_netcdf(root / "state" / split / f"sample_{i:04d}.nc")
            xr.Dataset(
                {
                    "inflow_angle": (("time",), rng.standard_normal(t).astype("f4")),
                    "velocity_magnitude": (
                        ("time",),
                        (7 + rng.standard_normal(t)).astype("f4"),
                    ),
                },
                coords=dict(time=np.arange(t) * 1.0),
            ).to_netcdf(root / "param" / split / f"sample_{i:04d}.nc")
    # training-data config carrying the trained domain + step size for the loader
    OmegaConf.save(
        OmegaConf.create(
            {
                "domain": {
                    "nx": nx,
                    "ny": ny,
                    "nz": nz,
                    "bounds": [[0.0, nx], [0.0, ny], [0.0, nz]],
                },
                "time": {
                    "simulation_time": float(t),
                    "output_frequency": 1.0,
                    "spinup_time": 0.0,
                },
            }
        ),
        root / "config.yaml",
    )


def _shrink_dft_for_cpu(cfg) -> None:
    """CPU smoke shapes for the DFT fine-tune (mirrors the plan-01/-02 shrinkers).

    dft.yaml owns the trainer/dataset block shape; these keys match
    lora_nextstep.yaml (the DFT trainer reuses the same Trainer)."""
    cfg.dataset.pushforward_steps = 1
    cfg.dataloader.batch_size = 2
    cfg.dataloader.num_workers = 0
    cfg.trainer.num_epochs = 2
    cfg.trainer.device = "cpu"
    cfg.trainer.amp = False
    cfg.trainer.compile_model = False
    cfg.trainer.patience = None
    cfg.trainer.pushforward_epochs_per_step = None
    cfg.trainer.lr_warmup_epochs = None
    cfg.trainer.cudnn_benchmark = False
    cfg.trainer.tf32 = False
    cfg.trainer.resume = False


def test_dft_finetune_end_to_end(tmp_path, monkeypatch):
    """compose finetuning.yaml (finetune_mode=dft) -> run -> exported dir loads
    into NeuralSurrogateForwardModel + rolls out a finite trajectory."""
    data_dir = tmp_path / "data"
    _write_transition_dataset(data_dir)
    ae_dir = _make_ae_model_dir(tmp_path, encode_geometry=True)

    cfg = _compose_dft(
        ae_dir,
        data_dir,
        [
            "model_name=tadpole_stepper_ft_test",
            "architecture.encoder_crop_size=16",
            "lora.rank=4",
            "lora.alpha=8",
        ],
    )
    _shrink_dft_for_cpu(cfg)

    # The dft finetune_mode bundles Trainer + MSELoss (like lora_nextstep).
    assert cfg.trainer._target_.split(".")[-1] == "Trainer"
    assert cfg.loss._target_.split(".")[-1] == "MSELoss"
    assert cfg.architecture._target_.split(".")[-1] == "TadpoleTimeStepper"

    monkeypatch.chdir(tmp_path)
    _load_finetune_run()(cfg)

    out = tmp_path / "model_weights" / "tadpole_stepper_ft_test"
    assert (out / "weights.pt").exists()
    assert (out / "adapter" / "adapter_config.json").exists()
    assert (out / "adapter" / "adapter_model.safetensors").exists()
    assert (out / "config.yaml").exists()

    saved = OmegaConf.load(out / "config.yaml")
    assert saved.architecture._target_.split(".")[-1] == "TadpoleTimeStepper"
    # ESMDA deploy must not depend on the AE dir existing: the exported config
    # rebuilds the net from the merged weights alone.
    assert bool(saved.architecture.skip_pretrained_load) is True
    assert saved.architecture.get("pretrained_ae_dir") in (None, "null")

    # merged weights.pt is a plain state dict loadable into a fresh stepper.
    from hydra.utils import instantiate

    fresh = instantiate(
        saved.architecture,
        n_state_channels=len(STATE_VARS),
        n_params=len(PARAM_VARS),
    )
    fresh.load_state_dict(torch.load(out / "weights.pt"))
    assert isinstance(fresh, TadpoleTimeStepper)

    # and the exported dir drops into NeuralSurrogateForwardModel + rolls out.
    from neural_surrogates import NeuralSurrogateForwardModel

    model = NeuralSurrogateForwardModel(
        spinup_forward_model=_StubSpinup(),
        nx=NX,
        ny=NY,
        nz=NZ,
        bounds=[[0.0, NX], [0.0, NY], [0.0, NZ]],
        simulation_time=float(T),
        output_frequency=1.0,
        model_dir=out,
    )
    result = model(params=_params())
    assert result.sizes["time"] == T
    assert isinstance(model.model, TadpoleTimeStepper)
    for v in STATE_VARS:
        assert np.isfinite(result[v].values).all()
