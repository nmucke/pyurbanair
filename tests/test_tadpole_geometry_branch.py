"""Geometry-branch conditioning of the Tadpole autoencoder.

The branch replaces "fold the geometry through the encoder and reconstruct it"
with "condition the encoder/decoder on multi-resolution geometry features". Four
properties are worth locking down, and this file is organised around them:

* **shapes** -- the branch emits four features at strides ``(1, 2, 4, 16)``, and
  the fold turns them into the crop batch the vendored encoder expects;
* **no-op** -- with ``geometry_branch=None`` (the default) nothing changes: no
  projection sub-modules exist, so the ``state_dict`` key set is the old one and
  HF-style *strict* loading of the vendored encoder/decoder still works;
* **identity at init** -- every projection is zero-initialised, so a fresh
  branch-mode autoencoder reconstructs *exactly* what it would with the branch
  disconnected (``geom_feats=None``);
* **connectivity** -- the loss nevertheless reaches the branch *and* the
  projections, i.e. the zeros are a starting point, not a dead path.

Same conventions as ``test_autoencoder_pretraining.py``: ``importorskip`` for the
vendored autoencoder's runtime deps, CPU only, ``CROP = 16`` and tiny shapes.
"""

from __future__ import annotations

import importlib.util
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest
import xarray as xr
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

torch = pytest.importorskip("torch")
pytest.importorskip("diffusers")
pytest.importorskip("timm")
einops = pytest.importorskip("einops")

from neural_surrogates import GeometryBranch, TadpoleAE

if TYPE_CHECKING:
    # `torch` above is a *variable* (importorskip returns a module object), so
    # `torch.Tensor` is not a usable annotation; import the names statically for
    # the type checker only (`from __future__ import annotations` keeps every
    # annotation lazy at runtime). Mirrors test_autoencoder_adversarial.py.
    from torch import Tensor
    from torch.nn import Module

_WORKTREE = Path(__file__).resolve().parents[1]
_CONF = _WORKTREE / "conf"
_SCRIPT = _WORKTREE / "scripts" / "neural_surrogate" / "pretrain_autoencoder.py"

STATE_VARS = ("u", "v", "w")
C = len(STATE_VARS)
CROP = 16  # encoder_crop_size must be a multiple of 16
GRID = (16, 16, 32)  # non-cubic: U/V/W differ, so a fold-order bug shows up
WIDTH = 4  # tiny branch: out_dims (4, 8, 16, 32)


def _ae(**kw: Any) -> TadpoleAE:
    """Deterministic (``latent_type="mode"``) autoencoder with identity stats."""
    kw.setdefault("encoder_crop_size", CROP)
    kw.setdefault("latent_type", "mode")
    ae = TadpoleAE(n_state_channels=C, size="S", sdf_clamp_cells=8, **kw)
    ae.set_normalization([0.0] * C, [1.0] * C)
    ae.eval()
    return ae


def _branch_ae(sdf_features: str = "none", **kw: Any) -> TadpoleAE:
    return _ae(
        encode_geometry=False,
        sdf_features=sdf_features,
        geometry_branch={"width": WIDTH},
        **kw,
    )


def _branch(ae: TadpoleAE) -> GeometryBranch:
    """The AE's geometry branch, asserted present (a typing convenience)."""
    branch = ae.geometry_branch
    assert isinstance(branch, GeometryBranch)
    return branch


def _feats(ae: TadpoleAE, geom: Tensor, state: Tensor) -> list[Tensor]:
    """The branch features, asserted present (a typing convenience)."""
    feats: list[Tensor] | None = ae._branch_features(geom, None, state)
    assert feats is not None
    return feats


def _inputs(b: int = 1, seed: int = 0) -> tuple[Tensor, Tensor]:
    g = torch.Generator().manual_seed(seed)
    state = torch.randn(b, C, *GRID, generator=g)
    geom = (torch.rand(b, *GRID, generator=g) > 0.2).float()
    return state, geom


# --------------------------------------------------------------------------- #
# GeometryBranch itself + the fold.
# --------------------------------------------------------------------------- #


def test_branch_output_shapes_and_strides() -> None:
    """Four levels at strides (1, 2, 4, 16), with the documented widths."""
    branch = GeometryBranch(in_channels=2, width=WIDTH)
    assert branch.out_dims == (WIDTH, 2 * WIDTH, 4 * WIDTH, 8 * WIDTH)
    feats = branch(torch.randn(2, 2, *GRID))
    assert len(feats) == 4
    for f, dim, stride in zip(feats, branch.out_dims, GeometryBranch.strides):
        assert f.shape[0] == 2 and f.shape[1] == dim
        assert tuple(f.shape[2:]) == tuple(d // stride for d in GRID)


def test_branch_rejects_wrong_channel_count() -> None:
    branch = GeometryBranch(in_channels=2, width=WIDTH)
    with pytest.raises(ValueError, match="expected 2"):
        branch(torch.randn(1, 1, *GRID))


def test_fold_order_matches_state_fold() -> None:
    """`_fold_geom_feats` must tile in the SAME `(B C U V W)` order as the state
    fold -- otherwise every crop is conditioned on a different crop's geometry.

    The reference builds the answer the other way round: repeat the single
    geometry feature across the C folded state channels *first*, then apply the
    upstream state-fold rearrange verbatim."""
    ae = _branch_ae()
    b = 2
    # A stride-1 "feature" with a single channel: directly comparable to the
    # state fold, whose crops are also single-channel.
    feat = torch.randn(b, 1, *GRID)
    folded = ae._fold_geom_feats([feat], C)[0]

    reference = einops.rearrange(
        feat.expand(b, C, *GRID),
        "B C (U Xc) (V Yc) (W Zc) -> (B C U V W) 1 Xc Yc Zc",
        Xc=CROP,
        Yc=CROP,
        Zc=CROP,
    )
    assert folded.shape == reference.shape
    assert torch.equal(folded, reference)


def test_fold_shapes_across_levels() -> None:
    """Every level folds to the same crop batch, at its own resolution."""
    ae = _branch_ae()
    state, geom = _inputs(b=2)
    folded = ae._fold_geom_feats(_feats(ae, geom, state), C)
    n_crops = 2 * C * (GRID[0] // CROP) * (GRID[1] // CROP) * (GRID[2] // CROP)
    for f, dim, stride in zip(folded, _branch(ae).out_dims, (1, 2, 4, 16)):
        assert f.shape == (n_crops, dim, *([max(CROP // stride, 1)] * 3))


# --------------------------------------------------------------------------- #
# Construction contract.
# --------------------------------------------------------------------------- #


def test_branch_and_encode_geometry_are_exclusive() -> None:
    with pytest.raises(ValueError, match="mutually exclusive"):
        _ae(encode_geometry=True, geometry_branch={"width": WIDTH})


def test_sdf_features_allowed_with_branch() -> None:
    """SDF channels feed the branch, so they no longer require encode_geometry --
    but without either consumer they are still an error."""
    ae = _branch_ae(sdf_features="both")
    assert _branch(ae).in_channels == 1 + 4
    assert ae.n_geometry_channels == 0  # geometry never enters the working space
    with pytest.raises(ValueError, match="sdf_features requires"):
        _ae(encode_geometry=False, sdf_features="sdf")


def test_working_space_is_state_only_in_branch_mode() -> None:
    ae = _branch_ae(sdf_features="sdf")
    state, geom = _inputs()
    with torch.no_grad():
        recon, target = ae(state, geom, working_space=True)
    assert recon.shape[1] == C and target.shape[1] == C


# --------------------------------------------------------------------------- #
# The no-op rule.
# --------------------------------------------------------------------------- #


def test_default_build_creates_no_projection_modules() -> None:
    """`geometry_branch=None` must leave the module tree (and hence every
    existing checkpoint) exactly as it was: no projections, no branch."""
    ae = _ae()
    assert ae.geometry_branch is None
    assert ae.ae.encoder.conv_encoder.geom_proj is None
    assert ae.ae.decoder.conv_decoder.geom_proj is None
    assert ae.ae.decoder.geom_latent_proj is None
    stray = [k for k in ae.state_dict() if "geom" in k]
    assert stray == [], f"unexpected geometry keys in the default state dict: {stray}"


def test_default_encoder_decoder_load_strictly() -> None:
    """HF-style strict loading still works for an unconditioned build -- the
    guarantee that the vendored `state_dict` key set is untouched."""
    from neural_surrogates.architectures._tadpole.architecture.p3d import (
        _KLP3DEncoder,
        _P3DDecoder,
    )

    ae = _ae()
    _KLP3DEncoder("S").load_state_dict(ae.ae.encoder.state_dict())
    _P3DDecoder("S").load_state_dict(ae.ae.decoder.state_dict())


def test_default_forward_unchanged_by_the_geom_feats_argument() -> None:
    """The default path never builds features, so `forward` is the old one."""
    torch.manual_seed(7)
    ae = _ae()
    state, geom = _inputs()
    assert ae._branch_features(geom, None, state) is None
    with torch.no_grad():
        direct = ae(state, geom)
        via_ae = ae.ae(
            ae._pad_to_crop_multiple(ae._assemble_working_input(state, geom, None))[0],
            geom_feats=None,
        )
    assert torch.equal(
        direct, ae._denormalize_state(via_ae[:, :C]) * ae._batched_mask(geom, state)
    )


def test_branch_weights_are_separate_from_encoder_decoder() -> None:
    """The branch is its own module (saved as geometry_branch.pt); only the
    zero-init projections live inside the encoder/decoder state dicts."""
    ae = _branch_ae()
    enc_keys = ae.ae.encoder.state_dict()
    dec_keys = ae.ae.decoder.state_dict()
    assert any("conv_encoder.geom_proj" in k for k in enc_keys)
    assert any("conv_decoder.geom_proj" in k for k in dec_keys)
    assert any(k.startswith("geom_latent_proj") for k in dec_keys)
    assert not any(k.startswith("geometry_branch") for k in enc_keys | dec_keys.keys())


# --------------------------------------------------------------------------- #
# Identity at init + connectivity.
# --------------------------------------------------------------------------- #


def test_identity_at_init() -> None:
    """Zero-init projections => a fresh branch-mode AE reconstructs EXACTLY what
    it does with the branch disconnected (`geom_feats=None`)."""
    torch.manual_seed(3)
    ae = _branch_ae(sdf_features="sdf")
    state, geom = _inputs()
    with torch.no_grad():
        conditioned = ae(state, geom)
        x, orig = ae._pad_to_crop_multiple(
            ae._assemble_working_input(state, geom, None)
        )
        unconditioned = ae.ae(x, geom_feats=None)[..., : orig[0], : orig[1], : orig[2]]
        unconditioned = ae._denormalize_state(unconditioned[:, :C]) * ae._batched_mask(
            geom, state
        )
    assert torch.equal(conditioned, unconditioned)


def _projection_modules(ae: TadpoleAE) -> tuple[tuple[Module, str], ...]:
    return (
        (ae.ae.encoder.conv_encoder.geom_proj, "encoder conv stem"),
        (ae.ae.decoder.conv_decoder.geom_proj, "decoder up-path"),
        (ae.ae.decoder.geom_latent_proj, "decoder latent input"),
    )


def test_projection_gradients_at_init() -> None:
    """At init the loss already moves the decoder-side projections:
    ``dL/dW = upstream * feature`` is non-zero even though ``W == 0``. Every
    projection is in the graph (populated ``.grad``, not ``None``); the encoder's
    and the latent one read exactly zero here only because upstream zero-inits
    the transformer decoder's ``final_layer.out_proj``, which makes a *randomly
    initialised* decoder's output independent of its input -- a property of the
    vendored init, not of this wiring (the next test nudges it and everything
    lights up)."""
    torch.manual_seed(5)
    ae = _branch_ae(sdf_features="sdf").train()
    state, geom = _inputs(b=2)
    ae(state, geom).pow(2).mean().backward()

    for module, name in _projection_modules(ae):
        grads = [p.grad for p in module.parameters() if p.grad is not None]
        assert len(grads) == len(list(module.parameters())), f"{name} not in the graph"
    conv_up = ae.ae.decoder.conv_decoder.geom_proj
    assert any(p.grad.abs().max() > 0 for p in conv_up.parameters())
    assert all(
        p.grad is not None for p in _branch(ae).parameters()
    ), "the geometry branch is detached from the graph"


def test_gradients_reach_branch_and_all_projections() -> None:
    """One step off zero-init (upstream's ``out_proj`` and our projections) and
    the loss trains the branch itself as well as every injection point."""
    torch.manual_seed(5)
    ae = _branch_ae(sdf_features="sdf").train()
    with torch.no_grad():  # stand in for the first optimizer steps
        for module, _ in _projection_modules(ae):
            for p in module.parameters():
                p.add_(torch.randn_like(p) * 0.01)
        out_proj = ae.ae.decoder.transformer_decoder.final_layer.out_proj
        out_proj.weight.add_(torch.randn_like(out_proj.weight) * 0.01)

    state, geom = _inputs(b=2)
    ae(state, geom).pow(2).mean().backward()

    for module, name in _projection_modules(ae):
        grads = [p.grad for p in module.parameters() if p.grad is not None]
        assert grads, f"no gradient reached the {name} projection"
        assert any(g.abs().max() > 0 for g in grads), f"zero gradient in {name}"
    branch_grads = [p.grad for p in _branch(ae).parameters() if p.grad is not None]
    assert branch_grads, "no gradient reached the geometry branch"
    assert any(g.abs().max() > 0 for g in branch_grads)


def test_max_internal_batchsize_chunks_geometry_alongside_state() -> None:
    """Chunking the folded batch must chunk the features with it: the chunked and
    unchunked forwards agree (deterministic latent)."""
    torch.manual_seed(11)
    ae = _branch_ae()
    state, geom = _inputs(b=2)
    with torch.no_grad():
        whole = ae(state, geom)
        ae.ae.max_internal_batchsize = 3  # < the 2*C*U*V*W folded crops
        chunked = ae(state, geom)
    torch.testing.assert_close(whole, chunked)


def test_frozen_branch_features_are_cached() -> None:
    """A frozen branch (the DFT setting) is evaluated once per geometry; a
    trainable one is always recomputed so it stays attached to the graph."""
    ae = _branch_ae()
    state, geom = _inputs()
    first = ae._branch_features(geom, None, state)
    assert ae._branch_features(geom, None, state) is not first  # trainable: no cache

    for p in _branch(ae).parameters():
        p.requires_grad_(False)
    frozen = ae._branch_features(geom, None, state)
    assert ae._branch_features(geom, None, state) is frozen
    # a different geometry object busts the size-1 cache
    _, other = _inputs(seed=1)
    assert ae._branch_features(other, None, state) is not frozen


# --------------------------------------------------------------------------- #
# End-to-end: the pre-train script writes geometry_branch.pt.
# --------------------------------------------------------------------------- #

NZ, NY, NX, T = 16, 16, 16, 4


def _load_pretrain_run() -> Callable[..., None]:
    spec = importlib.util.spec_from_file_location("pretrain_ae_branch_test", _SCRIPT)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    run: Callable[..., None] = mod.run
    return run


def _write_dataset(root: Path) -> None:
    rng = np.random.default_rng(0)
    for split, n in {"train": 2, "val": 1}.items():
        (root / "state" / split).mkdir(parents=True, exist_ok=True)
        blank = np.zeros((NZ, NY, NX), "f4")
        blank[0] = 1.0  # bottom layer is obstacle (blanking=1)
        for i in range(n):
            data: dict[str, tuple[tuple[str, ...], np.ndarray]] = {
                v: (
                    ("time", "z", "y", "x"),
                    rng.standard_normal((T, NZ, NY, NX)).astype("f4"),
                )
                for v in STATE_VARS
            }
            data["blanking"] = (("z", "y", "x"), blank)
            xr.Dataset(
                data,
                coords=dict(
                    time=np.arange(T) * 1.0,
                    z=np.arange(NZ),
                    y=np.arange(NY),
                    x=np.arange(NX),
                ),
            ).to_netcdf(root / "state" / split / f"sample_{i:04d}.nc")


def test_pretrain_end_to_end_branch_mode(tmp_path: Path, monkeypatch: Any) -> None:
    data_dir = tmp_path / "data"
    _write_dataset(data_dir)

    with initialize_config_dir(version_base=None, config_dir=str(_CONF)):
        cfg = compose(
            config_name="neural_surrogate/pretrain_autoencoder",
            overrides=[
                f"dataset.root_dir={data_dir}",
                "model_name=tadpole_ae_branch_test",
            ],
        )
    OmegaConf.set_struct(cfg, False)
    cfg.architecture.encoder_crop_size = CROP
    cfg.architecture.encode_geometry = False
    cfg.architecture.geometry_branch = {"width": WIDTH}
    cfg.dataset.sdf_features = cfg.architecture.sdf_features
    cfg.dataset.sdf_clamp_cells = cfg.architecture.sdf_clamp_cells
    cfg.dataloader.batch_size = 2
    cfg.dataloader.num_workers = 0
    cfg.trainer.num_epochs = 1
    cfg.trainer.device = "cpu"
    cfg.trainer.amp = False
    cfg.trainer.compile_model = False
    cfg.trainer.patience = None
    cfg.trainer.lr_warmup_epochs = None
    cfg.trainer.cudnn_benchmark = False
    cfg.trainer.tf32 = False
    cfg.trainer.resume = False

    monkeypatch.chdir(tmp_path)
    _load_pretrain_run()(cfg)

    out = tmp_path / "model_weights" / "tadpole_ae_branch_test"
    assert (out / "geometry_branch.pt").exists()
    for name in ("weights.pt", "encoder.pt", "decoder.pt", "config.yaml"):
        assert (out / name).exists()

    # the exported branch reloads into a freshly built one, and the encoder
    # checkpoint carries the projections that go with it
    from hydra.utils import instantiate

    fresh = instantiate(
        OmegaConf.load(out / "config.yaml").architecture, n_state_channels=C
    )
    fresh.geometry_branch.load_state_dict(torch.load(out / "geometry_branch.pt"))
    fresh.ae.encoder.load_state_dict(torch.load(out / "encoder.pt"))
    fresh.ae.decoder.load_state_dict(torch.load(out / "decoder.pt"))
