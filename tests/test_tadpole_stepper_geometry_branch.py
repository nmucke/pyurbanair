"""Geometry-branch conditioning on the time-stepper side (``TadpoleTimeStepper``).

The branch mode replaces the folded geometry channels with a small *frozen*
``GeometryBranch`` whose 4-level feature pyramid is injected into the frozen
encoder/decoder (through their zero-init 1x1x1 projections) and into the latent
subnetwork (through a zero-init spatial FiLM). What these tests lock down:

* **No-op rule** -- ``geometry_branch=None`` (the default) leaves the module tree
  and the forward output exactly as they were: no extra state-dict keys, no
  ``geom_*`` kwargs reaching the vendored DFT.
* **Identity at init** -- in branch mode every injection is zero-init, so
  ``stepper(state, params, geometry)`` is still *bit-identical* to
  ``stepper._ae_reference_recon(state, geometry)`` (which applies the same branch
  features, because the projections belong to the frozen AE).
* **The FiLM is wired** -- zero at init, but gradients reach it and a non-zero
  FiLM actually changes the output.
* **The branch is frozen and loaded** -- ``geometry_branch.pt`` is mandatory
  (fail loud when missing) and its parameters never train.
* **AE -> stepper handoff** -- a branch-mode AE export (encoder.pt / decoder.pt /
  geometry_branch.pt) reconstructs identically through the stepper's AE path.
* **geom_cond grid** -- the branch's stride-16 level lands on the DFT's unfolded
  latent grid.

Gated with ``importorskip`` on the vendored Tadpole runtime deps, and the
branch-mode tests additionally skip while the ``GeometryBranch`` module (the
autoencoder side of this feature) is absent. Style follows
``tests/test_ae_to_timestepper.py``.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType
from typing import Any, cast

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("diffusers")
pytest.importorskip("timm")
pytest.importorskip("einops")

from neural_surrogates import TadpoleAE, TadpoleTimeStepper
from neural_surrogates.architectures.tadpole_stepper import ParamConditionedSubnetwork
from omegaconf import DictConfig, OmegaConf

_WORKTREE = Path(__file__).resolve().parents[1]
_SCRIPT = _WORKTREE / "scripts" / "neural_surrogate" / "finetune_neural_surrogate.py"

STATE_VARS = ("u", "v", "w")
CROP = 16  # encoder_crop_size must be a multiple of 16 (encoder downsamples /16)
BRANCH = {"width": 4}  # tiny branch: out_dims (4, 8, 16, 32)


def _load_finetune_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "finetune_ns_geom_branch_under_test", _SCRIPT
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _subnetwork(m: TadpoleTimeStepper) -> ParamConditionedSubnetwork:
    """The stepper's latent subnetwork, typed (nn.Module attribute access is
    ``Tensor | Module`` to a type checker)."""
    sub = m.dft.subnetwork
    assert isinstance(sub, ParamConditionedSubnetwork)
    return sub


def _film(m: TadpoleTimeStepper) -> Any:
    """The subnetwork's spatial-FiLM conv (only built in branch mode).

    Typed ``Any``: ``torch`` arrives through ``importorskip`` here, so its names
    are not resolvable as annotations."""
    film = _subnetwork(m).geom_film
    assert film is not None
    return film


def _branch(m: TadpoleTimeStepper) -> Any:
    assert m.geometry_branch is not None
    return m.geometry_branch


def _stepper(**kw: Any) -> TadpoleTimeStepper:
    """A fresh (random-init, no AE dir) stepper on CPU smoke shapes."""
    kw.setdefault("encoder_crop_size", CROP)
    kw.setdefault("sdf_clamp_cells", 8)
    kw.setdefault("latent_type", "mode")
    return TadpoleTimeStepper(
        n_state_channels=3,
        n_params=2,
        size="S",
        pretrained_ae_dir=None,
        skip_pretrained_load=True,
        **kw,
    )


def _branch_stepper(**kw: Any) -> TadpoleTimeStepper:
    """A branch-mode stepper (geometry is fed to the branch, not folded)."""
    kw.setdefault("geometry_branch", dict(BRANCH))
    kw.setdefault("encode_geometry", False)
    return _stepper(**kw)


def _inputs(b: int = 1, grid: tuple = (16, 16, 16), n_params: int = 2) -> tuple:
    state = torch.randn(b, 3, *grid)
    params = torch.randn(b, n_params) if n_params else None
    geom = (torch.rand(b, *grid) > 0.2).float()
    return state, params, geom


# --------------------------------------------------------------------------- #
# No-op rule: geometry_branch=None must change nothing at all.
# --------------------------------------------------------------------------- #


def test_geometry_branch_none_is_a_no_op() -> None:
    """The default build has no branch machinery: same state-dict keys as a build
    that never mentions the knob, no geometry attributes on the DFT, and an
    unchanged forward."""
    default = _stepper()
    explicit_none = _stepper(geometry_branch=None)

    assert default.geometry_branch is None
    assert default.geometry_branch_cfg is None
    assert default.dft.geom_in_dims is None
    assert set(default.state_dict()) == set(explicit_none.state_dict())
    # No projection / FiLM parameters exist at all (not merely zeroed ones).
    assert not [k for k in default.state_dict() if "geom" in k]

    explicit_none.load_state_dict(default.state_dict())
    default.eval()
    explicit_none.eval()
    state, params, geom = _inputs()
    with torch.no_grad():
        assert torch.equal(
            default(state, params, geom), explicit_none(state, params, geom)
        )


def test_geometry_branch_requires_encode_geometry_off() -> None:
    """The two geometry paths are mutually exclusive -- asking for both raises."""
    with pytest.raises(ValueError, match="mutually exclusive"):
        _stepper(geometry_branch=dict(BRANCH), encode_geometry=True)


def test_branch_mode_channel_bookkeeping() -> None:
    """In branch mode the working input is state-only and the SDF channels are
    allowed (they feed the branch, not the encoder)."""
    m = _branch_stepper(sdf_features="sdf")
    assert m.n_geometry_channels == 0
    assert m.n_geom_feature_channels == 1
    assert m.geometry_branch is not None
    # branch in_channels = mask + SDF channels
    assert m.dft.geom_in_dims == tuple(_branch(m).out_dims)


# --------------------------------------------------------------------------- #
# Identity at init (the load-bearing DFT-wiring invariant) in branch mode.
# --------------------------------------------------------------------------- #


def test_identity_at_init_parity_branch_mode() -> None:
    """Every branch injection (encoder/decoder projections + the subnetwork FiLM)
    is zero-init, so the branch-mode forward is still bit-identical to the pure-AE
    reconstruction -- which applies the same branch features, because the
    projections are part of the frozen AE."""
    m = _branch_stepper().eval()
    m.set_normalization([0, 0, 0], [1, 1, 1], [0, 0], [1, 1])
    state, params, geom = _inputs()
    with torch.no_grad():
        out = m(state, params, geom)
        ref = m._ae_reference_recon(state, geom)
    assert out.shape == state.shape
    assert torch.isfinite(out).all()
    assert torch.equal(out, ref)  # exact, bitwise


def test_identity_at_init_parity_branch_mode_non_divisible_grid() -> None:
    """Same invariant on a grid that needs internal padding (the branch features
    are computed on the padded grid, so the strides still line up)."""
    m = _branch_stepper().eval()
    m.set_normalization([0, 0, 0], [1, 1, 1], [0, 0], [1, 1])
    state, params, geom = _inputs(grid=(18, 16, 20))
    with torch.no_grad():
        out = m(state, params, geom)
        ref = m._ae_reference_recon(state, geom)
    assert out.shape == state.shape
    assert torch.equal(out, ref)


def test_branch_mode_chunked_path_parity() -> None:
    """Chunking the folded crops (max_internal_batchsize) must chunk the geometry
    features alongside them -- same weights, same output."""
    m = _branch_stepper().eval()
    m.set_normalization([0, 0, 0], [1, 1, 1], [0, 0], [1, 1])
    state, params, geom = _inputs()
    with torch.no_grad():
        unchunked = m(state, params, geom)
        m.dft.max_internal_batchsize = 2  # force the chunk path
        chunked = m(state, params, geom)
    assert torch.allclose(unchunked, chunked, atol=1e-5)


# --------------------------------------------------------------------------- #
# The subnetwork's spatial FiLM: zero at init, but trainable and load-bearing.
# --------------------------------------------------------------------------- #


def test_geom_film_zero_at_init_and_subnetwork_output_is_zero() -> None:
    """The FiLM conv is zero-init (identity conditioning) and the subnetwork's
    total output is still exactly 0 at init."""
    m = _branch_stepper()
    sub = _subnetwork(m)
    assert sub.geom_cond_dim == _branch(m).out_dims[3]
    assert torch.count_nonzero(_film(m).weight) == 0
    assert torch.count_nonzero(_film(m).bias) == 0

    x = torch.randn(1, sub.in_dim, 1, 1, 1)
    cond = torch.randn(1, sub.geom_cond_dim, 1, 1, 1)
    with torch.no_grad():
        out = sub(x, params=torch.randn(1, 2), geom_cond=cond)
    assert torch.count_nonzero(out) == 0


def _dezero_projections(m: TadpoleTimeStepper) -> None:
    """Break the zero-init projections that make an *untrained* net degenerate.

    Two of them sit between the FiLM and the loss: the subnetwork's own
    ``out_proj`` (zero by ``init_zero_proj``) and the vendored decoder's DiT-style
    ``final_layer.out_proj`` (zero-init upstream, which is why a freshly built
    decoder's output does not depend on its latent at all). With both left at zero
    every upstream gradient is structurally zero and any "the FiLM matters" test
    would pass vacuously, so give them small random weights first -- this stands in
    for "after a few training steps"."""
    with torch.no_grad():
        torch.nn.init.normal_(_subnetwork(m).seqmodel.out_proj.weight, std=0.05)
        decoder = cast(Any, m.dft.decoder)
        final_layer = decoder.transformer_decoder.transformer.final_layer
        torch.nn.init.normal_(final_layer.out_proj.weight, std=0.05)


def test_geom_film_changes_output_once_non_zero() -> None:
    """A non-zero FiLM (what training produces) genuinely modulates the latent, so
    the stepper is no longer the plain AE reconstruction."""
    m = _branch_stepper().eval()
    m.set_normalization([0, 0, 0], [1, 1, 1], [0, 0], [1, 1])
    _dezero_projections(m)
    with torch.no_grad():
        torch.nn.init.normal_(_film(m).weight, std=0.5)
    state, params, geom = _inputs()
    with torch.no_grad():
        out = m(state, params, geom)
        ref = m._ae_reference_recon(state, geom)
    assert torch.isfinite(out).all()
    assert not torch.allclose(out, ref)


def test_gradients_reach_the_film_and_the_subnetwork() -> None:
    """The FiLM conv and the subnetwork sit on the autograd graph (see
    ``_dezero_projections`` for why the zero-init projections are broken first)."""
    m = _branch_stepper()
    m.set_normalization([0, 0, 0], [1, 1, 1], [0, 0], [1, 1])
    _dezero_projections(m)
    sub = _subnetwork(m)
    state, params, geom = _inputs()
    m(state, params, geom).sum().backward()

    assert _film(m).weight.grad is not None
    assert torch.count_nonzero(_film(m).weight.grad) > 0
    assert sub.seqmodel.out_proj.weight.grad is not None
    assert torch.count_nonzero(sub.seqmodel.out_proj.weight.grad) > 0
    # the frozen branch itself never accumulates gradients
    assert all(p.grad is None for p in _branch(m).parameters())


# --------------------------------------------------------------------------- #
# Loading / freezing the pre-trained branch.
# --------------------------------------------------------------------------- #


def test_branch_is_frozen_and_loaded_from_the_ae_dir(tmp_path: Path) -> None:
    """``geometry_branch.pt`` is loaded verbatim and the branch stays frozen."""
    ae_dir = _make_ae_model_dir(tmp_path, geometry_branch=dict(BRANCH))
    m = TadpoleTimeStepper(
        n_state_channels=3,
        n_params=2,
        size="S",
        pretrained_ae_dir=str(ae_dir),
        encoder_crop_size=CROP,
        encode_geometry=False,
        geometry_branch=dict(BRANCH),
        latent_type="mode",
    )
    saved = torch.load(ae_dir / "geometry_branch.pt", map_location="cpu")
    loaded = _branch(m).state_dict()
    assert set(saved) == set(loaded)
    for k, v in saved.items():
        assert torch.equal(v, loaded[k])
    assert not any(p.requires_grad for p in _branch(m).parameters())
    # the projections that consume its features are frozen with the rest of the
    # AE (encoder_ft_state / decoder_ft_state = "frozen" covers them)
    assert not any(
        p.requires_grad
        for n, p in m.named_parameters()
        if "geom_proj" in n or "geom_latent_proj" in n
    )


def test_missing_geometry_branch_file_raises_actionable(tmp_path: Path) -> None:
    """An AE dir without geometry_branch.pt cannot supply the frozen branch."""
    ae_dir = _make_ae_model_dir(tmp_path, geometry_branch=dict(BRANCH))
    (ae_dir / "geometry_branch.pt").unlink()
    with pytest.raises(FileNotFoundError, match="geometry_branch.pt"):
        TadpoleTimeStepper(
            n_state_channels=3,
            n_params=2,
            size="S",
            pretrained_ae_dir=str(ae_dir),
            encoder_crop_size=CROP,
            encode_geometry=False,
            geometry_branch=dict(BRANCH),
        )


def test_branch_weights_travel_in_the_state_dict(tmp_path: Path) -> None:
    """The ESMDA deploy path (skip_pretrained_load + pretrained_ae_dir=null)
    rebuilds a branch-mode stepper from weights.pt alone -- the branch is a
    submodule, so its weights are in there."""
    trained = _branch_stepper()
    assert any(k.startswith("geometry_branch.") for k in trained.state_dict())
    trained.set_normalization([0, 0, 0], [1, 1, 1], [0, 0], [1, 1])
    trained.eval()
    state, params, geom = _inputs()
    with torch.no_grad():
        out = trained(state, params, geom)

    deployed = _branch_stepper()  # random init, no AE dir
    deployed.load_state_dict(trained.state_dict())
    deployed.eval()
    with torch.no_grad():
        assert torch.allclose(out, deployed(state, params, geom), atol=1e-6)


# --------------------------------------------------------------------------- #
# geom_cond grid check.
# --------------------------------------------------------------------------- #


def test_geom_cond_lands_on_the_latent_grid() -> None:
    """The branch's stride-16 level is exactly the DFT's unfolded latent grid."""
    m = _branch_stepper().eval()
    m.set_normalization([0, 0, 0], [1, 1, 1], [0, 0], [1, 1])
    state, _, geom = _inputs(grid=(18, 16, 20))  # padded to (32, 16, 32) at CROP=16
    with torch.no_grad():
        branch = m._geom_branch_kwargs(state, geom, None)
        x = m._assemble_working_input(state, geom, None)
        x_pad, _ = m._pad_to_crop_multiple(x)
    assert tuple(branch["geom_cond"].shape[2:]) == tuple(
        s // 16 for s in x_pad.shape[2:]
    )
    m._check_geom_cond_grid(branch["geom_cond"], x_pad)  # no raise


def test_geom_cond_grid_mismatch_raises() -> None:
    """A level-3 feature off the latent grid fails loud rather than broadcasting."""
    m = _branch_stepper()
    x_pad = torch.zeros(1, 3, 16, 16, 16)
    bad = torch.zeros(1, _branch(m).out_dims[3], 2, 2, 2)  # latent grid is 1^3
    with pytest.raises(ValueError, match="latent grid"):
        m._check_geom_cond_grid(bad, x_pad)


# --------------------------------------------------------------------------- #
# AE -> stepper handoff: the stepper's AE path reproduces the AE's own forward.
# --------------------------------------------------------------------------- #


def _make_ae_model_dir(root: Path, *, geometry_branch: dict | None = None) -> Path:
    """Fabricate a pre-trained AE ``model_dir`` the way ``pretrain_autoencoder.py``
    writes one (weights.pt + encoder.pt + decoder.pt + config.yaml, plus
    geometry_branch.pt in branch mode) -- no real pre-training."""
    branch_kwargs: dict = (
        {} if geometry_branch is None else {"geometry_branch": geometry_branch}
    )
    ae = TadpoleAE(
        n_state_channels=len(STATE_VARS),
        size="S",
        encoder_crop_size=CROP,
        latent_type="mode",
        encode_geometry=geometry_branch is None,
        sdf_features="none",
        normalize=True,
        **branch_kwargs,
    )
    ae.set_normalization([0.1, 0.2, 0.3], [1.0, 1.1, 1.2])
    if geometry_branch is not None:
        # Stand in for "this AE was actually trained": with the shipped zero-init
        # the geometry injections (and the decoder's DiT final layer) contribute
        # nothing, and a reconstruction-parity check would hold even if the
        # projections were never applied. Random weights make the parity real.
        with torch.no_grad():
            raw = cast(Any, ae.ae)
            torch.nn.init.normal_(
                raw.decoder.transformer_decoder.final_layer.out_proj.weight, std=0.05
            )
            projections = (
                list(raw.encoder.conv_encoder.geom_proj)
                + list(raw.decoder.conv_decoder.geom_proj)
                + [raw.decoder.geom_latent_proj]
            )
            for proj in projections:
                torch.nn.init.normal_(proj.weight, std=0.05)

    model_dir = root / "model_weights" / "tadpole_ae_branch"
    model_dir.mkdir(parents=True)
    torch.save(ae.state_dict(), model_dir / "weights.pt")
    ae.ae.save_separate_weights(
        str(model_dir / "encoder.pt"), str(model_dir / "decoder.pt")
    )
    if geometry_branch is not None:
        ae_branch = ae.geometry_branch
        assert ae_branch is not None
        torch.save(ae_branch.state_dict(), model_dir / "geometry_branch.pt")

    OmegaConf.save(
        OmegaConf.create(
            {
                "architecture": {
                    "_target_": "neural_surrogates.TadpoleAE",
                    "size": "S",
                    "encoder_crop_size": CROP,
                    "latent_type": "mode",
                    "encode_geometry": geometry_branch is None,
                    "sdf_features": "none",
                    "normalize": True,
                    "geometry_branch": geometry_branch,
                },
                "dataset": {
                    "root_dir": str(root / "ae_data"),
                    "state_vars": list(STATE_VARS),
                    "sdf_features": "none",
                },
            }
        ),
        model_dir / "config.yaml",
    )
    return model_dir


def test_ae_dir_handoff_reconstruction_parity(tmp_path: Path) -> None:
    """A branch-mode AE and a stepper built from its export reconstruct the same
    field: the encoder/decoder projections travel in encoder.pt/decoder.pt and the
    branch in geometry_branch.pt, so the stepper's AE path IS the AE."""
    ae_dir = _make_ae_model_dir(tmp_path, geometry_branch=dict(BRANCH))
    ae = TadpoleAE(
        n_state_channels=len(STATE_VARS),
        size="S",
        encoder_crop_size=CROP,
        latent_type="mode",
        encode_geometry=False,
        normalize=True,
        geometry_branch=dict(BRANCH),
    )
    ae.load_state_dict(torch.load(ae_dir / "weights.pt", map_location="cpu"))
    ae.eval()

    stepper = TadpoleTimeStepper(
        n_state_channels=3,
        n_params=2,
        size="S",
        pretrained_ae_dir=str(ae_dir),
        encoder_crop_size=CROP,
        encode_geometry=False,
        geometry_branch=dict(BRANCH),
        latent_type="mode",
    ).eval()

    state, params, geom = _inputs()
    with torch.no_grad():
        ae_recon = ae(state, geom)
        ref = stepper._ae_reference_recon(state, geom)
        out = stepper(state, params, geom)
    # Unlike the zero-init parity tests above, this AE has a *random* decoder
    # final layer, so its output genuinely depends on the latent and the check
    # is sensitive to last-bit kernel rounding. All three paths now hand the
    # decoder a contiguous latent (see model/dft.py), which makes them bit-exact
    # on the dev box; the tolerance leaves float32 headroom for other CPUs.
    assert torch.allclose(ae_recon, ref, rtol=1e-5, atol=1e-5)
    assert torch.allclose(ae_recon, out, rtol=1e-5, atol=1e-5)
    assert torch.allclose(ref, out, rtol=1e-5, atol=1e-5)

    # ... and the parity above is non-trivial: the geometry injection really is
    # applied on the stepper's (skip-wrapped) encoder path.
    with torch.no_grad():
        torch.nn.init.zeros_(
            cast(Any, stepper.dft.encoder).conv_encoder.conv_encoder.geom_proj[0].weight
        )
        no_geom = stepper._ae_reference_recon(state, geom)
    assert not torch.allclose(ae_recon, no_geom)


# --------------------------------------------------------------------------- #
# Fine-tune script: the AE <-> stepper geometry_branch cross-check.
# --------------------------------------------------------------------------- #


def _ae_arch(**overrides: Any) -> DictConfig:
    node = {
        "size": "S",
        "encode_geometry": True,
        "sdf_features": "none",
        "normalize": True,
        "geometry_branch": None,
    }
    node.update(overrides)
    return OmegaConf.create(node)


def test_cross_check_branch_only_on_the_ae_raises() -> None:
    """A branch-mode AE paired with a plain stepper must fail loud."""
    mod = _load_finetune_module()
    model = _stepper()  # geometry_branch=None
    ae_arch = _ae_arch(geometry_branch={"width": 4})
    with pytest.raises(ValueError, match="geometry_branch"):
        mod._check_ae_stepper_match(ae_arch, model)


def test_cross_check_branch_config_mismatch_raises() -> None:
    """Same knob, different branch config -> the frozen projections would not
    match the branch that feeds them."""
    mod = _load_finetune_module()
    model = _branch_stepper()
    ae_arch = _ae_arch(
        encode_geometry=False, geometry_branch={"width": BRANCH["width"] * 2}
    )
    with pytest.raises(ValueError, match="geometry_branch"):
        mod._check_ae_stepper_match(ae_arch, model)


def test_cross_check_matching_branch_passes() -> None:
    """Equal branch configs (and equal everything else) pass."""
    mod = _load_finetune_module()
    model = _branch_stepper()
    ae_arch = _ae_arch(encode_geometry=False, geometry_branch=dict(BRANCH))
    mod._check_ae_stepper_match(ae_arch, model)  # no raise


def test_cross_check_both_null_passes() -> None:
    """The no-op path stays clean: both sides null."""
    mod = _load_finetune_module()
    mod._check_ae_stepper_match(_ae_arch(), _stepper())  # no raise


# --------------------------------------------------------------------------- #
# LoRA preset safety: the new 1x1x1 projections must never be adapted (peft 0.19
# cannot merge a 1x1x1 Conv3d LoRA -- see finetuning/targets.py).
# --------------------------------------------------------------------------- #


def test_trainable_modules_cover_the_film_but_not_the_branch() -> None:
    """dft.yaml's ``trainable_modules`` (``subnetwork`` + the scales) unfreezes the
    spatial FiLM along with the rest of the subnetwork, and leaves the frozen
    branch (and the frozen encoder/decoder projections) alone."""
    mod = _load_finetune_module()
    tokens = list(
        OmegaConf.load(
            _WORKTREE / "conf/neural_surrogate/finetune_mode/dft.yaml"
        ).trainable_modules
    )
    m = _branch_stepper()
    m.requires_grad_(False)
    assert mod._unfreeze_trainable_modules(m, tokens) > 0
    trainable = {n for n, p in m.named_parameters() if p.requires_grad}
    assert "dft.subnetwork.geom_film.weight" in trainable
    assert not [n for n in trainable if n.startswith("geometry_branch.")]
    assert not [n for n in trainable if "geom_proj" in n or "geom_latent_proj" in n]


def test_lora_preset_skips_the_geometry_projections() -> None:
    resolve_target_modules = pytest.importorskip(
        "neural_surrogates.finetuning"
    ).resolve_target_modules
    m = _branch_stepper()
    targets = resolve_target_modules(m, preset="tadpole_encdec")
    named = dict(m.named_modules())
    for name in targets:
        module = named[name]
        assert not (
            isinstance(module, torch.nn.Conv3d)
            and all(k == 1 for k in module.kernel_size)
        ), f"{name} is a 1x1x1 Conv3d and must not be LoRA-adapted"
    assert not [t for t in targets if "geom" in t]
