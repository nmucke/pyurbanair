"""``TadpoleLatentGenerator`` (plan 07, phase 2A): representation + model tests.

What is locked down here:

* **Representation parity** -- for every AE spatial mode (``local`` / ``global``
  / ``halo``) crossed with both geometry paths (folded ``encode_geometry`` with
  SDF channels, and a ``geometry_branch``), ``decode_latents(encode_latents(x))``
  reproduces ``TadpoleAE.forward`` within fp32 tolerance -- with nontrivial state
  *and* latent statistics, active branch projections, a non-divisible
  rectangular grid (padding) and chunked ``max_internal_batchsize``. Geometry
  channels never appear in ``z``.
* **Conditioning** -- ``geometry_condition`` equals the conditioning derived
  inside ``encode_latents`` exactly (one shared code path).
* **Velocity net** -- explicit ``hidden_size < D`` is rejected; the output
  projection moves on the first AdamW step and the FiLM / geometry FiLM receive
  gradients on the second; the frozen AE never changes and stays in eval mode.
* **Statistics** -- constant latent channels get a floored std and encode to
  finite values.
* **Sampling** -- reproducible per-member noise across batch sizes, distinct
  seeds give distinct states, ``num_steps`` override, masking/shape.
* **Self-contained reload** -- ``skip_pretrained_load=True, ae_kwargs=...`` +
  ``load_state_dict(strict=True)`` reproduces samples and every buffer.

Gated with ``importorskip`` on the vendored Tadpole runtime deps
(``diffusers`` / ``timm`` / ``einops``), like the neighbouring Tadpole tests.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("diffusers")
pytest.importorskip("timm")
pytest.importorskip("einops")

from hydra.utils import instantiate
from neural_surrogates import LatentEncoding, TadpoleAE, TadpoleLatentGenerator
from omegaconf import OmegaConf

STATE_VARS = ("u", "v", "w")
CROP = 16
C, P, HP = 3, 2, 4
GRID = (16, 16, 32)
RECT_GRID = (16, 24, 40)  # not a multiple of CROP along y/x -> padding

# The tiny velocity net used throughout: D=768 (3 channels x Cl=256 for size S)
# stays as the token width (hidden_size=None -> D), only depth/heads shrink.
NET = dict(n_layers=1, num_heads=2, film_hidden=8, time_embed_dim=8)

MODES = ("local", "global", "halo")
GEOMS = ("fold", "branch")


@pytest.fixture(autouse=True)
def _seed():
    torch.manual_seed(7)
    yield


def _ae_arch(spatial_mode: str, geometry: str, latent_type: str = "mode") -> dict:
    arch: dict[str, Any] = {
        "_target_": "neural_surrogates.TadpoleAE",
        "size": "S",
        "encoder_crop_size": CROP,
        "latent_type": latent_type,
        "normalize": True,
        "sdf_clamp_cells": 8.0,
        "spatial_mode": spatial_mode,
        "halo_size": 16,
    }
    if geometry == "fold":
        arch.update(encode_geometry=True, sdf_features="both", geometry_branch=None)
    else:
        arch.update(
            encode_geometry=False,
            sdf_features="sdf",
            geometry_branch={"width": 8},
        )
    return arch


def _make_ae_export(
    root: Path, spatial_mode: str, geometry: str, *, latent_type: str = "mode"
) -> Path:
    """Fabricate a ``pretrain_autoencoder.py``-shaped AE export (weights.pt +
    config.yaml) for one spatial mode x geometry path. Zero-init heads are
    randomised so the parity checks cannot pass vacuously."""
    arch = _ae_arch(spatial_mode, geometry, latent_type)
    kwargs = {k: v for k, v in arch.items() if k != "_target_"}
    ae = TadpoleAE(n_state_channels=C, **kwargs)
    with torch.no_grad():
        raw: Any = ae.ae
        torch.nn.init.normal_(
            raw.decoder.transformer_decoder.final_layer.out_proj.weight, std=0.05
        )
        if geometry == "branch":
            projections = (
                list(raw.encoder.conv_encoder.geom_proj)
                + list(raw.decoder.conv_decoder.geom_proj)
                + [raw.decoder.geom_latent_proj]
            )
            for proj in projections:
                torch.nn.init.normal_(proj.weight, std=0.05)
    ae.set_normalization([0.1, -0.2, 0.3], [1.0, 1.5, 0.7])

    model_dir = root / "model_weights" / f"ae_{spatial_mode}_{geometry}"
    model_dir.mkdir(parents=True)
    torch.save(ae.state_dict(), model_dir / "weights.pt")
    OmegaConf.save(
        OmegaConf.create(
            {
                "architecture": arch,
                "dataset": {
                    "root_dir": str(root / "ae_data"),
                    "state_vars": list(STATE_VARS),
                    "sdf_features": arch["sdf_features"],
                    "sdf_clamp_cells": 8.0,
                },
            }
        ),
        model_dir / "config.yaml",
    )
    return model_dir


def _reference_ae(ae_dir: Path) -> TadpoleAE:
    """The AE exactly as exported, with ``latent_type="mode"`` for determinism."""
    cfg = OmegaConf.load(ae_dir / "config.yaml")
    kwargs = OmegaConf.to_container(cfg.architecture, resolve=True)
    assert isinstance(kwargs, dict)
    kwargs.pop("_target_")
    kwargs["latent_type"] = "mode"
    ae = TadpoleAE(n_state_channels=C, **kwargs)
    ae.load_state_dict(torch.load(ae_dir / "weights.pt", map_location="cpu"))
    return ae.eval()


def _generator(ae_dir: Path, **kw: Any) -> TadpoleLatentGenerator:
    kw = {**NET, **kw}
    return TadpoleLatentGenerator(
        n_state_channels=C,
        n_params=P,
        param_history_steps=HP,
        pretrained_ae_dir=str(ae_dir),
        **kw,
    )


def _install_nontrivial_latent_stats(m: TadpoleLatentGenerator) -> None:
    g = torch.Generator().manual_seed(11)
    n = m.working_latent_dim
    mean = torch.randn(n, generator=g)
    std = 0.5 + torch.rand(n, generator=g)
    m.set_latent_normalization(mean, std)


def _inputs(b: int = 1, grid: tuple = GRID) -> tuple:
    state = torch.randn(b, C, *grid) * 2.0 + 0.5
    geom = torch.ones(b, *grid)
    geom[:, 3:9, 2:7, 5:12] = 0.0  # one obstacle block
    params_hist = torch.randn(b, HP, P) * 3.0 + 10.0
    return state, geom, params_hist


def _sdf_feats(m: TadpoleLatentGenerator, geom: torch.Tensor) -> torch.Tensor:
    return m.ae._sdf_features(geom)


# --------------------------------------------------------------------------- #
# 1. Representation parity against the AE, all modes x geometry paths.
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("geometry", GEOMS)
@pytest.mark.parametrize("spatial_mode", MODES)
def test_encode_decode_matches_ae_forward(tmp_path, spatial_mode, geometry):
    ae_dir = _make_ae_export(tmp_path, spatial_mode, geometry)
    ref = _reference_ae(ae_dir)
    m = _generator(ae_dir).eval()
    _install_nontrivial_latent_stats(m)
    assert m.ae_fingerprint is not None and len(m.ae_fingerprint) == 64
    assert m.spatial_mode == spatial_mode
    assert m.latent_channels == 256
    assert m.state_latent_dim == C * 256
    if geometry == "fold":
        assert m.n_geometry_channels == 5  # mask + sdf + 3 grad
        assert m.geom_latent_dim == 5 * 256
        assert m.geom_cond_dim == m.geom_latent_dim
    else:
        assert m.n_geometry_channels == 0
        assert m.geom_latent_dim == 0
        assert m.geom_cond_dim == 8 * 8  # branch level-3 width

    state, geom, _ = _inputs(b=2, grid=RECT_GRID)
    feats = _sdf_feats(m, geom)
    # chunk the frozen encoder/decoder batch on both sides
    m.ae.ae.max_internal_batchsize = 2
    ref.ae.max_internal_batchsize = 2
    with torch.no_grad():
        enc = m.encode_latents(state, geom, feats)
        assert enc.z is not None
        # geometry channels are never generated: D == C * Cl, latent grid = pad/16
        assert enc.z.shape == (2, C * 256, 1, 2, 3)
        assert enc.orig_shape == RECT_GRID
        assert torch.isfinite(enc.z).all()
        out = m.decode_latents(enc.z, enc)
        expected = ref(state, geom, feats)
    assert out.shape == state.shape
    torch.testing.assert_close(out, expected, atol=1e-4, rtol=1e-4)
    # obstacle cells are zero
    assert torch.count_nonzero(out * (1 - geom.unsqueeze(1))) == 0
    # ... and the parity is non-trivial: the latent stats really are applied
    # (a wrong denormalisation would decode garbage).
    with torch.no_grad():
        wrong = m.decode_latents(enc.z * 1.5, enc)
    assert not torch.allclose(wrong, expected, atol=1e-3)


def test_local_mode_equals_ae_fold_path_exactly_zero_halo(tmp_path):
    """Local mode goes through ``encode_spatial`` with zero-halo regions -- the
    very same crops the AE's own fold sees -- so the full-grid latent equals the
    AE's folded latent, crop for crop."""
    ae_dir = _make_ae_export(tmp_path, "local", "fold")
    ref = _reference_ae(ae_dir)
    m = _generator(ae_dir).eval()
    _install_nontrivial_latent_stats(m)
    state, geom, _ = _inputs(b=1, grid=GRID)
    feats = _sdf_feats(m, geom)
    with torch.no_grad():
        z_raw, geom_raw, *_ = m._encode_raw(state, geom, feats)
        folded = ref.encode(state, geom, feats)  # (B*C_work*U*V*W, Cl, 1, 1, 1)
    # AE fold order is (B C U V W); the full grid is (B, C*Cl, U, V, W).
    work = torch.cat([z_raw, geom_raw], dim=1)
    b, cw = 1, m.n_working_channels
    u, v, w = z_raw.shape[2:]
    ae_grid = folded.reshape(b, cw, u, v, w, 256).permute(0, 1, 5, 2, 3, 4)
    ae_grid = ae_grid.reshape(b, cw * 256, u, v, w)
    torch.testing.assert_close(work, ae_grid, atol=1e-5, rtol=1e-5)


def test_export_latent_type_sample_is_forced_to_mode(tmp_path):
    ae_dir = _make_ae_export(tmp_path, "local", "branch", latent_type="sample")
    m = _generator(ae_dir).eval()
    assert m.ae_export_latent_type == "sample"
    assert m.ae_kwargs["latent_type"] == "mode"
    assert m.ae.latent_type == "mode" and m.ae.ae.latent_type == "mode"
    _install_nontrivial_latent_stats(m)
    state, geom, _ = _inputs()
    with torch.no_grad():
        a = m.encode_latents(state, geom).z
        b = m.encode_latents(state, geom).z
    assert torch.equal(a, b)  # deterministic latents


# --------------------------------------------------------------------------- #
# 2. geometry_condition == the conditioning inside encode_latents.
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("geometry", GEOMS)
def test_geometry_condition_matches_encode_latents(tmp_path, geometry):
    ae_dir = _make_ae_export(tmp_path, "halo", geometry)
    m = _generator(ae_dir).eval()
    _install_nontrivial_latent_stats(m)
    state, geom, _ = _inputs(b=2, grid=RECT_GRID)
    feats = _sdf_feats(m, geom)
    with torch.no_grad():
        enc = m.encode_latents(state, geom, feats)
        cond = m.geometry_condition(geom, feats, batch_size=2)
        # a single shared mask expanded to the batch is the same thing
        cond1 = m.geometry_condition(geom[:1], feats[:1], batch_size=2)
    assert isinstance(cond, LatentEncoding) and cond.z is None
    assert cond.orig_shape == enc.orig_shape
    assert torch.equal(cond.mask, enc.mask)
    for c in (cond, cond1):
        assert torch.equal(c.geom_cond, enc.geom_cond)
        if geometry == "fold":
            assert torch.equal(c.geom_latents, enc.geom_latents)
            assert c.decoder_geom_feats is None
        else:
            assert c.geom_latents is None
            assert len(c.decoder_geom_feats) == 4
            for a, b in zip(c.decoder_geom_feats, enc.decoder_geom_feats):
                assert torch.equal(a, b)


# --------------------------------------------------------------------------- #
# 3. Rejections.
# --------------------------------------------------------------------------- #


def test_rejects_hidden_size_below_latent_width(tmp_path):
    ae_dir = _make_ae_export(tmp_path, "local", "branch")
    with pytest.raises(ValueError, match="hidden_size"):
        _generator(ae_dir, hidden_size=144)
    # None -> D rounded UP to a multiple of num_heads
    m = _generator(ae_dir, num_heads=5)
    assert m.hidden_size == 770 and m.hidden_size % 5 == 0
    assert m.velocity_net.seqmodel.input_proj.in_features == 768
    assert m.velocity_net.seqmodel.out_proj.out_features == 768
    m2 = _generator(ae_dir, hidden_size=800, num_heads=3)
    assert m2.hidden_size == 801


def test_rejects_ae_without_geometry_path(tmp_path):
    ae_dir = _make_ae_export(tmp_path, "local", "branch")
    arch = _ae_arch("local", "fold")
    arch.update(encode_geometry=False, sdf_features="none", geometry_branch=None)
    ae = TadpoleAE(
        n_state_channels=C, **{k: v for k, v in arch.items() if k != "_target_"}
    )
    torch.save(ae.state_dict(), ae_dir / "weights.pt")
    cfg = OmegaConf.load(ae_dir / "config.yaml")
    cfg.architecture = arch
    OmegaConf.save(cfg, ae_dir / "config.yaml")
    with pytest.raises(ValueError, match="geometry path"):
        _generator(ae_dir)


def test_rejects_both_dir_and_kwargs_or_neither(tmp_path):
    ae_dir = _make_ae_export(tmp_path, "local", "branch")
    with pytest.raises(ValueError, match="exactly one"):
        TadpoleLatentGenerator(
            C, P, HP, pretrained_ae_dir=str(ae_dir), ae_kwargs={}, **NET
        )
    with pytest.raises(ValueError, match="exactly one"):
        TadpoleLatentGenerator(C, P, HP, **NET)


def test_rejects_noise_and_generator_and_bad_params_hist(tmp_path):
    ae_dir = _make_ae_export(tmp_path, "local", "branch")
    m = _generator(ae_dir).eval()
    _install_nontrivial_latent_stats(m)
    _, geom, params_hist = _inputs()
    with pytest.raises(ValueError, match="not both"):
        m.sample(
            params_hist,
            geom,
            initial_noise=torch.zeros(1, m.state_latent_dim, 1, 1, 2),
            generator=torch.Generator().manual_seed(0),
        )
    with pytest.raises(ValueError, match="params_hist"):
        m.sample(params_hist[:, :-1], geom, num_steps=1)
    with pytest.raises(ValueError, match="params_hist"):
        m.sample(params_hist.transpose(1, 2), geom, num_steps=1)
    bad = params_hist.clone()
    bad[0, 0, 0] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        m.sample(bad, geom, num_steps=1)
    with pytest.raises(ValueError, match="initial_noise"):
        m.sample(
            params_hist, geom, initial_noise=torch.zeros(1, m.state_latent_dim, 1, 1, 1)
        )


def test_attention_budget_raises_with_context(tmp_path):
    ae_dir = _make_ae_export(tmp_path, "local", "branch")
    m = _generator(ae_dir, max_latent_tokens=3).eval()
    _install_nontrivial_latent_stats(m)
    state, geom, params_hist = _inputs(b=2)  # 2 * (1, 1, 2) = 4 tokens > 3
    with pytest.raises(ValueError, match=r"4 tokens > max_latent_tokens=3"):
        m(state, params_hist, geom)
    m.max_latent_tokens = 4
    v_pred, v_target = m(state, params_hist, geom)
    assert v_pred.shape == v_target.shape


def test_encode_and_sample_require_installed_latent_stats(tmp_path):
    ae_dir = _make_ae_export(tmp_path, "local", "branch")
    m = _generator(ae_dir).eval()
    state, geom, params_hist = _inputs()
    assert not bool(m.latent_stats_installed)
    with pytest.raises(RuntimeError, match="latent normalisation"):
        m.encode_latents(state, geom)
    with pytest.raises(RuntimeError, match="latent normalisation"):
        m.sample(params_hist, geom, num_steps=1)
    with pytest.raises(ValueError, match="finite"):
        m.set_latent_normalization(
            torch.full((m.working_latent_dim,), float("nan")),
            torch.ones(m.working_latent_dim),
        )
    with pytest.raises(ValueError, match="expected"):
        m.set_latent_normalization(torch.zeros(3), torch.ones(3))


# --------------------------------------------------------------------------- #
# 4. Flow objective, gradients, frozen AE.
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("geometry", GEOMS)
def test_forward_trains_output_projection_then_conditioning(tmp_path, geometry):
    ae_dir = _make_ae_export(tmp_path, "local", geometry)
    m = _generator(ae_dir)
    _install_nontrivial_latent_stats(m)
    m.set_normalization(None, None, [10.0, 10.0], [3.0, 3.0])
    m.train()
    assert m.training and m.ae.training is False
    assert all(not p.requires_grad for p in m.ae.parameters())
    assert all(p.requires_grad for p in m.velocity_net.parameters())
    n_params = m.count_parameters()
    assert n_params == sum(p.numel() for p in m.velocity_net.parameters())
    assert f"{n_params:,}" in repr(m)

    state, geom, params_hist = _inputs(b=2)
    feats = _sdf_feats(m, geom)
    # Snapshot the frozen AE AFTER a warm-up encode: the vendored encoder fills
    # a lazily computed relative-position bias buffer on its first forward
    # (a cache, not a weight), which would otherwise show up as a "change".
    with torch.no_grad():
        m.encode_latents(state, geom, feats)
    ae_before = {k: v.clone() for k, v in m.ae.state_dict().items()}
    opt = torch.optim.AdamW([p for p in m.parameters() if p.requires_grad], lr=1e-3)
    sub = m.velocity_net
    out_proj = sub.seqmodel.out_proj
    w0 = out_proj.weight.clone()

    v_pred, v_target = m(state, params_hist, geom, feats)
    assert v_pred.shape == v_target.shape == (2, m.state_latent_dim, 1, 1, 2)
    assert torch.isfinite(v_pred).all() and torch.isfinite(v_target).all()
    assert not v_target.requires_grad and v_pred.requires_grad
    loss = torch.nn.functional.mse_loss(v_pred, v_target)
    loss.backward()
    assert out_proj.weight.grad is not None
    assert torch.count_nonzero(out_proj.weight.grad) > 0
    # zero-init out_proj blocks upstream gradients on the very first step
    assert torch.count_nonzero(sub.param_mlp[-1].weight.grad) == 0
    opt.step()
    opt.zero_grad()
    assert not torch.equal(out_proj.weight, w0)

    v_pred, v_target = m(state, params_hist, geom, feats)
    torch.nn.functional.mse_loss(v_pred, v_target).backward()
    assert torch.count_nonzero(sub.param_mlp[-1].weight.grad) > 0
    assert sub.geom_film is not None
    assert torch.count_nonzero(sub.geom_film.weight.grad) > 0
    opt.step()

    for k, v in m.ae.state_dict().items():
        assert torch.equal(v, ae_before[k]), k
    assert all(p.grad is None for p in m.ae.parameters())
    assert m.ae.training is False
    m.eval()
    m.train(True)
    assert m.ae.training is False


def test_forward_generator_reproducible_and_autocast_safe(tmp_path):
    ae_dir = _make_ae_export(tmp_path, "global", "fold")
    m = _generator(ae_dir).eval()
    _install_nontrivial_latent_stats(m)
    state, geom, params_hist = _inputs(b=2)
    g1 = torch.Generator().manual_seed(3)
    g2 = torch.Generator().manual_seed(3)
    a = m(state, params_hist, geom, generator=g1)
    b = m(state, params_hist, geom, generator=g2)
    assert torch.equal(a[1], b[1]) and torch.equal(a[0], b[0])
    # a caller's autocast touches the velocity net only; targets stay fp32
    with torch.autocast("cpu", dtype=torch.bfloat16):
        v_pred, v_target = m(
            state, params_hist, geom, generator=torch.Generator().manual_seed(3)
        )
    assert v_target.dtype == torch.float32
    assert torch.equal(v_target, a[1])
    assert torch.isfinite(v_pred.float()).all()


# --------------------------------------------------------------------------- #
# 5. Latent statistics: constant channels.
# --------------------------------------------------------------------------- #


def test_constant_channel_statistics_are_floored(tmp_path):
    ae_dir = _make_ae_export(tmp_path, "local", "fold")
    m = _generator(ae_dir, latent_eps=1e-6).eval()
    # zero state + all-fluid geometry: every 16^3 crop is identical, so every
    # working latent channel is constant across batch and space.
    # (The SDF block of an all-fluid domain still varies spatially, so only the
    # D state channels are guaranteed constant.)
    state = torch.zeros(2, C, *GRID)
    geom = torch.ones(2, *GRID)
    feats = _sdf_feats(m, geom)
    batches = [(state, geom, feats), (state[:1], geom[:1], feats[:1])]
    d = m.state_latent_dim
    with pytest.warns(UserWarning, match="near-constant"):
        mean, std = m.compute_latent_normalization(batches)
    assert bool(m.latent_stats_installed)
    assert torch.isfinite(mean).all() and torch.isfinite(std).all()
    assert torch.equal(std[:d], torch.full((d,), 1e-6))
    assert (std >= 1e-6).all()
    with torch.no_grad():
        enc = m.encode_latents(state, geom, feats)
    assert torch.isfinite(enc.z).all()
    assert torch.isfinite(enc.geom_cond).all()
    assert enc.z.abs().max() < 10.0  # (raw - mean) / eps with raw == mean


def test_compute_latent_normalization_matches_manual(tmp_path):
    ae_dir = _make_ae_export(tmp_path, "local", "branch")
    m = _generator(ae_dir).eval()
    batches = [_inputs(b=2)[:2] + (None,), _inputs(b=1)[:2] + (None,)]
    with torch.no_grad():
        raws = [m._encode_raw(s, g, None)[0].double() for s, g, _ in batches]
    allraw = torch.cat(raws, dim=0)
    exp_mean = allraw.mean(dim=(0, 2, 3, 4))
    exp_std = allraw.var(dim=(0, 2, 3, 4), unbiased=False).sqrt()
    mean, std = m.compute_latent_normalization(iter(batches), max_batches=None)
    torch.testing.assert_close(mean.double(), exp_mean, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(
        std.double(), exp_std.clamp_min(1e-6), atol=1e-5, rtol=1e-5
    )
    # max_batches limits the pass
    mean1, _ = m.compute_latent_normalization(iter(batches), max_batches=1)
    torch.testing.assert_close(
        mean1.double(), raws[0].mean(dim=(0, 2, 3, 4)), atol=1e-5, rtol=1e-5
    )
    with pytest.raises(ValueError, match="no batches"):
        m.compute_latent_normalization([])


# --------------------------------------------------------------------------- #
# 6. Sampling.
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("geometry", GEOMS)
def test_sampling_noise_is_per_member_and_batch_invariant(tmp_path, geometry):
    ae_dir = _make_ae_export(tmp_path, "local", geometry)
    m = _generator(ae_dir, num_sampling_steps=3).eval()
    _install_nontrivial_latent_stats(m)
    m.set_normalization(None, None, [10.0, 10.0], [3.0, 3.0])
    # make the (zero-init) velocity net non-trivial so the ODE actually moves
    with torch.no_grad():
        torch.nn.init.normal_(m.velocity_net.seqmodel.out_proj.weight, std=0.05)
        torch.nn.init.normal_(m.velocity_net.param_mlp[-1].weight, std=0.05)
        torch.nn.init.normal_(m.velocity_net.geom_film.weight, std=0.05)
    _, geom, params_hist = _inputs(b=3)
    feats = _sdf_feats(m, geom)
    latent_grid = m.latent_grid_for(GRID)
    assert latent_grid == (1, 1, 2)
    noise = torch.randn(3, m.state_latent_dim, *latent_grid)

    calls = []
    hook = m.velocity_net.register_forward_pre_hook(lambda _, a: calls.append(1))
    batched = m.sample(params_hist, geom, feats, initial_noise=noise)
    assert len(calls) == 3  # num_sampling_steps default honoured
    calls.clear()
    m.sample(params_hist, geom, feats, initial_noise=noise, num_steps=5)
    assert len(calls) == 5
    hook.remove()

    assert batched.shape == (3, C, *GRID)
    assert torch.isfinite(batched).all()
    assert torch.count_nonzero(batched * (1 - geom.unsqueeze(1))) == 0
    for i in range(3):
        single = m.sample(
            params_hist[i : i + 1],
            geom[i : i + 1],
            feats[i : i + 1],
            initial_noise=noise[i : i + 1],
        )
        torch.testing.assert_close(single, batched[i : i + 1], atol=1e-4, rtol=1e-4)
    # distinct members got distinct states, and distinct seeds differ too
    assert not torch.allclose(batched[0], batched[1])
    s1 = m.sample(params_hist, geom, feats, generator=torch.Generator().manual_seed(1))
    s2 = m.sample(params_hist, geom, feats, generator=torch.Generator().manual_seed(2))
    s1b = m.sample(params_hist, geom, feats, generator=torch.Generator().manual_seed(1))
    assert not torch.allclose(s1, s2)
    torch.testing.assert_close(s1, s1b, atol=1e-6, rtol=1e-6)
    # a single shared (unbatched) mask is accepted for the whole batch
    shared = m.sample(params_hist, geom[0], initial_noise=noise)
    assert shared.shape == (3, C, *GRID)


# --------------------------------------------------------------------------- #
# 7. Self-contained reload (deploy path) + Hydra round trip.
# --------------------------------------------------------------------------- #


def test_self_contained_reload_reproduces_samples(tmp_path):
    ae_dir = _make_ae_export(tmp_path, "halo", "branch")
    m = _generator(ae_dir, num_sampling_steps=2).eval()
    _install_nontrivial_latent_stats(m)
    m.set_normalization(None, None, [10.0, 12.0], [3.0, 4.0])
    with torch.no_grad():
        torch.nn.init.normal_(m.velocity_net.seqmodel.out_proj.weight, std=0.05)
    _, geom, params_hist = _inputs(b=2, grid=RECT_GRID)
    noise = torch.randn(2, m.state_latent_dim, *m.latent_grid_for(RECT_GRID))
    ref = m.sample(params_hist, geom, initial_noise=noise)
    torch.save(m.state_dict(), tmp_path / "weights.pt")

    # ae_kwargs are plain YAML-serialisable types with no download / dir needed
    node = {
        "_target_": "neural_surrogates.TadpoleLatentGenerator",
        "param_history_steps": HP,
        "skip_pretrained_load": True,
        "pretrained_ae_dir": None,
        "ae_kwargs": m.ae_kwargs,
        "num_sampling_steps": 2,
        **NET,
    }
    yaml = OmegaConf.to_yaml(OmegaConf.create(node))
    cfg = OmegaConf.create(yaml)
    assert cfg.ae_kwargs.pretrained == "none"
    assert cfg.ae_kwargs.spatial_mode == "halo"
    fresh = instantiate(cfg, n_state_channels=C, n_params=P)
    assert isinstance(fresh, TadpoleLatentGenerator)
    assert fresh.ae_fingerprint is None
    assert not bool(fresh.latent_stats_installed)
    fresh.load_state_dict(torch.load(tmp_path / "weights.pt"), strict=True)
    fresh.eval()
    assert bool(fresh.latent_stats_installed)
    assert torch.equal(fresh.latent_mean, m.latent_mean)
    assert torch.equal(fresh.latent_std, m.latent_std)
    assert torch.equal(fresh.param_mean, m.param_mean)
    assert torch.equal(fresh.param_std, m.param_std)
    assert all(not p.requires_grad for p in fresh.ae.parameters())
    out = fresh.sample(params_hist, geom, initial_noise=noise)
    torch.testing.assert_close(out, ref, atol=1e-5, rtol=1e-5)
