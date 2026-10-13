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
* **adaLN-Zero velocity net** (``velocity_conditioning="adaln"``) -- zero
  output at init, output projection trained first and the modulation next,
  ``tau`` changes the velocity once trained, checkpointing leaves gradients
  unchanged, the default build is the unchanged FiLM net, and the
  self-contained reload round-trips.

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

# The fabricated AE export (+ its architecture node) is shared with the
# trainer / deploy tests through the fixtures module.
from tests.neural_surrogates._latent_generator_fixtures import CROP, NET, STATE_VARS, C
from tests.neural_surrogates._latent_generator_fixtures import ae_arch as _ae_arch
from tests.neural_surrogates._latent_generator_fixtures import (
    make_ae_export as _make_ae_export,
)

P, HP = 2, 4
GRID = (16, 16, 32)
RECT_GRID = (16, 24, 40)  # not a multiple of CROP along y/x -> padding

MODES = ("local", "global", "halo")
GEOMS = ("fold", "branch")


@pytest.fixture(autouse=True)
def _seed():
    torch.manual_seed(7)
    yield


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


def test_conditioning_schema_is_installed_and_validated(tmp_path):
    """``params_hist`` is a bare tensor: only the installed schema can catch a
    reordered conditioning vector or a history saved at another cadence."""
    ae_dir = _make_ae_export(tmp_path, "local", "branch")
    m = _generator(ae_dir).eval()
    _install_nontrivial_latent_stats(m)
    _, geom, params_hist = _inputs(b=1)
    feats = _sdf_feats(m, geom)
    names = ("inflow_angle", "velocity_magnitude")
    dt = 5.0

    # (a) Nothing installed: a supplied claim has nothing to check against.
    assert m.param_names is None and m.history_dt_seconds is None
    with pytest.raises(ValueError, match="carries none"):
        m.sample(params_hist, geom, feats, num_steps=1, param_names=names)
    with pytest.raises(ValueError, match="carries none"):
        m.sample(params_hist, geom, feats, num_steps=1, history_dt_seconds=dt)

    # (b) The schema must describe the P conditioning columns.
    with pytest.raises(ValueError, match="n_params"):
        m.set_conditioning_schema((*names, "extra"), dt)
    with pytest.raises(ValueError, match="history_dt_seconds"):
        m.set_conditioning_schema(names, 0.0)
    m.set_conditioning_schema(names, dt)
    assert m.param_names == names and m.history_dt_seconds == dt
    # Provenance, not weights: nothing widens the strict state dict.
    assert not [k for k in m.state_dict() if "param_names" in k or "history_dt" in k]

    # (c) Order-sensitive names, cadence within 1e-6 relative.
    with pytest.raises(ValueError, match="order matters"):
        m.sample(params_hist, geom, feats, num_steps=1, param_names=names[::-1])
    with pytest.raises(ValueError, match="trained cadence"):
        m.sample(params_hist, geom, feats, num_steps=1, history_dt_seconds=5.001)
    out = m.sample(
        params_hist,
        geom,
        feats,
        num_steps=1,
        param_names=names,
        history_dt_seconds=dt * (1 + 1e-9),
    )
    assert out.shape == (1, C, *GRID)

    # (d) velocity() takes (and checks) the same claim.
    cond = m.geometry_condition(geom, feats, batch_size=1)
    z = torch.zeros(1, m.state_latent_dim, *m.latent_grid_for(GRID))
    tau = torch.zeros(1)
    with pytest.raises(ValueError, match="order matters"):
        m.velocity(z, tau, params_hist, cond, param_names=names[::-1])
    with pytest.raises(ValueError, match="trained cadence"):
        m.velocity(z, tau, params_hist, cond, history_dt_seconds=1.0)
    v = m.velocity(z, tau, params_hist, cond, param_names=names, history_dt_seconds=dt)
    assert v.shape == z.shape
    # A model whose schema records no cadence refuses a cadence claim.
    m.set_conditioning_schema(names, None)
    with pytest.raises(ValueError, match="no cadence"):
        m.sample(params_hist, geom, feats, num_steps=1, history_dt_seconds=dt)


def test_encode_and_sample_require_installed_latent_stats(tmp_path):
    ae_dir = _make_ae_export(tmp_path, "local", "branch")
    m = _generator(ae_dir).eval()
    state, geom, params_hist = _inputs()
    assert not bool(m.latent_stats_installed)
    with pytest.raises(RuntimeError, match="latent normalisation"):
        m.encode_latents(state, geom)
    with pytest.raises(RuntimeError, match="latent normalisation"):
        m.sample(params_hist, geom, num_steps=1)
    with pytest.raises(RuntimeError, match="latent normalisation"):
        m.forward_cached(
            torch.zeros(1, m.state_latent_dim, 1, 1, 2),
            torch.zeros(1, m.geom_cond_dim, 1, 1, 2),
            params_hist,
        )
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


def test_forward_uses_linear_interpolation_and_exact_velocity_target(
    tmp_path: Path, monkeypatch: Any
) -> None:
    ae_dir = _make_ae_export(tmp_path, "global", "fold")
    m = _generator(ae_dir).eval()
    _install_nontrivial_latent_stats(m)
    state, geom, params_hist = _inputs(b=2)
    cond = m.encode_latents(state, geom)
    assert cond.z is not None

    seed = 29
    expected_rng = torch.Generator().manual_seed(seed)
    z0 = torch.randn(
        cond.z.shape,
        generator=expected_rng,
        device=cond.z.device,
        dtype=cond.z.dtype,
    )
    tau = torch.rand(
        cond.z.shape[0],
        generator=expected_rng,
        device=cond.z.device,
        dtype=cond.z.dtype,
    )
    expected_z_tau = (1 - tau[:, None, None, None, None]) * z0 + tau[
        :, None, None, None, None
    ] * cond.z
    seen = {}

    def capture_velocity(z: Any, flow_time: Any, params: Any, geom_cond: Any) -> Any:
        seen["z"] = z
        seen["tau"] = flow_time
        return torch.zeros_like(z)

    monkeypatch.setattr(m, "_velocity", capture_velocity)
    _, target = m(
        state,
        params_hist,
        geom,
        generator=torch.Generator().manual_seed(seed),
    )

    assert torch.equal(seen["tau"], tau)
    assert torch.allclose(seen["z"], expected_z_tau)
    assert torch.allclose(target, cond.z - z0)


@pytest.mark.parametrize("geometry", GEOMS)
def test_forward_cached_equals_forward(tmp_path, geometry):
    """A latent-cache item (raw ``_encode_raw`` latents + raw geometry
    conditioning) gives exactly :meth:`forward`'s draw for the same seed."""
    ae_dir = _make_ae_export(tmp_path, "local", geometry)
    m = _generator(ae_dir).eval()
    _install_nontrivial_latent_stats(m)
    state, geom, params_hist = _inputs(b=2)
    feats = _sdf_feats(m, geom)
    z_raw, geom_raw, branch_cond, *_ = m._encode_raw(state, geom, feats)
    cached_geom = branch_cond if geometry == "branch" else geom_raw
    expected = m(
        state, params_hist, geom, feats, generator=torch.Generator().manual_seed(5)
    )
    got = m.forward_cached(
        z_raw, cached_geom, params_hist, generator=torch.Generator().manual_seed(5)
    )
    assert torch.equal(got[0], expected[0]) and torch.equal(got[1], expected[1])


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


def _ae_in_mode(ae_dir: Path, mode: str) -> TadpoleAE:
    """The exported AE's weights, run in spatial ``mode`` (same state-dict keys)."""
    cfg = OmegaConf.load(ae_dir / "config.yaml")
    kwargs = OmegaConf.to_container(cfg.architecture, resolve=True)
    assert isinstance(kwargs, dict)
    kwargs.pop("_target_")
    kwargs.update(latent_type="mode", spatial_mode=mode)
    ae = TadpoleAE(n_state_channels=C, **kwargs)
    ae.load_state_dict(torch.load(ae_dir / "weights.pt", map_location="cpu"))
    return ae.eval()


def _ae_decode(
    ae: TadpoleAE,
    m: TadpoleLatentGenerator,
    enc: LatentEncoding,
    state: torch.Tensor,
    geom: torch.Tensor,
    feats: torch.Tensor,
) -> torch.Tensor:
    """``enc.z`` decoded through ``ae``'s OWN ``decode`` path (its fold layout
    in local mode, its feature folding), denormalised and masked."""
    from einops import rearrange

    b = state.shape[0]
    full = m._denormalize_latents(enc.z, 0)
    if enc.geom_latents is not None:
        full = torch.cat([full, enc.geom_latents], dim=1)
    cw = full.shape[1] // m.latent_channels
    folded = full.reshape(b * cw, m.latent_channels, *full.shape[2:])
    gf = ae._branch_features(geom, feats, state)
    if gf is not None:
        gf = ae._fold_geom_feats(gf, cw)
    if ae.spatial_mode == "local":  # one latent cell per (cubic CROP) tile
        u, v, w = folded.shape[2:]
        tiles = rearrange(folded, "N L U V W -> (N U V W) L 1 1 1")
        decoded = rearrange(
            ae.decode(tiles, gf),
            "(N U V W) 1 a b c -> N 1 (U a) (V b) (W c)",
            U=u,
            V=v,
            W=w,
        )
    else:
        decoded = ae.decode(folded, gf)
    d, h, w = enc.orig_shape
    recon = decoded.reshape(b, cw, *decoded.shape[2:])[:, :C, :d, :h, :w]
    return ae._denormalize_state(recon) * geom.unsqueeze(1)


@pytest.mark.parametrize("geometry", GEOMS)
def test_decode_spatial_mode_override(tmp_path, geometry):
    """``decode_latents(spatial_mode=, halo_size=)`` decodes the SAME latent
    grid with another decoder policy: ``None`` is the trained mode exactly (and
    so is naming it), each mode equals that mode's own AE decode path on a
    padded (non-crop-multiple) grid, obstacles stay zero, and halo 0 is local."""
    ae_dir = _make_ae_export(tmp_path, "local", geometry)
    m = _generator(ae_dir).eval()
    _install_nontrivial_latent_stats(m)
    state, geom, _ = _inputs(b=2, grid=RECT_GRID)
    feats = _sdf_feats(m, geom)
    with torch.no_grad():
        enc = m.encode_latents(state, geom, feats)
        trained = m.decode_latents(enc.z, enc)
        assert torch.equal(m.decode_latents(enc.z, enc, spatial_mode="local"), trained)
        decoded = {}
        for mode in MODES:
            out = m.decode_latents(enc.z, enc, spatial_mode=mode)
            assert out.shape == state.shape
            assert torch.isfinite(out).all()
            assert torch.count_nonzero(out * (1 - geom.unsqueeze(1))) == 0
            ref = _ae_decode(_ae_in_mode(ae_dir, mode), m, enc, state, geom, feats)
            torch.testing.assert_close(out, ref, atol=1e-4, rtol=1e-4)
            decoded[mode] = out
        assert not torch.allclose(decoded["global"], trained)
        assert not torch.allclose(decoded["halo"], trained)
        torch.testing.assert_close(
            m.decode_latents(enc.z, enc, spatial_mode="halo", halo_size=0),
            trained,
            atol=1e-6,
            rtol=1e-6,
        )
    with pytest.raises(ValueError, match="spatial_mode"):
        m.decode_latents(enc.z, enc, spatial_mode="patchwise")
    with pytest.raises(ValueError, match="halo_size"):
        m.decode_latents(enc.z, enc, spatial_mode="halo", halo_size=8)


def test_decode_override_inherited_halo_and_global_ae(tmp_path):
    """A halo AE: ``None`` and naming ``halo`` agree, ``halo_size`` overrides
    the trained halo. A global AE pads only to stride 16, so a crop wider than
    16 leaves the (16, 32, 48) padded grid non-tileable in x: local / halo
    decode it with a shorter last tile instead of refusing or padding."""
    h_dir = _make_ae_export(tmp_path / "halo", "halo", "branch")
    h = _generator(h_dir).eval()
    _install_nontrivial_latent_stats(h)
    state, geom, _ = _inputs(b=1, grid=RECT_GRID)
    feats = _sdf_feats(h, geom)
    with torch.no_grad():
        enc = h.encode_latents(state, geom, feats)
        trained = h.decode_latents(enc.z, enc)
        assert torch.equal(h.decode_latents(enc.z, enc, spatial_mode="halo"), trained)
        assert torch.equal(h.decode_latents(enc.z, enc, halo_size=16), trained)
        assert not torch.allclose(h.decode_latents(enc.z, enc, halo_size=0), trained)

    g_dir = _make_ae_export(tmp_path / "global", "global", "fold")
    g = _generator(g_dir).eval()
    _install_nontrivial_latent_stats(g)
    g.encoder_crop_size = (16, 32, 32)  # decoder tiles only; global encodes whole
    with torch.no_grad():
        enc = g.encode_latents(state, geom, _sdf_feats(g, geom))
        assert enc.z.shape[2:] == (1, 2, 3)  # (16, 32, 48) / 16: not 32-tileable
        for mode in MODES:
            out = g.decode_latents(enc.z, enc, spatial_mode=mode)
            assert out.shape == state.shape
            assert torch.isfinite(out).all()
            assert torch.count_nonzero(out * (1 - geom.unsqueeze(1))) == 0


def _moving_generator(tmp_path: Path) -> TadpoleLatentGenerator:
    ae_dir = _make_ae_export(tmp_path, "local", "fold")
    m = _generator(ae_dir).eval()
    _install_nontrivial_latent_stats(m)
    m.set_normalization(None, None, [10.0, 10.0], [3.0, 3.0])
    with torch.no_grad():
        torch.nn.init.normal_(m.velocity_net.seqmodel.out_proj.weight, std=0.05)
        torch.nn.init.normal_(m.velocity_net.param_mlp[-1].weight, std=0.05)
    return m


def test_sample_decode_override_uses_the_same_latents(tmp_path):
    m = _moving_generator(tmp_path)
    _, geom, params_hist = _inputs(b=2)
    noise = torch.randn(2, m.state_latent_dim, *m.latent_grid_for(GRID))
    latents = []
    orig = m.decode_latents

    def spy(z, cond, **kw):  # type: ignore[no-untyped-def]
        latents.append(z.clone())
        return orig(z, cond, **kw)

    m.decode_latents = spy  # type: ignore[method-assign]
    base = m.sample(params_hist, geom, initial_noise=noise, num_steps=2)
    same = m.sample(
        params_hist,
        geom,
        initial_noise=noise,
        num_steps=2,
        decode_spatial_mode="local",
        decode_halo_size=32,
    )
    glob = m.sample(
        params_hist, geom, initial_noise=noise, num_steps=2, decode_spatial_mode="global"
    )
    assert torch.equal(same, base)  # local ignores the halo size
    assert torch.equal(latents[0], latents[2])
    assert not torch.allclose(glob, base)
    with pytest.raises(ValueError, match="halo_size"):
        m.sample(params_hist, geom, decode_spatial_mode="halo", decode_halo_size=-16)


def test_heun_sampler(tmp_path):
    """Heun = Euler predictor + trapezoidal corrector on every interval: 2 NFE
    per step, tau = i*dt then (i+1)*dt, and it matches a manual integration."""
    m = _moving_generator(tmp_path)
    _, geom, params_hist = _inputs(b=2)
    noise = torch.randn(2, m.state_latent_dim, *m.latent_grid_for(GRID))
    taus: list[torch.Tensor] = []
    hook = m.velocity_net.register_forward_pre_hook(lambda _, a: taus.append(1))
    euler = m.sample(params_hist, geom, initial_noise=noise, num_steps=3)
    assert len(taus) == 3
    taus.clear()
    heun = m.sample(params_hist, geom, initial_noise=noise, num_steps=3, sampler="heun")
    assert len(taus) == 3 * 2
    hook.remove()
    assert not torch.allclose(heun, euler)
    assert m.sample(
        params_hist, geom, initial_noise=noise, num_steps=3, sampler="euler"
    ).equal(euler)

    cond = m.geometry_condition(geom, batch_size=2)
    z, dt = noise.clone(), 1.0 / 3
    with torch.no_grad():
        for i in range(3):
            t0, t1 = torch.full((2,), i * dt), torch.full((2,), (i + 1) * dt)
            v0 = m.velocity(z, t0, params_hist, cond)
            v1 = m.velocity(z + dt * v0, t1, params_hist, cond)
            z = z + 0.5 * dt * (v0 + v1)
        manual = m.decode_latents(z, cond)
    torch.testing.assert_close(heun, manual, atol=1e-5, rtol=1e-5)
    with pytest.raises(ValueError, match="sampler"):
        m.sample(params_hist, geom, initial_noise=noise, sampler="rk4")


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
    # a node without the key (every pre-adaln artifact) rebuilds the FiLM net
    assert fresh.velocity_conditioning == "input_film"
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


# --------------------------------------------------------------------------- #
# 7. adaLN-Zero velocity conditioning (velocity_conditioning="adaln").
# --------------------------------------------------------------------------- #


def test_default_velocity_net_is_the_unchanged_film_subnetwork(tmp_path):
    from neural_surrogates.architectures.tadpole_stepper import (
        ParamConditionedSubnetwork,
    )

    ae_dir = _make_ae_export(tmp_path, "local", "branch")
    default = _generator(ae_dir)
    assert default.velocity_conditioning == "input_film"
    assert isinstance(default.velocity_net, ParamConditionedSubnetwork)
    explicit = _generator(ae_dir, velocity_conditioning="input_film")
    # weights trained before the option existed still load strictly
    explicit.load_state_dict(default.state_dict(), strict=True)
    prefixes = {k.split(".")[1] for k in default.state_dict() if k.startswith("vel")}
    assert prefixes == {"seqmodel", "param_mlp", "geom_film"}
    with pytest.raises(ValueError, match="velocity_conditioning"):
        _generator(ae_dir, velocity_conditioning="cross_attention")


@pytest.mark.parametrize("geometry", GEOMS)
def test_adaln_zero_at_init_trains_output_projection_then_modulation(
    tmp_path, geometry
):
    from neural_surrogates.architectures.latent_velocity_adaln import AdaLNVelocityNet

    ae_dir = _make_ae_export(tmp_path, "local", geometry)
    m = _generator(ae_dir, velocity_conditioning="adaln", n_layers=2)
    assert isinstance(m.velocity_net, AdaLNVelocityNet)
    assert "velocity_conditioning='adaln'" in repr(m)
    assert m.count_parameters() == sum(p.numel() for p in m.velocity_net.parameters())
    _install_nontrivial_latent_stats(m)
    m.set_normalization(None, None, [10.0, 10.0], [3.0, 3.0])
    m.train()
    state, geom, params_hist = _inputs(b=2)
    feats = _sdf_feats(m, geom)
    net = m.velocity_net
    modulations = [b.modulation[-1] for b in net.blocks] + [net.out_modulation[-1]]
    opt = torch.optim.AdamW([p for p in m.parameters() if p.requires_grad], lr=1e-3)

    v_pred, v_target = m(state, params_hist, geom, feats)
    assert torch.count_nonzero(v_pred) == 0  # every block is the identity
    torch.nn.functional.mse_loss(v_pred, v_target).backward()
    assert torch.count_nonzero(net.out_proj.weight.grad) > 0
    for layer in modulations:  # blocked by the zero output projection
        assert torch.count_nonzero(layer.weight.grad) == 0
    assert torch.count_nonzero(net.geom_proj.weight.grad) == 0
    opt.step()
    opt.zero_grad()

    v_pred, v_target = m(state, params_hist, geom, feats)
    torch.nn.functional.mse_loss(v_pred, v_target).backward()
    for layer in modulations:
        assert torch.count_nonzero(layer.weight.grad) > 0
    assert torch.count_nonzero(net.geom_proj.weight.grad) > 0
    # the embedding c reaches the loss only through the (still zero) modulation
    assert torch.count_nonzero(net.cond_mlp[0].weight.grad) == 0
    opt.step()
    opt.zero_grad()

    v_pred, v_target = m(state, params_hist, geom, feats)
    torch.nn.functional.mse_loss(v_pred, v_target).backward()
    assert torch.count_nonzero(net.cond_mlp[0].weight.grad) > 0


def test_adaln_tau_changes_the_velocity_after_training_steps(tmp_path):
    ae_dir = _make_ae_export(tmp_path, "local", "fold")
    m = _generator(ae_dir, velocity_conditioning="adaln")
    _install_nontrivial_latent_stats(m)
    m.set_normalization(None, None, [10.0, 10.0], [3.0, 3.0])
    state, geom, params_hist = _inputs(b=2)
    cond = m.encode_latents(state, geom, _sdf_feats(m, geom))
    z = torch.randn_like(cond.z)

    def v(tau: float) -> torch.Tensor:
        return m.velocity(z, torch.full((2,), tau), params_hist, cond)

    opt = torch.optim.AdamW([p for p in m.parameters() if p.requires_grad], lr=1e-2)
    m.train()
    for _ in range(3):
        v_pred, v_target = m(state, params_hist, geom, _sdf_feats(m, geom))
        torch.nn.functional.mse_loss(v_pred, v_target).backward()
        opt.step()
        opt.zero_grad()
    m.eval()
    with torch.no_grad():
        assert not torch.allclose(v(0.0), v(1.0), atol=1e-6)
        assert not torch.allclose(v(0.1), v(0.2), atol=1e-6)


def test_adaln_checkpointing_keeps_outputs_and_gradients(tmp_path):
    ae_dir = _make_ae_export(tmp_path, "local", "branch")
    plain = _generator(ae_dir, velocity_conditioning="adaln", n_layers=2)
    ckpt = _generator(
        ae_dir, velocity_conditioning="adaln", n_layers=2, use_checkpoint=True
    )
    with torch.no_grad():
        for p in plain.velocity_net.parameters():
            p.normal_(std=0.05)
    ckpt.load_state_dict(plain.state_dict(), strict=True)
    z = torch.randn(2, plain.state_latent_dim, 1, 1, 2)
    cond = torch.randn(2, plain.cond_dim)
    geom_cond = torch.randn(2, plain.geom_cond_dim, 1, 1, 2)
    outs = []
    for m in (plain, ckpt):
        out = m.velocity_net(z, cond, geom_cond=geom_cond)
        out.square().sum().backward()
        outs.append(out)
    torch.testing.assert_close(outs[0], outs[1])
    for (name, a), b in zip(
        plain.velocity_net.named_parameters(), ckpt.velocity_net.parameters()
    ):
        torch.testing.assert_close(a.grad, b.grad, msg=name)


def test_adaln_self_contained_reload_reproduces_samples(tmp_path):
    ae_dir = _make_ae_export(tmp_path, "halo", "fold")
    m = _generator(ae_dir, velocity_conditioning="adaln", num_sampling_steps=2).eval()
    _install_nontrivial_latent_stats(m)
    m.set_normalization(None, None, [10.0, 12.0], [3.0, 4.0])
    with torch.no_grad():  # move the zero-init layers so sampling is non-trivial
        for p in m.velocity_net.parameters():
            if torch.count_nonzero(p) == 0:
                p.normal_(std=0.05)
    _, geom, params_hist = _inputs(b=2, grid=RECT_GRID)
    noise = torch.randn(2, m.state_latent_dim, *m.latent_grid_for(RECT_GRID))
    ref = m.sample(params_hist, geom, initial_noise=noise)
    torch.save(m.state_dict(), tmp_path / "weights.pt")

    node = {
        "_target_": "neural_surrogates.TadpoleLatentGenerator",
        "param_history_steps": HP,
        "skip_pretrained_load": True,
        "pretrained_ae_dir": None,
        "ae_kwargs": m.ae_kwargs,
        "num_sampling_steps": 2,
        "velocity_conditioning": "adaln",
        **NET,
    }
    cfg = OmegaConf.create(OmegaConf.to_yaml(OmegaConf.create(node)))
    fresh = instantiate(cfg, n_state_channels=C, n_params=P)
    assert fresh.velocity_conditioning == "adaln"
    fresh.load_state_dict(torch.load(tmp_path / "weights.pt"), strict=True)
    out = fresh.eval().sample(params_hist, geom, initial_noise=noise)
    torch.testing.assert_close(out, ref, atol=1e-5, rtol=1e-5)
