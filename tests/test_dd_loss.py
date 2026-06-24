"""Tests for the Eq (9) domain-decomposition loss and :class:`PatchTrainer`.

Covers each loss term (non-negative, finite, differentiable), the interface
term's zero-when-agreeing / positive-when-disagreeing behaviour incl. periodic
wrap, the divergence term's zero on a divergence-free field, the coarse term's
zero at the restriction target, and an overfit smoke test that the
:class:`PatchTrainer` drives the total loss down on a tiny synthetic
full-field ``TransitionDataset``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
xr = pytest.importorskip("xarray")

from neural_surrogates import (
    DomainDecomposed,
    DomainDecompositionLoss,
    PatchTrainer,
    TransitionDataset,
)
from neural_surrogates.decomposition import DomainDecomposition

N_STATE = 3
N_PARAMS = 2

# Tiny y-periodic decomposition: interior 4, halo 2, taper 1, coarsen 2.
DECOMP = dict(
    interior_size=4,
    halo=2,
    taper=1,
    coarsen_factor=2,
    n_pos=3,
    periodic_axes=(False, True, False),
)
# Sub-nets are Hydra `_target_` config nodes (any registered architecture); the
# wrapper instantiates them and injects channel counts / extra_in_channels.
_UNET = "neural_surrogates.UNetConvNeXt"
FINE = dict(_target_=_UNET, base_channels=4, channel_mults=[1, 2], depths=[1, 1], residual=True)
COARSE = dict(_target_=_UNET, base_channels=4, channel_mults=[1, 2], depths=[1, 1], residual=True)

# y length divisible by interior_size (8 % 4 == 0); multiple patches per axis.
GRID = (8, 8, 8)


def _model(seed: int = 0, **kw) -> DomainDecomposed:
    torch.manual_seed(seed)
    return DomainDecomposed(
        N_STATE,
        N_PARAMS,
        decomposition=dict(DECOMP),
        fine_net=dict(FINE),
        coarse_net=dict(COARSE),
        **kw,
    )


def _inputs(grid=GRID, b=2, seed=0):
    g = torch.Generator().manual_seed(seed)
    state = torch.randn(b, N_STATE, *grid, generator=g)
    params = torch.randn(b, N_PARAMS, generator=g)
    geometry = (torch.rand(b, 1, *grid, generator=g) > 0.2).float()
    return state, params, geometry


# --------------------------------------------------------------------------- #
# Per-term: non-negative, finite, differentiable.
# --------------------------------------------------------------------------- #
def test_terms_nonneg_finite_and_grad_flows():
    model = _model().train()
    state, params, geometry = _inputs()
    target = torch.randn_like(state)

    state_next, info = model(state, params, geometry, return_intermediates=True)
    loss_fn = DomainDecompositionLoss()
    total, terms = loss_fn(
        info=info,
        state_next=state_next,
        target_next=target,
        geometry=geometry,
        dd=model.dd,
    )

    for name, val in terms.items():
        assert torch.isfinite(val), f"{name} not finite"
        assert float(val) >= 0.0, f"{name} negative"

    total.backward()
    assert any(
        p.grad is not None and p.grad.abs().sum() > 0
        for p in model.fine_net.parameters()
    ), "no grad reached fine net"
    assert any(
        p.grad is not None and p.grad.abs().sum() > 0
        for p in model.coarse_net.parameters()
    ), "no grad reached coarse net"


def test_total_is_weighted_sum():
    model = _model().eval()
    state, params, geometry = _inputs()
    target = torch.randn_like(state)
    with torch.no_grad():
        state_next, info = model(
            state, params, geometry, return_intermediates=True
        )
    loss_fn = DomainDecompositionLoss(
        lambda_interface=0.1, lambda_divergence=0.01, lambda_coarse=1.0
    )
    total, terms = loss_fn(
        info=info,
        state_next=state_next,
        target_next=target,
        geometry=geometry,
        dd=model.dd,
    )
    expected = (
        terms["one_step"]
        + 0.1 * terms["interface"]
        + 0.01 * terms["divergence"]
        + 1.0 * terms["coarse"]
    )
    assert torch.allclose(terms["total"], expected, atol=1e-6)


# --------------------------------------------------------------------------- #
# Interface term: 0 when patches agree, > 0 when they disagree (+ periodic wrap)
# --------------------------------------------------------------------------- #
def _patch_blocks(dd: DomainDecomposition, grid, b=1, fill="const"):
    probe = torch.zeros(b, 1, *grid)
    m = dd.num_patches(probe)
    blk = dd.interior_size + 2 * dd.taper
    if fill == "const":
        blocks = torch.full((b * m, N_STATE, blk, blk, blk), 0.7)
    else:
        g = torch.Generator().manual_seed(7)
        blocks = torch.randn(b * m, N_STATE, blk, blk, blk, generator=g)
    return blocks, m


def test_interface_zero_when_constant():
    dd = DomainDecomposition(**DECOMP)
    # populate the plan cache for GRID via any operator
    dd.plan(torch.zeros(1, 1, *GRID))
    blocks, m = _patch_blocks(dd, GRID, fill="const")
    info = {"patch_pred": blocks, "num_patches": m, "dd": dd}
    loss_fn = DomainDecompositionLoss()
    val = loss_fn._interface_term(info, dd)
    assert float(val) == pytest.approx(0.0, abs=1e-7)


def test_interface_positive_when_disagree():
    dd = DomainDecomposition(**DECOMP)
    dd.plan(torch.zeros(1, 1, *GRID))
    blocks, m = _patch_blocks(dd, GRID, fill="rand")
    info = {"patch_pred": blocks, "num_patches": m, "dd": dd}
    loss_fn = DomainDecompositionLoss()
    val = loss_fn._interface_term(info, dd)
    assert float(val) > 0.0


def test_interface_zero_for_globally_consistent_patches():
    """The load-bearing semantic: when every patch's block is the restriction of
    one shared global field, adjacent patches read *identical* cells on their
    overlap band, so the interface disagreement is exactly zero.

    This pins the band-matching geometry (high ``2t`` band of ``i`` vs low band
    of its ``+`` neighbour ``j``): a wrong pairing (e.g. low-i vs low-j) would
    compare *different* global cells of the spatially varying field and give a
    clearly nonzero value. The field is non-constant on every axis (so wrong
    bands disagree) and periodic in y (so the seam bands stay consistent)."""
    dd = DomainDecomposition(**DECOMP)
    nz, ny, nx = GRID
    z = torch.arange(nz, dtype=torch.float32).view(1, 1, nz, 1, 1)
    y = torch.arange(ny, dtype=torch.float32).view(1, 1, 1, ny, 1)
    x = torch.arange(nx, dtype=torch.float32).view(1, 1, 1, 1, nx)
    # smooth, varies on all axes; periodic in y via sin(2π y / Ny).
    field = 0.3 * z + torch.sin(2 * torch.pi * y / ny) + 0.2 * x
    field = field.expand(1, N_STATE, nz, ny, nx).contiguous()

    # patch_pred is the extended (n+2h) restriction cropped to the (n+2t) PoU
    # footprint -- exactly what DomainDecomposed builds internally.
    ext = dd.restrict(field)
    lo = dd.halo - dd.taper
    blk = dd.interior_size + 2 * dd.taper
    patch_pred = ext[:, :, lo : lo + blk, lo : lo + blk, lo : lo + blk].contiguous()

    info = {"patch_pred": patch_pred, "num_patches": dd.num_patches(field), "dd": dd}
    loss_fn = DomainDecompositionLoss()
    val = loss_fn._interface_term(info, dd)
    assert float(val) == pytest.approx(0.0, abs=1e-5)


def test_interface_counts_periodic_wrap_faces():
    """With y periodic, every y-face has a (wrapped) neighbour, so the y axis
    contributes interface terms even at the grid edge; the term is finite and
    differentiable through the wrapped neighbour block."""
    dd = DomainDecomposition(**DECOMP)
    dd.plan(torch.zeros(1, 1, *GRID))
    blk = dd.interior_size + 2 * dd.taper
    m = dd.num_patches(torch.zeros(1, 1, *GRID))
    blocks = torch.randn(m, N_STATE, blk, blk, blk, requires_grad=True)
    info = {"patch_pred": blocks, "num_patches": m, "dd": dd}
    loss_fn = DomainDecompositionLoss()
    val = loss_fn._interface_term(info, dd)
    assert torch.isfinite(val) and float(val.detach()) > 0
    val.backward()
    # gradient must reach the patches that only couple via the periodic wrap.
    assert blocks.grad is not None and blocks.grad.abs().sum() > 0

    # Sanity: y is periodic so every patch has both +y and -y neighbours; the
    # neighbour table should have no -1 on the y slots.
    neighbors = dd.neighbor_indices(torch.zeros(1, 1, *GRID))
    assert (neighbors[:, 2] >= 0).all()  # -y
    assert (neighbors[:, 3] >= 0).all()  # +y


# --------------------------------------------------------------------------- #
# Divergence term: 0 for constant velocity, > 0 for a divergent field.
# --------------------------------------------------------------------------- #
def test_divergence_zero_for_constant_velocity():
    loss_fn = DomainDecompositionLoss(mask_loss=False)
    state = torch.zeros(1, N_STATE, *GRID)
    state[:, 0] = 2.0  # u
    state[:, 1] = -1.0  # v
    state[:, 2] = 0.5  # w
    fluid = torch.ones(1, 1, *GRID, dtype=torch.bool)
    val = loss_fn._divergence_term(state, fluid)
    assert float(val) == pytest.approx(0.0, abs=1e-7)


def test_divergence_matches_known_value_for_divergent_field():
    """``u = x`` (x-velocity linear in x) has physical divergence
    ``du/dx = 1`` everywhere, so the squared-divergence term is exactly 1.0.

    Channels are ``(u, v, w) = (x-vel, y-vel, z-vel)``; ``u`` is channel 0 and
    must be differenced along x (dim 4)."""
    loss_fn = DomainDecompositionLoss(mask_loss=False)
    nz, ny, nx = GRID
    xs = torch.arange(nx, dtype=torch.float32).view(1, 1, 1, 1, nx)
    state = torch.zeros(1, N_STATE, nz, ny, nx)
    state[:, 0] = xs  # u (x-velocity) linear in x -> du/dx = 1
    fluid = torch.ones(1, 1, *GRID, dtype=torch.bool)
    val = loss_fn._divergence_term(state, fluid)
    assert float(val) == pytest.approx(1.0, abs=1e-6)


def test_divergence_zero_for_shear_flow():
    """A shear flow ``u = u(z)`` (x-velocity varying along z, v=w=0) is
    divergence-free: ``du/dx = dv/dy = dw/dz = 0``. This discriminates the
    channel<->axis mapping -- the old transposed code differenced channel 0
    along z and would report a *nonzero* divergence for this field."""
    loss_fn = DomainDecompositionLoss(mask_loss=False)
    nz, ny, nx = GRID
    zs = torch.arange(nz, dtype=torch.float32).view(1, 1, nz, 1, 1)
    state = torch.zeros(1, N_STATE, nz, ny, nx)
    state[:, 0] = zs  # u (x-velocity) varies along z -> du/dx still 0
    fluid = torch.ones(1, 1, *GRID, dtype=torch.bool)
    val = loss_fn._divergence_term(state, fluid)
    assert float(val) == pytest.approx(0.0, abs=1e-7)


# --------------------------------------------------------------------------- #
# Coarse term: 0 at the restriction target.
# --------------------------------------------------------------------------- #
def test_coarse_zero_at_restriction_target():
    dd = DomainDecomposition(**DECOMP)
    target = torch.randn(2, N_STATE, *GRID)
    coarse_target = dd.restrict_coarse(target)
    info = {"coarse_pred": coarse_target.clone()}
    loss_fn = DomainDecompositionLoss()
    val = loss_fn._coarse_term(info, target, dd)
    assert float(val) == pytest.approx(0.0, abs=1e-7)


def test_coarse_equals_manual_mse():
    """The coarse term is exactly MSE(coarse_pred, restrict_coarse(target)) for
    an arbitrary (independent) coarse_pred -- pins the formula, not just that
    ``a - a == 0``."""
    dd = DomainDecomposition(**DECOMP)
    target = torch.randn(2, N_STATE, *GRID)
    g = torch.Generator().manual_seed(3)
    coarse_pred = torch.randn(
        2, N_STATE, *dd.restrict_coarse(target).shape[2:], generator=g
    )
    info = {"coarse_pred": coarse_pred}
    loss_fn = DomainDecompositionLoss()
    val = loss_fn._coarse_term(info, target, dd)
    expected = (coarse_pred - dd.restrict_coarse(target)).pow(2).mean()
    assert torch.allclose(val, expected, atol=1e-7)


def test_coarse_positive_when_mismatched():
    dd = DomainDecomposition(**DECOMP)
    target = torch.randn(2, N_STATE, *GRID)
    coarse_target = dd.restrict_coarse(target)
    info = {"coarse_pred": coarse_target + 1.0}  # constant offset of 1 -> MSE 1
    loss_fn = DomainDecompositionLoss()
    val = loss_fn._coarse_term(info, target, dd)
    assert float(val) == pytest.approx(1.0, abs=1e-6)


# --------------------------------------------------------------------------- #
# Overfit smoke: PatchTrainer drives the total loss down.
# --------------------------------------------------------------------------- #
T_LEN = 4
STATE_VARS = ("u", "v", "w")
PARAM_VARS = ("inflow_angle", "velocity_magnitude")


def _write_traj(state_dir: Path, param_dir: Path, idx: int, seed: int) -> None:
    state_dir.mkdir(parents=True, exist_ok=True)
    param_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    dims = ("time", "z", "y", "x")
    shape = (T_LEN, *GRID)
    data_vars = {
        v: (dims, rng.standard_normal(shape).astype(np.float32))
        for v in STATE_VARS
    }
    obstacle = np.zeros(GRID, dtype=np.float32)
    obstacle[0:2, 2:4, 2:4] = 1.0
    data_vars["blanking"] = (
        dims,
        np.broadcast_to(obstacle, shape).copy(),
    )
    xr.Dataset(data_vars).to_netcdf(state_dir / f"sample_{idx:04d}.nc")
    pvars = {
        "inflow_angle": (("time",), rng.uniform(-60, 60, T_LEN).astype(np.float32)),
        "velocity_magnitude": (
            ("time",),
            rng.uniform(1, 5, T_LEN).astype(np.float32),
        ),
    }
    xr.Dataset(pvars).to_netcdf(param_dir / f"sample_{idx:04d}.nc")


@pytest.fixture
def field_root(tmp_path: Path) -> Path:
    root = tmp_path / "fields"
    for idx in range(2):
        _write_traj(
            root / "state" / "train",
            root / "param" / "train",
            idx,
            seed=10 + idx,
        )
    return root


def test_patch_trainer_overfits(field_root: Path):
    from torch.utils.data import DataLoader

    ds = TransitionDataset(
        root_dir=field_root,
        split="train",
        state_vars=STATE_VARS,
        param_vars=PARAM_VARS,
        geometry_var="blanking",
        pushforward_steps=1,
    )
    loader = DataLoader(ds, batch_size=2, shuffle=False)

    model = _model(seed=1).train()
    loss_fn = DomainDecompositionLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

    # Measure the initial total loss over the (fixed) dataset.
    def epoch_loss() -> float:
        model.eval()
        total, n = 0.0, 0
        with torch.no_grad():
            for batch in loader:
                geom = batch["geometry"][0]
                geom_b = geom.expand(batch["state_n"].shape[0], *geom.shape)
                sn, info = model(
                    batch["state_n"],
                    batch["params_n"][:, 0, :],
                    geom_b,
                    return_intermediates=True,
                )
                loss, _ = loss_fn(
                    info=info,
                    state_next=sn,
                    target_next=batch["state_next"],
                    geometry=geom_b,
                    dd=model.dd,
                )
                total += float(loss)
                n += 1
        model.train()
        return total / max(n, 1)

    initial = epoch_loss()

    trainer = PatchTrainer(
        model=model,
        train_loader=loader,
        val_loader=loader,
        optimizer=optimizer,
        loss_fn=loss_fn,
        num_epochs=20,
        device="cpu",
        weights_path=None,
    )
    trainer.fit()

    final = epoch_loss()
    assert final < initial, f"loss did not decrease: {initial} -> {final}"
