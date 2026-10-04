"""Unit tests for :class:`DomainDecomposed` (the DD surrogate wrapper).

Covers the full-grid ``forward(state, params, geometry) -> state_next``
contract: output shape equals input shape on several grids (incl. larger than
one patch and not a multiple of the interior size), grid-flexibility (one
instance runs two grids), eval-mode determinism, obstacle cells exactly zero,
and gradients reaching both the fine and the coarse net.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from neural_surrogates import DomainDecomposed

N_STATE = 3
N_PARAMS = 2

# Sub-nets are Hydra `_target_` config nodes (any registered architecture); the
# wrapper instantiates them and injects channel counts / extra_in_channels.
_UNET = "neural_surrogates.UNetConvNeXt"
DECOMP = dict(interior_size=8, halo=3, taper=2, coarsen_factor=2, n_pos=3)
FINE = dict(_target_=_UNET, base_channels=8, channel_mults=[1, 2], depths=[1, 1], residual=True)
COARSE = dict(_target_=_UNET, base_channels=8, channel_mults=[1, 2], depths=[1, 1], residual=True)

# one patch, multiple patches, and a non-multiple-of-n grid.
GRIDS = [(8, 8, 8), (8, 16, 16), (10, 20, 13)]


def _model(**kw):
    return DomainDecomposed(
        N_STATE,
        N_PARAMS,
        decomposition=DECOMP,
        fine_net=FINE,
        coarse_net=COARSE,
        **kw,
    )


def _inputs(grid, b=2, seed=0):
    g = torch.Generator().manual_seed(seed)
    state = torch.randn(b, N_STATE, *grid, generator=g)
    params = torch.randn(b, N_PARAMS, generator=g)
    geometry = (torch.rand(b, 1, *grid, generator=g) > 0.3).float()
    return state, params, geometry


@pytest.mark.parametrize("grid", GRIDS)
def test_forward_shape(grid):
    model = _model().eval()
    state, params, geometry = _inputs(grid)
    out = model(state, params, geometry)
    assert out.shape == state.shape


def test_grid_flexibility_single_instance():
    """One model instance runs two different grids without rebuild errors."""
    model = _model().eval()
    for grid in [(8, 16, 16), (10, 12, 20)]:
        state, params, geometry = _inputs(grid)
        out = model(state, params, geometry)
        assert out.shape == state.shape


def test_geometry_3d_input_accepted():
    """geometry given as (B, *grid) (no channel dim) is handled."""
    model = _model().eval()
    state, params, geometry = _inputs((8, 16, 16))
    out = model(state, params, geometry.squeeze(1))
    assert out.shape == state.shape


def test_eval_determinism():
    model = _model().eval()
    state, params, geometry = _inputs((8, 16, 16))
    with torch.no_grad():
        a = model(state, params, geometry)
        b = model(state, params, geometry)
    assert torch.allclose(a, b)


def test_obstacle_cells_exactly_zero():
    model = _model().eval()
    state, params, geometry = _inputs((8, 16, 16))
    out = model(state, params, geometry)
    solid = geometry.expand_as(out) == 0
    assert torch.count_nonzero(out[solid]) == 0


def test_gradients_reach_both_nets():
    model = _model().train()
    state, params, geometry = _inputs((8, 16, 16))
    out = model(state, params, geometry)
    out.pow(2).mean().backward()

    def has_grad(net):
        return any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in net.parameters()
        )

    assert has_grad(model.fine_net), "no gradient reached the fine net"
    assert has_grad(model.coarse_net), "no gradient reached the coarse net"


def test_domain_flexible_attribute():
    assert DomainDecomposed.domain_flexible is True
    assert _model().domain_flexible is True


def test_divergence_projection_runs_when_enabled():
    model = _model(divergence_projection=True).eval()
    state, params, geometry = _inputs((8, 16, 16))
    out = model(state, params, geometry)
    assert out.shape == state.shape


def test_periodic_y_forward_runs():
    """A y-periodic DD model runs end-to-end (y length divisible by n)."""
    model = DomainDecomposed(
        N_STATE,
        N_PARAMS,
        decomposition={**DECOMP, "periodic_axes": (False, True, False)},
        fine_net=FINE,
        coarse_net=COARSE,
    ).eval()
    grid = (8, 16, 24)  # y=16 divisible by interior_size=8
    state, params, geometry = _inputs(grid)
    out = model(state, params, geometry)
    assert out.shape == state.shape


def test_set_normalization_forwards_to_both():
    model = DomainDecomposed(
        N_STATE,
        N_PARAMS,
        decomposition=DECOMP,
        fine_net={**FINE, "normalize": True},
        coarse_net={**COARSE, "normalize": True},
    )
    sm = torch.zeros(N_STATE)
    ss = torch.ones(N_STATE) * 2.0
    pm = torch.zeros(N_PARAMS)
    ps = torch.ones(N_PARAMS) * 3.0
    model.set_normalization(sm, ss, pm, ps)
    assert torch.allclose(model.fine_net.state_std, ss)
    assert torch.allclose(model.coarse_net.state_std, ss)
    assert torch.allclose(model.fine_net.param_std, ps)
    assert torch.allclose(model.coarse_net.param_std, ps)


# --------------------------------------------------------------------------- #
# Any registered architecture is usable as a sub-net (instantiated via Hydra).
# --------------------------------------------------------------------------- #
def test_mixed_architecture_subnets():
    """A different family (SimpleConv) drives the fine net while a UNet drives
    the coarse net: the wrapper instantiates each via Hydra and injects the
    context channels through SimpleConv's `extra_in_channels` contract."""
    from neural_surrogates import SimpleConv

    model = DomainDecomposed(
        N_STATE,
        N_PARAMS,
        decomposition=DECOMP,
        fine_net=dict(_target_="neural_surrogates.SimpleConv", kernel_size=3),
        coarse_net=COARSE,
    ).eval()
    assert isinstance(model.fine_net, SimpleConv)
    # fine stem widened by context (C) + positional (n_pos) channels.
    assert model.fine_net.extra_in_channels == N_STATE + model.dd.n_pos
    state, params, geometry = _inputs((8, 16, 16))
    with torch.no_grad():
        out = model(state, params, geometry)
    assert out.shape == state.shape
    assert float((out * (1 - geometry)).abs().max()) == 0.0  # obstacles zero


def test_subnet_requires_target():
    """A sub-net config without a `_target_` is a clear error, not a crash."""
    with pytest.raises(ValueError, match="_target_"):
        DomainDecomposed(
            N_STATE, N_PARAMS, decomposition=DECOMP,
            fine_net=dict(base_channels=4), coarse_net=COARSE,
        )


def test_fine_net_without_extra_channels_support_errors():
    """A fine-net architecture that cannot ingest the context channels raises a
    helpful error (it lacks the `extra_in_channels` contract)."""
    with pytest.raises(ValueError, match="extra_in_channels"):
        DomainDecomposed(
            N_STATE, N_PARAMS, decomposition=DECOMP,
            fine_net=dict(_target_="torch.nn.Identity"), coarse_net=COARSE,
        )
