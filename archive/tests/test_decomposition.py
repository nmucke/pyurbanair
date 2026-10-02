"""Unit tests for :class:`DomainDecomposition` (pure-torch DD operators).

Covers the partition-of-unity correctness (the #1 risk): ``Σ_i w_i ≡ 1`` on
several shapes incl. non-multiples of the interior size, a non-trivial blend
across a seam, the restrict∘extend_merge round-trip of an arbitrary field, the
positional encoding distinguishing edge from interior patches, the
``any_fluid`` coarse-geometry rule, and per-shape plan caching.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from neural_surrogates import DomainDecomposition

# A few grids: multiples of n, and deliberately not multiples of n.
GRIDS = [(8, 16, 16), (8, 8, 8), (10, 20, 13), (5, 11, 9), (7, 9, 17)]


def _dd(**kw):
    base = dict(
        interior_size=8, halo=3, taper=2, coarsen_factor=2, n_pos=3
    )
    base.update(kw)
    return DomainDecomposition(**base)


@pytest.mark.parametrize("grid", GRIDS)
def test_partition_of_unity_sums_to_one(grid):
    """Merging constant-1 extended blocks reproduces 1 everywhere (Σw≡1)."""
    dd = _dd()
    ref = torch.zeros(1, 1, *grid)
    blocks = dd.restrict(ref)
    ones = torch.ones_like(blocks)
    merged = dd.extend_merge(ones, ref)
    assert merged.shape == ref.shape
    assert torch.allclose(merged, torch.ones_like(merged), atol=1e-6)


@pytest.mark.parametrize("grid", GRIDS)
def test_field_round_trips(grid):
    """restrict then extend_merge of an *arbitrary* field reproduces it exactly.

    A constant field round-trips trivially; a random field is the stronger
    pin because it would expose a wrong halo-crop offset (``lo = h - t``) or a
    mis-aligned scatter index -- every interior cell is covered by patches that
    all read the *same* global value there, so the normalised blend is exact.
    """
    dd = _dd()
    g = torch.Generator().manual_seed(0)
    field = torch.randn(2, 3, *grid, generator=g)
    merged = dd.extend_merge(dd.restrict(field), field)
    assert merged.shape == field.shape
    assert torch.allclose(merged, field, atol=1e-5)


def test_window_is_strict_blend_across_seam():
    """Across a seam the window is strictly in (0, 1): a non-trivial blend
    (not a hard disjoint switch)."""
    dd = _dd(interior_size=8, halo=4, taper=3)
    grid = (8, 16, 16)  # two patches along y -> a seam at y=8
    ref = torch.zeros(1, 1, *grid)
    blocks = dd.restrict(ref)
    # The taper ramp must be strictly inside (0, 1) and rise monotonically.
    win = dd.plan(ref).window
    ramp = win[0, 0, win.shape[2] // 2, win.shape[3] // 2, : dd.taper]
    assert (ramp > 0).all() and (ramp < 1).all()
    assert torch.all(ramp[1:] > ramp[:-1])  # strictly increasing
    # A single patch's normalised contribution is a genuine blend: weight only
    # patch 0 and check the overlapped seam cells fall strictly in (0, 1)
    # because the neighbouring (value-0) patch shares the overlap.
    m = dd.plan(ref).num_patches
    one_patch = torch.zeros(m, 1, *blocks.shape[2:])
    one_patch[0] = 1.0
    merged = dd.extend_merge(one_patch, ref)
    vals = merged[merged > 0]
    assert (vals <= 1.0 + 1e-6).all()
    assert (vals < 1.0 - 1e-6).any()  # some seam cells are a strict blend


def test_positional_edge_differs_from_interior():
    """Positional encoding distinguishes an edge patch from an interior one."""
    dd = _dd(interior_size=8, halo=3, n_pos=3)
    grid = (8, 24, 24)  # cx=cy=3 -> a true interior patch exists along y,x
    ref = torch.zeros(1, 1, *grid)
    pos = dd.positional(ref)  # (M, n_pos, e, e, e)
    plan = dd.plan(ref)
    cz, cy, cx = plan.counts

    def flat(pz, py, px):
        return (pz * cy + py) * cx + px

    edge = pos[flat(0, 0, 0)]
    interior = pos[flat(0, 1, 1)]
    assert not torch.allclose(edge, interior)
    # the central (interior) patch's y/x encodings straddle zero, the corner
    # patch's are biased negative (near the low boundary).
    assert interior.mean() > edge.mean()


def test_positional_respects_n_pos():
    dd = _dd(n_pos=2)
    ref = torch.zeros(1, 1, 8, 16, 16)
    assert dd.positional(ref).shape[1] == 2


def test_any_fluid_coarse_geometry():
    """any_fluid keeps a coarse cell fluid if any fine cell is fluid."""
    dd = _dd(coarsen_factor=2, geometry_coarsen="any_fluid")
    m = torch.zeros(1, 1, 4, 4, 4)
    m[0, 0, 0, 0, 0] = 1.0  # one fluid cell in the (0,0,0) 2^3 window
    coarse = dd.restrict_coarse_geom(m)
    assert coarse.shape == (1, 1, 2, 2, 2)
    assert coarse[0, 0, 0, 0, 0] == 1.0  # window had a fluid cell
    assert coarse[0, 0, 1, 1, 1] == 0.0  # window all solid


def test_all_fluid_coarse_geometry():
    dd = _dd(coarsen_factor=2, geometry_coarsen="all_fluid")
    m = torch.zeros(1, 1, 4, 4, 4)
    m[0, 0, 0, 0, 0] = 1.0  # only one of eight cells fluid
    coarse = dd.restrict_coarse_geom(m)
    assert coarse[0, 0, 0, 0, 0] == 0.0  # not ALL fluid
    m[0, 0, :2, :2, :2] = 1.0  # fill the whole window
    coarse = dd.restrict_coarse_geom(m)
    assert coarse[0, 0, 0, 0, 0] == 1.0


def test_plan_cache_reused_per_shape():
    """The plan object is cached and reused for the same shape/device/dtype,
    and rebuilt on a shape change."""
    dd = _dd()
    a = torch.zeros(1, 1, 8, 16, 16)
    p1 = dd.plan(a)
    p2 = dd.plan(torch.zeros(2, 3, 8, 16, 16))  # same spatial shape
    assert p1 is p2  # cache hit
    p3 = dd.plan(torch.zeros(1, 1, 8, 8, 8))  # different shape
    assert p3 is not p1
    # back to the first shape rebuilds (one-entry cache)
    p4 = dd.plan(a)
    assert p4 is not p1


@pytest.mark.parametrize("grid", GRIDS)
def test_restrict_block_shape(grid):
    dd = _dd()
    ref = torch.zeros(2, 3, *grid)
    blocks = dd.restrict(ref)
    e = dd.interior_size + 2 * dd.halo
    m = dd.num_patches(ref)
    assert blocks.shape == (2 * m, 3, e, e, e)


def test_neighbor_indices():
    dd = _dd()
    ref = torch.zeros(1, 1, 8, 16, 16)  # cz=1, cy=2, cx=2
    nb = dd.neighbor_indices(ref)
    assert nb.shape == (dd.num_patches(ref), 6)
    # patch (0,0,0)=index 0: -z,+z boundary (-1); +y and +x have neighbours.
    assert nb[0, 0] == -1 and nb[0, 1] == -1  # z faces on boundary
    assert nb[0, 3] >= 0  # +y neighbour exists
    assert nb[0, 5] >= 0  # +x neighbour exists


def test_periodic_neighbors_wrap():
    dd = _dd(periodic_axes=(False, True, False))
    ref = torch.zeros(1, 1, 8, 16, 16)  # cy=2
    nb = dd.neighbor_indices(ref)
    # along periodic y, the -y face of py=0 wraps to py=1 (not -1)
    assert nb[0, 2] >= 0


# --------------------------------------------------------------------------- #
# Periodic-axis suite. Tensor layout is (B, C, z, y, x); we run y-periodic,
# i.e. periodic_axes=(False, True, False), the user's typical configuration.
# --------------------------------------------------------------------------- #

PERIODIC = (False, True, False)
# y length must be divisible by interior_size (n=8); x/z are free.
PERIODIC_GRIDS = [(8, 16, 24), (8, 24, 13), (5, 8, 9), (10, 32, 17)]


def _pdd(**kw):
    return _dd(periodic_axes=PERIODIC, **kw)


@pytest.mark.parametrize("grid", PERIODIC_GRIDS)
def test_periodic_partition_of_unity_including_seam(grid):
    """Σ_i w_i ≡ 1 everywhere, INCLUDING the cells straddling the y-seam."""
    dd = _pdd()
    ref = torch.zeros(1, 1, *grid)
    ones = torch.ones_like(dd.restrict(ref))
    merged = dd.extend_merge(ones, ref)
    assert merged.shape == ref.shape
    assert torch.allclose(merged, torch.ones_like(merged), atol=1e-6)
    # explicitly the y-boundary neighbourhoods (the wrap seam)
    assert torch.allclose(merged[:, :, :, 0, :], torch.ones_like(merged[:, :, :, 0, :]), atol=1e-6)
    assert torch.allclose(merged[:, :, :, -1, :], torch.ones_like(merged[:, :, :, -1, :]), atol=1e-6)


@pytest.mark.parametrize("grid", PERIODIC_GRIDS)
def test_periodic_smooth_field_no_seam_discontinuity(grid):
    """A field continuous across the y-wrap (sin(2π y/Ny)) round-trips through
    restrict→extend_merge with no seam discontinuity at y=0 / y=Ny-1."""
    dd = _pdd()
    ny = grid[1]
    yy = torch.arange(ny, dtype=torch.float32)
    field = (
        torch.sin(2.0 * torch.pi * yy / ny)
        .view(1, 1, 1, ny, 1)
        .expand(1, 1, grid[0], ny, grid[2])
        .contiguous()
    )
    out = dd.extend_merge(dd.restrict(field), field)
    assert out.shape == field.shape
    # global round-trip is exact (the field is smooth and constant in z/x)
    assert torch.allclose(out, field, atol=1e-5)
    # and specifically no jump at the seam
    assert (out[:, :, :, 0, :] - field[:, :, :, 0, :]).abs().max() < 1e-5
    assert (out[:, :, :, -1, :] - field[:, :, :, -1, :]).abs().max() < 1e-5


def test_periodic_halo_reads_wrapped_opposite_side():
    """A y-boundary tile's halo cells equal the wrapped opposite-side cells."""
    dd = _pdd(interior_size=8, halo=3)
    grid = (8, 16, 16)
    # a y-varying field so wrapped cells are identifiable
    yy = torch.arange(grid[1], dtype=torch.float32)
    field = (
        yy.view(1, 1, 1, grid[1], 1)
        .expand(1, 1, grid[0], grid[1], grid[2])
        .contiguous()
    )
    blocks = dd.restrict(field)  # (M, 1, e, e, e), B=1
    h = dd.halo
    # patch (pz=0, py=0, px=0) is index 0. Its extended block covers global y in
    # [-h, n+h); the low-halo cell y_local=0 -> global y=-h -> wraps to Ny-h.
    b0 = blocks[0, 0]  # (e, e, e)
    zc, xc = 3, 3  # any interior z, x
    assert torch.allclose(b0[zc, 0, xc], field[0, 0, zc, grid[1] - h, xc])
    # the high-halo of the LAST y-tile wraps to y=0.
    plan = dd.plan(field)
    cz, cy, cx = plan.counts
    last = blocks[(cy - 1) * cx]  # patch (0, cy-1, 0)
    n = dd.interior_size
    # y_local = n + 2h - 1 -> global y = (cy-1)*n + (n+h-1) = Ny-1+h -> wraps h-1
    assert torch.allclose(last[0, zc, -1, xc], field[0, 0, zc, h - 1, xc])


def test_periodic_positional_no_boundary_spike():
    """The periodic-axis positional channel is continuous across the seam and
    stays bounded (no wall spike), while a non-periodic axis still ramps to a
    boundary value beyond that bound."""
    dd = _pdd(interior_size=8, halo=3, taper=2, n_pos=3)
    grid = (8, 24, 24)
    ref = torch.zeros(1, 1, *grid)
    pos = dd.positional(ref)  # (M, 3, e, e, e): ch0=z, ch1=y(periodic), ch2=x
    plan = dd.plan(ref)
    cz, cy, cx = plan.counts
    h, n = dd.halo, dd.interior_size

    def flat(pz, py, px):
        return (pz * cy + py) * cx + px

    # y channel (periodic) is bounded by the sin range [-1, 1]: no ±(>1) wall.
    assert pos[:, 1].abs().max() <= 1.0 + 1e-6
    # and is continuous across the wrap: the last y-tile's high halo (wraps to
    # y=0..) equals the first y-tile's interior-start column.
    last_y = pos[flat(0, cy - 1, 1)][1]
    first_y = pos[flat(0, 0, 1)][1]
    seam_diff = (last_y[:, n + h : n + 2 * h, :] - first_y[:, h : 2 * h, :]).abs().max()
    assert seam_diff < 1e-5

    # x channel (non-periodic) DOES reach a boundary value beyond the sin bound:
    # the extreme x-tiles' signed-distance encoding exceeds 1 in magnitude.
    x_first = pos[flat(0, 1, 0)][2]
    x_last = pos[flat(0, 1, cx - 1)][2]
    assert x_first.min() < -1.0
    assert x_last.max() > 1.0


def test_periodic_axis_indivisible_raises():
    """interior_size must divide a periodic axis length exactly."""
    dd = DomainDecomposition(
        interior_size=16, halo=4, taper=2, periodic_axes=(False, True, False)
    )
    with pytest.raises(ValueError, match="periodic axis 'y' length 50"):
        dd.restrict(torch.zeros(1, 1, 8, 50, 32))
    # a divisible length is fine on the same axis.
    out = dd.restrict(torch.zeros(1, 1, 8, 48, 32))
    assert out.shape[0] == dd.num_patches(torch.zeros(1, 1, 8, 48, 32))


def test_periodic_axis_no_padding_applied():
    """A periodic axis is never padded (its ring length is preserved)."""
    dd = _pdd()
    ref = torch.zeros(1, 1, 8, 24, 13)  # y=24 divisible by n=8; x=13 not
    plan = dd.plan(ref)
    assert plan.pad[1] == 0  # periodic y: no pad
    assert plan.padded[1] == 24
    assert plan.pad[2] > 0  # non-periodic x: padded to multiple of n
