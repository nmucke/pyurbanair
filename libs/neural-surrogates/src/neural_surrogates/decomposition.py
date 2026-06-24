"""Two-level overlapping domain decomposition operators (pure torch).

This module implements the spatial bookkeeping for a domain-decomposed
neural surrogate (companion PDF §2 + §5): it splits a full ``(Nz, Ny, Nx)``
grid into a uniform batch of overlapping patches, runs the per-patch network
on that batch, and stitches the patch outputs back into a single full grid
with a partition-of-unity (PoU) blend.

There is **no I/O here** -- everything operates on torch tensors so the
operators can sit *inside* a model whose external contract is still
``forward(state, params, geometry) -> state_next`` on full grids.

Design (see :class:`DomainDecomposition`):

* **Uniform tiling via padding.** Each spatial axis is padded up to a
  multiple of ``interior_size`` (``n``), then ``halo`` (``h``) cells are added
  on every side (``boundary_mode``, ``'circular'`` per periodic axis). The
  extended blocks are then exactly ``(n + 2h)`` on every axis, so a single
  batched ``forward`` covers every patch and the result crops cleanly back to
  the original grid.
* **PoU windows.** Patch outputs are merged on a slightly larger
  ``(n + 2*taper)`` footprint (stride ``n``, overlap ``2*taper``) carrying a
  separable Hann taper, scatter-added into the padded grid and normalised by
  the overlap-sum so ``Σ_i w_i ≡ 1`` at every grid cell. Strict-``n``
  interiors would be disjoint and give no blend, hence the ``taper`` band.

Three pad/crop bookkeepings (interior→multiple-of-``n``, halo ``h``, and the
``taper`` band of the merge) all round-trip to the **exact** original
``(Nz, Ny, Nx)``.

Buffers (windows, positional encodings) are built lazily per spatial shape and
cached in a small per-shape :class:`_Plan`; they are rebuilt on the correct
device/dtype whenever the shape, device or dtype changes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
import torch.nn.functional as F


def _hann_taper(taper: int, *, device, dtype) -> torch.Tensor:
    """1-D ramp rising over ``taper`` cells, flat ``1`` over the interior, and
    falling over ``taper`` cells, for a window of length ``n + 2*taper``.

    Returns just the rising ramp of length ``taper`` (in ``(0, 1)``); the full
    separable window is assembled from it. With ``taper == 0`` the ramp is
    empty and the window is all-ones.
    """
    if taper <= 0:
        return torch.ones(0, device=device, dtype=dtype)
    # Half a Hann window over 2*taper points, sampled strictly inside (0, 1)
    # so the blend is non-trivial at the seam (never exactly 0 or 1).
    k = torch.arange(1, taper + 1, device=device, dtype=dtype)
    return 0.5 * (1.0 - torch.cos(torch.pi * k / (taper + 1)))


def _window_1d(n: int, taper: int, *, device, dtype) -> torch.Tensor:
    """Separable 1-D PoU factor of length ``n + 2*taper``: rising ramp, flat
    interior, falling ramp. Strictly positive everywhere."""
    ramp = _hann_taper(taper, device=device, dtype=dtype)
    flat = torch.ones(n, device=device, dtype=dtype)
    return torch.cat([ramp, flat, ramp.flip(0)])


@dataclass
class _Plan:
    """Cached per-shape tiling layout. All integers are derived once per
    distinct ``(grid, device, dtype)`` and reused across calls."""

    grid: tuple[int, int, int]
    # interior-padded grid (multiple of n on every axis)
    padded: tuple[int, int, int]
    pad: tuple[int, int, int]  # interior padding added (z, y, x)
    counts: tuple[int, int, int]  # number of patches per axis
    num_patches: int
    coarse: tuple[int, int, int]  # coarse grid after avg_pool3d(r)
    coarse_pad: tuple[int, int, int]  # padding added before coarse pooling
    window: torch.Tensor  # separable PoU window (1, 1, n+2t, n+2t, n+2t)
    positional: torch.Tensor  # (M, n_pos, n+2h, n+2h, n+2h)
    neighbors: torch.Tensor  # (M, 6) flat indices of (-z,+z,-y,+y,-x,+x) nbrs
    # flat destination indices of every merge-window cell into the taper-padded
    # accumulator, for the vectorised overlap-add scatter; (M * (n+2t)**3,).
    merge_index: torch.Tensor


class DomainDecomposition:
    """Two-level overlapping domain decomposition for full-grid surrogates.

    Parameters
    ----------
    interior_size:
        Patch interior edge length ``n`` (the stride of the tiling). The grid
        is padded to a multiple of ``n`` on every axis.
    halo:
        Halo width ``h``: cells added on every side of an interior to form the
        ``(n + 2h)`` extended block the fine net consumes.
    taper:
        PoU overlap half-width ``δ``: the merge footprint is ``(n + 2δ)`` and
        adjacent merge windows overlap by ``2δ`` cells with a Hann taper. Must
        satisfy ``taper <= halo`` so the taper band lies inside the extended
        block.
    coarsen_factor:
        Average-pool factor ``r`` for the coarse level.
    n_pos:
        Number of positional-encoding channels (a signed, normalised axis
        coordinate along each of the three axes, ``-1``..``+1`` low-to-high
        face; ``sin`` of the wrapped angle on a periodic axis; capped at 3).
    boundary_mode:
        ``F.pad`` mode for the halo fill on non-periodic axes (e.g.
        ``'replicate'``, ``'reflect'``, ``'constant'``).
    periodic_axes:
        Per-axis ``(z, y, x)`` periodicity. Periodic axes are filled
        circularly in the halo (global periodicity lives here, never on the
        inner net).
    geometry_coarsen:
        ``'any_fluid'`` (default): a coarse cell is fluid if *any* fine cell in
        its window is fluid, ``(avg_pool3d(m, r) > 0).float()``. ``'all_fluid'``
        requires every fine cell to be fluid.

    Notes
    -----
    The object is stateless apart from a one-entry per-shape cache; call any
    operator with full-grid tensors of shape ``(B, C, Nz, Ny, Nx)`` and the
    plan is built (or reused) for that shape/device/dtype automatically.
    """

    def __init__(
        self,
        interior_size: int = 16,
        halo: int = 4,
        taper: int = 2,
        coarsen_factor: int = 4,
        n_pos: int = 3,
        boundary_mode: str = "replicate",
        periodic_axes: Sequence[bool] = (False, False, False),
        geometry_coarsen: str = "any_fluid",
    ) -> None:
        if interior_size < 1:
            raise ValueError("interior_size must be >= 1")
        if halo < 0:
            raise ValueError("halo must be >= 0")
        if taper < 0:
            raise ValueError("taper must be >= 0")
        if taper > halo:
            raise ValueError(
                f"taper ({taper}) must be <= halo ({halo}) so the PoU band "
                "lies inside the extended block"
            )
        if coarsen_factor < 1:
            raise ValueError("coarsen_factor must be >= 1")
        if geometry_coarsen not in ("any_fluid", "all_fluid"):
            raise ValueError(
                f"geometry_coarsen must be 'any_fluid' or 'all_fluid', "
                f"got {geometry_coarsen!r}"
            )
        periodic = tuple(bool(p) for p in periodic_axes)
        if len(periodic) != 3:
            raise ValueError("periodic_axes must have length 3 (z, y, x)")

        self.interior_size = int(interior_size)
        self.halo = int(halo)
        self.taper = int(taper)
        self.coarsen_factor = int(coarsen_factor)
        self.n_pos = min(int(n_pos), 3)
        self.boundary_mode = boundary_mode
        self.periodic = periodic
        self.geometry_coarsen = geometry_coarsen

        self._plan: _Plan | None = None
        self._cache_key: tuple | None = None

    # ------------------------------------------------------------------ #
    # Plan construction / caching
    # ------------------------------------------------------------------ #
    def _build_plan(
        self, grid: tuple[int, int, int], *, device, dtype
    ) -> _Plan:
        n = self.interior_size
        h = self.halo
        t = self.taper
        r = self.coarsen_factor

        # Periodic axes must NOT be padded: padding changes the ring length and
        # corrupts the wrap. Require interior_size to divide the periodic axis
        # length exactly; non-periodic axes pad up to a multiple of n as usual.
        axis_names = ("z", "y", "x")
        pad_list = []
        for a, g in enumerate(grid):
            if self.periodic[a]:
                if g % n != 0:
                    raise ValueError(
                        f"periodic axis {axis_names[a]!r} length {g} not "
                        f"divisible by interior_size {n}"
                    )
                pad_list.append(0)
            else:
                pad_list.append((n - g % n) % n)
        pad = tuple(pad_list)
        padded = tuple(g + p for g, p in zip(grid, pad))
        counts = tuple(p // n for p in padded)
        num_patches = counts[0] * counts[1] * counts[2]

        # Coarse grid: pad the *original* grid up to a multiple of r, then pool.
        coarse_pad = tuple((r - g % r) % r for g in grid)
        coarse = tuple((g + p) // r for g, p in zip(grid, coarse_pad))

        # Separable PoU window over the (n + 2t) merge footprint.
        w1 = _window_1d(n, t, device=device, dtype=dtype)
        window = (
            w1.view(-1, 1, 1)
            * w1.view(1, -1, 1)
            * w1.view(1, 1, -1)
        ).view(1, 1, n + 2 * t, n + 2 * t, n + 2 * t)

        positional = self._build_positional(
            grid, counts, device=device, dtype=dtype
        )
        neighbors = self._build_neighbors(counts, device=device)
        merge_index = self._build_merge_index(
            counts, padded, device=device
        )

        return _Plan(
            grid=grid,
            padded=padded,
            pad=pad,
            counts=counts,
            num_patches=num_patches,
            coarse=coarse,
            coarse_pad=coarse_pad,
            window=window,
            positional=positional,
            neighbors=neighbors,
            merge_index=merge_index,
        )

    def _build_neighbors(
        self, counts: tuple[int, int, int], *, device
    ) -> torch.Tensor:
        """Patch adjacency table ``(M, 6)``: flat indices of the
        ``(-z, +z, -y, +y, -x, +x)`` neighbours, ``-1`` where a face is on the
        global boundary and that axis is not periodic."""
        cz, cy, cx = counts

        def flat(pz: int, py: int, px: int) -> int:
            return (pz * cy + py) * cx + px

        out = torch.full((cz * cy * cx, 6), -1, dtype=torch.long, device=device)
        for pz in range(cz):
            for py in range(cy):
                for px in range(cx):
                    idx = flat(pz, py, px)
                    coord = (pz, py, px)
                    for axis in range(3):
                        for d, slot in ((-1, 2 * axis), (1, 2 * axis + 1)):
                            nc = list(coord)
                            nc[axis] += d
                            if 0 <= nc[axis] < counts[axis]:
                                out[idx, slot] = flat(*nc)
                            elif self.periodic[axis]:
                                nc[axis] %= counts[axis]
                                out[idx, slot] = flat(*nc)
        return out

    def _build_merge_index(
        self,
        counts: tuple[int, int, int],
        padded: tuple[int, int, int],
        *,
        device,
    ) -> torch.Tensor:
        """Flat destination indices for the vectorised merge scatter-add.

        Patch ``p = (pz, py, px)`` (row-major, ``z`` slowest) places its
        ``(n+2t)`` merge window at origin ``(pz*n, py*n, px*n)`` in the accumulator,
        which is the interior-padded grid grown by ``taper`` on every side. This
        returns, for every patch and every local window cell ``(a, b, c)``, the
        flat index into the ``(acc_z, acc_y, acc_x)`` accumulator volume, ordered
        ``(p, a, b, c)`` to match ``merge_blocks.reshape(B*M, C, ...)``.
        """
        n = self.interior_size
        t = self.taper
        edge = n + 2 * t
        cz, cy, cx = counts
        acc_y = padded[1] + 2 * t
        acc_x = padded[2] + 2 * t
        la = torch.arange(edge, device=device)
        # global accumulator coordinate of each (patch, local-cell) pair per axis
        zc = (torch.arange(cz, device=device) * n).view(cz, 1) + la  # (cz, edge)
        yc = (torch.arange(cy, device=device) * n).view(cy, 1) + la  # (cy, edge)
        xc = (torch.arange(cx, device=device) * n).view(cx, 1) + la  # (cx, edge)
        Z = zc.view(cz, 1, 1, edge, 1, 1)
        Y = yc.view(1, cy, 1, 1, edge, 1)
        X = xc.view(1, 1, cx, 1, 1, edge)
        flat = (Z * acc_y + Y) * acc_x + X  # (cz, cy, cx, edge, edge, edge)
        return flat.reshape(-1).to(torch.long)

    def _build_positional(
        self,
        grid: tuple[int, int, int],
        counts: tuple[int, int, int],
        *,
        device,
        dtype,
    ) -> torch.Tensor:
        """Per-patch positional encoding of shape ``(M, n_pos, n+2h³)``.

        Channel ``a`` encodes the position of each extended-block cell along
        axis ``a``:

        * **Non-periodic axis** -- a signed, normalised axis *coordinate*
          (``-1`` at the low face, ``+1`` at the high face; monotonic, not a
          distance-to-nearest-boundary), so an edge patch differs from an
          interior one. Halo cells beyond the grid extrapolate linearly.
        * **Periodic axis** -- ``sin(2π · wrapped_coord / g)``: a smooth,
          wrap-continuous encoding that does NOT spike at the seam, so a tile
          next to the periodic seam is not told it sits against a wall.

        Halo cells share the periodic tile's wrapped global coordinate, so the
        encoding is continuous across the seam.
        """
        n = self.interior_size
        h = self.halo
        ext = n + 2 * h
        if self.n_pos == 0:
            return torch.zeros(
                counts[0] * counts[1] * counts[2], 0, ext, ext, ext,
                device=device, dtype=dtype,
            )

        # Global coordinate (in fine-cell units) of every cell of every
        # extended block, per axis. Block p starts its interior at p*n, so its
        # extended block starts at p*n - h.
        per_axis = []  # list of (count_a, ext) per-axis encodings
        for a in range(3):
            local = torch.arange(ext, device=device, dtype=dtype) - h
            starts = (
                torch.arange(counts[a], device=device, dtype=dtype) * n
            ).view(-1, 1)
            coords = starts + local.view(1, -1)  # (count_a, ext) global coords
            g = grid[a]
            if self.periodic[a]:
                # Periodic: sin of the wrapped angle -> continuous across the
                # seam, no spurious boundary. (coords may be negative in halos;
                # sin wraps it naturally.)
                enc = torch.sin(2.0 * torch.pi * coords / g)
            else:
                denom = max(g - 1, 1)
                # signed distance to boundary, normalised to ~[-1, 1]
                enc = 2.0 * coords / denom - 1.0
            per_axis.append(enc)

        cz, cy, cx = counts
        # Build (M, n_pos, ext, ext, ext) by broadcasting each axis encoding.
        z = per_axis[0]  # (cz, ext)
        y = per_axis[1]  # (cy, ext)
        x = per_axis[2]  # (cx, ext)

        # patch index ordering: (pz, py, px) flattened row-major (z slowest).
        pz = torch.arange(cz, device=device).view(cz, 1, 1).expand(cz, cy, cx).reshape(-1)
        py = torch.arange(cy, device=device).view(1, cy, 1).expand(cz, cy, cx).reshape(-1)
        px = torch.arange(cx, device=device).view(1, 1, cx).expand(cz, cy, cx).reshape(-1)

        zc = z[pz]  # (M, ext)
        yc = y[py]  # (M, ext)
        xc = x[px]  # (M, ext)

        m = pz.numel()
        chans = []
        if self.n_pos >= 1:
            chans.append(zc.view(m, ext, 1, 1).expand(m, ext, ext, ext))
        if self.n_pos >= 2:
            chans.append(yc.view(m, 1, ext, 1).expand(m, ext, ext, ext))
        if self.n_pos >= 3:
            chans.append(xc.view(m, 1, 1, ext).expand(m, ext, ext, ext))
        return torch.stack(chans, dim=1).contiguous()

    def plan(self, x: torch.Tensor) -> _Plan:
        """Return the cached :class:`_Plan` for ``x``'s spatial shape, building
        (and caching) it on ``x``'s device/dtype if needed."""
        grid = tuple(int(s) for s in x.shape[-3:])
        key = (grid, x.device, x.dtype)
        if self._cache_key != key:
            self._plan = self._build_plan(grid, device=x.device, dtype=x.dtype)
            self._cache_key = key
        assert self._plan is not None
        return self._plan

    def num_patches(self, x: torch.Tensor) -> int:
        """Number of patches ``M`` for ``x``'s spatial shape."""
        return self.plan(x).num_patches

    # ------------------------------------------------------------------ #
    # Restriction (full grid -> extended-block batch)
    # ------------------------------------------------------------------ #
    def _pad_extended(self, x: torch.Tensor, plan: _Plan) -> torch.Tensor:
        """Pad ``x`` (B, C, Nz, Ny, Nx) to the interior-multiple grid, then
        add ``halo`` on every side honouring per-axis boundary mode."""
        n = self.interior_size
        h = self.halo
        pz, py, px = plan.pad
        # interior padding (low side 0, high side `pad`) using the same
        # boundary semantics as the halo fill.
        x = self._pad_axes(x, (0, pz, 0, py, 0, px))
        # halo on every side
        x = self._pad_axes(x, (h, h, h, h, h, h))
        return x

    def _pad_axes(
        self, x: torch.Tensor, amounts: tuple[int, int, int, int, int, int]
    ) -> torch.Tensor:
        """Pad ``x`` by ``(lo_z, hi_z, lo_y, hi_y, lo_x, hi_x)`` using the
        circular mode on periodic axes and ``boundary_mode`` elsewhere.

        Done axis-by-axis because ``F.pad`` takes a single mode for the whole
        call; mixing circular and replicate per axis needs separate passes.
        """
        loz, hiz, loy, hiy, lox, hix = amounts
        # F.pad spec order is (x_lo, x_hi, y_lo, y_hi, z_lo, z_hi).
        # x axis
        if lox or hix:
            mode = "circular" if self.periodic[2] else self.boundary_mode
            x = F.pad(x, (lox, hix, 0, 0, 0, 0), mode=mode)
        if loy or hiy:
            mode = "circular" if self.periodic[1] else self.boundary_mode
            x = F.pad(x, (0, 0, loy, hiy, 0, 0), mode=mode)
        if loz or hiz:
            mode = "circular" if self.periodic[0] else self.boundary_mode
            x = F.pad(x, (0, 0, 0, 0, loz, hiz), mode=mode)
        return x

    def _unfold_blocks(
        self, x: torch.Tensor, plan: _Plan, edge: int
    ) -> torch.Tensor:
        """Extract a uniform ``(B*M, C, edge, edge, edge)`` batch from the
        padded grid ``x`` via strided slicing (stride = interior_size).

        ``edge`` is the block edge length (``n+2h`` for restriction, ``n+2t``
        for the merge crop). The grid must already be padded so that block
        ``p`` along an axis occupies ``[p*n, p*n + edge)``.
        """
        n = self.interior_size
        b, c = x.shape[0], x.shape[1]
        cz, cy, cx = plan.counts
        # unfold creates a strided view of overlapping windows.
        blocks = (
            x.unfold(2, edge, n)
            .unfold(3, edge, n)
            .unfold(4, edge, n)
        )  # (B, C, cz, cy, cx, edge, edge, edge)
        blocks = blocks.permute(0, 2, 3, 4, 1, 5, 6, 7).contiguous()
        return blocks.view(b * cz * cy * cx, c, edge, edge, edge)

    def restrict(self, x: torch.Tensor) -> torch.Tensor:
        """Restrict a full grid ``(B, C, Nz, Ny, Nx)`` to the extended-block
        batch ``(B*M, C, n+2h, n+2h, n+2h)``."""
        plan = self.plan(x)
        edge = self.interior_size + 2 * self.halo
        xp = self._pad_extended(x, plan)
        return self._unfold_blocks(xp, plan, edge)

    def restrict_coarse(self, x: torch.Tensor) -> torch.Tensor:
        """Average-pool a full grid to the coarse level
        ``(B, C, ceil(Nz/r), ...)`` (after padding to a multiple of ``r``)."""
        plan = self.plan(x)
        r = self.coarsen_factor
        cz, cy, cx = plan.coarse_pad
        xp = self._pad_axes(x, (0, cz, 0, cy, 0, cx))
        return F.avg_pool3d(xp, r)

    def restrict_coarse_geom(self, m: torch.Tensor) -> torch.Tensor:
        """Coarsen a geometry/fluid mask to the coarse level.

        ``any_fluid`` (default) keeps a coarse cell fluid if *any* fine cell in
        its window is fluid -- ``(avg_pool3d(m, r) > 0).float()`` -- so thin
        fluid corridors stay visible to the coarse pressure field. ``all_fluid``
        requires the whole window to be fluid.
        """
        plan = self.plan(m)
        r = self.coarsen_factor
        cz, cy, cx = plan.coarse_pad
        mp = self._pad_axes(m, (0, cz, 0, cy, 0, cx))
        pooled = F.avg_pool3d(mp, r)
        if self.geometry_coarsen == "any_fluid":
            return (pooled > 0).to(m.dtype)
        return (pooled >= 1.0 - 1e-6).to(m.dtype)

    def prolong(
        self, coarse: torch.Tensor, x_like: torch.Tensor
    ) -> torch.Tensor:
        """Trilinearly upsample a coarse field back to the fine grid of
        ``x_like`` (shape ``(B, C, Nz, Ny, Nx)``)."""
        grid = tuple(int(s) for s in x_like.shape[-3:])
        return F.interpolate(
            coarse, size=grid, mode="trilinear", align_corners=False
        )

    def positional(self, x: torch.Tensor) -> torch.Tensor:
        """Per-patch positional encoding ``(M, n_pos, n+2h, n+2h, n+2h)`` for
        ``x``'s spatial shape (on its device/dtype)."""
        return self.plan(x).positional

    def neighbor_indices(self, x: torch.Tensor) -> torch.Tensor:
        """Patch adjacency: ``(M, 6)`` flat patch indices of the
        ``(-z, +z, -y, +y, -x, +x)`` neighbours, ``-1`` where a face is on the
        global boundary (and that axis is not periodic). Cached in the plan."""
        return self.plan(x).neighbors

    # ------------------------------------------------------------------ #
    # Merge (extended-block batch -> full grid, windowed scatter-add)
    # ------------------------------------------------------------------ #
    def extend_merge(self, blocks: torch.Tensor, x_like: torch.Tensor) -> torch.Tensor:
        """Merge per-patch extended blocks back into a full grid.

        ``blocks`` is ``(B*M, C, n+2h, n+2h, n+2h)`` (the fine-net output on the
        extended footprint). The central ``(n+2t)`` sub-block of each is
        multiplied by the separable Hann PoU window and scatter-added into a
        grid padded by ``taper`` on each side.

        On a **periodic** axis the ``taper`` overhang of the boundary tiles is
        then **wrapped-added** to the opposite end (the first and last tiles
        blend across the seam), so ``Σ_i w_i ≡ 1`` holds *including* across the
        seam and the merge is C0-continuous there. On a **non-periodic** axis
        the overhang is dropped and the half-ramp is restored to 1 by the
        overlap-sum normalisation, as before. The result is finally normalised
        and cropped back to the exact original grid.
        """
        plan = self.plan(x_like)
        n = self.interior_size
        h = self.halo
        t = self.taper
        edge = n + 2 * t
        b = x_like.shape[0]
        c = blocks.shape[1]
        m = plan.num_patches
        device, dtype = blocks.device, blocks.dtype

        # Crop the extended (n+2h) blocks down to the (n+2t) merge footprint.
        lo = h - t
        merge_blocks = blocks[
            :, :, lo : lo + edge, lo : lo + edge, lo : lo + edge
        ]
        # apply window
        win = plan.window  # (1, 1, edge, edge, edge), already on device/dtype
        merge_blocks = merge_blocks * win  # (B*M, C, edge, edge, edge)

        # Padded accumulation grid: interior-padded grid + `taper` on each side
        # so the negative offsets of edge-patch windows land inside.
        pz, py, px = plan.padded
        acc_z, acc_y, acc_x = pz + 2 * t, py + 2 * t, px + 2 * t
        acc_vol = acc_z * acc_y * acc_x

        # Vectorised overlap-add: scatter every (patch, local-cell) value into the
        # flattened accumulator with a single index_add_. `merge_index` holds the
        # precomputed flat destinations in (patch, a, b, c) order, matching the
        # reshape below. Duplicate indices (the overlaps) accumulate, which is
        # exactly the windowed scatter-add. Patch p's (n+2t) window starts at p*n
        # in the taper-padded accumulator (the window begins `taper` cells before
        # the interior, cancelled by the `taper` left-pad).
        idx = plan.merge_index  # (m * edge**3,) long
        src = merge_blocks.reshape(b, m, c, edge, edge, edge)
        src = src.permute(0, 2, 1, 3, 4, 5).reshape(b, c, m * edge ** 3)
        acc = torch.zeros(b, c, acc_vol, device=device, dtype=dtype)
        acc.index_add_(2, idx, src)
        acc = acc.view(b, c, acc_z, acc_y, acc_x)

        # Overlap-sum of the windows: the same scatter with the window as source.
        # It is batch-independent, so accumulate once (b=1) and broadcast in the
        # division below. The window is tiled across the m patches.
        win_src = (
            win.reshape(1, 1, 1, edge ** 3)
            .expand(1, 1, m, edge ** 3)
            .reshape(1, 1, m * edge ** 3)
        )
        norm = torch.zeros(1, 1, acc_vol, device=device, dtype=dtype)
        norm.index_add_(2, idx, win_src)
        norm = norm.view(1, 1, acc_z, acc_y, acc_x)

        # Fold the taper overhang of periodic axes back onto the opposite end
        # BEFORE normalising, so the wrapped tiles contribute to both the value
        # (acc) and the weight (norm). The accumulator spans [0, L+2t) on each
        # axis with the interior at [t, t+L); for a periodic axis of padded
        # length L the low overhang [0, t) belongs to global [L-t, L) and the
        # high overhang [L+t, L+2t) belongs to global [0, t).
        for axis, L in enumerate(plan.padded):
            if not self.periodic[axis]:
                continue
            acc = self._wrap_fold_axis(acc, axis, t, L)
            norm = self._wrap_fold_axis(norm, axis, t, L)

        # Normalise so Σ_i w_i ≡ 1, then strip the taper pad and interior pad.
        acc = acc / norm.clamp_min(1e-12)
        acc = acc[:, :, t : t + pz, t : t + py, t : t + px]  # interior-padded
        gz, gy, gx = plan.grid
        return acc[:, :, :gz, :gy, :gx]

    @staticmethod
    def _wrap_fold_axis(
        a: torch.Tensor, axis: int, t: int, length: int
    ) -> torch.Tensor:
        """Wrap-add the two ``taper`` overhang bands of ``a`` along spatial
        ``axis`` (0,1,2 -> tensor dim 2,3,4) back onto the opposite interior
        end, returning a tensor whose overhang bands have been zeroed.

        The accumulator along this axis has length ``length + 2t`` with the
        interior at ``[t, t+length)``. The low band ``[0, t)`` is added to
        ``[t+length-t, t+length)`` and the high band ``[t+length, length+2t)``
        to ``[t, 2t)``.
        """
        if t == 0:
            return a
        dim = 2 + axis
        a = a.clone()
        low = a.narrow(dim, 0, t)
        high = a.narrow(dim, t + length, t)
        a.narrow(dim, t + length - t, t).add_(low)
        a.narrow(dim, t, t).add_(high)
        # zero the consumed overhang so the later crop sees a clean interior
        a.narrow(dim, 0, t).zero_()
        a.narrow(dim, t + length, t).zero_()
        return a
