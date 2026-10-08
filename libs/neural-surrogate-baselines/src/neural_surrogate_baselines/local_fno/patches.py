"""Overlapping horizontal patches for Local-FNO (Qin et al. 2025, Fig. 3).

The domain is covered by a grid of ``core x core`` cores (horizontal only;
patches span the full height). Each patch is its core plus ``overlap`` cells
on every side. A prediction keeps only each patch's core, so every overlap
cell is predicted by the neighbour whose core contains it ("split evenly").

Before tiling, each horizontal axis is extended to whole cores plus the
overlap: by wrapping on a periodic axis, by repeating the edge cell
otherwise. Wrapped cells are real data; repeated ones are not, which
:meth:`PatchGrid.valid` records so a loss can skip them.
"""

from __future__ import annotations

import dataclasses
import math

import torch


@dataclasses.dataclass(frozen=True)
class PatchGrid:
    """The patch tiling of a ``(ny, nx)`` horizontal grid."""

    ny: int
    nx: int
    core: int
    overlap: int
    periodic_y: bool
    periodic_x: bool

    @property
    def size(self) -> int:
        return self.core + 2 * self.overlap

    @property
    def n_y(self) -> int:
        return math.ceil(self.ny / self.core)

    @property
    def n_x(self) -> int:
        return math.ceil(self.nx / self.core)

    @property
    def n_patches(self) -> int:
        return self.n_y * self.n_x

    def _indices(self, n: int, n_cores: int, periodic: bool) -> torch.Tensor:
        idx = torch.arange(-self.overlap, n_cores * self.core + self.overlap)
        return idx % n if periodic else idx.clamp(0, n - 1)

    def _inside(self, n: int, n_cores: int, periodic: bool) -> torch.Tensor:
        idx = torch.arange(-self.overlap, n_cores * self.core + self.overlap)
        return (
            torch.ones_like(idx, dtype=torch.bool)
            if periodic
            else (idx >= 0) & (idx < n)
        )

    def pad(self, x: torch.Tensor) -> torch.Tensor:
        """``(..., ny, nx)`` -> ``(..., n_y*core + 2*overlap, n_x*core + 2*overlap)``."""
        iy = self._indices(self.ny, self.n_y, self.periodic_y).to(x.device)
        ix = self._indices(self.nx, self.n_x, self.periodic_x).to(x.device)
        return x.index_select(-2, iy).index_select(-1, ix)

    def extract(self, x: torch.Tensor) -> torch.Tensor:
        """``(B, C, nz, ny, nx)`` -> ``(B*n_patches, C, nz, size, size)``."""
        b, c, nz = x.shape[:3]
        p = self.pad(x).unfold(3, self.size, self.core).unfold(4, self.size, self.core)
        # (B, C, nz, n_y, n_x, size, size) -> (B, n_y, n_x, C, nz, size, size)
        p = p.permute(0, 3, 4, 1, 2, 5, 6)
        return p.reshape(b * self.n_patches, c, nz, self.size, self.size)

    def stitch(self, patches: torch.Tensor) -> torch.Tensor:
        """Keep each patch's core: ``(B*n_patches, C, nz, size, size)`` -> ``(B, C, nz, ny, nx)``."""
        c, nz = patches.shape[1:3]
        o, k = self.overlap, self.core
        cores = patches[..., o : o + k, o : o + k]
        cores = cores.reshape(-1, self.n_y, self.n_x, c, nz, k, k)
        # (B, n_y, n_x, C, nz, k, k) -> (B, C, nz, n_y, k, n_x, k)
        full = cores.permute(0, 3, 4, 1, 5, 2, 6).reshape(
            -1, c, nz, self.n_y * k, self.n_x * k
        )
        return full[..., : self.ny, : self.nx]

    def valid(self, nz: int, device: torch.device | None = None) -> torch.Tensor:
        """``(n_patches, 1, nz, size, size)``: True where a patch cell is real data."""
        iy = self._inside(self.ny, self.n_y, self.periodic_y)
        ix = self._inside(self.nx, self.n_x, self.periodic_x)
        grid = (iy[:, None] & ix[None, :]).to(device)
        grid = grid.unfold(0, self.size, self.core).unfold(1, self.size, self.core)
        grid = grid.reshape(self.n_patches, 1, 1, self.size, self.size)
        return grid.expand(-1, -1, nz, -1, -1)
