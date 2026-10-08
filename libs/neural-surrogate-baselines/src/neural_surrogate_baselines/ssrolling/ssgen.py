"""The small-scale generator (SSGen) of SSRollingUrbanNet (Park & Lee, Phys.
Fluids 38, 085167, 2026, Eqs. 1–8).

A small 2D CNN that restores the small scales a patch-based backbone smooths
out. It works on the backbone's gridded prediction with height folded into
channels, so its convolutions run over (y, x) and fit any horizontal grid:

1. high-pass: the field minus its 3×3 horizontal mean;
2. stem: 3×3 conv, GroupNorm, GELU, dropout;
3. building gate: the solid mask through a per-level 3×3 conv, GroupNorm and
   GELU, averaged over height and projected (``f_b``); then
   ``x' = g⊙x + gate·g⊙f_b`` with ``g = σ(Conv1×1([x, f_b]))``;
4. three refine blocks ``x <- GELU(GN(x + Conv1×1(GELU(GroupedConv3×3(x)))))``;
5. head: 1×1 conv, GroupNorm, GELU, 1×1 conv back to the folded channels.

The model adds the result to the backbone's prediction (Eq. 8).

Not specified by the paper, chosen here: 8 GroupNorm groups and zero padding.
The paper's 9.5 M parameters (at 272 channels) do not follow from its
description; we don't try to match them.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class _Refine(nn.Module):
    def __init__(self, channels: int, groups: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, padding=1, groups=groups)
        self.mix = nn.Conv2d(channels, channels, 1)
        self.norm = nn.GroupNorm(groups, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.gelu(self.norm(x + self.mix(F.gelu(self.conv(x)))))


class SSGen(nn.Module):
    def __init__(
        self,
        channels: int,
        hidden: int = 64,
        gate: float = 0.2,
        dropout: float = 0.1,
        building_channels: int = 8,
        groups: int = 8,
        n_blocks: int = 3,
    ) -> None:
        """``channels``: the folded ``C * nz`` channels of the prediction."""
        super().__init__()
        self.gate = float(gate)
        self.stem = nn.Sequential(
            nn.Conv2d(channels, hidden, 3, padding=1),
            nn.GroupNorm(groups, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.building = nn.Sequential(
            nn.Conv3d(1, building_channels, (1, 3, 3), padding=(0, 1, 1)),
            nn.GroupNorm(groups, building_channels),
            nn.GELU(),
        )
        self.building_proj = nn.Conv2d(building_channels, hidden, 1)
        self.gate_conv = nn.Conv2d(2 * hidden, hidden, 1)
        self.blocks = nn.Sequential(*(_Refine(hidden, groups) for _ in range(n_blocks)))
        self.head = nn.Sequential(
            nn.Conv2d(hidden, hidden, 1),
            nn.GroupNorm(groups, hidden),
            nn.GELU(),
            nn.Conv2d(hidden, channels, 1),
        )

    def forward(self, q: torch.Tensor, solid: torch.Tensor) -> torch.Tensor:
        """The correction to ``q`` ``(B, C, nz, ny, nx)``, given the solid
        mask ``(B, nz, ny, nx)`` (1 = building)."""
        shape = q.shape
        q = q.flatten(1, 2)
        q = q - F.avg_pool2d(q, 3, stride=1, padding=1, count_include_pad=False)
        x = self.stem(q)
        f_b = self.building_proj(self.building(solid.unsqueeze(1)).mean(dim=2))
        g = torch.sigmoid(self.gate_conv(torch.cat([x, f_b], dim=1)))
        x = g * x + self.gate * g * f_b
        return self.head(self.blocks(x)).view(shape)
