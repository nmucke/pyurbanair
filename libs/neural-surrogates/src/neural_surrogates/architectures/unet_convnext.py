"""3D UNet with ConvNeXt blocks for the neural surrogate.

Same external contract as `SimpleConv`: takes `(state, params, geometry)`
and predicts `state_next`. The geometry mask is concatenated to the state
along the channel dimension at the stem; inflow parameters are injected
as a per-channel bias inside every ConvNeXt block (learned linear
projection from `params` to channel space, broadcast over spatial dims).
"""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn.functional as F
from torch import nn


class _ConvNeXtBlock3d(nn.Module):
    def __init__(
        self,
        channels: int,
        n_params: int,
        kernel_size: int,
        expansion: int,
    ) -> None:
        super().__init__()
        self.dwconv = nn.Conv3d(
            channels,
            channels,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=channels,
        )
        self.norm = nn.GroupNorm(1, channels)
        hidden = channels * expansion
        self.pwconv1 = nn.Conv3d(channels, hidden, kernel_size=1)
        self.act = nn.GELU()
        self.pwconv2 = nn.Conv3d(hidden, channels, kernel_size=1)
        self.param_proj = nn.Linear(n_params, channels) if n_params > 0 else None

    def forward(self, x: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.dwconv(x)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.param_proj is not None:
            bias = self.param_proj(params)
            spatial = (1,) * (x.dim() - 2)
            x = x + bias.view(bias.shape[0], bias.shape[1], *spatial)
        return residual + x


class _Stage(nn.Module):
    def __init__(
        self,
        channels: int,
        n_params: int,
        depth: int,
        kernel_size: int,
        expansion: int,
    ) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(
            [
                _ConvNeXtBlock3d(channels, n_params, kernel_size, expansion)
                for _ in range(depth)
            ]
        )

    def forward(self, x: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x, params)
        return x


class UNetConvNeXt(nn.Module):
    """3D UNet whose encoder/decoder stages are stacks of ConvNeXt blocks.

    Parameters
    ----------
    n_state_channels:
        Number of state channels (e.g. 3 for `u, v, w`).
    n_params:
        Number of inflow parameters, broadcast-added as per-channel bias
        inside every ConvNeXt block.
    base_channels:
        Channel count at the highest-resolution stage. Subsequent stages
        multiply this by ``channel_mults``.
    channel_mults:
        Channel multipliers, one per stage. Length determines depth of
        the UNet (number of downsampling steps = ``len(channel_mults) - 1``).
    depths:
        ConvNeXt-block count per stage. Must match ``len(channel_mults)``.
    kernel_size:
        Depthwise-conv kernel size inside each ConvNeXt block.
    expansion:
        Channel-expansion ratio of the pointwise MLP inside each block.
    """

    def __init__(
        self,
        n_state_channels: int,
        n_params: int,
        base_channels: int = 16,
        channel_mults: Sequence[int] = (1, 2, 4),
        depths: Sequence[int] = (1, 1, 1),
        kernel_size: int = 7,
        expansion: int = 4,
    ) -> None:
        super().__init__()
        if len(channel_mults) != len(depths):
            raise ValueError(
                f"channel_mults ({len(channel_mults)}) and depths ({len(depths)}) "
                "must have the same length"
            )
        if len(channel_mults) < 1:
            raise ValueError("channel_mults must have at least one stage")

        self.n_state_channels = n_state_channels
        self.n_params = n_params
        channels = [base_channels * m for m in channel_mults]
        self.n_levels = len(channels) - 1

        self.stem = nn.Conv3d(
            n_state_channels + 1, channels[0], kernel_size=3, padding=1
        )

        self.encoder_stages = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        for i in range(self.n_levels):
            self.encoder_stages.append(
                _Stage(channels[i], n_params, depths[i], kernel_size, expansion)
            )
            self.downsamples.append(
                nn.Conv3d(channels[i], channels[i + 1], kernel_size=2, stride=2)
            )

        self.bottleneck = _Stage(
            channels[-1], n_params, depths[-1], kernel_size, expansion
        )

        self.upsamples = nn.ModuleList()
        self.fuse = nn.ModuleList()
        self.decoder_stages = nn.ModuleList()
        for i in reversed(range(self.n_levels)):
            self.upsamples.append(
                nn.ConvTranspose3d(
                    channels[i + 1], channels[i], kernel_size=2, stride=2
                )
            )
            self.fuse.append(nn.Conv3d(channels[i] * 2, channels[i], kernel_size=1))
            self.decoder_stages.append(
                _Stage(channels[i], n_params, depths[i], kernel_size, expansion)
            )

        self.head = nn.Conv3d(channels[0], n_state_channels, kernel_size=1)

    def _pad_to_multiple(
        self, x: torch.Tensor, multiple: int
    ) -> tuple[torch.Tensor, tuple[int, int, int]]:
        d, h, w = x.shape[-3:]
        pad_d = (multiple - d % multiple) % multiple
        pad_h = (multiple - h % multiple) % multiple
        pad_w = (multiple - w % multiple) % multiple
        if pad_d or pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h, 0, pad_d))
        return x, (pad_d, pad_h, pad_w)

    def forward(
        self,
        state: torch.Tensor,
        params: torch.Tensor,
        geometry: torch.Tensor,
    ) -> torch.Tensor:
        if geometry.dim() == state.dim() - 1:
            geometry = geometry.unsqueeze(1)
        x = torch.cat([state, geometry], dim=1)

        orig_spatial = x.shape[-3:]
        x, _ = self._pad_to_multiple(x, 2**self.n_levels)

        x = self.stem(x)

        skips = []
        for stage, down in zip(self.encoder_stages, self.downsamples):
            x = stage(x, params)
            skips.append(x)
            x = down(x)

        x = self.bottleneck(x, params)

        for up, fuse, stage, skip in zip(
            self.upsamples, self.fuse, self.decoder_stages, reversed(skips)
        ):
            x = up(x)
            x = fuse(torch.cat([x, skip], dim=1))
            x = stage(x, params)

        x = self.head(x)

        d, h, w = orig_spatial
        x = x[..., :d, :h, :w]
        return x
