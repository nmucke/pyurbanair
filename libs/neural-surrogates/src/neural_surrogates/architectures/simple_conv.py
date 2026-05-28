"""Minimal single-layer convolutional surrogate.

Stands up an end-to-end training loop without committing to an architecture.
The geometry mask is concatenated with the state along the channel
dimension; each inflow parameter is broadcast-added to one output channel.
"""

from __future__ import annotations

import torch
from torch import nn


class SimpleConv(nn.Module):
    def __init__(
        self,
        n_state_channels: int,
        n_params: int,
        kernel_size: int = 3,
    ) -> None:
        super().__init__()
        if n_params > n_state_channels:
            raise ValueError(
                f"n_params ({n_params}) must be <= n_state_channels "
                f"({n_state_channels}); each parameter is added to one output channel"
            )
        self.n_state_channels = n_state_channels
        self.n_params = n_params
        self.conv = nn.Conv3d(
            in_channels=n_state_channels + 1,
            out_channels=n_state_channels,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
        )

    def forward(
        self,
        state: torch.Tensor,
        params: torch.Tensor,
        geometry: torch.Tensor,
    ) -> torch.Tensor:
        if geometry.dim() == state.dim() - 1:
            geometry = geometry.unsqueeze(1)
        x = torch.cat([state, geometry], dim=1)
        y = self.conv(x)

        spatial_dims = y.dim() - 2
        broadcast_shape = (-1, 1) + (1,) * spatial_dims
        pieces = [
            params[:, i : i + 1].view(broadcast_shape) for i in range(self.n_params)
        ]
        n_pad = self.n_state_channels - self.n_params
        if n_pad > 0:
            pieces.append(
                params.new_zeros((params.shape[0], n_pad) + (1,) * spatial_dims)
            )
        return y + torch.cat(pieces, dim=1)
