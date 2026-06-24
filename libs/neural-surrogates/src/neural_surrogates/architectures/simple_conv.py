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
        extra_in_channels: int = 0,
    ) -> None:
        super().__init__()
        if n_params > n_state_channels:
            raise ValueError(
                f"n_params ({n_params}) must be <= n_state_channels "
                f"({n_state_channels}); each parameter is added to one output channel"
            )
        if extra_in_channels < 0:
            raise ValueError("extra_in_channels must be >= 0")
        self.n_state_channels = n_state_channels
        self.n_params = n_params
        # ``extra_in_channels`` are raw input-only channels (e.g. a coarse-context
        # field + positional encoding fed by the domain-decomposition wrapper).
        # They widen the input stem but NOT the output: with the default 0 the
        # stem and state-dict are byte-identical to the original.
        self.extra_in_channels = int(extra_in_channels)
        self.conv = nn.Conv3d(
            in_channels=n_state_channels + 1 + self.extra_in_channels,
            out_channels=n_state_channels,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
        )

    def forward(
        self,
        state: torch.Tensor,
        params: torch.Tensor,
        geometry: torch.Tensor,
        extra: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if (extra is None) != (self.extra_in_channels == 0):
            raise ValueError(
                "extra must be provided iff extra_in_channels > 0 "
                f"(extra_in_channels={self.extra_in_channels}, "
                f"extra={'None' if extra is None else 'tensor'})"
            )
        if extra is not None and extra.shape[1] != self.extra_in_channels:
            raise ValueError(
                f"extra has {extra.shape[1]} channels, expected "
                f"{self.extra_in_channels}"
            )
        if geometry.dim() == state.dim() - 1:
            geometry = geometry.unsqueeze(1)
        pieces = [state, geometry]
        if extra is not None:
            # raw, unnormalised, unmasked -- concatenated right after geometry.
            pieces.append(extra)
        x = torch.cat(pieces, dim=1)
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
