"""Pointwise state mixing added alongside DFT's gated skip connections."""

from collections.abc import Sequence

import torch
from torch import nn


class SkipMixingAdapter(nn.Module):
    """Mix aligned state features, with an exactly zero initial correction."""

    def __init__(self, channels: int, width: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(channels)
        self.input_proj = nn.Conv3d(channels, width, 1)
        self.activation = nn.GELU()
        self.output_proj = nn.Conv3d(width, channels, 1)
        nn.init.zeros_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm(x.movedim(1, -1)).movedim(-1, 1)
        return self.output_proj(self.activation(self.input_proj(x)))


class TadpoleSkipMixing(nn.Module):
    """Separate adapters for skips at strides 1, 2, 4 and 8.

    Inputs have batch order B*Cin for one spatial region. Only the first
    n_state_channels participate; geometry receives zero corrections.
    """

    def __init__(
        self,
        n_state_channels: int,
        n_input_channels: int,
        feature_dims: Sequence[int],
        width: int = 32,
        levels: Sequence[int] = (4, 8),
    ) -> None:
        super().__init__()
        if isinstance(width, bool) or not isinstance(width, int) or width <= 0:
            raise ValueError("skip_mixing.width must be a positive integer")
        levels = tuple(levels)
        if (
            not levels
            or any(
                type(level) is not int or level not in (1, 2, 4, 8) for level in levels
            )
            or len(set(levels)) != len(levels)
        ):
            raise ValueError(
                "skip_mixing.levels must be distinct strides from [1, 2, 4, 8]"
            )
        self.n_state_channels = n_state_channels
        self.n_input_channels = n_input_channels
        self.adapters = nn.ModuleDict(
            {
                str(stride): SkipMixingAdapter(n_state_channels * features, width)
                for stride, features in zip((1, 2, 4, 8), feature_dims)
                if stride in levels
            }
        )

    def forward(
        self, skips: list[list[torch.Tensor]]
    ) -> list[list[torch.Tensor | None]]:
        corrections: list[list[torch.Tensor | None]] = [[], []]
        for group, group_skips in enumerate(skips):
            for level, skip in enumerate(group_skips):
                key = str(2 ** (2 * group + level))
                if key not in self.adapters:
                    corrections[group].append(None)
                    continue
                features = skip.shape[1]
                grouped = skip.reshape(-1, self.n_input_channels, *skip.shape[1:])
                states = grouped[:, : self.n_state_channels].flatten(1, 2)
                mixed = self.adapters[key](states).reshape(
                    grouped.shape[0], self.n_state_channels, features, *skip.shape[2:]
                )
                if self.n_input_channels > self.n_state_channels:
                    mixed = torch.cat(
                        (mixed, torch.zeros_like(grouped[:, self.n_state_channels :])),
                        dim=1,
                    )
                corrections[group].append(mixed.reshape_as(skip))
        return corrections
