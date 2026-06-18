"""3D UNet with ConvNeXt blocks for the neural surrogate.

Same external contract as `SimpleConv`: takes `(state, params, geometry)`
and predicts `state_next`. The geometry mask is concatenated to the state
along the channel dimension at the stem; inflow parameters condition every
ConvNeXt block, either as a per-channel bias (legacy) or as FiLM
scale/shift from a shared parameter embedding (``conditioning="film"``).

With ``normalize=True`` the model standardises state and params with
buffered training-split statistics (installed via ``set_normalization``,
saved with the weights); with ``residual=True`` the head predicts the
state increment instead of the next state. Both default to off so that
checkpoints trained before these options existed still load.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


def _circular_spec(
    pads: tuple[int, int, int],
) -> tuple[int, int, int, int, int, int] | None:
    """``F.pad`` spec (last dim first) circularly padding the given (z, y, x)
    amounts, or ``None`` when nothing needs wrapping."""
    pz, py, px = pads
    if not (pz or py or px):
        return None
    return (px, px, py, py, pz, pz)


class _ConvNeXtBlock3d(nn.Module):
    def __init__(
        self,
        channels: int,
        cond_dim: int,
        kernel_size: int,
        expansion: int,
        separable_dwconv: bool = False,
        conditioning: str = "bias",
        norm_groups: int = 1,
        periodic: tuple[bool, bool, bool] = (False, False, False),
    ) -> None:
        super().__init__()
        if separable_dwconv:
            # Factorized depthwise conv: three axis-aligned k-tap convs
            # (3k taps/voxel) instead of one dense k^3 stencil. Same
            # receptive field; for k=7 this is ~16x fewer MACs in the
            # depthwise step, which dominates block cost at low channel
            # counts in 3D.
            kernels = (
                (kernel_size, 1, 1),
                (1, kernel_size, 1),
                (1, 1, kernel_size),
            )
        else:
            kernels = ((kernel_size,) * 3,)
        # Periodic axes are wrapped with circular F.pad in the forward pass;
        # the convs themselves only zero-pad the non-periodic axes. Module
        # structure (and hence state-dict keys) is identical either way.
        convs = []
        self._dw_circ_specs: list[tuple[int, ...] | None] = []
        for k3 in kernels:
            pads = tuple(k // 2 for k in k3)
            convs.append(
                nn.Conv3d(
                    channels,
                    channels,
                    kernel_size=k3,
                    padding=tuple(
                        0 if per else p for p, per in zip(pads, periodic)
                    ),
                    groups=channels,
                )
            )
            self._dw_circ_specs.append(
                _circular_spec(
                    tuple(p if per else 0 for p, per in zip(pads, periodic))
                )
            )
        self.dwconv = convs[0] if len(convs) == 1 else nn.Sequential(*convs)
        self.norm = nn.GroupNorm(norm_groups, channels)
        hidden = channels * expansion
        self.pwconv1 = nn.Conv3d(channels, hidden, kernel_size=1)
        self.act = nn.GELU()
        self.pwconv2 = nn.Conv3d(hidden, channels, kernel_size=1)
        self.param_proj = None
        self.film = None
        if cond_dim > 0:
            if conditioning == "film":
                # FiLM: per-channel scale/shift applied right after the norm.
                # Zero-init so every block starts as an unconditioned identity
                # modulation and learns its conditioning from there.
                self.film = nn.Linear(cond_dim, 2 * channels)
                nn.init.zeros_(self.film.weight)
                nn.init.zeros_(self.film.bias)
            elif conditioning == "bias":
                self.param_proj = nn.Linear(cond_dim, channels)
            else:
                raise ValueError(
                    f"conditioning must be 'bias' or 'film', got {conditioning!r}"
                )

    def _dwconv_forward(self, x: torch.Tensor) -> torch.Tensor:
        convs = (
            self.dwconv
            if isinstance(self.dwconv, nn.Sequential)
            else (self.dwconv,)
        )
        for conv, spec in zip(convs, self._dw_circ_specs):
            if spec is not None:
                x = F.pad(x, spec, mode="circular")
            x = conv(x)
        return x

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self._dwconv_forward(x)
        x = self.norm(x)
        spatial = (1,) * (x.dim() - 2)
        if self.film is not None:
            gamma, beta = self.film(cond).chunk(2, dim=1)
            x = x * (1.0 + gamma.view(*gamma.shape, *spatial)) + beta.view(
                *beta.shape, *spatial
            )
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.param_proj is not None:
            bias = self.param_proj(cond)
            x = x + bias.view(bias.shape[0], bias.shape[1], *spatial)
        return residual + x


class _Stage(nn.Module):
    def __init__(
        self,
        channels: int,
        cond_dim: int,
        depth: int,
        kernel_size: int,
        expansion: int,
        separable_dwconv: bool = False,
        conditioning: str = "bias",
        norm_groups: int = 1,
        periodic: tuple[bool, bool, bool] = (False, False, False),
    ) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(
            [
                _ConvNeXtBlock3d(
                    channels,
                    cond_dim,
                    kernel_size,
                    expansion,
                    separable_dwconv,
                    conditioning,
                    norm_groups,
                    periodic,
                )
                for _ in range(depth)
            ]
        )

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x, cond)
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
    separable_dwconv:
        Replace each block's dense ``k^3`` depthwise conv with three
        axis-aligned ``k``-tap depthwise convs (same receptive field,
        ``3k`` instead of ``k^3`` taps per voxel). Weights are not
        compatible between the two settings.
    conditioning:
        ``"bias"`` (legacy): per-channel bias added after the block MLP,
        projected directly from the params. ``"film"``: params pass through
        a shared embedding MLP once per forward; each block applies a
        zero-initialised per-channel scale/shift right after its norm.
    param_embed_dim:
        Width of the shared parameter embedding (``"film"`` only).
    norm_groups:
        Number of groups for each block's GroupNorm. The legacy value 1
        normalises over channels and space jointly; 8 is the conventional
        choice. State-dict compatible either way.
    normalize:
        Standardise state and params with buffered training statistics
        (install via :meth:`set_normalization`); the output is mapped back
        to physical units. Adds buffers to the state dict, so checkpoints
        trained without it need ``normalize=False``.
    residual:
        Predict the state increment rather than the next state; the
        identity rollout becomes the zero-output solution.
    periodic_axes:
        Spatial axes (subset of ``"z", "y", "x"``; tensor layout is
        ``(B, C, z, y, x)``) with periodic boundaries. Spatial convs wrap
        circularly along these axes instead of zero-padding, i.e. the
        last cell's neighbour is the first cell (no equality between
        boundary values is imposed). Weight-compatible with models
        trained without it (only padding semantics change). Sizes along
        periodic axes must be divisible by ``2^(len(channel_mults)-1)``.
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
        separable_dwconv: bool = False,
        conditioning: str = "bias",
        param_embed_dim: int = 64,
        norm_groups: int = 1,
        normalize: bool = False,
        residual: bool = False,
        periodic_axes: Sequence[str] = (),
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
        self.normalize = normalize
        self.residual = residual
        axes = tuple(periodic_axes or ())
        unknown = set(axes) - {"z", "y", "x"}
        if unknown:
            raise ValueError(
                f"periodic_axes entries must be in ('z', 'y', 'x'), got {sorted(unknown)}"
            )
        self._periodic = tuple(a in axes for a in ("z", "y", "x"))
        channels = [base_channels * m for m in channel_mults]
        self.n_levels = len(channels) - 1

        if normalize:
            self.register_buffer("state_mean", torch.zeros(n_state_channels))
            self.register_buffer("state_std", torch.ones(n_state_channels))
            self.register_buffer("param_mean", torch.zeros(max(n_params, 1)))
            self.register_buffer("param_std", torch.ones(max(n_params, 1)))

        self.param_embed = None
        cond_dim = n_params
        if conditioning == "film" and n_params > 0:
            self.param_embed = nn.Sequential(
                nn.Linear(n_params, param_embed_dim),
                nn.GELU(),
                nn.Linear(param_embed_dim, param_embed_dim),
            )
            cond_dim = param_embed_dim

        self.stem = nn.Conv3d(
            n_state_channels + 1,
            channels[0],
            kernel_size=3,
            padding=tuple(0 if per else 1 for per in self._periodic),
        )
        self._stem_circ_spec = _circular_spec(
            tuple(1 if per else 0 for per in self._periodic)
        )

        self.encoder_stages = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        stage_kwargs = dict(
            kernel_size=kernel_size,
            expansion=expansion,
            separable_dwconv=separable_dwconv,
            conditioning=conditioning,
            norm_groups=norm_groups,
            periodic=self._periodic,
        )
        for i in range(self.n_levels):
            self.encoder_stages.append(
                _Stage(channels[i], cond_dim, depths[i], **stage_kwargs)
            )
            self.downsamples.append(
                nn.Conv3d(channels[i], channels[i + 1], kernel_size=2, stride=2)
            )

        self.bottleneck = _Stage(channels[-1], cond_dim, depths[-1], **stage_kwargs)

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
                _Stage(channels[i], cond_dim, depths[i], **stage_kwargs)
            )

        self.head = nn.Conv3d(channels[0], n_state_channels, kernel_size=1)

    def set_normalization(
        self,
        state_mean,
        state_std,
        param_mean,
        param_std,
    ) -> None:
        """Install training-split standardisation statistics into the buffers.

        Accepts array-likes (numpy or tensor). Zero stds (e.g. a parameter
        that is constant across the training split) are replaced by 1 so the
        standardisation is a no-op for that channel rather than a division
        by zero.
        """
        if not self.normalize:
            print(
                "UNetConvNeXt(normalize=False): ignoring normalization stats"
            )
            return

        def _fill(buffer: torch.Tensor, values) -> None:
            vals = torch.as_tensor(
                np.asarray(values), dtype=buffer.dtype, device=buffer.device
            ).reshape(-1)
            if vals.numel() != buffer.numel():
                raise ValueError(
                    f"expected {buffer.numel()} values, got {vals.numel()}"
                )
            buffer.copy_(vals)

        _fill(self.state_mean, state_mean)
        _fill(self.state_std, state_std)
        if self.n_params > 0:
            _fill(self.param_mean, param_mean)
            _fill(self.param_std, param_std)
        self.state_std.copy_(
            torch.where(self.state_std > 0, self.state_std, torch.ones_like(self.state_std))
        )
        self.param_std.copy_(
            torch.where(self.param_std > 0, self.param_std, torch.ones_like(self.param_std))
        )

    def _pad_to_multiple(
        self, x: torch.Tensor, multiple: int
    ) -> tuple[torch.Tensor, tuple[int, int, int]]:
        d, h, w = x.shape[-3:]
        pad_d = (multiple - d % multiple) % multiple
        pad_h = (multiple - h % multiple) % multiple
        pad_w = (multiple - w % multiple) % multiple
        # Zero-padding a periodic axis would break the wrap (the strided
        # down/upsamples would see a corrupted period), so periodic sizes
        # must already be divisible.
        for name, size, pad, per in zip(
            ("z", "y", "x"), (d, h, w), (pad_d, pad_h, pad_w), self._periodic
        ):
            if per and pad:
                raise ValueError(
                    f"size along periodic axis '{name}' ({size}) must be "
                    f"divisible by 2^n_levels = {multiple}"
                )
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
        # Zero out obstacle interiors: uDALES fielddumps carry junk values
        # there, and standardisation below would otherwise shift solid cells
        # off zero.
        state = state * geometry

        if self.normalize:
            ch = (1, -1) + (1,) * (state.dim() - 2)
            x = (state - self.state_mean.view(ch)) / self.state_std.view(ch)
            x = x * geometry
            if self.n_params > 0:
                params = (params - self.param_mean) / self.param_std
        else:
            x = state

        cond = self.param_embed(params) if self.param_embed is not None else params

        x = torch.cat([x, geometry], dim=1)

        orig_spatial = x.shape[-3:]
        x, _ = self._pad_to_multiple(x, 2**self.n_levels)

        if self._stem_circ_spec is not None:
            x = F.pad(x, self._stem_circ_spec, mode="circular")
        x = self.stem(x)

        skips = []
        for stage, down in zip(self.encoder_stages, self.downsamples):
            x = stage(x, cond)
            skips.append(x)
            x = down(x)

        x = self.bottleneck(x, cond)

        for up, fuse, stage, skip in zip(
            self.upsamples, self.fuse, self.decoder_stages, reversed(skips)
        ):
            x = up(x)
            x = fuse(torch.cat([x, skip], dim=1))
            x = stage(x, cond)

        x = self.head(x)

        d, h, w = orig_spatial
        x = x[..., :d, :h, :w]

        # Map the head output back to physical units. With ``residual`` the
        # network predicts the (normalised) increment, so the identity
        # rollout corresponds to zero output.
        if self.normalize:
            x = x * self.state_std.view(ch)
            if not self.residual:
                x = x + self.state_mean.view(ch)
        if self.residual:
            x = state + x

        x = geometry * x

        return x
