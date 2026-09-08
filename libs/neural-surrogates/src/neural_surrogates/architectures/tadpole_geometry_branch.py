"""Geometry side-branch for the Tadpole autoencoder / time-stepper.

The default Tadpole path *folds* the geometry block ``[mask, (sdf, grad-sdf)]``
into the single-channel encoder as extra channels and reconstructs it, so the
obstacle field is squeezed through the same 16x-compressing latent as the flow
and the state crops themselves never see it (see ``tadpole_ae.py``). The
**geometry branch** is the alternative: a small trainable conv net that keeps the
geometry in *feature* space at several resolutions, which the encoder/decoder
then read as an additive conditioning signal at matching strides -- "condition,
don't predict".

:class:`GeometryBranch` maps the raw geometry block ``(B, in_channels, X, Y, Z)``
(on the **padded** grid, i.e. every spatial dim already a multiple of
``encoder_crop_size`` and hence of 16) to four feature maps at strides

``(1, 2, 4, 16)``

relative to that grid. Those strides are not arbitrary -- they are exactly the
resolutions the vendored P3D stack exposes:

* stride 1 / 2 / 4 are the three stages of the encoder's conv stem
  (``feature_embed`` -> ``downsampling_layers[0]`` -> ``downsampling_layers[1]``)
  and, mirrored, of the decoder's conv up-path;
* stride 16 is the latent grid (the conv stem's 4x times the transformer's 4x),
  where the level-3 feature conditions the decoder's latent input and (in the
  time-stepper) the latent subnetwork's FiLM.

The consumers own the projections that turn these features into the host's
channel counts (zero-initialised ``1x1x1`` convs living *inside* the vendored
encoder/decoder, so they travel in ``encoder.pt`` / ``decoder.pt``); this module
only produces the shared, host-agnostic features and is checkpointed separately
as ``geometry_branch.pt``.

The branch is deliberately small and plain (strided conv + GroupNorm + GELU): the
geometry cue is a bounded, low-frequency field, and every level's output is
consumed through a zero-init projection, so capacity here is not what limits the
model -- identity-at-init and cheap re-evaluation per step are.
"""

from __future__ import annotations

from typing import Sequence

import torch
from torch import nn


def _num_groups(channels: int) -> int:
    """Largest group count in ``(8, 4, 2, 1)`` that divides ``channels``.

    ``GroupNorm`` requires the channel count to be divisible by the group count;
    ``out_dims`` is user-configurable, so pick a divisor rather than assume 8.
    """
    for g in (8, 4, 2, 1):
        if channels % g == 0:
            return g
    return 1  # pragma: no cover - unreachable (1 always divides)


def _block(in_channels: int, out_channels: int, stride: int) -> nn.Sequential:
    """Strided conv -> norm -> activation -> conv, one branch level."""
    return nn.Sequential(
        nn.Conv3d(in_channels, out_channels, 3, stride, 1),
        nn.GroupNorm(_num_groups(out_channels), out_channels),
        nn.GELU(),
        nn.Conv3d(out_channels, out_channels, 3, 1, 1),
    )


class GeometryBranch(nn.Module):
    """Multi-resolution geometry features for the Tadpole encoder/decoder.

    Parameters
    ----------
    in_channels:
        Channels of the raw geometry block: ``1 + n_sdf_feature_channels(mode)``
        (the mask plus any SDF / gradient channels). The host derives it from its
        own ``sdf_features`` setting.
    width:
        Base feature width; the default ``out_dims`` is
        ``(width, 2*width, 4*width, 8*width)``.
    out_dims:
        Explicit per-level feature widths (4 ints), overriding ``width``.

    Attributes
    ----------
    strides:
        ``(1, 2, 4, 16)`` -- the downsampling of each returned level relative to
        the (padded) input grid.
    out_dims:
        The four feature widths, in level order.
    """

    strides: tuple[int, int, int, int] = (1, 2, 4, 16)

    def __init__(
        self,
        in_channels: int,
        width: int = 32,
        out_dims: Sequence[int] | None = None,
    ) -> None:
        super().__init__()

        if in_channels < 1:
            raise ValueError(f"in_channels must be >= 1, got {in_channels}")
        if out_dims is None:
            dims = (width, 2 * width, 4 * width, 8 * width)
        else:
            values = tuple(int(d) for d in out_dims)
            if len(values) != 4:
                raise ValueError(
                    f"out_dims must have 4 entries (one per level), got {len(values)}"
                )
            dims = (values[0], values[1], values[2], values[3])
        self.in_channels = int(in_channels)
        self.out_dims: tuple[int, int, int, int] = dims

        d0, d1, d2, d3 = self.out_dims
        # Level 0 (stride 1) is the stem; 1 and 2 each halve; level 3 drops the
        # remaining 4x in one block (two strided convs) so it lands on the latent
        # grid without an intermediate feature nobody consumes.
        self.level_0 = _block(self.in_channels, d0, 1)
        self.level_1 = _block(d0, d1, 2)
        self.level_2 = _block(d1, d2, 2)
        self.level_3 = nn.Sequential(
            nn.Conv3d(d2, d3, 3, 2, 1),
            nn.GroupNorm(_num_groups(d3), d3),
            nn.GELU(),
            nn.Conv3d(d3, d3, 3, 2, 1),
        )

    def forward(self, geom_block: torch.Tensor) -> list[torch.Tensor]:
        """``(B, in_channels, X, Y, Z)`` -> the four features, coarsest last.

        ``X/Y/Z`` must already be padded to a multiple of the host's
        ``encoder_crop_size`` (hence of 16), so level ``i`` comes out at exactly
        ``X / strides[i]`` per axis.
        """
        if geom_block.shape[1] != self.in_channels:
            raise ValueError(
                f"geometry block has {geom_block.shape[1]} channels, expected "
                f"{self.in_channels}"
            )
        f0 = self.level_0(geom_block)
        f1 = self.level_1(f0)
        f2 = self.level_2(f1)
        f3 = self.level_3(f2)
        return [f0, f1, f2, f3]
