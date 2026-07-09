"""Shared field-IO helpers for the Tadpole wrappers.

``TadpoleAE`` (plan 02, snapshot pre-training) and ``TadpoleTimeStepper``
(plan 03, AE -> time-stepper) both wrap a vendored Tadpole encoder/decoder and
therefore need the *same* pre/post-processing around it: obstacle masking,
per-channel state z-scoring, geometry/SDF channel assembly, and padding the grid
up to a multiple of ``encoder_crop_size`` so the upstream fold tiles cleanly.

Those helpers are collected here as a plain (non-``nn.Module``) mixin so both
wrappers inherit one implementation. The mixin reads instance attributes the host
class is responsible for setting in ``__init__``:

``encoder_crop_size``, ``normalize``, ``encode_geometry``,
``sdf_features_enabled``, ``sdf_feature_mode``, ``sdf_clamp_cells``,
``n_geom_feature_channels`` and (when ``normalize``) the ``state_mean`` /
``state_std`` buffers.

The behaviour is byte-identical to the original ``TadpoleAE`` methods -- this
module is a pure extraction, not a rewrite.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import torch
import torch.nn.functional as F
from neural_surrogates.sdf import sdf_features as compute_sdf_features


class _TadpoleFieldIO:
    """Mask / normalise / assemble / pad helpers shared by the Tadpole wrappers."""

    # Instance attributes the host class (TadpoleAE / TadpoleTimeStepper) sets in
    # its ``__init__``; declared here (annotation only, no assignment) so this
    # mixin type-checks against the shared contract without creating class-level
    # defaults that would shadow the hosts' buffers/attrs.
    encoder_crop_size: int
    normalize: bool
    encode_geometry: bool
    sdf_features_enabled: bool
    sdf_feature_mode: str
    sdf_clamp_cells: float
    n_geom_feature_channels: int
    state_mean: torch.Tensor
    state_std: torch.Tensor

    # -- spatial padding --------------------------------------------------- #

    def _pad_to_crop_multiple(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, tuple[int, int, int]]:
        """Zero-pad ``(D, H, W)`` up to a multiple of ``encoder_crop_size`` so the
        autoencoder tiles cleanly; return the padded tensor and the original
        spatial shape for cropping the reconstruction back."""
        mult = self.encoder_crop_size
        d, h, w = x.shape[-3:]
        pad_d, pad_h, pad_w = ((mult - s % mult) % mult for s in (d, h, w))
        if pad_d or pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h, 0, pad_d))
        return x, (d, h, w)

    # -- input assembly ---------------------------------------------------- #

    def _geometry_channels(
        self,
        geometry: torch.Tensor,
        geom_features: torch.Tensor | None,
        b: int,
        grid: Sequence[int],
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Raw geometry block ``(B, n_geometry_channels, *grid)``: the mask,
        followed by the selected SDF channels (self-computed if not supplied)."""
        mask = geometry.to(dtype=dtype)  # (B, *grid)
        pieces = [mask.unsqueeze(1)]  # (B, 1, *grid)
        if self.sdf_features_enabled:
            if geom_features is None:
                geom_features = self._sdf_features(geometry)
            geom_features = geom_features.to(dtype=dtype)
            if geom_features.shape[1] != self.n_geom_feature_channels:
                raise ValueError(
                    f"geom_features has {geom_features.shape[1]} channels, expected "
                    f"{self.n_geom_feature_channels}"
                )
            if geom_features.shape[0] == 1 and b != 1:
                geom_features = geom_features.expand(b, *geom_features.shape[1:])
            pieces.append(geom_features)
        return torch.cat(pieces, dim=1)

    def _sdf_features(self, geometry: torch.Tensor) -> torch.Tensor:
        """Compute ``(B, C, *grid)`` SDF features from a ``(B, *grid)`` /
        ``(*grid,)`` mask (inference/analysis convenience; training ships them)."""
        g = geometry
        if g.dim() == 3:  # single (z, y, x)
            return compute_sdf_features(
                g, clamp_cells=self.sdf_clamp_cells, mode=self.sdf_feature_mode
            ).unsqueeze(0)
        feats = [
            compute_sdf_features(
                g[i], clamp_cells=self.sdf_clamp_cells, mode=self.sdf_feature_mode
            )
            for i in range(g.shape[0])
        ]
        return torch.stack(feats, dim=0)

    def _normalize_state(self, state: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Mask obstacles to zero and z-score each state channel (masked again so
        obstacle cells stay exactly zero)."""
        state = state * mask
        if not self.normalize:
            return state
        ch = (1, -1) + (1,) * (state.dim() - 2)
        x = (state - self.state_mean.view(ch)) / self.state_std.view(ch)
        return x * mask

    def _denormalize_state(self, x: torch.Tensor) -> torch.Tensor:
        if not self.normalize:
            return x
        ch = (1, -1) + (1,) * (x.dim() - 2)
        return x * self.state_std.view(ch) + self.state_mean.view(ch)

    def _fold_dims(self, x: torch.Tensor) -> tuple[int, int, int, int, int]:
        cs = self.encoder_crop_size
        b, c = x.shape[0], x.shape[1]
        u = max(x.shape[2] // cs, 1)
        v = max(x.shape[3] // cs, 1)
        w = max(x.shape[4] // cs, 1)
        return b, c, u, v, w

    def _assemble_working_input(
        self,
        state: torch.Tensor,
        geometry: torch.Tensor,
        geom_features: torch.Tensor | None,
    ) -> torch.Tensor:
        """Build the autoencoder's working-space input
        ``[normalised state, (geometry block)]`` -> ``(B, Cin, *grid)``."""
        if geometry.dim() == state.dim() - 1:  # (B, *grid) alongside (B, C, *grid)
            pass
        elif geometry.dim() == state.dim() - 2:  # unbatched (*grid,)
            geometry = geometry.unsqueeze(0).expand(state.shape[0], *geometry.shape)
        mask = geometry.unsqueeze(1).to(dtype=state.dtype)  # (B, 1, *grid)
        x = self._normalize_state(state, mask)
        if self.encode_geometry:
            geom_block = self._geometry_channels(
                geometry, geom_features, state.shape[0], state.shape[2:], state.dtype
            )
            x = torch.cat([x, geom_block], dim=1)
        return x

    @staticmethod
    def _batched_mask(geometry: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        """``(B, 1, *grid)`` fluid mask broadcastable over the state channels."""
        if geometry.dim() == state.dim() - 2:  # (*grid,)
            geometry = geometry.unsqueeze(0).expand(state.shape[0], *geometry.shape)
        return geometry.unsqueeze(1).to(dtype=state.dtype)

    # -- normalisation-buffer install helper ------------------------------- #

    @staticmethod
    def _to_buffer(buf: torch.Tensor, value, eps: float | None = None) -> torch.Tensor:
        """Coerce ``value`` to ``buf``'s dtype/device/shape (optionally floored)."""
        t = torch.as_tensor(
            np.asarray(value), dtype=buf.dtype, device=buf.device
        ).reshape(-1)
        if t.numel() != buf.numel():
            raise ValueError(f"expected {buf.numel()} values, got {t.numel()}")
        if eps is not None:
            t = t.clamp_min(eps)
        return t
