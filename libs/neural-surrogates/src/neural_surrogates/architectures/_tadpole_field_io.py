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
``n_geom_feature_channels``, ``geometry_branch`` (``None`` when off) and (when
``normalize``) the ``state_mean`` / ``state_std`` buffers.

Geometry-branch mode
--------------------
When the host builds a
:class:`~neural_surrogates.architectures.tadpole_geometry_branch.GeometryBranch`,
geometry is no longer folded through the encoder as extra channels: it is turned
into multi-resolution features that condition the encoder/decoder additively.
Two helpers support that -- :meth:`_TadpoleFieldIO._branch_features` (raw
geometry block -> padded -> branch -> the 4 *unfolded* features) and
:meth:`_TadpoleFieldIO._fold_geom_feats` (those features folded into the crop
batch, in the same ``(B C U V W)`` order as the state fold) -- and
:meth:`_TadpoleFieldIO._assemble_working_input` then returns the **state channels
only**.

The behaviour on the default (no-branch) path is byte-identical to the original
``TadpoleAE`` methods -- this module is a pure extraction, not a rewrite.
"""

from __future__ import annotations

from typing import Any, Sequence

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
    # ``None`` when geometry-branch conditioning is off. Read through
    # ``getattr`` below so a host that predates the branch still works.
    geometry_branch: torch.nn.Module | None

    # Size-1 cache for a FROZEN branch's features (the DFT setting): a
    # ``(geometry_tensor, (b, *grid), features)`` triple. Unlike the attributes
    # above this one is owned by the mixin, so it gets a class-level default --
    # holding a reference to the keyed geometry keeps that object alive, so
    # identity (``is``) can never alias a recycled tensor (same argument as
    # ``P3D._sdf_cache``). A plain attribute, never a buffer: it stays out of the
    # state dict.
    _branch_cache: (
        tuple[
            tuple[torch.Tensor, torch.Tensor | None],
            tuple[Any, ...],
            list[torch.Tensor],
        ]
        | None
    ) = None

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

    def _expand_geometry(
        self, geometry: torch.Tensor, state: torch.Tensor
    ) -> torch.Tensor:
        """Normalise a mask to ``(B, *grid)`` against ``state``'s batch."""
        if geometry.dim() == state.dim() - 2:  # unbatched (*grid,)
            return geometry.unsqueeze(0).expand(state.shape[0], *geometry.shape)
        if geometry.dim() == state.dim() - 1:  # (B, *grid) alongside (B, C, *grid)
            # A single-mask (1, *grid) alongside a batched state must be expanded
            # to B, not left to broadcast silently over the batch.
            if geometry.shape[0] == 1 and state.shape[0] != 1:
                return geometry.expand(state.shape[0], *geometry.shape[1:])
            return geometry
        raise ValueError(
            f"geometry shape {tuple(geometry.shape)} is not a mask for state "
            f"shape {tuple(state.shape)}; expected (*grid,) or (B, *grid) with "
            f"grid == {tuple(state.shape[2:])}"
        )

    def _assemble_working_input(
        self,
        state: torch.Tensor,
        geometry: torch.Tensor,
        geom_features: torch.Tensor | None,
    ) -> torch.Tensor:
        """Build the autoencoder's working-space input
        ``[normalised state, (geometry block)]`` -> ``(B, Cin, *grid)``.

        In geometry-branch mode the geometry block is **not** appended: geometry
        conditions the encoder/decoder through the branch instead of being folded
        and reconstructed, so the working space is the state channels only (the
        host's ``encode_geometry`` is forced off in that mode anyway; the branch
        check below states the intent at the point it matters)."""
        geometry = self._expand_geometry(geometry, state)
        mask = geometry.unsqueeze(1).to(dtype=state.dtype)  # (B, 1, *grid)
        x = self._normalize_state(state, mask)
        if self.encode_geometry and getattr(self, "geometry_branch", None) is None:
            geom_block = self._geometry_channels(
                geometry, geom_features, state.shape[0], state.shape[2:], state.dtype
            )
            x = torch.cat([x, geom_block], dim=1)
        return x

    # -- geometry branch --------------------------------------------------- #

    def _branch_features(
        self,
        geometry: torch.Tensor,
        geom_features: torch.Tensor | None,
        state: torch.Tensor,
    ) -> list[torch.Tensor] | None:
        """The 4 **unfolded** geometry-branch features on the padded grid.

        ``None`` when the host has no branch (the default path). Otherwise the
        raw geometry block ``[mask, (sdf, grad-sdf)]`` is assembled exactly as
        the folded path assembles it, zero-padded up to the crop multiple (so the
        features line up with the padded state), and run through the branch.

        A **frozen** branch (every parameter ``requires_grad=False`` -- the DFT
        setting, where the geometry is fixed for a whole rollout) is evaluated
        once and cached on the geometry tensor's identity + the batch/grid shape.
        While the branch is trainable (AE pre-training) it is always recomputed:
        the cached features would be stale after the first optimizer step and
        would detach the branch from the graph."""
        branch = getattr(self, "geometry_branch", None)
        if branch is None:
            return None

        # Key the cache on the tensors the CALLER handed us: `_expand_geometry`
        # can return a fresh view, which would make an identity key miss every
        # time. `geom_features` is part of the key too -- it also feeds the block.
        cache_key = (geometry, geom_features)
        shape_key = (state.shape[0],) + tuple(state.shape[2:]) + (state.dtype,)
        trainable = any(p.requires_grad for p in branch.parameters())
        if not trainable:
            cached = self._branch_cache
            if cached is not None:
                cached_key, cached_shape, feats = cached
                if (
                    cached_key[0] is geometry
                    and cached_key[1] is geom_features
                    and cached_shape == shape_key
                ):
                    return feats

        geometry = self._expand_geometry(geometry, state)
        block = self._geometry_channels(
            geometry, geom_features, state.shape[0], state.shape[2:], state.dtype
        )
        block, _ = self._pad_to_crop_multiple(block)
        features: list[torch.Tensor] = branch(block)
        if not trainable:
            self._branch_cache = (cache_key, shape_key, features)
        return features

    def _fold_geom_feats(
        self, feats: list[torch.Tensor], n_channels: int
    ) -> list[torch.Tensor]:
        """Fold branch features into the encoder's crop batch.

        Each ``(B, F, X, Y, Z)`` level is tiled exactly like the state fold
        ``"B C (U Xc) (V Yc) (W Zc) -> (B C U V W) 1 Xc Yc Zc"`` -- same
        ``(B C U V W)`` ordering -- but with the crop size divided by that
        level's stride (read off level 0, which is at stride 1) and the feature
        dimension kept. The single geometry feature is shared by all
        ``n_channels`` folded state channels, so it is expanded over ``C``:
        result ``(B*C*U*V*W, F, Xc/s, Yc/s, Zc/s)``."""
        from einops import rearrange

        cs = self.encoder_crop_size
        grid = feats[0].shape[2]
        folded: list[torch.Tensor] = []
        for f in feats:
            stride = max(grid // f.shape[2], 1)
            crop = max(cs // stride, 1)
            u = max(f.shape[2] // crop, 1)
            v = max(f.shape[3] // crop, 1)
            w = max(f.shape[4] // crop, 1)
            t = rearrange(
                f, "B F (U Xc) (V Yc) (W Zc) -> B U V W F Xc Yc Zc", U=u, V=v, W=w
            )
            t = t.unsqueeze(1).expand(-1, n_channels, *((-1,) * (t.dim() - 1)))
            folded.append(t.reshape(-1, *t.shape[5:]))
        return folded

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
