"""Whole-domain and halo processing without changing Tadpole's module tree.

Channels are folded into the batch independently. Halo encoders contribute only
their central latent cells; decoders read neighboring cells from the assembled
(optionally evolved) latent grid. Overlap therefore supplies context on both
sides of the bottleneck. This is a finite-context approximation, not numerical
equivalence to whole-domain processing with its attention/normalization context.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Any, Sequence

import torch

from .tadpole_skip_mixing import TadpoleSkipMixing

STRIDE = 16


def validate_spatial_mode(mode: str, halo_size: int) -> None:
    if mode not in ("local", "global", "halo"):
        raise ValueError("spatial_mode must be 'local', 'global', or 'halo'")
    if mode == "halo" and (
        isinstance(halo_size, bool)
        or not isinstance(halo_size, int)
        or halo_size < 0
        or halo_size % STRIDE
    ):
        raise ValueError("halo_size must be a nonnegative multiple of 16")


@dataclass
class Region:
    core: tuple[slice, ...]
    outer: tuple[slice, ...]

    def slices(self, stride: int = 1, *, relative: bool = False) -> tuple[Any, ...]:
        source = self.core if relative else self.outer
        return (Ellipsis,) + tuple(
            slice(
                (s.start - (o.start if relative else 0)) // stride,
                (s.stop - (o.start if relative else 0)) // stride,
            )
            for s, o in zip(source, self.outer)
        )

    def destination(self, stride: int = 1) -> tuple[Any, ...]:
        return (Ellipsis,) + tuple(
            slice(s.start // stride, s.stop // stride) for s in self.core
        )


@dataclass
class SpatialResiduals:
    """Patch-aligned DFT skips for the full-grid latent returned by encode."""

    regions: list[Region]
    skips: list[list[list[torch.Tensor]]]


def regions_for(shape: Sequence[int], mode: str, crop: int, halo: int) -> list[Region]:
    if mode == "global":
        whole = tuple(slice(0, n) for n in shape)
        return [Region(whole, whole)]
    if mode == "local":
        halo = 0
    return [
        Region(
            tuple(slice(s, s + crop) for s in start),
            tuple(
                slice(max(0, s - halo), min(n, s + crop + halo))
                for s, n in zip(start, shape)
            ),
        )
        for start in product(*(range(0, n, crop) for n in shape))
    ]


def _geom_patch(
    features: list[torch.Tensor] | None, region: Region, start: int, stop: int
) -> list[torch.Tensor] | None:
    if features is None:
        return None
    return [
        f[start:stop][region.slices(stride)].contiguous()
        for f, stride in zip(features, (1, 2, 4, STRIDE))
    ]


def encode_spatial(
    model: Any,
    x: torch.Tensor,
    mode: str,
    crop: int,
    halo: int,
    features: list[torch.Tensor] | None = None,
    *,
    dft: bool = False,
    latent_type: str | None = None,
    return_kl: bool = False,
) -> tuple[torch.Tensor, SpatialResiduals, torch.Tensor | None]:
    """Encode expanded patches and gather one latent per central grid cell."""
    folded = x.flatten(0, 1).unsqueeze(1)
    regions = regions_for(x.shape[2:], mode, crop, halo)
    latent_grid: torch.Tensor | None = None
    kl_grid: torch.Tensor | None = None
    all_skips = []
    limit = model.max_internal_batchsize or folded.shape[0]
    for region in regions:
        latents, kls, skips = [], [], []
        for start in range(0, folded.shape[0], limit):
            stop = start + limit
            patch = folded[start:stop][region.slices()].contiguous()
            geom = _geom_patch(features, region, start, stop)
            encoded = model.encoder(patch, latent_type="distribution", geom_feats=geom)
            if dft:
                dist, residuals = encoded
                skips.append(residuals)
            else:
                dist = encoded
            kind = latent_type or model.latent_type
            if kind not in ("sample", "mode"):
                raise ValueError("latent_type must be 'sample' or 'mode'")
            latent = dist.sample() if kind == "sample" else dist.mode()
            latents.append(latent[region.slices(STRIDE, relative=True)])
            if return_kl:
                kls.append(dist.kl_elem()[region.slices(STRIDE, relative=True)])
        core_latent = torch.cat(latents)
        if latent_grid is None:
            grid_shape = tuple(n // STRIDE for n in x.shape[2:])
            latent_grid = core_latent.new_zeros(*core_latent.shape[:2], *grid_shape)
            if return_kl:
                kl_grid = torch.zeros_like(latent_grid)
        latent_grid[region.destination(STRIDE)] = core_latent
        if return_kl:
            assert kl_grid is not None
            kl_grid[region.destination(STRIDE)] = torch.cat(kls)
        if dft:
            all_skips.append(
                [
                    [
                        torch.cat([chunk[level][i] for chunk in skips])
                        for i in range(len(skips[0][level]))
                    ]
                    for level in range(len(skips[0]))
                ]
            )
    assert latent_grid is not None
    return latent_grid, SpatialResiduals(regions, all_skips), kl_grid


def decode_spatial(
    model: Any,
    latent: torch.Tensor,
    mode: str,
    crop: int,
    halo: int,
    features: list[torch.Tensor] | None = None,
    *,
    residuals: SpatialResiduals | None = None,
    zero_skips: bool = False,
    skip_mixing: TadpoleSkipMixing | None = None,
) -> torch.Tensor:
    """Decode with neighboring latent context and keep only each central core."""
    shape = tuple(n * STRIDE for n in latent.shape[2:])
    regions = (
        residuals.regions
        if residuals is not None
        else regions_for(shape, mode, crop, halo)
    )
    output = latent.new_zeros(latent.shape[0], 1, *shape)
    limit = model.max_internal_batchsize or latent.shape[0]
    for i, region in enumerate(regions):
        additions = (
            skip_mixing(residuals.skips[i])
            if skip_mixing is not None and residuals is not None and not zero_skips
            else None
        )
        pieces = []
        for start in range(0, latent.shape[0], limit):
            stop = start + limit
            patch = latent[start:stop][region.slices(STRIDE)].contiguous()
            geom = _geom_patch(features, region, start, stop)
            if residuals is None:
                decoded = model.decoder(patch, geom_feats=geom)
            else:
                skips = [
                    [
                        (
                            torch.zeros_like(t[start:stop])
                            if zero_skips
                            else t[start:stop]
                        )
                        for t in level
                    ]
                    for level in residuals.skips[i]
                ]
                extra = {}
                if additions is not None:
                    extra["skip_additions"] = [
                        [None if t is None else t[start:stop] for t in level]
                        for level in additions
                    ]
                decoded = model.decoder(patch, skips, geom_feats=geom, **extra)
            pieces.append(decoded[region.slices(relative=True)])
        output[region.destination()] = torch.cat(pieces)
    return output
