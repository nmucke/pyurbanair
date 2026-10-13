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
from torch.utils.checkpoint import checkpoint as _checkpoint

from ._tadpole_crop import CropShape
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

    @property
    def shape(self) -> tuple[int, ...]:
        return tuple(s.stop - s.start for s in self.outer)

    def destination(self, stride: int = 1) -> tuple[Any, ...]:
        return (Ellipsis,) + tuple(
            slice(s.start // stride, s.stop // stride) for s in self.core
        )


# One encoder/decoder call: the region indices it stacks (each with the folded
# batch slice start:stop), in batch order region-major.
Call = tuple[list[int], int, int]


@dataclass
class SpatialResiduals:
    """Call-aligned DFT skips for the full-grid latent returned by encode."""

    regions: list[Region]
    skips: list[list[list[torch.Tensor]]]
    calls: list[Call]


def regions_for(
    shape: Sequence[int], mode: str, crop: CropShape, halo: int
) -> list[Region]:
    if mode == "global":
        whole = tuple(slice(0, n) for n in shape)
        return [Region(whole, whole)]
    if mode == "local":
        halo = 0
    return [
        Region(
            tuple(slice(s, s + size) for s, size in zip(start, crop)),
            tuple(
                slice(max(0, s - halo), min(n, s + size + halo))
                for s, n, size in zip(start, shape, crop)
            ),
        )
        for start in product(*(range(0, n, size) for n, size in zip(shape, crop)))
    ]


def tile_calls(
    regions: list[Region], n: int, limit: int | None, batch_tiles: bool
) -> list[Call]:
    """Partition the (region, folded crop) work into encoder/decoder calls.

    Default: one call per region, its ``n`` folded crops chunked by ``limit``.
    ``batch_tiles`` stacks regions of equal outer shape into one call (all
    ``n`` crops each), at most ``limit // n`` regions per call when ``limit``
    is set: far fewer kernel launches, the same function up to batched-kernel
    rounding.
    """
    if not batch_tiles:
        step = limit or n
        return [
            ([i], start, min(start + step, n))
            for i in range(len(regions))
            for start in range(0, n, step)
        ]
    groups: dict[tuple[int, ...], list[int]] = {}
    for i, region in enumerate(regions):
        groups.setdefault(region.shape, []).append(i)
    per = max(1, limit // n) if limit else len(regions)
    return [
        (idx[k : k + per], 0, n)
        for idx in groups.values()
        for k in range(0, len(idx), per)
    ]


def _call(fn: Any, recompute: bool) -> Any:
    """Run ``fn()``; with ``recompute`` (and autograd on) keep only its inputs
    and outputs and recompute its activations in backward, one call at a time."""
    if recompute and torch.is_grad_enabled():
        return _checkpoint(fn, use_reentrant=False)
    return fn()


def _gather(t: torch.Tensor, call: Call, regions: list[Region], stride: int = 1):  # type: ignore[no-untyped-def]
    """The call's patches of ``t`` (folded batch first), stacked region-major."""
    idx, start, stop = call
    parts = [t[start:stop][regions[i].slices(stride)] for i in idx]
    return (parts[0] if len(parts) == 1 else torch.cat(parts)).contiguous()


def _geom_patch(
    features: list[torch.Tensor] | None, call: Call, regions: list[Region]
) -> list[torch.Tensor] | None:
    if features is None:
        return None
    return [
        _gather(f, call, regions, stride)
        for f, stride in zip(features, (1, 2, 4, STRIDE))
    ]


def encode_spatial(
    model: Any,
    x: torch.Tensor,
    mode: str,
    crop: CropShape,
    halo: int,
    features: list[torch.Tensor] | None = None,
    *,
    dft: bool = False,
    latent_type: str | None = None,
    return_kl: bool = False,
    recompute: bool = False,
    batch_tiles: bool = False,
) -> tuple[torch.Tensor, SpatialResiduals, torch.Tensor | None]:
    """Encode expanded patches and gather one latent per central grid cell.

    ``recompute`` checkpoints each encoder call: training then keeps only each
    call's core latents and skips, not its activations (same values and
    gradients). ``batch_tiles``: see :func:`tile_calls`.
    """
    kind = latent_type or model.latent_type
    if kind not in ("sample", "mode"):
        raise ValueError("latent_type must be 'sample' or 'mode'")
    folded = x.flatten(0, 1).unsqueeze(1)
    regions = regions_for(x.shape[2:], mode, crop, halo)
    calls = tile_calls(
        regions, folded.shape[0], model.max_internal_batchsize, batch_tiles
    )
    latent_grid: torch.Tensor | None = None
    kl_grid: torch.Tensor | None = None
    all_skips = []
    for call in calls:
        idx, start, stop = call
        patch = _gather(folded, call, regions)
        geom = _geom_patch(features, call, regions)
        cores = [regions[i].slices(STRIDE, relative=True) for i in idx]

        def encode(patch=patch, geom=geom, cores=cores, m=stop - start):  # type: ignore[no-untyped-def]
            encoded = model.encoder(patch, latent_type="distribution", geom_feats=geom)
            dist, residuals = encoded if dft else (encoded, None)
            latent = dist.sample() if kind == "sample" else dist.mode()
            kl = dist.kl_elem() if return_kl else None
            return (
                [latent[j * m : (j + 1) * m][c] for j, c in enumerate(cores)],
                (
                    [kl[j * m : (j + 1) * m][c] for j, c in enumerate(cores)]
                    if kl is not None
                    else None
                ),
                residuals,
            )

        latents, kls, residuals = _call(encode, recompute)
        if len(calls) == 1 and len(regions) == 1:
            # One call on one whole-grid region (global mode, or a grid of one
            # tile): its core latent is the grid; skip the zero-fill + copy.
            # Contiguous like the assembled grid (the posterior mode is a view).
            latent_grid = latents[0].contiguous()
            if return_kl:
                assert kls is not None
                kl_grid = kls[0].contiguous()
        elif latent_grid is None:
            grid_shape = tuple(n // STRIDE for n in x.shape[2:])
            latent_grid = latents[0].new_zeros(
                folded.shape[0], latents[0].shape[1], *grid_shape
            )
            if return_kl:
                kl_grid = torch.zeros_like(latent_grid)
        for j, i in enumerate(idx):
            if len(calls) == 1 and len(regions) == 1:
                break
            latent_grid[start:stop][regions[i].destination(STRIDE)] = latents[j]
            if return_kl:
                assert kl_grid is not None and kls is not None
                kl_grid[start:stop][regions[i].destination(STRIDE)] = kls[j]
        if dft:
            all_skips.append(residuals)
    assert latent_grid is not None
    return latent_grid, SpatialResiduals(regions, all_skips, calls), kl_grid


def decode_spatial(
    model: Any,
    latent: torch.Tensor,
    mode: str,
    crop: CropShape,
    halo: int,
    features: list[torch.Tensor] | None = None,
    *,
    residuals: SpatialResiduals | None = None,
    zero_skips: bool = False,
    skip_mixing: TadpoleSkipMixing | None = None,
    recompute: bool = False,
    batch_tiles: bool = False,
) -> torch.Tensor:
    """Decode with neighboring latent context and keep only each central core.

    ``recompute`` / ``batch_tiles``: see :func:`encode_spatial`; with
    ``residuals`` the encoder's call partition is reused.
    """
    shape = tuple(n * STRIDE for n in latent.shape[2:])
    n = latent.shape[0]
    if residuals is not None:
        regions, calls = residuals.regions, residuals.calls
    else:
        regions = regions_for(shape, mode, crop, halo)
        calls = tile_calls(regions, n, model.max_internal_batchsize, batch_tiles)
    # One call on one whole-grid region returns its decoded output directly
    # (no full-resolution zero-fill + copy).
    whole = len(calls) == 1 and len(regions) == 1
    output = None if whole else latent.new_zeros(n, 1, *shape)
    mixed: dict[int, list[list[torch.Tensor | None]]] = {}

    def additions(k: int) -> list[list[torch.Tensor | None]] | None:
        """Skip-mixing corrections for call ``k``; mixing groups each sample's
        channels, so a chunk of one region is mixed over the whole region."""
        if skip_mixing is None or residuals is None or zero_skips:
            return None
        idx, start, stop = calls[k]
        if start == 0 and stop == n:
            full: list[list[torch.Tensor | None]] = skip_mixing(residuals.skips[k])
            return full
        i = idx[0]
        if i not in mixed:
            own = [c for c, call in enumerate(calls) if call[0] == [i]]
            mixed[i] = skip_mixing(
                [
                    [
                        torch.cat([residuals.skips[c][g][lv] for c in own])
                        for lv in range(len(level))
                    ]
                    for g, level in enumerate(residuals.skips[k])
                ]
            )
        return [
            [None if t is None else t[start:stop] for t in level] for level in mixed[i]
        ]

    for k, call in enumerate(calls):
        idx, start, stop = call
        patch = _gather(latent, call, regions, STRIDE)
        geom = _geom_patch(features, call, regions)
        cores = [regions[i].slices(relative=True) for i in idx]
        extra: dict[str, Any] = {}
        skips = None
        if residuals is not None:
            skips = [
                [torch.zeros_like(t) if zero_skips else t for t in level]
                for level in residuals.skips[k]
            ]
            add = additions(k)
            if add is not None:
                extra["skip_additions"] = add

        def decode(patch=patch, geom=geom, skips=skips, extra=extra, cores=cores, m=stop - start):  # type: ignore[no-untyped-def]
            if skips is None:
                decoded = model.decoder(patch, geom_feats=geom)
            else:
                decoded = model.decoder(patch, skips, geom_feats=geom, **extra)
            return [decoded[j * m : (j + 1) * m][c] for j, c in enumerate(cores)]

        pieces = _call(decode, recompute)
        if output is None:
            return pieces[0]
        for j, i in enumerate(idx):
            output[start:stop][regions[i].destination()] = pieces[j]
    assert output is not None
    return output
