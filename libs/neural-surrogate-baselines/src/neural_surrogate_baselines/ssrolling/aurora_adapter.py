"""3DSwinUrbanNet (Park, Kim, Tsiotras & Lee, Phys. Fluids 38, 045136, 2026):
Microsoft Aurora behind our stepper contract
``forward(state, params, geometry, geom_features=None) -> (B, C, nz, ny, nx)``.

The paper's model is Aurora with AuroraSmall's depths and heads and
``embed_dim = 512`` (451 M parameters with Aurora's default LoRA). It takes two
snapshots and predicts the next one directly, not an increment. Aurora is used
unpatched; this module maps our fields to its ``Batch`` and back:

* atmospheric variables: the state channels and the 3D solid mask, one Aurora
  level per cell layer (levels ``1 … nz``); the predicted mask is dropped;
* surface and static variable: the building-height map, the solid fraction of
  each column (Aurora needs at least one surface variable);
* metadata: pseudo-degrees at a fixed scale (see ``_DEG_PER_CELL``) and a
  constant time and lead time, so they carry no information beyond position.

Not specified by the paper, chosen here:

* inflow parameters (the papers have none): an MLP embedding added to the
  backbone's conditioning vector ``backbone.time_mlp(lead time)``, which drives
  the adaptive LayerNorm of every Swin block;
* z-score normalisation with our training-split statistics, zero inside
  buildings. Aurora's own normalisation runs with identity statistics,
  registered under prefixed names (``nsb_*``) so its tables for weather
  variables are never touched;
* the lateral boundaries. Aurora's shifted windows wrap around its W axis
  (``warped=True`` is hard-coded), so a periodic axis is mapped to W (y by
  default, then H = x); without one the wrap is switched off per block. H and W
  are padded up to a multiple of the patch size (circular along a periodic
  axis, else by repeating the edge) and the prediction is cropped back.
  Aurora's wrap only connects the two edges of a stage whose patch count along
  W is a multiple of the window width (12); at other widths its padding keeps
  the wrapped columns apart, as for a non-periodic axis.
"""

from __future__ import annotations

import functools
from datetime import datetime
from typing import Any, Sequence

import numpy as np
import numpy.typing as npt
import torch
import torch.nn.functional as F
from torch import nn

# Our variables in Aurora's batch, prefixed so they never clash with its own.
_SOLID = "nsb_solid"
_HEIGHT = "nsb_height"
_HEIGHT_STATIC = "nsb_hstatic"

# Pseudo-coordinates, the same for every grid: cell (i, j) of Aurora's (H, W)
# sits at lat = _ORIGIN_DEG + (H - 1 - i) * _DEG_PER_CELL (decreasing, as
# Aurora requires) and lon = _ORIGIN_DEG + j * _DEG_PER_CELL. Aurora encodes
# patch centres with wavelengths 0.01°–720° (`pos_expansion`, values must be 0
# or ≥ 0.01 in magnitude) and the root area of each patch from 1e-4 km
# upward (`scale_expansion`). At 0.01° per cell, neighbouring 4-cell patches
# are 0.04° apart, resolved by the shortest wavelength, and a patch's root
# area is about 6 km. Starting at 1° keeps every centre above the 0.01° floor
# and near the equator, where the patch area hardly changes (< 0.2 % over 256
# cells). Grids of up to ~8900 cells per axis stay within ±90°.
_DEG_PER_CELL = 0.01
_ORIGIN_DEG = 1.0
_TIME = datetime(2000, 1, 1)


class UrbanAurora(nn.Module):
    def __init__(
        self,
        n_state_channels: int,
        n_params: int,
        num_history_steps: int = 2,
        embed_dim: int = 512,
        num_heads: int = 16,
        encoder_depths: Sequence[int] = (2, 6, 2),
        encoder_num_heads: Sequence[int] = (4, 8, 16),
        decoder_depths: Sequence[int] = (2, 6, 2),
        decoder_num_heads: Sequence[int] = (16, 8, 4),
        window_size: Sequence[int] = (2, 6, 12),
        patch_size: int = 4,
        latent_levels: int = 4,
        use_lora: bool = True,
        periodic_axes: Sequence[str] = ("y",),
    ) -> None:
        super().__init__()
        # Imported here so the package imports without Aurora installed.
        from aurora import Aurora
        from aurora.model.swin3d import Swin3DTransformerBlock

        if int(num_history_steps) != 2:
            raise ValueError(
                f"Aurora takes two input snapshots; got num_history_steps={num_history_steps}"
            )
        periodic = set(periodic_axes)
        if not periodic <= {"y", "x"}:
            raise ValueError(
                f"periodic_axes must be in ('y', 'x'), got {sorted(periodic)}"
            )
        if len(periodic) == 2:
            raise ValueError("Aurora wraps around one lateral axis only")
        self.n_state_channels = int(n_state_channels)
        self.n_params = int(n_params)
        self.num_history_steps = 2
        self.patch_size = int(patch_size)
        self.periodic_y = "y" in periodic
        self.periodic_x = "x" in periodic
        # Aurora's W axis is the one its windows wrap around.
        self.y_on_w = self.periodic_y
        # Any nx, ny at the trained spacing: the windows and patches tile any
        # grid, and the pseudo-coordinates have a fixed scale.
        self.domain_flexible = True
        # The padding and window shapes depend on the grid.
        self.compile_dynamic = False

        self.register_buffer("state_mean", torch.zeros(self.n_state_channels))
        self.register_buffer("state_std", torch.ones(self.n_state_channels))
        self.register_buffer("param_mean", torch.zeros(max(self.n_params, 1)))
        self.register_buffer("param_std", torch.ones(max(self.n_params, 1)))

        self._state_names = tuple(f"nsb_state{i}" for i in range(self.n_state_channels))
        # Aurora's default lead time (6 h) is constant, so it carries no
        # information; its autocast stays off, our trainer's AMP covers it.
        self.aurora = Aurora(
            surf_vars=(_HEIGHT,),
            static_vars=(_HEIGHT_STATIC,),
            atmos_vars=(*self._state_names, _SOLID),
            window_size=tuple(int(w) for w in window_size),
            encoder_depths=tuple(encoder_depths),
            encoder_num_heads=tuple(encoder_num_heads),
            decoder_depths=tuple(decoder_depths),
            decoder_num_heads=tuple(decoder_num_heads),
            latent_levels=int(latent_levels),
            patch_size=self.patch_size,
            embed_dim=int(embed_dim),
            num_heads=int(num_heads),
            max_history_size=2,
            use_lora=use_lora,
        )
        if not periodic:
            for block in self.aurora.modules():
                if isinstance(block, Swin3DTransformerBlock):
                    block.forward = functools.partial(
                        type(block).forward, block, warped=False
                    )

        # The current batch's inflow embedding, set around each Aurora call.
        self._param_embedding: torch.Tensor | None = None
        if self.n_params > 0:
            self.param_embed = nn.Sequential(
                nn.Linear(self.n_params, int(embed_dim)),
                nn.SiLU(),
                nn.Linear(int(embed_dim), int(embed_dim)),
            )
            self.aurora.backbone.time_mlp.register_forward_hook(
                self._add_param_embedding
            )

    def set_normalization(
        self,
        state_mean: npt.ArrayLike,
        state_std: npt.ArrayLike,
        param_mean: npt.ArrayLike,
        param_std: npt.ArrayLike,
    ) -> None:
        """Install training-split statistics; zero stds become 1."""
        pairs = [(self.state_mean, state_mean), (self.state_std, state_std)]
        if self.n_params > 0:
            pairs += [(self.param_mean, param_mean), (self.param_std, param_std)]
        for buffer, values in pairs:
            vals = torch.as_tensor(np.asarray(values), dtype=buffer.dtype).reshape(-1)
            if vals.numel() != buffer.numel():
                raise ValueError(
                    f"expected {buffer.numel()} values, got {vals.numel()}"
                )
            buffer.copy_(vals)
        for std in (self.state_std, self.param_std):
            std.copy_(torch.where(std > 0, std, torch.ones_like(std)))

    def _add_param_embedding(
        self, module: nn.Module, inputs: Any, output: torch.Tensor
    ) -> torch.Tensor:
        assert self._param_embedding is not None, "call the stepper, not its Aurora"
        return output + self._param_embedding.to(output.dtype)

    def _layout(self, x: torch.Tensor) -> torch.Tensor:
        """Our (…, y, x) to Aurora's (…, H, W) and back (its own inverse)."""
        return x.transpose(-1, -2) if self.y_on_w else x

    def _pad(self, x: torch.Tensor) -> torch.Tensor:
        """Pad (…, ny, nx) at the far ends up to multiples of the patch size."""
        py, px = (-x.shape[-2]) % self.patch_size, (-x.shape[-1]) % self.patch_size
        if not (py or px):
            return x
        lead = x.shape[:-2]
        x = x.reshape(-1, 1, *x.shape[-2:])
        x = F.pad(x, (0, px, 0, 0), mode="circular" if self.periodic_x else "replicate")
        x = F.pad(x, (0, 0, 0, py), mode="circular" if self.periodic_y else "replicate")
        return x.reshape(*lead, *x.shape[-2:])

    def _register_identity_stats(self, nz: int) -> None:
        """Identity entries in Aurora's (module-global) statistics tables for
        our variables, adding missing keys only."""
        from aurora.normalisation import level_to_str, locations, scales

        names = (*self._state_names, _SOLID)
        keys = [_HEIGHT, _HEIGHT_STATIC] + [
            f"{name}_{level_to_str(level)}"
            for name in names
            for level in range(1, nz + 1)
        ]
        for key in keys:
            if key not in locations:
                locations[key], scales[key] = 0.0, 1.0

    @staticmethod
    def _fluid_mask(state: torch.Tensor, geometry: torch.Tensor) -> torch.Tensor:
        """``(B, nz, ny, nx)`` fluid mask in the state's dtype."""
        if geometry.dim() == state.dim():  # (B, 1, *grid)
            geometry = geometry[:, 0]
        return geometry.to(state.dtype)

    def predict_normalised(
        self, state: torch.Tensor, params: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        """Aurora's next state ``(B, C, nz, ny, nx)`` in z-scored units, zero
        in buildings (``mask``: ``(B, nz, ny, nx)``, 1 = fluid)."""
        from aurora import Batch, Metadata

        B, _, nz, ny, nx = state.shape
        C, T = self.n_state_channels, self.num_history_steps
        mean = self.state_mean.view(1, 1, C, 1, 1, 1)
        std = self.state_std.view(1, 1, C, 1, 1, 1)
        fluid = mask[:, None, None]
        x = (state.view(B, T, C, nz, ny, nx) - mean) / std * fluid
        x = self._layout(self._pad(x))  # (B, T, C, nz, H, W)
        solid = self._layout(self._pad(1.0 - mask))  # (B, nz, H, W)
        height = solid.mean(dim=1)  # (B, H, W)
        if bool((height != height[:1]).any()):
            raise ValueError(
                "Aurora's static variables are shared by the batch: every member "
                "needs the same geometry"
            )
        H, W = height.shape[-2:]
        self._register_identity_stats(nz)
        lat = _ORIGIN_DEG + _DEG_PER_CELL * torch.arange(
            H - 1, -1, -1, dtype=torch.float64
        )
        lon = _ORIGIN_DEG + _DEG_PER_CELL * torch.arange(W, dtype=torch.float64)
        batch = Batch(
            surf_vars={_HEIGHT: height[:, None].expand(B, T, H, W)},
            static_vars={_HEIGHT_STATIC: height[0]},
            atmos_vars={
                **{name: x[:, :, i] for i, name in enumerate(self._state_names)},
                _SOLID: solid[:, None].expand(B, T, nz, H, W),
            },
            metadata=Metadata(
                lat=lat,
                lon=lon,
                time=(_TIME,) * B,
                atmos_levels=tuple(range(1, nz + 1)),
            ),
        )
        if self.n_params > 0:
            self._param_embedding = self.param_embed(
                (params - self.param_mean) / self.param_std
            )
        try:
            pred = self.aurora(batch)
        finally:
            self._param_embedding = None
        # Each variable is (B, 1, nz, H, W): the one predicted time.
        out = torch.stack(
            [pred.atmos_vars[name][:, 0] for name in self._state_names], 1
        )
        out = self._layout(out)[..., :ny, :nx]
        return out.to(state.dtype) * mask[:, None]

    def _denormalise(self, q: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        mean = self.state_mean.view(1, -1, 1, 1, 1)
        std = self.state_std.view(1, -1, 1, 1, 1)
        return (q * std + mean) * mask[:, None]

    def forward(
        self,
        state: torch.Tensor,
        params: torch.Tensor,
        geometry: torch.Tensor,
        geom_features: torch.Tensor | None = None,
    ) -> torch.Tensor:
        mask = self._fluid_mask(state, geometry)
        return self._denormalise(self.predict_normalised(state, params, mask), mask)
