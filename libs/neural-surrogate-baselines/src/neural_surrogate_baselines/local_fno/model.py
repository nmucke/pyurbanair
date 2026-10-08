"""Local-FNO (Qin et al., Build. Environ. 273, 112668, 2025) as a stepper.

The paper's model, in our stepper contract
``forward(state, params, geometry, geom_features=None) -> (B, C, nz, ny, nx)``:

* input per patch: the two latest frames, the SDF of the buildings and (our
  addition, the paper has none) the inflow parameters as constant channels;
* lifting ``Q`` and projection ``P``: pointwise MLPs with one hidden layer;
* ``n_layers`` Fourier layers ``v <- σ(M(K v) + W v + b + v)`` (their Eq. 5);
* the domain is tiled into overlapping horizontal patches (``patches.py``),
  predicted by one shared network, and stitched from the patch cores;
* the next state is predicted directly, not as an increment (as in the paper).

Not specified by the paper, chosen here: GELU for σ, hidden width
``mlp_ratio * width`` for ``Q``, ``P`` and ``M``, z-score normalisation.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import numpy.typing as npt
import torch
import torch.nn.functional as F
from neural_surrogate_baselines.local_fno.patches import PatchGrid
from neural_surrogate_baselines.local_fno.spectral import SpectralConv3d
from neural_surrogates.sdf import n_sdf_feature_channels, normalize_sdf_mode
from neural_surrogates.sdf import sdf_features as compute_sdf_features
from torch import nn


def _pointwise_mlp(c_in: int, hidden: int, c_out: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv3d(c_in, hidden, kernel_size=1),
        nn.GELU(),
        nn.Conv3d(hidden, c_out, kernel_size=1),
    )


class FourierLayer(nn.Module):
    """``v <- σ(M(K v) + W v + b + v)``."""

    def __init__(self, width: int, modes: tuple[int, int, int], hidden: int) -> None:
        super().__init__()
        self.spectral = SpectralConv3d(width, width, modes)
        self.ffn = _pointwise_mlp(width, hidden, width)
        self.linear = nn.Conv3d(width, width, kernel_size=1)

    def forward(self, v: torch.Tensor) -> torch.Tensor:
        return F.gelu(self.ffn(self.spectral(v)) + self.linear(v) + v)


class LocalFNOStepper(nn.Module):
    def __init__(
        self,
        n_state_channels: int,
        n_params: int,
        num_history_steps: int = 2,
        width: int = 36,
        modes: Sequence[int] = (8, 16, 16),
        n_layers: int = 4,
        mlp_ratio: float = 2.0,
        patch_core: int = 64,
        patch_overlap: int = 6,
        periodic_axes: Sequence[str] = ("y",),
        sdf_features: bool | str = "sdf",
        sdf_clamp_cells: float = 32.0,
    ) -> None:
        super().__init__()
        unknown = set(periodic_axes) - {"y", "x"}
        if unknown:
            raise ValueError(
                f"periodic_axes must be in ('y', 'x'), got {sorted(unknown)}"
            )
        if len(modes) != 3:
            raise ValueError(f"modes must be (z, y, x), got {modes}")
        self.n_state_channels = int(n_state_channels)
        self.n_params = int(n_params)
        self.num_history_steps = int(num_history_steps)
        self.patch_core = int(patch_core)
        self.patch_overlap = int(patch_overlap)
        self.periodic_y = "y" in periodic_axes
        self.periodic_x = "x" in periodic_axes
        self.sdf_feature_mode = normalize_sdf_mode(sdf_features)
        self.sdf_clamp_cells = float(sdf_clamp_cells)
        self.n_geom_feature_channels = n_sdf_feature_channels(self.sdf_feature_mode)
        # Patches make the network independent of the domain size: only the
        # cell spacing must match the training data.
        self.domain_flexible = True
        # The patch count depends on the grid; let inductor specialise per grid.
        self.compile_dynamic = False
        self._sdf_cache: tuple[torch.Tensor, torch.Tensor] | None = None

        self.register_buffer("state_mean", torch.zeros(self.n_state_channels))
        self.register_buffer("state_std", torch.ones(self.n_state_channels))
        self.register_buffer("param_mean", torch.zeros(max(self.n_params, 1)))
        self.register_buffer("param_std", torch.ones(max(self.n_params, 1)))

        hidden = int(round(mlp_ratio * width))
        c_in = (
            self.num_history_steps * self.n_state_channels
            + self.n_geom_feature_channels
            + self.n_params
        )
        self.lifting = _pointwise_mlp(c_in, hidden, width)
        self.layers = nn.ModuleList(
            FourierLayer(width, tuple(int(m) for m in modes), hidden)  # type: ignore[arg-type]
            for _ in range(n_layers)
        )
        self.projection = _pointwise_mlp(width, hidden, self.n_state_channels)

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

    def patch_grid(self, ny: int, nx: int) -> PatchGrid:
        return PatchGrid(
            ny,
            nx,
            self.patch_core,
            self.patch_overlap,
            self.periodic_y,
            self.periodic_x,
        )

    def patchify(self, field: torch.Tensor) -> torch.Tensor:
        """``(B, C, nz, ny, nx)`` -> the matching patches, e.g. for a target."""
        return self.patch_grid(*field.shape[-2:]).extract(field)

    def _sdf_features_for(self, geometry: torch.Tensor) -> torch.Tensor:
        """``(B, C_g, *grid)`` SDF features of the fluid mask, cached.

        The EDT runs once per geometry: the key is a copy of the mask, compared
        by value, so an in-place edit of the caller's mask is never served stale.
        """
        cached = self._sdf_cache
        if cached is not None:
            key, feat = cached
            if (
                key.shape == geometry.shape
                and key.device == geometry.device
                and torch.equal(key, geometry)
            ):
                return feat

        def sdf(g: torch.Tensor) -> torch.Tensor:
            # The EDT needs a float32/64 mask (the model may run in bf16).
            return compute_sdf_features(
                g.float(), clamp_cells=self.sdf_clamp_cells, mode=self.sdf_feature_mode
            )

        # An ensemble stacks identical masks: one EDT for all equal members.
        first = sdf(geometry[0])
        feats = [first] + [
            first if torch.equal(g, geometry[0]) else sdf(g) for g in geometry[1:]
        ]
        feat = torch.stack(feats).to(geometry.device)
        self._sdf_cache = (geometry.clone(), feat)
        return feat

    def forward_patches(
        self,
        state: torch.Tensor,
        params: torch.Tensor,
        geometry: torch.Tensor,
        geom_features: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """The raw per-patch predictions ``(B*n_patches, C, nz, size, size)``
        in physical units, before stitching (the paper's training target)."""
        if geometry.dim() == state.dim():  # (B, 1, *grid)
            geometry = geometry[:, 0]
        mask = geometry.unsqueeze(1).to(state.dtype)
        mean = self.state_mean.repeat(self.num_history_steps).view(1, -1, 1, 1, 1)
        std = self.state_std.repeat(self.num_history_steps).view(1, -1, 1, 1, 1)
        x = (state * mask - mean) / std * mask
        if self.n_geom_feature_channels > 0:
            if geom_features is None:
                geom_features = self._sdf_features_for(geometry)
            x = torch.cat([x, geom_features.to(x.dtype)], dim=1)
        grid = self.patch_grid(*state.shape[-2:])
        x = grid.extract(x)
        if self.n_params > 0:
            p = (params - self.param_mean) / self.param_std
            p = p.repeat_interleave(grid.n_patches, dim=0)
            x = torch.cat(
                [x, p[:, :, None, None, None].expand(-1, -1, *x.shape[2:])], 1
            )
        v = self.lifting(x)
        for layer in self.layers:
            v = layer(v)
        out = self.projection(v)
        return out * self.state_std.view(1, -1, 1, 1, 1) + self.state_mean.view(
            1, -1, 1, 1, 1
        )

    def forward(
        self,
        state: torch.Tensor,
        params: torch.Tensor,
        geometry: torch.Tensor,
        geom_features: torch.Tensor | None = None,
    ) -> torch.Tensor:
        patches = self.forward_patches(state, params, geometry, geom_features)
        out = self.patch_grid(*state.shape[-2:]).stitch(patches)
        if geometry.dim() == state.dim():
            geometry = geometry[:, 0]
        return out * geometry.unsqueeze(1).to(out.dtype)
