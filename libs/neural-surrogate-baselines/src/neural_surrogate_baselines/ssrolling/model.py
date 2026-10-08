"""SSRollingUrbanNet (Park & Lee, Phys. Fluids 38, 085167, 2026) as a stepper:
3DSwinUrbanNet (``aurora_adapter.UrbanAurora``) plus the small-scale
generator (``ssgen.SSGen``) added to its prediction. ``ssgen=False`` is the
3DSwinUrbanNet baseline itself.

The paper runs SSGen "in physical space on the unpatchified coarse
prediction", i.e. on the grid rather than on patch tokens; its data is
min-max scaled. We run it on the z-scored grid, where Aurora predicts, with
the buildings zeroed before and after.

SSGen folds height into its channels, so it is built for a fixed number of
vertical levels, ``n_levels`` (default 32: our corpora's 128 m at 4 m). The
model is built before it sees data and must load saved weights at once, so
this is a constructor argument, checked against ``nz`` on every call.
"""

from __future__ import annotations

from typing import Any

import torch
from neural_surrogate_baselines.ssrolling.aurora_adapter import UrbanAurora
from neural_surrogate_baselines.ssrolling.ssgen import SSGen


class SSRollingUrbanNet(UrbanAurora):
    def __init__(
        self,
        n_state_channels: int,
        n_params: int,
        ssgen: bool = True,
        ssgen_channels: int = 64,
        ssgen_gate: float = 0.2,
        ssgen_dropout: float = 0.1,
        n_levels: int = 32,
        **backbone: Any,
    ) -> None:
        """``backbone``: the ``UrbanAurora`` arguments (``embed_dim``,
        ``num_heads``, depths, ``window_size``, ``periodic_axes``, …)."""
        super().__init__(n_state_channels, n_params, **backbone)
        self.n_levels = int(n_levels)
        self.ssgen = (
            SSGen(
                self.n_state_channels * self.n_levels,
                hidden=ssgen_channels,
                gate=ssgen_gate,
                dropout=ssgen_dropout,
            )
            if ssgen
            else None
        )

    def forward(
        self,
        state: torch.Tensor,
        params: torch.Tensor,
        geometry: torch.Tensor,
        geom_features: torch.Tensor | None = None,
    ) -> torch.Tensor:
        mask = self._fluid_mask(state, geometry)
        q = self.predict_normalised(state, params, mask)
        if self.ssgen is not None:
            if q.shape[2] != self.n_levels:
                raise ValueError(
                    f"SSGen is built for n_levels={self.n_levels} vertical levels, "
                    f"the grid has nz={q.shape[2]}: set the architecture's n_levels"
                )
            q = q + self.ssgen(q, 1.0 - mask)
        return self._denormalise(q, mask)
