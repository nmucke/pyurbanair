"""Datasets of the re-implemented papers' training recipes."""

from __future__ import annotations

import numpy as np
import torch
from neural_surrogates.datasets.transition import TransitionDataset


class RolloutTransitionDataset(TransitionDataset):
    """A :class:`TransitionDataset` whose items also carry every intermediate
    target, for losses on each step of a K-step rollout (SSRollingUrbanNet).

    Each item adds ``state_targets`` ``(K, C, *grid)``: the snapshots at
    ``t+1 … t+K``. ``state_next`` stays the last of them, and every other key
    is unchanged; ``transition_collate`` stacks the new key to
    ``(B, K, C, *grid)``. The history and the targets come from one read.
    """

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        traj, t = self._index[idx]
        K = self.pushforward_steps
        H = self.num_history_steps
        snap = self._get_state_ds(traj).isel(time=slice(t - H + 1, t + K + 1))
        channels = np.stack(
            [np.asarray(snap[v].values) for v in self.state_vars], axis=1
        )
        frames = torch.from_numpy(channels).to(self.dtype)
        item = {
            "state_n": frames[:H].flatten(0, 1),
            "state_next": frames[-1],
            "state_targets": frames[H:],
            "params_n": self._params[traj][t : t + K],
            "geometry": self.geometry_for(traj),
        }
        features = self.geom_features_for(traj)
        if features is not None:
            item["geom_features"] = features
        return item
