"""Per-patch PyTorch dataset for the domain-decomposed training objective.

This module exposes :class:`PatchTransitionDataset`, a sibling of
:class:`neural_surrogates.data.TransitionDataset` that reads the **same**
on-disk ``training_data/`` split (see ``docs/training_data.md``) but yields
one sample *per spatial patch* of a two-level overlapping domain
decomposition (companion PDF §2 + §5, Eq 9). It subclasses
``TransitionDataset`` purely to reuse its file-walking, parameter loading,
geometry resolution and lazy two-snapshot NetCDF reads -- it is its own
class in its own file so the plain transition dataset stays uncluttered.

Each ``__getitem__`` restricts a single ``(t, t+K)`` transition to one patch
``p`` of a :class:`~neural_surrogates.decomposition.DomainDecomposition` and
returns the fine extended block, its interior delta target, the patch
geometry, positional encoding, neighbour indices and the (global, per-
``(traj, t)``) coarse fields needed for the coarse supervision term.

Scope
-----
``K = 1`` is the supported and tested path. A ``K > 1`` patch pushforward is
**not** modelled correctly here: rolling a patch forward in time needs the
full PoU merge (a patch's halo at step ``t+1`` depends on its neighbours'
interiors), which only the model owns. ``pushforward_steps`` is still
accepted and the ``t``/``t+K`` delta is produced for completeness, but the
``delta_target`` is then simply ``interior(S_{t+K}) - interior(S_t)`` and
carries the caveat above.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import torch

from neural_surrogates.data import TransitionDataset
from neural_surrogates.decomposition import DomainDecomposition


class PatchTransitionDataset(TransitionDataset):
    """Per-patch transition samples over a ``training_data/`` split.

    Flattens every trajectory in a split into ``(traj, t, patch)`` triples:
    a trajectory with ``T`` saved steps and ``M`` patches per grid
    contributes ``(T - K) * M`` samples, and the dataset length is the sum
    across all trajectories in ``<root>/state/<split>/``. The patch count
    ``M`` is taken from ``self.dd.num_patches`` on the split's spatial grid
    (all trajectories in a split share the grid; this is asserted).

    Parameters
    ----------
    root_dir, split, state_vars, param_vars, geometry_var, cache, dtype,
    pushforward_steps:
        Forwarded verbatim to :class:`TransitionDataset`.
    decomposition:
        Either a ready :class:`DomainDecomposition` or a ``dict`` of its
        constructor kwargs (e.g. ``{"interior_size": 4, "halo": 2, ...}``).

    Item schema (``__getitem__`` returns a ``dict[str, torch.Tensor]``)::

        state_n_patch  (C, n+2h, n+2h, n+2h)   R_p S_t           extended block
        delta_target   (C, n,    n,    n)       interior(S_{t+K} - S_t)
        geometry_patch (1, n+2h, n+2h, n+2h)    R_p m
        positional     (n_pos, n+2h, n+2h, n+2h) e_p
        coarse_input   (C, *coarse_grid)        R_H S_t
        coarse_geom    (1, *coarse_grid)        R_H m
        coarse_target  (C, *coarse_grid)        R_H S_{t+K}
        neighbors      (6,) long                dd.neighbor_indices()[p]
        params_n       (K, P)                   params at t .. t+K-1
        patch_index    () long                  p
        traj_index     () long                  traj

    where ``C = len(state_vars)``, ``n = interior_size``, ``h = halo`` and
    ``coarse_grid`` is the coarsened spatial shape. The coarse fields are
    global (identical for every patch of a ``(traj, t)``) but ``r**3``
    smaller, so returning them per item is cheap.

    The interior delta is computed as the **centre crop** ``[h : h+n]`` of
    the extended blocks of ``S_{t+K}`` and ``S_t`` -- the centre of an
    extended block is exactly the patch interior -- which avoids a second
    restriction operator.
    """

    def __init__(
        self,
        root_dir: str | Path,
        split: str,
        state_vars: Sequence[str] = ("u", "v", "w"),
        param_vars: Sequence[str] | None = None,
        geometry_var: str | None = "blanking",
        cache: bool = False,
        dtype: torch.dtype = torch.float32,
        pushforward_steps: int = 1,
        decomposition: DomainDecomposition | dict | None = None,
    ) -> None:
        super().__init__(
            root_dir=root_dir,
            split=split,
            state_vars=state_vars,
            param_vars=param_vars,
            geometry_var=geometry_var,
            cache=cache,
            dtype=dtype,
            pushforward_steps=pushforward_steps,
        )

        if decomposition is None:
            self.dd = DomainDecomposition()
        elif isinstance(decomposition, DomainDecomposition):
            self.dd = decomposition
        elif isinstance(decomposition, dict):
            self.dd = DomainDecomposition(**decomposition)
        else:
            raise TypeError(
                "decomposition must be a DomainDecomposition, a dict of its "
                f"kwargs, or None; got {type(decomposition)!r}"
            )

        # Derive the per-trajectory patch count from the split's spatial grid.
        # All trajectories in a split share the grid (the case fixes the
        # domain); assert it so a mismatched split fails loudly rather than
        # silently mis-indexing.
        self._grid = self._read_spatial_grid(self._state_files[0])
        for state_path in self._state_files[1:]:
            grid = self._read_spatial_grid(state_path)
            if grid != self._grid:
                raise ValueError(
                    f"trajectory {state_path.name} has spatial grid {grid}, "
                    f"expected {self._grid} (all trajectories in a split must "
                    "share the grid)"
                )

        probe = torch.zeros(1, 1, *self._grid, dtype=self.dtype)
        self._num_patches = int(self.dd.num_patches(probe))
        self._neighbors = self.dd.neighbor_indices(probe)

        # Re-flatten the base (traj, t) index into (traj, t, patch).
        self._patch_index: list[tuple[int, int, int]] = [
            (traj, t, p)
            for (traj, t) in self._index
            for p in range(self._num_patches)
        ]

    def _read_spatial_grid(self, state_path: Path) -> tuple[int, int, int]:
        """``(Nz, Ny, Nx)`` of the first state variable in ``state_path``."""
        import xarray as xr

        with xr.open_dataset(state_path) as ds:
            da = ds[self.state_vars[0]].isel(time=0)
        return tuple(int(s) for s in da.shape[-3:])

    @property
    def num_patches(self) -> int:
        """Patches per trajectory ``M`` for this split's grid."""
        return self._num_patches

    def __len__(self) -> int:
        return len(self._patch_index)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        traj, t, p = self._patch_index[idx]
        K = self.pushforward_steps
        n = self.dd.interior_size
        h = self.dd.halo

        ds = self._get_state_ds(traj)
        snap = ds.isel(time=[t, t + K])
        # (2, C, Nz, Ny, Nx): time endpoints stacked on axis 0, vars on axis 1.
        channels = np.stack(
            [np.asarray(snap[v].values) for v in self.state_vars], axis=1
        )
        pair = torch.from_numpy(channels).to(self.dtype)
        state_t = pair[0:1]  # (1, C, Nz, Ny, Nx)
        state_tk = pair[1:2]  # (1, C, Nz, Ny, Nx)

        geom = self._geometry.to(self.dtype).view(1, 1, *self._grid)

        # Restrict to the extended-block batch (B=1 -> M blocks) and pick p.
        ext_t = self.dd.restrict(state_t)[p]  # (C, n+2h, n+2h, n+2h)
        ext_tk = self.dd.restrict(state_tk)[p]
        geom_patch = self.dd.restrict(geom)[p]  # (1, n+2h, n+2h, n+2h)

        # Interior is the centre crop [h : h+n] of the extended block.
        interior_t = ext_t[:, h : h + n, h : h + n, h : h + n]
        interior_tk = ext_tk[:, h : h + n, h : h + n, h : h + n]
        delta_target = interior_tk - interior_t  # (C, n, n, n)

        positional = self.dd.positional(state_t)[p]  # (n_pos, n+2h, n+2h, n+2h)

        coarse_input = self.dd.restrict_coarse(state_t)[0]  # (C, *coarse)
        coarse_target = self.dd.restrict_coarse(state_tk)[0]
        coarse_geom = self.dd.restrict_coarse_geom(geom)[0]  # (1, *coarse)

        return {
            "state_n_patch": ext_t,
            "delta_target": delta_target,
            "geometry_patch": geom_patch,
            "positional": positional,
            "coarse_input": coarse_input,
            "coarse_geom": coarse_geom,
            "coarse_target": coarse_target,
            "neighbors": self._neighbors[p],
            "params_n": self._params[traj][t : t + K],
            "patch_index": torch.tensor(p, dtype=torch.long),
            "traj_index": torch.tensor(traj, dtype=torch.long),
        }
