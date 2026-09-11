"""Parameter-history-conditioned snapshot dataset (plan 07, generative spin-up).

:class:`SnapshotHistoryDataset` is :class:`~neural_surrogates.datasets.snapshot.SnapshotDataset`
plus the physical conditioning a conditional generator needs: every snapshot at
trajectory time ``t`` ships the ``Hp`` (``param_history_steps``) parameter rows
at the saved times ``t-Hp+1 … t`` as ``params_hist`` ``(Hp, P)``, **oldest
first** and ending at the snapshot's own time. A generator trained on these
pairs can later synthesise a developed flow state for ESMDA cold starts from a
member's forcing history alone.

Because the conditioning is *physical* (a parameter row per saved time), the
dataset is strict where the transition datasets are lenient:

* state and parameter files are paired by sample id (the ``sample_XXXX`` stem),
  never by sorted position -- a missing partner on either side raises;
* the two files' ``time`` coordinates must agree per sample, be strictly
  increasing and be finite (as must the parameter values themselves);
* the saved cadence must be consistent: the median ``dt`` across every
  trajectory is stored as ``history_dt_seconds`` and every ``dt`` must lie
  within ``cadence_rtol`` of it. Real corpora are *slightly* non-uniform
  (``pyudales_idealized`` saves at 0, 4.85, 9.92, 15.00, ... s), so an exact
  check would reject valid data, while a loose one would let a corpus with a
  mixed cadence train a generator whose ``Hp`` rows span an ill-defined
  duration.

``param_vars`` is required and *ordered*: the column order of ``params_hist``
is the order given, which the generator artifact records as its conditioning
schema so deployment can never substitute a different convention.

Anchors start at ``t = Hp-1`` so every history is fully recorded. With
``constant_prehistory=True`` anchors start at ``t = 0`` and the missing leading
rows repeat the first recorded parameter row -- valid **only** when the data's
provenance guarantees the forcing was constant at those values before the
first saved time (e.g. a constant-forcing spin-up that ends exactly at the
first saved state); the training config carries the flag so the choice is
explicit and auditable. ``time_stride`` thins the *anchors* only; histories
always use contiguous saved steps.

Per-trajectory ``_params`` ``(T, P)`` tables are kept exactly as
``TransitionDataset`` keeps them, so ``training/data_utils.get_normalization_stats``
yields parameter mean/std over all saved times unchanged; ``sample_index``,
``grid_shape`` and the geometry dedup are inherited, so
:class:`~neural_surrogates.datasets.sampler.TrajectoryBatchSampler` buckets
multi-geometry splits as before.

Random cropping is deliberately unsupported (``random_crop_size`` must be
``None``): a global latent generator trained on relocated crops needs its own
assessment of coordinates and global correlations (plan 07 defers it).
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import torch
import xarray as xr
from neural_surrogates.datasets._params import load_param_table
from neural_surrogates.datasets.snapshot import SnapshotDataset, snapshot_collate


def snapshot_history_collate(batch: list[dict]) -> dict:
    """Collate history items: shared geometry (+ SDF) once, ``params_hist`` stacked.

    Identical to :func:`~neural_surrogates.datasets.snapshot.snapshot_collate`
    (which it delegates to): a batch whose items share one geometry tensor
    ships it once as ``(1, *grid)`` / ``(1, C, *grid)``; every other key --
    ``state`` and ``params_hist`` -- goes through the default collate, so
    ``params_hist`` arrives as ``(B, Hp, P)``.
    """
    return snapshot_collate(batch)


class SnapshotHistoryDataset(SnapshotDataset):
    """Single snapshots plus their ``(Hp, P)`` parameter history.

    Each item is a ``dict`` of ``torch.Tensor``: ``state`` ``(C, *grid)``,
    ``geometry`` ``(*grid,)``, optional ``geom_features`` ``(C, *grid)`` (all as
    in :class:`SnapshotDataset`) and ``params_hist`` ``(Hp, P)`` -- the
    parameter rows at saved times ``t-Hp+1 … t``, oldest first, columns in
    ``param_vars`` order. Scalar (static) parameters are broadcast across the
    history.

    ``param_vars`` and ``param_history_steps`` are required (``None`` raises)
    but keep default-valued positions so Hydra / keyword construction reads
    like the sibling datasets.
    """

    def __init__(
        self,
        root_dir: str | Path,
        split: str,
        state_vars: Sequence[str] = ("u", "v", "w"),
        param_vars: Sequence[str] | None = None,
        param_history_steps: int | None = None,
        geometry_var: str | None = "blanking",
        cache: bool = False,
        dtype: torch.dtype = torch.float32,
        time_stride: int = 1,
        random_crop_size: int | None = None,
        sdf_features: bool | str = "none",
        sdf_clamp_cells: float = 32.0,
        cadence_rtol: float = 0.05,
        constant_prehistory: bool = False,
    ) -> None:
        if param_vars is None or len(tuple(param_vars)) == 0:
            raise ValueError(
                "SnapshotHistoryDataset needs an explicit, ordered param_vars "
                "(the conditioning schema); got none"
            )
        if param_history_steps is None or int(param_history_steps) < 1:
            raise ValueError(
                f"param_history_steps must be an int >= 1, got {param_history_steps}"
            )
        if random_crop_size is not None:
            raise NotImplementedError(
                "SnapshotHistoryDataset does not support random_crop_size (plan 07 "
                "trains the generator on full snapshots); pass None"
            )
        if cadence_rtol < 0:
            raise ValueError(f"cadence_rtol must be >= 0, got {cadence_rtol}")
        super().__init__(
            root_dir,
            split,
            state_vars=state_vars,
            geometry_var=geometry_var,
            cache=cache,
            dtype=dtype,
            time_stride=time_stride,
            random_crop_size=None,
            sdf_features=sdf_features,
            sdf_clamp_cells=sdf_clamp_cells,
        )
        self.param_history_steps = int(param_history_steps)
        self.cadence_rtol = float(cadence_rtol)
        self.constant_prehistory = bool(constant_prehistory)

        # -- pair state/param files by sample id, never by sorted position ---- #
        param_dir = self.root / "param" / split
        if not param_dir.is_dir():
            raise FileNotFoundError(f"missing param split dir: {param_dir}")
        by_stem = {p.stem: p for p in param_dir.glob("sample_*.nc")}
        state_stems = [p.stem for p in self._state_files]
        missing = [s for s in state_stems if s not in by_stem]
        if missing:
            raise FileNotFoundError(
                f"split '{split}' under {self.root}: state samples without a "
                f"param partner: {missing}"
            )
        extra = sorted(set(by_stem) - set(state_stems))
        if extra:
            raise ValueError(
                f"split '{split}' under {self.root}: param samples without a "
                f"state partner: {extra}"
            )
        self._param_files: list[Path] = [by_stem[s] for s in state_stems]

        # -- per-sample validation: times, params, lengths ------------------- #
        wanted = tuple(param_vars)
        self.param_names = ()
        self._params = []
        self._times: list[np.ndarray] = []
        hp = self.param_history_steps
        min_len = 1 if self.constant_prehistory else hp
        for state_path, param_path, t_len in zip(
            self._state_files, self._param_files, self._traj_lengths
        ):
            t_state = self._read_time_coord(state_path)
            t_param = self._read_time_coord(param_path)
            if t_state.shape != t_param.shape or not np.allclose(
                t_state, t_param, rtol=1e-6, atol=1e-6
            ):
                raise ValueError(
                    f"time coordinates differ between {state_path.name} "
                    f"({t_state.shape[0]} steps) and its param file "
                    f"({t_param.shape[0]} steps); state/param files must be "
                    "saved on identical times"
                )
            if not np.all(np.isfinite(t_state)) or not np.all(np.diff(t_state) > 0):
                raise ValueError(
                    f"time coordinate of {state_path.name} must be finite and "
                    "strictly increasing"
                )
            if t_len < min_len:
                raise ValueError(
                    f"trajectory {state_path.name} has {t_len} time steps; need "
                    f"at least param_history_steps = {hp} (or set "
                    "constant_prehistory=True when provenance allows it)"
                )
            params, names = load_param_table(param_path, t_len, wanted, self.dtype)
            if not self.param_names:
                self.param_names = names
            elif names != self.param_names:
                raise ValueError(
                    f"param variable set differs between samples: "
                    f"{self.param_names} vs {names} (in {param_path})"
                )
            if not torch.isfinite(params).all():
                bad = [
                    name
                    for name, col in zip(names, params.unbind(-1))
                    if not torch.isfinite(col).all()
                ]
                raise ValueError(
                    f"non-finite parameter values in {param_path.name}: {bad}"
                )
            self._params.append(params)
            self._times.append(t_state)

        # -- consistent saved cadence across the whole split ----------------- #
        self.history_dt_seconds = self._validate_cadence()

        # Anchors: every history must be fully recorded unless the caller
        # vouches for a constant pre-history. ``time_stride`` thins anchors only.
        start = 0 if self.constant_prehistory else hp - 1
        self._index = [
            (traj, t)
            for traj, t_len in enumerate(self._traj_lengths)
            for t in range(start, t_len, self.time_stride)
        ]

    # -- validation helpers ------------------------------------------------ #

    @staticmethod
    def _read_time_coord(path: Path) -> np.ndarray:
        with xr.open_dataset(path) as ds:
            if "time" not in ds.coords:
                raise ValueError(
                    f"{path.name} has no 'time' coordinate; the history dataset "
                    "pairs state and params by physical time and cannot align "
                    "by index"
                )
            return np.asarray(ds["time"].values, dtype=np.float64)

    def _validate_cadence(self) -> float:
        """Median saved ``dt`` across all trajectories; every ``dt`` must lie
        within ``cadence_rtol`` of it (raises naming the offending sample and
        step otherwise)."""
        dts = [np.diff(t) for t in self._times if t.shape[0] > 1]
        if not dts:
            raise ValueError(
                "cannot determine the saved cadence: no trajectory has more "
                "than one time step"
            )
        median = float(np.median(np.concatenate(dts)))
        if not median > 0:
            raise ValueError(f"saved cadence must be positive, got median dt {median}")
        tol = self.cadence_rtol * median
        for traj, dt in zip(
            (i for i, t in enumerate(self._times) if t.shape[0] > 1), dts
        ):
            off = np.flatnonzero(np.abs(dt - median) > tol)
            if off.size:
                k = int(off[0])
                raise ValueError(
                    f"inconsistent saved cadence in {self._state_files[traj].name}: "
                    f"dt[{k}] = {dt[k]:.6g} s (between t[{k}] and t[{k + 1}]) is "
                    f"more than {self.cadence_rtol:g} away from the median "
                    f"{median:.6g} s"
                )
        return median

    # -- items ------------------------------------------------------------- #

    def params_hist_for(self, traj: int, t: int) -> torch.Tensor:
        """``(Hp, P)`` parameter rows at saved steps ``t-Hp+1 … t`` (oldest
        first). With ``constant_prehistory`` the rows before step 0 repeat the
        first recorded row."""
        hp = self.param_history_steps
        table = self._params[traj]
        t0 = t - hp + 1
        if t0 >= 0:
            return table[t0 : t + 1]
        if not self.constant_prehistory:
            raise IndexError(
                f"anchor t={t} of trajectory {traj} has only {t + 1} recorded "
                f"steps of the {hp}-step history"
            )
        pad = table[0:1].expand(-t0, table.shape[1])
        return torch.cat([pad, table[: t + 1]], dim=0)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        item = super().__getitem__(idx)
        traj, t = self._index[idx]
        item["params_hist"] = self.params_hist_for(traj, t)
        return item
