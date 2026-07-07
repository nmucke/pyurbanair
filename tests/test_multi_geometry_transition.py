"""Multi-geometry (varying-domain) support in the transition-training stack.

The random-geometry training data gives every trajectory its own grid and
geometry mask. Covers:

* :class:`TransitionDataset` — per-trajectory geometry/SDF with content dedup
  (equal masks share one tensor object, so single-geometry splits behave
  byte-identically to the fixed-domain dataset);
* :func:`transition_collate` — fails loud on a mixed-geometry batch;
* :class:`TrajectoryBatchSampler` — same-trajectory batches, cell-budget batch
  sizing, full coverage, and lazy pickup of ``set_pushforward_steps``;
* :class:`BaseTraining._prepare_batch` — refreshes the device geometry cache
  when the batch's geometry changes (the silent-stale-mask trap);
* ``_refork_loader`` — rebuilds a loader around its custom batch sampler;
* the train script's ``_compute_normalization_stats`` — per-trajectory masks.
"""

from __future__ import annotations

import importlib.util
import math
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

torch = pytest.importorskip("torch")

from neural_surrogates import Trainer, TrajectoryBatchSampler
from neural_surrogates.datasets.transition import TransitionDataset, transition_collate
from torch.utils.data import DataLoader

STATE_VARS = ("u", "v", "w")
PARAM_VARS = ("inflow_angle", "velocity_magnitude")
T_LEN = 4
# Trajectories 0 and 1 share grid + mask (dedup pair); trajectory 2 has its
# own, larger grid.
GRIDS = [(4, 8, 8), (4, 8, 8), (4, 8, 16)]

_WORKTREE = Path(__file__).resolve().parents[1]
_TRAIN_SCRIPT = _WORKTREE / "scripts" / "neural_surrogate" / "train_neural_surrogate.py"


def _write_sample(root: Path, split: str, idx: int, grid: tuple, seed: int) -> None:
    state_dir = root / "state" / split
    param_dir = root / "param" / split
    state_dir.mkdir(parents=True, exist_ok=True)
    param_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    nz, ny, nx = grid
    dims = ("time", "z", "y", "x")
    shape = (T_LEN, nz, ny, nx)
    data_vars = {
        v: (dims, rng.standard_normal(shape).astype(np.float32)) for v in STATE_VARS
    }
    obstacle = np.zeros(grid, dtype=np.float32)
    obstacle[0:2, 2:4, 2 : 2 + nx // 4] = 1.0
    data_vars["blanking"] = (dims, np.broadcast_to(obstacle, shape).copy())
    xr.Dataset(data_vars).to_netcdf(state_dir / f"sample_{idx:04d}.nc")
    pvars = {
        "inflow_angle": (("time",), rng.uniform(-60, 60, T_LEN).astype(np.float32)),
        "velocity_magnitude": (("time",), rng.uniform(1, 5, T_LEN).astype(np.float32)),
    }
    xr.Dataset(pvars).to_netcdf(param_dir / f"sample_{idx:04d}.nc")


@pytest.fixture
def data_root(tmp_path: Path) -> Path:
    root = tmp_path / "data"
    for idx, grid in enumerate(GRIDS):
        _write_sample(root, "train", idx, grid, seed=10 + idx)
    return root


def _make_ds(root: Path, **overrides) -> TransitionDataset:
    kwargs = dict(
        root_dir=root,
        split="train",
        state_vars=STATE_VARS,
        param_vars=PARAM_VARS,
        geometry_var="blanking",
        pushforward_steps=1,
    )
    kwargs.update(overrides)
    return TransitionDataset(**kwargs)


# -- dataset ------------------------------------------------------------------


def test_per_trajectory_geometry_and_dedup(data_root: Path) -> None:
    ds = _make_ds(data_root, sdf_features=True)
    # Trajectories 0/1 share one mask object (content dedup), 2 has its own.
    assert ds.geometry_for(0) is ds.geometry_for(1)
    assert ds.geometry_for(2) is not ds.geometry_for(0)
    assert len(ds._geometries) == 2
    for traj, grid in enumerate(GRIDS):
        assert ds.grid_shape(traj) == grid
        assert tuple(ds.geometry_for(traj).shape) == grid
    # SDF features follow the mask sharing and shapes.
    assert ds.geom_features_for(0) is ds.geom_features_for(1)
    assert ds.geom_features_for(2).shape == (4, *GRIDS[2])


def test_items_carry_their_trajectory_geometry(data_root: Path) -> None:
    ds = _make_ds(data_root)
    per_traj = T_LEN - 1
    item0 = ds[0]  # trajectory 0
    item2 = ds[2 * per_traj]  # trajectory 2
    assert item0["geometry"] is ds.geometry_for(0)
    assert item2["geometry"] is ds.geometry_for(2)
    assert item2["state_n"].shape == (len(STATE_VARS), *GRIDS[2])


def test_collate_rejects_mixed_geometry_batch(data_root: Path) -> None:
    ds = _make_ds(data_root)
    per_traj = T_LEN - 1
    # Same trajectory (and its dedup twin): fine.
    batch = transition_collate([ds[0], ds[1], ds[per_traj]])
    assert batch["geometry"].shape == (1, *GRIDS[0])
    # Mixing trajectory 2's different geometry: loud failure.
    with pytest.raises(ValueError, match="share one geometry"):
        transition_collate([ds[0], ds[2 * per_traj]])


# -- sampler ------------------------------------------------------------------


def test_sampler_batches_are_same_trajectory_and_cover_all(data_root: Path) -> None:
    ds = _make_ds(data_root)
    sampler = TrajectoryBatchSampler(ds, batch_size=2, shuffle=True, seed=1)
    batches = list(sampler)
    seen: list[int] = []
    for batch in batches:
        trajs = {ds.sample_index[i][0] for i in batch}
        assert len(trajs) == 1
        seen.extend(batch)
    assert sorted(seen) == list(range(len(ds)))
    assert len(sampler) == len(batches)
    # A second epoch advances the shuffle seed and still covers every sample.
    batches2 = list(sampler)
    assert sampler._epoch == 2
    assert sorted(i for b in batches2 for i in b) == list(range(len(ds)))


def test_sampler_cell_budget_scales_batch_size(data_root: Path) -> None:
    ds = _make_ds(data_root)
    budget = 2 * math.prod(GRIDS[0])  # 2 small-grid samples; 1 large-grid sample
    sampler = TrajectoryBatchSampler(ds, cell_budget=budget, shuffle=False)
    for batch in sampler:
        traj = ds.sample_index[batch[0]][0]
        expected = 2 if GRIDS[traj] == GRIDS[0] else 1
        assert len(batch) <= expected


def test_sampler_tracks_pushforward_curriculum(data_root: Path) -> None:
    ds = _make_ds(data_root, pushforward_steps=2)
    sampler = TrajectoryBatchSampler(ds, batch_size=4, shuffle=False)
    assert sum(len(b) for b in sampler) == len(GRIDS) * (T_LEN - 2)
    ds.set_pushforward_steps(1)
    # No rebuild: the sampler reads the dataset's index lazily.
    assert sum(len(b) for b in sampler) == len(GRIDS) * (T_LEN - 1)


def test_sampler_requires_a_bound(data_root: Path) -> None:
    ds = _make_ds(data_root)
    with pytest.raises(ValueError, match="batch_size, cell_budget"):
        TrajectoryBatchSampler(ds)


def test_dataloader_end_to_end(data_root: Path) -> None:
    ds = _make_ds(data_root, sdf_features=True)
    loader = DataLoader(
        ds,
        batch_sampler=TrajectoryBatchSampler(ds, batch_size=2, shuffle=True),
        collate_fn=transition_collate,
    )
    n_items = 0
    for batch in loader:
        b, _, nz, ny, nx = batch["state_n"].shape
        assert batch["geometry"].shape == (1, nz, ny, nx)
        assert batch["geom_features"].shape == (1, 4, nz, ny, nx)
        n_items += b
    assert n_items == len(ds)


# -- trainer geometry cache ---------------------------------------------------


class _IdentityModel(torch.nn.Module):
    n_geom_feature_channels = 4

    def __init__(self) -> None:
        super().__init__()
        self.p = torch.nn.Parameter(torch.zeros(1))

    def forward(self, state, params, geometry, geom_features=None):
        return state


def _make_trainer(model) -> Trainer:
    dummy = DataLoader([0, 1], batch_size=1)
    return Trainer(
        model=model,
        train_loader=dummy,
        val_loader=dummy,
        optimizer=torch.optim.SGD(model.parameters(), lr=0.1),
        loss_fn=torch.nn.MSELoss(),
        num_epochs=1,
        device="cpu",
    )


def test_trainer_refreshes_geometry_per_batch(data_root: Path) -> None:
    """A change of batch geometry must refresh the device cache — a stale mask
    would silently train every batch against the first geometry."""
    ds = _make_ds(data_root, sdf_features=True)
    per_traj = T_LEN - 1
    trainer = _make_trainer(_IdentityModel())
    batch_a = transition_collate([ds[0], ds[1]])
    batch_b = transition_collate([ds[2 * per_traj]])

    _, _, _, geom_a = trainer._prepare_batch(batch_a)
    assert geom_a.shape == (2, *GRIDS[0])
    torch.testing.assert_close(trainer._geometry, ds.geometry_for(0))
    feat_a = trainer._geom_features

    _, _, _, geom_b = trainer._prepare_batch(batch_b)
    assert geom_b.shape == (1, *GRIDS[2])
    torch.testing.assert_close(trainer._geometry, ds.geometry_for(2))
    assert trainer._geom_features is not feat_a
    assert trainer._geom_features.shape == (4, *GRIDS[2])
    assert trainer._fluid_mask.shape == GRIDS[2]

    # Content-equal but distinct object (what DataLoader workers produce):
    # revalidates without replacing the cached device tensors.
    geom_device = trainer._geometry
    batch_b2 = dict(batch_b)
    batch_b2["geometry"] = batch_b["geometry"].clone()
    trainer._prepare_batch(batch_b2)
    assert trainer._geometry is geom_device


def test_refork_loader_keeps_custom_batch_sampler(data_root: Path) -> None:
    ds = _make_ds(data_root)
    sampler = TrajectoryBatchSampler(ds, batch_size=2, shuffle=True)
    loader = DataLoader(ds, batch_sampler=sampler, collate_fn=transition_collate)
    reforked = Trainer._refork_loader(loader)
    assert reforked.batch_sampler is sampler
    assert reforked.dataset is ds
    assert reforked.collate_fn is transition_collate


# -- normalization stats ------------------------------------------------------


def test_normalization_stats_use_per_trajectory_masks(data_root: Path) -> None:
    spec = importlib.util.spec_from_file_location("train_ns_stats", _TRAIN_SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    ds = _make_ds(data_root)
    s_mean, s_std, p_mean, p_std = mod._compute_normalization_stats(ds)

    # Reference: gather every trajectory's fluid values per channel directly.
    ref = {v: [] for v in STATE_VARS}
    for traj in range(len(GRIDS)):
        fluid = ds.geometry_for(traj).numpy().astype(bool)
        with xr.open_dataset(ds._state_files[traj]) as raw:
            for v in STATE_VARS:
                vals = np.asarray(raw[v].values)
                ref[v].append(vals[:, fluid].ravel())
    for c, v in enumerate(STATE_VARS):
        allv = np.concatenate(ref[v]).astype(np.float64)
        assert s_mean[c] == pytest.approx(allv.mean(), rel=1e-6)
        assert s_std[c] == pytest.approx(allv.std(), rel=1e-6)
    assert p_mean.shape == (len(PARAM_VARS),)
    assert p_std.shape == (len(PARAM_VARS),)
