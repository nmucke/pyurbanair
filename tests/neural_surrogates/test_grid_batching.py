"""Batches that mix trajectories of one grid shape (``group_by="grid"``).

On CPU smoke shapes via the shared fixtures in ``_latent_generator_fixtures``,
with trajectories 0 and 1 on one grid but with different masks and trajectory
2 on its own grid: the snapshot collate ships per-sample geometry and SDF
features for a mixed batch, the trainer's device geometry cache never serves a
stale mask across shared / mixed batches, latent-cache batches carry each
sample's own geometry conditioning, a mixed snapshot batch and its cached form
score the same loss, and the latent generator trains with grid batching.

Gated with ``importorskip`` on the vendored Tadpole runtime deps.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import xarray as xr

torch = pytest.importorskip("torch")
pytest.importorskip("diffusers")
pytest.importorskip("timm")
pytest.importorskip("einops")

from neural_surrogates import (
    LatentCacheDataset,
    LatentFlowMatchingTrainer,
    SnapshotHistoryDataset,
    TadpoleLatentGenerator,
    TrajectoryBatchSampler,
    load_latent_stats,
    prepare_latent_cache,
    snapshot_history_collate,
)
from torch.utils.data import DataLoader

from tests.neural_surrogates._latent_generator_fixtures import (
    GRID,
    HP,
    NET,
    PARAM_VARS,
    STATE_VARS,
    C,
    P,
    make_ae_export,
    write_history_dataset,
)

GRIDS = [GRID, GRID, (16, 16, 48)]


@pytest.fixture(autouse=True)
def _seed():
    torch.manual_seed(3)
    yield


def _move_obstacle(path: Path) -> None:
    """Shift the obstacle block of one sample so its mask differs from its
    same-grid neighbour's."""
    with xr.open_dataset(path) as ds:
        ds = ds.load()
    ds["blanking"] = ds["blanking"].roll(x=4, roll_coords=False)
    ds.to_netcdf(path)


@pytest.fixture(scope="module")
def inputs(tmp_path_factory) -> tuple[Path, Path]:
    tmp = tmp_path_factory.mktemp("grid_batching")
    ae_dir = make_ae_export(tmp, "local", "fold")
    data_dir = write_history_dataset(
        tmp / "data", splits={"train": 3, "val": 3}, grids=GRIDS
    )
    for split in ("train", "val"):
        _move_obstacle(data_dir / "state" / split / "sample_0001.nc")
    return ae_dir, data_dir


def _dataset(data_dir: Path, split: str) -> SnapshotHistoryDataset:
    return SnapshotHistoryDataset(
        data_dir,
        split,
        state_vars=STATE_VARS,
        param_vars=PARAM_VARS,
        param_history_steps=HP,
        sdf_features="both",
        sdf_clamp_cells=8.0,
    )


def _sampler(ds: Any, **kw: Any) -> TrajectoryBatchSampler:
    kw.setdefault("batch_size", 8)
    kw.setdefault("group_by", "grid")
    return TrajectoryBatchSampler(ds, **kw)


def _loader(ds: Any, collate: Any = snapshot_history_collate, **kw: Any) -> DataLoader:
    return DataLoader(ds, batch_sampler=_sampler(ds, **kw), collate_fn=collate)


def _model(ae_dir: Path) -> TadpoleLatentGenerator:
    return TadpoleLatentGenerator(
        n_state_channels=C,
        n_params=P,
        param_history_steps=HP,
        pretrained_ae_dir=str(ae_dir),
        **NET,
    )


def _trainer(model, train_loader, val_loader) -> LatentFlowMatchingTrainer:
    params = [p for p in model.parameters() if p.requires_grad]
    return LatentFlowMatchingTrainer(
        model,
        train_loader,
        val_loader,
        optimizer=torch.optim.AdamW(params, lr=1e-3),
        loss_fn=torch.nn.MSELoss(),
        num_epochs=1,
        device="cpu",
    )


def _trajs(ds: Any, batch: list[int]) -> list[int]:
    return [ds.sample_index[i][0] for i in batch]


def test_mixed_batch_ships_per_sample_geometry(inputs):
    ae_dir, data_dir = inputs
    ds = _dataset(data_dir, "train")
    assert ds.grid_shape(0) == ds.grid_shape(1)
    assert not torch.equal(ds.geometry_for(0), ds.geometry_for(1))

    sampler = _sampler(ds, shuffle=False)
    batches = list(sampler)
    trajs = _trajs(ds, batches[0])
    assert set(trajs) == {0, 1}  # 4 frames each, one grid
    mixed = snapshot_history_collate([ds[i] for i in batches[0]])
    b = len(batches[0])
    assert mixed["geometry"].shape == (b, *GRID)
    assert mixed["geom_features"].shape[:2] == (b, 4)
    for row, traj in enumerate(trajs):
        assert torch.equal(mixed["geometry"][row], ds.geometry_for(traj))
        assert torch.equal(mixed["geom_features"][row], ds.geom_features_for(traj))

    # The trainer's device cache: shared (one trajectory) -> mixed -> shared on
    # the other geometry. Every call returns the batch's own masks.
    trainer = _trainer(_model(ae_dir), _loader(ds), _loader(ds, shuffle=False))
    one = {
        traj: snapshot_history_collate(
            [ds[i] for i in batches[0] if _trajs(ds, [i])[0] == traj]
        )
        for traj in (0, 1)
    }
    for batch, expected in (
        (one[0], [0] * 4),
        (mixed, trajs),
        (one[1], [1] * 4),
        (mixed, trajs),
        (one[0], [0] * 4),
    ):
        _, geometry, features = trainer._prepare_snapshot_batch(batch)
        assert geometry.shape == (len(expected), *GRID)
        for row, traj in enumerate(expected):
            assert torch.equal(geometry[row], ds.geometry_for(traj))
            assert torch.equal(features[row], ds.geom_features_for(traj))


def test_latent_cache_mixed_batch_and_training(inputs, tmp_path):
    ae_dir, data_dir = inputs
    model = _model(ae_dir)
    datasets = {split: _dataset(data_dir, split) for split in ("train", "val")}
    root = prepare_latent_cache(model, datasets, tmp_path / "cache", device="cpu")
    cache = {split: LatentCacheDataset(root, ds) for split, ds in datasets.items()}
    train = cache["train"]
    assert not torch.equal(train._geoms[0], train._geoms[1])

    sampler = _sampler(train, shuffle=True, seed=1)
    plan = list(sampler)
    sampler.set_epoch(0)  # the loader replays the planned epoch
    loader = DataLoader(train, batch_sampler=sampler)
    mixed = 0
    for batch_idx, batch in zip(plan, loader, strict=True):
        trajs = _trajs(train, batch_idx)
        assert len({train.grid_shape(t) for t in trajs}) == 1
        mixed += len(set(trajs)) > 1
        assert batch["geom"].shape[0] == batch["latent"].shape[0] == len(trajs)
        for row, traj in enumerate(trajs):
            assert torch.equal(batch["geom"][row], train._geoms[traj])
    assert mixed

    # A mixed snapshot batch and its cached form give the same loss: the
    # per-sample geometry reaches the encoder as the cache saw it.
    snap_train = _loader(datasets["train"], shuffle=False)
    trainer = _trainer(model, snap_train, _loader(datasets["val"], shuffle=False))
    model.set_latent_normalization(*load_latent_stats(root))
    indices = next(iter(snap_train.batch_sampler))
    assert len(set(_trajs(train, indices))) == 2
    snap = snapshot_history_collate([datasets["train"][i] for i in indices])
    cached = torch.utils.data.default_collate([train[i] for i in indices])
    model.eval()
    with torch.no_grad():
        expected = trainer._forward(snap, torch.Generator().manual_seed(1))
        got = trainer._forward(cached, torch.Generator().manual_seed(1))
    torch.testing.assert_close(got, expected, rtol=1e-4, atol=1e-5)

    # One epoch on cached grid batches, and one on snapshot grid batches.
    for train_loader, val_loader in (
        (
            DataLoader(train, batch_sampler=_sampler(train)),
            DataLoader(
                cache["val"], batch_sampler=_sampler(cache["val"], shuffle=False)
            ),
        ),
        (_loader(datasets["train"]), _loader(datasets["val"], shuffle=False)),
    ):
        history = _trainer(model, train_loader, val_loader).fit()
        assert len(history["val"]) == 1
        assert all(torch.isfinite(torch.tensor(v)) for v in history["val"])
