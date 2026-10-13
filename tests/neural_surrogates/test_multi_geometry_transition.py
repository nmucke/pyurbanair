"""Multi-geometry (varying-domain) support in the transition-training stack.

The random-geometry training data gives every trajectory its own grid and
geometry mask. Covers:

* :class:`TransitionDataset` — per-trajectory geometry/SDF with content dedup
  (equal masks share one tensor object, so single-geometry splits behave
  byte-identically to the fixed-domain dataset);
* :func:`transition_collate` — fails loud on a mixed-geometry batch;
* :class:`TrajectoryBatchSampler` — same-trajectory batches, cell-budget batch
  sizing, full coverage, and lazy pickup of ``set_pushforward_steps``; its
  ``group_by="grid"`` mode (same-grid batches across trajectories) and the
  unchanged default order;
* :class:`BaseTraining._prepare_batch` — refreshes the device geometry cache
  when the batch's geometry changes (the silent-stale-mask trap);
* ``_refork_loader`` — rebuilds a loader around its custom batch sampler;
* the library's ``_compute_normalization_stats`` — per-trajectory masks.
"""

from __future__ import annotations

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


def test_geometry_lives_in_one_shared_buffer(data_root: Path) -> None:
    """Masks and SDF features are shared before any loader worker starts, so
    workers ship handles to them instead of each keeping its own shared copy
    of every geometry it served; values are those of the files."""
    ds = _make_ds(data_root, sdf_features=True)
    for tensors in (ds._geometries, ds._geom_features):
        assert all(t.is_shared() for t in tensors)
        assert len({t.untyped_storage().data_ptr() for t in tensors}) == 1
    for traj, path in enumerate(ds._state_files):
        assert torch.equal(ds.geometry_for(traj), ds._load_geometry(path))
    loader = DataLoader(
        ds,
        batch_sampler=TrajectoryBatchSampler(ds, batch_size=2, shuffle=False),
        collate_fn=transition_collate,
        num_workers=1,
    )
    for batch in loader:
        traj = 0 if batch["geometry"].shape[-1] == 8 else 2
        assert torch.equal(batch["geometry"][0], ds.geometry_for(traj))
        assert torch.equal(batch["geom_features"][0], ds.geom_features_for(traj))


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


def _reference_trajectory_batches(
    sampler: TrajectoryBatchSampler, generator: torch.Generator | None
) -> list[list[int]]:
    """The trajectory-mode algorithm as it was before ``group_by`` existed: the
    default must keep drawing exactly these batches for a seed."""
    by_traj: dict[int, list[int]] = {}
    for flat, (traj, _t) in enumerate(sampler.dataset.sample_index):
        by_traj.setdefault(traj, []).append(flat)
    batches: list[list[int]] = []
    for traj, flats in by_traj.items():
        if generator is not None:
            order = torch.randperm(len(flats), generator=generator).tolist()
            flats = [flats[i] for i in order]
        size = sampler.batch_size or 1
        if sampler.cell_budget is not None:
            size = max(1, sampler.cell_budget // math.prod(GRIDS[traj]))
            if sampler.batch_size is not None:
                size = min(sampler.batch_size, size)
        for start in range(0, len(flats), size):
            batch = flats[start : start + size]
            if not (sampler.drop_last and len(batch) < size):
                batches.append(batch)
    if generator is not None:
        order = torch.randperm(len(batches), generator=generator).tolist()
        batches = [batches[i] for i in order]
    return batches


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(batch_size=2),
        dict(batch_size=2, drop_last=True),
        dict(cell_budget=2 * math.prod(GRIDS[0])),
    ],
)
def test_sampler_default_order_is_unchanged(data_root: Path, kwargs: dict) -> None:
    ds = _make_ds(data_root)
    sampler = TrajectoryBatchSampler(ds, shuffle=True, seed=5, **kwargs)
    assert sampler.group_by == "trajectory"
    for epoch in range(3):
        expected = _reference_trajectory_batches(
            sampler, torch.Generator().manual_seed(5 + epoch)
        )
        assert list(sampler) == expected
    unshuffled = TrajectoryBatchSampler(ds, shuffle=False, **kwargs)
    assert list(unshuffled) == _reference_trajectory_batches(unshuffled, None)


def test_grid_sampler_mixes_trajectories_never_shapes(data_root: Path) -> None:
    ds = _make_ds(data_root)
    sampler = TrajectoryBatchSampler(
        ds, batch_size=4, shuffle=True, seed=1, group_by="grid"
    )
    for _ in range(2):  # two epochs: different shuffles, same invariants
        batches = list(sampler)
        assert len(sampler) == len(batches)
        assert sorted(i for b in batches for i in b) == list(range(len(ds)))
        mixed = False
        for batch in batches:
            trajs = {ds.sample_index[i][0] for i in batch}
            assert len({GRIDS[t] for t in trajs}) == 1
            mixed |= len(trajs) > 1
        # Trajectories 0 and 1 have 3 samples each on one grid: a batch of 4
        # from that grid must hold both.
        assert mixed
    assert sampler._epoch == 2


def test_grid_sampler_respects_cell_budget_and_drop_last(data_root: Path) -> None:
    ds = _make_ds(data_root)
    per_traj = T_LEN - 1
    budget = 2 * math.prod(GRIDS[0])  # 2 small-grid samples; 1 large-grid sample
    sampler = TrajectoryBatchSampler(ds, cell_budget=budget, group_by="grid")
    sizes = {GRIDS[0]: 2, GRIDS[2]: 1}
    batches = list(sampler)
    for batch in batches:
        assert len(batch) == sizes[GRIDS[ds.sample_index[batch[0]][0]]]
    assert len(batches) == len(sampler) == 2 * per_traj // 2 + per_traj
    # drop_last applies per grid: the small grid's 6 samples give one full
    # batch of 4, the large grid's 3 samples none.
    dropping = TrajectoryBatchSampler(ds, batch_size=4, drop_last=True, group_by="grid")
    batches = list(dropping)
    assert len(dropping) == len(batches) == 1
    assert len(batches[0]) == 4
    assert dropping.batch_sizes(0) == dropping.batch_sizes() == [4]


def test_grid_sampler_is_deterministic(data_root: Path) -> None:
    ds = _make_ds(data_root)
    per_traj = T_LEN - 1
    # shuffle=False: grid-major, flat order within a grid; never advances.
    fixed = TrajectoryBatchSampler(ds, batch_size=4, shuffle=False, group_by="grid")
    expected = [[0, 1, 2, 3], [4, 5], [6, 7, 8]]
    assert 2 * per_traj == 6 and list(fixed) == expected
    assert list(fixed) == expected and fixed._epoch == 0
    # A seed fixes every epoch's shuffle; set_epoch replays it.
    a = TrajectoryBatchSampler(ds, batch_size=4, seed=3, group_by="grid")
    b = TrajectoryBatchSampler(ds, batch_size=4, seed=3, group_by="grid")
    epochs = [list(a), list(a)]
    assert epochs == [list(b), list(b)]
    b.set_epoch(1)
    assert list(b) == epochs[1]


def test_sampler_rejects_unknown_group_by(data_root: Path) -> None:
    ds = _make_ds(data_root)
    with pytest.raises(ValueError, match="group_by"):
        TrajectoryBatchSampler(ds, batch_size=2, group_by="geometry")


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
    # The normalization pass now lives in the library (imported by both the train
    # and fine-tune scripts) rather than as a script-local helper.
    from neural_surrogates.training.data_utils import _compute_normalization_stats

    ds = _make_ds(data_root)
    s_mean, s_std, p_mean, p_std = _compute_normalization_stats(ds)

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


class _RecordingModel(_IdentityModel):
    """Records the geometry tensors each call receives (identity matters: the
    Tadpole stepper caches its frozen geometry branch on it)."""

    def __init__(self) -> None:
        super().__init__()
        self.seen: list[tuple[torch.Tensor, torch.Tensor | None]] = []

    def forward(self, state, params, geometry, geom_features=None):
        self.seen.append((geometry, geom_features))
        return state


def test_trainer_reuses_broadcast_geometry(data_root: Path) -> None:
    """Every model call of a batch -- and of the next batch on the same
    geometry and batch size -- gets the same broadcast tensors; a new geometry
    gets new ones."""
    ds = _make_ds(data_root, sdf_features=True, pushforward_steps=3)
    model = _RecordingModel()
    trainer = _make_trainer(model)
    batch_a = transition_collate([ds[0]])
    batch_a2 = transition_collate([ds[1]])  # trajectory 1: same grid + mask
    batch_b = transition_collate([ds[2]])  # trajectory 2: its own geometry

    trainer._forward(batch_a)
    trainer._forward(batch_a2)
    assert len(model.seen) == 6  # three rollout steps per batch
    geometry, features = model.seen[0]
    assert features is not None and features.shape == (1, 4, *GRIDS[0])
    assert all(g is geometry and f is features for g, f in model.seen)

    trainer._forward(batch_b)
    assert model.seen[-1][0] is not geometry
    assert model.seen[-1][0].shape == (1, *GRIDS[2])


@pytest.mark.parametrize("loss_fn", [torch.nn.MSELoss(), torch.nn.L1Loss()])
def test_masked_loss_matches_boolean_indexing(data_root: Path, loss_fn) -> None:
    """The sync-free masked loss equals the loss over the fluid cells only,
    in value and gradient, even with non-finite junk inside the obstacles."""
    ds = _make_ds(data_root, sdf_features=True)
    batch = transition_collate([ds[6], ds[7]])  # trajectory 2
    fluid = ds.geometry_for(2).bool()
    batch["state_next"] = batch["state_next"].masked_fill(~fluid, float("nan"))

    class Scaled(_IdentityModel):
        def forward(self, state, params, geometry, geom_features=None):
            return state * (1.0 + self.p)

    model = Scaled()
    trainer = _make_trainer(model)
    trainer.loss_fn = loss_fn
    loss = trainer._forward(batch)
    (grad,) = torch.autograd.grad(loss, model.p)

    pred = batch["state_n"] * (1.0 + model.p)
    expected = loss_fn(pred[..., fluid], batch["state_next"][..., fluid])
    (expected_grad,) = torch.autograd.grad(expected, model.p)
    torch.testing.assert_close(loss, expected)
    torch.testing.assert_close(grad, expected_grad)


class _ScaledModel(_IdentityModel):
    def forward(self, state, params, geometry, geom_features=None):
        return state * (1.0 + self.p)


def _run_accum_epoch(loader: DataLoader, lr: float, **trainer_kwargs):
    """One SGD epoch of a ``_ScaledModel`` trainer; returns the trainer and
    its number of optimizer steps."""
    model = _ScaledModel()
    trainer = Trainer(
        model=model,
        train_loader=loader,
        val_loader=loader,
        optimizer=torch.optim.SGD(model.parameters(), lr=lr),
        loss_fn=torch.nn.MSELoss(),
        num_epochs=1,
        device="cpu",
        amp=False,
        **trainer_kwargs,
    )
    steps = 0
    step = trainer.optimizer.step

    def counting_step(*args, **kwargs):
        nonlocal steps
        steps += 1
        return step(*args, **kwargs)

    trainer.optimizer.step = counting_step  # type: ignore[method-assign]
    trainer._train_epoch()
    return trainer, steps


def _sample_losses(batch: dict, p: torch.Tensor) -> list[torch.Tensor]:
    """The masked MSE of a ``_ScaledModel`` with parameter ``p``, per sample."""
    fluid = batch["geometry"][0].bool()
    pred = batch["state_n"] * (1.0 + p)
    return [
        torch.nn.functional.mse_loss(
            pred[b][..., fluid], batch["state_next"][b][..., fluid]
        )
        for b in range(pred.shape[0])
    ]


def test_grad_accum_averages_micro_batches(data_root: Path) -> None:
    """``grad_accum_steps=k`` takes one optimizer step per k batches on their
    mean gradient; a short last group is averaged over its own size."""
    ds = _make_ds(data_root, sdf_features=True)
    loader = DataLoader(
        ds,
        batch_sampler=TrajectoryBatchSampler(ds, batch_size=1, shuffle=False),
        collate_fn=transition_collate,
    )
    batches = list(loader)  # 9 single-sample batches -> groups of 4, 4, 1
    lr = 0.1
    trainer, steps = _run_accum_epoch(loader, lr, grad_accum_steps=4)

    p = torch.zeros(1, requires_grad=True)
    for group in (batches[0:4], batches[4:8], batches[8:9]):
        grad = torch.zeros(1)
        for batch in group:
            (loss,) = _sample_losses(batch, p)
            grad += torch.autograd.grad(loss, p)[0] / len(group)
        p = (p - lr * grad).detach().requires_grad_(True)
    assert steps == 3
    assert trainer._optimizer_steps_per_epoch() == 3
    torch.testing.assert_close(trainer.model.p.detach(), p.detach())


@pytest.mark.parametrize("window", [None, (2, 3), (0, 2), (7, 5)])
def test_profile_window(data_root: Path, tmp_path: Path, window) -> None:
    """``profile_steps=(start, count)`` profiles that window of the first
    epoch into ``profile.txt`` (a window running past the epoch end is closed
    there) without changing training; off writes nothing."""
    ds = _make_ds(data_root, sdf_features=True)
    loader = DataLoader(
        ds,
        batch_sampler=TrajectoryBatchSampler(ds, batch_size=1, shuffle=False),
        collate_fn=transition_collate,
    )

    class Scaled(_IdentityModel):
        def forward(self, state, params, geometry, geom_features=None):
            return state * (1.0 + self.p)

    params = []
    for profile_steps in (None, window):
        model = Scaled()
        out = tmp_path / ("off" if profile_steps is None else "on")
        trainer = Trainer(
            model=model,
            train_loader=loader,
            val_loader=loader,
            optimizer=torch.optim.SGD(model.parameters(), lr=0.1),
            loss_fn=torch.nn.MSELoss(),
            num_epochs=2,
            device="cpu",
            amp=False,
            grad_accum_steps=2,
            weights_path=out / "weights.pt",
            profile_steps=profile_steps,
        )
        trainer._train_epoch()
        trainer._train_epoch()  # one-shot: the second epoch is not profiled
        params.append(model.p.detach().clone())
        profile = out / "profile.txt"
        if profile_steps is None:
            assert not profile.exists()
        else:
            text = profile.read_text()
            count = min(window[1], 9 - window[0])  # 9 batches per epoch
            assert text.startswith(f"profiled {count} micro-batches")
            assert "Self CPU time total" in text and "aten::" in text
    torch.testing.assert_close(params[0], params[1])


@pytest.mark.parametrize(
    "target, group_batches",
    [
        (4, (3, 4)),  # 2+1+2 = 5 overshoots, then 1+1+1+1 = 4
        (6, (4, 3)),  # 2+1+2+1 = 6, then a partial last group of 1+1+1
    ],
)
def test_grad_accum_samples_averages_over_samples(
    data_root: Path, target: int, group_batches: tuple[int, ...]
) -> None:
    """``grad_accum_samples=N`` steps once at least N samples are in, on the
    per-sample mean gradient, so variable micro-batches weigh by their size;
    the epoch's last partial group is flushed."""
    ds = _make_ds(data_root, sdf_features=True)
    # 256-cell trajectories 0/1 get batches of 2 (then 1), the 512-cell
    # trajectory 2 batches of 1: sizes 2, 1, 2, 1, 1, 1, 1.
    sampler = TrajectoryBatchSampler(ds, batch_size=3, cell_budget=512, shuffle=False)
    loader = DataLoader(ds, batch_sampler=sampler, collate_fn=transition_collate)
    batches = list(loader)
    assert [len(b["state_n"]) for b in batches] == [2, 1, 2, 1, 1, 1, 1]
    lr = 0.1
    trainer, steps = _run_accum_epoch(loader, lr, grad_accum_samples=target)

    p = torch.zeros(1, requires_grad=True)
    start = 0
    for size in group_batches:
        losses = [
            loss for b in batches[start : start + size] for loss in _sample_losses(b, p)
        ]
        start += size
        (grad,) = torch.autograd.grad(torch.stack(losses).mean(), p)
        p = (p - lr * grad).detach().requires_grad_(True)
    assert steps == len(group_batches) == trainer._optimizer_steps_per_epoch()
    torch.testing.assert_close(trainer.model.p.detach(), p.detach())


@pytest.mark.parametrize("kwargs", [{}, {"grad_accum_samples": 1}])
def test_no_accumulation_steps_every_batch(data_root: Path, kwargs: dict) -> None:
    """Without accumulation (the default) and with ``grad_accum_samples=1``
    every batch takes its own step on its own mean loss."""
    ds = _make_ds(data_root, sdf_features=True)
    sampler = TrajectoryBatchSampler(ds, batch_size=3, cell_budget=512, shuffle=False)
    loader = DataLoader(ds, batch_sampler=sampler, collate_fn=transition_collate)
    lr = 0.1
    trainer, steps = _run_accum_epoch(loader, lr, **kwargs)

    p = torch.zeros(1, requires_grad=True)
    for batch in loader:
        (grad,) = torch.autograd.grad(torch.stack(_sample_losses(batch, p)).mean(), p)
        p = (p - lr * grad).detach().requires_grad_(True)
    assert steps == len(loader) == trainer._optimizer_steps_per_epoch()
    torch.testing.assert_close(trainer.model.p.detach(), p.detach())


def test_grad_accum_steps_and_samples_are_exclusive() -> None:
    model = _IdentityModel()
    dummy = DataLoader([0, 1], batch_size=1)
    with pytest.raises(ValueError, match="not both"):
        Trainer(
            model=model,
            train_loader=dummy,
            val_loader=dummy,
            optimizer=torch.optim.SGD(model.parameters(), lr=0.1),
            loss_fn=torch.nn.MSELoss(),
            num_epochs=1,
            device="cpu",
            grad_accum_steps=2,
            grad_accum_samples=8,
        )


def test_per_step_warmup_counts_steps_at_the_curriculum_horizon(
    data_root: Path,
) -> None:
    """The per-step warmup spans the optimizer steps the curriculum's first
    epochs really take: at horizon 1 a trajectory has more samples than at
    the dataset's built (target) horizon, so the LR must reach its peak only
    after the last step of epoch 0, not before it."""
    ds = _make_ds(data_root, sdf_features=True, pushforward_steps=3)
    loader = DataLoader(
        ds,
        batch_sampler=TrajectoryBatchSampler(ds, batch_size=1, shuffle=False),
        collate_fn=transition_collate,
    )
    model = _ScaledModel()
    trainer = Trainer(
        model=model,
        train_loader=loader,
        val_loader=loader,
        optimizer=torch.optim.SGD(model.parameters(), lr=0.1),
        loss_fn=torch.nn.MSELoss(),
        num_epochs=3,
        device="cpu",
        amp=False,
        pushforward_epochs_per_step=1,
        pushforward_start_steps=1,
        lr_warmup_epochs=1,
        lr_warmup_start=0.01,
        lr_warmup_per_step=True,
        grad_accum_steps=2,
    )
    # Horizon 1: 3 samples per trajectory -> 9 batches -> 5 steps (horizon 3
    # would give 3 batches -> 2 steps). The dataset keeps its built horizon.
    assert trainer.scheduler._milestones == [5]
    assert ds.pushforward_steps == 3
    lrs: list[float] = []
    step = trainer.optimizer.step

    def recording_step(*args, **kwargs):
        lrs.append(trainer.optimizer.param_groups[0]["lr"])
        return step(*args, **kwargs)

    trainer.optimizer.step = recording_step  # type: ignore[method-assign]
    trainer._validate = lambda: 0.0  # type: ignore[method-assign]
    trainer.fit()
    # Epoch 0 ramps over all 5 steps, epoch 1 (horizon 2: 6 batches) is at peak.
    assert lrs[:5] == pytest.approx([0.01 + 0.09 * i / 5 for i in range(5)])
    assert lrs[5:8] == pytest.approx([0.1] * 3)


def test_resumed_run_continues_the_sampler_shuffles(
    data_root: Path, tmp_path: Path
) -> None:
    """The trainer seeds each epoch's shuffle by its epoch number, so a run
    resumed from a checkpoint draws the same batches as the uninterrupted
    run instead of replaying the shuffles of its first epochs."""
    ds = _make_ds(data_root, sdf_features=True)

    def run(out: Path, num_epochs: int) -> list[list[int]]:
        orders: list[list[int]] = []

        class Recording(TrajectoryBatchSampler):
            def __iter__(self):  # type: ignore[no-untyped-def]
                batches = list(super().__iter__())
                orders.append([i for batch in batches for i in batch])
                return iter(batches)

        sampler = Recording(ds, batch_size=1, shuffle=True, seed=3)
        loader = DataLoader(ds, batch_sampler=sampler, collate_fn=transition_collate)
        model = _ScaledModel()
        trainer = Trainer(
            model=model,
            train_loader=loader,
            val_loader=loader,
            optimizer=torch.optim.SGD(model.parameters(), lr=0.1),
            loss_fn=torch.nn.MSELoss(),
            num_epochs=num_epochs,
            device="cpu",
            amp=False,
            weights_path=out / "weights.pt",
            resume=True,
        )
        trainer._validate = lambda: 0.0  # type: ignore[method-assign]
        trainer.fit()
        return orders

    full = run(tmp_path / "full", num_epochs=3)
    run(tmp_path / "resumed", num_epochs=1)
    resumed = run(tmp_path / "resumed", num_epochs=3)
    assert len(full) == 3 and len(resumed) == 2
    assert full[0] != full[1]
    assert resumed == full[1:]


def test_per_step_warmup_counts_the_shuffled_groups(data_root: Path) -> None:
    """With ``grad_accum_samples`` over shuffled variable batches the groups
    overshoot the target differently than in unshuffled order; the warmup
    counts the epoch's own shuffle, so it ends exactly with epoch 0."""
    ds = _make_ds(data_root, sdf_features=True)
    # Sizes 2, 1, 2, 1, 1, 1, 1 unshuffled (2 groups of >= 4 samples), drawn
    # as 1, 2, 1, 2, 1, 1, 1 in epoch 0 of seed 0 (3 groups).
    sampler = TrajectoryBatchSampler(ds, batch_size=3, cell_budget=512, seed=0)
    loader = DataLoader(ds, batch_sampler=sampler, collate_fn=transition_collate)
    model = _ScaledModel()
    trainer = Trainer(
        model=model,
        train_loader=loader,
        val_loader=loader,
        optimizer=torch.optim.SGD(model.parameters(), lr=0.1),
        loss_fn=torch.nn.MSELoss(),
        num_epochs=2,
        device="cpu",
        amp=False,
        lr_warmup_epochs=1,
        lr_warmup_start=0.01,
        lr_warmup_per_step=True,
        grad_accum_samples=4,
    )
    assert trainer._optimizer_steps_per_epoch() == 2
    assert trainer.scheduler._milestones == [3]
    lrs: list[float] = []
    step = trainer.optimizer.step

    def recording_step(*args, **kwargs):
        lrs.append(trainer.optimizer.param_groups[0]["lr"])
        return step(*args, **kwargs)

    trainer.optimizer.step = recording_step  # type: ignore[method-assign]
    trainer._validate = lambda: 0.0  # type: ignore[method-assign]
    trainer.fit()
    assert lrs[:3] == pytest.approx([0.01, 0.04, 0.07])
    assert lrs[3:] == pytest.approx([0.1] * (len(lrs) - 3))
