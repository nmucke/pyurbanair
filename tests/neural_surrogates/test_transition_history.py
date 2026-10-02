"""History window (``num_history_steps``) on :class:`TransitionDataset`.

Covers the dataset half of the history-conditioned surrogate:

* ``H=1`` is byte-identical to the historyless construction (same items, same
  length) — the no-op-when-absent rule;
* ``H>1`` stacks the window ``t-H+1 … t`` onto ``state_n``'s channel axis
  **oldest first**, with the sample count shortened at the *front*;
* the horizon curriculum (``set_pushforward_steps``) keeps the ``H`` offset;
* too-short trajectories and the unsupported patch dataset fail loud.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import xarray as xr

torch = pytest.importorskip("torch")

from neural_surrogates.datasets.transition import TransitionDataset, transition_collate

NZ = NY = NX = 8
T_LEN = 6
STATE_VARS = ("u", "v", "w")
PARAM_VARS = ("inflow_angle", "velocity_magnitude")
C = len(STATE_VARS)
N_TRAJ = 2


def _write_sample(state_dir: Path, param_dir: Path, idx: int, seed: int) -> None:
    state_dir.mkdir(parents=True, exist_ok=True)
    param_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    dims = ("time", "z", "y", "x")
    shape = (T_LEN, NZ, NY, NX)
    data_vars = {
        v: (dims, rng.standard_normal(shape).astype(np.float32)) for v in STATE_VARS
    }
    obstacle = np.zeros((NZ, NY, NX), dtype=np.float32)
    obstacle[0:2, 2:4, 2:4] = 1.0
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
    for idx in range(N_TRAJ):
        _write_sample(root / "state" / "train", root / "param" / "train", idx, 20 + idx)
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


def _raw_frames(root: Path, traj: int) -> torch.Tensor:
    """``(T, C, *grid)`` straight off disk, bypassing the dataset."""
    path = root / "state" / "train" / f"sample_{traj:04d}.nc"
    with xr.open_dataset(path) as ds:
        stacked = np.stack([np.asarray(ds[v].values) for v in STATE_VARS], axis=1)
    return torch.from_numpy(stacked).to(torch.float32)


# -- H=1 is the historyless dataset -----------------------------------------


def test_h1_items_identical_to_default(data_root: Path) -> None:
    default = _make_ds(data_root)
    explicit = _make_ds(data_root, num_history_steps=1)

    assert len(explicit) == len(default) == N_TRAJ * (T_LEN - 1)
    assert explicit.sample_index == default.sample_index
    for idx in range(len(default)):
        a, b = default[idx], explicit[idx]
        assert set(a) == set(b)
        for key in a:
            assert torch.equal(a[key], b[key]), f"{key} differs at sample {idx}"
    assert default[0]["state_n"].shape == (C, NZ, NY, NX)


def test_h1_with_sdf_features_identical_to_default(data_root: Path) -> None:
    default = _make_ds(data_root, sdf_features=True, sdf_clamp_cells=8.0)
    explicit = _make_ds(
        data_root, sdf_features=True, sdf_clamp_cells=8.0, num_history_steps=1
    )
    a, b = default[3], explicit[3]
    assert (
        set(a)
        == set(b)
        == {
            "state_n",
            "state_next",
            "params_n",
            "geometry",
            "geom_features",
        }
    )
    for key in a:
        assert torch.equal(a[key], b[key])


# -- H>1 layout and index arithmetic ----------------------------------------


def test_h3_shape_length_and_frame_order(data_root: Path) -> None:
    H, K = 3, 2
    ds = _make_ds(data_root, num_history_steps=H, pushforward_steps=K)

    assert len(ds) == N_TRAJ * (T_LEN - K - (H - 1))
    # Anchors start at t = H-1 and run to T-K-1 for every trajectory.
    assert ds.sample_index == [
        (traj, t) for traj in range(N_TRAJ) for t in range(H - 1, T_LEN - K)
    ]

    item = ds[0]
    assert item["state_n"].shape == (H * C, NZ, NY, NX)
    assert item["state_next"].shape == (C, NZ, NY, NX)
    assert item["params_n"].shape == (K, len(PARAM_VARS))

    # Frame order, checked against the raw netCDF: oldest first, newest last.
    for flat, (traj, t) in enumerate(ds.sample_index):
        frames = _raw_frames(data_root, traj)
        got = ds[flat]
        for h in range(H):
            block = got["state_n"][h * C : (h + 1) * C]
            assert torch.equal(block, frames[t - (H - 1) + h])
        assert torch.equal(got["state_n"][-C:], frames[t])
        assert torch.equal(got["state_next"], frames[t + K])
        assert torch.equal(got["params_n"], ds._params[traj][t : t + K])


def test_h3_first_sample_starts_at_history_offset(data_root: Path) -> None:
    ds = _make_ds(data_root, num_history_steps=3)
    frames = _raw_frames(data_root, 0)
    # The t=0 and t=1 anchors are gone: the first sample's newest frame is t=2.
    assert ds.sample_index[0] == (0, 2)
    assert torch.equal(ds[0]["state_n"][-C:], frames[2])


# -- curriculum -------------------------------------------------------------


def test_history_offset_survives_pushforward_curriculum(data_root: Path) -> None:
    H = 3
    ds = _make_ds(data_root, num_history_steps=H, pushforward_steps=1)
    assert len(ds) == N_TRAJ * (T_LEN - 1 - (H - 1))

    ds.set_pushforward_steps(2)
    assert ds.pushforward_steps == 2
    assert ds.num_history_steps == H
    assert len(ds) == N_TRAJ * (T_LEN - 2 - (H - 1))
    assert ds.sample_index[0] == (0, H - 1)

    frames = _raw_frames(data_root, 0)
    item = ds[0]
    assert item["state_n"].shape == (H * C, NZ, NY, NX)
    assert torch.equal(item["state_n"][-C:], frames[H - 1])
    assert torch.equal(item["state_next"], frames[H - 1 + 2])


def test_curriculum_rejects_horizon_that_no_longer_fits(data_root: Path) -> None:
    ds = _make_ds(data_root, num_history_steps=3)
    with pytest.raises(ValueError, match="num_history_steps"):
        ds.set_pushforward_steps(T_LEN - 2)
    # The failed switch left the dataset on its previous horizon.
    assert ds.pushforward_steps == 1


# -- guards -----------------------------------------------------------------


def test_too_short_trajectory_names_both_knobs(data_root: Path) -> None:
    with pytest.raises(ValueError) as excinfo:
        _make_ds(data_root, num_history_steps=3, pushforward_steps=T_LEN - 2)
    message = str(excinfo.value)
    assert "pushforward_steps" in message
    assert "num_history_steps" in message


@pytest.mark.parametrize("bad", [0, -2])
def test_num_history_steps_must_be_positive(data_root: Path, bad: int) -> None:
    with pytest.raises(ValueError, match="num_history_steps must be >= 1"):
        _make_ds(data_root, num_history_steps=bad)


def test_patch_dataset_rejects_history(data_root: Path) -> None:
    from neural_surrogates.datasets.patch import PatchTransitionDataset

    dd_kwargs = dict(
        interior_size=4,
        halo=2,
        taper=1,
        coarsen_factor=2,
        n_pos=3,
        periodic_axes=(False, True, False),
    )
    with pytest.raises(NotImplementedError, match="num_history_steps"):
        PatchTransitionDataset(
            root_dir=data_root,
            split="train",
            state_vars=STATE_VARS,
            param_vars=PARAM_VARS,
            geometry_var="blanking",
            num_history_steps=2,
            decomposition=dd_kwargs,
        )
    # H=1 still builds (the kwarg is accepted, not merely tolerated).
    ok = PatchTransitionDataset(
        root_dir=data_root,
        split="train",
        state_vars=STATE_VARS,
        param_vars=PARAM_VARS,
        geometry_var="blanking",
        num_history_steps=1,
        decomposition=dd_kwargs,
    )
    assert ok.num_history_steps == 1


# -- collate ----------------------------------------------------------------


def test_collate_stacks_history_channels(data_root: Path) -> None:
    H = 3
    ds = _make_ds(data_root, num_history_steps=H)
    items = [ds[0], ds[1]]
    batch = transition_collate(items)

    assert batch["state_n"].shape == (2, H * C, NZ, NY, NX)
    assert batch["state_next"].shape == (2, C, NZ, NY, NX)
    assert batch["geometry"].shape == (1, NZ, NY, NX)
    for i, item in enumerate(items):
        assert torch.equal(batch["state_n"][i], item["state_n"])
        assert torch.equal(batch["state_next"][i], item["state_next"])
