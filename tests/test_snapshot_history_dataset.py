"""Plan 07 phase 1: :class:`SnapshotHistoryDataset` + the shared param reader.

Covers plan 07 §5.1 -- sample/time pairing, oldest-first history, verified
plateau padding (``constant_prehistory``), static (scalar) params, explicit
variable ordering, cadence validation, ``time_stride``, shared-geometry
collate, multi-geometry bucketing, ``get_normalization_stats`` on the new
dataset, the shared snapshot batch preparation, and unchanged
:class:`TransitionDataset` behaviour after ``_load_params`` was hoisted into
``datasets/_params.load_param_table``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import xarray as xr

torch = pytest.importorskip("torch")

from neural_surrogates import (
    BaseTraining,
    SnapshotHistoryDataset,
    TrajectoryBatchSampler,
    TransitionDataset,
    snapshot_history_collate,
)
from neural_surrogates.datasets._params import load_param_table
from neural_surrogates.training.data_utils import get_normalization_stats
from torch.utils.data import DataLoader

STATE_VARS = ("u", "v", "w")
# Two time-varying params plus one static scalar (uDALES-style
# ``pressure_gradient_magnitude``).
PARAM_VARS = ("inflow_angle", "velocity_magnitude", "pressure_gradient_magnitude")
GRID = (4, 6, 8)
T_LEN = 8
DT = 5.0
HP = 3
N_TRAJ = 2


def _write_sample(
    root: Path,
    idx: int,
    *,
    split: str = "train",
    grid: tuple[int, int, int] = GRID,
    t_len: int = T_LEN,
    times: np.ndarray | None = None,
    param_times: np.ndarray | None = None,
    param_stem: str | None = None,
    seed: int = 0,
    param_override: dict | None = None,
) -> None:
    """One ``sample_XXXX`` state + param pair with an explicit ``time`` coord."""
    state_dir = root / "state" / split
    param_dir = root / "param" / split
    state_dir.mkdir(parents=True, exist_ok=True)
    param_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    nz, ny, nx = grid
    if times is None:
        times = np.arange(t_len) * DT
    if param_times is None:
        param_times = times
    dims = ("time", "z", "y", "x")
    state: dict[str, tuple] = {
        v: (dims, rng.standard_normal((t_len, nz, ny, nx)).astype("f4"))
        for v in STATE_VARS
    }
    obstacle = np.zeros(grid, dtype="f4")
    obstacle[0, 1:3, 1 : 1 + nx // 4] = 1.0
    state["blanking"] = (("z", "y", "x"), obstacle)
    xr.Dataset(
        state,
        coords=dict(
            time=np.asarray(times, dtype="f8"),
            z=np.arange(nz),
            y=np.arange(ny),
            x=np.arange(nx),
        ),
    ).to_netcdf(state_dir / f"sample_{idx:04d}.nc")
    pvars = {
        "inflow_angle": (("time",), rng.uniform(-60, 60, t_len)),
        "velocity_magnitude": (("time",), rng.uniform(1, 5, t_len)),
        "pressure_gradient_magnitude": 1e-3 * (idx + 1),
    }
    if param_override:
        pvars.update(param_override)
    stem = param_stem or f"sample_{idx:04d}"
    xr.Dataset(pvars, coords=dict(time=np.asarray(param_times, dtype="f8"))).to_netcdf(
        param_dir / f"{stem}.nc"
    )


@pytest.fixture
def data_root(tmp_path: Path) -> Path:
    root = tmp_path / "data"
    for idx in range(N_TRAJ):
        _write_sample(root, idx, seed=100 + idx)
    return root


def _make_ds(root: Path, **overrides) -> SnapshotHistoryDataset:
    kwargs: dict = dict(
        root_dir=root,
        split="train",
        state_vars=STATE_VARS,
        param_vars=PARAM_VARS,
        param_history_steps=HP,
        geometry_var="blanking",
    )
    kwargs.update(overrides)
    return SnapshotHistoryDataset(**kwargs)


def _raw_params(root: Path, traj: int, names=PARAM_VARS) -> np.ndarray:
    """``(T, P)`` straight off the param file, bypassing the dataset."""
    path = root / "param" / "train" / f"sample_{traj:04d}.nc"
    with xr.open_dataset(path) as ds:
        t_len = ds.sizes["time"]
        cols = [np.broadcast_to(np.asarray(ds[n].values), (t_len,)) for n in names]
    return np.stack(cols, axis=-1).astype(np.float64)


def _raw_state(root: Path, traj: int, t: int) -> torch.Tensor:
    path = root / "state" / "train" / f"sample_{traj:04d}.nc"
    with xr.open_dataset(path) as ds:
        return torch.from_numpy(
            np.stack([np.asarray(ds[v].isel(time=t).values) for v in STATE_VARS])
        ).float()


# -- required knobs ---------------------------------------------------------


def test_required_knobs_and_unsupported_crop(data_root: Path) -> None:
    with pytest.raises(ValueError, match="param_vars"):
        _make_ds(data_root, param_vars=None)
    with pytest.raises(ValueError, match="param_history_steps"):
        _make_ds(data_root, param_history_steps=None)
    with pytest.raises(ValueError, match="param_history_steps"):
        _make_ds(data_root, param_history_steps=0)
    with pytest.raises(NotImplementedError, match="random_crop_size"):
        _make_ds(data_root, random_crop_size=4)


# -- items: oldest-first history, anchors, static params, ordering ----------


def test_items_oldest_first_history_matches_raw_file(data_root: Path) -> None:
    ds = _make_ds(data_root)
    assert ds.param_names == PARAM_VARS
    assert ds.param_history_steps == HP
    assert ds.history_dt_seconds == pytest.approx(DT)
    # Anchors start at t = Hp-1 by default.
    assert len(ds) == N_TRAJ * (T_LEN - HP + 1)
    assert ds.sample_index[0] == (0, HP - 1)
    assert all(t >= HP - 1 for _, t in ds.sample_index)

    for idx in range(len(ds)):
        traj, t = ds.sample_index[idx]
        item = ds[idx]
        assert set(item) == {"state", "geometry", "params_hist"}
        assert item["state"].shape == (len(STATE_VARS), *GRID)
        assert item["geometry"].shape == GRID
        assert item["params_hist"].shape == (HP, len(PARAM_VARS))
        torch.testing.assert_close(item["state"], _raw_state(data_root, traj, t))
        expected = _raw_params(data_root, traj)[t - HP + 1 : t + 1]
        np.testing.assert_allclose(
            item["params_hist"].double().numpy(), expected, rtol=1e-6
        )
    # The last row of the history is the snapshot's own parameters.
    traj, t = ds.sample_index[-1]
    np.testing.assert_allclose(
        ds[-1]["params_hist"][-1].double().numpy(), _raw_params(data_root, traj)[t]
    )


def test_static_scalar_param_broadcast_across_history(data_root: Path) -> None:
    ds = _make_ds(data_root)
    col = PARAM_VARS.index("pressure_gradient_magnitude")
    for idx in range(len(ds)):
        traj, _ = ds.sample_index[idx]
        column = ds[idx]["params_hist"][:, col]
        assert torch.all(column == column[0])
        assert column[0].item() == pytest.approx(1e-3 * (traj + 1))


def test_explicit_variable_ordering_is_honoured(data_root: Path) -> None:
    fwd = _make_ds(data_root)
    rev = _make_ds(data_root, param_vars=tuple(reversed(PARAM_VARS)))
    assert rev.param_names == tuple(reversed(PARAM_VARS))
    for idx in (0, len(fwd) - 1):
        torch.testing.assert_close(
            rev[idx]["params_hist"], fwd[idx]["params_hist"].flip(-1)
        )
    sub = _make_ds(data_root, param_vars=("velocity_magnitude",))
    assert sub[0]["params_hist"].shape == (HP, 1)
    torch.testing.assert_close(sub[0]["params_hist"][:, 0], fwd[0]["params_hist"][:, 1])


def test_constant_prehistory_pads_with_first_row(data_root: Path) -> None:
    default = _make_ds(data_root)
    padded = _make_ds(data_root, constant_prehistory=True)
    assert len(padded) == N_TRAJ * T_LEN
    assert padded.sample_index[0] == (0, 0)
    raw = _raw_params(data_root, 0)
    # t=0: every row is the first recorded row.
    hist0 = padded[0]["params_hist"].double().numpy()
    np.testing.assert_allclose(hist0, np.repeat(raw[:1], HP, axis=0))
    # t=1: Hp-2 padded rows, then rows 0 and 1.
    hist1 = padded[1]["params_hist"].double().numpy()
    np.testing.assert_allclose(hist1[: HP - 2], np.repeat(raw[:1], HP - 2, axis=0))
    np.testing.assert_allclose(hist1[HP - 2 :], raw[:2])
    # Fully-recorded anchors are identical to the default dataset's items.
    for traj, t in default.sample_index:
        a = default[default.sample_index.index((traj, t))]
        b = padded[padded.sample_index.index((traj, t))]
        torch.testing.assert_close(a["params_hist"], b["params_hist"])
        torch.testing.assert_close(a["state"], b["state"])


def test_short_trajectory_needs_constant_prehistory(tmp_path: Path) -> None:
    root = tmp_path / "data"
    _write_sample(root, 0, t_len=2, seed=1)
    with pytest.raises(ValueError, match="param_history_steps"):
        _make_ds(root, param_history_steps=3)
    ds = _make_ds(root, param_history_steps=3, constant_prehistory=True)
    assert len(ds) == 2
    assert ds[1]["params_hist"].shape == (3, len(PARAM_VARS))


# -- pairing and validation ---------------------------------------------------


def test_missing_param_partner_raises(data_root: Path) -> None:
    (data_root / "param" / "train" / "sample_0001.nc").unlink()
    with pytest.raises(FileNotFoundError, match="sample_0001"):
        _make_ds(data_root)


def test_mismatched_sample_ids_raise(data_root: Path) -> None:
    # Same file *count*, different ids: sorted-position pairing would silently
    # align sample_0001's state with sample_0007's params.
    param_dir = data_root / "param" / "train"
    (param_dir / "sample_0001.nc").rename(param_dir / "sample_0007.nc")
    with pytest.raises(FileNotFoundError, match="sample_0001"):
        _make_ds(data_root)


def test_mismatched_time_coordinates_raise(tmp_path: Path) -> None:
    root = tmp_path / "data"
    _write_sample(root, 0, seed=1)
    _write_sample(root, 1, seed=2, param_times=np.arange(T_LEN) * DT + 0.5)
    with pytest.raises(ValueError, match="time coordinates differ.*sample_0001"):
        _make_ds(root)
    # A param file with one extra saved time (same cadence) is caught by the
    # time-coordinate check before any parameter is read.
    root2 = tmp_path / "data2"
    _write_sample(root2, 0, seed=1)
    _write_sample(root2, 1, seed=2)
    longer = np.arange(T_LEN + 1) * DT
    xr.Dataset(
        {
            "inflow_angle": (("time",), np.linspace(-10.0, 10.0, T_LEN + 1)),
            "velocity_magnitude": (("time",), np.linspace(1.0, 2.0, T_LEN + 1)),
            "pressure_gradient_magnitude": 1e-3,
        },
        coords=dict(time=longer),
    ).to_netcdf(root2 / "param" / "train" / "sample_0001.nc")
    with pytest.raises(ValueError, match="time coordinates differ.*sample_0001"):
        _make_ds(root2)


def test_missing_time_coordinate_raises(tmp_path: Path) -> None:
    root = tmp_path / "data"
    _write_sample(root, 0, seed=1)
    path = root / "param" / "train" / "sample_0000.nc"
    with xr.open_dataset(path) as ds:
        stripped = ds.load().drop_vars("time")
    path.unlink()
    stripped.to_netcdf(path)
    with pytest.raises(ValueError, match="'time' coordinate"):
        _make_ds(root)


def test_non_finite_params_raise(tmp_path: Path) -> None:
    root = tmp_path / "data"
    vals = np.linspace(1.0, 2.0, T_LEN)
    vals[3] = np.nan
    _write_sample(
        root, 0, seed=1, param_override={"velocity_magnitude": (("time",), vals)}
    )
    with pytest.raises(ValueError, match="non-finite.*velocity_magnitude"):
        _make_ds(root)


def test_non_increasing_times_raise(tmp_path: Path) -> None:
    root = tmp_path / "data"
    times = np.arange(T_LEN) * DT
    times[4] = times[3]
    _write_sample(root, 0, seed=1, times=times)
    with pytest.raises(ValueError, match="strictly increasing"):
        _make_ds(root)


# -- cadence ------------------------------------------------------------------


def test_cadence_jitter_within_tolerance_gives_median(tmp_path: Path) -> None:
    root = tmp_path / "data"
    rng = np.random.default_rng(3)
    all_dts = []
    for idx in range(N_TRAJ):
        # pyudales_idealized-style saved cadence: ~5 s with ~3% jitter.
        dts = DT * (1.0 + rng.uniform(-0.03, 0.03, T_LEN - 1))
        all_dts.append(dts)
        times = np.concatenate([[0.0], np.cumsum(dts)])
        _write_sample(root, idx, seed=idx, times=times)
    ds = _make_ds(root)
    assert ds.history_dt_seconds == pytest.approx(
        float(np.median(np.concatenate(all_dts)))
    )
    # Tightening the tolerance below the jitter rejects the same data.
    with pytest.raises(ValueError, match="inconsistent saved cadence"):
        _make_ds(root, cadence_rtol=0.001)


def test_cadence_outlier_raises_with_sample_and_index(tmp_path: Path) -> None:
    root = tmp_path / "data"
    _write_sample(root, 0, seed=1)
    times = np.arange(T_LEN) * DT
    times[5:] += 0.5 * DT  # one 7.5 s gap: 50% off the 5 s median
    _write_sample(root, 1, seed=2, times=times)
    with pytest.raises(ValueError, match=r"sample_0001.*dt\[4\]"):
        _make_ds(root)


# -- time_stride ----------------------------------------------------------------


def test_time_stride_thins_anchors_only(data_root: Path) -> None:
    full = _make_ds(data_root)
    strided = _make_ds(data_root, time_stride=2)
    expected = [(traj, t) for traj in range(N_TRAJ) for t in range(HP - 1, T_LEN, 2)]
    assert strided.sample_index == expected
    for idx, (traj, t) in enumerate(strided.sample_index):
        # Histories stay contiguous in saved steps (not strided).
        ref = full[full.sample_index.index((traj, t))]
        torch.testing.assert_close(strided[idx]["params_hist"], ref["params_hist"])
        np.testing.assert_allclose(
            strided[idx]["params_hist"].double().numpy(),
            _raw_params(data_root, traj)[t - HP + 1 : t + 1],
        )


# -- collate ------------------------------------------------------------------


def test_collate_ships_shared_geometry_once_and_stacks_history(
    data_root: Path,
) -> None:
    ds = _make_ds(data_root, sdf_features="both", sdf_clamp_cells=4.0)
    items = [ds[i] for i in range(3)]
    assert items[0]["geometry"] is items[1]["geometry"]  # deduped mask
    batch = snapshot_history_collate(items)
    assert batch["state"].shape == (3, len(STATE_VARS), *GRID)
    assert batch["geometry"].shape == (1, *GRID)
    assert batch["geom_features"].shape == (1, 4, *GRID)
    assert batch["params_hist"].shape == (3, HP, len(PARAM_VARS))
    for i, item in enumerate(items):
        torch.testing.assert_close(batch["params_hist"][i], item["params_hist"])


def test_prepare_snapshot_batch_on_history_batch(data_root: Path) -> None:
    ds = _make_ds(data_root, sdf_features="sdf", sdf_clamp_cells=4.0)
    loader = DataLoader(ds, batch_size=4, collate_fn=snapshot_history_collate)
    batch = next(iter(loader))
    model = torch.nn.Linear(1, 1)
    dummy = DataLoader([0, 1], batch_size=1)
    trainer = BaseTraining(
        model=model,
        train_loader=dummy,
        val_loader=dummy,
        optimizer=torch.optim.SGD(model.parameters(), lr=0.1),
        loss_fn=torch.nn.MSELoss(),
        num_epochs=1,
        device="cpu",
    )
    state, geometry, features = trainer._prepare_snapshot_batch(batch)
    assert state.shape == (4, len(STATE_VARS), *GRID)
    assert geometry.shape == (4, *GRID)
    assert features is not None and features.shape == (4, 1, *GRID)
    torch.testing.assert_close(geometry[0], ds.geometry_for(0))
    # Cached shared geometry is reused on the next content-equal batch.
    cached = trainer._geometry
    trainer._prepare_snapshot_batch(batch)
    assert trainer._geometry is cached
    # params_hist is untouched by the shared prep (the generator trainer moves
    # it itself) and is still (B, Hp, P).
    assert batch["params_hist"].shape == (4, HP, len(PARAM_VARS))


# -- multi-geometry ---------------------------------------------------------------


def test_multi_geometry_bucketing_with_trajectory_batch_sampler(
    tmp_path: Path,
) -> None:
    root = tmp_path / "data"
    grids = [(4, 6, 8), (4, 6, 8), (4, 8, 12)]
    for idx, grid in enumerate(grids):
        _write_sample(root, idx, grid=grid, seed=10 + idx)
    ds = _make_ds(root)
    assert ds.geometry_for(0) is ds.geometry_for(1)
    assert ds.grid_shape(2) == (4, 8, 12)
    sampler = TrajectoryBatchSampler(ds, batch_size=4, shuffle=True, seed=0)
    loader = DataLoader(ds, batch_sampler=sampler, collate_fn=snapshot_history_collate)
    seen = 0
    for batch in loader:
        b = batch["state"].shape[0]
        seen += b
        assert batch["geometry"].shape[0] == 1
        assert batch["state"].shape[2:] == batch["geometry"].shape[1:]
        assert batch["params_hist"].shape == (b, HP, len(PARAM_VARS))
    assert seen == len(ds)


# -- normalization stats ----------------------------------------------------------


def test_get_normalization_stats_covers_all_saved_times(data_root: Path) -> None:
    ds = _make_ds(data_root)
    state_mean, state_std, param_mean, param_std = get_normalization_stats(ds)
    assert state_mean.shape == state_std.shape == (len(STATE_VARS),)
    all_rows = np.concatenate([_raw_params(data_root, i) for i in range(N_TRAJ)])
    np.testing.assert_allclose(param_mean, all_rows.mean(axis=0), rtol=1e-5)
    np.testing.assert_allclose(param_std, all_rows.std(axis=0), rtol=1e-5)
    # Stats are keyed on the dataset class, so the cached file names it.
    cached = np.load(data_root / "normalization_stats" / "train.npz")
    assert "SnapshotHistoryDataset" in str(cached["signature"])


# -- shared reader + unchanged TransitionDataset --------------------------------


def test_load_param_table_matches_transition_dataset(data_root: Path) -> None:
    tds = TransitionDataset(
        data_root, "train", state_vars=STATE_VARS, param_vars=PARAM_VARS
    )
    assert tds.param_names == PARAM_VARS
    for traj in range(N_TRAJ):
        table, names = load_param_table(
            data_root / "param" / "train" / f"sample_{traj:04d}.nc",
            T_LEN,
            PARAM_VARS,
            torch.float32,
        )
        assert names == PARAM_VARS
        torch.testing.assert_close(tds._params[traj], table)
        np.testing.assert_allclose(
            table.double().numpy(), _raw_params(data_root, traj), rtol=1e-6
        )
    # Items still slice that table: params_n == table[t:t+K].
    for idx in (0, len(tds) - 1):
        traj, t = tds.sample_index[idx]
        torch.testing.assert_close(tds[idx]["params_n"], tds._params[traj][t : t + 1])
    # The history dataset's tables equal TransitionDataset's on the same split.
    hds = _make_ds(data_root)
    for a, b in zip(hds._params, tds._params):
        torch.testing.assert_close(a, b)
    # ``param_vars=None`` still takes every data variable in file order.
    table, names = load_param_table(
        data_root / "param" / "train" / "sample_0000.nc", T_LEN, None, torch.float64
    )
    assert names == PARAM_VARS and table.dtype == torch.float64


def test_load_param_table_errors_unchanged(tmp_path: Path) -> None:
    path = tmp_path / "sample_0000.nc"
    xr.Dataset(
        {
            "inflow_angle": (("time",), np.arange(5.0)),
            "matrix": (("time", "k"), np.zeros((5, 2))),
        }
    ).to_netcdf(path)
    with pytest.raises(ValueError, match="has length 5, expected 4"):
        load_param_table(path, 4, ("inflow_angle",), torch.float32)
    with pytest.raises(ValueError, match="unsupported shape"):
        load_param_table(path, 5, ("matrix",), torch.float32)
