"""Tests for :class:`PatchTransitionDataset` over a synthetic pylbm layout."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import xarray as xr
from neural_surrogates.datasets.patch import PatchTransitionDataset
from neural_surrogates.decomposition import DomainDecomposition

# Small grid where interior_size (4) divides the periodic (y) axis length (8).
NZ, NY, NX = 8, 8, 8
T_LEN = 4
STATE_VARS = ("u", "v", "w")
PARAM_VARS = ("inflow_angle", "velocity_magnitude")

DD_KWARGS = dict(
    interior_size=4,
    halo=2,
    taper=1,
    coarsen_factor=2,
    n_pos=3,
    periodic_axes=(False, True, False),  # (z, y, x) -> y periodic
)


def _write_sample(
    state_dir: Path,
    param_dir: Path,
    idx: int,
    *,
    with_blanking: bool,
    seed: int,
) -> None:
    state_dir.mkdir(parents=True, exist_ok=True)
    param_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)

    dims = ("time", "z", "y", "x")
    shape = (T_LEN, NZ, NY, NX)
    data_vars = {
        v: (dims, rng.standard_normal(shape).astype(np.float32)) for v in STATE_VARS
    }
    if with_blanking:
        # Obstacle indicator: 1 inside a small solid block, 0 in fluid.
        obstacle = np.zeros((NZ, NY, NX), dtype=np.float32)
        obstacle[0:2, 2:4, 2:4] = 1.0
        obstacle = np.broadcast_to(obstacle, shape).copy()
        data_vars["blanking"] = (dims, obstacle)
    else:
        # Fallback path: exact zeros inside an obstacle so the non-zero-state
        # heuristic recovers a mask. Zero a block across every state var/time.
        for v in STATE_VARS:
            data_vars[v][1][:, 0:2, 2:4, 2:4] = 0.0

    xr.Dataset(data_vars).to_netcdf(state_dir / f"sample_{idx:04d}.nc")

    pdims = ("time",)
    pvars = {
        "inflow_angle": (pdims, rng.uniform(-60, 60, T_LEN).astype(np.float32)),
        "velocity_magnitude": (
            pdims,
            rng.uniform(1, 5, T_LEN).astype(np.float32),
        ),
    }
    xr.Dataset(pvars).to_netcdf(param_dir / f"sample_{idx:04d}.nc")


@pytest.fixture
def blanking_root(tmp_path: Path) -> Path:
    root = tmp_path / "blanking"
    for idx in range(2):
        _write_sample(
            root / "state" / "train",
            root / "param" / "train",
            idx,
            with_blanking=True,
            seed=100 + idx,
        )
    return root


@pytest.fixture
def fallback_root(tmp_path: Path) -> Path:
    root = tmp_path / "fallback"
    for idx in range(2):
        _write_sample(
            root / "state" / "train",
            root / "param" / "train",
            idx,
            with_blanking=False,
            seed=200 + idx,
        )
    return root


def _make_ds(root: Path, **overrides) -> PatchTransitionDataset:
    kwargs = dict(
        root_dir=root,
        split="train",
        state_vars=STATE_VARS,
        param_vars=PARAM_VARS,
        geometry_var="blanking",
        pushforward_steps=1,
        decomposition=dict(DD_KWARGS),
    )
    kwargs.update(overrides)
    return PatchTransitionDataset(**kwargs)


def test_length(blanking_root: Path) -> None:
    ds = _make_ds(blanking_root)
    dd = DomainDecomposition(**DD_KWARGS)
    import torch

    m = dd.num_patches(torch.zeros(1, 1, NZ, NY, NX))
    # 2 trajectories, (T - K) start times each, M patches each.
    assert ds.num_patches == m
    assert len(ds) == 2 * (T_LEN - 1) * m


def test_item_keys_and_shapes(blanking_root: Path) -> None:
    ds = _make_ds(blanking_root)
    n = DD_KWARGS["interior_size"]
    h = DD_KWARGS["halo"]
    ext = n + 2 * h
    c = len(STATE_VARS)
    dd = DomainDecomposition(**DD_KWARGS)
    import torch

    coarse_grid = tuple(dd.plan(torch.zeros(1, 1, NZ, NY, NX)).coarse)

    item = ds[0]
    expected_keys = {
        "state_n_patch",
        "delta_target",
        "geometry_patch",
        "positional",
        "coarse_input",
        "coarse_geom",
        "coarse_target",
        "neighbors",
        "params_n",
        "patch_index",
        "traj_index",
    }
    assert set(item) == expected_keys

    assert item["state_n_patch"].shape == (c, ext, ext, ext)
    assert item["delta_target"].shape == (c, n, n, n)
    assert item["geometry_patch"].shape == (1, ext, ext, ext)
    assert item["positional"].shape == (DD_KWARGS["n_pos"], ext, ext, ext)
    assert item["coarse_input"].shape == (c, *coarse_grid)
    assert item["coarse_geom"].shape == (1, *coarse_grid)
    assert item["coarse_target"].shape == (c, *coarse_grid)
    assert item["neighbors"].shape == (6,)
    assert item["params_n"].shape == (1, len(PARAM_VARS))
    assert item["patch_index"].shape == ()
    assert item["traj_index"].shape == ()


def test_delta_target_matches_raw_netcdf(blanking_root: Path) -> None:
    """delta_target == interior(S_{t+K}) - interior(S_t) from raw arrays."""
    import torch

    ds = _make_ds(blanking_root)
    n = DD_KWARGS["interior_size"]
    counts = (NZ // n, NY // n, NX // n)  # (2, 2, 2) here

    # Pick a known (traj, t, patch): traj 1, t 1, patch (pz,py,px)=(1,0,1).
    traj, t = 1, 1
    pz, py, px = 1, 0, 1
    p = (pz * counts[1] + py) * counts[2] + px

    # Locate the flat dataset index for this (traj, t, p).
    flat = ds._patch_index.index((traj, t, p))
    item = ds[flat]

    raw = xr.open_dataset(blanking_root / "state" / "train" / f"sample_{traj:04d}.nc")
    # interior slice for this patch (stride n, no halo).
    zsl = slice(pz * n, pz * n + n)
    ysl = slice(py * n, py * n + n)
    xsl = slice(px * n, px * n + n)
    s_t = np.stack(
        [raw[v].isel(time=t).values[zsl, ysl, xsl] for v in STATE_VARS], axis=0
    )
    s_tk = np.stack(
        [raw[v].isel(time=t + 1).values[zsl, ysl, xsl] for v in STATE_VARS],
        axis=0,
    )
    raw.close()
    expected = torch.from_numpy((s_tk - s_t).astype(np.float32))

    assert torch.allclose(item["delta_target"], expected, atol=1e-6)
    assert int(item["patch_index"]) == p
    assert int(item["traj_index"]) == traj


def test_neighbors_periodic_y_and_boundary(blanking_root: Path) -> None:
    """y-axis neighbours wrap; x/z boundary neighbours are -1."""
    ds = _make_ds(blanking_root)
    cz, cy, cx = NZ // 4, NY // 4, NX // 4  # (2, 2, 2)

    def flat(pz: int, py: int, px: int) -> int:
        return (pz * cy + py) * cx + px

    # Interior-ish corner patch (0, 0, 0): order is (-z,+z,-y,+y,-x,+x).
    p = flat(0, 0, 0)
    item = ds[ds._patch_index.index((0, 0, p))]
    nb = item["neighbors"].tolist()
    # -z boundary -> -1, +z -> (1,0,0)
    assert nb[0] == -1
    assert nb[1] == flat(1, 0, 0)
    # y periodic: -y of py=0 wraps to py=1, +y also wraps to py=1
    assert nb[2] == flat(0, 1, 0)
    assert nb[3] == flat(0, 1, 0)
    # -x boundary -> -1, +x -> (0,0,1)
    assert nb[4] == -1
    assert nb[5] == flat(0, 0, 1)

    # A patch whose every face has a valid neighbour exists only because y
    # wraps; with 2x2x2 every patch touches a boundary on z and x. Confirm the
    # number of valid (non -1) faces: z and x each contribute exactly one
    # in-grid neighbour, y contributes two (wrapped) -> 4 valid faces.
    assert sum(1 for v in nb if v != -1) == 4


def test_blanking_geometry_present(blanking_root: Path) -> None:
    ds = _make_ds(blanking_root)
    # Geometry mask should mark the solid block as 0 (obstacle).
    assert float(ds.geometry_for(0)[0, 2, 2]) == 0.0
    assert float(ds.geometry_for(0)[4, 0, 0]) == 1.0
    # Patch geometry is restricted from the same mask.
    assert ds[0]["geometry_patch"].shape[0] == 1


def test_fallback_geometry_constructs(fallback_root: Path) -> None:
    # geometry_var present in kwargs but absent in the file -> non-zero-state
    # fallback path must construct and yield items.
    ds = _make_ds(fallback_root)
    assert len(ds) == 2 * (T_LEN - 1) * ds.num_patches
    item = ds[0]
    assert item["geometry_patch"].shape[0] == 1
    # The zeroed obstacle block is recovered as 0 in the fallback mask.
    assert float(ds.geometry_for(0)[0, 2, 2]) == 0.0
