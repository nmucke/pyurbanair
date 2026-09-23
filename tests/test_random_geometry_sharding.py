"""Sharded random-geometry generation must reproduce the one-process corpus.

Drives generate_random_geometries_training_data.run() end to end with the
solver swapped for a deterministic fake, so the test exercises the plan /
simulate / finalize bookkeeping (shard assignment, resume, frozen config)
rather than CFD.
"""

from __future__ import annotations

import pathlib
import sys
from typing import Any

import numpy as np
import pytest
import trimesh
import xarray as xr
from hydra import compose, initialize
from omegaconf import DictConfig

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "scripts"))
sys.path.insert(
    0, str(pathlib.Path(__file__).resolve().parents[1] / "scripts" / "neural_surrogate")
)

import generate_random_geometries_training_data as gen  # noqa: E402

_NUM_SHARDS = 3


class _FakeForwardModel:
    """Writes a state whose values depend on the geometry, params and spinup."""

    def __init__(self, **kw: Any) -> None:
        self.kw = kw

    def __call__(self, *, state: object, params: xr.Dataset, sim_name: str) -> None:
        kw = self.kw
        if kw.get("fail"):
            raise RuntimeError("injected solver failure")
        dt = kw["output_frequency"]
        time = np.arange(0.0, kw["simulation_time"] + dt / 2, dt)
        speed = np.interp(
            time, params["time"].values, params["velocity_magnitude"].values
        )
        shape = (time.size, kw["nz"], kw["ny"], kw["nx"])
        base = speed[:, None, None, None] + kw["spinup_time"] / 1000.0
        ds = xr.Dataset(
            {
                name: (("time", "z", "y", "x"), np.broadcast_to(base + k, shape))
                for k, name in enumerate(("u", "v", "w", "pres"))
            },
            coords={"time": time},
        )
        results_dir = pathlib.Path(kw["results_dir"])
        results_dir.mkdir(parents=True, exist_ok=True)
        ds.to_netcdf(results_dir / f"{sim_name}.nc")


@pytest.fixture  # type: ignore[misc]
def pool_dir(tmp_path: pathlib.Path) -> pathlib.Path:
    stl_dir = tmp_path / "pool"
    stl_dir.mkdir()
    rows = ["name,Lx_m,Ly_m,z_max_m"]
    for i, (lx, ly) in enumerate([(32, 32), (64, 32), (48, 48), (96, 64), (32, 80)]):
        box = trimesh.creation.box(extents=(8.0, 8.0, 8.0))
        box.apply_translation((lx / 2, ly / 2, 4.0))
        box.export(stl_dir / f"g{i}.stl")
        rows.append(f"g{i},{lx},{ly},8.0")
    (stl_dir / "manifest.csv").write_text("\n".join(rows) + "\n")
    return stl_dir


@pytest.fixture  # type: ignore[misc]
def fake_solver(monkeypatch: pytest.MonkeyPatch) -> set[str]:
    """Patch the solver; returns a mutable set of geometry names to fail."""
    failing: set[str] = set()

    def _instantiate(node: object, **kw: Any) -> _FakeForwardModel | None:
        if "forward_model" in kw:  # the `prepare` call
            return None
        stl = pathlib.Path(kw.get("stl_path", ""))
        return _FakeForwardModel(**kw, fail=stl.stem in failing)

    monkeypatch.setattr(gen, "instantiate", _instantiate)
    monkeypatch.setattr(gen, "clean_outputs", lambda **_: None)
    monkeypatch.setattr(gen, "animate_state", lambda **_: None)
    return failing


def _cfg(pool_dir: pathlib.Path, out: pathlib.Path, *extra: str) -> DictConfig:
    with initialize(version_base=None, config_path="../conf"):
        return compose(
            config_name="neural_surrogate/training_data",
            overrides=[
                "model=pylbm",
                "training_data.geometry.source=idealized",
                f"training_data.geometry.stl_dir={pool_dir}",
                "training_data.geometry.resolution=4.0",
                "training_data.geometry.z_size=64.0",
                "training_data.geometry.upstream_padding=0.0",
                "training_data.geometry.downstream_padding=0.0",
                "training_data.geometry.lateral_padding=0.0",
                # 3 training geometries for 7 sims -> multi-sample groups.
                "training_data.num_train=7",
                "training_data.num_val=1",
                "training_data.num_test=1",
                "training_data.simulation_time=20.0",
                "training_data.output_frequency=5.0",
                "training_data.spinup_time=5.0",
                "training_data.adaptive_spinup.max_spinup_time=100.0",
                "training_data.params_sampler.seconds_per_knot=10.0",
                f"training_data.output_dir={out}",
                f"paths.experiment_dir={out.parent / (out.name + '_scratch')}",
                *extra,
            ],
        )


def _stage(pool_dir: pathlib.Path, out: pathlib.Path, stage: str, *extra: str) -> None:
    gen.run(
        _cfg(
            pool_dir,
            out,
            f"training_data.sharding.stage={stage}",
            f"training_data.sharding.num_shards={_NUM_SHARDS}",
            *extra,
        )
    )


def _sample_files(root: pathlib.Path) -> list[pathlib.Path]:
    return sorted(p.relative_to(root) for p in root.glob("*/*/sample_*.nc"))


def test_sharded_matches_single_process(
    tmp_path: pathlib.Path, pool_dir: pathlib.Path, fake_solver: set[str]
) -> None:
    single = tmp_path / "single"
    gen.run(_cfg(pool_dir, single))

    sharded = tmp_path / "sharded"
    _stage(pool_dir, sharded, "plan")
    # Out of order on purpose: shard results must not depend on run order.
    for k in reversed(range(_NUM_SHARDS)):
        _stage(pool_dir, sharded, "simulate", f"training_data.sharding.shard_index={k}")
    _stage(pool_dir, sharded, "finalize")

    files = _sample_files(single)
    assert len(files) == 2 * 9
    assert files == _sample_files(sharded)
    for rel in files + [pathlib.Path("params.nc"), pathlib.Path("sampled_params.nc")]:
        xr.testing.assert_identical(
            xr.load_dataset(single / rel), xr.load_dataset(sharded / rel)
        )
    assert (single / "geometries.csv").read_text() == (
        sharded / "geometries.csv"
    ).read_text()
    assert "sharding" not in (sharded / "config.yaml").read_text()


def test_shards_partition_the_groups(
    tmp_path: pathlib.Path, pool_dir: pathlib.Path
) -> None:
    cfg = _cfg(pool_dir, tmp_path / "out")
    groups = gen._group_samples(gen._plan_samples(cfg))
    costs = [float(len(g) * (i + 1)) for i, g in enumerate(groups)]
    shard_of = gen._assign_shards(costs, _NUM_SHARDS)
    assert shard_of == gen._assign_shards(costs, _NUM_SHARDS)
    assert sorted(set(shard_of)) == list(range(_NUM_SHARDS))
    assert len(shard_of) == len(groups)


def test_resume_and_failures(
    tmp_path: pathlib.Path, pool_dir: pathlib.Path, fake_solver: set[str]
) -> None:
    out = tmp_path / "out"
    _stage(pool_dir, out, "plan")
    for k in range(_NUM_SHARDS):
        _stage(pool_dir, out, "simulate", f"training_data.sharding.shard_index={k}")
    reference = {p: xr.load_dataset(out / p) for p in _sample_files(out)}

    # A sample lost mid-run (e.g. shard killed at the time limit) blocks
    # finalize, which names the shard to resubmit.
    lost = next(p for p in reference if p.parts[0] == "param")
    (out / lost).unlink()
    with pytest.raises(RuntimeError, match="shard"):
        _stage(pool_dir, out, "finalize")

    # Rerunning every shard redoes only the missing sample, identically.
    stl = xr.load_dataset(out / "state" / lost.parts[1] / lost.name).attrs[
        "geometry_stl"
    ]
    fake_solver.add(pathlib.Path(stl).stem)
    with pytest.raises(RuntimeError, match="1 geometry group"):
        for k in range(_NUM_SHARDS):
            _stage(pool_dir, out, "simulate", f"training_data.sharding.shard_index={k}")
    fake_solver.clear()
    for k in range(_NUM_SHARDS):
        _stage(pool_dir, out, "simulate", f"training_data.sharding.shard_index={k}")
    for p, ds in reference.items():
        xr.testing.assert_identical(ds, xr.load_dataset(out / p))
    _stage(pool_dir, out, "finalize")
    assert (out / "params.nc").exists()


def test_plan_refuses_different_config(
    tmp_path: pathlib.Path, pool_dir: pathlib.Path, fake_solver: set[str]
) -> None:
    out = tmp_path / "out"
    _stage(pool_dir, out, "plan")
    # Same config (runtime keys aside) is reused, not redrawn.
    _stage(pool_dir, out, "plan", "model.forward_model.verbose=true")
    with pytest.raises(ValueError, match="training_data.seed"):
        _stage(pool_dir, out, "plan", "training_data.seed=1")
