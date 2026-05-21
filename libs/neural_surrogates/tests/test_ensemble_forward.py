"""D2 ensemble batching: in-memory concat, on-disk files, Path-based state."""

from __future__ import annotations

import numpy as np
import xarray

from neural_surrogates.data.generate import CorpusWriter
from neural_surrogates.data.grid import GridMeta, build_static_channels
from neural_surrogates.ensemble_forward_model import EnsembleForwardModel
from neural_surrogates.forward_model import ForwardModel
from neural_surrogates.training.train import run as train_run
from neural_surrogates.utils.schema import ContractSchema, ParamSchema


def _make_checkpoint(tmp_path):
    z, y, x = 4, 4, 4
    grid = GridMeta(nx=x, ny=y, nz=z, bounds=((0.0, x), (0.0, y), (0.0, z)))
    schema = ParamSchema(names=("inflow_angle", "velocity_magnitude"))
    var_names = ("u", "v", "w")
    mask = np.zeros((z, y, x), dtype=np.float32)
    static = build_static_channels(mask, grid=grid)
    contract = ContractSchema("pylbm", schema, var_names)
    corpus_path = tmp_path / "corpus"
    writer = CorpusWriter(corpus_path, grid, contract, var_names, static, mask)
    rng = np.random.default_rng(0)
    for i in range(4):
        f = rng.standard_normal((8, 3, z, y, x)).astype(np.float32)
        p = rng.standard_normal((8, schema.conditioning_dim)).astype(np.float32)
        writer.add_trajectory(f"t{i}", f, p, np.arange(8.0), "train" if i < 3 else "val")
    writer.finalize(extra={"source_solver_name": "pylbm"})
    ckpt = tmp_path / "ckpt"
    train_run({
        "corpus_path": str(corpus_path), "checkpoint_dir": str(ckpt),
        "history_len": 1, "num_epochs": 1, "batch_size": 2,
        "horizon_schedule": [{"epoch": 0, "horizon": 1, "warmup": 0}],
        "arch": {"name": "unet3d", "base_channels": 4,
                 "channel_multipliers": [1, 2], "embed_dim": 8},
    })
    return ckpt, grid


def _params_ensemble(n: int) -> xarray.Dataset:
    return xarray.Dataset(
        data_vars={
            "inflow_angle": ("ensemble", np.linspace(0, 90, n)),
            "velocity_magnitude": ("ensemble", np.linspace(1, 5, n)),
        },
        coords={"ensemble": np.arange(n)},
    )


def _forward_model(ckpt, grid, results_dir=None):
    return ForwardModel(
        ckpt, nx=grid.nx, ny=grid.ny, nz=grid.nz, bounds=grid.bounds,
        simulation_time=3.0, output_frequency=1.0, device="cpu",
        results_dir=results_dir,
    )


def test_ensemble_in_memory_concat(tmp_path) -> None:
    ckpt, grid = _make_checkpoint(tmp_path)
    fm = _forward_model(ckpt, grid)
    ens = EnsembleForwardModel(fm, ensemble_size=4)
    out = ens.run_ensemble(params=_params_ensemble(4), sim_name="state")
    assert out.sizes["ensemble"] == 4
    assert out.sizes["time"] == 3
    assert set(("u", "v", "w")).issubset(out.data_vars)


def test_ensemble_chunked_matches_unchunked(tmp_path) -> None:
    ckpt, grid = _make_checkpoint(tmp_path)
    params = _params_ensemble(4)
    full = EnsembleForwardModel(_forward_model(ckpt, grid), ensemble_size=4).run_ensemble(
        params=params
    )
    chunked = EnsembleForwardModel(
        _forward_model(ckpt, grid), ensemble_size=4, vmap_chunk_size=2
    ).run_ensemble(params=params)
    np.testing.assert_allclose(
        full["u"].values, chunked["u"].values, rtol=1e-4, atol=1e-5
    )


def test_ensemble_on_disk_writes_per_member(tmp_path) -> None:
    ckpt, grid = _make_checkpoint(tmp_path)
    results_dir = tmp_path / "results"
    fm = _forward_model(ckpt, grid, results_dir=results_dir)
    ens = EnsembleForwardModel(fm, ensemble_size=3, results_dir=results_dir)
    out = ens.run_ensemble(params=_params_ensemble(3), sim_name="state")
    assert out is None
    for i in range(3):
        assert (results_dir / f"state_{i}.nc").exists()
    # get_states re-opens them and concatenates
    stacked = ens.get_states()
    assert stacked.sizes["ensemble"] == 3
