"""P3 end-to-end: run_forward_model with model=neural_surrogate (CPU, no GPU).

Trains a tiny checkpoint on a synthetic corpus in ``tmp_path`` (no committed
binary), then drives ``scripts/run_forward_model.run`` through the real
pyurbanair Hydra config and asserts the surrogate honors the BaseForwardModel
contract (time-axis + grid + variables).
"""

from __future__ import annotations

import pathlib

import numpy as np
import pytest


def _train_tiny_checkpoint(tmp_path: pathlib.Path, grid_shape, ckpt_dir: pathlib.Path):
    from neural_surrogates.data.generate import CorpusWriter
    from neural_surrogates.data.grid import GridMeta, build_static_channels
    from neural_surrogates.training.train import run as train_run
    from neural_surrogates.utils.schema import ContractSchema, ParamSchema

    z, y, x = grid_shape
    grid = GridMeta(nx=x, ny=y, nz=z, bounds=((0.0, x), (0.0, y), (0.0, z)))
    schema = ParamSchema(names=("inflow_angle", "velocity_magnitude"))
    var_names = ("u", "v", "w")
    mask = np.zeros((z, y, x), dtype=np.float32)
    mask[:1, :2, :2] = 1.0
    static = build_static_channels(mask, grid=grid)
    contract = ContractSchema("pylbm", schema, var_names)

    corpus_path = tmp_path / "corpus"
    writer = CorpusWriter(corpus_path, grid, contract, var_names, static, mask)
    rng = np.random.default_rng(0)
    for i in range(5):
        fields = rng.standard_normal((10, 3, z, y, x)).astype(np.float32)
        params = rng.standard_normal((10, schema.conditioning_dim)).astype(np.float32)
        writer.add_trajectory(
            f"t{i}", fields, params, np.arange(10.0), "train" if i < 4 else "val"
        )
    writer.finalize(extra={"source_solver_name": "pylbm"})

    train_run(
        {
            "corpus_path": str(corpus_path),
            "checkpoint_dir": str(ckpt_dir),
            "history_len": 2,
            "num_epochs": 1,
            "batch_size": 2,
            "horizon_schedule": [{"epoch": 0, "horizon": 1, "warmup": 0}],
            "arch": {
                "name": "unet3d",
                "base_channels": 4,
                "channel_multipliers": [1, 2],
                "num_res_blocks_per_level": 1,
                "embed_dim": 8,
            },
        }
    )


def test_run_forward_model_neural_surrogate(tmp_path, compose_test_cfg) -> None:
    from scripts.run_forward_model import run

    grid_shape = (4, 6, 8)  # (Z, Y, X)
    ckpt_dir = tmp_path / "ckpt"
    _train_tiny_checkpoint(tmp_path, grid_shape, ckpt_dir)

    z, y, x = grid_shape
    cfg = compose_test_cfg(
        [
            "model=neural_surrogate",
            f"model.checkpoint_path={ckpt_dir}",
            "model.forward_model.device=cpu",
            "run.skip_viz=true",
            f"domain.nx={x}",
            f"domain.ny={y}",
            f"domain.nz={z}",
            f"domain.bounds=[[0.0,{x}.0],[0.0,{y}.0],[0.0,{z}.0]]",
            "time.simulation_time=3.0",
            "time.output_frequency=1.0",
        ]
    )

    run(cfg)  # state=None -> canned cold start; should not raise

    fm_state = None
    # Re-run directly to capture the returned state (run() does viz only).
    from hydra.utils import instantiate
    from pyurbanair.config.hydra_helpers import create_true_params

    forward_model = instantiate(cfg.model.forward_model, results_dir=None)
    forward_model.ensure_loaded()
    true_params = create_true_params(cfg.model.name, cfg.params.true)
    fm_state = forward_model(params=true_params)

    assert fm_state.sizes["time"] == 3
    assert set(("u", "v", "w")).issubset(fm_state.data_vars)
    assert fm_state["u"].transpose("time", "z", "y", "x").shape == (3, z, y, x)
    # solid cells (mask) are re-zeroed
    assert np.allclose(fm_state["u"].isel(z=0, y=0, x=0).values, 0.0)


def test_grid_mismatch_raises(tmp_path, compose_test_cfg) -> None:
    from hydra.utils import instantiate

    ckpt_dir = tmp_path / "ckpt"
    _train_tiny_checkpoint(tmp_path, (4, 6, 8), ckpt_dir)

    cfg = compose_test_cfg(
        [
            "model=neural_surrogate",
            f"model.checkpoint_path={ckpt_dir}",
            "model.forward_model.device=cpu",
            "domain.nx=9",  # wrong
            "domain.ny=6",
            "domain.nz=4",
            "domain.bounds=[[0.0,9.0],[0.0,6.0],[0.0,4.0]]",
        ]
    )
    forward_model = instantiate(cfg.model.forward_model, results_dir=None)
    with pytest.raises(ValueError, match="does not match the checkpoint grid"):
        forward_model.ensure_loaded()
