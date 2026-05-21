"""End-to-end Zarr-corpus + trainer + checkpoint test (CPU, tiny model).

Exercises generate.CorpusWriter/open_corpus, the curriculum loop in
training.train.run, and checkpoint save — the P1+P2 plumbing wired together.
"""

from __future__ import annotations

import numpy as np

from neural_surrogates.data.generate import CorpusWriter, open_corpus
from neural_surrogates.data.grid import GridMeta, build_static_channels
from neural_surrogates.training.checkpoint import load_checkpoint
from neural_surrogates.training.train import run as train_run
from neural_surrogates.utils.schema import ContractSchema, ParamSchema


def _write_synthetic_corpus(path) -> None:
    z, y, x = 4, 4, 4
    grid = GridMeta(nx=x, ny=y, nz=z, bounds=((0.0, x), (0.0, y), (0.0, z)))
    schema = ParamSchema(names=("inflow_angle", "velocity_magnitude"))
    p = schema.conditioning_dim
    var_names = ("u", "v", "w")
    mask = np.zeros((z, y, x), dtype=np.float32)
    mask[:1, :2, :2] = 1.0
    static = build_static_channels(mask, grid=grid)
    contract = ContractSchema(
        source_solver_name="pylbm", param_schema=schema, state_var_names=var_names
    )

    writer = CorpusWriter(path, grid, contract, var_names, static, mask)
    rng = np.random.default_rng(0)
    for i in range(5):
        T = 10
        fields = rng.standard_normal((T, 3, z, y, x)).astype(np.float32)
        params = rng.standard_normal((T, p)).astype(np.float32)
        times = np.arange(T, dtype=float)
        split = "train" if i < 4 else "val"
        writer.add_trajectory(f"t{i}", fields, params, times, split)
    writer.finalize(extra={"source_solver_name": "pylbm"})


def test_train_run_writes_checkpoint(tmp_path) -> None:
    corpus_path = tmp_path / "corpus"
    _write_synthetic_corpus(corpus_path)

    # sanity: corpus reads back
    corpus = open_corpus(corpus_path)
    assert corpus.var_names == ("u", "v", "w")
    assert len(corpus.split_ids("train")) == 4

    cfg = {
        "corpus_path": str(corpus_path),
        "checkpoint_dir": str(tmp_path / "ckpt"),
        "history_len": 2,
        "seed": 0,
        "batch_size": 2,
        "stride": 1,
        "num_epochs": 3,
        "off_manifold_noise_std": 0.05,
        "optimizer": {"name": "adamw", "learning_rate": 1e-3, "weight_decay": 1e-5},
        "horizon_schedule": [
            {"epoch": 0, "horizon": 1, "warmup": 0},
            {"epoch": 2, "horizon": 3, "warmup": 1},
        ],
        "arch": {
            "name": "unet3d",
            "base_channels": 4,
            "channel_multipliers": [1, 2],
            "num_res_blocks_per_level": 1,
            "embed_dim": 8,
        },
    }
    metrics = train_run(cfg)

    assert len(metrics["train_loss"]) == 3
    assert all(np.isfinite(loss) for loss in metrics["train_loss"])
    assert len(metrics["rollout_error_vs_horizon"]) == 3  # final horizon = 3

    loaded = load_checkpoint(tmp_path / "ckpt", expected_architecture="unet3d")
    assert loaded.history_len == 2
    assert loaded.schema.source_solver_name == "pylbm"
    assert loaded.arch.history_len == 2
