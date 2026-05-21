"""Smoke test for the GATE evaluation (computes the three bars; CPU, tiny)."""

from __future__ import annotations

import numpy as np

from neural_surrogates.data.generate import CorpusWriter
from neural_surrogates.data.grid import GridMeta, build_static_channels
from neural_surrogates.training.train import run as train_run
from neural_surrogates.utils.schema import ContractSchema, ParamSchema


def _make_corpus_and_ckpt(tmp_path):
    z, y, x = 4, 4, 4
    grid = GridMeta(nx=x, ny=y, nz=z, bounds=((0.0, x), (0.0, y), (0.0, z)))
    schema = ParamSchema(names=("inflow_angle", "velocity_magnitude"))
    var_names = ("u", "v", "w")
    mask = np.zeros((z, y, x), dtype=np.float32)
    static = build_static_channels(mask, grid=grid)
    corpus_path = tmp_path / "corpus"
    writer = CorpusWriter(
        corpus_path, grid, ContractSchema("pylbm", schema, var_names),
        var_names, static, mask,
    )
    rng = np.random.default_rng(0)
    for i in range(6):
        f = rng.standard_normal((12, 3, z, y, x)).astype(np.float32)
        p = rng.standard_normal((12, schema.conditioning_dim)).astype(np.float32)
        writer.add_trajectory(f"t{i}", f, p, np.arange(12.0), "train" if i < 4 else "val")
    writer.finalize(extra={"source_solver_name": "pylbm"})
    ckpt = tmp_path / "ckpt"
    train_run({
        "corpus_path": str(corpus_path), "checkpoint_dir": str(ckpt),
        "history_len": 1, "num_epochs": 1, "batch_size": 4,
        "horizon_schedule": [{"epoch": 0, "horizon": 1, "warmup": 0}],
        "arch": {"name": "unet3d", "base_channels": 4,
                 "channel_multipliers": [1, 2], "embed_dim": 8},
    })
    return ckpt, corpus_path


def test_gate_runs_and_reports_all_bars(tmp_path) -> None:
    from scripts.eval_surrogate_gate import evaluate_gate

    ckpt, corpus_path = _make_corpus_and_ckpt(tmp_path)
    results = evaluate_gate(
        ckpt, corpus_path, split="val", horizon=4, b1=1e9, b2=1e9, cold_start_max=1e9
    )
    assert len(results["clean_rollout_error"]) == results["horizon"]
    assert len(results["ood_rollout_error"]) == results["horizon"]
    # generous bars -> all pass; structure is what we assert here
    assert results["pass_b1"] and results["pass_b2"] and results["pass_cold_start"]
    assert results["pass"] is True
