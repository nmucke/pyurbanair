"""Neural surrogates: training, evaluation and data generation."""

from __future__ import annotations

import json
import pathlib
from typing import Any

import pytest
import torch

from tests.conftest import TRAIN_ORDER, compose, load_script


@pytest.mark.parametrize("name", TRAIN_ORDER)  # type: ignore[misc]
def test_train_writes_a_rebuildable_model(name: str, trained: dict[str, Any]) -> None:
    eval_common = load_script("scripts/surrogate/eval_common.py")
    cfg = trained[name]
    model_dir = pathlib.Path(cfg.paths.weights_dir) / cfg.name
    assert (model_dir / "config.yaml").exists()
    model, _ = eval_common.load_model(model_dir, torch.device("cpu"))
    assert sum(p.numel() for p in model.parameters()) > 0


def test_autoencoder_exports_encoder_and_decoder(trained: dict[str, Any]) -> None:
    cfg = trained["train_autoencoder"]
    model_dir = pathlib.Path(cfg.paths.weights_dir) / cfg.name
    assert (model_dir / "encoder.pt").exists() and (model_dir / "decoder.pt").exists()


def test_evaluate_stepper(session_root: pathlib.Path, trained: dict[str, Any]) -> None:
    cfg = compose("surrogate/eval", "+test=eval", root=session_root)
    load_script("scripts/surrogate/evaluate_stepper.py").run(cfg)
    out = pathlib.Path(cfg.stepper.output_dir)
    for name in (
        "metrics.csv",
        "rmse.png",
        "slices.png",
        "tke_slices.png",
        "params.png",
    ):
        assert (out / name).exists(), name


def test_evaluate_autoencoder(
    session_root: pathlib.Path, trained: dict[str, Any]
) -> None:
    cfg = compose("surrogate/eval", "+test=eval", root=session_root)
    load_script("scripts/surrogate/evaluate_autoencoder.py").run(cfg)
    out = pathlib.Path(cfg.autoencoder.output_dir)
    metrics = json.loads((out / "metrics.json").read_text())
    assert set(metrics["per_variable"]) == {"u", "v", "w"}
    assert (out / "latent_stats.png").exists()


def test_evaluate_latent_generator(
    session_root: pathlib.Path, trained: dict[str, Any]
) -> None:
    cfg = compose(
        "surrogate/eval",
        "+test=eval",
        f"latent_generator.rollout_stepper_dir={session_root / 'weights' / 'stepper'}",
        root=session_root,
    )
    load_script("scripts/surrogate/evaluate_latent_generator.py").run(cfg)
    out = pathlib.Path(cfg.latent_generator.output_dir)
    summary = json.loads((out / "summary.json").read_text())
    assert "passed" in summary["acceptance"]
    assert [row["num_steps"] for row in summary["step_sweep"]] == [2, 5]
    assert set(summary["rollout_kinetic_energy"]) == {"real", "ae_recon", "generated"}


@pytest.mark.integration  # type: ignore[misc]
def test_generate_data(tmp_path: pathlib.Path) -> None:
    cfg = compose("surrogate/generate_data", "+test=generate_data", root=tmp_path)
    load_script("scripts/surrogate/generate_data.py").run(cfg)
    out = pathlib.Path(cfg.paths.results_dir)
    counts = {
        s: len(list((out / "state" / s).glob("*.nc"))) for s in ("train", "val", "test")
    }
    assert counts == {"train": 2, "val": 1, "test": 1}
