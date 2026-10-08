"""Neural surrogates: training, evaluation and data generation."""

from __future__ import annotations

import json
import pathlib
from typing import Any

import pytest
import torch
from omegaconf import OmegaConf

from tests.conftest import TRAIN_ORDER, compose, load_script


@pytest.mark.parametrize("name", TRAIN_ORDER)  # type: ignore[misc]
def test_train_writes_a_rebuildable_model(name: str, trained: dict[str, Any]) -> None:
    eval_common = load_script("scripts/utils/eval_common.py")
    cfg = trained[name]
    model_dir = pathlib.Path(cfg.paths.weights_dir) / cfg.name
    assert (model_dir / "config.yaml").exists()
    model, _ = eval_common.load_model(model_dir, torch.device("cpu"))
    assert sum(p.numel() for p in model.parameters()) > 0


def test_autoencoder_exports_encoder_and_decoder(trained: dict[str, Any]) -> None:
    eval_common = load_script("scripts/utils/eval_common.py")
    cfg = trained["train_autoencoder"]
    model_dir = pathlib.Path(cfg.paths.weights_dir) / cfg.name
    model, _ = eval_common.load_model(model_dir, torch.device("cpu"))  # weights.pt
    parts = {"encoder.pt": model.ae.encoder, "decoder.pt": model.ae.decoder}
    if model.geometry_branch is not None:
        parts["geometry_branch.pt"] = model.geometry_branch
    for file, module in parts.items():
        exported = torch.load(model_dir / file)
        expected = module.state_dict()
        assert exported.keys() == expected.keys(), file
        assert all(torch.equal(exported[k], expected[k]) for k in expected), file


def test_prechunk_prepare_only(
    tmp_path: pathlib.Path, training_data: pathlib.Path
) -> None:
    prepared = tmp_path / "prechunked"
    cfg = compose(
        "surrogate/train_autoencoder",
        "+test=train_autoencoder",
        f"paths.data_dir={training_data}",
        f"prechunk.output_root={prepared}",
        "prechunk.prepare_only=true",
        root=tmp_path,
    )
    load_script("scripts/surrogate/train.py").run(cfg)
    sources = sorted(
        p.relative_to(training_data) for p in training_data.glob("state/*/sample_*.nc")
    )
    assert sources
    assert sorted(p.relative_to(prepared) for p in prepared.glob("state/*/*.nc")) == (
        sources
    )
    manifest = json.loads((prepared / ".rechunk-manifest.json").read_text())
    assert manifest["complete"]
    assert sorted(manifest["files"]) == [str(p) for p in sources]
    assert not pathlib.Path(cfg.paths.weights_dir).exists()


def test_prechunked_training_keeps_the_source_root(
    tmp_path: pathlib.Path, training_data: pathlib.Path
) -> None:
    prepared = tmp_path / "prechunked"
    cfg = compose(
        "surrogate/train_autoencoder",
        "+test=train_autoencoder",
        f"paths.data_dir={training_data}",
        f"prechunk.output_root={prepared}",
        root=tmp_path,
    )
    load_script("scripts/surrogate/train.py").run(cfg)
    model_dir = pathlib.Path(cfg.paths.weights_dir) / cfg.name
    saved = OmegaConf.load(model_dir / "config.yaml")
    assert saved.dataset.root_dir == str(training_data)
    assert (prepared / "normalization_stats" / "train.npz").exists()
    assert (model_dir / "weights.pt").exists()


def test_prechunked_dft_reads_whole_frames_and_source_params(
    tmp_path: pathlib.Path, training_data: pathlib.Path, trained: dict[str, Any]
) -> None:
    ae = trained["train_autoencoder"]
    prepared = tmp_path / "prechunked"
    cfg = compose(
        "surrogate/train_dft",
        "+test=train_dft",
        f"paths.data_dir={training_data}",
        f"pretrained_dir={pathlib.Path(ae.paths.weights_dir) / ae.name}",
        f"prechunk.output_root={prepared}",
        root=tmp_path,
    )
    load_script("scripts/surrogate/train.py").run(cfg)
    manifest = json.loads((prepared / ".rechunk-manifest.json").read_text())
    assert manifest["complete"] and manifest["options"]["spatial_chunks"] is None
    assert not (prepared / "param").exists()  # params come from the source
    model_dir = pathlib.Path(cfg.paths.weights_dir) / cfg.name
    assert OmegaConf.load(model_dir / "config.yaml").dataset.root_dir == str(
        training_data
    )
    assert (model_dir / "weights.pt").exists()


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
    assert list(out.glob("vertical_profiles_*.png"))


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
