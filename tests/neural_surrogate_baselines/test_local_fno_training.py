"""Local-FNO through the unchanged training, evaluation and forward-run scripts."""

from __future__ import annotations

import csv
import pathlib
import shutil
from typing import Any

import numpy as np
import pytest
import torch
import xarray
from omegaconf import OmegaConf

from tests.conftest import compose, load_script, surrogate


@pytest.fixture(scope="module")  # type: ignore[misc]
def local_fno(session_root: pathlib.Path, training_data: pathlib.Path) -> Any:
    """Train the tiny Local-FNO once (two capped epochs, to see the LR halve)."""
    torch.set_num_threads(1)
    cfg = compose(
        "surrogate/baselines/local_fno/train",
        "+test=train",
        "trainer.num_epochs=2",
        "trainer.patience=null",
        root=session_root,
    )
    load_script("scripts/surrogate/train.py").run(cfg)
    return cfg


def test_trains_with_the_paper_schedule(local_fno: Any) -> None:
    model_dir = pathlib.Path(local_fno.paths.weights_dir) / local_fno.name
    with (model_dir / "metrics.csv").open() as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 2
    # StepLR halves the learning rate after every epoch.
    assert float(rows[1]["lr"]) == pytest.approx(0.5 * float(rows[0]["lr"]))
    assert all(np.isfinite(float(r["train_loss"])) for r in rows)


def test_rebuilds_from_the_saved_config(local_fno: Any) -> None:
    eval_common = load_script("scripts/utils/eval_common.py")
    model_dir = pathlib.Path(local_fno.paths.weights_dir) / local_fno.name
    model, _ = eval_common.load_model(model_dir, torch.device("cpu"))
    assert type(model).__name__ == "LocalFNOStepper"
    assert model.num_history_steps == 2
    # The training split's statistics travel with the weights.
    assert not torch.equal(model.state_std, torch.ones(3))


def test_evaluate_stepper(session_root: pathlib.Path, local_fno: Any) -> None:
    model_dir = pathlib.Path(local_fno.paths.weights_dir) / local_fno.name
    cfg = compose(
        "surrogate/eval",
        "+test=eval",
        f"stepper.models=[{model_dir}]",
        f"stepper.output_dir={session_root / 'local_fno_eval'}",
        root=session_root,
    )
    load_script("scripts/surrogate/evaluate_stepper.py").run(cfg)
    assert (pathlib.Path(cfg.stepper.output_dir) / "metrics.csv").exists()


def test_forward_run_from_a_training_state(
    tmp_path: pathlib.Path,
    session_root: pathlib.Path,
    training_data: pathlib.Path,
    local_fno: Any,
) -> None:
    """The forward model (the data-assimilation path) rolls Local-FNO out."""
    model_dir = pathlib.Path(local_fno.paths.weights_dir) / local_fno.name
    cfg = compose(
        "forward",
        "+test=forward",
        "model=neural_surrogate_tiny",
        *surrogate(session_root),
        f"model.forward_model.model_dir={model_dir}",
        "model.forward_model.spinup_source=training_data",
        f"forward.initial_state={training_data / 'state' / 'test' / 'sample_0000.nc'}",
        root=tmp_path,
    )
    load_script("scripts/run_forward.py").run(cfg)
    state = xarray.load_dataset(pathlib.Path(cfg.paths.results_dir) / "state.nc")
    assert {"u", "v", "w"} <= set(state.data_vars)
    assert np.isfinite(state.u.values).all()


def test_compare_against_persistence(
    session_root: pathlib.Path, local_fno: Any
) -> None:
    model_dir = pathlib.Path(local_fno.paths.weights_dir) / local_fno.name
    cfg = compose(
        "surrogate/baselines/compare",
        "device=cpu",
        f"models=[{model_dir}]",
        f"output_dir={session_root / 'compare'}",
        "levels=[0, 2]",
        root=session_root,
    )
    load_script("scripts/surrogate/baselines/compare.py").run(cfg)
    out = pathlib.Path(cfg.output_dir)
    for name in (
        "metrics.csv",
        "per_step.csv",
        "statistics.csv",
        "rmse.png",
        "energy.png",
        "spectra.png",
        "profiles.png",
    ):
        assert (out / name).exists(), name
    with (out / "metrics.csv").open() as f:
        rows = {(r["model"], r["sample"]): r for r in csv.DictReader(f)}
    assert float(rows[("persistence", "mean")]["rmse_vs_persistence"]) == pytest.approx(
        1.0
    )
    assert np.isfinite(float(rows[("local_fno", "mean")]["rmse"]))


def test_compare_refuses_models_trained_on_other_data(
    tmp_path: pathlib.Path, session_root: pathlib.Path, local_fno: Any
) -> None:
    model_dir = pathlib.Path(local_fno.paths.weights_dir) / local_fno.name
    other = tmp_path / "other"
    shutil.copytree(model_dir, other)
    cfg_other = OmegaConf.load(other / "config.yaml")
    cfg_other.dataset.param_vars = ["inflow_angle"]
    OmegaConf.save(cfg_other, other / "config.yaml")
    cfg = compose(
        "surrogate/baselines/compare",
        "device=cpu",
        f"models=[{model_dir},{other}]",
        f"output_dir={tmp_path / 'compare'}",
        root=session_root,
    )
    with pytest.raises(ValueError, match="different data"):
        load_script("scripts/surrogate/baselines/compare.py").run(cfg)
