"""SSRollingUrbanNet through the unchanged training, evaluation and forward-run
scripts: Roll-1, TL Roll-3 from it, then the rebuilt model rolled out."""

from __future__ import annotations

import csv
import pathlib
from typing import Any

import numpy as np
import pytest
import torch
import xarray
from omegaconf import OmegaConf

from tests.conftest import compose, load_script, surrogate


@pytest.fixture(scope="module")  # type: ignore[misc]
def roll1(session_root: pathlib.Path, training_data: pathlib.Path) -> Any:
    torch.set_num_threads(1)
    cfg = compose(
        "surrogate/baselines/ssrolling/train_roll1",
        "+test=train_roll1",
        root=session_root,
    )
    load_script("scripts/surrogate/train.py").run(cfg)
    return cfg


@pytest.fixture(scope="module")  # type: ignore[misc]
def tl_roll3(session_root: pathlib.Path, roll1: Any) -> Any:
    cfg = compose(
        "surrogate/baselines/ssrolling/finetune_roll3",
        "+test=finetune_roll3",
        root=session_root,
    )
    load_script("scripts/surrogate/train.py").run(cfg)
    return cfg


def _model_dir(cfg: Any) -> pathlib.Path:
    return pathlib.Path(cfg.paths.weights_dir, cfg.name)


def test_roll1_logs_both_loss_terms(roll1: Any) -> None:
    with (_model_dir(roll1) / "metrics.csv").open() as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 1
    for key in (
        "train_loss",
        "val_loss",
        "train_mse",
        "train_spec",
        "val_mse",
        "val_spec",
    ):
        assert np.isfinite(float(rows[0][key]))


def test_tl_roll3_continues_roll1(roll1: Any, tl_roll3: Any) -> None:
    saved = OmegaConf.load(_model_dir(tl_roll3) / "config.yaml")
    pretrained = OmegaConf.load(_model_dir(roll1) / "config.yaml")
    assert saved.architecture == pretrained.architecture
    assert saved.dataset.num_history_steps == 2
    assert saved.dataset.pushforward_steps == 3
    assert (_model_dir(tl_roll3) / "weights.pt").exists()


def test_the_rebuilt_model_rolls_out(
    training_data: pathlib.Path, tl_roll3: Any
) -> None:
    eval_common = load_script("scripts/utils/eval_common.py")
    model, cfg = eval_common.load_model(_model_dir(tl_roll3), torch.device("cpu"))
    assert type(model).__name__ == "SSRollingUrbanNet"
    assert model.ssgen is not None
    # The training split's statistics travel with the weights.
    assert not torch.equal(model.state_std, torch.ones(3))
    test = eval_common.open_dataset(cfg, None, "test")
    frames = torch.from_numpy(eval_common.load_states(test, 0, times=[0, 1]))
    state = frames.flatten(0, 1).unsqueeze(0)
    geometry = test.geometry_for(0).unsqueeze(0)
    params = test._params[0][:1]
    with torch.no_grad():
        for _ in range(3):
            pred = model(state, params, geometry)
            state = torch.cat([state[:, 3:], pred], dim=1)
    assert pred.shape == (1, 3, *geometry.shape[1:])
    assert torch.isfinite(pred).all()
    assert (pred.permute(1, 0, 2, 3, 4)[:, geometry == 0] == 0).all()


def test_forward_run_from_a_training_state(
    tmp_path: pathlib.Path,
    session_root: pathlib.Path,
    training_data: pathlib.Path,
    tl_roll3: Any,
) -> None:
    """The forward model (the data-assimilation path) rolls it out."""
    cfg = compose(
        "forward",
        "+test=forward",
        "model=neural_surrogate_tiny",
        *surrogate(session_root),
        f"model.forward_model.model_dir={_model_dir(tl_roll3)}",
        "model.forward_model.spinup_source=training_data",
        f"forward.initial_state={training_data / 'state' / 'test' / 'sample_0000.nc'}",
        root=tmp_path,
    )
    load_script("scripts/run_forward.py").run(cfg)
    state = xarray.load_dataset(pathlib.Path(cfg.paths.results_dir) / "state.nc")
    assert {"u", "v", "w"} <= set(state.data_vars)
    assert np.isfinite(state.u.values).all()
