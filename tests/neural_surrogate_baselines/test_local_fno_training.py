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


def test_epoch_caps() -> None:
    from neural_surrogate_baselines.local_fno.training import _Batches

    loader = torch.utils.data.DataLoader(list(range(10)), batch_size=1)
    train = _Batches(loader, 7, cycle=True)
    first, second = [int(b) for b in train], [int(b) for b in train]
    assert len(train) == 7 and first == list(range(7))
    assert second == [7, 8, 9, 0, 1, 2, 3]  # continues, then restarts
    val = _Batches(loader, 4, cycle=False)
    # Spread over the loader (its last batches too), the same every pass.
    assert [int(b) for b in val] == [0, 3, 6, 9] == [int(b) for b in val]
    assert len(val) == 4


def _compare(session_root: pathlib.Path, out: pathlib.Path, *overrides: str) -> Any:
    cfg = compose(
        "surrogate/baselines/compare",
        "device=cpu",
        f"output_dir={out}",
        *overrides,
        root=session_root,
    )
    load_script("scripts/surrogate/baselines/compare.py").run(cfg)
    return cfg


def test_compare_labels_clashing_names_by_path(
    tmp_path: pathlib.Path, session_root: pathlib.Path, local_fno: Any
) -> None:
    model_dir = pathlib.Path(local_fno.paths.weights_dir) / local_fno.name
    seeds = [tmp_path / f"seed{i}" / "local_fno" for i in range(2)]
    for seed in seeds:
        shutil.copytree(model_dir, seed)
    _compare(session_root, tmp_path / "out", f"models=[{seeds[0]},{seeds[1]}]")
    with (tmp_path / "out" / "metrics.csv").open() as f:
        models = {r["model"] for r in csv.DictReader(f)}
    assert models == {str(seeds[0]), str(seeds[1]), "persistence"}


def test_compare_guards(
    tmp_path: pathlib.Path, session_root: pathlib.Path, local_fno: Any
) -> None:
    model_dir = pathlib.Path(local_fno.paths.weights_dir) / local_fno.name
    reserved = tmp_path / "persistence"
    shutil.copytree(model_dir, reserved)
    with pytest.raises(ValueError, match="reserved"):
        _compare(session_root, tmp_path / "a", f"models=[{reserved}]")
    with pytest.raises(ValueError, match="max_steps"):
        _compare(session_root, tmp_path / "b", f"models=[{model_dir}]", "max_steps=0")
    # Levels outside the grid skip the spectra instead of failing.
    _compare(
        session_root,
        tmp_path / "c",
        f"models=[{model_dir}]",
        "levels=[99]",
        "max_steps=2",
    )
    assert (tmp_path / "c" / "metrics.csv").exists()
    assert not (tmp_path / "c" / "spectra.png").exists()


def test_compare_refuses_another_cadence(
    tmp_path: pathlib.Path,
    session_root: pathlib.Path,
    training_data: pathlib.Path,
    local_fno: Any,
) -> None:
    """Two corpora with the same folder name but another output frequency."""
    model_dir = pathlib.Path(local_fno.paths.weights_dir) / local_fno.name
    corpus = tmp_path / "elsewhere" / training_data.name
    shutil.copytree(training_data, corpus)
    data_cfg = OmegaConf.load(corpus / "config.yaml")
    data_cfg.time.output_frequency = 2 * float(data_cfg.time.output_frequency)
    OmegaConf.save(data_cfg, corpus / "config.yaml")
    other = tmp_path / "other"
    shutil.copytree(model_dir, other)
    cfg_other = OmegaConf.load(other / "config.yaml")
    cfg_other.paths.data_dir = str(corpus)
    OmegaConf.save(cfg_other, other / "config.yaml")
    with pytest.raises(ValueError, match="different data"):
        _compare(session_root, tmp_path / "out", f"models=[{model_dir},{other}]")
