"""Coefficient-only inference wiring, including native multi-window smoke tests."""

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import xarray as xr
from hydra.utils import instantiate
from omegaconf import OmegaConf

from pyurbanair.config.discrepancy import SGS_BIAS_PARAMETER_NAMES
from pyurbanair.config.hydra_helpers import inference_parameter_configs
from pyurbanair.config.run_record import validate_run_config


def _configure(cfg: Any, workflow: str) -> None:
    settings = {
        "enabled": True,
        "kind": "sgs_strain_rotation",
        "coefficient_model": "persistent",
        "canopy_height": 10.0,
        "height_band_over_H": [0.5, 2.0],
        "gradient_regularization": 0.01,
        "log_multiplier_cap": 0.4054651081081644,
        "prior_std": [0.15, 0.15, 0.15],
    }
    for role in ("truth_model", "assim_model"):
        OmegaConf.update(
            cfg, f"{role}.forward_model.model_discrepancy", settings, force_add=True
        )
        cfg[role].forward_model.temp_dir = str(Path(cfg.paths.experiment_dir) / role)
    cfg.params_to_estimate = list(SGS_BIAS_PARAMETER_NAMES)
    cfg.ensemble.failure.policy = "raise"
    cfg.run.truth_dir = None
    cfg.run.skip_viz = True
    if workflow == "filter_smoothing":
        cfg.filtering.mode = "state"
        cfg.filter_smoothing.num_assimilation_windows = 2
        cfg.filter_smoothing.beta = 2.0
        cfg.filter_smoothing.likelihood_allocation = "shared_budget"
    elif workflow == "filtering":
        cfg.filtering.mode = "joint"
        cfg.filtering.num_assimilation_windows = 2
    else:
        cfg.esmda.num_assimilation_windows = 2
    if workflow != "filtering":
        cfg.esmda.interval_seconds = None


@pytest.mark.parametrize("workflow", ["esmda", "filtering", "filter_smoothing"])  # type: ignore[misc]
def test_truth_parameters_remain_independent_of_inferred_coefficients(
    workflow: str, compose_test_cfg: Any
) -> None:
    overrides = ["params@prior_params=static", "params@truth_params=static_truth"]
    if workflow != "filtering":
        overrides.append("esmda/smoother=static")
    cfg = compose_test_cfg(overrides, config_name=f"run_{workflow}")
    _configure(cfg, workflow)
    before = OmegaConf.to_container(cfg.truth_params, resolve=True)
    validate_run_config(cfg, workflow)
    truth_cfg, prior_cfg = inference_parameter_configs(cfg)
    assert OmegaConf.to_container(truth_cfg, resolve=True) == before
    truth = instantiate(truth_cfg).sample(1)
    prior = instantiate(prior_cfg).sample(8)
    assert set(prior.data_vars) == set(SGS_BIAS_PARAMETER_NAMES)
    assert not set(SGS_BIAS_PARAMETER_NAMES).intersection(truth.data_vars)
    assert "velocity_magnitude" in truth
    assert prior.sizes == {"ensemble": 8}


def test_disk_ensemble_retains_each_members_discrepancy(tmp_path: Path) -> None:
    from scripts.esmda.run_esmda import _stream_concat_members

    files = []
    metadata = [{"coefficients": {"sgs_bias_b0": value}} for value in [0.1, -0.2]]
    for member, record in enumerate(metadata):
        path = tmp_path / f"member_{member}.nc"
        xr.Dataset(
            {"u": ("time", [float(member)])},
            coords={"time": [0.0]},
            attrs={"model_discrepancy": json.dumps(record)},
        ).to_netcdf(path)
        files.append(path)
    output = tmp_path / "ensemble.nc"
    _stream_concat_members(files, output)
    stored = xr.load_dataset(output)
    assert "model_discrepancy" not in stored.attrs
    assert json.loads(stored.attrs["model_discrepancy_by_member"]) == metadata


@pytest.mark.integration  # type: ignore[misc]
@pytest.mark.parametrize("workflow", ["esmda", "filtering", "filter_smoothing"])  # type: ignore[misc]
@pytest.mark.parametrize("disk", [False, True], ids=["memory", "disk"])  # type: ignore[misc]
def test_native_discrepancy_workflows(
    workflow: str, disk: bool, compose_test_cfg: Any
) -> None:
    """Run all inference paths across a warm boundary, using frozen test configs.

    This checks integration and parameter timing, not statistical recovery on
    this deliberately tiny grid. The named recovery experiment is separate.
    """
    import importlib

    overrides = [
        "model@truth_model=pyudales",
        "model@assim_model=pyudales",
        "params@prior_params=static",
        "params@truth_params=static_truth",
        "ensemble.ensemble_size=4",
        "ensemble.num_parallel_processes=2",
        f"run.ensemble_save_on_disk={str(disk).lower()}",
    ]
    if workflow != "filtering":
        overrides += [
            "esmda/smoother=static",
            "esmda.num_steps=2",
            "observation/aggregation=none",
        ]
    cfg = compose_test_cfg(overrides, config_name=f"run_{workflow}")
    _configure(cfg, workflow)
    for name, value in zip(SGS_BIAS_PARAMETER_NAMES, [0.1, -0.12, 0.08]):
        OmegaConf.update(
            cfg,
            f"truth_params.parameters.{name}",
            {
                "_target_": "pyurbanair.static_parameters.Constant",
                "value": value,
            },
            force_add=True,
        )
    module = importlib.import_module(f"scripts.{workflow}.run_{workflow}")
    module.run(cfg)
    out = Path(cfg.paths.results_dir)
    prior = xr.load_dataset(out / "prior_params.nc")
    posterior = xr.load_dataset(out / "posterior_params.nc")
    assert set(posterior.data_vars) == set(SGS_BIAS_PARAMETER_NAMES)
    assert all(np.isfinite(posterior[name]).all() for name in SGS_BIAS_PARAMETER_NAMES)
    assert any(
        not np.array_equal(prior[name], posterior[name])
        for name in SGS_BIAS_PARAMETER_NAMES
    )
    if workflow == "esmda":
        previous = xr.load_dataset(out / "windows/window_0_posterior_params.nc")
        following = xr.load_dataset(out / "windows/window_1_prior_params.nc")
        xr.testing.assert_equal(previous, following)
    else:
        applied = xr.load_dataset(out / "applied_params_history.nc")
        if workflow == "filtering":
            analyzed = xr.load_dataset(out / "params_history.nc")
            for name in SGS_BIAS_PARAMETER_NAMES:
                np.testing.assert_array_equal(
                    applied[name], analyzed[name].isel(cycle=slice(None, -1))
                )
        else:
            history = xr.load_dataset(out / "esmda_params_history.nc")
            cycles = applied.sizes["cycle"] // 2
            for window in range(2):
                for name in SGS_BIAS_PARAMETER_NAMES:
                    expected = history[name].isel(window=window, esmda_step=-1)
                    for cycle in range(window * cycles, (window + 1) * cycles):
                        np.testing.assert_array_equal(
                            applied[name].isel(cycle=cycle), expected
                        )
