"""Runner regressions for prescribed parameters outside the analysis selection."""

import importlib
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import xarray as xr
from hydra.utils import instantiate
from omegaconf import OmegaConf


class _PrescribedForwardModel:
    def __init__(self, simulation_time: float, output_frequency: float) -> None:
        self.simulation_time = simulation_time
        self.output_frequency = output_frequency
        self.params_seen: list[xr.Dataset] = []

    def __call__(self, params: xr.Dataset, state: Any = None) -> xr.Dataset:
        self.params_seen.append(params.copy(deep=True))
        velocity = params.velocity_magnitude
        if "time" in velocity.dims:
            velocity = velocity.mean("time")
        # The unestimated coefficient contributes directly to the prediction,
        # so accidentally dropping it changes the observations used by analysis.
        value = float(velocity) + 0.01 * float(params.sgs_bias_b0)
        times = np.arange(0, self.simulation_time, self.output_frequency)
        shape = (len(times), 2, 2, 2)
        return xr.Dataset(
            {
                name: (("time", "z", "y", "x"), np.full(shape, value))
                for name in ("u", "v")
            },
            coords={
                "time": times,
                "z": [0.0, 10.0],
                "y": [0.0, 20.0],
                "x": [0.0, 20.0],
            },
        )


class _PrescribedEnsembleModel:
    save_on_disk = False
    results_dir = None

    def __init__(self, forward_model: _PrescribedForwardModel) -> None:
        self.forward_model = forward_model
        self.params_seen: list[xr.Dataset] = []

    def run_ensemble(self, params: xr.Dataset, state: Any = None) -> xr.Dataset:
        self.params_seen.append(params.copy(deep=True))
        return xr.concat(
            [
                self.forward_model(params.isel(ensemble=member))
                for member in range(params.sizes["ensemble"])
            ],
            dim="ensemble",
        )

    def apply_failure_substitutions_to_params(self, params: Any) -> Any:
        return params

    def apply_failure_substitutions_to_state(self, state: Any) -> Any:
        return state


@pytest.mark.parametrize(  # type: ignore[misc]
    "workflow,smoother,mode,selection",
    [
        ("esmda", "static", "joint", ["velocity_magnitude"]),
        ("esmda", "state", "joint", ["velocity_magnitude"]),
        ("esmda", "dynamic", "joint", ["velocity_magnitude"]),
        ("esmda", "static", "joint", []),
        ("filtering", "static", "joint", ["velocity_magnitude"]),
        ("filtering", "static", "state", ["velocity_magnitude"]),
        ("filter_smoothing", "static", "joint", ["velocity_magnitude"]),
        ("filter_smoothing", "dynamic", "joint", ["velocity_magnitude"]),
    ],
)
def test_runners_apply_and_save_unestimated_parameters(
    workflow: str,
    smoother: str,
    mode: str,
    selection: list[str],
    compose_test_cfg: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dynamic = smoother == "dynamic"
    overrides = [
        "model@truth_model=pylbm",
        "model@assim_model=pylbm",
        f"params@prior_params={'dynamic' if dynamic else 'static'}",
        f"params@truth_params={'dynamic_truth' if dynamic else 'static_truth'}",
        "ensemble.ensemble_size=8",
        "run.ensemble_save_on_disk=false",
    ]
    if workflow != "filtering":
        overrides += [f"esmda/smoother={smoother}", "esmda.interval_seconds=null"]
    cfg = compose_test_cfg(overrides, config_name=f"run_{workflow}")
    cfg.params_to_estimate = selection
    if workflow == "esmda":
        cfg.esmda.num_assimilation_windows = 2
    else:
        cfg.filtering.mode = mode
        cfg.filtering.parameter_evolution = None
        if mode == "state":
            cfg.filtering.inflation = None
        if workflow == "filtering":
            cfg.filtering.num_assimilation_windows = 2
        else:
            cfg.filter_smoothing.num_assimilation_windows = 2
    block = "static_parameters" if dynamic else "parameters"
    for role, value in [("truth_params", 0.0), ("prior_params", -20.0)]:
        OmegaConf.update(
            cfg,
            f"{role}.{block}.sgs_bias_b0",
            {"_target_": "pyurbanair.static_parameters.Constant", "value": value},
            force_add=True,
        )

    module = importlib.import_module(f"scripts.{workflow}.run_{workflow}")
    models: list[_PrescribedForwardModel] = []
    ensembles: list[_PrescribedEnsembleModel] = []

    def instantiate_with_toy_backend(config: Any, **kwargs: Any) -> Any:
        target = str(config.get("_target_", ""))
        if target == "pylbm.forward_model.ForwardModel":
            model = _PrescribedForwardModel(
                float(kwargs.get("simulation_time", config.simulation_time)),
                float(config.output_frequency),
            )
            models.append(model)
            return model
        if target == "pylbm.ensemble_forward_model.EnsembleForwardModel":
            ensemble = _PrescribedEnsembleModel(kwargs["forward_model"])
            ensembles.append(ensemble)
            return ensemble
        if target == "pyurbanair.config.hydra_helpers.prepare_compile":
            return None
        return instantiate(config, **kwargs)

    monkeypatch.setattr(module, "instantiate", instantiate_with_toy_backend)
    monkeypatch.setattr(module, "clean_outputs", lambda **kwargs: None)
    module.run(cfg)

    assert len(models[0].params_seen) == 1
    assert float(models[0].params_seen[0].sgs_bias_b0) == 0.0
    out = Path(cfg.paths.results_dir)
    truth = xr.load_dataset(out / "true_params.nc")
    assert "sgs_bias_b0" in truth
    np.testing.assert_array_equal(truth.sgs_bias_b0, 0.0)
    for ensemble in ensembles:
        assert ensemble.params_seen
        for params in ensemble.params_seen:
            np.testing.assert_array_equal(params.sgs_bias_b0, -20.0)
            assert "inflow_angle" in params
    for window in range(2):
        prior = xr.load_dataset(out / f"windows/window_{window}_prior_params.nc")
        posterior = xr.load_dataset(
            out / f"windows/window_{window}_posterior_params.nc"
        )
        assert set(prior.data_vars) == set(posterior.data_vars)
        np.testing.assert_array_equal(posterior.sgs_bias_b0, -20.0)
        xr.testing.assert_equal(
            prior.inflow_angle.reset_coords(drop=True),
            posterior.inflow_angle.reset_coords(drop=True),
        )
        if smoother == "state" or mode == "state" or not selection:
            xr.testing.assert_equal(
                prior.reset_coords(drop=True), posterior.reset_coords(drop=True)
            )
        else:
            assert not np.array_equal(
                prior.velocity_magnitude, posterior.velocity_magnitude
            )
