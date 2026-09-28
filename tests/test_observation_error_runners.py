"""Small runner-facing checks for the opt-in observation likelihood."""

import pathlib
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import xarray as xr
from data_assimilation.observation_error import ResolvedObservationError
from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf

from pyurbanair.config.hydra_helpers import create_observation_error
from scripts.esmda.run_esmda import _save_obs_diagnostics
from scripts.filtering.run_filtering import _save_window_obs_diagnostics


class _ToyOperator:
    num_sensors = 1
    obs_states = ("u",)
    obs_z = (2.0,)


def _resolved() -> ResolvedObservationError:
    spec = create_observation_error(
        OmegaConf.create(
            {
                "observation_error": {
                    "instrument_std": 0.2,
                    "representation_std": 0.1,
                }
            }
        ),
        OmegaConf.create({"temporal_mode": "full"}),
        ("esmda.obs_error_std",),
    )
    assert spec is not None
    raw = xr.DataArray(
        np.array([[1.0], [1.0]]),
        dims=("time", "obs"),
        coords={"time": [0.0, 1.0], "obs": [0]},
    )
    from data_assimilation.observation_operator import AggregateObservations

    return spec.resolve(raw, _ToyOperator(), AggregateObservations(2.0))


def test_helper_rejects_unknown_error_key() -> None:
    cfg = OmegaConf.create(
        {"observation_error": {"instrument_std": 0.2, "correlation": 0.5}}
    )
    with pytest.raises(ValueError, match="Unknown observation_error keys"):
        create_observation_error(cfg, OmegaConf.create({"temporal_mode": "full"}), ())


def test_null_preserves_legacy_and_explicit_override_conflicts(
    monkeypatch: Any,
) -> None:
    obs = OmegaConf.create({"temporal_mode": "full"})
    assert (
        create_observation_error(
            OmegaConf.create({"observation_error": None}),
            obs,
            ("esmda.obs_error_std",),
        )
        is None
    )
    monkeypatch.setattr(HydraConfig, "initialized", lambda: True)
    monkeypatch.setattr(
        HydraConfig,
        "get",
        lambda: SimpleNamespace(
            overrides=SimpleNamespace(task=["esmda.obs_error_std=0.4"])
        ),
    )
    with pytest.raises(ValueError, match="legacy observation-error override"):
        create_observation_error(
            OmegaConf.create({"observation_error": {"instrument_std": 0.2}}),
            obs,
            ("esmda.obs_error_std",),
        )


def test_esmda_artifact_uses_propagated_physical_variance(
    tmp_path: pathlib.Path,
) -> None:
    resolved = _resolved()
    np.testing.assert_allclose(resolved.covariance_diag, [0.025])
    np.testing.assert_allclose(resolved.raw_instrument_std, 0.2)
    pred = np.array([[0.8, 1.0]])
    params = xr.Dataset({"a": (("esmda_step", "ensemble"), [[0.0, 0.0], [1.0, 1.0]])})
    _save_obs_diagnostics(
        tmp_path,
        0,
        np.array([1.1]),
        np.array([1.0]),
        resolved.std,
        [pred, pred],
        params,
        _ToyOperator(),
        resolved,
        2.0,
        np.array([[1.0, 1.1]]),
    )
    with xr.open_dataset(tmp_path / "window_0_obs.nc") as ds:
        np.testing.assert_allclose(ds.obs_error_std, np.sqrt(0.025))
        np.testing.assert_allclose(ds.obs_instrument_variance, [0.02])
        np.testing.assert_allclose(ds.obs_representation_variance, [0.005])
        np.testing.assert_allclose(ds.obs_innovation_prior, [0.2])
        assert ds.obs_bin_count.item() == 2
        assert ds.attrs["analysis_covariance_multiplier"] == 2.0
        np.testing.assert_allclose(ds.attrs["physical_nis_prior"], 0.04 / 0.045)
        assert "obs_innovation_analyzed" in ds
    with xr.open_dataset(tmp_path / "window_0_pred_obs.nc") as ds:
        assert ds.pred_obs_analyzed.dims == ("obs_index", "ensemble")


def test_filter_artifact_distinguishes_analyzed_state(tmp_path: pathlib.Path) -> None:
    # One aggregated bin in the ESMDA test becomes one ordinary filter frame.
    spec = create_observation_error(
        OmegaConf.create({"observation_error": {"instrument_std": 0.2}}),
        OmegaConf.create({"temporal_mode": "full"}),
        ("filtering.obs_error_std",),
    )
    assert spec is not None
    raw = xr.DataArray(
        np.array([[1.0]]),
        dims=("time", "obs"),
        coords={"time": [2.0], "obs": [0]},
    )
    frame = spec.resolve(raw, _ToyOperator())
    _save_window_obs_diagnostics(
        tmp_path,
        0,
        np.array([1.1]),
        np.array([1.0]),
        frame.std,
        np.array([[0.8, 1.0]]),
        np.array([[0.9, 1.05]]),
        _ToyOperator(),
        [frame],
        np.array([[1.0, 1.1]]),
    )
    with xr.open_dataset(tmp_path / "window_0_obs.nc") as ds:
        assert "obs_innovation_analyzed" in ds
        assert ds.attrs["physical_nis_prior_per_observation"] > 0.0
    with xr.open_dataset(tmp_path / "window_0_pred_obs.nc") as ds:
        assert ds.pred_obs_analyzed.dims == ("obs_index", "ensemble")


@pytest.mark.parametrize(  # type: ignore[misc,unused-ignore]
    "config_name", ["run_esmda", "run_filtering", "run_filter_smoothing"]
)
@pytest.mark.parametrize("config_root", ["conf", "tests/conf"])  # type: ignore[misc,unused-ignore]
def test_corrected_likelihood_composes_with_observation_components(
    config_name: str, config_root: str
) -> None:
    from hydra import compose, initialize_config_dir

    from pyurbanair.config.hydra_helpers import (
        create_aggregate_observations,
        create_observation_operator,
    )

    root = pathlib.Path(__file__).resolve().parents[1] / config_root
    with initialize_config_dir(version_base=None, config_dir=str(root)):
        cfg = compose(
            config_name=config_name,
            overrides=[
                "observation_error={instrument_std:0.25,representation_std:0.1}"
            ],
        )
    spec = create_observation_error(cfg, cfg.obs, ())
    assert spec is not None
    operator = create_observation_operator(
        cfg.obs, cfg.truth_model.solver_name, cfg.observation.operator
    )
    aggregate = create_aggregate_observations(cfg)
    raw = xr.DataArray(
        np.zeros((2, operator.observation_operator.num_obs)),
        dims=("time", "obs"),
        coords={"time": [0.0, 1.0]},
    )
    resolved = spec.resolve(raw, operator, aggregate)
    expected = 0.25**2 + 0.1**2
    if aggregate is not None:
        expected /= 2
    np.testing.assert_allclose(resolved.variance, expected)
    cfg.observation.operator = {
        "_target_": "data_assimilation.observation_operator.ObservationOperator"
    }
    with pytest.raises(ValueError, match="temporal observation/operator"):
        create_observation_error(cfg, cfg.obs, ())
