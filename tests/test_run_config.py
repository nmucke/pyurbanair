"""Configuration inspection and run-record contract without solver execution."""

from __future__ import annotations

import os
import pathlib
import subprocess
import sys
from typing import Any

import pytest
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from pyurbanair.config.run_record import validate_run_config, write_run_record

ROOT = pathlib.Path(__file__).resolve().parents[1]
CONF = ROOT / "conf"


def _compose(name: str, *overrides: str) -> Any:
    with initialize_config_dir(version_base=None, config_dir=str(CONF)):
        return compose(config_name=name, overrides=list(overrides))


def test_preview_imports_no_backend_and_creates_no_run_dir(
    tmp_path: pathlib.Path,
) -> None:
    code = """
import sys
from scripts.preview_config import main
sys.argv = ['preview_config.py', 'run_esmda', 'paths.results_dir=RUN_DIR']
main()
for name in ('pylbm', 'pyudales', 'pypalm', 'jax', 'data_assimilation'):
    assert name not in sys.modules, name
""".replace(
        "RUN_DIR", str(tmp_path / "run")
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=ROOT,
        env={**os.environ, "PYTHONPYCACHEPREFIX": str(tmp_path / "pycache")},
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "Selected configuration files:" in result.stdout
    assert "Deferred runtime arguments:" in result.stdout
    assert not (tmp_path / "run").exists()


def test_workflow_and_mode_validation_precede_artifacts(tmp_path: pathlib.Path) -> None:
    cfg = _compose("run_esmda", "experiment=esmda/barcelona_dynamic")
    with pytest.raises(ValueError, match="experiment.workflow"):
        validate_run_config(cfg, "filtering")
    bad = _compose(
        "run_esmda",
        "esmda/smoother=static",
        "params@prior_params=dynamic",
        f"paths.results_dir={tmp_path / 'invalid'}",
    )
    with pytest.raises(ValueError, match="esmda.smoother and prior_params"):
        validate_run_config(bad, "esmda")
    assert not (tmp_path / "invalid").exists()
    mixed = _compose(
        "run_esmda",
        "esmda/smoother=static",
        "params@prior_params=static",
        "params@truth_params=dynamic_truth",
    )
    with pytest.raises(ValueError, match="truth_params and prior_params"):
        validate_run_config(mixed, "esmda")
    fractional_cycle = _compose(
        "run_filter_smoothing",
        "time.simulation_time=3.1",
        "time.output_frequency=1",
    )
    with pytest.raises(ValueError, match="tiled exactly"):
        validate_run_config(fractional_cycle, "filter_smoothing")


def test_fixed_training_data_case_controls_output_name() -> None:
    cfg = _compose(
        "neural_surrogate/training_data",
        "training_data/geometry_mode=fixed",
        "case=barcelona",
    )
    validate_run_config(cfg, "surrogate_data")
    assert cfg.training_data.geometry.output_name == "barcelona"
    assert str(cfg.training_data.output_dir).endswith("pyudales_barcelona/")
    cfg.training_data.geometry.output_name = "realistic"
    with pytest.raises(ValueError, match="case_name"):
        validate_run_config(cfg, "surrogate_data")
    legacy = _compose(
        "neural_surrogate/training_data",
        "training_data/geometry_mode=fixed",
        "case=barcelona",
        "training_data.geometry.source=xie_and_castro",
    )
    with pytest.raises(ValueError, match="legacy fixed-case selector"):
        validate_run_config(legacy, "surrogate_data")


def test_probe_preview_requires_completed_run(tmp_path: pathlib.Path) -> None:
    cfg = _compose("run_probe_series", f"probes.run_dir={tmp_path}")
    with pytest.raises(ValueError, match="truth_access.yaml"):
        validate_run_config(cfg, "probe")


def test_probe_preview_skips_inherited_inference_validation(
    tmp_path: pathlib.Path,
) -> None:
    (tmp_path / "truth_access.yaml").write_text("")
    cfg = _compose(
        "run_probe_series",
        f"probes.run_dir={tmp_path}",
        "assim_model.forward_model.model_discrepancy.enabled=true",
        "esmda/smoother=state",
    )
    validate_run_config(cfg, "probe")


def test_grid_operator_and_explicit_none_aggregation() -> None:
    from pyurbanair.config.hydra_helpers import (
        create_aggregate_observations,
        create_observation_operator,
    )

    obs = OmegaConf.create(
        {
            "mode": "grid",
            "x_min": 0.0,
            "x_max": 1.0,
            "y_min": 2.0,
            "y_max": 3.0,
            "z": 0.5,
            "n_per_axis": 2,
            "states": ["u"],
        }
    )
    operator_cfg = OmegaConf.load(CONF / "observation/operator/temporal_grid.yaml")
    operator = create_observation_operator(obs, "pylbm", operator_cfg)
    assert len(operator.observation_operator.obs_x) == 4
    default_operator = OmegaConf.load(
        CONF / "observation/operator/temporal_points.yaml"
    )
    fallback = create_observation_operator(obs, "pylbm", default_operator)
    assert len(fallback.observation_operator.obs_x) == 4

    cfg = _compose("run_esmda", "observation/aggregation=none")
    assert cfg.esmda.interval_seconds is not None
    assert create_aggregate_observations(cfg) is None
    legacy_disabled = _compose("run_esmda", "esmda.interval_seconds=null")
    assert create_aggregate_observations(legacy_disabled) is None


def test_record_keeps_raw_and_resolved_config_and_runtime_args(
    tmp_path: pathlib.Path,
) -> None:
    cfg = _compose(
        "run_esmda", "experiment=esmda/barcelona_dynamic", "esmda.num_steps=2"
    )
    write_run_record(
        cfg,
        tmp_path / "run",
        "esmda",
        constructor_overrides=[
            {
                "role": "assim",
                "component": "esmda.smoother",
                "window": 0,
                "values": {"num_time_points": 3},
            }
        ],
    )
    raw = OmegaConf.load(tmp_path / "run" / "config.yaml")
    resolved = OmegaConf.load(tmp_path / "run" / "config.resolved.yaml")
    manifest = OmegaConf.load(tmp_path / "run" / "run_manifest.yaml")
    raw_values = OmegaConf.to_container(raw, resolve=False)
    assert isinstance(raw_values, dict)
    esmda_values = raw_values["esmda"]
    assert isinstance(esmda_values, dict)
    assert esmda_values["alpha"] == "${.num_steps}"
    assert resolved.esmda.alpha == 2
    assert manifest.experiment == "barcelona_dynamic"
    assert manifest.constructor_overrides[0]["values"].num_time_points == 3
    assert manifest.params_to_estimate == ["inflow_angle", "velocity_magnitude"]
    assert pathlib.Path(manifest.paths.output_dir) == (tmp_path / "run").resolve()
