import pathlib

import pytest


@pytest.mark.parametrize(  # type: ignore[misc]
    "truth_model,assim_model",
    [
        pytest.param("pylbm", "pylbm", id="pylbm_pylbm"),
        pytest.param("pyudales", "pyudales", id="pyudales_pyudales"),
        pytest.param("pylbm", "pyudales", id="pylbm_pyudales_cross"),
        pytest.param("pyudales", "pylbm", id="pyudales_pylbm_cross"),
        pytest.param(
            "pyudales", "neural_surrogate", id="pyudales_neural_surrogate"
        ),
    ],
)
def test_run_time_varying_parameters_rollout_esmda(
    truth_model: str,
    assim_model: str,
    tmp_path: pathlib.Path,
    compose_test_cfg,
    surrogate_model_dir_factory,
) -> None:
    """Rollout time-varying parameter ESMDA across multiple windows."""
    from scripts.run_time_varying_parameters_rollout_esmda import run

    overrides = [
        "esmda=time_varying_rollout",
        f"model@truth_model={truth_model}",
        f"model@assim_model={assim_model}",
        "run.skip_viz=true",
        "time_varying.num_time_points=3",
        "ensemble.ensemble_size=2",
        "ensemble.num_parallel_processes=2",
        "esmda.num_steps=1",
        "esmda.num_assimilation_windows=2",
        f"paths.base_results_dir={tmp_path}",
    ]
    if truth_model == "pylbm":
        overrides.append("truth_model.forward_model.cuda=false")
    if assim_model == "pylbm":
        overrides.append("assim_model.forward_model.cuda=false")
    if assim_model == "neural_surrogate":
        # Point the surrogate at a freshly-built trained-model folder so the
        # architecture, trained domain (== size=tiny) and output frequency are
        # all derived from it. Weights are random — ESMDA exercises the
        # interface, not physical accuracy.
        model_dir = surrogate_model_dir_factory(
            tmp_path,
            domain={
                "nx": 20,
                "ny": 20,
                "nz": 4,
                "bounds": [[0.0, 20.0], [0.0, 20.0], [0.0, 10.0]],
            },
            time={
                "simulation_time": 5.0,
                "output_frequency": 1.0,
                "spinup_time": 5.0,
            },
            param_vars=("inflow_angle", "velocity_magnitude"),
        )
        overrides.append(f"assim_model.forward_model.model_dir={model_dir}")

    run(compose_test_cfg(overrides))
