import pathlib

import pytest


@pytest.mark.parametrize(  # type: ignore[misc]
    "truth_model,assim_model",
    [
        pytest.param("pylbm", "pylbm", id="pylbm_pylbm"),
        pytest.param("pyudales", "pyudales", id="pyudales_pyudales"),
        pytest.param("pylbm", "pyudales", id="pylbm_pyudales_cross"),
        pytest.param("pyudales", "pylbm", id="pyudales_pylbm_cross"),
    ],
)
def test_run_time_varying_parameters_rollout_esmda(
    truth_model: str,
    assim_model: str,
    tmp_path: pathlib.Path,
    compose_test_cfg,
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

    run(compose_test_cfg(overrides))
