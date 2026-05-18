import pathlib

import pytest


@pytest.mark.parametrize(  # type: ignore[misc]
    "truth_model,assim_model,use_results_dir",
    [
        pytest.param("pylbm", "pylbm", False, id="pylbm_pylbm_in_memory"),
        pytest.param("pylbm", "pylbm", True, id="pylbm_pylbm_on_disk"),
        pytest.param("pyudales", "pyudales", False, id="pyudales_pyudales_in_memory"),
        pytest.param("pyudales", "pyudales", True, id="pyudales_pyudales_on_disk"),
        pytest.param("pylbm", "pyudales", False, id="pylbm_pyudales_cross"),
        pytest.param("pyudales", "pylbm", False, id="pyudales_pylbm_cross"),
    ],
)
def test_run_time_varying_parameter_esmda(
    truth_model: str,
    assim_model: str,
    use_results_dir: bool,
    tmp_path: pathlib.Path,
    compose_test_cfg,
) -> None:
    """Test run_time_varying_parameter_esmda.py with various model and storage combinations."""
    from scripts.run_time_varying_parameter_esmda import run

    overrides = [
        "esmda=time_varying_parameter",
        f"model@truth_model={truth_model}",
        f"model@assim_model={assim_model}",
        "run.skip_viz=true",
        "time_varying.num_time_points=3",
        "ensemble.ensemble_size=2",
        "ensemble.num_parallel_processes=2",
        "esmda.num_steps=1",
        "esmda.num_assimilation_windows=1",
    ]
    if truth_model == "pylbm":
        overrides.append("truth_model.forward_model.cuda=false")
    if assim_model == "pylbm":
        overrides.append("assim_model.forward_model.cuda=false")
    if use_results_dir:
        results_dir = tmp_path / "results"
        results_dir.mkdir()
        overrides.append(f"run.results_dir={results_dir}")

    run(compose_test_cfg(overrides))
