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
def test_run_state_and_parameter_esmda(
    truth_model: str,
    assim_model: str,
    tmp_path: pathlib.Path,
    compose_test_cfg,
) -> None:
    """Spin-up path: state+parameter ESMDA seeds its initial state ensemble
    from a warm-up call to the assim model and runs one assimilation window."""
    from scripts.run_state_and_parameter_esmda import run

    overrides = [
        f"model@truth_model={truth_model}",
        f"model@assim_model={assim_model}",
        # Smoother-mechanics test: global update (see test_localization for the
        # localized path; the default localization is degenerate at this size).
        "esmda.localization=null",
        "run.skip_viz=true",
        "ensemble.ensemble_size=2",
        "ensemble.num_parallel_processes=2",
        "esmda.num_steps=1",
        "esmda.num_assimilation_windows=1",
        f"paths.base_results_dir={tmp_path}",
    ]
    if truth_model == "pylbm":
        overrides.append("truth_model.forward_model.cuda=false")
    if assim_model == "pylbm":
        overrides.append("assim_model.forward_model.cuda=false")

    run(compose_test_cfg(overrides, config_name="run_state_and_parameter_esmda"))
