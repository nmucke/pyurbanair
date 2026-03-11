import pathlib
import sys

import numpy as np
import pytest
import xarray

# Patch scripts.config to use tests.config before any script imports
PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import tests.config as tests_config

sys.modules["scripts.config"] = tests_config


def _build_params_ensemble(model_name: str, ensemble_size: int) -> xarray.Dataset:
    base_true_params = tests_config.create_true_params(model_name)
    members: list[xarray.Dataset] = []
    for member_idx in range(ensemble_size):
        member = base_true_params.copy(deep=True)
        member["inflow_angle"] = member["inflow_angle"] + float(member_idx)
        members.append(member)
    return xarray.concat(members, dim="ensemble").assign_coords(
        ensemble=np.arange(ensemble_size)
    )


def _write_init_conditions(
    model_name: str,
    init_root: pathlib.Path,
    ensemble_size: int,
) -> None:
    subdir = "lbm" if model_name == "pylbm" else "udales"
    init_dir = init_root / subdir
    init_dir.mkdir(parents=True, exist_ok=True)

    params_ensemble = _build_params_ensemble(model_name, ensemble_size)
    params_ensemble.to_netcdf(init_dir / "params.nc")

    forward_model = tests_config.create_forward_model(model_name)
    tests_config.prepare_forward_model(model_name, forward_model)
    tests_config.clean_forward_model_outputs(model_name, forward_model)

    for member_idx in range(ensemble_size):
        member_params = params_ensemble.isel(ensemble=member_idx)
        state = forward_model(params=member_params)
        if state is None:
            raise RuntimeError(
                "Expected in-memory state while generating init conditions."
            )
        state.isel(time=-1).to_netcdf(init_dir / f"state_{member_idx}.nc")

    tests_config.clean_forward_model_outputs(model_name, forward_model)


@pytest.mark.parametrize(  # type: ignore[misc]
    "truth_model,assim_model",
    [
        pytest.param("pylbm", "pylbm", id="pylbm_pylbm"),
        pytest.param("pyudales", "pyudales", id="pyudales_pyudales"),
        pytest.param("pylbm", "pyudales", id="pylbm_pyudales_cross"),
        pytest.param("pyudales", "pylbm", id="pyudales_pylbm_cross"),
    ],
)
def test_run_rollout_esmda_with_init_conditions(
    truth_model: str,
    assim_model: str,
    tmp_path: pathlib.Path,
) -> None:
    """Test run_rollout_esmda.py with init conditions enabled."""
    from scripts.run_rollout_esmda import main

    original_argv = sys.argv
    original_ensemble = tests_config.ENSEMBLE.copy()
    original_esmda = tests_config.ESMDA.copy()

    init_conditions_root = tmp_path / "esmda_init_conditions"
    ensemble_size = 2
    _write_init_conditions(truth_model, init_conditions_root, ensemble_size)
    if assim_model != truth_model:
        _write_init_conditions(assim_model, init_conditions_root, ensemble_size)

    sys.argv = [
        "run_rollout_esmda",
        "--truth-model",
        truth_model,
        "--assim-model",
        assim_model,
        "--init-conditions",
        "--skip-viz",
    ]

    try:
        tests_config.ENSEMBLE["ensemble_size"] = ensemble_size
        tests_config.ENSEMBLE["num_parallel_processes"] = ensemble_size
        tests_config.ESMDA["num_steps"] = 1
        tests_config.ESMDA["num_assimilation_windows"] = 1
        tests_config.ESMDA["true_sim_id"] = 0
        tests_config.ESMDA["init_conditions_dir"] = str(init_conditions_root)

        main()
    finally:
        sys.argv = original_argv
        tests_config.ENSEMBLE.update(original_ensemble)
        tests_config.ESMDA.update(original_esmda)
