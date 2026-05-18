import pathlib

import numpy as np
import pytest
import xarray
from hydra.utils import instantiate
from pyurbanair.config.hydra_helpers import (
    clean_outputs,
    create_true_params,
)


def _build_params_ensemble(cfg, ensemble_size: int) -> xarray.Dataset:
    base_true_params = create_true_params(cfg.model.name, cfg.params.true)
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
    compose_test_cfg,
) -> None:
    overrides = [f"model={model_name}"]
    if model_name == "pylbm":
        overrides.append("model.forward_model.cuda=false")
    cfg = compose_test_cfg(overrides)

    subdir = cfg.model.init_subdir
    init_dir = init_root / subdir
    init_dir.mkdir(parents=True, exist_ok=True)

    params_ensemble = _build_params_ensemble(cfg, ensemble_size)
    params_ensemble.to_netcdf(init_dir / "params.nc")

    forward_model = instantiate(cfg.model.forward_model)
    instantiate(cfg.model.prepare, forward_model=forward_model)
    clean_outputs(cfg.model.name, forward_model)

    for member_idx in range(ensemble_size):
        member_params = params_ensemble.isel(ensemble=member_idx)
        state = forward_model(params=member_params)
        if state is None:
            raise RuntimeError(
                "Expected in-memory state while generating init conditions."
            )
        state.isel(time=-1).to_netcdf(init_dir / f"state_{member_idx}.nc")

    clean_outputs(cfg.model.name, forward_model)


@pytest.mark.parametrize(  # type: ignore[misc]
    "truth_model,assim_model",
    [
        pytest.param("pylbm", "pylbm", id="pylbm_pylbm"),
        pytest.param("pyudales", "pyudales", id="pyudales_pyudales"),
        pytest.param("pylbm", "pyudales", id="pylbm_pyudales_cross"),
        pytest.param("pyudales", "pylbm", id="pyudales_pylbm_cross"),
    ],
)
def test_run_state_and_parameter_esmda_with_init_conditions(
    truth_model: str,
    assim_model: str,
    tmp_path: pathlib.Path,
    compose_test_cfg,
) -> None:
    """Test run_state_and_parameter_esmda.py with init conditions enabled."""
    from scripts.run_state_and_parameter_esmda import run

    init_conditions_root = tmp_path / "esmda_init_conditions"
    ensemble_size = 2
    _write_init_conditions(
        truth_model,
        init_conditions_root,
        ensemble_size,
        compose_test_cfg,
    )
    if assim_model != truth_model:
        _write_init_conditions(
            assim_model,
            init_conditions_root,
            ensemble_size,
            compose_test_cfg,
        )

    overrides = [
        "esmda=state_and_parameter",
        f"model@truth_model={truth_model}",
        f"model@assim_model={assim_model}",
        "esmda.use_init_conditions=true",
        "run.skip_viz=true",
        f"ensemble.ensemble_size={ensemble_size}",
        f"ensemble.num_parallel_processes={ensemble_size}",
        "esmda.num_steps=1",
        "esmda.num_assimilation_windows=1",
        "esmda.true_sim_id=0",
        f"esmda.init_conditions_dir={init_conditions_root}",
    ]
    if truth_model == "pylbm":
        overrides.append("truth_model.forward_model.cuda=false")
    if assim_model == "pylbm":
        overrides.append("assim_model.forward_model.cuda=false")

    run(compose_test_cfg(overrides))
