import pathlib
import sys

import pytest

PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import tests.config as tests_config

sys.modules["scripts.config"] = tests_config

from pylbm.forward_model import ForwardModel as LBMForwardModel
from pyudales.forward_model import ForwardModel as UDALESForwardModel
from pyudales.utils.namoptions_utils import NamoptionsFile

from pyurbanair.utils.config_utils import (
    clean_forward_model_outputs,
    create_forward_model,
    create_true_params,
    model_args,
    prepare_forward_model,
)

import tests.config as config


@pytest.mark.parametrize("model", ["pylbm", "pyudales"])
def test_spinup_trims_output(model: str) -> None:
    """Verify spinup extends the run but trims output to simulation_time."""
    spinup_time = 2.0
    expected_steps = round(
        config.TIME["simulation_time"] / config.TIME["output_frequency"]
    )

    # --- Run without spinup ---
    fm = create_forward_model(model)
    prepare_forward_model(model, fm)
    clean_forward_model_outputs(model, fm)
    state_no_spinup = fm(params=create_true_params(model))
    assert state_no_spinup is not None
    assert state_no_spinup.sizes["time"] == expected_steps

    # --- Run with spinup ---
    args = model_args(model)
    args["spinup_time"] = spinup_time
    if model == "pylbm":
        fm_spinup = LBMForwardModel(**args)
    else:
        fm_spinup = UDALESForwardModel(**args)
    prepare_forward_model(model, fm_spinup)
    clean_forward_model_outputs(model, fm_spinup)

    # Assert spinup was actually configured (not silently ignored)
    if model == "pylbm":
        assert fm_spinup.spinup_time == spinup_time
    else:
        namoptions = NamoptionsFile(
            fm_spinup.dirs.experiment_dir
            / f"namoptions.{fm_spinup.dirs.experiment_name}"
        )
        runtime = float(namoptions.get_value("RUN", "runtime"))
        assert runtime == config.TIME["simulation_time"] + spinup_time

    state_spinup = fm_spinup(params=create_true_params(model))
    assert state_spinup is not None
    # Output is trimmed to simulation_time
    assert state_spinup.sizes["time"] == expected_steps
    # Time coordinate is rebased to start at 0
    assert int(state_spinup.time.values[0]) == 0
