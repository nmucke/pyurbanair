import pytest
from hydra.utils import instantiate
from pyudales.utils.namoptions_utils import NamoptionsFile
from pyurbanair.config.hydra_helpers import clean_outputs


@pytest.mark.parametrize("model", ["pylbm", "pyudales"])
def test_spinup_trims_output(model: str, compose_test_cfg) -> None:
    """Verify spinup extends the run but trims output to simulation_time."""
    spinup_time = 2.0

    # Static scalar params: the declarative sampler draws a single member, then
    # we drop the ensemble dim the way the forward model does for single runs.
    overrides = [f"model={model}", "params=static"]
    if model == "pylbm":
        overrides.append("model.forward_model.cuda=false")

    cfg = compose_test_cfg(overrides)
    expected_steps = round(cfg.time.simulation_time / cfg.time.output_frequency)
    true_params = instantiate(cfg.params).sample(1).isel(ensemble=0, drop=True)

    # --- Run without spinup ---
    fm = instantiate(cfg.model.forward_model)
    instantiate(cfg.model.prepare, forward_model=fm)
    clean_outputs(cfg.model.name, fm)
    state_no_spinup = fm(params=true_params)
    assert state_no_spinup is not None
    assert state_no_spinup.sizes["time"] == expected_steps

    # --- Run with spinup ---
    cfg_spinup = compose_test_cfg(
        overrides + [f"time.spinup_time={spinup_time}"],
    )
    fm_spinup = instantiate(cfg_spinup.model.forward_model)
    instantiate(cfg_spinup.model.prepare, forward_model=fm_spinup)
    clean_outputs(cfg_spinup.model.name, fm_spinup)

    # Assert spinup was actually configured (not silently ignored)
    if model == "pylbm":
        assert fm_spinup.spinup_time == spinup_time
    else:
        namoptions = NamoptionsFile(
            fm_spinup.dirs.experiment_dir
            / f"namoptions.{fm_spinup.dirs.experiment_name}"
        )
        runtime = float(namoptions.get_value("RUN", "runtime"))
        assert runtime == cfg_spinup.time.simulation_time + spinup_time

    state_spinup = fm_spinup(params=true_params)
    assert state_spinup is not None
    # Output is trimmed to simulation_time
    assert state_spinup.sizes["time"] == expected_steps
    # Time coordinate is rebased to start at 0
    assert int(state_spinup.time.values[0]) == 0
