import pathlib

import pytest
import xarray
from hydra.utils import instantiate
from pyudales.utils.namoptions_utils import NamoptionsFile

from pyurbanair.config.hydra_helpers import clean_outputs
from tests.conftest import compose

pytestmark = pytest.mark.integration


@pytest.mark.parametrize("model", ["pylbm_tiny", "pyudales_stock"])  # type: ignore[misc]
def test_spinup_trims_output(model: str, tmp_path: pathlib.Path) -> None:
    """Verify spinup extends the run but trims output to simulation_time."""
    spinup_time = 2.0

    overrides = ["+test=forward", f"model={model}"]
    cfg = compose(
        "forward", *overrides, "time.spinup_time=0.0", root=tmp_path / "no_spinup"
    )
    expected_steps = round(cfg.time.simulation_time / cfg.time.output_frequency)
    true_params = xarray.Dataset({"inflow_angle": 0.0, "velocity_magnitude": 5.0})

    # --- Run without spinup ---
    fm = instantiate(cfg.model.forward_model)
    instantiate(cfg.model.prepare, forward_model=fm)
    clean_outputs(cfg.model.name, fm)
    state_no_spinup = fm(params=true_params)
    assert state_no_spinup is not None
    assert state_no_spinup.sizes["time"] == expected_steps

    # --- Run with spinup ---
    cfg_spinup = compose(
        "forward",
        *overrides,
        f"time.spinup_time={spinup_time}",
        root=tmp_path / "spinup",
    )
    fm_spinup = instantiate(cfg_spinup.model.forward_model)
    instantiate(cfg_spinup.model.prepare, forward_model=fm_spinup)
    clean_outputs(cfg_spinup.model.name, fm_spinup)

    # Assert spinup was actually configured (not silently ignored)
    if cfg.model.name == "pylbm":
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
    # Time is rebased past the spinup: frames sit in (0, simulation_time], like
    # the run without spinup (uDALES's own output times jitter around tf).
    tf = cfg.time.output_frequency
    for state in (state_no_spinup, state_spinup):
        assert state.time.values[0] == pytest.approx(tf, abs=0.5 * tf)
        assert state.time.values[-1] == pytest.approx(
            cfg.time.simulation_time, abs=0.5 * tf
        )
