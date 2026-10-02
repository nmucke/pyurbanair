"""Forward-model parameter routing and executable selection without CFD."""

import json
from pathlib import Path
from typing import Any

import pytest
import xarray as xr
from pyudales.forward_model import ForwardModel

SETTINGS = {
    "enabled": True,
    "canopy_height": 10.0,
    "height_band_over_H": [0.5, 1.5],
    "gradient_regularization": 1e-6,
    "log_multiplier_cap": 1.0,
}


def make_model(tmp_path: Path, **kwargs: Any) -> ForwardModel:
    from pyudales.forward_model import ForwardModel

    case = tmp_path / "case"
    case.mkdir(exist_ok=True)
    (case / "namoptions.999").write_text(
        "&RUN\n nprocx=1\n nprocy=1\n runtime=3.0\n/\n"
        "&NAMSUBGRID\n lvreman=.true.\n lsmagorinsky=.false.\n/\n"
    )
    return ForwardModel(case_dir=case, temp_dir=tmp_path / "temp", ncpu=1, **kwargs)


def test_coefficients_survive_separately_and_reset(tmp_path: Path) -> None:
    from pyudales.utils.namoptions_utils import NamoptionsFile

    model = make_model(
        tmp_path, model_discrepancy=SETTINGS, params=xr.Dataset({"sgs_bias_b0": 0.1})
    )
    assert model.params is not None
    assert "sgs_bias_b0" not in model.params
    model._apply_discrepancy_settings(xr.Dataset({"sgs_bias_b1": 0.4}))
    path = model.dirs.experiment_dir / "namoptions.300"
    nml = NamoptionsFile(path)
    assert nml.get_value_as_float("NAMSUBGRID", "sgs_bias_b0") == 0.1
    assert nml.get_value_as_float("NAMSUBGRID", "sgs_bias_b1") == 0.4
    model._apply_discrepancy_settings(xr.Dataset())
    nml = NamoptionsFile(path)
    assert nml.get_value_as_float("NAMSUBGRID", "sgs_bias_b1") == 0.0
    assert nml.get_value_as_float("NAMSUBGRID", "sgs_bias_b0") == 0.1
    model.model_discrepancy = {"enabled": False}
    model._apply_discrepancy_settings(None)
    assert "sgs_bias_b0" not in NamoptionsFile(path).get_section_keys("NAMSUBGRID")


def test_prepare_selects_variant_and_members_keep_executable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import pyudales.forward_model as module
    from pyudales.utils.forward_model_utils import create_new_forward_model

    calls = []

    def fake_prepare(**kwargs: Any) -> Path:
        calls.append(kwargs)
        root = tmp_path / ("extended" if kwargs["discrepancy_enabled"] else "stock")
        exe = root / "build" / "u-dales"
        exe.parent.mkdir(parents=True, exist_ok=True)
        exe.write_text("executable")
        return exe

    monkeypatch.setattr(module, "prepare_solver", fake_prepare)
    monkeypatch.setattr(module, "validate_solver", lambda exe, enabled: exe.is_file())
    monkeypatch.setattr(
        module, "solver_source_dir", lambda exe: exe.parent.parent / "source"
    )
    model = make_model(tmp_path, model_discrepancy=SETTINGS)
    assert calls == []  # Construction is solver-free.
    model.prepare_solver()
    model.prepare_solver()
    assert len(calls) == 1
    member = create_new_forward_model(model, tmp_path / "members", "001")
    assert member.dirs.solver_executable == model.dirs.solver_executable
    assert (
        str(model.dirs.solver_executable)
        in (member.dirs.experiment_dir / "config.sh").read_text()
    )
    member.model_discrepancy = {"enabled": False}
    member._apply_discrepancy_settings(None)
    member.prepare_solver()
    assert calls[-1]["discrepancy_enabled"] is False
    assert "stock" in str(member.dirs.solver_executable)


def test_metadata_survives_netcdf(tmp_path: Path) -> None:
    model = make_model(tmp_path, model_discrepancy=SETTINGS)
    result = xr.Dataset({"u": ("time", [1.0])})
    model._record_discrepancy(result)
    path = tmp_path / "state.nc"
    result.to_netcdf(path)
    with xr.open_dataset(path) as stored:
        assert json.loads(stored.attrs["model_discrepancy"]) == json.loads(
            result.attrs["model_discrepancy"]
        )


def test_rejects_incompatible_closure_before_build(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="[Vv]reman"):
        make_model(tmp_path, model_discrepancy=SETTINGS, closure="smagorinsky")


@pytest.mark.parametrize("role", ["model", "assim_model"])  # type: ignore[misc]
def test_unsupported_discrepancy_backends_fail_early(role: str) -> None:
    from omegaconf import OmegaConf

    from tests.conftest import load_script

    check_config = load_script("scripts/utils/inconsistency_check.py").check_config
    cfg = OmegaConf.create(
        {role: {"name": "pylbm", "forward_model": {"model_discrepancy": SETTINGS}}}
    )
    with pytest.raises(ValueError, match="needs pyudales with vreman"):
        check_config(cfg, "forward")


@pytest.mark.integration  # type: ignore[misc]
@pytest.mark.parametrize("ncpu", [1, 2])  # type: ignore[misc]
def test_native_fixed_discrepancy_forecast_and_restart(
    tmp_path: Path, ncpu: int
) -> None:
    """Exercise automatic compilation/preprocessing, native diagnostics and warm start."""
    import numpy as np
    from hydra.utils import instantiate

    from tests.conftest import compose

    cfg = compose(
        "forward",
        "+test=forward",
        "model=pyudales_stock",
        f"model.forward_model.ncpu={ncpu}",
        root=tmp_path,
    )
    model = instantiate(
        cfg.model.forward_model, model_discrepancy=SETTINGS, results_dir=None
    )
    model.run_preprocessing()
    first = model.run_single(
        params=xr.Dataset({"sgs_bias_b0": 0.1, "sgs_bias_b1": 0.2, "sgs_bias_b2": -0.1})
    )
    assert all(np.isfinite(first[name]).all() for name in ("u", "v", "w"))
    metadata = json.loads(first.attrs["model_discrepancy"])
    assert "multiplier_min=" in metadata["native_diagnostics"]
    second = model.run_single(
        state=first.isel(time=[-1]), params=xr.Dataset({"sgs_bias_b0": -0.2})
    )
    assert all(np.isfinite(second[name]).all() for name in ("u", "v", "w"))
    metadata = json.loads(second.attrs["model_discrepancy"])
    assert metadata["coefficients"]["sgs_bias_b0"] == -0.2
    assert metadata["coefficients"]["sgs_bias_b1"] == 0.0


def test_disabled_discrepancy_ignores_unused_coefficients(tmp_path: Path) -> None:
    model = make_model(
        tmp_path, params=xr.Dataset({"sgs_bias_b0": ("time", [1.0, 2.0])})
    )
    model._apply_discrepancy_settings(
        xr.Dataset({"sgs_bias_b0": ("time", [float("nan")])})
    )
    assert model._discrepancy_metadata is None
