"""Native cross-stack handoff for hybrid smoothing/filtering, without CFD."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest
import xarray as xr
from data_assimilation.filter_smoothing.base import FilterSmoothing
from pyudales.ensemble_forward_model import EnsembleForwardModel
from pyudales.forward_model import ForwardModel
from pyudales.utils.inlet_turbulence_utils import read_elapsed_time, write_elapsed_time
from pyudales.utils.namoptions_utils import NamoptionsFile
from pyudales.utils.warm_start_utils import CARRY_DIRNAME
from pyudales.utils.window_checkpoint import validate_carry
from scipy.io import FortranFile

from pyurbanair.base_ensemble_forward_model import BaseForwardModel, FailurePolicy
from tests.test_udales_window_replay import _carry, _model


def _ensemble(
    members: list[ForwardModel], policy: FailurePolicy = "raise"
) -> EnsembleForwardModel:
    ensemble = object.__new__(EnsembleForwardModel)  # type: ignore[type-abstract]
    ensemble.ensemble_forward_models = cast(list[BaseForwardModel], members)
    ensemble.ensemble_size = len(members)
    ensemble._failure_policy = policy
    return ensemble


def _members(ensemble: EnsembleForwardModel) -> list[ForwardModel]:
    return cast(list[ForwardModel], ensemble.ensemble_forward_models)


def _pair(
    tmp_path: Path, n: int = 2
) -> tuple[EnsembleForwardModel, EnsembleForwardModel]:
    smoother = _ensemble(
        [_model(tmp_path / "smoother" / str(i), f"{i:03d}") for i in range(n)]
    )
    filtering = _ensemble(
        [_model(tmp_path / "filter" / str(i), f"{i:03d}") for i in range(n)]
    )
    for ensemble, runtime in ((smoother, 20.0), (filtering, 2.0)):
        for member in _members(ensemble):
            member._simulation_time = runtime
            nam = NamoptionsFile(
                member.dirs.experiment_dir / f"namoptions.{member.dirs.experiment_name}"
            )
            nam.set_value("RUN", "runtime", runtime)
            nam.write()
    return smoother, filtering


def _native_value(model: ForwardModel) -> float:
    restart = next((model.dirs.experiment_dir / CARRY_DIRNAME).glob("initd*"))
    with FortranFile(restart, "r") as handle:
        handle.read_record(np.uint8)
        handle.read_record(np.uint8)
        return float(handle.read_record(np.float64)[0])


def test_handoff_copies_each_accepted_endpoint_without_runtime_files(
    tmp_path: Path,
) -> None:
    smoother, filtering = _pair(tmp_path)
    destination_bytes = [
        (
            member.dirs.experiment_dir / f"namoptions.{member.dirs.experiment_name}"
        ).read_bytes()
        for member in _members(smoother)
    ]
    destination_runtime = [member._simulation_time for member in _members(smoother)]
    assert destination_runtime == [20.0, 20.0]
    assert all(member._simulation_time == 2.0 for member in _members(filtering))
    for index, base_member in enumerate(_members(filtering)):
        member = base_member
        _carry(member, float(index + 3))
        write_elapsed_time(member.dirs, float((index + 1) * 12))
        member.inlet_turbulence["seed"] = 100 + index
    smoother.synchronize_forecast_state_from(filtering)

    for index, base_member in enumerate(_members(smoother)):
        member = base_member
        validate_carry(member, required=True)
        assert _native_value(member) == float(index + 3)
        assert read_elapsed_time(member.dirs) == float((index + 1) * 12)
        assert member._elapsed_time == float((index + 1) * 12)
        assert member.inlet_turbulence["seed"] == 100 + index
        nam = member.dirs.experiment_dir / f"namoptions.{member.dirs.experiment_name}"
        assert nam.read_bytes() == destination_bytes[index]
        assert member._simulation_time == destination_runtime[index]

    # A second window replaces each prior native endpoint while keeping the
    # smoother's full-window solver settings.
    for index, member in enumerate(_members(filtering)):
        _carry(member, float(index + 30))
        write_elapsed_time(member.dirs, float((index + 1) * 24))
        member.inlet_turbulence["seed"] = 200 + index
    smoother.synchronize_forecast_state_from(filtering)
    for index, member in enumerate(_members(smoother)):
        assert _native_value(member) == float(index + 30)
        assert read_elapsed_time(member.dirs) == float((index + 1) * 24)
        assert member.inlet_turbulence["seed"] == 200 + index
        nam = member.dirs.experiment_dir / f"namoptions.{member.dirs.experiment_name}"
        assert nam.read_bytes() == destination_bytes[index]
        assert member._simulation_time == 20.0


def test_cold_source_clears_stale_destination_carry(tmp_path: Path) -> None:
    smoother, filtering = _pair(tmp_path, 1)
    destination = _members(smoother)[0]
    _carry(destination, 99.0)
    write_elapsed_time(destination.dirs, 45.0)
    smoother.synchronize_forecast_state_from(filtering)
    assert not (destination.dirs.experiment_dir / CARRY_DIRNAME).exists()
    assert read_elapsed_time(destination.dirs) == 0.0
    assert destination._elapsed_time == 0.0


def test_invalid_second_source_member_leaves_all_destinations_untouched(
    tmp_path: Path,
) -> None:
    smoother, filtering = _pair(tmp_path)
    for index, member in enumerate(_members(smoother)):
        _carry(member, float(index + 50))
    for index, member in enumerate(_members(filtering)):
        _carry(member, float(index + 1))
        write_elapsed_time(member.dirs, 10.0)
    second = _members(filtering)[1]
    next((second.dirs.experiment_dir / CARRY_DIRNAME).glob("initd*")).unlink()
    with pytest.raises(ValueError, match="Invalid discrepancy window carry"):
        smoother.synchronize_forecast_state_from(filtering)
    for index, member in enumerate(_members(smoother)):
        assert _native_value(member) == float(index + 50)


def test_mismatched_layout_rejected_before_copy(tmp_path: Path) -> None:
    smoother, filtering = _pair(tmp_path, 1)
    source = _members(filtering)[0]
    _carry(source, 3.0)
    write_elapsed_time(source.dirs, 12.0)
    destination = _members(smoother)[0]
    nam_path = (
        destination.dirs.experiment_dir
        / f"namoptions.{destination.dirs.experiment_name}"
    )
    nam = NamoptionsFile(nam_path)
    nam.set_value("DOMAIN", "itot", 10)
    nam.write()
    with pytest.raises(ValueError, match="matching grid and MPI ranks"):
        smoother.synchronize_forecast_state_from(filtering)
    assert not (destination.dirs.experiment_dir / CARRY_DIRNAME).exists()


@pytest.mark.parametrize(  # type: ignore[misc]
    "invalid", ["active", "policy", "count", "clock", "missing_carry"]
)
def test_handoff_rejects_unsupported_or_corrupt_source(
    tmp_path: Path, invalid: str
) -> None:
    smoother, filtering = _pair(tmp_path, 1)
    source = _members(filtering)[0]
    _carry(source, 1.0)
    write_elapsed_time(source.dirs, 5.0)
    if invalid == "active":
        source.begin_forecast_window()
        expected = "active forecast window"
    elif invalid == "policy":
        filtering._failure_policy = "resample_from_successes"
        expected = "failure.policy=raise"
    elif invalid == "count":
        smoother.ensemble_size = 2
        expected = "matching ensemble sizes"
    elif invalid == "missing_carry":
        from pyudales.utils.warm_start_utils import clear_carry

        clear_carry(source.dirs)
        expected = "missing native carry"
    else:
        # Discover the backend's named clock file rather than relying on a
        # string literal in the assertion below.
        from pyudales.utils.inlet_turbulence_utils import elapsed_time_path

        path = elapsed_time_path(source.dirs)
        path.write_text(json.dumps({"experiment_name": "wrong", "elapsed_time": 5}))
        expected = "invalid physical clock"
    try:
        with pytest.raises((RuntimeError, ValueError), match=expected):
            smoother.synchronize_forecast_state_from(filtering)
    finally:
        if invalid == "active":
            source.end_forecast_window(commit=False)
    assert not (_members(smoother)[0].dirs.experiment_dir / CARRY_DIRNAME).exists()


def test_hybrid_calls_handoff_before_smoother_and_rejects_joint_filter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    smoother_model = SimpleNamespace(
        forecast_window_replay_enabled=True,
        _failure_policy="raise",
        synchronize_forecast_state_from=lambda source: events.append("handoff"),
    )
    filter_model = SimpleNamespace(_failure_policy="raise")

    class Smoother:
        forward_model = smoother_model

        def __call__(self, **kwargs: Any) -> xr.Dataset:
            events.append("smoother")
            return cast(xr.Dataset, kwargs["params"])

    hybrid = cast(Any, object.__new__(FilterSmoothing))
    hybrid.smoother = Smoother()
    hybrid.filter = SimpleNamespace(mode="state", forward_model=filter_model)
    hybrid.tempering = SimpleNamespace(likelihood_allocation="filter_only")
    monkeypatch.setattr(hybrid, "_validate_observations", lambda value: value)
    monkeypatch.setattr(hybrid, "_check_tempering", lambda: None)
    monkeypatch.setattr(hybrid, "_run_static", lambda *args: "finished")
    params = xr.Dataset({"sgs_bias_b0": ("ensemble", [0.1, 0.2])})
    observations = [xr.DataArray(np.zeros((1, 1)), dims=("time", "obs"))]
    assert hybrid.run(params=params, observations=observations) == "finished"
    assert events == ["handoff", "smoother"]
    events.clear()
    hybrid.filter.mode = "joint"
    with pytest.raises(ValueError, match="state-only filter"):
        hybrid.run(params=params, observations=observations)
    assert events == []
