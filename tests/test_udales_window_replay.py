"""Native hidden-state transactions, without launching the CFD executable."""

import json
import pickle
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import xarray as xr
from pyudales.ensemble_forward_model import EnsembleForwardModel
from pyudales.forward_model import ForwardModel
from pyudales.utils.inlet_turbulence_utils import derive_seed, write_elapsed_time
from pyudales.utils.warm_start_utils import CARRY_DIRNAME, CARRY_META_NAME
from pyudales.utils.window_checkpoint import validate_carry
from scipy.io import FortranFile

from tests.test_udales_discrepancy_wiring import SETTINGS, make_model


def _model(tmp_path: Path, name: str = "300") -> ForwardModel:
    tmp_path.mkdir(parents=True, exist_ok=True)
    model = make_model(tmp_path, model_discrepancy=SETTINGS, experiment_name=name)
    nam = model.dirs.experiment_dir / f"namoptions.{name}"
    nam.write_text(nam.read_text() + "&DOMAIN\nitot=8\njtot=6\nktot=4\n/\n")
    return model


def _carry(model: ForwardModel, value: float) -> None:
    root = model.dirs.experiment_dir / CARRY_DIRNAME
    root.mkdir(exist_ok=True)
    name = f"initd00000003_000_000.{model.dirs.experiment_name}"
    with FortranFile(root / name, "w") as handle:
        handle.write_record(np.zeros(8 * 6 * 4))
        handle.write_record(np.zeros(8 * 6 * 4 * 5, dtype=np.int32))
        for _ in range(10):
            handle.write_record(np.full(10 * 8 * 5, value))
        handle.write_record(np.array([3.0, 0.1]))
    (root / CARRY_META_NAME).write_text(
        json.dumps(
            {
                "experiment_name": model.dirs.experiment_name,
                "grid": {"itot": 8, "jtot": 6, "ktot": 4},
                "files": [name],
            }
        )
    )


def _bytes(model: ForwardModel) -> dict[str, bytes]:
    root = model.dirs.experiment_dir
    return {
        str(path.relative_to(root)): path.read_bytes()
        for path in root.rglob("*")
        if path.is_file()
    }


@pytest.mark.parametrize("worker_copy", [False, True])  # type: ignore[misc]
def test_repeated_attempts_replay_disk_and_python_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, worker_copy: bool
) -> None:
    model = _model(tmp_path)
    _carry(model, 7.0)
    model._elapsed_time = 12.0
    write_elapsed_time(model.dirs, 12.0)
    baseline = _bytes(model)
    model.begin_forecast_window()
    starts = []

    def forecast(self: ForwardModel, **kwargs: Any) -> xr.Dataset:
        starts.append((_bytes(self), self._elapsed_time, self.spinup_time))
        _carry(self, 99.0)
        self._elapsed_time += 3.0
        write_elapsed_time(self.dirs, self._elapsed_time)
        (self.dirs.experiment_dir / "generated.inp").write_text("attempt")
        return xr.Dataset({"u": 1.0})

    monkeypatch.setattr(ForwardModel, "_run_single", forecast)
    for _ in range(2):
        member = pickle.loads(pickle.dumps(model)) if worker_copy else model
        member.run_single()
    assert starts[0] == starts[1]
    assert starts[0][0] == baseline
    model.end_forecast_window(commit=True)
    assert model._elapsed_time == 15.0
    assert not list(model.dirs.experiment_dir.parent.glob(".window_checkpoint_*"))
    model.end_forecast_window(commit=True)  # Endpoint accepted exactly once.


def test_failed_attempt_and_window_rollback_restore_original(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model = _model(tmp_path)
    baseline = _bytes(model)
    model.begin_forecast_window()

    def fail(**kwargs: Any) -> xr.Dataset:
        _carry(model, 99.0)
        model._elapsed_time = 123.0
        model.spinup_time = 99.0
        (model.dirs.experiment_dir / "stale.txt").write_text("stale")
        raise RuntimeError("solver failed")

    monkeypatch.setattr(model, "_run_single", fail)
    with pytest.raises(RuntimeError, match="solver failed"):
        model.run_single()
    assert _bytes(model) == baseline
    assert model._elapsed_time == 0.0
    model.end_forecast_window(commit=False)
    assert _bytes(model) == baseline


def test_donor_snapshot_identity_survives_next_attempt_and_rolls_back(
    tmp_path: Path,
) -> None:
    donor = _model(tmp_path / "donor", "001")
    failed = _model(tmp_path / "failed", "002")
    _carry(donor, 1.0)
    _carry(failed, 2.0)
    original = _bytes(failed)
    donor.begin_forecast_window()
    failed.begin_forecast_window()
    failed._forecast_window_start = donor._forecast_window_start
    _carry(donor, 99.0)
    failed.restore_forecast_window()
    validate_carry(failed, required=True)
    assert failed.inlet_turbulence["seed"] == derive_seed("001")
    with FortranFile(
        failed.dirs.experiment_dir / CARRY_DIRNAME / "initd00000003_000_000.002", "r"
    ) as handle:
        handle.read_record(np.uint8)
        handle.read_record(np.uint8)
        assert np.all(handle.read_record(np.float64) == 1.0)
    failed.end_forecast_window(commit=False)
    assert _bytes(failed) == original
    donor.end_forecast_window(commit=False)


@pytest.mark.parametrize("damage", ["missing_rank", "bad_record", "bad_metadata"])  # type: ignore[misc]
def test_invalid_carry_rejected_before_snapshot(tmp_path: Path, damage: str) -> None:
    model = _model(tmp_path)
    _carry(model, 1.0)
    root = model.dirs.experiment_dir / CARRY_DIRNAME
    if damage == "missing_rank":
        next(root.glob("initd*")).unlink()
    elif damage == "bad_record":
        next(root.glob("initd*")).write_bytes(b"broken")
    else:
        (root / CARRY_META_NAME).write_text("broken")
    baseline = _bytes(model)
    with pytest.raises(ValueError, match="Invalid discrepancy window carry"):
        model.begin_forecast_window()
    assert _bytes(model) == baseline
    assert not list(model.dirs.experiment_dir.parent.glob(".window_checkpoint_*"))


def test_snapshot_corruption_rejected_before_live_inputs_change(tmp_path: Path) -> None:
    model = _model(tmp_path)
    model.begin_forecast_window()
    checkpoint = model._forecast_window_start
    assert checkpoint is not None
    (checkpoint.root / "inputs" / "config.sh").write_text("corrupt")
    baseline = _bytes(model)
    with pytest.raises(ValueError, match="Corrupt discrepancy window checkpoint"):
        model.restore_forecast_window()
    assert _bytes(model) == baseline
    shutil.rmtree(checkpoint.root)


def test_disabled_window_is_byte_identical_noop(tmp_path: Path) -> None:
    model = make_model(tmp_path)
    baseline = _bytes(model)
    model.begin_forecast_window()
    model.restore_forecast_window()
    model.end_forecast_window(commit=True)
    assert _bytes(model) == baseline
    assert not list(model.dirs.experiment_dir.parent.glob(".window_checkpoint_*"))


def test_replay_failure_params_are_exact_donor_clones(tmp_path: Path) -> None:
    ensemble = object.__new__(EnsembleForwardModel)  # type: ignore[type-abstract]
    ensemble.ensemble_forward_models = [_model(tmp_path)]
    ensemble._last_failure_substitutions = {1: 0}
    ensemble._failure_jitter_scale = 100.0
    params = xr.Dataset({"sgs_bias_b0": ("ensemble", [0.2, 0.8]), "constant": 5.0})
    actual = ensemble.apply_failure_substitutions_to_params(params)
    np.testing.assert_array_equal(actual.sgs_bias_b0.values, [0.2, 0.2])
    assert float(actual.constant) == 5.0
    np.testing.assert_array_equal(params.sgs_bias_b0.values, [0.2, 0.8])


@pytest.mark.parametrize("value", [-1e-8, -1.0, np.inf, np.nan])  # type: ignore[misc]
def test_discrepancy_runtime_sgs_constant_rejected_before_rounding(
    tmp_path: Path, value: float
) -> None:
    model = _model(tmp_path)
    baseline = _bytes(model)
    with pytest.raises(ValueError, match="finite nonnegative"):
        model._apply_sgs_setting(xr.Dataset({"sgs_constant": value}))
    assert _bytes(model) == baseline


def test_runtime_params_and_template_restore_together(tmp_path: Path) -> None:
    model = _model(tmp_path)
    model.params = xr.Dataset({"inflow_speed": 2.0})
    model._warmstart_template_dir.mkdir()
    template = model._warmstart_template_dir / "initd00000001_000_000.300"
    template.write_bytes(b"original template")
    model.warmstart_template_file = template
    model.begin_forecast_window()
    model.params = xr.Dataset({"inflow_speed": 9.0, "sgs_constant": 99.0})
    template.write_bytes(b"changed")
    model.warmstart_template_file = None
    model.restore_forecast_window()
    assert float(model.params.inflow_speed) == 2.0
    assert "sgs_constant" not in model.params
    assert model.warmstart_template_file == template
    assert template.read_bytes() == b"original template"
    model.end_forecast_window(commit=False)


def test_member_clone_rebases_all_template_ranks(tmp_path: Path) -> None:
    from pyudales.utils.forward_model_utils import create_new_forward_model

    model = _model(tmp_path)
    model._warmstart_template_dir.mkdir()
    for rank in range(2):
        path = model._warmstart_template_dir / f"initd00000001_{rank:03d}_000.300"
        path.write_bytes(bytes([rank]))
    model.warmstart_template_file = (
        model._warmstart_template_dir / "initd00000001_000_000.300"
    )
    member = create_new_forward_model(model, tmp_path / "members", "001")
    assert member.warmstart_template_file is not None
    assert member.warmstart_template_file.read_bytes() == b"\x00"
    assert (
        member._warmstart_template_dir / "initd00000001_001_000.001"
    ).read_bytes() == b"\x01"
    assert member._warmstart_template_dir != model._warmstart_template_dir


@pytest.mark.parametrize("on_disk", [False, True])  # type: ignore[misc]
def test_ensemble_failure_propagates_start_identity_and_endpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, on_disk: bool
) -> None:
    from pyurbanair.base_ensemble_forward_model import BaseEnsembleForwardModel

    members = [_model(tmp_path / str(index), f"{index:03d}") for index in range(2)]
    for index, member in enumerate(members):
        _carry(member, float(index + 1))
    baseline = [_bytes(member) for member in members]
    ensemble = object.__new__(EnsembleForwardModel)  # type: ignore[type-abstract]
    ensemble.ensemble_forward_models = list(members)
    ensemble.ensemble_size = 2
    ensemble.begin_forecast_window()
    starts = [member._forecast_window_start for member in members]

    def forecast(self: Any, **kwargs: Any) -> xr.Dataset | None:
        self._last_failure_substitutions = {1: 0}
        _carry(members[0], 77.0)
        write_elapsed_time(members[0].dirs, 3.0)
        (members[0].dirs.experiment_dir / "model_discrepancy.json").write_text("{}")
        return None if on_disk else xr.Dataset({"u": ("ensemble", [1.0, 1.0])})

    monkeypatch.setattr(BaseEnsembleForwardModel, "run_ensemble", forecast)
    result = ensemble.run_ensemble()
    assert (result is None) == on_disk
    assert members[1]._forecast_window_start is starts[0]
    assert members[1]._forecast_window_original is starts[1]
    assert members[1]._elapsed_time == 3.0
    validate_carry(members[1], required=True)
    ensemble.restore_forecast_window()
    with FortranFile(
        members[1].dirs.experiment_dir / CARRY_DIRNAME / "initd00000003_000_000.001",
        "r",
    ) as handle:
        handle.read_record(np.uint8)
        handle.read_record(np.uint8)
        assert np.all(handle.read_record(np.float64) == 1.0)
    ensemble.end_forecast_window(commit=False)
    assert [_bytes(member) for member in members] == baseline


def test_commit_validates_all_members_before_releasing_rollback(tmp_path: Path) -> None:
    members = [_model(tmp_path / str(index), f"{index:03d}") for index in range(2)]
    ensemble = object.__new__(EnsembleForwardModel)  # type: ignore[type-abstract]
    ensemble.ensemble_forward_models = list(members)
    ensemble.begin_forecast_window()
    _carry(members[0], 10.0)
    with pytest.raises(ValueError, match="did not produce a native carry"):
        ensemble.end_forecast_window(commit=True)
    assert all(member._forecast_window_original is not None for member in members)
    ensemble.end_forecast_window(commit=False)
    assert not any(
        (member.dirs.experiment_dir / CARRY_DIRNAME).exists() for member in members
    )


def test_checkpoint_cleanup_failure_keeps_committed_ensemble(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from pyudales.utils.window_checkpoint import WindowCheckpoint

    members = [_model(tmp_path / str(index), f"{index:03d}") for index in range(2)]
    ensemble = object.__new__(EnsembleForwardModel)  # type: ignore[type-abstract]
    ensemble.ensemble_forward_models = list(members)
    ensemble.begin_forecast_window()
    checkpoints = [member._forecast_window_original for member in members]
    for member in members:
        _carry(member, 10.0)
        write_elapsed_time(member.dirs, 3.0)

    def cannot_remove(self: Any) -> None:
        raise OSError("scratch filesystem busy")

    monkeypatch.setattr(WindowCheckpoint, "remove", cannot_remove)
    ensemble.end_forecast_window(commit=True)
    assert all(member._forecast_window_original is None for member in members)
    assert all(member._elapsed_time == 3.0 for member in members)
    for checkpoint in checkpoints:
        assert checkpoint is not None
        shutil.rmtree(checkpoint.root)


def test_reused_disk_results_do_not_turn_cold_replay_into_warm_start(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    member = _model(tmp_path)
    results = tmp_path / "results"
    results.mkdir()
    xr.Dataset({"u": 999.0}).to_netcdf(results / "state_0.nc")
    ensemble = object.__new__(EnsembleForwardModel)  # type: ignore[type-abstract]
    ensemble.ensemble_forward_models = [member]
    ensemble.ensemble_size = 1
    ensemble.results_dir = results
    ensemble.save_on_disk = True
    ensemble._failure_policy = "raise"
    starts = []

    def forecast(**kwargs: Any) -> xr.Dataset:
        starts.append(kwargs["state"])
        return xr.Dataset({"u": 5.0})

    monkeypatch.setattr(member, "_run_single", forecast)
    # The legacy implicit disk warm start remains available outside a window.
    previous = ensemble.get_member_state(None, 0)
    assert previous is not None
    assert float(previous.u) == 999.0
    for _ in range(2):
        ensemble.begin_forecast_window()
        ensemble._run_ensemble_sequentially_on_disk(state=None, sim_name="state")
        ensemble.end_forecast_window(commit=False)
    assert starts == [None, None]


@pytest.mark.parametrize("clock", [[4.0, 0.1], [3.0, 0.0], [-1.0, 0.1]])  # type: ignore[misc]
def test_multirank_carry_rejects_inconsistent_or_invalid_clocks(
    tmp_path: Path, clock: list[float]
) -> None:
    from tests.test_udales_discrepancy_warmstart import _case, _records

    model = _model(tmp_path / "model")
    dirs, _, templates = _case(tmp_path / "ranks")
    model.dirs = dirs
    root = dirs.experiment_dir / CARRY_DIRNAME
    root.mkdir()
    for template in templates:
        shutil.copy2(template, root / template.name)
    records = _records(root / templates[1].name)
    records[-1] = np.asarray(clock, dtype=np.float64).view(np.uint8)
    with FortranFile(root / templates[1].name, "w") as handle:
        for record in records:
            handle.write_record(record)
    (root / CARRY_META_NAME).write_text(
        json.dumps(
            {
                "experiment_name": "000",
                "grid": {"itot": 8, "jtot": 6, "ktot": 4},
                "files": [path.name for path in templates],
            }
        )
    )
    with pytest.raises(ValueError, match="clock"):
        validate_carry(model)
