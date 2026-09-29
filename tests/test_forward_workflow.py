"""Forward artifact and initialization contracts without invoking a CFD solver."""

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import xarray as xr
from omegaconf import DictConfig, OmegaConf

from pyurbanair.workflows import forward


def _config(
    tmp_path: Path, ensemble: bool = False, dynamic: bool = False
) -> DictConfig:
    return OmegaConf.create(
        {
            "run": {
                "ensemble": ensemble,
                "rollout_steps": 1,
                "skip_viz": True,
                "results_dir": None,
            },
            "time": {"simulation_time": 2.0},
            "ensemble": {"ensemble_size": 2},
            "model": {
                "name": "fake",
                "forward_model": {"role": "model"},
                "ensemble_model": {"role": "ensemble"},
                "prepare": {"role": "prepare"},
            },
            "params": {
                "role": "params",
                **({"seconds_per_knot": 1.0} if dynamic else {}),
            },
            "paths": {"base_results_dir": str(tmp_path)},
        }
    )


def _state(members: int = 1) -> xr.Dataset:
    data = np.arange(members * 2, dtype=float).reshape(members, 2, 1, 1, 1)
    return xr.Dataset(
        {v: (("ensemble", "time", "z", "y", "x"), data) for v in ("u", "v", "w")},
        coords={
            "ensemble": np.arange(members),
            "time": [1.0, 2.0],
            "z": [0.5],
            "y": [0.5],
            "x": [0.5],
        },
    )


class _Sampler:
    time_coords = [0.0, 1.0]

    def __init__(self, dynamic: bool) -> None:
        self.dynamic = dynamic

    def sample(self, members: int) -> xr.Dataset:
        dims = ("ensemble", "time") if self.dynamic else ("ensemble",)
        values = np.arange(members, dtype=float) + 7
        coords: dict[str, Any] = {"ensemble": np.arange(members)}
        if self.dynamic:
            values = np.repeat(values[:, None], 2, axis=1)
            coords["time"] = self.time_coords
        return xr.Dataset(
            {v: (dims, values) for v in ("inflow_angle", "velocity_magnitude")},
            coords=coords,
        )

    def extrapolate(self, params: xr.Dataset, times: Any, key: Any) -> xr.Dataset:
        return params + 1


class _Model:
    def __init__(self, cfg: DictConfig, fail_on: int | None = None) -> None:
        self.cfg = cfg
        self.calls: list[xr.Dataset | None] = []
        self.fail_on = fail_on

    def __call__(
        self, *, params: xr.Dataset, state: xr.Dataset | None = None
    ) -> xr.Dataset:
        self.calls.append(state)
        if len(self.calls) == self.fail_on:
            raise RuntimeError("solver failed")
        result = _state(2 if self.cfg.run.ensemble else 1)
        return result if self.cfg.run.ensemble else result.isel(ensemble=0, drop=True)

    def run_ensemble(
        self, *, params: xr.Dataset, state: xr.Dataset | None, sim_name: str
    ) -> xr.Dataset:
        return self(params=params, state=state)


def _install(monkeypatch: pytest.MonkeyPatch, cfg: DictConfig, model: _Model) -> None:
    def instantiate(node: DictConfig, **kwargs: Any) -> Any:
        return (
            _Sampler("seconds_per_knot" in cfg.params)
            if node.role == "params"
            else model
        )

    monkeypatch.setattr(forward, "instantiate", instantiate)
    monkeypatch.setattr(forward, "clean_outputs", lambda **kwargs: None)


@pytest.mark.parametrize("ensemble", [False, True])  # type: ignore[misc, unused-ignore]
@pytest.mark.parametrize("dynamic", [False, True])  # type: ignore[misc, unused-ignore]
def test_complete_artifacts_all_windows_members(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, ensemble: bool, dynamic: bool
) -> None:
    cfg = _config(tmp_path, ensemble, dynamic)
    model = _Model(cfg)
    _install(monkeypatch, cfg, model)
    root = forward.run(
        cfg,
        complete_artifacts=True,
        output_dir=tmp_path,
        provenance={"choices": {"model": "fake"}, "overrides": ["model=fake"]},
    )
    index = json.loads((root / "artifact_index.json").read_text())
    assert index["status"] == "complete"
    assert len(index["artifacts"]) == (8 if ensemble else 4)
    assert model.calls[0] is None
    assert model.calls[1] is not None
    for artifact in index["artifacts"]:
        path = root / artifact["path"]
        assert forward._fingerprint(path) == artifact["sha256"]
        with xr.open_dataset(path) as ds:
            if artifact["kind"] == "params":
                expected = (
                    7
                    + (artifact["member"] or 0)
                    + (artifact["window"] if dynamic else 0)
                )
                assert np.all(ds.inflow_angle.values == expected)
    with xr.open_dataset(root / "state.nc") as ds:
        assert ds.sizes.get("ensemble", 1) == (2 if ensemble else 1)
        assert np.all(np.diff(ds.time.values) > 0)
    manifest = OmegaConf.load(root / "run_manifest.yaml")
    assert manifest.choices.model == "fake"
    assert manifest.cli_overrides == ["model=fake"]


def test_failure_preserves_intent_and_partial_outputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = _config(tmp_path)
    _install(monkeypatch, cfg, _Model(cfg, fail_on=2))
    with pytest.raises(RuntimeError, match="solver failed"):
        forward.run(cfg, complete_artifacts=True, output_dir=tmp_path)
    assert (tmp_path / "config.resolved.yaml").exists()
    index = json.loads((tmp_path / "artifact_index.json").read_text())
    assert index["status"] == "partial"
    assert len(index["artifacts"]) == 2
    assert (
        json.loads((tmp_path / "forward_status.json").read_text())["status"] == "failed"
    )


def test_constructor_failure_has_pre_run_record(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = _config(tmp_path)

    def fail(node: Any, **kwargs: Any) -> Any:
        assert (tmp_path / "run_manifest.yaml").is_file()
        raise RuntimeError("construction")

    monkeypatch.setattr(forward, "instantiate", fail)
    with pytest.raises(RuntimeError, match="construction"):
        forward.run(cfg, output_dir=tmp_path)


def test_initial_state_selection_history_and_members(tmp_path: Path) -> None:
    cfg = _config(tmp_path, ensemble=True)
    path = tmp_path / "initial.nc"
    _state(2).to_netcdf(path)
    model = SimpleNamespace(num_history_steps=2)
    state = forward.load_initial_state(path, cfg, model=model)
    assert state is not None and state.sizes["ensemble"] == 2
    np.testing.assert_equal(state.u.values, _state(2).u.values)
    with pytest.raises(ValueError, match="history frames"):
        forward.load_initial_state({"path": path, "time_index": 0}, cfg, model=model)
    cfg.run.ensemble = False
    with pytest.raises(ValueError, match="select a member"):
        forward.load_initial_state(path, cfg)
    chosen = forward.load_initial_state({"path": path, "member": 1}, cfg, model=model)
    assert chosen is not None and "ensemble" not in chosen.dims
    np.testing.assert_equal(chosen.u.values, _state(2).isel(ensemble=1).u.values)


def test_initial_state_validation(tmp_path: Path) -> None:
    cfg = _config(tmp_path)
    state = _state().isel(ensemble=0, drop=True)
    with pytest.raises(ValueError, match="missing variables"):
        forward.load_initial_state(state.drop_vars("w"), cfg)
    with pytest.raises(ValueError, match="strictly increasing"):
        forward.load_initial_state(state.assign_coords(time=[1.0, 1.0]), cfg)
    cfg.model.forward_model.spinup_source = "training_data"
    with pytest.raises(ValueError, match="no cold start"):
        forward.load_initial_state(None, cfg)


def test_cli_uses_shared_runner() -> None:
    from scripts.run_forward_model import run

    assert run is forward.run


def test_duplicate_disk_member_labels(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = _config(tmp_path, ensemble=True)

    class DiskModel(_Model):
        def __call__(
            self, *, params: xr.Dataset, state: xr.Dataset | None = None
        ) -> xr.Dataset:
            return (
                super()
                .__call__(params=params, state=state)
                .assign_coords(ensemble=[0, 0])
            )

    _install(monkeypatch, cfg, DiskModel(cfg))
    forward.run(cfg, complete_artifacts=True, output_dir=tmp_path)
    index = json.loads((tmp_path / "artifact_index.json").read_text())
    assert [
        entry["member"] for entry in index["artifacts"] if entry["kind"] == "state"
    ] == [0, 1, 0, 1]
    with xr.open_dataset(tmp_path / "state.nc") as ds:
        np.testing.assert_equal(ds.ensemble.values, [0, 1])


def test_streaming_flag_is_rejected(tmp_path: Path) -> None:
    cfg = _config(tmp_path)
    cfg.run.ensemble_save_on_disk = True
    with pytest.raises(ValueError, match="not implemented"):
        forward.run(cfg)


def test_plot_failure_keeps_successful_numerical_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from scripts import _common

    cfg = _config(tmp_path)
    cfg.run.skip_viz = False
    _install(monkeypatch, cfg, _Model(cfg))

    def fail(*args: Any, **kwargs: Any) -> None:
        raise RuntimeError("plot failed")

    monkeypatch.setattr(_common, "visualize_forward_state", fail)
    forward.run(cfg, complete_artifacts=True, output_dir=tmp_path)
    status = json.loads((tmp_path / "forward_status.json").read_text())
    assert status["status"] == "succeeded"
    assert status["visualization_error"] == "RuntimeError: plot failed"
    assert (tmp_path / "state.nc").is_file()


def test_actual_donor_parameters_are_persisted_without_jitter(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = _config(tmp_path, ensemble=True)
    model = _Model(cfg)
    model._last_failure_substitutions = {1: 0}  # type: ignore[attr-defined]
    _install(monkeypatch, cfg, model)
    forward.run(cfg, complete_artifacts=True, output_dir=tmp_path)
    with xr.open_dataset(tmp_path / "windows/0000/params_0001.nc") as params:
        assert float(params.inflow_angle.values[0]) == 7
    index = json.loads((tmp_path / "artifact_index.json").read_text())
    assert index["failure_substitutions"][0]["donors"] == {"1": 0}


def test_dataset_initial_state_is_saved_and_passed_to_solver(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = _config(tmp_path, ensemble=True)
    model = _Model(cfg)
    _install(monkeypatch, cfg, model)
    initial = _state(2)
    forward.run(
        cfg, complete_artifacts=True, initial_state=initial, output_dir=tmp_path
    )
    xr.testing.assert_equal(model.calls[0], initial)
    assert (tmp_path / "initial_state.nc").is_file()


def test_training_data_warm_start_does_not_construct_cfd(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = _config(tmp_path)
    cfg.model.name = "neural_surrogate"
    cfg.model.forward_model.spinup_source = "training_data"
    cfg.model.forward_model.spinup_forward_model = {"_target_": "unused.CFD"}
    model = _Model(cfg)
    seen: dict[str, Any] = {}

    def instantiate(node: DictConfig, **kwargs: Any) -> Any:
        if node.role == "params":
            return _Sampler(False)
        if node.role == "model":
            seen.update(kwargs)
        return model

    monkeypatch.setattr(forward, "instantiate", instantiate)
    monkeypatch.setattr(forward, "clean_outputs", lambda **kwargs: None)
    forward.run(
        cfg, initial_state=_state().isel(ensemble=0, drop=True), output_dir=tmp_path
    )
    assert seen["spinup_forward_model"] is None


def test_ensemble_constructor_paths_are_path_objects(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = _config(tmp_path, ensemble=True)
    cfg.model.ensemble_model.temp_dir = str(tmp_path / "scratch")
    cfg.model.ensemble_model.results_dir = str(tmp_path / "results")
    model = _Model(cfg)
    seen: dict[str, Any] = {}

    def instantiate(node: DictConfig, **kwargs: Any) -> Any:
        if node.role == "params":
            return _Sampler(False)
        if node.role == "ensemble":
            seen.update(kwargs)
        return model

    monkeypatch.setattr(forward, "instantiate", instantiate)
    monkeypatch.setattr(forward, "clean_outputs", lambda **kwargs: None)
    forward.run(cfg, output_dir=tmp_path)
    assert seen["temp_dir"] == tmp_path / "scratch"
    assert seen["results_dir"] == tmp_path / "results"


def test_generated_solver_inputs_preserve_each_window(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = _config(tmp_path)
    native = tmp_path / "scratch" / "infile.in"
    native.parent.mkdir()

    class NativeModel(_Model):
        dirs = SimpleNamespace(infile_path=native)

        def __call__(
            self, *, params: xr.Dataset, state: xr.Dataset | None = None
        ) -> xr.Dataset:
            result = super().__call__(params=params, state=state)
            native.write_text(f"window = {len(self.calls)}\n")
            return result

    _install(monkeypatch, cfg, NativeModel(cfg))
    forward.run(cfg, complete_artifacts=True, output_dir=tmp_path)
    index = json.loads((tmp_path / "artifact_index.json").read_text())
    snapshots = [item for item in index["artifacts"] if item["kind"] == "solver_input"]
    assert [item["window"] for item in snapshots] == [0, 1]
    assert [(tmp_path / item["path"]).read_text() for item in snapshots] == [
        "window = 1\n",
        "window = 2\n",
    ]
    assert snapshots[0]["sha256"] != snapshots[1]["sha256"]
    assert snapshots[0]["source_path"] == str(native)


def test_solver_input_snapshot_excludes_shared_paths(tmp_path: Path) -> None:
    shared = tmp_path / "shared" / "infile.in"
    shared.parent.mkdir()
    shared.write_text("shared input")
    owned = tmp_path / "run"
    owned.mkdir()
    model = SimpleNamespace(dirs=SimpleNamespace(infile_path=shared))
    index: dict[str, Any] = {"artifacts": []}
    forward._snapshot_solver_inputs(
        owned, index, 0, model, _Sampler(False).sample(1), False
    )
    assert index["artifacts"] == []
    assert index["solver_input_warnings"][0]["reason"] == "outside owned run root"


def _timed_values(times: list[float], values: list[float] | None = None) -> xr.Dataset:
    return xr.Dataset(
        {"value": ("time", times if values is None else values)}, coords={"time": times}
    )


def test_rollout_forecast_times_match_prescribed_parameters(tmp_path: Path) -> None:
    cfg = _config(tmp_path, dynamic=True)
    cfg.time.simulation_time = 3.0
    states = [
        _timed_values([1.0, 2.0, 3.0]),
        _timed_values([1.0, 2.0, 3.0], [4.0, 5.0, 6.0]),
    ]
    parameters = [
        _timed_values([0.0, 1.0, 2.0, 3.0]),
        _timed_values([0.0, 1.0, 2.0, 3.0], [3.0, 4.0, 5.0, 6.0]),
    ]
    combined = forward._concat_windows(states, cfg)
    forcing = forward._concat_windows(parameters, cfg).drop_duplicates("time")
    np.testing.assert_equal(combined.time.values, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    xr.testing.assert_equal(combined["value"], forcing["value"].sel(time=combined.time))
    index: dict[str, Any] = {"artifacts": []}
    for window in range(2):
        forward._persist_window(
            tmp_path, index, window, states[window], parameters[window], cfg
        )
        with xr.open_dataset(tmp_path / f"windows/{window:04d}/state_0000.nc") as saved:
            np.testing.assert_equal(
                saved.time.values, combined.time.values[window * 3 : (window + 1) * 3]
            )
            assert saved.attrs["time_reference"] == "global"


def test_boundary_including_windows_keep_the_shared_endpoint(tmp_path: Path) -> None:
    cfg = _config(tmp_path)
    cfg.time.simulation_time = 3.0
    first = _timed_values([0.0, 1.0, 2.0, 3.0])
    second = _timed_values([0.0, 1.0, 2.0, 3.0], [3.0, 4.0, 5.0, 6.0])
    combined = forward._concat_windows([first, second], cfg)
    np.testing.assert_equal(
        combined.time.values, [0.0, 1.0, 2.0, 3.0, 3.0, 4.0, 5.0, 6.0]
    )
    np.testing.assert_equal(combined["value"].values, combined.time.values)


def test_explicit_global_window_is_not_shifted_twice(tmp_path: Path) -> None:
    cfg = _config(tmp_path)
    cfg.time.simulation_time = 3.0
    first = _timed_values([1.0, 2.0, 3.0])
    second = _timed_values([4.0, 5.0, 6.0]).assign_attrs(time_reference="global")
    combined = forward._concat_windows([first, second], cfg)
    np.testing.assert_equal(combined.time.values, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    # A global endpoint exactly on a window boundary is unambiguous with metadata.
    endpoint = _timed_values([3.0]).assign_attrs(time_reference="global")
    assert forward._window_dataset(endpoint, 1, cfg).time.item() == 3.0


def test_initial_artifact_clock_tag_does_not_leak_to_forecasts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = _config(tmp_path)

    class CopyingModel(_Model):
        def __call__(
            self, *, params: xr.Dataset, state: xr.Dataset | None = None
        ) -> xr.Dataset:
            result = super().__call__(params=params, state=state)
            if state is not None:
                result.attrs.update(state.attrs)
            return result

    model = CopyingModel(cfg)
    _install(monkeypatch, cfg, model)
    initial = _state().isel(ensemble=0, drop=True).assign_attrs(time_reference="global")
    forward.run(
        cfg, complete_artifacts=True, initial_state=initial, output_dir=tmp_path
    )
    assert model.calls[0] is not None
    assert "time_reference" not in model.calls[0].attrs
    assert initial.attrs["time_reference"] == "global"
    with xr.open_dataset(tmp_path / "state.nc") as saved:
        np.testing.assert_equal(saved.time.values, [1.0, 2.0, 3.0, 4.0])


def test_singleton_ensemble_initial_state_gets_member_axis(tmp_path: Path) -> None:
    cfg = _config(tmp_path, ensemble=True)
    cfg.ensemble.ensemble_size = 1
    initial = _state().isel(ensemble=0, drop=True)
    selected = forward.load_initial_state(initial, cfg)
    assert selected is not None
    assert selected.sizes["ensemble"] == 1
    xr.testing.assert_equal(selected.isel(ensemble=0, drop=True), initial)


def _history_fixture(ensemble: bool) -> xr.Dataset:
    data = np.asarray([-2.0, -1.0, 0.0]).reshape(3, 1, 1, 1)
    initial = xr.Dataset(
        {v: (("time", "z", "y", "x"), data) for v in ("u", "v", "w")},
        coords={"time": [98.0, 99.0, 100.0], "x": [0.5], "y": [0.5], "z": [0.5]},
    )
    if ensemble:
        initial = xr.concat([initial, initial + 100.0], dim="ensemble").assign_coords(
            ensemble=[10, 20]
        )
    return initial


class _HistoryModel(_Model):
    num_history_steps = 3
    trained_output_frequency = 1.0

    def __call__(
        self, *, params: xr.Dataset, state: xr.Dataset | None = None
    ) -> xr.Dataset:
        self.calls.append(state)
        output = _history_fixture(bool(self.cfg.run.ensemble)).isel(time=[-1])
        for variable in output:
            output[variable] = output[variable] + len(self.calls)
        return output.assign_coords(time=[1.0])


@pytest.mark.parametrize("ensemble", [False, True])  # type: ignore[misc, unused-ignore]
def test_short_windows_keep_real_surrogate_history(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, ensemble: bool
) -> None:
    cfg = _config(tmp_path, ensemble=ensemble)
    cfg.time.simulation_time = 1.0
    cfg.run.rollout_steps = 3
    model = _HistoryModel(cfg)
    _install(monkeypatch, cfg, model)
    forward.run(
        cfg,
        complete_artifacts=True,
        initial_state=_history_fixture(ensemble),
        output_dir=tmp_path,
    )
    for window, supplied in enumerate(model.calls):
        assert supplied is not None and supplied.sizes["time"] == 3
        first = supplied.isel(ensemble=0) if ensemble else supplied
        np.testing.assert_equal(
            first.u.values.ravel(), np.arange(window - 2, window + 1)
        )
        if window:
            np.testing.assert_equal(
                supplied.time.values, np.arange(window - 2, window + 1)
            )
        if ensemble:
            np.testing.assert_equal(
                supplied.isel(ensemble=1).u.values.ravel(),
                np.arange(window - 2, window + 1) + 100,
            )
    with xr.open_dataset(tmp_path / "state.nc") as state:
        np.testing.assert_equal(state.time.values, [1.0, 2.0, 3.0, 4.0])


def test_short_cold_window_cannot_repeat_pad_missing_history(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = _config(tmp_path)
    cfg.time.simulation_time = 1.0
    model = _HistoryModel(cfg)
    _install(monkeypatch, cfg, model)
    with pytest.raises(ValueError, match="3 real history frames"):
        forward.run(cfg, complete_artifacts=True, output_dir=tmp_path)
    assert len(model.calls) == 1
    assert (
        json.loads((tmp_path / "artifact_index.json").read_text())["status"]
        == "partial"
    )


def test_carried_history_counts_toward_retained_memory_limit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = _config(tmp_path)
    cfg.time.simulation_time = 1.0
    initial = _history_fixture(False)
    forecast = initial.isel(time=[-1])
    cfg.run.max_retained_bytes = (
        initial.nbytes + forecast.nbytes + _Sampler(False).sample(1).nbytes - 1
    )
    model = _HistoryModel(cfg)
    _install(monkeypatch, cfg, model)
    with pytest.raises(MemoryError, match="max_retained_bytes"):
        forward.run(
            cfg, complete_artifacts=True, initial_state=initial, output_dir=tmp_path
        )
    assert len(model.calls) == 1
    assert (tmp_path / "windows/0000/state_0000.nc").exists()


def test_static_consolidated_params_retain_later_donor_changes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = _config(tmp_path, ensemble=True)

    class ChangedDonorModel(_Model):
        def __call__(
            self, *, params: xr.Dataset, state: xr.Dataset | None = None
        ) -> xr.Dataset:
            output = super().__call__(params=params, state=state)
            self._last_failure_substitutions = {1: 0} if len(self.calls) == 2 else {}
            return output

    _install(monkeypatch, cfg, ChangedDonorModel(cfg))
    forward.run(cfg, complete_artifacts=True, output_dir=tmp_path)
    with xr.open_dataset(tmp_path / "params.nc") as params:
        assert params.inflow_angle.dims == ("window", "ensemble")
        np.testing.assert_equal(params.window_start.values, [0.0, 2.0])
        np.testing.assert_equal(params.inflow_angle.sel(ensemble=1).values, [8.0, 7.0])
    index = json.loads((tmp_path / "artifact_index.json").read_text())
    assert index["consolidated_params_layout"] == "window"
