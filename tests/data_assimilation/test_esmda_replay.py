"""ESMDA must replay hidden solver state, while accepting analyzed inputs."""

import json
from pathlib import Path
from typing import Any, cast

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import xarray as xr
from data_assimilation.smoothing.esmda import ParameterESMDA

from pyurbanair.base_ensemble_forward_model import BaseEnsembleForwardModel
from pyurbanair.base_forward_model import BaseForwardModel


class HiddenStateEnsemble(BaseEnsembleForwardModel):
    """A forecast whose unobserved memory exposes accidental endpoint reuse."""

    def __init__(self, directory: Path | None = None, replay: bool = True) -> None:
        self._set_save_mode(directory)
        self.replay = replay
        self.hidden = np.array([2.0, 4.0, 6.0])
        self.clock = 10.0
        self.checkpoint: tuple[np.ndarray, float] | None = None
        self.starts: list[np.ndarray] = []
        self.inputs: list[np.ndarray] = []
        self.finishes: list[bool] = []
        self.fail_at: int | None = None
        self._last_failure_substitutions: dict[int, int] = {}
        self._failure_jitter_scale = 0.0

    def _create_new_forward_model(self, *args: Any, **kwargs: Any) -> BaseForwardModel:
        raise NotImplementedError("This test ensemble has no member models")

    def _pre_run_ensemble(self, *args: Any, **kwargs: Any) -> None:
        pass

    def _post_run_ensemble(self, sim_name: str) -> xr.Dataset | None:
        return None

    @property
    def forecast_window_replay_enabled(self) -> bool:
        return self.replay

    def begin_forecast_window(self) -> None:
        assert self.checkpoint is None
        self.checkpoint = (self.hidden.copy(), self.clock)

    def restore_forecast_window(self) -> None:
        assert self.checkpoint is not None
        self.hidden, self.clock = self.checkpoint[0].copy(), self.checkpoint[1]

    def end_forecast_window(self, commit: bool) -> None:
        self.finishes.append(commit)
        if not commit:
            self.restore_forecast_window()
        self.checkpoint = None

    def run_ensemble(
        self,
        state: xr.Dataset | Path | None = None,
        params: xr.Dataset | None = None,
        sim_name: str | None = "state",
    ) -> xr.Dataset | None:
        assert params is not None
        assert not isinstance(state, Path)
        self.starts.append(self.hidden.copy())
        initial = np.zeros(3) if state is None else np.asarray(state.u)[:, -1]
        self.inputs.append(initial.copy())
        endpoint = initial + self.hidden + params.a.values
        self.hidden += 7.0
        self.clock += 2.0
        if len(self.starts) == self.fail_at:
            raise RuntimeError("forecast failed after changing hidden state")
        result = xr.Dataset(
            {"u": (("ensemble", "time"), np.stack([initial, endpoint], axis=1))},
            coords={"ensemble": [0, 1, 2], "time": [0.0, 2.0]},
        )
        if self.save_on_disk:
            assert self.results_dir is not None
            for i in range(3):
                result.isel(ensemble=i, drop=True).to_netcdf(
                    self.results_dir / f"state_{i}.nc"
                )
            return None
        return result


def observe(state: xr.Dataset) -> np.ndarray:
    return np.asarray(state.u.isel(time=[-1]))


def make_smoother(
    model: HiddenStateEnsemble,
    parameter_names_to_estimate: tuple[str, ...] | None = None,
) -> ParameterESMDA:
    smoother = ParameterESMDA(
        observation_operator=cast(Any, observe),
        forward_model=model,
        C_D=jnp.array([1.0]),
        num_steps=2,
        rng_key=jax.random.PRNGKey(7),
        parameter_names_to_estimate=parameter_names_to_estimate,
    )
    smoother.collect_obs_diagnostics = True
    return smoother


def prior() -> xr.Dataset:
    return xr.Dataset({"a": ("ensemble", [-0.5, 0.0, 0.5])})


@pytest.mark.parametrize("disk", [False, True])  # type: ignore[misc]
@pytest.mark.parametrize("final", [False, True])  # type: ignore[misc]
def test_window_replays_and_commits_only_the_posterior(
    tmp_path: Path, disk: bool, final: bool
) -> None:
    model = HiddenStateEnsemble(tmp_path / "results" if disk else None)
    smoother = make_smoother(model)
    smoother(params=prior(), observations=np.array([4.0]), final_forecast=final)
    assert len(model.starts) == 2 + int(final)
    for hidden in model.starts:
        np.testing.assert_array_equal(hidden, [2.0, 4.0, 6.0])
    np.testing.assert_array_equal(
        model.hidden, [9.0, 11.0, 13.0] if final else [2, 4, 6]
    )
    assert model.clock == (12.0 if final else 10.0)
    assert model.finishes == [final]
    assert model.checkpoint is None
    assert len(smoother.pred_obs_history) == 2 + int(final)


@pytest.mark.parametrize("failure", ["forecast", "analysis", "diagnostics"])  # type: ignore[misc]
def test_failed_window_rolls_back(
    failure: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    model = HiddenStateEnsemble()
    smoother = make_smoother(model)

    def fail(*args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("injected failure")

    if failure == "forecast":
        model.fail_at = 2
    elif failure == "analysis":
        monkeypatch.setattr(smoother, "_one_step", fail)
    else:
        monkeypatch.setattr(smoother, "_final_time_smoothing_step", fail)
    with pytest.raises(RuntimeError):
        smoother(params=prior(), observations=np.array([4.0]))
    np.testing.assert_array_equal(model.hidden, [2, 4, 6])
    assert model.clock == 10.0
    assert model.finishes == [False]
    assert model.checkpoint is None


def test_analyzed_initial_state_is_injected_after_hidden_state_restore(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = HiddenStateEnsemble()
    smoother = make_smoother(model)

    def update(params: xr.Dataset, obs: Any, state: xr.Dataset) -> Any:
        return state.isel(time=[0]) + 1.0, params

    monkeypatch.setattr(smoother, "_one_step", update)
    smoother(params=prior(), observations=np.array([4.0]))
    np.testing.assert_array_equal(model.inputs, [[0, 0, 0], [1, 1, 1], [2, 2, 2]])
    np.testing.assert_array_equal(model.starts, [[2, 4, 6]] * 3)


def test_next_window_starts_from_accepted_endpoint() -> None:
    model = HiddenStateEnsemble()
    smoother = make_smoother(model)
    params, state = smoother(params=prior(), observations=np.array([4.0]))
    smoother(params=params, state=state.isel(time=[-1]), observations=np.array([5.0]))
    np.testing.assert_array_equal(model.starts[3:], [[9, 11, 13]] * 3)
    assert model.clock == 14.0
    assert model.finishes == [True, True]


def test_disabled_backend_retains_legacy_hidden_state_behavior() -> None:
    model = HiddenStateEnsemble(replay=False)
    make_smoother(model)(params=prior(), observations=np.array([4.0]))
    np.testing.assert_array_equal(model.starts, [[2, 4, 6], [9, 11, 13], [16, 18, 20]])
    assert model.finishes == []


def test_disk_and_memory_replays_have_identical_updates(tmp_path: Path) -> None:
    memory = make_smoother(HiddenStateEnsemble())
    disk = make_smoother(HiddenStateEnsemble(tmp_path / "results"))
    mem_params, _ = memory(params=prior(), observations=np.array([4.0]))
    disk_params = disk(params=prior(), observations=np.array([4.0]))
    xr.testing.assert_identical(mem_params, disk_params)
    np.testing.assert_array_equal(memory.pred_obs_history, disk.pred_obs_history)
    np.testing.assert_array_equal(memory.rng_key, disk.rng_key)


@pytest.mark.parametrize("selection", [("a",), ()])  # type: ignore[misc]
def test_disk_replay_applies_unestimated_parameters(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, selection: tuple[str, ...]
) -> None:
    params = prior()
    params["sgs_bias_b0"] = ("ensemble", [-20.123456789, -19.123456789, -18.123456789])
    params.sgs_bias_b0.attrs["units"] = "dimensionless"
    smoothers = []
    results = []
    for directory in (None, tmp_path / "results"):
        model = HiddenStateEnsemble(directory)
        forecast = model.run_ensemble
        applied: list[xr.Dataset] = []

        def capture(
            *, _forecast: Any = forecast, _applied: Any = applied, **kwargs: Any
        ) -> Any:
            _applied.append(kwargs["params"].copy(deep=True))
            return _forecast(**kwargs)

        monkeypatch.setattr(model, "run_ensemble", capture)
        smoother = make_smoother(model, parameter_names_to_estimate=selection)
        result = smoother(params=params, observations=np.array([4.0]))
        posterior = result if model.save_on_disk else result[0]
        for forecast_params in applied:
            xr.testing.assert_identical(forecast_params.sgs_bias_b0, params.sgs_bias_b0)
        xr.testing.assert_identical(posterior.sgs_bias_b0, params.sgs_bias_b0)
        if not selection:
            xr.testing.assert_identical(posterior, params)
        else:
            assert not np.array_equal(posterior.a, params.a)
        smoothers.append(smoother)
        results.append(posterior)
    xr.testing.assert_identical(results[0], results[1])
    np.testing.assert_array_equal(
        smoothers[0].pred_obs_history, smoothers[1].pred_obs_history
    )
    np.testing.assert_array_equal(smoothers[0].rng_key, smoothers[1].rng_key)


def test_commit_validation_failure_rolls_back(monkeypatch: pytest.MonkeyPatch) -> None:
    model = HiddenStateEnsemble()
    finish = model.end_forecast_window

    def reject_commit(commit: bool) -> None:
        if commit:
            raise ValueError("incomplete native endpoint")
        finish(commit=False)

    monkeypatch.setattr(model, "end_forecast_window", reject_commit)
    with pytest.raises(ValueError, match="incomplete native endpoint"):
        make_smoother(model)(params=prior(), observations=np.array([4.0]))
    np.testing.assert_array_equal(model.hidden, [2, 4, 6])
    assert model.clock == 10.0
    assert model.checkpoint is None


@pytest.mark.parametrize("history", [False, True])  # type: ignore[misc]
def test_posterior_donor_coefficients_match_returned_forecast(
    history: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    model = HiddenStateEnsemble()
    forecast = model.run_ensemble

    def final_donor(**kwargs: Any) -> xr.Dataset | None:
        result = forecast(**kwargs)
        if len(model.starts) == 3:
            model._last_failure_substitutions = {0: 2}
            assert result is not None
            result["u"][0] = result["u"][2]
        return result

    monkeypatch.setattr(model, "run_ensemble", final_donor)
    params, state = make_smoother(model)(
        params=prior(), observations=np.array([4.0]), return_params_history=history
    )
    if history:
        params = params.isel(esmda_step=-1)
    assert params.a.values[0] == params.a.values[2]
    np.testing.assert_array_equal(state.u.values[0], state.u.values[2])


def test_state_history_keeps_each_steps_own_attrs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Per-forecast provenance (pyudales' per-member discrepancy coefficients)
    must describe the step's own members, not the prior's."""
    model = HiddenStateEnsemble()
    forecast = model.run_ensemble

    def with_provenance(**kwargs: Any) -> xr.Dataset | None:
        result = forecast(**kwargs)
        assert result is not None
        result.attrs["coefficients"] = json.dumps(kwargs["params"].a.values.tolist())
        result.attrs["solver"] = "fake"
        return result

    monkeypatch.setattr(model, "run_ensemble", with_provenance)
    smoother = make_smoother(model)
    params, states = smoother(
        params=prior(),
        observations=np.array([4.0]),
        return_params_history=True,
        return_state_history=True,
    )
    # The stacked history keeps only what every step shares.
    assert states.attrs == {"solver": "fake"}
    steps = smoother.state_history_attrs
    assert len(steps) == states.sizes["esmda_step"] == 3
    for step, attrs in zip([0, -1], [steps[0], steps[-1]]):
        expected = params.isel(esmda_step=step).a.values.tolist()
        assert json.loads(attrs["coefficients"]) == expected
    assert steps[0]["coefficients"] != steps[-1]["coefficients"]
