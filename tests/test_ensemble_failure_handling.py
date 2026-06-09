"""Unit tests for ensemble member failure handling."""

import pathlib
import subprocess
from typing import Any, Optional

import numpy as np
import pytest
import xarray

from pyurbanair.base_ensemble_forward_model import BaseEnsembleForwardModel
from pyurbanair.base_forward_model import BaseForwardModel


class _StubForwardModel:
    """Minimal stand-in used as the ``forward_model`` template.

    The base ensemble class only reads ``results_dir`` and the absence of
    ``rollout_step`` / ``dirs`` attributes from the template, so this tiny
    stub is enough.
    """

    results_dir: Optional[pathlib.Path] = None


class _MockMember:
    """One ensemble member: returns a Dataset, or raises if its index is failing."""

    def __init__(self, index: int, fail_indices: set[int]) -> None:
        self.index = index
        self.fail_indices = fail_indices

    def __call__(
        self,
        state: Optional[xarray.Dataset] = None,
        params: Optional[xarray.Dataset] = None,
        sim_name: Optional[str] = None,
    ) -> xarray.Dataset:
        if self.index in self.fail_indices:
            raise subprocess.CalledProcessError(
                returncode=132,
                cmd=["mock", f"member_{self.index}"],
            )
        return xarray.Dataset(
            data_vars={"value": ("x", np.full(3, float(self.index)))},
            coords={"x": np.arange(3)},
        )


class _MockEnsemble(BaseEnsembleForwardModel):
    """Concrete ensemble whose members never touch the disk."""

    def __init__(self, ensemble_size: int, fail_indices: set[int]) -> None:
        self._fail_indices = fail_indices
        super().__init__(
            forward_model=_StubForwardModel(),  # type: ignore[arg-type]
            ensemble_size=ensemble_size,
            temp_dir=pathlib.Path("/tmp/_mock_ensemble_test"),
        )
        # Replace the auto-created list with mock members.
        self.ensemble_forward_models = [
            _MockMember(i, fail_indices) for i in range(ensemble_size)  # type: ignore[misc]
        ]

    def _create_new_forward_model(
        self,
        forward_model: BaseForwardModel,
        experiment_base_dir: pathlib.Path,
        experiment_name: str,
    ) -> Any:
        # Auto-create called by base __init__; we overwrite below anyway.
        return _MockMember(int(experiment_name), self._fail_indices)

    def _pre_run_ensemble(self, *_args: Any, **_kwargs: Any) -> None:
        return None

    def _post_run_ensemble(self, *_args: Any, **_kwargs: Any) -> None:
        return None


def _make_params(n: int) -> xarray.Dataset:
    rng = np.random.default_rng(0)
    return xarray.Dataset(
        data_vars={
            "alpha": ("ensemble", rng.normal(0.0, 1.0, n)),
            "beta": ("ensemble", rng.normal(5.0, 0.5, n)),
        },
        coords={"ensemble": np.arange(n)},
    )


def test_resample_substitutes_failed_states_with_donor_states() -> None:
    n = 16
    failed = {3, 7}
    ensemble = _MockEnsemble(n, fail_indices=failed)
    ensemble.configure_failure_policy(
        policy="resample_from_successes", jitter_scale=0.0, seed=123
    )

    states = ensemble._run_ensemble_sequentially_in_memory(
        params=_make_params(n), sim_name="state"
    )

    assert states.sizes["ensemble"] == n
    survivors = set(range(n)) - failed
    for j in failed:
        # Failed slot's value field equals the donor's index-valued vector.
        donor_value = float(states["value"].isel(ensemble=j).values[0])
        assert int(donor_value) in survivors, (
            f"failed member {j} should be cloned from a survivor, got {donor_value}"
        )

    assert set(ensemble._last_failure_substitutions.keys()) == failed
    assert all(d in survivors for d in ensemble._last_failure_substitutions.values())


def test_apply_substitutions_clones_params_with_jitter() -> None:
    n = 16
    failed = {2, 11}
    ensemble = _MockEnsemble(n, fail_indices=failed)
    ensemble.configure_failure_policy(
        policy="resample_from_successes", jitter_scale=0.05, seed=42
    )

    params = _make_params(n)
    _ = ensemble._run_ensemble_sequentially_in_memory(params=params, sim_name="state")

    new_params = ensemble.apply_failure_substitutions_to_params(params)

    # Original params unchanged (no in-place mutation).
    assert np.array_equal(params["alpha"].values, _make_params(n)["alpha"].values)

    subs = ensemble._last_failure_substitutions
    for j, donor in subs.items():
        for var in ("alpha", "beta"):
            donor_val = params[var].isel(ensemble=donor).item()
            new_val = new_params[var].isel(ensemble=j).item()
            std = float(params[var].std(dim="ensemble").values)
            # Jitter is small relative to the empirical std.
            assert abs(new_val - donor_val) < 0.5 * std + 1e-9
            # And not exactly zero — jitter actually applied.
            assert new_val != donor_val


def test_zero_jitter_yields_exact_clone() -> None:
    n = 8
    failed = {5}
    ensemble = _MockEnsemble(n, fail_indices=failed)
    ensemble.configure_failure_policy(
        policy="resample_from_successes", jitter_scale=0.0, seed=1
    )

    params = _make_params(n)
    _ = ensemble._run_ensemble_sequentially_in_memory(params=params, sim_name="state")
    new_params = ensemble.apply_failure_substitutions_to_params(params)

    donor = ensemble._last_failure_substitutions[5]
    for var in ("alpha", "beta"):
        assert (
            new_params[var].isel(ensemble=5).item()
            == params[var].isel(ensemble=donor).item()
        )


def _make_state(n: int) -> xarray.Dataset:
    """Per-member warm-start state: member i's field is the constant i.

    Includes a variable without an ``ensemble`` dim to check it passes through.
    """
    return xarray.Dataset(
        data_vars={
            "u": (("ensemble", "x"), np.repeat(np.arange(n)[:, None], 3, axis=1).astype(float)),
            "topo": ("x", np.array([1.0, 2.0, 3.0])),
        },
        coords={"ensemble": np.arange(n), "x": np.arange(3)},
    )


def test_apply_substitutions_clones_state_from_donor() -> None:
    n = 16
    failed = {3, 7}
    ensemble = _MockEnsemble(n, fail_indices=failed)
    ensemble.configure_failure_policy(
        policy="resample_from_successes", jitter_scale=0.0, seed=123
    )

    _ = ensemble._run_ensemble_sequentially_in_memory(
        params=_make_params(n), sim_name="state"
    )

    state = _make_state(n)
    new_state = ensemble.apply_failure_substitutions_to_state(state)

    # Original not mutated.
    assert np.array_equal(state["u"].values, _make_state(n)["u"].values)

    subs = ensemble._last_failure_substitutions
    for j, donor in subs.items():
        # Failed slot now holds an exact clone of the donor's field (no jitter).
        assert np.array_equal(
            new_state["u"].isel(ensemble=j).values,
            state["u"].isel(ensemble=donor).values,
        )
    # Variable without an ``ensemble`` dim is passed through untouched.
    assert np.array_equal(new_state["topo"].values, state["topo"].values)


def test_apply_state_substitutions_noops_on_none_and_no_failures() -> None:
    n = 6
    ensemble = _MockEnsemble(n, fail_indices=set())
    ensemble.configure_failure_policy(
        policy="resample_from_successes", jitter_scale=0.0, seed=7
    )
    _ = ensemble._run_ensemble_sequentially_in_memory(
        params=_make_params(n), sim_name="state"
    )

    # Cold start: None passes straight through.
    assert ensemble.apply_failure_substitutions_to_state(None) is None
    # No failures: state returned unchanged.
    state = _make_state(n)
    new_state = ensemble.apply_failure_substitutions_to_state(state)
    assert np.array_equal(new_state["u"].values, state["u"].values)


def test_raise_policy_propagates_called_process_error() -> None:
    ensemble = _MockEnsemble(8, fail_indices={4})
    # Default policy is "raise"; configure_failure_policy not called.
    with pytest.raises(subprocess.CalledProcessError):
        ensemble._run_ensemble_sequentially_in_memory(
            params=_make_params(8), sim_name="state"
        )


def test_all_members_failing_raises() -> None:
    n = 4
    ensemble = _MockEnsemble(n, fail_indices=set(range(n)))
    ensemble.configure_failure_policy(
        policy="resample_from_successes", jitter_scale=0.0, seed=0
    )
    with pytest.raises(RuntimeError, match="All ensemble members failed"):
        ensemble._run_ensemble_sequentially_in_memory(
            params=_make_params(n), sim_name="state"
        )


def test_apply_substitutions_handles_read_only_arrays() -> None:
    """Regression: JAX-backed Dataset values are read-only even after deep
    copy, which used to break in-place mutation."""
    n = 8
    failed = {6}
    ensemble = _MockEnsemble(n, fail_indices=failed)
    ensemble.configure_failure_policy(
        policy="resample_from_successes", jitter_scale=0.05, seed=3
    )

    base = _make_params(n)
    # Mark the underlying arrays read-only to mimic JAX-backed buffers.
    ro_data_vars = {}
    for name, da in base.data_vars.items():
        arr = np.array(da.values, copy=True)
        arr.setflags(write=False)
        ro_data_vars[name] = (da.dims, arr)
    params = xarray.Dataset(data_vars=ro_data_vars, coords=base.coords)

    _ = ensemble._run_ensemble_sequentially_in_memory(params=params, sim_name="state")
    new_params = ensemble.apply_failure_substitutions_to_params(params)

    # Substitution actually happened.
    donor = ensemble._last_failure_substitutions[6]
    for var in ("alpha", "beta"):
        donor_val = params[var].isel(ensemble=donor).item()
        new_val = new_params[var].isel(ensemble=6).item()
        assert new_val != donor_val  # jitter applied
        assert new_val != params[var].isel(ensemble=6).item()  # slot changed


def test_no_failures_leaves_substitutions_empty_and_params_untouched() -> None:
    n = 6
    ensemble = _MockEnsemble(n, fail_indices=set())
    ensemble.configure_failure_policy(
        policy="resample_from_successes", jitter_scale=0.05, seed=7
    )

    params = _make_params(n)
    states = ensemble._run_ensemble_sequentially_in_memory(
        params=params, sim_name="state"
    )

    assert ensemble._last_failure_substitutions == {}
    assert states.sizes["ensemble"] == n
    new_params = ensemble.apply_failure_substitutions_to_params(params)
    # No-op when nothing failed.
    assert np.array_equal(new_params["alpha"].values, params["alpha"].values)
    assert np.array_equal(new_params["beta"].values, params["beta"].values)
