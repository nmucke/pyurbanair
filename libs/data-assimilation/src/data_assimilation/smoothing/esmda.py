import os
import pathlib
import re
from abc import abstractmethod
from typing import Optional

import jax
import jax.numpy as jnp
import numpy as np
import xarray


def _group_ids_by_base_name(names: list[str]) -> jnp.ndarray:
    """Block id per row, grouping names that share a base (``_<int>`` stripped).

    Co-locates all time knots of one parameter (e.g. ``inflow_angle_0``,
    ``inflow_angle_1``, … -> one block) while leaving distinct parameters in
    separate blocks.  Static parameters (no numeric suffix) each form their
    own block, so block grouping then reduces to the per-row analysis.
    """
    base_names = [re.sub(r"_\d+$", "", name) for name in names]
    order: dict[str, int] = {}
    ids = []
    for base in base_names:
        ids.append(order.setdefault(base, len(order)))
    return jnp.asarray(ids, dtype=int)


def _block_grouping_enabled(localization) -> bool:
    """True when a localization is set and requests joint block updates."""
    return localization is not None and getattr(localization, "block_grouping", False)
from data_assimilation.localization.base import BaseLocalization
from data_assimilation.observation_operator import ObservationOperator
from data_assimilation.smoothing.base import BaseSmoothing
from pyurbanair.base_ensemble_forward_model import BaseEnsembleForwardModel


class _BaseESMDA(BaseSmoothing):
    """Shared ESMDA logic for parameter-only and joint state-parameter variants."""

    def __init__(
        self,
        observation_operator: ObservationOperator,
        forward_model: BaseEnsembleForwardModel,
        C_D: jnp.ndarray,
        num_steps: int = 3,
        alpha: Optional[float] = None,
        rng_key: Optional[jax.random.PRNGKey] = jax.random.PRNGKey(42),
        localization: Optional[BaseLocalization] = None,
    ) -> None:
        super().__init__(observation_operator, forward_model)

        self.alpha = num_steps if alpha is None else alpha
        self.C_D = C_D
        self.C_D_sqrt = jnp.sqrt(self.C_D)
        self.rng_key = rng_key
        self.num_steps = num_steps
        # Optional localization strategy. When None, the global (unlocalized)
        # Kalman update is used. When set, each augmented-state row is updated
        # via a local analysis driven by the strategy's inflation factors.
        self.localization = localization

        if self.forward_model.save_on_disk:
            self.base_results_dir = self.forward_model.results_dir
            for i in range(num_steps + 1):
                step_dir = self.base_results_dir / f"step_{i}"
                os.makedirs(step_dir, exist_ok=True)
                for state_file in step_dir.glob("state_*.nc"):
                    state_file.unlink(missing_ok=True)

    def _set_step_results_dir(self, step: int) -> None:
        """Point the forward model's results directory at the given step."""
        if self.forward_model.save_on_disk:
            self.forward_model.set_results_dir(self.base_results_dir / f"step_{step}")

    def get_state(self, ensemble_member: int, step: int) -> xarray.Dataset:
        """Get the state for one ensemble member at a given step."""
        return xarray.open_dataset(
            self.base_results_dir / f"step_{step}" / f"state_{ensemble_member}.nc"
        )

    def _compute_kalman_update(
        self,
        augmented: jnp.ndarray,
        pred_obs: jnp.ndarray,
        obs: jnp.ndarray,
        N_e: int,
        group_ids: Optional[jnp.ndarray] = None,
        localize_mask: Optional[jnp.ndarray] = None,
        row_coords: Optional[jnp.ndarray] = None,
        obs_coords: Optional[jnp.ndarray] = None,
    ) -> jnp.ndarray:
        """Compute the ESMDA Kalman update for an augmented state vector.

        Args:
            augmented: Array of shape (N_aug, N_e) — the augmented state
                (parameters only, or state + parameters).
            pred_obs: Array of shape (N_d, N_e) — predicted observations.
            obs: Array of shape (N_d,) — true observations.
            N_e: Ensemble size.
            group_ids: Optional block id per augmented row, shape (N_aug,),
                used only by a localized update with block grouping enabled.
                Rows sharing an id are updated jointly. ``None`` -> per-row.
            localize_mask: Optional boolean array, shape (N_aug,), forwarded to
                a localized update. Rows where ``False`` get the exact global
                update; ``True`` rows get the localized update. Ignored on the
                global (``localization is None``) path.
            row_coords: Optional augmented-row coordinates, shape (N_aug, 3),
                and ``obs_coords``: observation coordinates, shape (N_d, 3),
                forwarded to a distance-based localized update. Ignored on the
                global path and by coordinate-free strategies.

        Returns:
            Updated augmented array of the same shape.
        """
        N_d = obs.shape[0]

        aug_mean = jnp.mean(augmented, axis=1, keepdims=True)
        pred_obs_mean = jnp.mean(pred_obs, axis=1, keepdims=True)

        aug_dev = augmented - aug_mean
        pred_obs_dev = pred_obs - pred_obs_mean

        # Localized (local-analysis) update: each augmented row is updated
        # with only the observations the localization strategy deems relevant.
        if self.localization is not None:
            self.rng_key, subkey = jax.random.split(self.rng_key)
            return self.localization.localized_update(
                augmented=augmented,
                aug_dev=aug_dev,
                pred_obs=pred_obs,
                pred_obs_dev=pred_obs_dev,
                obs=obs,
                C_D=self.C_D,
                C_D_sqrt=self.C_D_sqrt,
                alpha=self.alpha,
                rng_key=subkey,
                group_ids=group_ids,
                localize_mask=localize_mask,
                row_coords=row_coords,
                obs_coords=obs_coords,
            )

        C_MD = jnp.dot(aug_dev, pred_obs_dev.T) / (N_e - 1)
        C_DD = jnp.dot(pred_obs_dev, pred_obs_dev.T) / (N_e - 1)

        self.rng_key, subkey = jax.random.split(self.rng_key)
        Z = jax.random.normal(subkey, (N_d, N_e))
        perturbed_obs = obs[:, None] + jnp.sqrt(self.alpha) * (self.C_D_sqrt @ Z)

        innovation = perturbed_obs - pred_obs
        C_DD_alpha = C_DD + self.alpha * self.C_D

        try:
            x = jnp.linalg.solve(C_DD_alpha, innovation)
        except jnp.linalg.LinAlgError:
            x = jnp.linalg.lstsq(C_DD_alpha, innovation, rcond=None)[0]

        return augmented + C_MD @ x

    def _observation_coords(self, n_d: int) -> jnp.ndarray:
        """Physical (x, y, z) coordinate of each of the ``n_d`` observations.

        Every observation is a sensor reading; its spatial location is the
        sensor's, independent of which state component (u/v/w) or time interval
        it belongs to.  In the flattened observation vector the sensor index is
        the innermost/fastest axis (see ``ObservationOperator`` and
        ``TemporalObservationOperator``), so observation ``j`` sits at sensor
        ``j % num_sensors``.  Tiling the sensor coordinates therefore reproduces
        the per-observation coordinates regardless of the number of state
        components or temporal intervals.

        Requires coordinate-based observations (``obs_x``/``obs_y``/``obs_z``).
        """
        base_op = getattr(
            self.observation_operator, "observation_operator", self.observation_operator
        )
        if not getattr(base_op, "use_interpolation", False):
            raise ValueError(
                "Distance-based localization requires coordinate-based "
                "observations (obs_x/obs_y/obs_z), not index-based ones."
            )
        sensor_xyz = np.stack(
            [base_op.obs_x, base_op.obs_y, base_op.obs_z], axis=1
        )  # (num_sensors, 3)
        num_sensors = base_op.num_sensors
        reps, remainder = divmod(int(n_d), int(num_sensors))
        if remainder != 0:
            raise ValueError(
                f"Observation count {n_d} is not a multiple of the sensor count "
                f"{num_sensors}; cannot map observations to sensor coordinates."
            )
        return jnp.asarray(np.tile(sensor_xyz, (reps, 1)))  # (n_d, 3)

    @abstractmethod
    def _one_step(
        self,
        params: xarray.Dataset,
        obs: jnp.ndarray,
        state: Optional[xarray.Dataset] = None,
    ) -> tuple[Optional[xarray.Dataset], xarray.Dataset]:
        """Perform one ESMDA assimilation step.

        Returns:
            (updated_state_or_None, updated_params)
        """
        raise NotImplementedError

    def _analysis(
        self,
        params: xarray.Dataset,
        observations: jnp.ndarray,
        state: Optional[xarray.Dataset] = None,
        return_params_history: bool = False,
        return_state_history: bool = False,
    ) -> xarray.Dataset | tuple[xarray.Dataset, xarray.Dataset]:
        """Perform the ESMDA analysis loop.

        Iterated joint estimation: the forecast at iteration ``i`` starts from
        the *current* initial-condition estimate. For the state-bearing variants
        (``StateAndParameterESMDA`` / ``StateAndTimeVaryingParameterESMDA``) the
        Kalman-updated initial condition from ``_one_step`` is fed forward, so it
        is actually used by the next forecast (and hence the next Kalman update)
        and propagates into the posterior forecast and the next window. For the
        parameter-only variants ``_one_step`` returns no state (``None``), so
        ``initial_state`` keeps the caller's pinned value and the loop reduces to
        the original parameter-only behavior.

        Note: feeding the analyzed IC forward warm-starts the next forecast from
        it (skipping a fresh spin-up after iteration 0). This is intentional — it
        is what makes the state estimate matter — but it means the analyzed field
        is integrated directly, so it must be a usable warm-start state for the
        forward model (as the cross-window carry-over already assumes).
        """
        if return_state_history and self.forward_model.save_on_disk:
            raise ValueError(
                "return_state_history is not supported in on-disk save mode: "
                "the per-step states live in the step_{i}/ directories "
                "(see get_state). Use an in-memory forward model "
                "(results_dir=None) to collect the state history."
            )

        initial_state = state

        params_history: list[xarray.Dataset] = [params] if return_params_history else []
        state_history: list[xarray.Dataset] = []

        for i in range(self.num_steps):
            self._set_step_results_dir(i)

            state = self._forecast_step(state=initial_state, params=params)
            params = self.forward_model.apply_failure_substitutions_to_params(params)

            if return_state_history:
                state_history.append(state)

            updated_state, params = self._one_step(
                params=params,
                obs=observations,
                state=state,
            )

            # Feed the Kalman-updated initial condition forward (state-bearing
            # variants only; param-only variants return ``None`` and keep the
            # caller's pinned IC).
            if updated_state is not None:
                initial_state = updated_state
            # Repair any diverged members in the IC used by the next forecast
            # (clones a donor's known-good field into each failed slot). No-op on
            # a cold start (``initial_state is None``) or when nothing failed.
            initial_state = self.forward_model.apply_failure_substitutions_to_state(
                initial_state
            )

            if return_params_history:
                params_history.append(params)

            print(f"ESMDA step {i} completed")

        # Final forecast from the analyzed initial condition + updated params.
        self._set_step_results_dir(self.num_steps)
        state = self._forecast_step(state=initial_state, params=params)

        if return_state_history:
            state_history.append(state)

        # Build return values
        result_params = (
            xarray.concat(params_history, dim="esmda_step", join="override")
            if return_params_history
            else params
        )

        if self.forward_model.save_on_disk:
            return result_params

        if return_state_history:
            result_state = xarray.concat(
                state_history, dim="esmda_step", join="override"
            )
            return result_params, result_state

        return result_params, state


class ParameterESMDA(_BaseESMDA):
    """Parameter-only ESMDA smoothing."""

    def _one_step(
        self,
        params: xarray.Dataset,
        obs: jnp.ndarray,
        state: Optional[xarray.Dataset] = None,
    ) -> tuple[Optional[xarray.Dataset], xarray.Dataset]:
        obs = jnp.asarray(obs)
        param_names = list(params.data_vars.keys())
        N_e = params.sizes["ensemble"]

        pred_obs = self._observation_step(
            state=state,
            results_dir=(
                self.forward_model.results_dir
                if self.forward_model.save_on_disk
                else None
            ),
        )
        pred_obs = jnp.asarray(pred_obs).T

        if pred_obs.ndim != 2 or pred_obs.shape[1] != N_e:
            raise ValueError(
                "Predicted observations shape does not match ensemble size. "
                f"Expected second dimension {N_e}, got {pred_obs.shape}. "
                "This usually indicates stale or unexpected files in the ESMDA "
                "results step directory."
            )

        params_array = jnp.array([params[name].values for name in param_names])

        # Block grouping: co-locate time knots of the same parameter so they
        # share one observation selection and transition (paper sec. 3b).
        group_ids = (
            _group_ids_by_base_name(param_names)
            if _block_grouping_enabled(self.localization)
            else None
        )
        params_updated = self._compute_kalman_update(
            params_array, pred_obs, obs, N_e, group_ids=group_ids
        )

        updated_data_vars = {
            name: ("ensemble", params_updated[i, :])
            for i, name in enumerate(param_names)
        }

        return None, xarray.Dataset(data_vars=updated_data_vars, coords=params.coords)


class TimeVaryingParameterESMDA(ParameterESMDA):
    """Parameter-only ESMDA for time-varying inflow parameters.

    Each time point of each parameter is treated as an independent scalar
    parameter during the Kalman update.  The flattening groups all time
    points of one parameter together (e.g. ``inflow_angle_0``, …,
    ``inflow_angle_{N_t-1}``, ``velocity_magnitude_0``, …) so that
    different physical quantities are never mixed.
    """

    def __init__(
        self,
        observation_operator: ObservationOperator,
        forward_model: BaseEnsembleForwardModel,
        C_D: jnp.ndarray,
        num_time_points: int,
        num_steps: int = 3,
        alpha: Optional[float] = None,
        rng_key: Optional[jax.random.PRNGKey] = jax.random.PRNGKey(42),
        pin_initial_time_point: bool = False,
        localization: Optional[BaseLocalization] = None,
    ) -> None:
        super().__init__(
            observation_operator=observation_operator,
            forward_model=forward_model,
            C_D=C_D,
            num_steps=num_steps,
            alpha=alpha,
            rng_key=rng_key,
            localization=localization,
        )
        self.num_time_points = num_time_points
        # When True, ``t=0`` of every time-varying parameter is excluded
        # from the Kalman-updated augmented state and reinserted unchanged
        # during unflatten — per ensemble member. Useful for preserving
        # cross-window continuity in rollout ESMDA.
        self.pin_initial_time_point = pin_initial_time_point

    def _flatten_time_varying_params(self, params: xarray.Dataset) -> xarray.Dataset:
        """Flatten ``(time, ensemble)`` params to scalar ``(ensemble,)`` vars.

        Each variable with a ``time`` dimension is expanded into
        ``{name}_0``, ``{name}_1``, … so that all time points of one
        parameter are contiguous.  Variables without a ``time`` dimension
        are passed through unchanged.

        When ``self.pin_initial_time_point`` is True, ``{name}_0`` is
        omitted so ``t=0`` never enters the augmented state.
        """
        start_idx = 1 if self.pin_initial_time_point else 0
        flat_data_vars: dict = {}
        for name in params.data_vars:
            if "time" in params[name].dims:
                for t_idx in range(start_idx, self.num_time_points):
                    flat_data_vars[f"{name}_{t_idx}"] = (
                        "ensemble",
                        jnp.asarray(params[name].isel(time=t_idx).values),
                    )
            else:
                flat_data_vars[name] = (
                    "ensemble",
                    jnp.asarray(params[name].values),
                )
        return xarray.Dataset(
            data_vars=flat_data_vars,
            coords={"ensemble": params.coords["ensemble"]},
        )

    def _unflatten_params(
        self,
        flat_params: xarray.Dataset,
        original_params: xarray.Dataset,
    ) -> xarray.Dataset:
        """Reverse :meth:`_flatten_time_varying_params`.

        When ``self.pin_initial_time_point`` is True, ``t=0`` was not
        flattened — reinsert it per-member from ``original_params``.
        """
        start_idx = 1 if self.pin_initial_time_point else 0
        data_vars: dict = {}
        for name in original_params.data_vars:
            if "time" in original_params[name].dims:
                time_slices: list = []
                if self.pin_initial_time_point:
                    time_slices.append(
                        jnp.asarray(original_params[name].isel(time=0).values)
                    )
                time_slices.extend(
                    jnp.asarray(flat_params[f"{name}_{t_idx}"].values)
                    for t_idx in range(start_idx, self.num_time_points)
                )
                data_vars[name] = (
                    ("time", "ensemble"),
                    jnp.stack(time_slices, axis=0),
                )
            else:
                data_vars[name] = flat_params[name]
        return xarray.Dataset(data_vars=data_vars, coords=original_params.coords)

    def _one_step(
        self,
        params: xarray.Dataset,
        obs: jnp.ndarray,
        state: Optional[xarray.Dataset] = None,
    ) -> tuple[Optional[xarray.Dataset], xarray.Dataset]:
        flat_params = self._flatten_time_varying_params(params)
        _, updated_flat = super()._one_step(flat_params, obs, state)
        updated_params = self._unflatten_params(updated_flat, params)
        return None, updated_params


class StateAndParameterESMDA(_BaseESMDA):
    """Joint state and parameter ESMDA smoothing."""

    def _get_states(
        self,
        state: Optional[xarray.Dataset] = None,
        results_dir: Optional[pathlib.Path] = None,
    ) -> xarray.Dataset:
        """Get ensemble states, selecting the first timestep.

        Both branches must select the same frame: the augmented Kalman vector
        holds the window's initial condition, and ``_analysis`` feeds the
        analyzed result forward as the next forecast's warm start. Reading a
        different frame on disk would re-assimilate the window's observations
        against a state from a different time.
        """
        if state is not None:
            return state.isel(time=0)
        if results_dir is not None:
            state_files = self._get_sorted_state_files(pathlib.Path(results_dir))
            if not state_files:
                raise FileNotFoundError(
                    f"No state_*.nc files found in results directory: {results_dir}"
                )
            states = [xarray.open_dataset(f).isel(time=0) for f in state_files]
            return xarray.concat(states, dim="ensemble", join="override")
        raise ValueError("Either state or results_dir must be provided.")

    def _flatten_state(self, state: xarray.Dataset) -> jnp.ndarray:
        """Flatten ensemble state to (degrees_of_freedom, ensemble_size)."""
        flat_vars = []
        for var_name in sorted(state.data_vars):
            variable = state[var_name]
            flat_var = variable.transpose("ensemble", ...).values.reshape(
                variable.sizes["ensemble"], -1
            )
            flat_vars.append(flat_var.T)
        return jnp.concatenate(flat_vars, axis=0)

    def _state_group_ids(self, state: xarray.Dataset) -> jnp.ndarray:
        """Block id per flattened state row, grouping co-located cells.

        Each variable flattens (in :meth:`_flatten_state`) to its per-cell
        positions in the same order, so the within-variable position is the
        cell id.  Sharing that id across variables groups the co-located
        components — e.g. ``u``/``v``/``w`` at one grid cell — into one block,
        giving them the joint, balanced update of the paper's grid-block
        local analysis.
        """
        per_var = []
        for var_name in sorted(state.data_vars):
            variable = state[var_name]
            n_cells = variable.size // variable.sizes["ensemble"]
            per_var.append(jnp.arange(n_cells, dtype=int))
        return jnp.concatenate(per_var)

    def _unflatten_state(
        self,
        states_array: jnp.ndarray,
        state_template: xarray.Dataset,
    ) -> xarray.Dataset:
        """Unflatten a (degrees_of_freedom, ensemble_size) array back to xarray."""
        new_data_vars = {}
        ensemble_size = states_array.shape[1]
        current_pos = 0

        for var_name in sorted(state_template.data_vars):
            template_var = state_template[var_name]
            var_flat_size = template_var.size // template_var.sizes["ensemble"]
            flat_var_chunk = states_array[current_pos : current_pos + var_flat_size, :]
            current_pos += var_flat_size

            dims_no_ensemble = [d for d in template_var.dims if d != "ensemble"]
            shape_no_ensemble = [template_var.sizes[d] for d in dims_no_ensemble]
            data = flat_var_chunk.T.reshape((ensemble_size, *shape_no_ensemble))

            new_dims_order = ["ensemble"] + dims_no_ensemble
            data_array = xarray.DataArray(data, dims=new_dims_order)
            new_data_vars[var_name] = data_array.transpose(*template_var.dims)

        return xarray.Dataset(new_data_vars, coords=state_template.coords)

    def _state_row_coords(self, state: xarray.Dataset) -> jnp.ndarray:
        """Physical ``(x, y, z)`` coordinate of each flattened state row.

        Mirrors :meth:`_flatten_state`'s ordering exactly: for each variable
        (sorted), the non-ensemble dims are flattened in C-order.  The grid
        coordinate along each axis is read from the dim's coordinate values; a
        dim's axis (x/y/z) is taken from the leading character of its name
        (``x``/``xt``/``xm``/``xu`` -> x, ``y``/``yt``/``ym``/``yv`` -> y,
        ``z``/``zt``/``zm`` -> z), which holds for every supported solver grid.
        Used only by distance-based localization.
        """
        coords_list = []
        for var_name in sorted(state.data_vars):
            variable = state[var_name]
            dims_no_ens = [d for d in variable.dims if d != "ensemble"]
            for d in dims_no_ens:
                if d not in state.coords:
                    raise ValueError(
                        f"State dimension '{d}' (variable '{var_name}') has no "
                        "coordinate values; distance-based localization needs "
                        "physical grid coordinates for every state dimension — "
                        "grid indices would be compared against sensor "
                        "coordinates in metres."
                    )
            coord_1d = [
                np.asarray(state[d].values, dtype=float) for d in dims_no_ens
            ]
            # meshgrid(indexing="ij").ravel() flattens in the same C-order as
            # transpose("ensemble", ...).reshape(ensemble, -1) in _flatten_state.
            grids = np.meshgrid(*coord_1d, indexing="ij") if coord_1d else []
            flat = [g.ravel() for g in grids]
            n_cells = flat[0].size if flat else variable.size // variable.sizes["ensemble"]
            per_axis = {d[0]: flat[i] for i, d in enumerate(dims_no_ens)}
            xyz = np.stack(
                [per_axis.get(axis, np.zeros(n_cells)) for axis in ("x", "y", "z")],
                axis=1,
            )
            coords_list.append(xyz)
        return jnp.asarray(np.concatenate(coords_list, axis=0))

    def _augmented_state_update(
        self,
        states_array: xarray.Dataset,
        flat_params: xarray.Dataset,
        pred_obs: jnp.ndarray,
        obs: jnp.ndarray,
        N_e: int,
    ) -> tuple[xarray.Dataset, xarray.Dataset]:
        """Build ``[state | params]``, apply the (state-only) Kalman update, split.

        ``flat_params`` is a Dataset of scalar ``(ensemble,)`` variables (static
        params, or time-varying params already flattened to ``{name}_{t}``).
        Localization — correlation- or distance-based — is applied to the STATE
        rows only; parameter rows always get the global update (``localize_mask``).
        Returns ``(updated_state, updated_flat_params)``.
        """
        param_names = list(flat_params.data_vars.keys())
        state_template = xarray.Dataset(
            {
                var: (
                    states_array[var].dims,
                    jnp.empty(states_array[var].shape, dtype=states_array[var].dtype),
                )
                for var in states_array.data_vars
            },
            coords={coord: states_array.coords[coord] for coord in states_array.coords},
        )

        states_flat = self._flatten_state(states_array)
        N_s = states_flat.shape[0]

        params_array = jnp.array([flat_params[name].values for name in param_names])
        augmented = jnp.concatenate([states_flat, params_array], axis=0)

        # State-only localization (no-op on the global path where localization is
        # None). Parameter rows are masked out so they receive the global update.
        localize_mask = None
        group_ids = None
        row_coords = None
        obs_coords = None
        if self.localization is not None:
            localize_mask = jnp.concatenate(
                [jnp.ones(N_s, dtype=bool), jnp.zeros(len(param_names), dtype=bool)]
            )
            # Block grouping: co-locate state rows by grid cell; each parameter
            # is its own block (and masked anyway), so it never joins a state one.
            if _block_grouping_enabled(self.localization):
                state_groups = self._state_group_ids(states_array)
                offset = int(state_groups.max()) + 1
                param_groups = offset + _group_ids_by_base_name(param_names)
                group_ids = jnp.concatenate([state_groups, param_groups])
            # Distance-based localization additionally needs grid/sensor coords.
            if getattr(self.localization, "requires_coordinates", False):
                state_coords = self._state_row_coords(states_array)  # (N_s, 3)
                param_coords = jnp.zeros((len(param_names), 3))  # masked out
                row_coords = jnp.concatenate([state_coords, param_coords], axis=0)
                obs_coords = self._observation_coords(obs.shape[0])

        augmented = self._compute_kalman_update(
            augmented,
            pred_obs,
            obs,
            N_e,
            group_ids=group_ids,
            localize_mask=localize_mask,
            row_coords=row_coords,
            obs_coords=obs_coords,
        )

        updated_states = self._unflatten_state(augmented[:N_s, :], state_template)
        updated_flat = xarray.Dataset(
            {
                name: ("ensemble", augmented[N_s + i, :])
                for i, name in enumerate(param_names)
            },
            coords={"ensemble": flat_params.coords["ensemble"]},
        )
        return updated_states, updated_flat

    def _one_step(
        self,
        params: xarray.Dataset,
        obs: jnp.ndarray,
        state: Optional[xarray.Dataset] = None,
    ) -> tuple[xarray.Dataset, xarray.Dataset]:
        obs = jnp.asarray(obs)
        results_dir = (
            self.forward_model.results_dir
            if self.forward_model.save_on_disk
            else None
        )

        pred_obs = self._observation_step(state=state, results_dir=results_dir)
        pred_obs = jnp.asarray(pred_obs).T

        states_array = self._get_states(state=state, results_dir=results_dir)
        N_e = params.sizes["ensemble"]

        # Static params are already scalar (ensemble,) vars: no flatten needed.
        updated_states, updated_flat = self._augmented_state_update(
            states_array, params, pred_obs, obs, N_e
        )
        return updated_states, xarray.Dataset(
            data_vars={name: updated_flat[name] for name in updated_flat.data_vars},
            coords=params.coords,
        )


class StateAndTimeVaryingParameterESMDA(
    StateAndParameterESMDA, TimeVaryingParameterESMDA
):
    """Joint state and time-varying-parameter ESMDA smoothing.

    Combines :class:`StateAndParameterESMDA` (the window's ``time=0`` initial-
    condition state is flattened into the augmented vector) with
    :class:`TimeVaryingParameterESMDA` (each time-varying parameter is flattened
    into per-time-knot scalars ``{name}_{t}``, respecting
    ``pin_initial_time_point``).

    Localization — correlation- or distance-based — when enabled is applied ONLY
    to the state rows: parameter rows always receive the plain global Kalman
    update (via ``localize_mask``).  The augmented update is built by the shared
    :meth:`StateAndParameterESMDA._augmented_state_update`; grid-block grouping
    co-locates the state rows by cell and keeps parameters in separate blocks
    (moot, since they are masked to the global update).

    The constructor is inherited from :class:`TimeVaryingParameterESMDA`
    (``StateAndParameterESMDA`` defines no ``__init__``); its signature is
    ``(observation_operator, forward_model, C_D, num_time_points,
    num_steps=3, alpha=None, rng_key=..., pin_initial_time_point=False,
    localization=None)``.
    """

    def _one_step(
        self,
        params: xarray.Dataset,
        obs: jnp.ndarray,
        state: Optional[xarray.Dataset] = None,
    ) -> tuple[xarray.Dataset, xarray.Dataset]:
        obs = jnp.asarray(obs)
        results_dir = (
            self.forward_model.results_dir
            if self.forward_model.save_on_disk
            else None
        )

        pred_obs = self._observation_step(state=state, results_dir=results_dir)
        pred_obs = jnp.asarray(pred_obs).T

        # Time=0 state ensemble + time-varying params flattened to per-knot
        # scalars (respecting pin_initial_time_point). The shared state-only
        # localized update handles the augmented [state | params] vector.
        states_array = self._get_states(state=state, results_dir=results_dir)
        flat_params = self._flatten_time_varying_params(params)
        N_e = flat_params.sizes["ensemble"]

        updated_states, updated_flat = self._augmented_state_update(
            states_array, flat_params, pred_obs, obs, N_e
        )
        updated_params = self._unflatten_params(updated_flat, params)
        return updated_states, updated_params
