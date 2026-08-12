"""Augmented-state structure transforms shared by smoothing and filtering.

The Kalman analysis operates on flat ``(N_rows, N_e)`` arrays; the rest of the
library speaks ``xarray.Dataset``. The two classes here own the flatten /
unflatten round-trips (and the row metadata the localized update needs: block
ids and physical row coordinates) so the ESMDA smoothers and the sequential
filters share one flattening order and one pinning semantics.

* :class:`ParamAugmentation` — parameter Dataset ⟷ flat scalar-variable
  Dataset ⟷ ``(N_p, N_e)`` array, with optional per-time-knot expansion of
  time-varying parameters.
* :class:`StateAugmentation` — state Dataset ⟷ ``(N_s, N_e)`` array, plus the
  per-row grid-cell block ids and physical coordinates.
"""

from collections.abc import Mapping
from typing import Optional

import jax.numpy as jnp
import numpy as np
import xarray


class ParamAugmentation:
    """Flatten/unflatten parameter Datasets for the augmented Kalman vector.

    With ``num_time_points=None`` (static parameters) the flatten/unflatten
    pair is the identity: every variable is already a scalar ``(ensemble,)``
    entry of the augmented vector. With ``num_time_points`` set, each variable
    with a ``time`` dimension is expanded into per-knot scalars ``{name}_{t}``
    so all time points of one parameter are contiguous; variables without a
    ``time`` dimension pass through unchanged.

    When ``pin_initial_time_point`` is True, ``t=0`` of every time-varying
    parameter is excluded from the flattened (augmented) representation and
    reinserted unchanged during unflatten — per ensemble member. Useful for
    preserving cross-window continuity in rollout ESMDA.
    """

    def __init__(
        self,
        num_time_points: Optional[int] = None,
        pin_initial_time_point: bool = False,
    ) -> None:
        if pin_initial_time_point and num_time_points is None:
            raise ValueError(
                "pin_initial_time_point requires num_time_points: pinning t=0 "
                "only applies to time-varying parameters."
            )
        self.num_time_points = num_time_points
        self.pin_initial_time_point = pin_initial_time_point

    def check_num_time_points(self, params: xarray.Dataset) -> None:
        """Validate the data's ``time`` size against the configured knot count.

        The flatten/unflatten round-trip iterates ``range(num_time_points)``;
        if the data has fewer knots the trailing ones are silently dropped, and
        if it has more they raise an opaque out-of-bounds error. Check once,
        loudly.
        """
        if self.num_time_points is None or "time" not in params.dims:
            return
        data_time = int(params.sizes["time"])
        if data_time != self.num_time_points:
            raise ValueError(
                f"num_time_points ({self.num_time_points}) does not match the "
                f"params' time dimension ({data_time}). The time-varying "
                "flatten/unflatten assumes they agree; set num_time_points to "
                "the sampled prior's knot count."
            )

    def flatten(self, params: xarray.Dataset) -> xarray.Dataset:
        """Flatten ``(time, ensemble)`` params to scalar ``(ensemble,)`` vars.

        Static configuration (``num_time_points=None``): returns ``params``
        unchanged. Otherwise each variable with a ``time`` dimension is
        expanded into ``{name}_0``, ``{name}_1``, … (``{name}_1`` first when
        ``pin_initial_time_point`` is True, so ``t=0`` never enters the
        augmented state).
        """
        if self.num_time_points is None:
            return params
        self.check_num_time_points(params)
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

    def unflatten(
        self,
        flat_params: xarray.Dataset,
        original_params: xarray.Dataset,
    ) -> xarray.Dataset:
        """Reverse :meth:`flatten`.

        When ``pin_initial_time_point`` is True, ``t=0`` was not flattened —
        reinsert it per-member from ``original_params``.
        """
        if self.num_time_points is None:
            return flat_params
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

    def group_ids(self, params: xarray.Dataset) -> jnp.ndarray:
        """Block id per flattened param row, grouping knots of one parameter.

        Mirrors :meth:`flatten`'s variable order exactly: a time-varying
        parameter's ``{name}_{t}`` knots all share one block id; each static
        parameter is its own block. Built from the true name->(param, knot)
        mapping, so it cannot merge unrelated parameters the way stripping a
        numeric name suffix could.
        """
        if self.num_time_points is None:
            return jnp.arange(len(params.data_vars), dtype=int)
        self.check_num_time_points(params)
        n_knots = self.num_time_points - (1 if self.pin_initial_time_point else 0)
        ids: list[int] = []
        for block_id, name in enumerate(params.data_vars):
            repeats = n_knots if "time" in params[name].dims else 1
            ids.extend([block_id] * repeats)
        return jnp.asarray(ids, dtype=int)

    @staticmethod
    def to_array(flat_params: xarray.Dataset) -> jnp.ndarray:
        """Stack the flat scalar variables into a ``(N_p, N_e)`` array."""
        return jnp.array([flat_params[name].values for name in flat_params.data_vars])

    @staticmethod
    def from_array(
        params_array: jnp.ndarray, flat_template: xarray.Dataset
    ) -> xarray.Dataset:
        """Rebuild a flat scalar-variable Dataset from a ``(N_p, N_e)`` array."""
        return xarray.Dataset(
            data_vars={
                name: ("ensemble", params_array[i, :])
                for i, name in enumerate(flat_template.data_vars)
            },
            coords={"ensemble": flat_template.coords["ensemble"]},
        )


class StateAugmentation:
    """Flatten/unflatten state Datasets for the augmented Kalman vector.

    Stateless: the flattening order is fixed (variables in sorted name order,
    each transposed to ``(ensemble, ...)`` and its non-ensemble dims flattened
    in C-order) so :meth:`flatten`, :meth:`unflatten`, :meth:`row_coords` and
    :meth:`group_ids` all agree on which physical cell a row is.
    """

    def flatten(self, state: xarray.Dataset) -> jnp.ndarray:
        """Flatten ensemble state to (degrees_of_freedom, ensemble_size)."""
        flat_vars = []
        for var_name in sorted(state.data_vars):
            variable = state[var_name]
            flat_var = variable.transpose("ensemble", ...).values.reshape(
                variable.sizes["ensemble"], -1
            )
            flat_vars.append(flat_var.T)
        return jnp.concatenate(flat_vars, axis=0)

    def unflatten(
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

    def flatten_snapshots(
        self, state: xarray.Dataset, snapshot_stride: int = 1
    ) -> jnp.ndarray:
        """Flatten every (member, time frame) into a snapshot column.

        Returns an array of shape (degrees_of_freedom, N_e * N_t) whose row
        ordering matches :meth:`flatten` exactly (sorted variables, spatial
        dims flattened in C-order), so a column is directly comparable to a
        flattened single-frame state vector. Time frames are thinned by
        ``snapshot_stride``.
        """
        state = state.isel(time=slice(None, None, snapshot_stride))
        flat_vars = []
        n_samples = state.sizes["ensemble"] * state.sizes["time"]
        for var_name in sorted(state.data_vars):
            variable = state[var_name]
            flat_var = variable.transpose("ensemble", "time", ...).values.reshape(
                n_samples, -1
            )
            flat_vars.append(flat_var.T)
        return jnp.concatenate(flat_vars, axis=0)

    def group_ids(self, state: xarray.Dataset) -> jnp.ndarray:
        """Block id per flattened state row, grouping truly co-located cells.

        Variables whose ordered spatial axes have identical physical coordinate
        arrays share the same per-cell block ids. This groups ``u``/``v``/``w``
        on pylbm's collocated grid, while keeping face-centred variables on the
        staggered uDALES/PALM grids in separate blocks. Merely sharing an array
        shape is insufficient: flat index ``i`` on ``xm`` and ``xt`` generally
        denotes different physical locations.

        The signature comparison is done once per variable, rather than by
        materializing and uniquing one xyz tuple per state row. It therefore
        preserves the cheap metadata construction needed for large CFD grids.
        """
        per_var: list[jnp.ndarray] = []
        grid_offsets: dict[tuple, int] = {}
        next_group = 0

        for var_name in sorted(state.data_vars):
            variable = state[var_name]
            dims_no_ensemble = [d for d in variable.dims if d != "ensemble"]
            signature_parts = []
            for dim in dims_no_ensemble:
                if dim not in state.coords:
                    raise ValueError(
                        f"State dimension '{dim}' (variable '{var_name}') has no "
                        "coordinate values; grid-block localization cannot "
                        "determine whether state rows are physically co-located."
                    )
                # Normalize numeric dtype so physically identical float32 and
                # float64 coordinate axes are recognized as the same grid.
                values = np.asarray(state[dim].values, dtype=float)
                signature_parts.append(
                    (str(dim)[0], values.dtype.str, values.shape, values.tobytes())
                )

            # Dimension order is part of the signature, matching flatten()'s
            # C-order. Equal signatures therefore imply equal flat-index ->
            # physical-location mappings.
            signature = tuple(signature_parts)
            n_cells = variable.size // variable.sizes["ensemble"]
            if signature not in grid_offsets:
                grid_offsets[signature] = next_group
                next_group += n_cells
            offset = grid_offsets[signature]
            per_var.append(offset + jnp.arange(n_cells, dtype=int))

        return jnp.concatenate(per_var)

    def row_scales(
        self, state: xarray.Dataset, variable_scales: Mapping[str, float]
    ) -> tuple[jnp.ndarray, dict[str, float]]:
        """Per-row characteristic scale, in :meth:`flatten` order.

        Returns the ``(N_s,)`` scale vector and the resolved per-variable scales
        (variables absent from ``variable_scales`` get ``1.0``). Lives here so
        the flattening order has exactly one owner: a reduction that divides
        rows by these scales must use the same ordering as :meth:`flatten`.
        """
        resolved = {
            str(name): float(variable_scales.get(str(name), 1.0))
            for name in sorted(state.data_vars)
        }
        rows = [
            jnp.full(
                (state[name].size // state[name].sizes["ensemble"],), resolved[name]
            )
            for name in sorted(state.data_vars)
        ]
        return (jnp.concatenate(rows) if rows else jnp.empty((0,))), resolved

    def row_coords(self, state: xarray.Dataset) -> jnp.ndarray:
        """Physical ``(x, y, z)`` coordinate of each flattened state row.

        Mirrors :meth:`flatten`'s ordering exactly: for each variable (sorted),
        the non-ensemble dims are flattened in C-order.  The grid coordinate
        along each axis is read from the dim's coordinate values; a dim's axis
        (x/y/z) is taken from the leading character of its name
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
            coord_1d = [np.asarray(state[d].values, dtype=float) for d in dims_no_ens]
            # meshgrid(indexing="ij").ravel() flattens in the same C-order as
            # transpose("ensemble", ...).reshape(ensemble, -1) in flatten().
            grids = np.meshgrid(*coord_1d, indexing="ij") if coord_1d else []
            flat = [g.ravel() for g in grids]
            n_cells = (
                flat[0].size if flat else variable.size // variable.sizes["ensemble"]
            )
            per_axis = {d[0]: flat[i] for i, d in enumerate(dims_no_ens)}
            xyz = np.stack(
                [per_axis.get(axis, np.zeros(n_cells)) for axis in ("x", "y", "z")],
                axis=1,
            )
            coords_list.append(xyz)
        return jnp.asarray(np.concatenate(coords_list, axis=0))
