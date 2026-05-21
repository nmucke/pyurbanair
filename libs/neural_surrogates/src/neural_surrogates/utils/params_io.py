"""Parameters Dataset -> dense per-step conditioning (framework side of §1.5).

The Fortran solvers accept a *sparse* time-varying parameter series and
interpolate between values at runtime (e.g. pylbm's ``write_uvel_time_file`` +
``m_inflow.F90``). The framework replicates this here — **not** in any network
(``docs/neural_surrogate_plan.md`` §1.5):

- a params Dataset whose vars carry a sparse ``time`` dim is **linearly**
  interpolated onto the dense per-step rollout grid;
- ``inflow_angle`` is encoded via **sin/cos** so the 359°->1° wrap doesn't
  sweep (interpolating sin/cos directly is both wrap-safe and the encoding);
- the **spin-up plateau** convention is replicated when spin-up frames precede
  the window;
- scalar params (no ``time`` dim) broadcast to every step.

The network always receives conditioning vectors **ready to embed**; it never
sees the sparse series. Which params are included is driven by the checkpoint's
``ParamSchema``, not ``model.name`` — so uDALES-trained checkpoints keep
``pressure_gradient_magnitude`` (§6.2).
"""

from __future__ import annotations

import numpy as np
import xarray

from .schema import ParamSchema


def _values_on_grid(
    da: xarray.DataArray,
    target_times: np.ndarray,
) -> np.ndarray:
    """Linearly interpolate a (possibly scalar) param onto ``target_times``."""
    if "time" in da.dims:
        src_times = np.asarray(da["time"].values, dtype=float)
        src_vals = np.asarray(da.values, dtype=float)
        order = np.argsort(src_times)
        # np.interp clamps outside the sparse range (constant extrapolation),
        # matching the solver's hold-last-value behavior at the ends.
        return np.interp(target_times, src_times[order], src_vals[order])
    return np.full(target_times.shape, float(np.asarray(da.values)))


def params_to_conditioning(
    params: xarray.Dataset,
    schema: ParamSchema,
    num_steps: int,
    output_frequency: float,
    spinup_outputs: int = 0,
) -> np.ndarray:
    """Build the dense ``[T, P]`` per-step conditioning sequence (§1.5).

    Args:
        params: Parameter Dataset; vars may have a sparse ``time`` dim or be
            scalar. Must contain every name in ``schema.names``.
        schema: The checkpoint's parameter contract (order + which are angular).
        num_steps: Number of output frames in the (trimmed) window.
        output_frequency: Spacing between output frames, in physical units.
        spinup_outputs: If > 0, prepend this many plateau frames held at the
            initial value (so total length is ``spinup_outputs + num_steps``).

    Returns:
        ``[spinup_outputs + num_steps, P]`` float32 array, columns ordered per
        ``schema`` with each angular param expanded to ``[sin, cos]``.
    """
    if num_steps <= 0:
        raise ValueError(f"num_steps must be positive, got {num_steps}.")
    target_times = np.arange(num_steps, dtype=float) * float(output_frequency)

    columns: list[np.ndarray] = []
    for name in schema.names:
        if name not in params:
            raise KeyError(
                f"params is missing {name!r} required by schema; "
                f"has {list(params.data_vars)}."
            )
        da = params[name]
        if schema.is_angular(name):
            radians = np.deg2rad(_values_on_grid(da, target_times))
            columns.append(np.sin(radians))
            columns.append(np.cos(radians))
        else:
            columns.append(_values_on_grid(da, target_times))

    seq = np.stack(columns, axis=1).astype(np.float32)  # [num_steps, P]

    if spinup_outputs > 0:
        plateau = np.repeat(seq[:1], spinup_outputs, axis=0)
        seq = np.concatenate([plateau, seq], axis=0)
    return seq


def conditioning_for_frames(
    params: xarray.Dataset,
    schema: ParamSchema,
    frame_times: np.ndarray,
) -> np.ndarray:
    """Encode conditioning at explicit ``frame_times`` (corpus per-frame params).

    Used by the data generator to store one **effective** conditioning vector
    per output frame (``docs/neural_surrogate_plan.md`` §5), aligned with the
    dense convention above so training tensors match inference.
    """
    columns: list[np.ndarray] = []
    for name in schema.names:
        da = params[name]
        if schema.is_angular(name):
            radians = np.deg2rad(_values_on_grid(da, frame_times))
            columns.append(np.sin(radians))
            columns.append(np.cos(radians))
        else:
            columns.append(_values_on_grid(da, frame_times))
    return np.stack(columns, axis=1).astype(np.float32)
