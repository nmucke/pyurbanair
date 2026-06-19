"""Critically-damped AR(2) prior with relaxation between windows.

Implements the recursive scheme described in Evensen (2024) for
inflow-forcing data assimilation with an LBM LES model:

    dz/dt = w,    dw/dt = -2 λ w - λ² z + η(t),    λ = √3 / l_corr

where ``η`` is white noise scaled so that ``z`` has zero mean and unit
variance in stationarity.  ``z`` is C¹-smooth with correlation length
``l_corr``.

Window 0 prior (Eq. 36 in the paper):
    x_j(t) = x_ext + Σ_ext z_j(t)
with ``z`` drawn from its stationary distribution.

Between windows (Eqs. 40-43):
    The AR(2) state ``(z, w)`` is carried forward across windows so the
    next window's draw is C¹-continuous with the previous one.  The
    next window's *prior mean* is blended with the previous posterior
    mean via an exponential relaxation toward ``x_ext``:

        x(t) = (1 - α(t)) x_ext + α(t) (μ_end + Σ_ext z_j(t)),
        α(t) = exp(-(t - t_0) / l_corr).

    At t = t_0 the prior matches μ_end; far into the window it relaxes
    back to x_ext with full external spread.
"""

from __future__ import annotations

import math
from typing import Optional

import jax
import jax.numpy as jnp
import numpy as np
import xarray

from .base import ParameterTimeSeries


def build_knot_times(
    start: float, end: float, seconds_per_knot: float
) -> jnp.ndarray:
    """Knot times spaced ``seconds_per_knot`` over ``[start, end]``.

    Regular knots fall on ``start, start + s, start + 2s, …`` up to the last
    multiple ``≤ end``.  When ``end`` is not an exact multiple of the spacing
    past ``start``, a final knot is appended *at* ``end`` — the trailing,
    sub-spacing interval onto which the parameter trajectory is linearly
    extrapolated (see :meth:`AR2RelaxationModel._append_linear_endpoint`).  A
    ``seconds_per_knot`` larger than the whole span degenerates to the two
    endpoints ``[start, end]``.
    """
    start = float(start)
    end = float(end)
    s = float(seconds_per_knot)
    span = end - start
    n_intervals = int(math.floor(span / s + 1e-9)) if s > 0 else 0
    if n_intervals <= 0:
        return jnp.asarray([start, end])
    reg = start + s * np.arange(n_intervals + 1)
    if reg[-1] < end - 1e-9:
        reg = np.append(reg, end)
    return jnp.asarray(reg)


class AR2RelaxationModel(ParameterTimeSeries):
    """AR(2) prior + posterior-anchored relaxation extrapolation."""

    def __init__(
        self,
        external_parameters: dict[str, dict[str, float]],
        simulation_time: float,
        seconds_per_knot: float,
        correlation_length: float,
        seed: int = 0,
    ) -> None:
        # The knots are the trajectory's time coordinates: the parameter takes a
        # new value every ``seconds_per_knot`` seconds. When the window length is
        # not an exact multiple of the spacing, the final knot sits at
        # ``simulation_time`` and is reached by linear extrapolation rather than a
        # fresh AR(2) step (see :meth:`_append_linear_endpoint`).
        self.simulation_time = float(simulation_time)
        self.seconds_per_knot = float(seconds_per_knot)
        time_coords = build_knot_times(0.0, simulation_time, seconds_per_knot)
        super().__init__(external_parameters, time_coords, seed)
        self.correlation_length = correlation_length
        self.lam = math.sqrt(3.0) / max(correlation_length, 1e-6)

        # Carried state: per-parameter terminal (z, w) of the most
        # recent draw.  ``None`` triggers a stationary cold start.
        self._state: dict[str, tuple[jnp.ndarray, jnp.ndarray]] = {}

    # ------------------------------------------------------------------
    # AR(2) integration
    # ------------------------------------------------------------------

    def _stationary_init(
        self, rng_key: jax.Array, ensemble_size: int
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Sample ``(z, w)`` from the stationary distribution.

        Cov(z) = 1, Cov(w) = λ², Cov(z, w) = 0.
        """
        key_z, key_w = jax.random.split(rng_key)
        z0 = jax.random.normal(key_z, (ensemble_size,))
        w0 = jax.random.normal(key_w, (ensemble_size,)) * self.lam
        return z0, w0

    def _integrate(
        self,
        time_coords: jnp.ndarray,
        z0: jnp.ndarray,
        w0: jnp.ndarray,
        rng_key: jax.Array,
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """Exact discrete-time integration of the critically-damped AR(2).

        Uses the closed-form transition matrix ``F = exp(A·dt)`` (which
        is exact for the double eigenvalue at ``-λ``) and the exact
        one-step process noise ``Q = P_stat - F P_stat F^T`` with
        ``P_stat = diag(1, λ²)``, factored via Cholesky.  Each grid step
        preserves the stationary covariance exactly without substepping,
        matching the reference implementation in Evensen's Dasys code
        (``m_smooth_random_series.F90``).
        """
        time_coords = jnp.asarray(time_coords)
        n_t = time_coords.shape[0]

        ensemble_size = z0.shape[0]
        intervals = jnp.diff(time_coords)
        # Two independent normals per step for the 2-D Cholesky factor.
        eps = jax.random.normal(rng_key, (n_t - 1, 2, ensemble_size))

        lam = self.lam
        lam2 = lam * lam

        def advance_interval(state, scan_input):
            z, w = state
            dt, eps_pair = scan_input
            e = jnp.exp(-lam * dt)
            a11 = e * (1.0 + lam * dt)
            a12 = e * dt
            a21 = -e * lam2 * dt
            a22 = e * (1.0 - lam * dt)

            q11 = 1.0 - (a11 * a11 + a12 * a12 * lam2)
            q12 = -(a11 * a21 + a12 * a22 * lam2)
            q22 = lam2 - (a21 * a21 + a22 * a22 * lam2)

            q11 = jnp.maximum(q11, 0.0)
            q22 = jnp.maximum(q22, 0.0)

            l11 = jnp.sqrt(q11)
            l11_safe = jnp.where(l11 > 0.0, l11, 1.0)
            l21 = jnp.where(l11 > 0.0, q12 / l11_safe, 0.0)
            l22 = jnp.sqrt(jnp.maximum(q22 - l21 * l21, 0.0))

            eps_z, eps_w = eps_pair[0], eps_pair[1]
            z_new = a11 * z + a12 * w + l11 * eps_z
            w_new = a21 * z + a22 * w + l21 * eps_z + l22 * eps_w
            return (z_new, w_new), z_new

        (z_final, w_final), z_grid = jax.lax.scan(
            advance_interval, (z0, w0), (intervals, eps)
        )
        z_traj = jnp.concatenate([z0[None, :], z_grid], axis=0)
        return z_traj, z_final, w_final

    # ------------------------------------------------------------------
    # Extrapolated endpoint (window length not a multiple of the spacing)
    # ------------------------------------------------------------------

    def _split_endpoint(
        self, times: jnp.ndarray
    ) -> tuple[jnp.ndarray, Optional[float]]:
        """Split an output grid into ``(regular_knots, end_time | None)``.

        A trailing interval shorter than ``seconds_per_knot`` marks the
        linearly-extrapolated endpoint: the AR(2) is integrated only over the
        regular knots and the final value is extrapolated onto ``end_time``.
        """
        times = jnp.asarray(times)
        if times.shape[0] >= 2:
            last_dt = float(times[-1] - times[-2])
            if last_dt < self.seconds_per_knot - 1e-6:
                return times[:-1], float(times[-1])
        return times, None

    @staticmethod
    def _append_linear_endpoint(
        values: jnp.ndarray, reg_times: jnp.ndarray, end_time: float
    ) -> jnp.ndarray:
        """Append a per-member linearly-extrapolated final row at ``end_time``.

        ``values`` is ``(n_reg, N_e)`` on ``reg_times``; the appended row is the
        linear extrapolation of the last two knots onto ``end_time``.
        """
        reg_times = jnp.asarray(reg_times)
        slope = (values[-1] - values[-2]) / (reg_times[-1] - reg_times[-2])
        end_val = values[-1] + slope * (end_time - reg_times[-1])
        return jnp.concatenate([values, end_val[None, :]], axis=0)

    # ------------------------------------------------------------------
    # ParameterTimeSeries API
    # ------------------------------------------------------------------

    def sample(self, ensemble_size: int) -> xarray.Dataset:
        """Cold-start AR(2) prior, eq. 36: x = x_ext + Σ_ext z."""
        time_coords = self.time_coords
        reg_times, end_time = self._split_endpoint(time_coords)
        keys = jax.random.split(self.rng_key, 2 * len(self.param_names))
        arrays: dict[str, jnp.ndarray] = {}
        self._state = {}

        for i, name in enumerate(self.param_names):
            init_key, integ_key = keys[2 * i], keys[2 * i + 1]
            z0, w0 = self._stationary_init(init_key, ensemble_size)
            z_traj, z_end, w_end = self._integrate(
                reg_times, z0, w0, integ_key
            )
            # z_traj is the unit-variance AR(2) anomaly; apply the (possibly
            # time-varying) external envelope x_ext(t) + Σ_ext(t)·z.
            mean_t, std_t = self._ext_profile(name, reg_times)
            vals = mean_t[:, None] + std_t[:, None] * z_traj
            if end_time is not None:
                vals = self._append_linear_endpoint(vals, reg_times, end_time)
            arrays[name] = vals
            self._state[name] = (z_end, w_end)

        return self._build_dataset(arrays, time_coords, ensemble_size)

    def extrapolate(
        self,
        posterior: xarray.Dataset,
        prediction_times: jnp.ndarray,
        rng_key: jax.Array,
    ) -> xarray.Dataset:
        """Posterior-anchored AR(2) draw blended via eq. 42.

        For continuity across windows the AR(2) state at the start of
        the new window is initialized PER MEMBER from the normalized
        end-of-window posterior (eq. 40):

            z_j(t_0) = (x_post_j(t_end) - μ_end) / σ_ext,
            w_j(t_0) ≈ (z_j(t_end) - z_j(t_end - Δt)) / Δt.

        With α(t_0) = 1 this guarantees the prior matches each member's
        own posterior value at the window boundary.  Far into the
        window α(t) → 0 and the prior relaxes back to ``x_ext``.
        """
        prediction_times = jnp.asarray(prediction_times)
        reg_times, end_time = self._split_endpoint(prediction_times)
        ensemble_size = posterior.sizes["ensemble"]
        t0 = reg_times[0]
        alpha = jnp.exp(
            -(reg_times - t0) / max(self.correlation_length, 1e-6)
        )

        keys = jax.random.split(rng_key, len(self.param_names))
        arrays: dict[str, jnp.ndarray] = {}
        passthrough = {
            n: posterior[n]
            for n in posterior.data_vars
            if "time" not in posterior[n].dims
        }
        new_state: dict[str, tuple[jnp.ndarray, jnp.ndarray]] = {}

        for key, name in zip(keys, self.param_names):
            x_ext = self._ext_scalar(name, "mean")
            std = self._ext_scalar(name, "std")
            std_safe = max(std, 1e-12)

            if name in posterior.data_vars and "time" in posterior[name].dims:
                y_post = jnp.asarray(
                    posterior[name].transpose("time", "ensemble").values
                )  # (N_t_post, N_e)
                mu_end = y_post[-1].mean()
                # Per-member normalized end-of-window state.
                z0 = (y_post[-1] - mu_end) / std_safe
                if y_post.shape[0] >= 2:
                    # Each timepoint is normalized by ITS OWN ensemble
                    # mean before differencing, matching the reference
                    # Fortran (m_ensemble_forcing.F90).
                    mu_prev = y_post[-2].mean()
                    z_prev = (y_post[-2] - mu_prev) / std_safe
                    post_times = jnp.asarray(posterior.coords["time"].values)
                    dt_post = post_times[-1] - post_times[-2]
                    w0 = (z0 - z_prev) / jnp.maximum(dt_post, 1e-6)
                else:
                    w0 = jnp.zeros_like(z0)
            else:
                # No posterior trajectory for this parameter — fall back
                # to a stationary cold start.
                init_key, key = jax.random.split(key)
                z0, w0 = self._stationary_init(init_key, ensemble_size)
                mu_end = jnp.asarray(x_ext)

            z_traj, z_end, w_end = self._integrate(
                reg_times, z0, w0, key
            )
            new_state[name] = (z_end, w_end)

            ar2_part = mu_end + std * z_traj  # (N_reg, N_e)
            ext_part = jnp.full_like(ar2_part, x_ext)
            vals = (
                alpha[:, None] * ar2_part + (1.0 - alpha[:, None]) * (ext_part + std * z_traj)
            )
            if end_time is not None:
                vals = self._append_linear_endpoint(vals, reg_times, end_time)
            arrays[name] = vals

        self._state = new_state
        return self._build_dataset(
            arrays, prediction_times, ensemble_size, passthrough=passthrough
        )
