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
import xarray

from .base import ParameterTimeSeries


class AR2RelaxationModel(ParameterTimeSeries):
    """AR(2) prior + posterior-anchored relaxation extrapolation."""

    def __init__(
        self,
        external_priors: dict[str, dict[str, float]],
        ensemble_size: int,
        correlation_length: float,
        substeps_per_interval: int = 8,
    ) -> None:
        super().__init__(external_priors, ensemble_size)
        self.correlation_length = correlation_length
        self.lam = math.sqrt(3.0) / max(correlation_length, 1e-6)
        # σ_η chosen so the stationary marginal variance of z is 1
        # (Lyapunov equation for the 2D linear SDE).
        self.sigma_eta = 2.0 * self.lam**1.5
        self.substeps_per_interval = max(int(substeps_per_interval), 1)

        # Carried state: per-parameter terminal (z, w) of the most
        # recent draw.  ``None`` triggers a stationary cold start.
        self._state: dict[str, tuple[jnp.ndarray, jnp.ndarray]] = {}

    # ------------------------------------------------------------------
    # AR(2) integration
    # ------------------------------------------------------------------

    def _stationary_init(
        self, rng_key: jax.Array
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Sample ``(z, w)`` from the stationary distribution.

        Cov(z) = 1, Cov(w) = λ², Cov(z, w) = 0.
        """
        key_z, key_w = jax.random.split(rng_key)
        z0 = jax.random.normal(key_z, (self.ensemble_size,))
        w0 = jax.random.normal(key_w, (self.ensemble_size,)) * self.lam
        return z0, w0

    def _integrate(
        self,
        time_coords: jnp.ndarray,
        z0: jnp.ndarray,
        w0: jnp.ndarray,
        rng_key: jax.Array,
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """Euler-Maruyama integration of the AR(2) SDE on ``time_coords``.

        Returns the trajectory ``z(t)`` of shape ``(N_t, N_e)`` and the
        terminal state ``(z_end, w_end)``.  Stores grid-point values
        only; ``substeps_per_interval`` sub-steps are taken between
        consecutive grid points for accuracy.
        """
        time_coords = jnp.asarray(time_coords)
        n_t = time_coords.shape[0]
        n_sub = self.substeps_per_interval

        intervals = jnp.diff(time_coords)
        # Pre-sample all noise increments: shape (n_t-1, n_sub, N_e).
        eps = jax.random.normal(
            rng_key, (n_t - 1, n_sub, self.ensemble_size)
        )

        lam = self.lam
        sigma_eta = self.sigma_eta

        def advance_interval(state, scan_input):
            z, w = state
            dt_total, eps_block = scan_input
            dt = dt_total / n_sub
            sqrt_dt = jnp.sqrt(jnp.maximum(dt, 0.0))

            def substep(carry, eps_k):
                zc, wc = carry
                z_new = zc + dt * wc
                w_new = (
                    wc
                    + dt * (-2.0 * lam * wc - lam**2 * zc)
                    + sigma_eta * sqrt_dt * eps_k
                )
                return (z_new, w_new), None

            (z_end, w_end), _ = jax.lax.scan(substep, (z, w), eps_block)
            return (z_end, w_end), z_end

        (z_final, w_final), z_grid = jax.lax.scan(
            advance_interval, (z0, w0), (intervals, eps)
        )
        z_traj = jnp.concatenate([z0[None, :], z_grid], axis=0)
        return z_traj, z_final, w_final

    # ------------------------------------------------------------------
    # ParameterTimeSeries API
    # ------------------------------------------------------------------

    def sample_prior(
        self,
        time_coords: jnp.ndarray,
        rng_key: jax.Array,
    ) -> xarray.Dataset:
        """Cold-start AR(2) prior, eq. 36: x = x_ext + Σ_ext z."""
        time_coords = jnp.asarray(time_coords)
        keys = jax.random.split(rng_key, 2 * len(self.param_names))
        arrays: dict[str, jnp.ndarray] = {}
        self._state = {}

        for i, name in enumerate(self.param_names):
            init_key, integ_key = keys[2 * i], keys[2 * i + 1]
            z0, w0 = self._stationary_init(init_key)
            z_traj, z_end, w_end = self._integrate(
                time_coords, z0, w0, integ_key
            )
            spec = self._ext(name)
            arrays[name] = spec["mean"] + spec["std"] * z_traj
            self._state[name] = (z_end, w_end)

        return self._build_dataset(arrays, time_coords)

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
        t0 = prediction_times[0]
        alpha = jnp.exp(
            -(prediction_times - t0) / max(self.correlation_length, 1e-6)
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
            spec = self._ext(name)
            x_ext = spec["mean"]
            std = spec["std"]
            std_safe = max(std, 1e-12)

            if name in posterior.data_vars and "time" in posterior[name].dims:
                y_post = jnp.asarray(
                    posterior[name].transpose("time", "ensemble").values
                )  # (N_t_post, N_e)
                mu_end = y_post[-1].mean()
                # Per-member normalized end-of-window state.
                z0 = (y_post[-1] - mu_end) / std_safe
                if y_post.shape[0] >= 2:
                    post_times = jnp.asarray(posterior.coords["time"].values)
                    dt_post = post_times[-1] - post_times[-2]
                    z_prev = (y_post[-2] - mu_end) / std_safe
                    w0 = (z0 - z_prev) / jnp.maximum(dt_post, 1e-6)
                else:
                    w0 = jnp.zeros_like(z0)
            else:
                # No posterior trajectory for this parameter — fall back
                # to a stationary cold start.
                init_key, key = jax.random.split(key)
                z0, w0 = self._stationary_init(init_key)
                mu_end = jnp.asarray(x_ext)

            z_traj, z_end, w_end = self._integrate(
                prediction_times, z0, w0, key
            )
            new_state[name] = (z_end, w_end)

            ar2_part = mu_end# + std * z_traj  # (N_t, N_e)
            ext_part = jnp.full_like(ar2_part, x_ext)
            arrays[name] = (
                alpha[:, None] * ar2_part + (1.0 - alpha[:, None]) * (ext_part + std * z_traj)
            )

        self._state = new_state
        return self._build_dataset(arrays, prediction_times, passthrough=passthrough)
