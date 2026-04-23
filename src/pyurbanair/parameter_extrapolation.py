"""Extrapolation of time-varying parameters between assimilation windows.

Three methods are provided and selected via the ``method`` argument:

- ``"linear_trend_gp"``: per-member least-squares linear trend plus a
  zero-mean Gaussian-process residual with heteroscedastic observation
  noise from the cross-ensemble spread.  The trend continues into the
  extrapolation window (optionally exponentially damped), so the
  prediction does not collapse to a constant.

- ``"ar1"``: per-member first-order autoregressive model fit by
  ordinary least squares on the posterior trajectory and rolled forward
  deterministically from the last training value.  With the estimated
  AR coefficient clipped to ``(-phi_max, phi_max)``.

- ``"ornstein_uhlenbeck"``: per-member OU-with-drift SDE
  ``dx = θ_e (x*_e − x) dt + σ dW_e``, fitted by OLS on the posterior
  trajectory with ensemble-pooled diffusion ``σ``.  Rolled forward
  stochastically (Euler–Maruyama) with independent Brownian increments
  per member, so the ensemble spread grows across the extrapolation
  window instead of collapsing.
"""

from typing import Optional

import jax
import jax.numpy as jnp
import xarray


# ---------------------------------------------------------------------------
# GP helpers
# ---------------------------------------------------------------------------


def _rbf_kernel(
    x1: jnp.ndarray,
    x2: jnp.ndarray,
    correlation_length: float,
) -> jnp.ndarray:
    dt = x1[:, None] - x2[None, :]
    return jnp.exp(-0.5 * (dt / jnp.maximum(correlation_length, 1e-6)) ** 2)


def _gp_predict(
    x_train: jnp.ndarray,
    y_train: jnp.ndarray,
    x_pred: jnp.ndarray,
    correlation_length: float,
    amplitude: float,
    noise_var: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    amp_sq = amplitude**2
    K_tt = amp_sq * _rbf_kernel(x_train, x_train, correlation_length)
    K_tt = K_tt + jnp.diag(noise_var) + 1e-8 * jnp.eye(K_tt.shape[0])

    K_pt = amp_sq * _rbf_kernel(x_pred, x_train, correlation_length)
    K_pp = amp_sq * _rbf_kernel(x_pred, x_pred, correlation_length)

    L = jnp.linalg.cholesky(K_tt)
    alpha = jax.scipy.linalg.cho_solve((L, True), y_train)
    V = jax.scipy.linalg.solve_triangular(L, K_pt.T, lower=True)

    mean = K_pt @ alpha
    var = jnp.diag(K_pp) - jnp.sum(V**2, axis=0)
    std = jnp.sqrt(jnp.maximum(var, 0.0))

    return mean, std


def _linear_trend_at(
    t: jnp.ndarray,
    a: jnp.ndarray,
    b: jnp.ndarray,
    t_end: float,
    slope_damping_time: Optional[float],
) -> jnp.ndarray:
    """Evaluate (possibly slope-damped) linear trend at ``t``.

    For ``t <= t_end`` this is ``a + b*t``.  For ``t > t_end`` the slope
    decays exponentially with time-scale ``slope_damping_time``, so the
    extrapolated mean asymptotes to ``a + b*t_end + b*tau`` rather than
    running away.  When ``slope_damping_time`` is ``None`` or non-positive,
    plain linear extrapolation is used.
    """
    if slope_damping_time is None or slope_damping_time <= 0:
        return a + b * t
    tau = slope_damping_time
    delta = jnp.maximum(t - t_end, 0.0)
    return a + b * t_end + b * tau * (1.0 - jnp.exp(-delta / tau))


# ---------------------------------------------------------------------------
# Method: linear_trend_gp
# ---------------------------------------------------------------------------


def _extrapolate_linear_trend_gp(
    params: xarray.Dataset,
    prediction_times: jnp.ndarray,
    correlation_length: Optional[float],
    include_std: bool,
    kernel_amplitudes: Optional[dict[str, float]],
    continuity_jitter: float,
    slope_damping_time: Optional[float],
) -> xarray.Dataset:
    time_coords = jnp.asarray(params.coords["time"].values)
    t_end = float(time_coords[-1])

    if correlation_length is None:
        time_span = float(time_coords[-1] - time_coords[0])
        correlation_length = time_span / 4.0

    prediction_times = jnp.asarray(prediction_times)

    data_vars: dict = {}
    ensemble_coords = params.coords["ensemble"]

    t = time_coords
    t_mean = t.mean()
    t_centered = t - t_mean
    denom = jnp.sum(t_centered**2)

    for name in params.data_vars:
        if "time" not in params[name].dims:
            data_vars[name] = params[name]
            continue

        da = params[name].transpose("time", "ensemble")
        y_train = jnp.asarray(da.values)  # (N_t, N_e)

        # Per-member least-squares linear fit y = a + b*t
        y_means = y_train.mean(axis=0, keepdims=True)  # (1, N_e)
        b = jnp.sum(t_centered[:, None] * (y_train - y_means), axis=0) / denom
        a = y_means[0] - b * t_mean  # (N_e,)

        trend_train = a[None, :] + b[None, :] * t[:, None]  # (N_t, N_e)
        residual = y_train - trend_train

        # Heteroscedastic observation noise from cross-ensemble spread;
        # anchor the last point so extrapolation interpolates exactly
        # through each member's own end-of-window value.
        sigma_t = jnp.asarray(da.std(dim="ensemble").values)
        noise_var = jnp.maximum(sigma_t**2, continuity_jitter)
        noise_var = noise_var.at[-1].set(continuity_jitter)

        if kernel_amplitudes is not None and name in kernel_amplitudes:
            amplitude = float(kernel_amplitudes[name])
        else:
            amplitude = float(jnp.std(residual))
        amplitude = max(amplitude, 1e-6)

        def _predict(r):
            return _gp_predict(
                time_coords,
                r,
                prediction_times,
                correlation_length,
                amplitude,
                noise_var,
            )

        res_means, res_stds = jax.vmap(_predict, in_axes=1, out_axes=1)(residual)

        trend_pred = jax.vmap(
            lambda ae, be: _linear_trend_at(
                prediction_times, ae, be, t_end, slope_damping_time
            ),
            in_axes=(0, 0),
            out_axes=1,
        )(a, b)  # (N_p, N_e)

        means = trend_pred + res_means

        data_vars[name] = (("time", "ensemble"), means)
        if include_std:
            data_vars[f"{name}_std"] = (("time", "ensemble"), res_stds)

    return xarray.Dataset(
        data_vars=data_vars,
        coords={"time": prediction_times, "ensemble": ensemble_coords},
    )


# ---------------------------------------------------------------------------
# Method: ar1
# ---------------------------------------------------------------------------


def _extrapolate_ar1(
    params: xarray.Dataset,
    prediction_times: jnp.ndarray,
    phi_max: float,
) -> xarray.Dataset:
    prediction_times = jnp.asarray(prediction_times)

    data_vars: dict = {}
    ensemble_coords = params.coords["ensemble"]

    for name in params.data_vars:
        if "time" not in params[name].dims:
            data_vars[name] = params[name]
            continue

        da = params[name].transpose("time", "ensemble")
        y_train = jnp.asarray(da.values)  # (N_t, N_e)

        # Per-member AR(1) fit: y_{i+1} - μ = φ (y_i - μ) + ε
        mu = y_train.mean(axis=0)  # (N_e,)
        yc = y_train - mu[None, :]
        num = jnp.sum(yc[1:] * yc[:-1], axis=0)
        den = jnp.sum(yc[:-1] ** 2, axis=0)
        phi = jnp.where(den > 1e-12, num / jnp.maximum(den, 1e-12), 0.0)
        phi = jnp.clip(phi, -phi_max, phi_max)

        # prediction_times[0] equals the last training time, so k=0 gives
        # x_pred[0] = y_train[-1] exactly (continuity with the posterior).
        last = y_train[-1]  # (N_e,)
        k = jnp.arange(prediction_times.shape[0])  # (N_p,)
        powers = phi[None, :] ** k[:, None]  # (N_p, N_e)
        means = mu[None, :] + powers * (last - mu)[None, :]

        data_vars[name] = (("time", "ensemble"), means)

    return xarray.Dataset(
        data_vars=data_vars,
        coords={"time": prediction_times, "ensemble": ensemble_coords},
    )


# ---------------------------------------------------------------------------
# Method: ornstein_uhlenbeck
# ---------------------------------------------------------------------------


def _extrapolate_ornstein_uhlenbeck(
    params: xarray.Dataset,
    prediction_times: jnp.ndarray,
    rng_key: "jax.Array",
    phi_max: float,
) -> xarray.Dataset:
    prediction_times = jnp.asarray(prediction_times)
    n_pred = prediction_times.shape[0]

    data_vars: dict = {}
    ensemble_coords = params.coords["ensemble"]

    var_names = [n for n in params.data_vars if "time" in params[n].dims]
    var_keys = jax.random.split(rng_key, max(len(var_names), 1))

    passthrough = {n: params[n] for n in params.data_vars if "time" not in params[n].dims}
    data_vars.update(passthrough)

    for name, key in zip(var_names, var_keys):
        da = params[name].transpose("time", "ensemble")
        y_train = jnp.asarray(da.values)  # (N_t, N_e)
        n_t, n_e = y_train.shape

        x_prev = y_train[:-1, :]  # (N_t-1, N_e)
        x_next = y_train[1:, :]

        # Per-member OLS regression of x_next on x_prev:
        #   x_next = c_e + φ_e · x_prev + noise
        mean_prev = x_prev.mean(axis=0, keepdims=True)  # (1, N_e)
        mean_next = x_next.mean(axis=0, keepdims=True)
        cov = jnp.sum((x_prev - mean_prev) * (x_next - mean_next), axis=0)
        var_prev = jnp.sum((x_prev - mean_prev) ** 2, axis=0)
        phi = jnp.where(var_prev > 1e-12, cov / jnp.maximum(var_prev, 1e-12), 0.0)
        phi = jnp.clip(phi, -phi_max, phi_max)
        c = mean_next[0] - phi * mean_prev[0]  # (N_e,)

        # Pooled residual std across the whole ensemble of increments.
        residuals = x_next - (c[None, :] + phi[None, :] * x_prev)
        sigma = jnp.sqrt(jnp.mean(residuals**2))

        # Stochastic rollout.  prediction_times[0] == time_coords[-1] so
        # x_pred[0] = y_train[-1] gives exact per-member continuity.
        eps = jax.random.normal(key, (n_pred - 1, n_e))
        x0 = y_train[-1, :]

        def step(x_prev_state, eps_k):
            x_new = c + phi * x_prev_state + sigma * eps_k
            return x_new, x_new

        _, x_rest = jax.lax.scan(step, x0, eps)  # x_rest shape: (N_p-1, N_e)
        means = jnp.concatenate([x0[None, :], x_rest], axis=0)  # (N_p, N_e)

        data_vars[name] = (("time", "ensemble"), means)

    return xarray.Dataset(
        data_vars=data_vars,
        coords={"time": prediction_times, "ensemble": ensemble_coords},
    )


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def extrapolate_parameters(
    params: xarray.Dataset,
    prediction_times: jnp.ndarray,
    method: str = "linear_trend_gp",
    correlation_length: Optional[float] = None,
    include_std: bool = False,
    kernel_amplitudes: Optional[dict[str, float]] = None,
    continuity_jitter: float = 1e-6,
    slope_damping_time: Optional[float] = None,
    ar1_phi_max: float = 0.999,
    rng_key: Optional["jax.Array"] = None,
    ou_phi_max: float = 0.999,
) -> xarray.Dataset:
    """Extrapolate time-varying parameters from one window to the next.

    Args:
        params: Posterior parameters with dims ``(time, ensemble)``.
            Variables without a ``time`` dimension pass through unchanged.
        prediction_times: 1-D array of time coordinates to predict at.
            By convention ``prediction_times[0] == params.time[-1]`` so
            the first predicted value matches each member's end-of-window
            posterior exactly (continuity between windows).
        method: ``"linear_trend_gp"`` or ``"ar1"``.
        correlation_length: RBF length scale for ``"linear_trend_gp"``.
            Defaults to one quarter of the training time span.
        include_std: If ``True``, ``"linear_trend_gp"`` also returns the
            GP-residual predictive std as ``{name}_std``.  Ignored by
            ``"ar1"``.
        kernel_amplitudes: Optional per-variable signal std override for
            ``"linear_trend_gp"``; defaults to the ensemble std of the
            detrended residual.
        continuity_jitter: Noise floor at training points in
            ``"linear_trend_gp"``; also the forced noise at the final
            training point so the GP interpolates exactly through it.
        slope_damping_time: Exponential decay time-scale for the linear
            trend's slope past ``t_end`` in ``"linear_trend_gp"``.
            ``None`` (default) means no damping — plain linear
            extrapolation.
        ar1_phi_max: Stationarity clip for ``"ar1"``.  The estimated AR
            coefficient is clipped to ``(-phi_max, phi_max)``.
        rng_key: JAX PRNG key for stochastic methods.  Required for
            ``"ornstein_uhlenbeck"``; ignored by the other methods.
        ou_phi_max: Stationarity clip for ``"ornstein_uhlenbeck"``.
    """
    if method == "linear_trend_gp":
        return _extrapolate_linear_trend_gp(
            params,
            prediction_times,
            correlation_length,
            include_std,
            kernel_amplitudes,
            continuity_jitter,
            slope_damping_time,
        )
    if method == "ar1":
        return _extrapolate_ar1(params, prediction_times, ar1_phi_max)
    if method == "ornstein_uhlenbeck":
        if rng_key is None:
            raise ValueError(
                "extrapolate_parameters(method='ornstein_uhlenbeck') requires "
                "rng_key for the stochastic Brownian increments."
            )
        return _extrapolate_ornstein_uhlenbeck(
            params, prediction_times, rng_key, ou_phi_max
        )
    raise ValueError(
        f"Unknown extrapolation method: {method!r}. "
        "Expected 'linear_trend_gp', 'ar1', or 'ornstein_uhlenbeck'."
    )
