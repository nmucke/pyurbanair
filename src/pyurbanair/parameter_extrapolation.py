"""Gaussian process extrapolation of time-varying parameters."""

from typing import Optional

import jax
import jax.numpy as jnp
import xarray


def _rbf_kernel(
    x1: jnp.ndarray,
    x2: jnp.ndarray,
    correlation_length: float,
) -> jnp.ndarray:
    """Squared-exponential (RBF) kernel matrix.

    Args:
        x1: 1-D array of shape ``(n,)``.
        x2: 1-D array of shape ``(m,)``.
        correlation_length: Kernel length scale.

    Returns:
        Kernel matrix of shape ``(n, m)``.
    """
    dt = x1[:, None] - x2[None, :]
    return jnp.exp(-0.5 * (dt / jnp.maximum(correlation_length, 1e-6)) ** 2)


def _gp_predict(
    x_train: jnp.ndarray,
    y_train: jnp.ndarray,
    x_pred: jnp.ndarray,
    correlation_length: float,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Predict at *x_pred* given observations at *x_train*.

    Args:
        x_train: Training time coordinates, shape ``(N_t,)``.
        y_train: Training values, shape ``(N_t,)``.
        x_pred: Prediction time coordinates, shape ``(N_p,)``.
        correlation_length: RBF kernel length scale.

    Returns:
        Tuple of (predictive_mean, predictive_std), each shape ``(N_p,)``.
    """
    K_tt = _rbf_kernel(x_train, x_train, correlation_length)
    K_tt = K_tt + 1e-6 * jnp.eye(K_tt.shape[0])

    K_pt = _rbf_kernel(x_pred, x_train, correlation_length)
    K_pp = _rbf_kernel(x_pred, x_pred, correlation_length)

    L = jnp.linalg.cholesky(K_tt)
    alpha = jax.scipy.linalg.cho_solve((L, True), y_train)
    V = jax.scipy.linalg.solve_triangular(L, K_pt.T, lower=True)

    mean = K_pt @ alpha
    var = jnp.diag(K_pp) - jnp.sum(V**2, axis=0)
    std = jnp.sqrt(jnp.maximum(var, 0.0))

    return mean, std


def extrapolate_parameters(
    params: xarray.Dataset,
    prediction_times: jnp.ndarray,
    correlation_length: Optional[float] = None,
    include_std: bool = False,
) -> xarray.Dataset:
    """Extrapolate time-varying parameters using Gaussian process regression.

    Fits an independent GP (squared-exponential kernel) to each ensemble
    member's trajectory and predicts at ``prediction_times``.

    Args:
        params: Posterior parameters with dims ``(time, ensemble)``.
            Variables without a ``time`` dimension are passed through
            unchanged.
        prediction_times: 1-D array of time coordinates to predict at.
        correlation_length: RBF kernel length scale in the same units as
            ``params.coords["time"]``.  Defaults to one quarter of the
            training time span.
        include_std: If ``True``, include predictive standard deviations
            as additional ``{name}_std`` variables.

    Returns:
        ``xarray.Dataset`` with dims ``(time, ensemble)`` where
        ``time`` corresponds to *prediction_times*.
    """
    time_coords = jnp.asarray(params.coords["time"].values)

    if correlation_length is None:
        time_span = float(time_coords[-1] - time_coords[0])
        correlation_length = time_span / 4.0

    prediction_times = jnp.asarray(prediction_times)

    # Vectorise GP prediction over ensemble members.
    def _predict_ensemble(y_train_ensemble: jnp.ndarray):
        """y_train_ensemble: shape (N_t, N_e)."""
        def _predict_single(y: jnp.ndarray):
            return _gp_predict(time_coords, y, prediction_times, correlation_length)
        means, stds = jax.vmap(_predict_single, in_axes=1, out_axes=1)(
            y_train_ensemble,
        )
        return means, stds

    data_vars: dict = {}
    ensemble_coords = params.coords["ensemble"]

    for name in params.data_vars:
        if "time" not in params[name].dims:
            data_vars[name] = params[name]
            continue

        da = params[name].transpose("time", "ensemble")
        y_train = jnp.asarray(da.values)  # (N_t, N_e)

        # Subtract per-trajectory mean so the zero-mean GP extrapolates
        # back toward each member's own level instead of toward 0.
        y_mean = y_train.mean(axis=0, keepdims=True)
        means, stds = _predict_ensemble(y_train - y_mean)
        means = means + y_mean

        data_vars[name] = (("time", "ensemble"), means)
        if include_std:
            data_vars[f"{name}_std"] = (("time", "ensemble"), stds)

    return xarray.Dataset(
        data_vars=data_vars,
        coords={"time": prediction_times, "ensemble": ensemble_coords},
    )
