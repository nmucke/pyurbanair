"""Metric definitions (spec §3).

All field metrics operate on standardized |U| DataArrays with dims
``(time, z, y, x)`` already interpolated onto the common (truth) grid, with an
optional boolean ``mask`` of solid cells to exclude.
"""

from __future__ import annotations

import numpy as np
import xarray as xr


# ---------------------------------------------------------------------------
# Field metrics
# ---------------------------------------------------------------------------
def field_rmse_timeseries(
    model: xr.DataArray, truth: xr.DataArray, mask: np.ndarray | None = None
) -> np.ndarray:
    """Per-time RMSE of |U| over fluid cells. Inputs share grid & time axis."""
    m = np.asarray(model.values, dtype=float)
    t = np.asarray(truth.values, dtype=float)
    diff = m - t
    if mask is not None:
        solid = np.broadcast_to(mask, diff.shape)
        diff = np.where(solid, np.nan, diff)
    axes = tuple(range(1, diff.ndim))  # all but time
    return np.sqrt(np.nanmean(diff**2, axis=axes))


def field_rmse(
    model: xr.DataArray, truth: xr.DataArray, mask: np.ndarray | None = None
) -> float:
    """Horizon-mean |U| field RMSE over fluid cells."""
    return float(np.nanmean(field_rmse_timeseries(model, truth, mask)))


def normalized_field_rmse(
    model: xr.DataArray, truth: xr.DataArray, mask: np.ndarray | None = None
) -> float:
    """Field RMSE normalized by the truth |U| RMS (fluid cells)."""
    t = np.asarray(truth.values, dtype=float)
    if mask is not None:
        t = np.where(np.broadcast_to(mask, t.shape), np.nan, t)
    rms = float(np.sqrt(np.nanmean(t**2)))
    if rms == 0 or not np.isfinite(rms):
        return float("nan")
    return field_rmse(model, truth, mask) / rms


# ---------------------------------------------------------------------------
# Parameter metrics (posterior-mean trajectory vs truth)
# ---------------------------------------------------------------------------
def _truth_on(post_da: xr.DataArray, true_da: xr.DataArray) -> np.ndarray:
    """Interpolate truth param onto the posterior time axis."""
    tp = np.asarray(post_da["time"].values, dtype=float)
    if "time" in true_da.dims:
        tt = np.asarray(true_da["time"].values, dtype=float)
        return np.interp(tp, tt, np.asarray(true_da.values, dtype=float))
    return np.full_like(tp, float(true_da.values))


def param_metrics(
    post: xr.Dataset, true: xr.Dataset, param: str, prior: xr.Dataset | None = None
) -> dict:
    """RMSE, bias, prior->posterior reduction, spread, and ±1σ/±2σ coverage."""
    if param not in post.data_vars or param not in true.data_vars:
        return {}
    pda = post[param].transpose("ensemble", "time")
    members = np.asarray(pda.values, dtype=float)  # (ens, time)
    mean = members.mean(0)
    std = members.std(0)
    truth = _truth_on(post[param], true[param])

    err = mean - truth
    rmse = float(np.sqrt(np.mean(err**2)))
    bias = float(np.mean(err))
    spread = float(np.mean(std))

    cov1 = float(np.mean(np.abs(truth - mean) <= std))
    cov2 = float(np.mean(np.abs(truth - mean) <= 2 * std))

    out = dict(rmse=rmse, bias=bias, spread=spread, coverage1=cov1, coverage2=cov2)

    if prior is not None and param in prior.data_vars:
        pri = prior[param].transpose("ensemble", "time")
        pmean = np.asarray(pri.values, dtype=float).mean(0)
        # truth onto prior time
        ptruth = _truth_on(prior[param], true[param])
        prmse = float(np.sqrt(np.mean((pmean - ptruth) ** 2)))
        out["prior_rmse"] = prmse
        out["reduction"] = (1.0 - rmse / prmse) if prmse > 0 else float("nan")
    return out


# ---------------------------------------------------------------------------
# Sensor metrics
# ---------------------------------------------------------------------------
def sensor_rmse(model_ts: np.ndarray, truth_ts: np.ndarray) -> float:
    """RMSE over (sensor, time) of a sampled scalar (e.g. |U|)."""
    return float(np.sqrt(np.nanmean((model_ts - truth_ts) ** 2)))


def spread_skill(spread_ts: np.ndarray, rmse_ts: np.ndarray, n_members: int) -> float:
    """Finite-ensemble-corrected RMS spread / RMS error (calibration ~ 1)."""
    if n_members < 1:
        raise ValueError(f"n_members must be positive, got {n_members}")
    num = float(np.sqrt(np.nanmean(np.asarray(spread_ts, dtype=float) ** 2)))
    den = float(np.sqrt(np.nanmean(np.asarray(rmse_ts, dtype=float) ** 2)))
    if den <= 0 or not np.isfinite(den):
        return float("nan")
    return float(np.sqrt((n_members + 1) / n_members)) * num / den
