"""Probabilistic ensemble scores and the metric bundles built on them.

Fair CRPS/CRPSS, energy score, z-scores, ranks, spread--skill, the hit rate
``q``, and the parameter / sensor bundles that assemble them for
``run_summary.yaml``.

From WP1.1 the pairwise estimators divide by ``M(M-1)`` rather than ``M**2``,
because the biased form's optimum is a collapsed ensemble -- the exact failure
these scores exist to detect. WP0.2 moves the biased forms in unchanged, so
until WP1.1 lands what is here is the old estimator. Formulas in
``docs/plans/esmda_turbulence_evaluation.md`` §3--§6; rollout in
``phase1_metrics_and_figures.md``.

Populated in WP0.2 (move), extended in phase 1.
"""

# mypy: ignore-errors
# Moved wholesale in WP0.2 from ``src/pyurbanair/utils/da_metrics.py``,
# ``scripts/figspec/metrics.py``, ``src/pyurbanair/plotting.py`` and
# ``scripts/esmda/_esmda_common.py`` -- largely unannotated code predating the
# strict mypy config. Waived rather than annotated as part of a pure refactor;
# dropping the waiver is later cleanup.

from __future__ import annotations

import numpy as np

# Both spellings: the moved sources disagreed (``figspec.metrics`` used ``xr``,
# ``plotting``/``_esmda_common`` used ``xarray``) and WP0.2 keeps every moved
# signature verbatim. Collapse to one alias in a later cleanup.
import xarray
import xarray as xr

# ---------------------------------------------------------------------------
# Per-knot diagnostic skill metrics for time-varying parameter assimilation.
#
# These operate on numpy arrays of shape ``(ensemble, time)`` and a truth array
# of shape ``(time,)``. They are intentionally pure-numpy (no JAX) so they can
# be applied to ``xarray.Dataset`` outputs after an ESMDA run.
# ---------------------------------------------------------------------------


def per_knot_error(ens: np.ndarray, truth: np.ndarray) -> np.ndarray:
    """Per-knot magnitude of (ensemble mean - truth)."""
    return np.abs(ens.mean(axis=0) - truth)


def per_knot_spread(ens: np.ndarray) -> np.ndarray:
    """Per-knot ensemble standard deviation (ddof=1)."""
    if ens.shape[0] < 2:
        return np.zeros(ens.shape[1])
    return ens.std(axis=0, ddof=1)


def crps_ensemble(ens: np.ndarray, truth: np.ndarray) -> np.ndarray:
    """Per-knot sample CRPS using the energy-form estimator.

    ``CRPS(F, y) = E|X - y| - 0.5 * E|X - X'|`` where ``X, X'`` are
    independent draws from ``F``. With a finite ensemble of size ``N`` the
    pairwise term is computed as the mean of ``|x_i - x_j|`` over all
    ``(i, j)`` pairs. ``ens`` is ``(n_ensemble, n_x)`` and ``truth`` is
    ``(n_x,)``; one score is returned per ``x`` location (lower is better, in
    the units of the scored quantity).

    Dtype is the *caller's* policy: nothing here casts, so a float32 ensemble
    is scored in float32 (the pairwise term then differs from the float64
    result at the ~1e-7 level). This matters because the two functions merged
    here disagreed on exactly that point -- ``da_metrics.per_knot_crps`` did
    not cast, ``plotting._crps_ensemble`` upcast to float64 -- so callers that
    need the upcast now do it explicitly.
    """
    n = ens.shape[0]
    term1 = np.mean(np.abs(ens - truth[None, :]), axis=0)
    if n < 2:
        return term1
    diffs = np.abs(ens[:, None, :] - ens[None, :, :])
    term2 = 0.5 * diffs.mean(axis=(0, 1))
    return term1 - term2


def per_knot_in_band(
    ens: np.ndarray, truth: np.ndarray, alpha: float = 0.9
) -> np.ndarray:
    """Boolean per-knot indicator: truth in central ``alpha`` ensemble band."""
    lo = np.quantile(ens, 0.5 - alpha / 2.0, axis=0)
    hi = np.quantile(ens, 0.5 + alpha / 2.0, axis=0)
    return (truth >= lo) & (truth <= hi)


def summary_scalars(
    ens: np.ndarray, truth: np.ndarray, alpha: float = 0.9
) -> dict[str, float]:
    """Time-averaged skill scalars for one parameter at one ESMDA step."""
    err = per_knot_error(ens, truth)
    spr = per_knot_spread(ens)
    crps = crps_ensemble(ens, truth)
    band = per_knot_in_band(ens, truth, alpha=alpha)
    return {
        "time_avg_error": float(np.sqrt(np.mean(err**2))),
        "time_avg_spread": float(np.mean(spr)),
        "mean_crps": float(np.mean(crps)),
        "coverage": float(np.mean(band)),
    }


# ---------------------------------------------------------------------------
# Sweep-figure field metrics (``docs/figure_specs.md`` §3).
#
# All field metrics operate on standardized |U| DataArrays with dims
# ``(time, z, y, x)`` already interpolated onto the common (truth) grid, with an
# optional boolean ``mask`` of solid cells to exclude.
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


def spread_skill(spread_ts: np.ndarray, rmse_ts: np.ndarray) -> float:
    """Mean spread / mean RMSE (calibration ~ 1)."""
    num = float(np.nanmean(spread_ts))
    den = float(np.nanmean(rmse_ts))
    return num / den if den > 0 else float("nan")


# ---------------------------------------------------------------------------
# Per-knot ensemble parameter / sensor bundles (the numbers the figures draw).
#
# These are a *different* estimator family from the mean-trajectory helpers
# above (per-knot member RMSE / CRPS vs mean-trajectory RMSE with ddof=0) with
# different callers; unifying them is deferred to phase 1.
# ---------------------------------------------------------------------------

# Parameters drawn by the parameter figures, in panel order. The two inflow
# drivers first, then the (constant-in-time) model-error knobs
# (docs/esmda_model_error_parameters.md). Only those actually present in the
# posterior are plotted, so single-model / inflow-only runs are unchanged.
_PLOTTED_PARAMS = (
    "inflow_angle",
    "velocity_magnitude",
    "vertical_inflow_exponent",
    "sgs_constant",
)


def _plotted_param_names(
    esmda_params: xarray.Dataset,
    true_params: xarray.Dataset | None = None,
) -> list[str]:
    """Ordered estimable parameters present in ``esmda_params`` (and the truth)."""
    names = [p for p in _PLOTTED_PARAMS if p in esmda_params.data_vars]
    if true_params is not None:
        names = [p for p in names if p in true_params.data_vars]
    return names


def _param_members_and_x(da: xarray.DataArray):
    """Return ``(x, members)`` for a parameter, members shaped ``(n_ensemble, n_x)``.

    ``x`` is the ``time`` coordinate when the parameter is time-varying and
    carries one, otherwise a plain index (e.g. one point per assimilation window
    for a static parameter stacked across windows).
    """
    da = da.transpose("ensemble", ...)
    members = np.asarray(da.values).reshape(da.sizes["ensemble"], -1)
    non_ens = [d for d in da.dims if d != "ensemble"]
    if non_ens == ["time"] and "time" in da.coords:
        x = np.asarray(da["time"].values, dtype=float)
    else:
        x = np.arange(members.shape[1])
    return x, members


def compute_parameter_metrics(
    esmda_params: xarray.Dataset,
    true_params: xarray.Dataset,
    prior_params: xarray.Dataset | None = None,
) -> dict[str, dict[str, np.ndarray]]:
    """Per-parameter posterior error series (RMSE & CRPS) of the ensemble vs truth.

    Returns ``{param: {"x", "rmse", "crps", ["prior_rmse"]}}`` with one value per
    posterior x-location (``time`` for a time-varying parameter, else one per
    assimilation window). Both measures reduce over the ensemble:

      * **RMSE** -- ``sqrt(mean_i (x_i - y)**2)`` over members ``x_i`` about the
        truth ``y`` (deterministic accuracy; captures bias and spread together).
      * **CRPS** -- empirical continuous ranked probability score of the ensemble
        against the truth (probabilistic skill; rewards a sharp, calibrated
        ensemble). Same units as the parameter.

    The truth is interpolated onto the posterior's x-axis when the two are
    sampled differently, so static (single-value) and time-varying parameters
    are handled uniformly. ``prior_params`` (if given and sampled on the same
    x-grid) adds the prior's RMSE for an improvement reference. These are the
    same numbers :func:`evaluation.figures.plot_parameter_error` draws.
    """
    metrics: dict[str, dict[str, np.ndarray]] = {}
    for param_name in _plotted_param_names(esmda_params, true_params):
        x_est, members = _param_members_and_x(esmda_params[param_name])

        true_da = true_params[param_name]
        if "ensemble" in true_da.dims:
            true_da = true_da.isel(ensemble=0)
        x_true, true_members = _param_members_and_x(true_da.expand_dims("ensemble"))
        truth = true_members[0]

        # Align the truth onto the posterior's x-axis: a static truth (single
        # point) becomes a constant; a differently-sampled time-varying truth is
        # linearly interpolated.
        order = np.argsort(x_true)
        truth_on_est = np.interp(x_est, np.asarray(x_true)[order], truth[order])

        entry: dict[str, np.ndarray] = {
            "x": x_est,
            "rmse": np.sqrt(np.mean((members - truth_on_est[None, :]) ** 2, axis=0)),
            # The parameter artifacts are float32 on disk; the CRPS here has
            # always been scored in float64 (see crps_ensemble on the
            # caller-owned dtype policy), so cast explicitly.
            "crps": crps_ensemble(np.asarray(members, dtype=float), truth_on_est),
        }
        if prior_params is not None and param_name in prior_params.data_vars:
            _, prior_members = _param_members_and_x(prior_params[param_name])
            if prior_members.shape[1] == truth_on_est.shape[0]:
                entry["prior_rmse"] = np.sqrt(
                    np.mean((prior_members - truth_on_est[None, :]) ** 2, axis=0)
                )
        metrics[param_name] = entry
    return metrics


def compute_sensor_metrics(
    true_sensor: xarray.DataArray,
    ensemble_sensor: xarray.DataArray,
) -> dict[str, np.ndarray]:
    """True vs ensemble |U| at sensors plus per-time RMSE/CRPS over the sensors.

    Returns ``{"time", "members", "ens_mean", "truth", "rmse", "crps"}`` where
    ``members`` is ``(ensemble, time, sensor)``, ``ens_mean``/``truth`` are
    ``(time, sensor)`` and ``rmse``/``crps`` are ``(time,)`` (reduced over the
    sensors). The truth is linearly interpolated onto the ensemble's time axis
    per sensor when the two are sampled differently. These are the same numbers
    :func:`evaluation.figures.plot_sensor_timeseries` draws.

      * **RMSE** -- ``sqrt(mean_s (mean_ens - truth)**2)`` of the ensemble mean
        about the truth (deterministic accuracy).
      * **CRPS** -- mean over sensors of the empirical continuous ranked
        probability score of the ensemble against the truth (probabilistic
        skill), in |U| units.
    """
    ens = ensemble_sensor.transpose("ensemble", "time", "sensor")
    members = np.asarray(ens.values, dtype=float)  # (E, T, S)
    t_ens = np.asarray(ens["time"].values, dtype=float)

    true_da = true_sensor.transpose("time", "sensor")
    truth_raw = np.asarray(true_da.values, dtype=float)  # (Tt, S)
    t_true = np.asarray(true_da["time"].values, dtype=float)

    n_sensors = members.shape[2]

    # Align the truth onto the ensemble time axis (per sensor) so differing
    # cadences/lengths between truth and assimilation grids still line up.
    order = np.argsort(t_true)
    truth = np.column_stack(
        [np.interp(t_ens, t_true[order], truth_raw[order, s]) for s in range(n_sensors)]
    )  # (T, S)

    ens_mean = members.mean(axis=0)  # (T, S)
    n_time = ens_mean.shape[0]
    rmse = np.sqrt(np.mean((ens_mean - truth) ** 2, axis=1))  # (T,)
    # ``members`` is already float64 here, so no cast is needed (unlike the
    # parameter bundle above).
    crps = np.array(
        [
            float(np.mean(crps_ensemble(members[:, t, :], truth[t, :])))
            for t in range(n_time)
        ]
    )  # (T,)

    return {
        "time": t_ens,
        "members": members,
        "ens_mean": ens_mean,
        "truth": truth,
        "rmse": rmse,
        "crps": crps,
    }


# ---------------------------------------------------------------------------
# Vector (u,v,w) sensor error metrics
# ---------------------------------------------------------------------------


def _energy_score(members, truth):
    """Per-timestep energy score, averaged over sensors.

    The energy score is the multivariate generalization of the CRPS (Gneiting &
    Raftery 2007): for a vector forecast ensemble ``{v_m}`` and truth ``v``,

        ES = mean_m ||v_m - v|| - 0.5 * mean_{m,m'} ||v_m - v_{m'}||,

    which reduces to the CRPS in 1-D. It rewards both accuracy (term 1) and a
    calibrated spread (term 2), in the same |U| units as the velocity.

    Args:
        members: ``(component, ensemble, time, sensor)`` aligned member vectors.
        truth: ``(component, time, sensor)`` aligned truth vectors.

    Returns:
        ``(time,)`` energy score, averaged over the sensors (matching the
        per-time, over-sensors reduction of ``compute_sensor_metrics``).
    """
    n_time = members.shape[2]
    es = np.empty(n_time)
    # Loop over time so the pairwise term never materializes more than
    # ``(component, ensemble, ensemble, sensor)`` at once.
    for t in range(n_time):
        m = members[:, :, t, :]  # (C, E, S)
        v = truth[:, t, :]  # (C, S)
        d_truth = np.sqrt(np.sum((m - v[:, None, :]) ** 2, axis=0))  # (E, S)
        term1 = d_truth.mean(axis=0)  # (S,)
        diff = m[:, :, None, :] - m[:, None, :, :]  # (C, E, E, S)
        d_pair = np.sqrt(np.sum(diff**2, axis=0))  # (E, E, S)
        term2 = 0.5 * d_pair.mean(axis=(0, 1))  # (S,)
        es[t] = float((term1 - term2).mean())  # average over sensors
    return es


def vector_sensor_metrics(truth_comp, ensemble_comp):
    """Full-vector ``(u, v, w)`` sensor error, reduced over sensors per timestep.

    One scalar per sensor per timestep is formed from the whole velocity vector
    (not just its magnitude), then reduced over sensors:

      * ``rmse(t) = sqrt(mean_s || <v>_ens - v_truth ||^2)`` -- the ensemble-mean
        vector error, obtained by combining the per-component
        :func:`compute_sensor_metrics` RMSEs as ``sqrt(sum_c rmse_c**2)`` (so the
        shared, time-aligning metric is reused unchanged).
      * ``energy_score(t)`` -- the multivariate CRPS (:func:`_energy_score`) over
        the aligned member/truth vectors.

    Args:
        truth_comp: ``(component, time, sensor)`` truth series.
        ensemble_comp: ``(component, ensemble, time, sensor)`` ensemble series.

    Returns:
        ``{"rmse": (T,), "energy_score": (T,)}``.
    """
    components = [str(c) for c in np.asarray(truth_comp["component"].values)]
    # Per-component metrics reuse the shared compute_sensor_metrics, which also
    # time-aligns the truth onto the ensemble axis identically for each
    # component, so the returned members/truth stack into consistent vectors.
    per = {
        c: compute_sensor_metrics(
            truth_comp.sel(component=c), ensemble_comp.sel(component=c)
        )
        for c in components
    }
    rmse = np.sqrt(np.sum([per[c]["rmse"] ** 2 for c in components], axis=0))  # (T,)
    members = np.stack([per[c]["members"] for c in components], axis=0)  # (C,E,T,S)
    truth = np.stack([per[c]["truth"] for c in components], axis=0)  # (C,T,S)
    return {"rmse": rmse, "energy_score": _energy_score(members, truth)}


# ---------------------------------------------------------------------------
# Series reductions for the run summary
# ---------------------------------------------------------------------------


def series_stats(arr):
    """{mean, final, max, min} of a 1-D series, or ``None`` if it has no values.

    ``final`` is the last element (the end-of-rollout value); the rest reduce
    over the whole series. NaNs are ignored.
    """
    a = np.asarray(arr, dtype=float).ravel()
    finite = a[np.isfinite(a)]
    if finite.size == 0:
        return None
    return {
        "mean": float(finite.mean()),
        "final": float(a[-1]) if np.isfinite(a[-1]) else None,
        "max": float(finite.max()),
        "min": float(finite.min()),
    }


def parameter_metric_summary(posterior_params, true_params, prior_params):
    """Per-parameter RMSE/CRPS summary stats (posterior, with a prior reference)."""
    metrics = compute_parameter_metrics(posterior_params, true_params, prior_params)
    summary = {}
    for name, m in metrics.items():
        entry = {"rmse": series_stats(m["rmse"]), "crps": series_stats(m["crps"])}
        if "prior_rmse" in m:
            prior_mean = float(np.nanmean(m["prior_rmse"]))
            post_mean = float(np.nanmean(m["rmse"]))
            entry["prior_rmse_mean"] = prior_mean
            entry["rmse_reduction_vs_prior"] = (
                float(1.0 - post_mean / prior_mean) if prior_mean > 0 else None
            )
        summary[name] = entry
    return summary
