"""Probabilistic ensemble scores and the metric bundles built on them.

Here today: the CRPS and energy-score estimators, per-knot skill metrics, the
sweep-figure field/parameter/sensor metrics, and the parameter / sensor bundles
that assemble them for ``run_summary.yaml`` (moved in WP0.2); CRPSS and
spread--skill (WP1.1); the §3 parameter calibration bundle -- z-score,
normalized error, contraction ratio (WP1.2); ranks and the §4.2 window-statistic
summary (WP1.3); the §4.1 mean-field hit rate ``q`` (WP1.4); the §5 normalized
data mismatch ``O_N`` (WP2.2).

Since WP1.1 the pairwise estimators divide by ``M(M-1)`` rather than ``M**2``,
because the biased form's optimum is a collapsed ensemble -- the exact failure
these scores exist to detect. Numbers produced before that change are ~O(1/M)
larger and are **not** comparable with current ones; ``metrics_version: 2`` in
``run_summary.yaml`` marks the boundary. Formulas in
``docs/plans/esmda_turbulence_evaluation.md`` §3--§6; rollout in
``phase1_metrics_and_figures.md``.

Populated in WP0.2 (move), extended through phase 1.
"""

# mypy: ignore-errors
# Moved wholesale in WP0.2 from ``src/pyurbanair/utils/da_metrics.py``,
# ``scripts/figspec/metrics.py``, ``src/pyurbanair/plotting.py`` and
# ``scripts/esmda/_esmda_common.py`` -- largely unannotated code predating the
# strict mypy config. Waived rather than annotated as part of a pure refactor;
# dropping the waiver is later cleanup.

from __future__ import annotations

import logging
import warnings

import numpy as np
import xarray as xr

logger = logging.getLogger(__name__)

# Estimator-semantics marker written to ``run_summary.yaml`` by the metric
# stages. Bump it whenever an existing summary key changes meaning (the keys
# themselves are additive only), so numbers from either side of the change are
# never silently compared. 1 (or an absent marker) = the pre-WP1.1 biased
# ``M**2`` pairwise scores and mean-of-stds spread; 2 = the fair ``M(M-1)``
# estimators and root-mean-variance spread in this module.
METRICS_VERSION = 2

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
    """Per-knot *fair* sample CRPS using the energy-form estimator.

    ``CRPS(F, y) = E|X - y| - 0.5 * E|X - X'|`` where ``X, X'`` are
    independent draws from ``F``. With a finite ensemble of size ``N`` the
    pairwise term averages ``|x_i - x_j|`` over the ``N(N - 1)`` *ordered
    off-diagonal* pairs (Ferro 2014), not over all ``N**2`` pairs: the
    ``N**2`` form silently includes the ``N`` zero diagonal entries, which
    shrinks the spread term and so rewards an under-dispersed ensemble --
    the exact failure this score is used to detect. ``ens`` is
    ``(n_ensemble, n_x)`` and ``truth`` is ``(n_x,)``; one score is returned
    per ``x`` location (lower is better, in the units of the scored
    quantity).

    A single member has no pairwise term and the score degenerates to the
    absolute error.

    *Floating* dtype is the caller's policy: a float32 ensemble against a
    float32 truth is scored in float32 (differing from the float64 result at
    the ~1e-7 level), and a float32 ensemble against a float64 truth promotes
    to float64 as it always did. This matters because the two functions merged
    here disagreed on exactly that point -- ``da_metrics.per_knot_crps`` did
    not cast, ``plotting._crps_ensemble`` upcast to float64 -- so callers that
    need the upcast now do it explicitly. Integer and float16 inputs are the
    one exception: both terms are *differences*, which wrap or overflow in a
    narrow or unsigned dtype, so they are promoted to a signed float.
    """
    n = ens.shape[0]
    # Both terms are differences, so they are formed in a signed floating dtype
    # -- never in the inputs' own dtype, which for an unsigned or narrow integer
    # ensemble would wrap or overflow silently. Floating inputs are unaffected:
    # float32 in stays float32 (see the dtype contract above), and a float32
    # ensemble against a float64 truth still promotes as it always did.
    acc = np.result_type(ens.dtype, np.asarray(truth).dtype, np.float32)
    ens = ens.astype(acc, copy=False)
    truth = np.asarray(truth).astype(acc, copy=False)

    term1 = np.mean(np.abs(ens - truth[None, :]), axis=0)
    if n < 2:
        return term1
    # For sorted scalar samples x_(0) <= ... <= x_(n-1),
    #     sum_{i,j} |x_i - x_j| = 2 * sum_i (2i - n + 1) * x_(i),
    # so the half-sum below is exactly ``0.5 * sum_{i != j} |x_i - x_j|``
    # divided by ``n(n-1)`` -- the diagonal contributes nothing to either form.
    # Algebraically identical to averaging the pairwise-difference tensor, but
    # O(n log n) instead of allocating an ``(n, n, n_x)`` array (this is called
    # per timestep on sensor ensembles).
    ordered = np.sort(ens, axis=0)
    # The weights sum to zero, so subtracting any per-column constant leaves the
    # result unchanged in exact arithmetic -- but not in floating point: without
    # it the sum cancels catastrophically for data far from the origin (e.g. an
    # inflow angle near 270 deg in float32, where the error grows ~1e4x).
    # Centering on the median restores the accuracy of the pairwise form.
    centered = ordered - ordered[n // 2]
    weights = (2 * np.arange(n) - n + 1).reshape((n,) + (1,) * (centered.ndim - 1))
    term2 = np.sum(weights.astype(acc, copy=False) * centered, axis=0) / (n * (n - 1))
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
    """Time-averaged skill scalars for one parameter at one ESMDA step.

    ``time_avg_spread`` is the root of the *mean variance*, not the mean of the
    per-knot standard deviations: averaging stds is biased low by Jensen, so
    the old form made every ensemble look more confident than it was and is not
    comparable with ``time_avg_error`` (which is already an RMS).
    """
    err = per_knot_error(ens, truth)
    spr = per_knot_spread(ens)
    crps = crps_ensemble(ens, truth)
    band = per_knot_in_band(ens, truth, alpha=alpha)
    return {
        "time_avg_error": float(np.sqrt(np.mean(err**2))),
        "time_avg_spread": float(np.sqrt(np.mean(spr**2))),
        "mean_crps": float(np.mean(crps)),
        "coverage": float(np.mean(band)),
    }


def ensemble_uniqueness(members: np.ndarray) -> dict:
    """Duplicate-member diagnostics for one ensemble.

    A resampling policy that replaces a diverged member with a copy of a
    surviving one (pypalm does this) leaves an ensemble whose nominal size ``M``
    overstates its degrees of freedom: the pairwise ``M(M-1)`` corrections in
    the fair scores above are then too generous. Counting the duplicates is
    near-free, so it is reported per run and per window.

    Args:
        members: ``(n_members, n_features)`` flattened member vectors.

    Returns:
        ``{"n_members", "n_unique", "min_pairwise", "median_pairwise"}``.
        ``n_unique`` is an **exact**, bitwise row count -- resampled clones are
        bit-identical, while legitimately close members must stay distinct;
        judging "how close is too close" is what the pairwise L2 distances are
        for. The distances are ``None`` below two members and whenever every
        pair is non-finite; pairs involving a diverged (NaN) member are dropped
        rather than poisoning the reduction. Bitwise matching is why ``0.0`` and
        ``-0.0`` count as two members at distance zero -- numerically equal, but
        not the same bytes, so not the clone this is looking for.
    """
    values = np.asarray(members)
    if values.ndim != 2:
        raise ValueError(
            "members must have shape (n_members, n_features); "
            f"got array with shape {values.shape}"
        )

    n_members = int(values.shape[0])
    # Compare raw bytes, not values: a diverged member is a row of NaNs, and
    # ``np.unique`` would call two identical such rows distinct (NaN != NaN) --
    # losing the signal for exactly the ensemble that most needs flagging.
    rows = np.ascontiguousarray(values)
    row_bytes = np.dtype((np.void, rows.dtype.itemsize * rows.shape[1]))
    n_unique = int(np.unique(rows.view(row_bytes)).size)
    if n_members < 2:
        return {
            "n_members": n_members,
            "n_unique": n_unique,
            "min_pairwise": None,
            "median_pairwise": None,
        }

    # Row-by-row rather than an (M, M, K) difference tensor: M is the ensemble
    # size (tens), K is every parameter knot of every window concatenated, and
    # the tensor reaches hundreds of MB on a long run.
    as_float = values.astype(float, copy=False)
    pairwise = np.array(
        [
            np.linalg.norm(as_float[i] - as_float[j])
            for i in range(n_members)
            for j in range(i + 1, n_members)
        ]
    )
    finite = pairwise[np.isfinite(pairwise)]
    return {
        "n_members": n_members,
        "n_unique": n_unique,
        "min_pairwise": float(finite.min()) if finite.size else None,
        "median_pairwise": float(np.median(finite)) if finite.size else None,
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


def spread_skill(spread_ts: np.ndarray, rmse_ts: np.ndarray, n_members: int) -> float:
    """Finite-ensemble-corrected spread--skill ratio (calibration ~ 1).

    ``SSR = sqrt((M + 1) / M) * RMS(spread) / RMS(rmse)`` (Fortin et al. 2014).
    Both series are reduced as roots of mean *squares*, since spread and error
    combine in variance; the factor corrects for the ensemble mean being
    estimated from the same ``M`` members whose spread is measured. Without it
    a perfectly calibrated ensemble scores ``sqrt(M / (M + 1)) < 1`` and reads
    as over-confident. ``SSR < 1`` is over-confident, ``> 1`` over-dispersive.

    ``n_members`` is required rather than defaulted: there is no ensemble size
    for which omitting the correction is right, and a silent default would
    reintroduce the bias it exists to remove.
    """
    if n_members < 1:
        raise ValueError(f"n_members must be positive, got {n_members}")
    num = float(np.sqrt(np.nanmean(np.asarray(spread_ts, dtype=float) ** 2)))
    den = float(np.sqrt(np.nanmean(np.asarray(rmse_ts, dtype=float) ** 2)))
    if not np.isfinite(den) or den <= 0:
        return float("nan")
    return float(np.sqrt((n_members + 1) / n_members)) * num / den


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
    esmda_params: xr.Dataset,
    true_params: xr.Dataset | None = None,
) -> list[str]:
    """Ordered estimable parameters present in ``esmda_params`` (and the truth)."""
    names = [p for p in _PLOTTED_PARAMS if p in esmda_params.data_vars]
    if true_params is not None:
        names = [p for p in names if p in true_params.data_vars]
    return names


def _param_members_and_x(da: xr.DataArray):
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


def _aligned_parameter_members(
    esmda_params: xr.Dataset,
    true_params: xr.Dataset,
    prior_params: xr.Dataset | None = None,
):
    """Yield ``(name, x, members, truth_on_x, prior_members)`` per parameter.

    The one place the truth is put on the estimate's x-axis, so the error series
    and the §3 bundle below cannot drift apart in how they align it: a static
    (single-point) truth becomes a constant, a differently-sampled time-varying
    truth is linearly interpolated.

    ``prior_members`` is ``None`` unless the prior carries the parameter *and*
    is sampled on the same x-grid -- a prior of a different length cannot be
    compared knot-by-knot, and silently broadcasting it would invent a
    correspondence that does not exist.
    """
    for param_name in _plotted_param_names(esmda_params, true_params):
        x_est, members = _param_members_and_x(esmda_params[param_name])

        true_da = true_params[param_name]
        if "ensemble" in true_da.dims:
            true_da = true_da.isel(ensemble=0)
        x_true, true_members = _param_members_and_x(true_da.expand_dims("ensemble"))
        truth = true_members[0]

        order = np.argsort(x_true)
        truth_on_est = np.interp(x_est, np.asarray(x_true)[order], truth[order])

        prior_members = None
        if prior_params is not None and param_name in prior_params.data_vars:
            _, candidate = _param_members_and_x(prior_params[param_name])
            if candidate.shape[1] == truth_on_est.shape[0]:
                prior_members = candidate

        yield param_name, x_est, members, truth_on_est, prior_members


def compute_parameter_metrics(
    esmda_params: xr.Dataset,
    true_params: xr.Dataset,
    prior_params: xr.Dataset | None = None,
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
    for name, x_est, members, truth_on_est, prior_members in _aligned_parameter_members(
        esmda_params, true_params, prior_params
    ):
        entry: dict[str, np.ndarray] = {
            "x": x_est,
            "rmse": np.sqrt(np.mean((members - truth_on_est[None, :]) ** 2, axis=0)),
            # The parameter artifacts are float32 on disk; the CRPS here has
            # always been scored in float64 (see crps_ensemble on the
            # caller-owned dtype policy), so cast explicitly.
            "crps": crps_ensemble(np.asarray(members, dtype=float), truth_on_est),
        }
        if prior_members is not None:
            entry["prior_rmse"] = np.sqrt(
                np.mean((prior_members - truth_on_est[None, :]) ** 2, axis=0)
            )
            # Same float64 cast as the posterior CRPS above, so the two are
            # scored identically and their ratio (the CRPSS) is meaningful.
            entry["prior_crps"] = crps_ensemble(
                np.asarray(prior_members, dtype=float), truth_on_est
            )
        metrics[name] = entry
    return metrics


def _ensemble_std(members: np.ndarray) -> np.ndarray:
    """Per-knot ensemble standard deviation (``ddof=1``), ``nan`` below 2 members.

    ``nan`` rather than :func:`per_knot_spread`'s ``0``: every use here is as a
    *divisor*, and a zero divisor would turn "the spread is unknown" into "the
    ensemble is infinitely confident" -- the opposite reading.
    """
    if members.shape[0] < 2:
        return np.full(members.shape[1], np.nan)
    return np.asarray(members.std(axis=0, ddof=1))


def _normalized(numerator: np.ndarray, spread: np.ndarray) -> np.ndarray:
    """``numerator / spread`` where the spread is a usable scale, else ``nan``.

    A non-positive or non-finite spread means the ratio is undefined, not
    infinite: it is what a *pinned* parameter (zero prior spread by
    construction) and a fully collapsed posterior both look like.

    The reductions cannot tell the two apart -- both :func:`series_stats` and
    :func:`z_score_stats` filter on ``np.isfinite``, which drops ``inf`` and
    ``nan`` alike, so ``run_summary.yaml`` reads ``null`` either way. What the
    guard buys is the *arrays*: the per-knot bundle is what the parameter
    figures plot, where one ``inf`` rescales an axis into uselessness while a
    ``nan`` is simply not drawn. Dividing under the guard rather than after it
    also keeps numpy's divide-by-zero warning off every pinned-parameter run.
    """
    usable = np.isfinite(spread) & (spread > 0)
    return np.where(usable, numerator / np.where(usable, spread, 1.0), np.nan)


def parameter_bundle(
    post: np.ndarray,
    prior: np.ndarray | None,
    truth: np.ndarray,
) -> dict[str, np.ndarray]:
    """Posterior-quality diagnostics for one parameter, per knot (metrics doc §3).

    The scores above ask *how close* the posterior is; these ask whether it is
    **honest about its own uncertainty**, which is the failure mode ESMDA
    actually has. A posterior can halve its RMSE and still be wrong, if it
    halved its spread by more.

    Args:
        post: ``(M, K)`` posterior members over ``M`` ensemble members and ``K``
            knots (time knots for a time-varying parameter, else one per
            assimilation window).
        prior: ``(M, K)`` prior members on the same knots, or ``None``.
        truth: ``(K,)`` true parameter values, already on the same knots.

    Returns:
        Per-knot ``(K,)`` arrays, ``nan`` wherever a scale is degenerate:

        * ``z_score`` -- ``(θ* − θ̄ᵃ)/σᵃ``. The single most informative scalar:
          ``|z| > 3`` together with a small ``contraction_ratio`` is the
          canonical over-confident posterior. Judge a *pooled* set against
          :func:`calibrated_z_std`, **not** against N(0, 1) -- z is standard
          normal only as ``M → ∞``, and at the ensemble sizes run here the
          difference is large enough to invert the reading.
        * ``posterior_std`` -- ``σᵃ`` (``ddof=1``), reported so a reader can
          tell a large ``|z|`` caused by bias from one caused by collapse.

        and, when ``prior`` is given, also

        * ``normalized_error`` -- ``(θ̄ᵃ − θ*)/σᵇ``, the error in units of prior
          uncertainty, so parameters in different physical units compare.
        * ``contraction_ratio`` -- ``σᵃ/σᵇ``. ``≪ 1`` = the data informed the
          parameter; ``≈ 1`` = unidentifiable.
        * ``prior_std`` -- ``σᵇ``.
    """
    post = np.asarray(post, dtype=float)
    truth = np.asarray(truth, dtype=float)
    if post.ndim != 2:
        raise ValueError(f"post must have shape (n_members, n_knots); got {post.shape}")
    if truth.shape != (post.shape[1],):
        raise ValueError(
            f"truth must have shape ({post.shape[1]},) to match post; "
            f"got {truth.shape}"
        )

    post_mean = post.mean(axis=0)
    post_std = _ensemble_std(post)
    bundle = {
        "z_score": _normalized(truth - post_mean, post_std),
        "posterior_std": post_std,
    }

    if prior is None:
        return bundle

    prior = np.asarray(prior, dtype=float)
    if prior.ndim != 2 or prior.shape[1] != post.shape[1]:
        raise ValueError(
            f"prior must have shape (n_members, {post.shape[1]}) to match post; "
            f"got {prior.shape}"
        )
    prior_std = _ensemble_std(prior)
    bundle["prior_std"] = prior_std
    bundle["normalized_error"] = _normalized(post_mean - truth, prior_std)
    bundle["contraction_ratio"] = _normalized(post_std, prior_std)
    return bundle


def compute_parameter_bundles(
    esmda_params: xr.Dataset,
    true_params: xr.Dataset,
    prior_params: xr.Dataset | None = None,
) -> dict[str, dict[str, np.ndarray]]:
    """Per-parameter :func:`parameter_bundle`, from the saved parameter datasets.

    The dataset front end of :func:`parameter_bundle`, sharing
    :func:`compute_parameter_metrics`' truth alignment and parameter selection
    so the two blocks of ``run_summary.yaml`` always describe the same knots.
    """
    return {
        name: parameter_bundle(members, prior_members, truth_on_est)
        for name, _x, members, truth_on_est, prior_members in (
            _aligned_parameter_members(esmda_params, true_params, prior_params)
        )
    }


def compute_sensor_metrics(
    true_sensor: xr.DataArray,
    ensemble_sensor: xr.DataArray,
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
    """Per-timestep *fair* energy score, averaged over sensors.

    The energy score is the multivariate generalization of the CRPS (Gneiting &
    Raftery 2007): for a vector forecast ensemble ``{v_m}`` and truth ``v``,

        ES = mean_m ||v_m - v|| - 0.5 * sum_{m != m'} ||v_m - v_m'|| / (M(M-1)),

    which reduces to the CRPS in 1-D. It rewards both accuracy (term 1) and a
    calibrated spread (term 2), in the same |U| units as the velocity. Like
    :func:`crps_ensemble` the pairwise term excludes the zero diagonal
    (Ferro 2014); with a single member it vanishes and the score degenerates to
    the Euclidean error.

    Args:
        members: ``(component, ensemble, time, sensor)`` aligned member vectors.
        truth: ``(component, time, sensor)`` aligned truth vectors.

    Returns:
        ``(time,)`` energy score, averaged over the sensors (matching the
        per-time, over-sensors reduction of ``compute_sensor_metrics``).
    """
    n_ens = members.shape[1]
    n_time = members.shape[2]
    single_member = n_ens < 2  # no pairwise term; loop-invariant
    es = np.empty(n_time)
    # Loop over time so the pairwise term never materializes more than
    # ``(component, ensemble, ensemble, sensor)`` at once.
    for t in range(n_time):
        m = members[:, :, t, :]  # (C, E, S)
        v = truth[:, t, :]  # (C, S)
        d_truth = np.sqrt(np.sum((m - v[:, None, :]) ** 2, axis=0))  # (E, S)
        term1 = d_truth.mean(axis=0)  # (S,)
        if single_member:
            es[t] = float(term1.mean())
            continue
        diff = m[:, :, None, :] - m[:, None, :, :]  # (C, E, E, S)
        d_pair = np.sqrt(np.sum(diff**2, axis=0))  # (E, E, S)
        # Zero diagonal, so the full sum over n(n-1) is the off-diagonal mean.
        term2 = 0.5 * d_pair.sum(axis=(0, 1)) / (n_ens * (n_ens - 1))  # (S,)
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


def calibrated_z_std(n_members: int) -> float | None:
    """Std of the z-scores a *calibrated* ``n_members`` ensemble produces.

    The reference to judge :func:`z_score_stats`' ``std`` against -- and it is
    **not 1** at the ensemble sizes this repo runs. ``z = (θ* − θ̄ᵃ)/σᵃ`` is only
    standard normal in the limit: at finite ``M`` the numerator carries the
    ensemble mean's own sampling error and the denominator is a noisy estimate
    of the true spread, so for a calibrated ensemble
    ``z ~ sqrt(1 + 1/M) · t_(M-1)``, whose std is

        sqrt((1 + 1/M) · (M - 1) / (M - 3)).

    That is 1.02 at ``M = 64`` (the production default) and 1.03 at ``M = 50``,
    but 1.26 at ``M = 8`` and *infinite* at ``M ≤ 3`` -- Student's t has no
    variance there. ``None`` in that case: the sample std of z does not
    converge, so no yardstick exists and reporting one would invent it. A
    2-member ensemble scoring ``std`` in the hundreds is the *calibrated*
    result, not a failure.

    The reference is exact (confirmed against a 400k-replicate simulation at
    ``M`` = 4, 8, 16, 50, 64) but it is a *population* std, and the sample std
    measured against it converges slowly just above the cutoff: ``t_3`` has no
    fourth moment, so at ``M = 4`` a calibrated run's sample std has a median
    of 1.71 against a reference of 1.94. Read the two as comparable from about
    ``M = 8``.
    """
    if n_members <= 3:
        return None
    return float(np.sqrt((1.0 + 1.0 / n_members) * (n_members - 1) / (n_members - 3)))


def z_score_stats(z: np.ndarray, n_members: int) -> dict | None:
    """Calibration reduction of a set of z-scores, or ``None`` if it has none.

    Not :func:`series_stats`: a z-score set is judged by its *distribution*, so
    this reports the moments to read it by rather than a series' endpoint.
    Compare ``std`` against ``expected_std``, never against 1 -- see
    :func:`calibrated_z_std` for why the two differ, and by how much.

    ``n_members`` is required rather than defaulted, for the same reason
    :func:`spread_skill` requires it: there is no ensemble size at which the
    ``M → ∞`` reference is the right one, so a silent default would bake in the
    misreading it exists to prevent.

    ``None`` below **four** members, not merely a null ``std``. A calibrated
    ensemble's z-scores are ``sqrt(1 + 1/M) · t_(M-1)``, and a ``t`` has a mean
    only above 1 dof and a variance only above 2 -- so at ``M = 2`` (the CI
    smoke shape) z is *Cauchy*, whose sample mean never concentrates no matter
    how many knots are pooled: a perfectly calibrated 2-member run produces
    ``|mean| > 5`` about 13 % of the time and ``max_abs > 3`` about 68 % of it.
    Reporting any one of those moments while nulling the others just moves the
    misreading to whichever key the reader falls back on. Nothing about a
    z-score set is interpretable at that size; ``contraction_ratio`` is, and is
    where the reader is pointed. ``None`` also when no knot has a finite z.

    ``std`` alone is ``None`` below two finite knots (undefined, not zero). It
    is a *sample* std over ``n`` -- see the caveat in ``docs/scripts_and_configs.md``
    on how wide its own sampling range is when ``n`` is small.
    """
    expected = calibrated_z_std(n_members)
    if expected is None:
        return None
    values = np.asarray(z, dtype=float).ravel()
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return None
    return {
        "n": int(finite.size),
        "mean": float(finite.mean()),
        "std": float(finite.std(ddof=1)) if finite.size > 1 else None,
        "expected_std": expected,
        "max_abs": float(np.abs(finite).max()),
    }


def _bundle_summary(name: str, bundle: dict[str, np.ndarray], n_members: int) -> dict:
    """Reduce one :func:`parameter_bundle` to the ``run_summary.yaml`` entries."""
    entry: dict = {"z_score": z_score_stats(bundle["z_score"], n_members)}
    # Below four members the entry is null by policy, not by degeneracy, and
    # the caller has already said so once for the whole run.
    if entry["z_score"] is None and calibrated_z_std(n_members) is not None:
        # No knot has a usable posterior spread. Either the ensemble collapsed,
        # or -- the likelier cause in practice -- a diverged member is NaN,
        # which poisons the mean and the spread at every knot at once. Worth a
        # line either way, since the resulting ``null`` otherwise reads as
        # "not computed".
        logger.warning(
            "%s: no finite z-score at any knot -- the posterior spread is zero, "
            "undefined, or poisoned by a non-finite member, so the calibration "
            "of this parameter is unknown",
            name,
        )
    for key in ("normalized_error", "contraction_ratio"):
        if key in bundle:
            entry[key] = series_stats(bundle[key])
    return entry


def _skill_score(
    post: np.ndarray, prior: np.ndarray, name: str, what: str
) -> tuple[float, float | None]:
    """``(mean prior score, 1 - mean post / mean prior)``; ``None`` skill if prior is 0.

    A zero reference is not an error: the fair CRPS is *identically* zero at
    ``M = 2`` whenever the truth falls between the two members (term1 and the
    pairwise term coincide), which is the CI smoke shape. The skill is genuinely
    undefined there, so it is emitted as ``None`` and logged rather than
    special-cased away.

    **Both means are taken over the knots finite in *both* series**, never over
    each series' own finite set. A diverged member is ``nan`` from the knot it
    blew up at onwards while the prior stays finite everywhere, so averaging the
    two independently builds ``1 - mean(post)/mean(prior)`` out of two different
    horizons -- and inflates the score exactly when the knot that dropped out is
    the hard one the prior scored worst at. Intersecting is preferred over the
    sibling ``window_statistics_summary`` guard's outright ``None``, because a
    knot here is one sample of a horizon rather than a whole assimilation
    window: the surviving knots still carry a meaningful score, whereas nulling
    would lose the parameter entirely over a single diverged member. The
    returned reference is the mean over those same knots, so it is always the
    denominator that was actually used -- never a full-horizon prior mean
    standing next to a partial-horizon skill.

    ``prior_mean`` is ``nan`` when no knot is finite in both (the skill is
    ``None`` then, via the undefined branch). Callers must write that ``nan`` as
    ``null``: a bare ``.nan`` in the YAML reads as a measured value to anything
    that does not check. Nothing on that path warns, so no caller needs a
    ``catch_warnings`` wrapper.
    """
    post_knots = np.isfinite(post)
    prior_knots = np.isfinite(prior)
    both = post_knots & prior_knots
    if not np.array_equal(post_knots, prior_knots):
        logger.warning(
            "%s: the prior %s is finite at %d knots and the posterior at %d, so "
            "their skill score is taken over the %d knots finite in both -- it "
            "covers less than the full horizon",
            name,
            what,
            int(prior_knots.sum()),
            int(post_knots.sum()),
            int(both.sum()),
        )
    if both.any():
        # Plain ``mean`` on the selection rather than ``nanmean`` on the whole:
        # the empty case is handled below instead of by numpy, which reports it
        # as a "Mean of empty slice" RuntimeWarning on stderr -- indistinguishable
        # from a crash when it surfaces in the middle of a metric stage.
        prior_mean = float(prior[both].mean())
        post_mean = float(post[both].mean())
    else:
        prior_mean = post_mean = float("nan")
    if not prior_mean > 0:
        logger.warning(
            "%s: prior %s is %.3g, so its skill score is undefined (null). At "
            "M=2 the fair CRPS is exactly 0 whenever the truth is bracketed by "
            "the two members.",
            name,
            what,
            prior_mean,
        )
        return prior_mean, None
    return prior_mean, float(1.0 - post_mean / prior_mean)


# ---------------------------------------------------------------------------
# Window-statistic scoring (metrics doc §4.2)
# ---------------------------------------------------------------------------

# Ties in a rank are broken by a draw, which must not depend on OS entropy:
# re-running the metric stage on the same run directory has to reproduce the
# same ``run_summary.yaml``. Same reasoning as the bootstrap's seed.
_DEFAULT_TIE_SEED = 0


def ensemble_rank(ens: np.ndarray, truth: np.ndarray, rng=None) -> np.ndarray:
    """Rank of the truth within an ensemble: an integer in ``[0, M]`` per knot.

    The third calibration view, beside the CRPS (sharpness *and* accuracy) and
    the z-score (the first two moments): a rank uses only the ordering, so it
    sees a *shape* failure -- a bimodal or skewed posterior that happens to have
    the right mean and spread -- that neither of the others can. Pooled over
    many knots the ranks of a calibrated ensemble are uniform on ``0..M``;
    a U-shape means under-dispersion, a dome over-dispersion, and a pile at one
    end a bias.

    Ties get a uniform draw among the ranks they are consistent with, rather
    than a fixed rule: a deterministic tie-break piles the mass at one end of
    the histogram and reads as a bias that is not there. They are essentially
    impossible for a continuous statistic and routine for a collapsed one, so
    the draw is seeded (``_DEFAULT_TIE_SEED``) and the result reproduces.

    Args:
        ens: ``(M, K)`` members.
        truth: ``(K,)`` truth.
        rng: A ``numpy`` generator or a seed; ``None`` means the fixed default.

    Returns:
        ``(K,)`` float ranks -- float, not int, because a knot where the truth
        or any member is non-finite has no rank and is ``nan``. Drop those
        before pooling; a rank histogram of an ensemble whose members are
        partly NaN is not a calibration statement.
    """
    ens = np.asarray(ens, dtype=float)
    truth = np.asarray(truth, dtype=float)
    if ens.ndim != 2 or truth.shape != (ens.shape[1],):
        raise ValueError(
            f"ens must be (M, K) and truth (K,); got {ens.shape} and {truth.shape}"
        )
    generator = (
        rng
        if isinstance(rng, np.random.Generator)
        else np.random.default_rng(_DEFAULT_TIE_SEED if rng is None else rng)
    )
    usable = np.isfinite(truth) & np.isfinite(ens).all(axis=0)
    below = (ens < truth[None, :]).sum(axis=0)
    ties = (ens == truth[None, :]).sum(axis=0)
    # An untied knot has ``ties == 0``, so its draw is deterministically 0 and
    # its rank cannot depend on how many ties preceded it. (It does not follow
    # that the *generator state* is data-independent: numpy short-circuits
    # ``integers(0, 1)`` without consuming entropy, so a generator shared with
    # a later caller would see a stream that depends on the tie count. Every
    # call site here builds its own.)
    drawn = generator.integers(0, ties + 1)
    return np.where(usable, below + drawn, np.nan)


def _flat_members(members: xr.DataArray) -> np.ndarray:
    """``(window, ensemble, sensor)`` -> ``(M, K)``, ensemble first, K = W*S.

    The transpose is by *name*, so it does not matter that ``window_statistics``
    puts ``window`` first (it is the concat dim) -- but the remaining dims keep
    their relative order, which is what makes ``K`` window-major and lets
    :func:`_score_window_statistic` reshape it back to ``(window, sensor)``.
    """
    ordered = members.transpose("ensemble", ...)
    values = np.asarray(ordered.values, dtype=float)
    return values.reshape(values.shape[0], -1)


def _score_window_statistic(members, truth, floor, name):
    """Score one statistic's ``(window, sensor)`` grid; entry + its per-window CRPS.

    ``members`` is ``(ensemble, window, sensor)``, ``truth`` ``(window, sensor)``
    and ``floor`` the matching per-member bootstrap std (or ``None``). The
    reductions differ by what the quantity *is*: the CRPS and the
    identifiability ratio are averaged over sensors and reported as a series
    over windows (so ``final`` is the end-of-run value, as everywhere else in
    the summary), while the z-scores are a *distribution* and are pooled.
    """
    member_values = _flat_members(members)
    truth_values = np.asarray(truth.values, dtype=float).reshape(-1)
    n_members, n_windows = member_values.shape[0], members.sizes["window"]

    crps = crps_ensemble(member_values, truth_values)
    spread = _ensemble_std(member_values)
    z = _normalized(truth_values - member_values.mean(axis=0), spread)
    ranks = ensemble_rank(member_values, truth_values)

    def _per_window(flat):
        """Sensor-averaged series over windows, from a flat ``(W*S,)`` array.

        A window with no frames at all (a truncated run -- invariant 3) is
        all-``nan`` across its sensors, which is the documented null and not
        something to warn about.
        """
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            return np.nanmean(flat.reshape(n_windows, -1), axis=1)

    crps_per_window = _per_window(crps)
    finite_ranks = ranks[np.isfinite(ranks)].astype(int)
    entry = {
        "crps": series_stats(crps_per_window),
        "z_score": z_score_stats(z, n_members),
        # Counts per rank, not the raw rank list. A rank histogram is what
        # figure D1 consumes, and pooling it over sensors and windows loses
        # nothing D1 uses -- while the raw list is ~4x larger on disk (measured:
        # 7053 lines of run_summary.yaml at Barcelona scale, since the YAML
        # writer puts every int on its own line). ``M+1`` bins rather than the
        # ~10 D1 draws: binning is exact here and the figure can coarsen, but
        # not the other way round.
        "rank_counts": np.bincount(finite_ranks, minlength=n_members + 1).tolist(),
    }

    if floor is not None:
        # Median over members, not mean: one diverged member's window can have a
        # wild sampling std, and the floor is meant to describe a typical
        # member's window rather than the worst one. A column with no measured
        # floor at all (a window too short to block, which is the CI smoke
        # shape) is the documented null path, not something to warn about.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            within = np.nanmedian(_flat_members(floor), axis=0)
        identifiability = series_stats(_per_window(_normalized(spread, within)))
        if identifiability is not None:
            entry["identifiability"] = identifiability
            # Thresholded on the value that is actually written, not on a
            # separate reduction of the same array: the docs tell the reader to
            # compare the emitted number against 3, and a warning derived from a
            # different pooling can disagree with it on skewed sensors.
            if identifiability["mean"] < 3.0:
                logger.warning(
                    "%s: across-member spread is only %.2gx its own sampling "
                    "noise (mean over windows of the sensor average) -- this "
                    "statistic is not identifiable at this window length, so "
                    "read its CRPS and ranks as noise, not as calibration",
                    name,
                    identifiability["mean"],
                )
    return entry, crps_per_window


def window_statistics_summary(
    truth_stats,
    posterior_stats,
    prior_stats=None,
    posterior_sampling_std=None,
    prior_sampling_std=None,
    label="",
):
    """The ``sensor_statistics`` block for one sensor set (metrics doc §4.2).

    Takes the dicts :func:`evaluation.sensors.window_statistics` and
    :func:`evaluation.sensors.window_sampling_std` produce -- statistic name ->
    ``(window, quantity, [ensemble,] sensor)`` -- and scores every
    ``statistic × quantity`` combination as its own key (``mean_u``,
    ``variance_magnitude``, ...), because they are separately identifiable: a
    parameter that fixes the mean wind while leaving the resolved variance
    halved is the failure this block exists to name.

    The prior half is scored identically, and the posterior entries carry the
    skill against it (``crps_reduction_vs_prior``, positive when assimilation
    helped). Absent prior artifacts are a no-op, not an error: the prior state
    is only saved under ``run.save_prior_state``.

    Args:
        truth_stats: ``window_statistics`` of the truth series.
        posterior_stats: ``window_statistics`` of the posterior members.
        prior_stats: The same for the prior members, or ``None``.
        posterior_sampling_std: ``window_sampling_std`` of the posterior members,
            or ``None`` to skip the identifiability guard.
        prior_sampling_std: The same for the prior.
        label: Prefix for the log lines (e.g. the sensor-set name), so a run
            with several sets does not emit indistinguishable warnings.

    Returns:
        ``{"n_members", "n_windows", "num_sensors", "posterior"[, "prior"]}``,
        or ``{}`` when the posterior carries no ensemble dimension.
    """
    sample = posterior_stats["mean"]
    if "ensemble" not in sample.dims:
        # Invariant 3: an ensemble-mean-only artifact (an old run, a state file
        # written without the member axis) has nothing to score probabilistically,
        # and must not abort the metric stage -- which would cost the parameter
        # and state blocks too, not just this one.
        logger.warning(
            "%sno ensemble dimension in the window statistics -- the sensor "
            "statistics need the members, so the block is omitted",
            f"{label}: " if label else "",
        )
        return {}
    quantities = [
        str(q) for q in np.asarray(posterior_stats["mean"]["quantity"].values)
    ]

    def _score(stats, sampling_std, half):
        entries, raw = {}, {}
        for statistic, members in stats.items():
            for quantity in quantities:
                name = f"{statistic}_{quantity}"
                floor = (
                    sampling_std[statistic].sel(quantity=quantity)
                    if sampling_std is not None
                    else None
                )
                entries[name], raw[name] = _score_window_statistic(
                    members.sel(quantity=quantity),
                    truth_stats[statistic].sel(quantity=quantity),
                    floor,
                    # Qualified, because a run with two sensor sets and a saved
                    # prior emits up to 32 of these warnings and a bare
                    # "mean_u" cannot be traced to the one that produced it.
                    f"{label}/{half}/{name}" if label else f"{half}/{name}",
                )
        return entries, raw

    posterior, posterior_crps = _score(
        posterior_stats, posterior_sampling_std, "posterior"
    )
    summary = {
        "n_members": int(sample.sizes["ensemble"]),
        "n_windows": int(sample.sizes["window"]),
        "num_sensors": int(sample.sizes["sensor"]),
        "posterior": posterior,
    }

    if prior_stats is None:
        return summary

    prior, prior_crps = _score(prior_stats, prior_sampling_std, "prior")
    summary["prior"] = prior
    for name, entry in posterior.items():
        post_windows = np.isfinite(posterior_crps[name])
        prior_windows = np.isfinite(prior_crps[name])
        if not np.array_equal(post_windows, prior_windows):
            # The two halves cover different windows -- a prior state file that
            # exists but stops short leaves its last window empty. Averaging a
            # 2-window prior against a 3-window posterior puts two horizons in
            # one skill score, which is the comparison this WP exists to avoid.
            logger.warning(
                "%s: the prior covers %d windows and the posterior %d, so their "
                "skill score would compare two different horizons (null)",
                name,
                int(prior_windows.sum()),
                int(post_windows.sum()),
            )
            entry["prior_crps_mean"] = None
            entry["crps_reduction_vs_prior"] = None
            continue
        # An all-empty prior (every window truncated away) reduces to nan, but
        # quietly: _skill_score handles the empty case itself rather than
        # letting nanmean warn, so no catch_warnings wrapper is needed here.
        prior_mean, skill = _skill_score(
            posterior_crps[name], prior_crps[name], name, "sensor-statistic CRPS"
        )
        # ``null``, never ``.nan``: the reductions elsewhere in this block
        # filter on ``np.isfinite``, and a bare nan in the YAML reads as a
        # measured value to anything that does not check.
        entry["prior_crps_mean"] = prior_mean if np.isfinite(prior_mean) else None
        entry["crps_reduction_vs_prior"] = skill
    return summary


def parameter_metric_summary(posterior_params, true_params, prior_params):
    """Per-parameter accuracy *and* calibration summary (metrics doc §3).

    The prior-relative entries are skill scores: ``1 - post/prior``, positive
    when assimilation helped. The CRPS one (``crps_reduction_vs_prior``, the
    CRPSS of the metrics doc §3) is the probabilistic counterpart of the RMSE
    reduction -- it also penalizes a posterior that got closer to the truth by
    collapsing, which the RMSE of the ensemble members cannot see.

    WP1.2 adds the calibration half from :func:`parameter_bundle` -- per
    parameter ``z_score``, ``normalized_error`` and ``contraction_ratio``, plus
    a ``pooled`` entry whose z-scores are concatenated across every parameter
    and knot. Pooling across parameters is legitimate precisely because a
    z-score is dimensionless; read the pooled ``n`` as a count, though, not as
    an effective sample size, since adjacent knots of one time-varying
    parameter are strongly correlated and a many-knot parameter outweighs a
    static one.
    """
    metrics = compute_parameter_metrics(posterior_params, true_params, prior_params)
    bundles = compute_parameter_bundles(posterior_params, true_params, prior_params)
    # ``.get``, not ``[...]``: invariant 3. A posterior without an ensemble
    # dimension (an old ensemble-mean-only artifact) selected no parameters and
    # returned ``{}`` before this WP, and must not now abort the whole metric
    # stage -- which would cost the state and sensor blocks too.
    n_members = int(posterior_params.sizes.get("ensemble", 0))
    if metrics and calibrated_z_std(n_members) is None:
        logger.warning(
            "Ensemble of %d: no z-score summary is interpretable below 4 "
            "members (z is Cauchy at M=2, so even its mean never converges), "
            "so every z_score entry is null. Read the contraction ratio "
            "instead -- it is well defined at any size.",
            n_members,
        )
    summary = {}
    pooled_z: list[np.ndarray] = []
    for name, m in metrics.items():
        entry = {"rmse": series_stats(m["rmse"]), "crps": series_stats(m["crps"])}
        # ``null``, never ``.nan``, for the same reason as the window block
        # above: a diverged member leaves every posterior knot non-finite, and a
        # bare nan in the YAML reads as a measured value to anything that does
        # not check. This only ever fires on a run that has no comparable knot
        # left, so no run whose numbers meant anything changes value.
        if "prior_rmse" in m:
            prior_mean, skill = _skill_score(m["rmse"], m["prior_rmse"], name, "RMSE")
            entry["prior_rmse_mean"] = prior_mean if np.isfinite(prior_mean) else None
            entry["rmse_reduction_vs_prior"] = skill
        if "prior_crps" in m:
            prior_mean, skill = _skill_score(m["crps"], m["prior_crps"], name, "CRPS")
            entry["prior_crps_mean"] = prior_mean if np.isfinite(prior_mean) else None
            entry["crps_reduction_vs_prior"] = skill
        # Same parameter selection and truth alignment as ``metrics`` (both go
        # through _aligned_parameter_members), so the lookup always hits.
        entry.update(_bundle_summary(name, bundles[name], n_members))
        pooled_z.append(bundles[name]["z_score"])
        summary[name] = entry
    if pooled_z:
        # Sits beside the parameter names, as the plan specifies. Safe because
        # the parameters are a closed whitelist (_PLOTTED_PARAMS) that will
        # never contain "pooled", and every consumer indexes this block by an
        # explicit parameter name rather than iterating it.
        summary["pooled"] = {
            "z_score": z_score_stats(np.concatenate(pooled_z), n_members)
        }
    return summary


# ---------------------------------------------------------------------------
# Mean-field scoring (WP1.4, metrics doc §4.1)
# ---------------------------------------------------------------------------


def hit_rate(predicted, observed, tolerance_d=0.25, tolerance_w=0.0):
    """VDI 3783/9 hit rate ``q``: the fraction of points that match.

    The one standards-based number the metrics doc asks of a mean *field*. A
    point counts as a hit when it is within a **relative** tolerance ``D`` or a
    **absolute** one ``W`` of the observation:

        ``|p - o| / |o| <= D``  or  ``|p - o| <= W``

    The two are an ``or`` for a reason: a relative test alone is unreachable
    wherever the observed value passes through zero (in an urban canopy, every
    recirculation boundary), and an absolute test alone is meaningless in the
    free stream. The guideline's acceptance threshold is ``q >= 0.66`` at
    ``D = 0.25``.

    ``W`` is what makes ``q`` a statement rather than a convention. Set from the
    truth's own sampling uncertainty -- the block-bootstrap standard error of
    its time-mean, ``sigma_u/sqrt(N_eff)`` (see
    :func:`~evaluation.turbulence.block_bootstrap_std`) -- it turns the score
    into "indistinguishable from the truth within the truth's own noise", which
    is the strongest claim a finite averaging window supports. The default
    ``0.0`` leaves the relative test alone, i.e. the guideline's fallback when
    no floor was measured; it is deliberately not a physical constant, because
    no single velocity scale is right for every case.

    Args:
        predicted: Model values, any shape.
        observed: Reference (truth) values, broadcastable to ``predicted``.
        tolerance_d: Relative tolerance ``D``; the guideline's 0.25.
        tolerance_w: Absolute tolerance ``W``, broadcastable to ``predicted``
            (so a per-component floor can be applied to a stacked field in one
            call).

    Returns:
        ``{"q": float | None, "n_points": int}``. Points where either side is
        non-finite are excluded from both -- that is the masked solid cells and
        the edges an interpolated truth could not reach, which must not be
        counted as hits *or* misses. ``q`` is ``None`` when nothing is left,
        which is the honest answer for a fully masked region rather than 0.0.

    Raises:
        ValueError: If ``tolerance_d`` is negative, or ``tolerance_w`` is
            negative anywhere (a tolerance is a distance).
    """
    if tolerance_d < 0:
        raise ValueError(f"tolerance_d must be non-negative, got {tolerance_d}")
    w = np.asarray(tolerance_w, dtype=float)
    if np.any(w < 0):
        raise ValueError("tolerance_w must be non-negative")

    p = np.asarray(predicted, dtype=float)
    o = np.asarray(observed, dtype=float)
    scored = np.isfinite(p) & np.isfinite(o)
    # A non-finite W (no floor measured for that component) falls back to the
    # relative test alone rather than disqualifying the point: the relative
    # criterion is the guideline's, the absolute one is the refinement.
    w = np.where(np.isfinite(w), w, 0.0)
    n_points = int(scored.sum())
    if n_points == 0:
        return {"q": None, "n_points": 0}

    error = np.abs(p - o)
    with warnings.catch_warnings():
        # |o| == 0 at a stagnation point: the relative test is then False (inf
        # or nan, neither <= D) and the absolute one decides, which is exactly
        # the division of labour the ``or`` exists for.
        warnings.simplefilter("ignore", RuntimeWarning)
        relative = error / np.abs(o)
    hits = scored & ((relative <= tolerance_d) | (error <= w))
    return {"q": float(hits.sum() / n_points), "n_points": n_points}


# ---------------------------------------------------------------------------
# ESMDA health: the normalized data mismatch (metrics doc §5)
# ---------------------------------------------------------------------------

# Posterior target for O_N and the half-width of the band around it. Under the
# chi-squared argument (Emerick & Reynolds; Evensen) a correctly conditioned
# posterior member has ``E[O_N] = 1/2`` with ``Var[O_N] = 1/(2 N_d)``, so the
# 3-sigma band is ``1/2 ± 3/sqrt(2 N_d)``.
DATA_MISMATCH_TARGET = 0.5

# The advisory collapse test: an across-member IQR below this fraction of the
# band's half-width means the members agree on their own mismatch to within the
# noise the target is defined at. Paired with an off-target median, that is
# ensemble collapse rather than convergence.
_COLLAPSE_IQR_FRACTION = 0.1

# Fewest pooled values an IQR-based collapse verdict may rest on. At M=2 the
# "quartiles" are just the two values scaled, so a smoke run's two near-identical
# members trip the test every time -- and the master plan's smoke caution asks
# for ``None`` + a log line on a degenerate shape rather than a special case.
_COLLAPSE_MIN_VALUES = 8


def data_mismatch(obs, pred_obs, obs_error_std):
    """Normalized data mismatch ``O_N`` per ensemble member (metrics doc §5).

    ``O_N(θ_m) = (1/2N_d)·(d − g(θ_m))ᵀ C_D⁻¹ (d − g(θ_m))`` with the
    **un-inflated** ``C_D`` — the mean over observations of the squared
    standardized residual, halved.

    The one diagnostic that separates the two ways an ES-MDA run fails, which no
    RMSE can: ``O_N`` well above ½ is under-fitting; ``O_N`` well below it means
    the ensemble is fitting the observation *noise* — an over-aggressive
    schedule, or a missing localization letting spurious long-range correlations
    consume the residual.

    Args:
        obs: The assimilated observations ``d``, shape ``(N_d,)``.
        pred_obs: Predicted observations ``g(θ_m)``, shape ``(N_d, M)``.
        obs_error_std: ``sqrt(diag(C_D))``, shape ``(N_d,)`` or a scalar.

    Returns:
        ``(M,)`` array of ``O_N``, one per member. Observations with a
        non-finite or non-positive error std, or a non-finite value on either
        side, are dropped from the average; a member left with none scores
        ``nan`` rather than 0.

    Raises:
        ValueError: If ``pred_obs`` is not 2-D or its observation axis does not
            match ``obs``.
    """
    d = np.asarray(obs, dtype=float).ravel()
    g = np.asarray(pred_obs, dtype=float)
    if g.ndim != 2:
        raise ValueError(f"pred_obs must be 2-D (N_d, M), got shape {g.shape}.")
    if g.shape[0] != d.size:
        raise ValueError(
            f"pred_obs has {g.shape[0]} observations but obs has {d.size}."
        )

    sigma = np.broadcast_to(np.asarray(obs_error_std, dtype=float).ravel(), d.shape)
    # A zero/negative/non-finite sigma would divide the residual by nothing. It
    # cannot reach here from a validated C_D (the smoother rejects those at
    # construction), but these arrays may equally come off an old or hand-edited
    # run dir, which is not a contract this function controls.
    usable = np.isfinite(sigma) & (sigma > 0.0) & np.isfinite(d)
    if not np.any(usable):
        return np.full(g.shape[1], np.nan)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        standardized = ((d[:, None] - g) / sigma[:, None]) ** 2
        standardized = np.where(
            usable[:, None] & np.isfinite(standardized), standardized, np.nan
        )
        # An all-nan column -- a member whose forecast failed at every sensor --
        # scores nan, which the summary then drops.
        return np.nanmean(standardized, axis=0) / 2.0


def data_mismatch_target_band(n_obs):
    """Half-width ``3/√(2N_d)`` of the 3-sigma band around the ½ target.

    ``None`` when ``n_obs`` is not positive: there is no band without
    observations, and reporting one would invent a tolerance.
    """
    n = int(n_obs)
    if n <= 0:
        return None
    return float(3.0 / np.sqrt(2.0 * n))


def _collapse_verdict(per_window, band):
    """Whether EVERY window's final iteration shows a collapsed member spread.

    The collapse signature is an across-member IQR that has fallen to a fraction
    of the target band's half-width *while the median is off target*: members
    agreeing on a wrong answer to within the noise the target is defined at.
    Identical members sitting **on** target are converged, not collapsed.

    Judged per window and ``and``-ed, because the IQR is only meaningful across
    members of one window: pooling a rollout's windows measures the drift of
    ``O_N`` from window to window, which is not a spread at all.

    ``None`` -- the verdict abstains -- when there is no band, no data, or fewer
    than :data:`_COLLAPSE_MIN_VALUES` members in a window. At the smoke shape's
    ``M = 2`` the "quartiles" are just the two members scaled, so a verdict there
    would fire on every CI run; the master plan's smoke caution asks for ``None``
    plus a log line rather than a special case.
    """
    windows = [
        np.atleast_2d(np.asarray(w, dtype=float))
        for w in (per_window if per_window is not None else [])
    ]
    if band is None or not windows:
        return None

    verdicts = []
    for values in windows:
        finite = [row[np.isfinite(row)] for row in values]
        final = next((f for f in reversed(finite) if f.size > 0), None)
        if final is None:
            continue
        if final.size < _COLLAPSE_MIN_VALUES:
            logger.info(
                "data_mismatch: %d member(s) at a window's final iteration is too "
                "few for an IQR-based collapse verdict (need %d); reporting "
                "collapsed=None",
                int(final.size),
                _COLLAPSE_MIN_VALUES,
            )
            return None
        iqr = float(np.subtract(*np.percentile(final, [75, 25])))
        median = float(np.median(final))
        verdicts.append(
            iqr < _COLLAPSE_IQR_FRACTION * band
            and abs(median - DATA_MISMATCH_TARGET) > band
        )

    return bool(verdicts and all(verdicts))


def data_mismatch_summary(per_step, n_obs, per_window=None):
    """Reduce per-iteration ``O_N`` values to the ``data_mismatch`` block.

    Args:
        per_step: Sequence of length ``L`` (the ESMDA iterations, index 0 the
            prior forecast and −1 the posterior forecast), each entry that
            iteration's per-member ``O_N`` values — pooled over the run's
            windows, so a rollout contributes ``W·M`` values per iteration.
        n_obs: ``N_d`` per window, which sets the target band.
        per_window: Optional list of ``(L, M)`` arrays, one per window, used for
            the ``collapsed`` verdict alone — see :func:`_collapse_verdict` for
            why that one cannot be taken off the pooled ``per_step``. Omitted,
            ``collapsed`` is ``None`` rather than a pooled approximation of it.

    Returns:
        The block described in ``phase2_obs_persistence.md``, or ``None`` when
        no iteration has a finite value (an empty or fully failed history).

    The three flags are **advisory**, and the block carries that in its own
    ``caveat`` field: the χ² target assumes ``C_D`` covers representativeness
    error as well as instrument error, and here it does not
    (``esmda.obs_error_std`` is a single instrument-scale number). A ``C_D``
    that is too small makes an otherwise healthy run look under-fitted. Read the
    *trend* across iterations and the across-member spread — neither of which a
    constant mis-scaling of ``C_D`` moves — before reading the flags.
    """
    steps = [
        np.asarray(v, dtype=float).ravel()
        for v in (per_step if per_step is not None else [])
    ]
    finite = [s[np.isfinite(s)] for s in steps]
    if not any(f.size for f in finite):
        return None

    medians = [float(np.median(f)) if f.size else None for f in finite]
    iqrs = [
        float(np.subtract(*np.percentile(f, [75, 25]))) if f.size else None
        for f in finite
    ]
    minima = [float(np.min(f)) if f.size else None for f in finite]

    band = data_mismatch_target_band(n_obs)
    # The last iteration that produced anything -- not simply ``[-1]``, which
    # would report ``None`` for a run whose posterior forecast failed outright.
    # Which one that was is published as ``final_step_index``: the flags below
    # are named ``*_final``, and on such a run they describe an earlier
    # iteration, which the reader has no other way to discover.
    final_step = next(
        (i for i in reversed(range(len(finite))) if finite[i].size > 0), None
    )
    final_median = None if final_step is None else medians[final_step]

    # Off-target counts as over/under-fitting only outside the band; with no
    # band (no observations) there is nothing to judge against, hence ``None``
    # rather than a default-False that would read as "checked, and fine".
    underfit = overfit = None
    if band is not None and final_median is not None:
        underfit = bool(final_median > DATA_MISMATCH_TARGET + band)
        overfit = bool(final_median < DATA_MISMATCH_TARGET - band)

    # ``collapsed`` is an ACROSS-MEMBER verdict, so on a rollout it is taken per
    # window rather than from ``per_step`` -- which pools every window's members,
    # and whose IQR is therefore dominated by the (entirely healthy) drift of
    # O_N from one window to the next. A 10-window run in which every window's
    # members sat on top of each other would report a large pooled IQR and
    # ``collapsed: False``, the exact conflation D3 un-pools to avoid.
    collapsed = _collapse_verdict(per_window, band)

    return {
        "per_step_median": medians,
        "per_step_iqr": iqrs,
        "per_step_min": minima,
        "target": DATA_MISMATCH_TARGET,
        "target_band": band,
        "num_observations": int(n_obs),
        "final_step_index": final_step,
        "underfit_final": underfit,
        "overfit_final": overfit,
        "collapsed": collapsed,
        "caveat": "no_representativeness_error",
    }
