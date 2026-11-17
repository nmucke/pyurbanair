"""Turbulence statistics of the sensor time series as observations.

Besides the raw frames (``raw``) and their interval means (``bins``, an
:class:`AggregateObservations`), a window's sensor series carries second
moments (``tke``, the variances, and ``stress``, the covariances between
components) and spectra (``spectral``) that constrain closure and roughness
parameters which the means barely see. :class:`ObservationStatistics` maps a
raw ``(..., "time", "obs")`` series to the chosen statistics, stacked along
``"time"`` in the fixed order raw, bins, tke, stress, spectral, and
:meth:`ObservationStatistics.resolve_error` gives the matching block-diagonal
(diagonal) observation error. ``raw`` and ``bins`` are refused together: the
bins are means of the same frames, so the pair counts the data twice. The
statistics are allowed with ``raw`` but are partly redundant with it.

Every block keeps the sensor as the innermost axis of the ``obs`` axis: the
``(component, sensor)`` layout of the observation operator, and for
``stress`` ``(pair, sensor)``. With two components there is one pair, so the
stress row fills only the first ``num_sensors`` slots; a boolean
``slot_used`` coordinate marks the rest, which ``flatten_observations`` drops.
Observation ``j`` of the flattened vector thus sits at sensor
``j % num_sensors``, and distance localization and the adaptive observation
error carry over. See ``docs/data_assimilation.md`` (§2,
Turbulence-statistics observations).
"""

import itertools
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence, cast

import numpy as np
import xarray
from data_assimilation.observation_error import ObservationErrorSpec
from data_assimilation.observation_operator import AggregateObservations
from scipy import linalg, signal

#: Statistic types in output order.
STATISTIC_TYPES = ("raw", "bins", "tke", "stress", "spectral")

_TKE_DEFAULTS: dict[str, Any] = {
    "normalize": True,
    "log": True,
    "std": "auto",
    "std_floor": 0.05,
    "pool_std": "height",
    "representation_std": 0.0,
}
#: No ``log``: the stresses are signed.
_STRESS_DEFAULTS: dict[str, Any] = {
    "normalize": True,
    "std": "auto",
    "std_floor": 0.05,
    "pool_std": "height",
    "representation_std": 0.0,
}
_SPECTRAL_DEFAULTS: dict[str, Any] = {
    "segment_seconds": None,
    "num_bands": 4,
    "f_min": None,
    "f_max": None,
    "normalize": True,
    "log": True,
    "std": "auto",
    "std_floor": 0.1,
    "pool_std": "height",
    "representation_std": 0.0,
}
#: Welch configuration (scipy's welch defaults, made explicit so the
#: per-segment spectrogram matches it exactly).
_WINDOW = "hann"
_DETREND = "constant"
#: Largest relative deviation of a frame spacing from the median spacing the
#: spectra accept (uDALES writes slightly irregular output times).
_MAX_TIME_JITTER = 0.1
#: Largest relative difference of a prediction's frame spacing from the
#: observed series' the spectral bands accept.
_MAX_DT_MISMATCH = 0.01
#: Guard for degenerate (constant) series in predictions, so a log stays finite.
_TINY = np.finfo(float).tiny

#: Row function of a statistic block: ``(..., T, O)`` frames and the
#: per-obs white-noise variance to add (or None) -> ``(..., rows, O)``.
_Rows = Callable[[np.ndarray, np.ndarray | None], np.ndarray]
#: Sampling std of a block from one observed ``(T, O)`` series -> ``(rows, O)``.
_Std = Callable[[np.ndarray], np.ndarray]


def _positive(value: Any, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite positive number.") from exc
    if not np.isfinite(result) or result <= 0:
        raise ValueError(f"{name} must be a finite positive number.")
    return result


def _non_negative(value: Any, name: str) -> float:
    result = float(value)
    if not np.isfinite(result) or result < 0:
        raise ValueError(f"{name} must be finite and non-negative.")
    return result


def _options(
    given: Mapping[str, Any] | None, defaults: dict[str, Any], name: str
) -> dict[str, Any]:
    """Defaults updated with ``given``, validated."""
    options = dict(defaults)
    if given is not None:
        unknown = set(given) - set(defaults)
        if unknown:
            raise ValueError(f"Unknown {name} options: {sorted(unknown)}")
        options.update(given)
    if options["std"] != "auto":
        options["std"] = _positive(options["std"], f"{name}.std")
    for key in ("std_floor", "representation_std"):
        options[key] = _non_negative(options[key], f"{name}.{key}")
    if options["pool_std"] not in ("height", "none"):
        raise ValueError(f"{name}.pool_std must be 'height' or 'none'.")
    options["normalize"] = bool(options["normalize"])
    if "log" in defaults:
        options["log"] = bool(options["log"])
    if options["pool_std"] == "height" and not (
        options.get("log") or options["normalize"]
    ):
        needs = "log or normalize" if "log" in defaults else "normalize"
        raise ValueError(
            f"{name}.pool_std 'height' needs {needs}: stds in absolute "
            "units differ between sensors of different turbulence levels."
        )
    return options


def _sampling_errors(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Sampling errors of the window statistics of a ``(T, O)`` series.

    Returns the std of ``log var_t`` and the variance of ``mean_t`` per
    column, from the large-sample (Bartlett) formulas for a Gaussian process,

        var(s**2) / sigma**4 = (2 / T) sum_tau rho(tau)**2,
        var(mean) / sigma**2 = (1 / T) sum_tau rho(tau),

    with the autocorrelation ``rho`` estimated from the series and summed
    over ``|tau|`` up to its first zero crossing (at most ``T / 4`` lags).
    """
    num_frames = values.shape[0]
    anomaly = values - values.mean(axis=0)
    spectrum = np.abs(np.fft.rfft(anomaly, n=2 * num_frames, axis=0)) ** 2
    autocovariance = np.fft.irfft(spectrum, axis=0)[:num_frames] / num_frames
    variance = np.maximum(autocovariance[0], _TINY)
    rho = autocovariance / variance
    max_lag = max(num_frames // 4, 2)
    # Lags before the first non-positive autocorrelation (lag 0 included).
    keep = np.cumprod(rho[:max_lag] > 0, axis=0).astype(bool)
    rho = np.where(keep, rho[:max_lag], 0.0)
    log_variance_std = np.sqrt(2.0 * (2.0 * np.sum(rho**2, axis=0) - 1.0) / num_frames)
    mean_variance = variance * (2.0 * np.sum(rho, axis=0) - 1.0) / num_frames
    return np.asarray(log_variance_std), np.asarray(mean_variance)


def _fluctuations(
    values: np.ndarray, partition: Sequence[Sequence[int]] | None
) -> tuple[np.ndarray, float]:
    """Fluctuations of ``(..., T, O)`` frames about the means of a partition.

    ``partition`` lists the frame ids of every bin (``None``: one bin, the
    whole window); each frame has its bin's arithmetic mean removed. Returns
    the ``(..., T, O)`` residual series and the fraction of a white noise's
    variance that ``mean_t(residual**2)`` keeps, ``(T - B) / T`` for ``B``
    bins (removing a mean of ``n`` frames removes ``1 / n`` of their noise
    variance).
    """
    num_frames = values.shape[-2]
    if partition is None:
        partition = [range(num_frames)]
    residual = np.empty_like(values)
    for frame_ids in partition:
        ids = list(frame_ids)
        residual[..., ids, :] = values[..., ids, :] - values[..., ids, :].mean(
            axis=-2, keepdims=True
        )
    return residual, (num_frames - len(partition)) / num_frames


def _residual_sampling_error(
    residual: np.ndarray, partition: Sequence[Sequence[int]] | None
) -> np.ndarray:
    """Relative std of ``mean_t(residual**2)`` per column of a ``(T, O)``
    residual series from :func:`_fluctuations` with the same ``partition``.

    The fluctuations' autocovariance ``gamma`` is the residual's (as in
    :func:`_sampling_errors`, kept up to its first non-positive lag) plus the
    variance ``m`` the removed bin means took out of it,
    ``m = sum_b n_b m_b / T`` with ``m_b = sum_tau (1 - |tau| / n_b)
    gamma(tau) / n_b`` (linear in ``m``: solved in closed form). With
    ``Sigma`` the Toeplitz matrix of ``gamma`` and ``Q`` the bin-mean removal,
    the quadratic form ``mean_t(residual**2)`` of a Gaussian process has

        var / mean**2 = 2 tr((Q Sigma)**2) / tr(Q Sigma)**2.

    Plain Bartlett on the residual (whose autocorrelation the mean removal
    shortens) underestimates the error once the bins span a few correlation
    times.
    """
    num_frames, num_obs = residual.shape
    if partition is None:
        partition = [range(num_frames)]
    spectrum = np.abs(np.fft.rfft(residual, n=2 * num_frames, axis=0)) ** 2
    max_lag = max(num_frames // 4, 2)
    autocovariance = np.fft.irfft(spectrum, axis=0)[:max_lag] / num_frames
    keep = np.cumprod(autocovariance > 0, axis=0).astype(bool)
    lags = np.arange(max_lag)[:, None]
    weighted = np.zeros(num_obs)
    weights = np.zeros(num_obs)
    for frame_ids in partition:
        n = len(frame_ids)
        w = np.where(keep, np.clip(1.0 - lags / n, 0.0, None), 0.0)
        w[1:] *= 2.0  # +-tau
        weighted += np.sum(w * autocovariance, axis=0)
        weights += np.sum(w, axis=0)
    removed = weighted / np.maximum(num_frames - weights, 1.0)
    gamma = np.zeros((num_frames, num_obs))
    gamma[:max_lag] = np.where(keep, autocovariance + removed, 0.0)
    relative = np.empty(num_obs)
    for j in range(num_obs):
        q_sigma = _fluctuations(linalg.toeplitz(gamma[:, j]), partition)[0]
        relative[j] = np.sqrt(2.0 * np.sum(q_sigma * q_sigma.T)) / max(
            float(np.trace(q_sigma)), _TINY
        )
    return relative


def _pair_sampling_moments(
    first: np.ndarray, second: np.ndarray, partition: Sequence[Sequence[int]] | None
) -> tuple[np.ndarray, np.ndarray]:
    """Expectation and covariance of ``(mean_t(a' b'), mean_t(a'**2),
    mean_t(b'**2))`` for the ``(T,)`` fluctuations ``a'``, ``b'`` of two
    components from :func:`_fluctuations` with the same ``partition``.

    The two-component generalisation of :func:`_residual_sampling_error`:
    the auto- and cross-covariances ``gamma_ik(tau) = cov(x_i(t), x_k(t +
    tau))`` are the residual's sample ones plus the (cross-)variance the
    removed bin means took out (solved the same way), kept up to the first
    non-positive lag of the autocovariances (the cross-covariance up to the
    shorter of the two). With ``Sigma_ik`` their Toeplitz matrices and ``Q``
    the bin-mean removal, Isserlis' theorem for a Gaussian process gives
    ``E mean_t(x_i x_j) = tr(Q Sigma_ij) / T`` and

        cov(mean_t(x_i x_j), mean_t(x_k x_l))
            = (tr(Q Sigma_ik Q Sigma_lj) + tr(Q Sigma_il Q Sigma_kj)) / T**2.
    """
    num_frames = first.size
    if partition is None:
        partition = [range(num_frames)]
    spectra = np.fft.rfft(np.stack([first, second], axis=1), n=2 * num_frames, axis=0)
    max_lag = max(num_frames // 4, 2)
    # lagged[tau, i, k] = mean_t x_i(t) x_k(t + tau), tau >= 0.
    lagged = (
        np.fft.irfft(np.conj(spectra)[:, :, None] * spectra[:, None, :], axis=0)[
            :max_lag
        ]
        / num_frames
    )
    keep_auto = np.cumprod(np.diagonal(lagged, axis1=1, axis2=2) > 0, axis=0)
    keep = (keep_auto[:, :, None] * keep_auto[:, None, :]).astype(bool)
    # Both signs of every lag: tau and -tau (lag 0 once).
    two_sided = lagged + lagged.transpose(0, 2, 1)
    two_sided[0] /= 2.0
    lags = np.arange(max_lag)[:, None, None]
    sides = np.where(lags > 0, 2.0, 1.0)
    weighted = np.zeros((2, 2))
    weights = np.zeros((2, 2))
    for frame_ids in partition:
        w = np.where(keep, np.clip(1.0 - lags / len(frame_ids), 0.0, None), 0.0)
        weighted += np.sum(w * two_sided, axis=0)
        weights += np.sum(w * sides, axis=0)
    removed = weighted / np.maximum(num_frames - weights, 1.0)
    gamma = np.zeros((num_frames, 2, 2))
    gamma[:max_lag] = np.where(keep, lagged + removed, 0.0)
    # q[i][k] = Q Sigma_ik, Sigma_ik[t, s] = gamma_ik(s - t).
    q = [
        [
            _fluctuations(linalg.toeplitz(gamma[:, k, i], gamma[:, i, k]), partition)[0]
            for k in range(2)
        ]
        for i in range(2)
    ]

    def trace(x: np.ndarray, y: np.ndarray) -> float:
        return float(np.sum(x * y.T))

    mean = np.array([np.trace(q[0][1]), np.trace(q[0][0]), np.trace(q[1][1])])
    v_ab = trace(q[0][0], q[1][1]) + trace(q[0][1], q[0][1])
    covariance = np.array(
        [
            [v_ab, 2 * trace(q[0][0], q[0][1]), 2 * trace(q[0][1], q[1][1])],
            [0.0, 2 * trace(q[0][0], q[0][0]), 2 * trace(q[0][1], q[1][0])],
            [0.0, 0.0, 2 * trace(q[1][1], q[1][1])],
        ]
    )
    covariance = np.triu(covariance) + np.triu(covariance, 1).T
    return mean / num_frames, covariance / num_frames**2


def _with_noise(value: np.ndarray, noise: np.ndarray | None) -> np.ndarray:
    """Add an expected white-noise contribution; the result must be positive
    (a constant or non-finite series has no tke, stress or spectrum to
    compare)."""
    if noise is not None:
        value = value + noise
    if not np.all(value > 0):
        raise ValueError(
            "A tke, stress or spectral statistic is undefined: a constant or non-finite "
            "series (e.g. a failed member or a sensor inside a building)."
        )
    return value


@dataclass(frozen=True)
class _SpectralSetup:
    """Welch segments and band membership of one window's frames."""

    num_frames: int
    dt: float
    nperseg: int
    noverlap: int
    band_ids: np.ndarray  # band of every Welch frequency, -1 outside all bands
    noise_factors: np.ndarray  # (num_bands,) band-mean white-noise PSD / variance
    band_centres: np.ndarray  # (num_bands,) geometric band centre, Hz
    overlap_correlation: float  # squared correlation of adjacent segments


@dataclass(frozen=True)
class ResolvedStatisticsError:
    """Block-diagonal error of one window's statistics.

    ``covariance_diag`` follows the flattened layout of
    :class:`ObservationStatistics` (rows raw, bins, tke, stress, spectral,
    sensor innermost). ``noise_variance`` is the per-observation raw-frame instrument
    variance (what :attr:`ObservationStatistics.noise_variance` should be set
    to).
    """

    covariance_diag: np.ndarray
    noise_variance: np.ndarray
    block_slices: dict[str, slice]
    #: Rows of each block (its slice holds a multiple of num_sensors per row).
    block_rows: dict[str, int]
    provenance: str

    @property
    def std(self) -> np.ndarray:
        return np.asarray(np.sqrt(self.covariance_diag))


class ObservationStatistics:
    """Raw frames, interval means and turbulence statistics of sensor series.

    Applied, like :class:`AggregateObservations` (which it can replace as the
    smoothers' ``aggregate_observations``), to the real and the predicted
    observations. Per component ``c`` and sensor ``s`` over all frames of the
    window:

    - ``raw``: the frames unchanged (no aggregation);
    - ``bins``: the ``aggregation`` interval means;
    - ``tke``: one row, ``k_cs = 0.5 mean_t(u'_cs**2)`` with ``u'`` the
      fluctuation about the window mean (the population variance,
      ``ddof=0``), or with ``bins`` about the mean of each bin's frames (the
      arithmetic mean whatever the aggregation ``mode``): the variation
      between the bin means is what the bins observe, so it is left out. The
      resolved TKE of sensor ``s`` is ``sum_c k_cs``. Options ``normalize``
      (default; divide by ``0.5 |U_s|**2``, ``|U_s|**2 = sum_c
      mean_t(u_cs)**2`` over the window; needs ``num_sensors``), ``log``;
    - ``stress``: one row, per pair ``(a, b)`` of observed components (in
      operator order: (0, 1), (0, 2), (1, 2)) and sensor ``s`` the covariance
      ``mean_t(u'_as u'_bs)`` of the tke fluctuations (the same partition).
      Option ``normalize`` (default): the correlation coefficient, divided by
      ``sqrt(mean_t(u'_as**2) mean_t(u'_bs**2))``. Needs ``num_sensors`` and
      two or three components; with two the row is narrower than ``obs``
      (``slot_used``);
    - ``spectral``: one row per band, the mean one-sided Welch PSD (Hann
      window, constant detrend, 50% overlap) in ``num_bands`` log-spaced bands
      between ``f_min`` (default the first nonzero Welch frequency) and
      ``f_max`` (default Nyquist). Options ``segment_seconds`` (default a
      quarter of the window), ``normalize`` (divide by the series variance:
      the spectral shape only), ``log``.

    Instrument noise: the observed (noisy) series' statistics are taken as
    they are; the predictions instead get the expected white-noise
    contribution of ``noise_variance`` added before normalisation and log
    (tke variance ``+ (T - B) / T sigma**2`` for ``T`` frames in ``B`` bins,
    one without ``bins``, as for the spectral normalising variance; band PSD
    ``+`` the exact Welch level of white noise), so both sides carry the same
    bias. The noise is independent
    between components, so the stress covariance gets nothing; with
    ``normalize`` its denominator's variances get the tke term. ``raw`` and
    ``bins`` are never changed.

    An ``observed=True`` call fixes the spectral bands (frame count, spacing,
    segments) for the following predictions, which must have the same frame
    count and spacing; before any observed call a prediction uses its own.

    ``std``, ``pool_std``, ``representation_std`` and ``std_floor`` set the
    tke, stress and spectral error blocks, see :meth:`resolve_error`.

    Args:
        types: Non-empty subset of ``("raw", "bins", "tke", "stress",
            "spectral")``; ``raw`` and ``bins`` are mutually exclusive.
        aggregation: The interval aggregation, required with ``bins`` and
            ignored without.
        tke: ``tke`` options (``None``: defaults).
        spectral: ``spectral`` options (``None``: defaults).
        num_sensors: Sensors per component, required by ``stress`` and the
            tke ``normalize`` option to tell the components of a sensor.
        noise_variance: Raw-frame instrument variance, a scalar or one value
            per obs (``None`` or 0: no noise), added to the predicted tke,
            stress and spectral statistics.
        stress: ``stress`` options (``None``: defaults).
    """

    def __init__(
        self,
        types: Sequence[str],
        aggregation: AggregateObservations | None = None,
        tke: Mapping[str, Any] | None = None,
        spectral: Mapping[str, Any] | None = None,
        num_sensors: int | None = None,
        noise_variance: np.ndarray | float | None = None,
        stress: Mapping[str, Any] | None = None,
    ) -> None:
        if isinstance(types, str):
            raise ValueError("types must be a list of statistic types.")
        types = list(types)
        if not types:
            raise ValueError("types must name at least one statistic type.")
        unknown = set(types) - set(STATISTIC_TYPES)
        if unknown:
            raise ValueError(
                f"Unknown statistic types {sorted(unknown)}; "
                f"choose from {list(STATISTIC_TYPES)}."
            )
        if len(types) != len(set(types)):
            raise ValueError(f"Duplicate statistic types in {types}.")
        if "raw" in types and "bins" in types:
            raise ValueError(
                "'raw' and 'bins' cannot be combined: the bins are means of "
                "the same frames, so the pair counts the data twice."
            )
        if "bins" in types and aggregation is None:
            raise ValueError("The 'bins' statistic requires an aggregation.")
        self.types = tuple(t for t in STATISTIC_TYPES if t in types)
        self.aggregation = aggregation if "bins" in self.types else None
        self._allow_interval_count_change = False
        self.tke = _options(tke, _TKE_DEFAULTS, "tke")
        self.stress = _options(stress, _STRESS_DEFAULTS, "stress")
        self.spectral = _options(spectral, _SPECTRAL_DEFAULTS, "spectral")
        num_bands = int(self.spectral["num_bands"])
        if num_bands != self.spectral["num_bands"] or num_bands < 1:
            raise ValueError("spectral.num_bands must be an integer >= 1.")
        self.spectral["num_bands"] = num_bands
        for key in ("segment_seconds", "f_min", "f_max"):
            if self.spectral[key] is not None:
                self.spectral[key] = _positive(self.spectral[key], f"spectral.{key}")
        if num_sensors is not None and int(num_sensors) < 1:
            raise ValueError("num_sensors must be a positive integer.")
        self.num_sensors = None if num_sensors is None else int(num_sensors)
        if "tke" in self.types and self.tke["normalize"] and self.num_sensors is None:
            raise ValueError("tke.normalize requires num_sensors.")
        if "stress" in self.types and self.num_sensors is None:
            raise ValueError("stress requires num_sensors.")
        self.noise_variance = noise_variance
        #: Spectral setup of the last observed series, reused by predictions.
        self._observed_setup: _SpectralSetup | None = None

    @property
    def allow_interval_count_change(self) -> bool:
        """Forwarded to the wrapped aggregation (see AggregateObservations)."""
        if self.aggregation is not None:
            return bool(self.aggregation.allow_interval_count_change)
        return self._allow_interval_count_change

    @allow_interval_count_change.setter
    def allow_interval_count_change(self, value: bool) -> None:
        self._allow_interval_count_change = bool(value)
        if self.aggregation is not None:
            self.aggregation.allow_interval_count_change = bool(value)

    # ------------------------------------------------------------------
    # tke
    # ------------------------------------------------------------------

    def _speed2(self, mean: np.ndarray) -> np.ndarray:
        """``|U_s|**2`` of every ``(c, s)`` entry from ``(..., O)`` means."""
        num_sensors = cast(int, self.num_sensors)
        if mean.shape[-1] % num_sensors:
            raise ValueError("Observation count is not a multiple of num_sensors.")
        per_component = mean.reshape(*mean.shape[:-1], -1, num_sensors)
        speed2 = np.square(per_component).sum(axis=-2, keepdims=True)
        return np.broadcast_to(speed2, per_component.shape).reshape(mean.shape)

    def _partition(
        self, observations: xarray.DataArray
    ) -> tuple[tuple[int, ...], ...] | None:
        """Frame ids of the ``bins`` intervals (``None`` without bins): the
        tke fluctuations are taken about these means. Bins of one frame each
        leave no fluctuation at all and are refused."""
        if self.aggregation is None:
            return None
        bins = self.aggregation.bins(observations, allow_interval_count_change=True)
        partition = tuple(bin_.frame_ids for bin_ in bins)
        if all(len(frame_ids) == 1 for frame_ids in partition):
            raise ValueError(
                "tke and stress with bins need bins of more than one frame: every "
                f"{self.aggregation.interval_seconds:g} s bin holds a single frame "
                "(the interval is not longer than the output spacing), so every "
                "fluctuation about the bin means is zero. Lengthen the "
                "aggregation interval or drop bins."
            )
        return partition

    def _tke(
        self,
        values: np.ndarray,
        noise: np.ndarray | None,
        partition: Sequence[Sequence[int]] | None,
    ) -> np.ndarray:
        """``(..., T, O)`` frames -> ``(..., 1, O)`` tke row."""
        residual, noise_factor = _fluctuations(values, partition)
        noise = None if noise is None else noise_factor * noise
        tke = 0.5 * _with_noise(np.mean(residual**2, axis=-2), noise)
        if self.tke["normalize"]:
            tke = tke / np.maximum(0.5 * self._speed2(values.mean(axis=-2)), _TINY)
        if self.tke["log"]:
            tke = np.log(tke)
        return tke[..., None, :]

    def _tke_std(
        self, values: np.ndarray, partition: Sequence[Sequence[int]] | None
    ) -> np.ndarray:
        """Sampling std of the tke row of one ``(T, O)`` series: the relative
        std of the residual variance (:func:`_residual_sampling_error`), with
        ``normalize`` that of ``|U_s|**2`` added (independent components,
        ``var |U|**2 = sum_c 4 mean_c**2 var(mean_c)``, Bartlett's
        ``var(mean_c)`` from :func:`_sampling_errors`), times the row value
        unless it is a log (delta method)."""
        relative = _residual_sampling_error(
            _fluctuations(values, partition)[0], partition
        )
        mean_variance = _sampling_errors(values)[1]
        if self.tke["normalize"]:
            mean = values.mean(axis=0)
            speed2 = self._speed2(mean)
            speed2_variance = self._speed2(np.sqrt(4.0 * mean**2 * mean_variance))
            relative = np.sqrt(
                relative**2 + speed2_variance / np.maximum(speed2, _TINY) ** 2
            )
        std = relative[None, :]
        if not self.tke["log"]:
            std = std * self._tke(values, None, partition)[0]
        return std

    # ------------------------------------------------------------------
    # stress
    # ------------------------------------------------------------------

    def _pairs(self, num_obs: int) -> tuple[np.ndarray, np.ndarray]:
        """Component ids ``(a, b)`` of every stress pair, in operator order."""
        num_sensors = cast(int, self.num_sensors)
        num_components, remainder = divmod(num_obs, num_sensors)
        if remainder:
            raise ValueError("Observation count is not a multiple of num_sensors.")
        if not 2 <= num_components <= 3:
            raise ValueError(
                f"stress needs two or three observed velocity components (got "
                f"{num_components}): it is the covariance between components."
            )
        pairs = np.array(list(itertools.combinations(range(num_components), 2)))
        return pairs[:, 0], pairs[:, 1]

    def _by_pair(self, array: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """``(..., O)`` -> the ``(..., P * S)`` entries of the components
        ``a`` and ``b`` of every pair, sensor innermost."""
        a, b = self._pairs(array.shape[-1])
        per_component = array.reshape(
            *array.shape[:-1], -1, cast(int, self.num_sensors)
        )
        shape = (*array.shape[:-1], -1)
        return per_component[..., a, :].reshape(shape), per_component[
            ..., b, :
        ].reshape(shape)

    def _stress(
        self,
        values: np.ndarray,
        noise: np.ndarray | None,
        partition: Sequence[Sequence[int]] | None,
    ) -> np.ndarray:
        """``(..., T, O)`` frames -> ``(..., 1, P * S)`` stress row. The noise
        is independent between components: only the normalising variances
        get it (as the tke row does). The variances are checked with or
        without ``normalize``: a constant or non-finite series has no stress."""
        residual, noise_factor = _fluctuations(values, partition)
        first, second = self._by_pair(residual)
        stress = np.mean(first * second, axis=-2)
        noise = None if noise is None else noise_factor * noise
        variance = _with_noise(np.mean(residual**2, axis=-2), noise)
        if self.stress["normalize"]:
            var_a, var_b = self._by_pair(variance)
            stress = stress / np.sqrt(var_a * var_b)
        return stress[..., None, :]

    def _stress_std(
        self, values: np.ndarray, partition: Sequence[Sequence[int]] | None
    ) -> np.ndarray:
        """Sampling std of the stress row of one ``(T, O)`` series, from the
        Gaussian moments of :func:`_pair_sampling_moments`: the covariance's
        own, or for the correlation coefficient ``r = c / sqrt(v_a v_b)``
        (``normalize``) the delta method, ``g.T V g`` with
        ``g = (1 / sqrt(v_a v_b), -r / (2 v_a), -r / (2 v_b))``."""
        first, second = self._by_pair(_fluctuations(values, partition)[0])
        std = np.empty(first.shape[-1])
        for j in range(std.size):
            mean, covariance = _pair_sampling_moments(
                first[:, j], second[:, j], partition
            )
            if self.stress["normalize"]:
                c, v_a, v_b = mean[0], max(mean[1], _TINY), max(mean[2], _TINY)
                r = c / np.sqrt(v_a * v_b)
                gradient = np.array(
                    [1.0 / np.sqrt(v_a * v_b), -0.5 * r / v_a, -0.5 * r / v_b]
                )
                std[j] = np.sqrt(max(gradient @ covariance @ gradient, 0.0))
            else:
                std[j] = np.sqrt(max(covariance[0, 0], 0.0))
        return std[None, :]

    # ------------------------------------------------------------------
    # spectral
    # ------------------------------------------------------------------

    def _spectral_setup(self, times: np.ndarray) -> _SpectralSetup:
        """Frame spacing, Welch segments and the bands of one window."""
        if times.size < 2:
            raise ValueError("Spectral statistics need at least two frames.")
        spacing = np.diff(times)
        dt = float(np.median(spacing))
        if dt <= 0 or np.any(np.abs(spacing - dt) > _MAX_TIME_JITTER * dt):
            raise ValueError(
                "Spectral statistics need near-uniform frame spacing (within "
                f"{_MAX_TIME_JITTER:.0%} of the median {dt:.4g} s)."
            )
        segment_seconds = self.spectral["segment_seconds"]
        if segment_seconds is None:
            nperseg = times.size // 4
        else:
            nperseg = int(round(segment_seconds / dt))
        if nperseg < 4 or nperseg > times.size:
            raise ValueError(
                f"Window of {times.size} frames is too short for Welch segments "
                f"of {nperseg} frames (need 4 <= segment <= window)."
            )
        freqs = np.fft.rfftfreq(nperseg, dt)
        f_min = self.spectral["f_min"] or float(freqs[1])
        f_max = self.spectral["f_max"] or 0.5 / dt
        if not f_min < f_max:
            raise ValueError(f"spectral f_min {f_min} must be below f_max {f_max}.")
        num_bands = self.spectral["num_bands"]
        edges = np.geomspace(f_min, f_max, num_bands + 1)
        # Half-open [lo, hi) bands, the last one closed; a relative tolerance
        # keeps the default edges (Welch frequencies themselves) inside.
        band_ids = np.searchsorted(edges, freqs * (1 + 1e-9), side="right") - 1
        band_ids[band_ids == num_bands] = num_bands - 1
        band_ids[freqs > f_max * (1 + 1e-9)] = -1
        # Expected one-sided Welch PSD of unit-variance white noise per bin:
        # the segment-mean removal takes |W_k|^2 / (N sum w^2) of the power
        # (W the window's DFT); DC and the (even-segment) Nyquist bin are not
        # doubled.
        window = signal.get_window(_WINDOW, nperseg)
        power = np.sum(window**2)
        factors = (
            2.0 * dt * (1.0 - np.abs(np.fft.rfft(window)) ** 2 / (nperseg * power))
        )
        factors[0] /= 2.0
        if nperseg % 2 == 0:
            factors[-1] /= 2.0
        noise_factors = np.empty(num_bands)
        for b in range(num_bands):
            in_band = band_ids == b
            if not in_band.any():
                raise ValueError(
                    f"Spectral band {b} [{edges[b]:.4g}, {edges[b + 1]:.4g}] Hz "
                    "holds no Welch frequency: use fewer bands, longer segments "
                    "or a wider frequency range."
                )
            noise_factors[b] = factors[in_band].mean()
        noverlap = nperseg // 2
        step = nperseg - noverlap
        overlap = np.sum(window[step:] * window[: nperseg - step]) / power
        return _SpectralSetup(
            num_frames=times.size,
            dt=dt,
            nperseg=nperseg,
            noverlap=noverlap,
            band_ids=band_ids,
            noise_factors=noise_factors,
            band_centres=np.sqrt(edges[:-1] * edges[1:]),
            overlap_correlation=float(overlap**2),
        )

    def _setup_for(self, times: np.ndarray, observed: bool) -> _SpectralSetup:
        """The observed series' setup for predictions, a new one when observed."""
        if observed or self._observed_setup is None:
            setup = self._spectral_setup(times)
            if observed:
                self._observed_setup = setup
            return setup
        setup = self._observed_setup
        dt = float(np.median(np.diff(times))) if times.size > 1 else np.nan
        if times.size != setup.num_frames or not (
            abs(dt - setup.dt) <= _MAX_DT_MISMATCH * setup.dt
        ):
            raise ValueError(
                f"Predicted series ({times.size} frames, dt {dt:.4g} s) do not "
                f"match the observed one ({setup.num_frames} frames, dt "
                f"{setup.dt:.4g} s): their spectral bands would differ."
            )
        return setup

    def _band_segments(self, values: np.ndarray, setup: _SpectralSetup) -> np.ndarray:
        """``(..., T, O)`` frames -> ``(..., num_bands, O, K)`` band PSD of
        every Welch segment (their mean over ``K`` is the Welch band PSD)."""
        _, _, psd = signal.spectrogram(
            values,
            fs=1.0 / setup.dt,
            window=_WINDOW,
            nperseg=setup.nperseg,
            noverlap=setup.noverlap,
            detrend=_DETREND,
            scaling="density",
            mode="psd",
            axis=-2,
        )
        return np.stack(
            [
                psd[..., setup.band_ids == b, :, :].mean(axis=-3)
                for b in range(setup.noise_factors.size)
            ],
            axis=-3,
        )

    def _spectral_rows(
        self, values: np.ndarray, setup: _SpectralSetup, noise: np.ndarray | None
    ) -> np.ndarray:
        """``(..., T, O)`` frames -> ``(..., num_bands, O)`` band rows."""
        bands = self._band_segments(values, setup).mean(axis=-1)
        band_noise = None if noise is None else setup.noise_factors[:, None] * noise
        bands = _with_noise(bands, band_noise)
        if self.spectral["normalize"]:
            # The population variance about the window mean, with its exact
            # (T - 1) / T share of the noise (as the tke row).
            residual, noise_factor = _fluctuations(values, None)
            noise = None if noise is None else noise_factor * noise
            variance = _with_noise(np.mean(residual**2, axis=-2), noise)
            bands = bands / variance[..., None, :]
        if self.spectral["log"]:
            bands = np.log(bands)
        return cast(np.ndarray, bands)

    def _spectral_std(self, values: np.ndarray, setup: _SpectralSetup) -> np.ndarray:
        """Sampling std of the band rows of one ``(T, O)`` series.

        The relative std of band ``b`` comes from the scatter of its ``K``
        Welch segments (adjacent ones correlated ``c``),

            r_b = std_m(P_m) / (sqrt(K) mean_m(P_m)) * sqrt(1 + 2 c (K - 1) / K).

        ``normalize`` divides by the variance, of relative std ``r_V``
        (:func:`_sampling_errors`), of which the band holds the fraction
        ``f_b``; with the bands independent,
        ``r**2 = (1 - f_b)**2 r_b**2 + max(r_V**2 - f_b**2 r_b**2, 0)``.
        ``log`` keeps the relative std, otherwise it is scaled by the row
        value (delta method).
        """
        segments = self._band_segments(values, setup)
        num_segments = segments.shape[-1]
        if num_segments < 2:
            raise ValueError(
                "The spectral sampling error needs at least two Welch "
                "segments: lower spectral.segment_seconds."
            )
        correction = 1.0 + 2.0 * setup.overlap_correlation * (num_segments - 1) / (
            num_segments
        )
        bands = segments.mean(axis=-1)
        relative = segments.std(axis=-1, ddof=1) / bands
        relative = relative * np.sqrt(correction / num_segments)
        variance = values.var(axis=0)
        if self.spectral["normalize"]:
            bins = np.bincount(setup.band_ids[setup.band_ids >= 0])
            df = 1.0 / (setup.nperseg * setup.dt)
            fraction = np.minimum(bins[:, None] * df * bands / variance, 1.0)
            # The band's own share of the variance, plus the rest of the
            # variance (independent of the band; its share clipped at zero).
            rest = (
                _sampling_errors(values)[0][None, :] ** 2 - (fraction * relative) ** 2
            )
            relative = np.sqrt(
                ((1.0 - fraction) * relative) ** 2 + np.maximum(rest, 0.0)
            )
        if self.spectral["log"]:
            return cast(np.ndarray, relative)
        return cast(np.ndarray, relative * self._spectral_rows(values, setup, None))

    # ------------------------------------------------------------------
    # Rows
    # ------------------------------------------------------------------

    def _statistic_rows(
        self, setup: _SpectralSetup | None, observations: xarray.DataArray
    ) -> list[tuple[str, _Rows, _Std, np.ndarray]]:
        """``(block, rows, sampling std, band_frequency)`` of tke, stress and
        spectral for the frames of ``observations``."""
        rows: list[tuple[str, _Rows, _Std, np.ndarray]] = []
        if {"tke", "stress"} & set(self.types):
            partition = self._partition(observations)
        if "tke" in self.types:
            rows.append(
                (
                    "tke",
                    lambda values, noise: self._tke(values, noise, partition),
                    lambda values: self._tke_std(values, partition),
                    np.full(1, np.nan),
                )
            )
        if "stress" in self.types:
            rows.append(
                (
                    "stress",
                    lambda values, noise: self._stress(values, noise, partition),
                    lambda values: self._stress_std(values, partition),
                    np.full(1, np.nan),
                )
            )
        if setup is not None:
            rows.append(
                (
                    "spectral",
                    lambda values, noise: self._spectral_rows(values, setup, noise),
                    lambda values: self._spectral_std(values, setup),
                    setup.band_centres,
                )
            )
        return rows

    def _noise(self, num_obs: int) -> np.ndarray | None:
        """``noise_variance`` as one value per obs, None without noise."""
        if self.noise_variance is None:
            return None
        noise = np.asarray(self.noise_variance, dtype=float)
        if noise.ndim > 1 or noise.size not in (1, num_obs):
            raise ValueError("noise_variance must be a scalar or one value per obs.")
        noise = np.broadcast_to(noise.reshape(-1), (num_obs,))
        if not np.all(np.isfinite(noise)) or np.any(noise < 0):
            raise ValueError("noise_variance must be finite and non-negative.")
        return None if not noise.any() else noise

    def __call__(
        self, observations: xarray.DataArray, *, observed: bool = False
    ) -> xarray.DataArray:
        """Statistics of ``(..., "time", "obs")`` raw sensor series.

        Returns ``(..., "time", "obs")`` with rows stacked along ``"time"``:
        the raw frames, the bin rows (exactly :class:`AggregateObservations`'
        output), the tke row, the stress row, then one spectral row per band.
        A ``"block"`` coordinate along ``"time"`` labels each row;
        ``"band_frequency"`` holds the spectral band centres (Hz, NaN on other
        rows). The statistic rows are labelled with the window's first frame
        time. A stress row narrower than ``obs`` (two components) is padded
        with zeros and a boolean ``"slot_used"`` coordinate on ``("time",
        "obs")`` marks the padding False (only then is it there), which
        ``flatten_observations`` drops.

        Args:
            observations: Raw frames with a seconds-valued ``time`` coordinate,
                optionally with leading dims (``ensemble``).
            observed: True for the noisy observed series (statistics as they
                are; fixes the spectral bands for the next predictions); False
                (the default, model predictions) adds the expected
                ``noise_variance`` contribution to the tke, stress and
                spectral rows.
        """
        if "time" not in observations.dims or "obs" not in observations.dims:
            raise ValueError("Observation statistics need 'time' and 'obs' dims.")
        if "time" not in observations.coords:
            raise ValueError("Observation statistics need a 'time' coordinate.")
        obs = observations.transpose(..., "time", "obs")
        times = np.asarray(obs["time"].values, dtype=float)
        noise = None if observed else self._noise(obs.sizes["obs"])
        setup = self._setup_for(times, observed) if "spectral" in self.types else None

        parts: list[tuple[str, xarray.DataArray, np.ndarray]] = []
        if "raw" in self.types:
            parts.append(("raw", obs, np.full(obs.sizes["time"], np.nan)))
        if self.aggregation is not None:
            binned = self.aggregation(obs)
            parts.append(("bins", binned, np.full(binned.sizes["time"], np.nan)))
        values = np.asarray(obs.values, dtype=float)
        num_obs = obs.sizes["obs"]
        widths = []
        for block, rows, _, frequency in self._statistic_rows(setup, obs):
            data = rows(values, noise)
            num_rows, width = data.shape[-2:]
            if width < num_obs:
                data = np.concatenate(
                    [data, np.zeros((*data.shape[:-1], num_obs - width))], axis=-1
                )
            widths += [width] * num_rows
            part = obs.isel(time=[0] * num_rows).copy(data=data)
            part = part.assign_coords(time=np.full(num_rows, times[0]))
            parts.append((block, part, frequency))

        labelled = [
            part.assign_coords(
                block=("time", [block] * part.sizes["time"]),
                band_frequency=("time", frequency),
            )
            for block, part, frequency in parts
        ]
        result = (
            labelled[0]
            if len(labelled) == 1
            else xarray.concat(
                labelled, dim="time", coords="minimal", compat="override", join="exact"
            )
        )
        if min(widths, default=num_obs) < num_obs:
            # The statistic rows are last: every earlier row is full.
            row_widths = np.full(result.sizes["time"], num_obs)
            row_widths[result.sizes["time"] - len(widths) :] = widths
            result = result.assign_coords(
                slot_used=(
                    ("time", "obs"),
                    np.arange(num_obs)[None, :] < row_widths[:, None],
                )
            )
        return cast(xarray.DataArray, result.transpose(*observations.dims))

    # ------------------------------------------------------------------
    # Layout and error
    # ------------------------------------------------------------------

    def _row_counts(self, observations: xarray.DataArray) -> dict[str, int]:
        counts: dict[str, int] = {}
        if "raw" in self.types:
            counts["raw"] = observations.sizes["time"]
        if self.aggregation is not None:
            counts["bins"] = len(
                self.aggregation.bins(observations, allow_interval_count_change=True)
            )
        if "tke" in self.types:
            counts["tke"] = 1
        if "stress" in self.types:
            counts["stress"] = 1
        if "spectral" in self.types:
            counts["spectral"] = self.spectral["num_bands"]
        return counts

    def _width(self, block: str, num_obs: int) -> int:
        """Entries per row of ``block``: ``num_obs``, for stress one per pair
        and sensor."""
        if block != "stress":
            return num_obs
        return int(self._pairs(num_obs)[0].size) * cast(int, self.num_sensors)

    def block_slices(self, observations: xarray.DataArray) -> dict[str, slice]:
        """Slices of each block in the flattened single-member vector."""
        num_obs = observations.sizes["obs"]
        slices = {}
        start = 0
        for block, count in self._row_counts(observations).items():
            size = count * self._width(block, num_obs)
            slices[block] = slice(start, start + size)
            start += size
        return slices

    @staticmethod
    def _pool_by_height(std: np.ndarray, heights: np.ndarray) -> np.ndarray:
        """Median of ``(rows, O)`` stds over the sensors at one height, per
        row and component (stress: pair; a sensor alone at its height keeps
        its own)."""
        num_sensors = heights.size
        per_component = std.reshape(std.shape[0], -1, num_sensors)
        pooled = per_component.copy()
        for height in np.unique(heights):
            same = heights == height
            pooled[..., same] = np.median(
                per_component[..., same], axis=-1, keepdims=True
            )
        return pooled.reshape(std.shape)

    def resolve_error(
        self,
        observations: xarray.DataArray,
        operator: Any,
        error: ObservationErrorSpec,
    ) -> ResolvedStatisticsError:
        """Block-diagonal error of the statistics of one observed series.

        ``observations``: the noisy observed ``("time", "obs")`` frames. The
        raw block is ``error.resolve(observations, operator)`` and the bins
        block ``error.resolve(observations, operator, aggregation)``,
        unchanged. A tke, stress or spectral block's std is

            sqrt(s**2 + representation_std**2), floored at std_floor,

        with ``s`` the configured float ``std``, or, for ``"auto"``,
        ``sqrt(2)`` times the sampling error of the observed statistic
        estimated from the series itself (tke: the variance of a quadratic
        form with the fluctuations' autocorrelation, see
        ``_residual_sampling_error``; stress: the Gaussian (Isserlis)
        moments of the pair's quadratic forms, see ``_stress_std``; spectral: the scatter of the Welch
        segments, see ``_spectral_std``): the observed and the predicted
        statistic each carry one. With ``pool_std: height`` the auto ``s`` is then replaced
        by its median over the sensors at the same height (per row and
        component or pair). The blocks are diagonal: the sampling errors of
        nearby sensors, of u, v, w, of a sensor's tke and stress and of its tke
        and low bands are in fact correlated, which this does not model.
        """
        resolved = error.resolve(observations, operator, self.aggregation)
        num_obs = observations.sizes["obs"]
        base = getattr(operator, "observation_operator", operator)
        num_sensors = int(base.num_sensors)
        if self.num_sensors is not None and num_sensors != self.num_sensors:
            raise ValueError("num_sensors does not match the operator.")
        times = np.asarray(observations["time"].values, dtype=float)
        values = np.asarray(observations.transpose("time", "obs").values, dtype=float)
        setup = self._spectral_setup(times) if "spectral" in self.types else None

        variances = []
        if "raw" in self.types or "bins" in self.types:
            variances.append(resolved.covariance_diag)
        descriptions = []
        for block, rows, sampling_std, frequency in self._statistic_rows(
            setup, observations
        ):
            options = {"tke": self.tke, "stress": self.stress}.get(block, self.spectral)
            num_rows = frequency.size
            # A constant series (no turbulence, no noise) has no defined
            # statistic: name the first such obs.
            bad = ~(np.isfinite(values).all(axis=0) & (values.var(axis=0) > 0))
            if bad.any():
                j = int(np.nonzero(bad)[0][0])
                raise ValueError(
                    f"The observed {block} statistic of obs {j} (sensor "
                    f"{j % num_sensors}) is undefined: a constant or non-finite "
                    "series."
                )
            if options["std"] == "auto":
                # The predicted statistic has its own sampling error, which the
                # ensemble spread misses when the members share one turbulence
                # realisation (common initial state and inflow): R = S_obs +
                # S_pred, about twice the observed series' sampling variance.
                std = np.sqrt(2.0) * sampling_std(values)
                if options["pool_std"] == "height":
                    heights = getattr(base, "obs_z", None)
                    if heights is None:  # index-based: equal index, equal height
                        heights = base.obs_ids_z
                    std = self._pool_by_height(std, np.asarray(heights, dtype=float))
            else:
                std = np.full((num_rows, self._width(block, num_obs)), options["std"])
            std = np.maximum(
                np.sqrt(np.square(std) + options["representation_std"] ** 2),
                options["std_floor"],
            )
            if not np.all(np.isfinite(std)) or np.any(std <= 0):
                raise ValueError(f"{block} error must be finite and positive.")
            variances.append(np.square(std).reshape(-1))
            log = f"log={options['log']}," if "log" in options else ""
            descriptions.append(
                f"{block}(std={options['std']},pool={options['pool_std']},"
                f"representation={options['representation_std']},"
                f"floor={options['std_floor']},{log}"
                f"normalize={options['normalize']})"
            )
        provenance = resolved.provenance
        if descriptions:
            provenance += "+statistics.v4:" + ",".join(descriptions)
        return ResolvedStatisticsError(
            covariance_diag=np.concatenate(variances),
            # The instrument std is constant in time (ObservationErrorSpec).
            noise_variance=np.asarray(resolved.raw_instrument_variance[0]),
            block_slices=self.block_slices(observations),
            block_rows=self._row_counts(observations),
            provenance=provenance,
        )
