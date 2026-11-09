"""Turbulence statistics of the sensor time series as observations.

Besides the raw frames (``raw``) and their interval means (``bins``, an
:class:`AggregateObservations`), a window's sensor series carries second
moments (``tke``) and spectra (``spectral``) that constrain closure and
roughness parameters which the means barely see. :class:`ObservationStatistics`
maps a raw ``(..., "time", "obs")`` series to the chosen statistics, stacked
along ``"time"`` in the fixed order raw, bins, tke, spectral, and
:meth:`ObservationStatistics.resolve_error` gives the matching block-diagonal
(diagonal) observation error. ``raw`` and ``bins`` are refused together: the
bins are means of the same frames, so the pair counts the data twice. ``tke``
and ``spectral`` are allowed with ``raw`` but are partly redundant with it.

Every block keeps the ``(component, sensor)`` layout of the observation
operator on the ``obs`` axis, so the sensor stays the innermost axis of the
flattened vector (observation ``j`` sits at sensor ``j % num_sensors``) and
distance localization and the adaptive observation error carry over. See
``docs/data_assimilation.md`` (§2, Turbulence-statistics observations).
"""

from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence, cast

import numpy as np
import xarray
from data_assimilation.observation_error import ObservationErrorSpec
from data_assimilation.observation_operator import AggregateObservations
from scipy import signal

#: Statistic types in output order.
STATISTIC_TYPES = ("raw", "bins", "tke", "spectral")

_TKE_DEFAULTS: dict[str, Any] = {
    "normalize": False,
    "log": True,
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
    options["log"] = bool(options["log"])
    if options["pool_std"] == "height" and not (options["log"] or options["normalize"]):
        raise ValueError(
            f"{name}.pool_std 'height' needs log or normalize: stds in absolute "
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


def _with_noise(value: np.ndarray, noise: np.ndarray | None) -> np.ndarray:
    """Add an expected white-noise contribution; the result must be positive
    (a constant or non-finite series has no tke or spectrum to compare)."""
    if noise is not None:
        value = value + noise
    if not np.all(value > 0):
        raise ValueError(
            "A tke or spectral statistic is undefined: a constant or non-finite "
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
    :class:`ObservationStatistics` (rows raw, bins, tke, spectral, sensor
    innermost). ``noise_variance`` is the per-observation raw-frame instrument
    variance (what :attr:`ObservationStatistics.noise_variance` should be set
    to).
    """

    covariance_diag: np.ndarray
    noise_variance: np.ndarray
    block_slices: dict[str, slice]
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
    - ``tke``: one row, ``k_cs = 0.5 var_t(u_cs)`` (population variance,
      ``ddof=0``); the resolved TKE of sensor ``s`` is ``sum_c k_cs``.
      Options ``normalize`` (divide by ``0.5 |U_s|**2``,
      ``|U_s|**2 = sum_c mean_t(u_cs)**2``; needs ``num_sensors``), ``log``;
    - ``spectral``: one row per band, the mean one-sided Welch PSD (Hann
      window, constant detrend, 50% overlap) in ``num_bands`` log-spaced bands
      between ``f_min`` (default the first nonzero Welch frequency) and
      ``f_max`` (default Nyquist). Options ``segment_seconds`` (default a
      quarter of the window), ``normalize`` (divide by the series variance:
      the spectral shape only), ``log``.

    Instrument noise: the observed (noisy) series' statistics are taken as
    they are; the predictions instead get the expected white-noise
    contribution of ``noise_variance`` added before normalisation and log
    (variance ``+ sigma**2``, band PSD ``+`` the exact Welch level of white
    noise), so both sides carry the same bias. ``raw`` and ``bins`` are never
    changed.

    An ``observed=True`` call fixes the spectral bands (frame count, spacing,
    segments) for the following predictions, which must have the same frame
    count and spacing; before any observed call a prediction uses its own.

    ``std``, ``pool_std``, ``representation_std`` and ``std_floor`` set the
    tke and spectral error blocks, see :meth:`resolve_error`.

    Args:
        types: Non-empty subset of ``("raw", "bins", "tke", "spectral")``;
            ``raw`` and ``bins`` are mutually exclusive.
        aggregation: The interval aggregation, required with ``bins`` and
            ignored without.
        tke: ``tke`` options (``None``: defaults).
        spectral: ``spectral`` options (``None``: defaults).
        num_sensors: Sensors per component, required by the tke ``normalize``
            option to sum the components of a sensor.
        noise_variance: Raw-frame instrument variance, a scalar or one value
            per obs (``None`` or 0: no noise), added to the predicted tke and
            spectral statistics.
    """

    def __init__(
        self,
        types: Sequence[str],
        aggregation: AggregateObservations | None = None,
        tke: Mapping[str, Any] | None = None,
        spectral: Mapping[str, Any] | None = None,
        num_sensors: int | None = None,
        noise_variance: np.ndarray | float | None = None,
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

    def _tke(self, values: np.ndarray, noise: np.ndarray | None) -> np.ndarray:
        """``(..., T, O)`` frames -> ``(..., 1, O)`` tke row."""
        tke = 0.5 * _with_noise(values.var(axis=-2), noise)
        if self.tke["normalize"]:
            tke = tke / np.maximum(0.5 * self._speed2(values.mean(axis=-2)), _TINY)
        if self.tke["log"]:
            tke = np.log(tke)
        return tke[..., None, :]

    def _tke_std(self, values: np.ndarray) -> np.ndarray:
        """Sampling std of the tke row of one ``(T, O)`` series: the relative
        std of the window variance (:func:`_sampling_errors`), with
        ``normalize`` that of ``|U_s|**2`` added (independent components,
        ``var |U|**2 = sum_c 4 mean_c**2 var(mean_c)``), times the row value
        unless it is a log (delta method)."""
        relative, mean_variance = _sampling_errors(values)
        if self.tke["normalize"]:
            mean = values.mean(axis=0)
            speed2 = self._speed2(mean)
            speed2_variance = self._speed2(np.sqrt(4.0 * mean**2 * mean_variance))
            relative = np.sqrt(
                relative**2 + speed2_variance / np.maximum(speed2, _TINY) ** 2
            )
        std = relative[None, :]
        if not self.tke["log"]:
            std = std * self._tke(values, None)[0]
        return std

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
            bands = bands / _with_noise(values.var(axis=-2), noise)[..., None, :]
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
        self, setup: _SpectralSetup | None
    ) -> list[tuple[str, _Rows, _Std, np.ndarray]]:
        """``(block, rows, sampling std, band_frequency)`` of tke and spectral."""
        rows: list[tuple[str, _Rows, _Std, np.ndarray]] = []
        if "tke" in self.types:
            rows.append(("tke", self._tke, self._tke_std, np.full(1, np.nan)))
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
        output), the tke row, then one spectral row per band. A ``"block"``
        coordinate along ``"time"`` labels each row; ``"band_frequency"``
        holds the spectral band centres (Hz, NaN on other rows). The tke and
        spectral rows are labelled with the window's first frame time.

        Args:
            observations: Raw frames with a seconds-valued ``time`` coordinate,
                optionally with leading dims (``ensemble``).
            observed: True for the noisy observed series (statistics as they
                are; fixes the spectral bands for the next predictions); False
                (the default, model predictions) adds the expected
                ``noise_variance`` contribution to the tke and spectral rows.
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
        for block, rows, _, frequency in self._statistic_rows(setup):
            data = rows(values, noise)
            num_rows = data.shape[-2]
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
        if "spectral" in self.types:
            counts["spectral"] = self.spectral["num_bands"]
        return counts

    def block_slices(self, observations: xarray.DataArray) -> dict[str, slice]:
        """Slices of each block in the flattened single-member vector."""
        num_obs = observations.sizes["obs"]
        slices = {}
        start = 0
        for block, count in self._row_counts(observations).items():
            slices[block] = slice(start, start + count * num_obs)
            start += count * num_obs
        return slices

    @staticmethod
    def _pool_by_height(std: np.ndarray, heights: np.ndarray) -> np.ndarray:
        """Median of ``(rows, O)`` stds over the sensors at one height, per
        row and component (a sensor alone at its height keeps its own)."""
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
        unchanged. A tke or spectral block's std is

            sqrt(s**2 + representation_std**2), floored at std_floor,

        with ``s`` the configured float ``std``, or, for ``"auto"``,
        ``sqrt(2)`` times the sampling error of the observed statistic
        estimated from the series itself (tke: Bartlett's formulas with the
        series' autocorrelation; spectral: the scatter of the Welch segments,
        see ``_spectral_std``): the observed and the predicted statistic each
        carry one. With ``pool_std: height`` the auto ``s`` is then replaced
        by its median over the sensors at the same height (per row and
        component). The blocks are diagonal: the sampling errors of nearby
        sensors, of u, v, w and of a sensor's tke and low bands are in fact
        correlated, which this does not model.
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
        for block, rows, sampling_std, frequency in self._statistic_rows(setup):
            options = self.tke if block == "tke" else self.spectral
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
                std = np.full((num_rows, num_obs), options["std"])
            std = np.maximum(
                np.sqrt(np.square(std) + options["representation_std"] ** 2),
                options["std_floor"],
            )
            if not np.all(np.isfinite(std)) or np.any(std <= 0):
                raise ValueError(f"{block} error must be finite and positive.")
            variances.append(np.square(std).reshape(-1))
            descriptions.append(
                f"{block}(std={options['std']},pool={options['pool_std']},"
                f"representation={options['representation_std']},"
                f"floor={options['std_floor']},log={options['log']},"
                f"normalize={options['normalize']})"
            )
        provenance = resolved.provenance
        if descriptions:
            provenance += "+statistics.v3:" + ",".join(descriptions)
        return ResolvedStatisticsError(
            covariance_diag=np.concatenate(variances),
            # The instrument std is constant in time (ObservationErrorSpec).
            noise_variance=np.asarray(resolved.raw_instrument_variance[0]),
            block_slices=self.block_slices(observations),
            provenance=provenance,
        )
