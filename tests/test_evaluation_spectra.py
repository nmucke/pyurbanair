"""WP3.2(b): probe spectra, the log-spectral distance and figure S4 (metrics doc §4.3).

The spectral check is the one item in the suite that catches an over-smoothed or
collapsed flow -- one that has the right time-mean and can be tuned to the right
total variance while carrying that variance at the wrong frequencies. That makes
it easy to ship something that *looks* right: a spectrum estimator quietly
disagreeing with itself between the truth and the members produces a plausible
figure and a meaningless number. So what is pinned here is the estimator's
discipline rather than the plotting:

  * **Welch recovers a known slope.** On synthetic colored noise with
    ``E(f) ~ f^s`` the estimator returns ``s`` over the band it is scored on, and
    its normalization is a density (``sum(E)*df`` is the variance)
    (``..._recovers_the_slope_...``, ``..._is_a_density``);
  * **the estimator is shared, not merely similar.** One segment *duration* for
    the truth and every member; a record that cannot carry it, or a frequency grid
    that does not line up, is refused rather than compared bin against
    non-matching bin (``..._refuses_a_segment_longer_than_the_record``,
    ``..._restricts_to_the_common_band_...``);
  * **LSD is exactly zero on identical spectra** and exactly ``10log10(r)`` on a
    constant ratio ``r`` -- the two values the formula can be checked against
    (``..._is_zero_...``, ``..._is_the_decibel_ratio``);
  * **sensors are paired on their coordinates, not on file order.** Truth at one
    point against a member at another renders exactly like a correct figure
    (``..._pairs_sensors_by_location``);
  * **master-plan invariant 3**: no probe records, a smoke-short record, a missing
    component or a sensor set with nothing in common all no-op with a log rather
    than raising, and the metric/figure stages skip their block
    (``..._no_ops_...``, ``..._is_absent_without_probe_records``);
  * **the SHIPPED probe cadence scores a band worth scoring.** Everything else
    here picks its own record length, so nothing exercised the estimator at the
    length `conf/run_probe_series.yaml` actually produces -- which is how a
    default landing exactly on the refusal floor shipped
    (``test_the_shipped_probe_cadence_scores_a_band_wider_than_the_refusal_floor``);
  * **the reported keys are present and finite on a short record** -- smoke-scale
    spectra are noise-dominated, so their *values* are not assertable and only
    their presence and finiteness are (``..._are_finite_on_a_short_record``);
  * **S4's guide slope is not drawn through the data** (the metrics doc §7.5
    requirement that is trivial to get wrong) and its annotated LSD is the same
    number the summary reports (``..._guide_segment_sits_above_the_data``,
    ``..._annotates_the_same_distance_the_metric_reports``).

The probe records are fabricated here rather than produced by a run: their schema
is fixed by ``scripts/esmda/run_probe_series.py`` and building them directly keeps
this a unit suite. Figure content is read back through a ``Figure.savefig`` spy,
as in ``test_evaluation_figures.py``.

Specification: ``docs/plans/esmda_evaluation/phase3_run_upgrades.md`` (WP3.2);
definitions and figure conventions: ``docs/plans/esmda_turbulence_evaluation.md``
§4.3 and §7.
"""

from __future__ import annotations

import logging
import pathlib
from collections.abc import Iterator, Sequence

import numpy as np
import pytest
import xarray
import yaml  # type: ignore[import-untyped]
from evaluation.figures import plot_spectra
from evaluation.turbulence import (
    _MIN_BAND_BINS,
    SPECTRAL_BAND_DECADE_BINS,
    SPECTRUM_CUTOFF_FRACTION,
    SPECTRUM_SEGMENTS,
    log_spectral_distance,
    median_spectrum,
    minimum_spectral_samples,
    probe_spectra,
    spectral_band_bins,
    spectral_metric_summary,
    welch_spectrum,
)
from matplotlib.figure import Figure

# Long enough that the estimator's own scatter is small: 2048-sample segments,
# 15 of them averaged at 50 % overlap.
_N_LONG = 1 << 14

# Short enough to be noise-dominated, long enough to leave the four in-band bins
# the metric requires -- the shape the plan's "assert presence and finiteness
# only" applies to.
_N_SHORT = 512


# ---------------------------------------------------------------------------
# Synthetic records
# ---------------------------------------------------------------------------


def _colored_noise(
    n_time: int, fs: float, slope: float, rng: np.random.Generator
) -> np.ndarray:
    """A series whose power spectral density follows ``f**slope``.

    Built in the frequency domain (random phases on a power-law amplitude) rather
    than by filtering, so the spectrum being recovered is a construction rather
    than another estimate.
    """
    frequencies = np.fft.rfftfreq(n_time, d=1.0 / fs)
    amplitude = np.zeros_like(frequencies)
    amplitude[1:] = frequencies[1:] ** (0.5 * slope)
    phase = rng.uniform(0.0, 2.0 * np.pi, frequencies.size)
    return np.asarray(np.fft.irfft(amplitude * np.exp(1j * phase), n=n_time))


def _field(
    shape: tuple[int, ...], fs: float, slope: float, rng: np.random.Generator
) -> np.ndarray:
    """``(..., time, sensor)`` colored noise, one independent series per column."""
    n_time, n_sensor = shape[-2], shape[-1]
    lead = int(np.prod(shape[:-2])) if len(shape) > 2 else 1
    columns = np.stack(
        [_colored_noise(n_time, fs, slope, rng) for _ in range(lead * n_sensor)]
    )
    return columns.reshape(shape[:-2] + (n_sensor, n_time)).swapaxes(-1, -2)


def _components(
    shape: tuple[int, ...], fs: float, slope: float, rng: np.random.Generator
) -> list[np.ndarray]:
    """``[u, v, w]`` colored-noise fields on one record's shape."""
    return [_field(shape, fs, slope, rng) for _ in range(3)]


def _probes(
    components: Sequence[np.ndarray],
    fs: float,
    *,
    sets: Sequence[str] | None = None,
    positions: np.ndarray | None = None,
    window_index: int = 2,
    with_sensor_coords: bool = True,
) -> xarray.Dataset:
    """A probe record in the schema ``run_probe_series.py`` writes."""
    u, v, w = components
    dims = ("ensemble", "time", "sensor") if u.ndim == 3 else ("time", "sensor")
    n_time, n_sensor = u.shape[-2], u.shape[-1]
    xyz = (
        positions
        if positions is not None
        else np.stack(
            [
                np.arange(n_sensor, dtype=float),
                np.full(n_sensor, 5.0),
                np.linspace(1.0, 4.0, n_sensor),
            ],
            axis=1,
        )
    )
    coords: dict[str, object] = {
        "time": np.arange(n_time, dtype=float) / fs,
        "sensor": np.arange(n_sensor),
    }
    if with_sensor_coords:
        coords["sensor_x"] = ("sensor", xyz[:, 0])
        coords["sensor_y"] = ("sensor", xyz[:, 1])
        coords["sensor_z"] = ("sensor", xyz[:, 2])
        coords["sensor_set"] = (
            "sensor",
            list(sets) if sets is not None else ["assimilation"] * n_sensor,
        )
    return xarray.Dataset(
        {"u": (dims, u), "v": (dims, v), "w": (dims, w)},
        coords=coords,
        attrs={
            "output_frequency": 1.0 / fs,
            "window_index": window_index,
            "member_source": "posterior",
            "spinup_time": 30.0,
        },
    )


@pytest.fixture  # type: ignore[misc]
def records() -> tuple[xarray.Dataset, xarray.Dataset]:
    """A truth record and a 6-member posterior record at the same cadence."""
    rng = np.random.default_rng(7)
    fs, n_sensor, n_members = 1.0, 3, 6
    truth = _probes(
        _components((_N_LONG, n_sensor), fs, -5.0 / 3.0, rng),
        fs,
        sets=["assimilation", "assimilation", "validation"],
    )
    posterior = _probes(
        _components((n_members, _N_LONG, n_sensor), fs, -5.0 / 3.0, rng), fs
    )
    return truth, posterior


def _copy_members(truth: xarray.Dataset, n_members: int) -> xarray.Dataset:
    """A member record whose every member is a copy of the truth's series."""
    stacked = {
        name: np.repeat(truth[name].values[None], n_members, axis=0)
        for name in ("u", "v", "w")
    }
    fs = 1.0 / float(truth.attrs["output_frequency"])
    return _probes(
        [stacked["u"], stacked["v"], stacked["w"]],
        fs,
        sets=[str(s) for s in np.asarray(truth["sensor_set"].values).ravel()],
    )


# ---------------------------------------------------------------------------
# welch_spectrum
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("slope", [-5.0 / 3.0, -1.0, 0.0])  # type: ignore[misc]
def test_welch_spectrum_recovers_the_slope_of_colored_noise(slope: float) -> None:
    """The whole diagnostic rests on this: a known power law comes back."""
    fs = 4.0
    series = _colored_noise(_N_LONG, fs, slope, np.random.default_rng(3))
    nperseg = _N_LONG // SPECTRUM_SEGMENTS
    frequencies, psd = welch_spectrum(series, fs, nperseg)

    # Scored over the band the metric uses, from a few bins in (the first bins of
    # a Hann-windowed segment carry the window's own leakage) up to the cutoff.
    cutoff = SPECTRUM_CUTOFF_FRACTION * 0.5 * fs
    band = (frequencies > 5.0 * fs / nperseg) & (frequencies < cutoff)
    fitted = np.polyfit(np.log10(frequencies[band]), np.log10(psd[band]), 1)[0]
    assert abs(fitted - slope) < 0.1, f"recovered {fitted:.3f}, built {slope:.3f}"


def test_welch_spectrum_is_a_density_not_a_periodogram() -> None:
    """``sum(E)*df`` is the variance: the premultiplied figure assumes a density."""
    fs = 2.0
    rng = np.random.default_rng(11)
    series = rng.normal(size=_N_LONG)
    frequencies, psd = welch_spectrum(series, fs, _N_LONG // SPECTRUM_SEGMENTS)
    df = float(frequencies[1] - frequencies[0])
    # The dropped DC bin costs the mean, which detrending removed anyway.
    assert psd.sum() * df == pytest.approx(series.var(), rel=0.05)


def test_welch_spectrum_drops_the_zero_frequency_bin() -> None:
    """DC carries leakage after detrending, and has no place on a log axis."""
    frequencies, psd = welch_spectrum(np.ones(64) * 3.0 + np.arange(64), 1.0, 16)
    assert frequencies[0] > 0.0
    assert frequencies.size == psd.shape[-1] == 8


def test_welch_spectrum_keeps_the_leading_dims_and_one_grid() -> None:
    """``(M, sensor, time)`` in, ``(M, sensor, freq)`` out, all on one grid."""
    rng = np.random.default_rng(5)
    series = rng.normal(size=(4, 3, 1024))
    frequencies, psd = welch_spectrum(series, 1.0, 128)
    assert psd.shape == (4, 3, frequencies.size)
    # Same (fs, nperseg) on a different record length -> the identical grid, which
    # is what makes the truth and the members comparable bin by bin.
    other, _ = welch_spectrum(rng.normal(size=512), 1.0, 128)
    assert np.array_equal(frequencies, other)


def test_welch_spectrum_refuses_a_segment_longer_than_the_record() -> None:
    """scipy would silently shorten it -- onto a different frequency grid."""
    with pytest.raises(ValueError, match="exceeds the 64 samples"):
        welch_spectrum(np.zeros(64), 1.0, 128)
    with pytest.raises(ValueError, match="at least 4 samples"):
        welch_spectrum(np.zeros(64), 1.0, 2)
    with pytest.raises(ValueError, match="positive frequency"):
        welch_spectrum(np.zeros(64), 0.0, 8)


def test_welch_spectrum_nulls_a_row_with_a_gap_in_it() -> None:
    """A member whose series has a gap gets no spectrum, not a spectrum over it."""
    rng = np.random.default_rng(1)
    series = rng.normal(size=(2, 256))
    series[1, 100] = np.nan
    _, psd = welch_spectrum(series, 1.0, 32)
    assert np.isfinite(psd[0]).all()
    assert not np.isfinite(psd[1]).any()


# ---------------------------------------------------------------------------
# log_spectral_distance
# ---------------------------------------------------------------------------


def test_log_spectral_distance_is_zero_on_identical_spectra() -> None:
    """Exactly zero, not nearly: the floor it is read against is not free."""
    rng = np.random.default_rng(2)
    spectrum = np.abs(rng.normal(size=(3, 20))) + 0.1
    assert log_spectral_distance(spectrum, spectrum.copy()) == pytest.approx(
        np.zeros(3), abs=0.0
    )
    assert float(log_spectral_distance(spectrum[0], spectrum[0].copy())) == 0.0


def test_log_spectral_distance_is_the_decibel_ratio() -> None:
    """A constant factor ``r`` between two spectra is ``10*log10(r)`` dB."""
    spectrum = np.linspace(1.0, 5.0, 32)
    assert float(log_spectral_distance(spectrum, 2.0 * spectrum)) == pytest.approx(
        10.0 * np.log10(2.0)
    )
    # Sign-free: the same distance whichever side carries the surplus.
    assert float(log_spectral_distance(2.0 * spectrum, spectrum)) == pytest.approx(
        10.0 * np.log10(2.0)
    )


def test_log_spectral_distance_broadcasts_over_members_and_sensors() -> None:
    truth = np.abs(np.random.default_rng(4).normal(size=(3, 16))) + 1.0
    members = np.stack([truth, 2.0 * truth, 4.0 * truth])
    distances = log_spectral_distance(truth, members)
    assert distances.shape == (3, 3)
    assert distances[0] == pytest.approx(np.zeros(3))
    assert distances[2] == pytest.approx(np.full(3, 10.0 * np.log10(4.0)))


def test_log_spectral_distance_drops_unusable_bins_rather_than_the_row() -> None:
    """One zero-energy bin is a property of one estimate, not of the pair."""
    truth = np.array([1.0, 2.0, 0.0, 4.0, np.nan])
    member = np.array([2.0, 4.0, 1.0, 8.0, 1.0])
    # Bin 2 (zero energy) and bin 4 (a gap) drop out; the three that remain are a
    # constant factor of two apart, and the distance is an rms magnitude.
    assert float(log_spectral_distance(truth, member)) == pytest.approx(
        10.0 * np.log10(2.0), abs=1e-12
    )
    assert not np.isfinite(float(log_spectral_distance(np.zeros(4), np.zeros(4))))


def test_log_spectral_distance_refuses_two_different_grids() -> None:
    with pytest.raises(ValueError, match="one frequency grid"):
        log_spectral_distance(np.ones(8), np.ones(9))


# ---------------------------------------------------------------------------
# probe_spectra
# ---------------------------------------------------------------------------


def test_probe_spectra_returns_matched_arrays_on_one_grid(
    records: tuple[xarray.Dataset, xarray.Dataset],
) -> None:
    truth, posterior = records
    bundle = probe_spectra(truth, posterior)
    assert bundle is not None

    n_freq = bundle["f"].size
    assert bundle["truth"].shape == (3, n_freq)
    assert bundle["posterior"].shape == (6, 3, n_freq)
    assert bundle["truth_halves"].shape == (2, 3, n_freq)
    assert bundle["prior"] is None
    assert bundle["sensor_sets"] == ("assimilation", "assimilation", "validation")

    # One shared segment duration, and the cutoff at a quarter of Nyquist.
    assert bundle["segment_seconds"] == pytest.approx(_N_LONG / SPECTRUM_SEGMENTS)
    assert bundle["f_cutoff"] == pytest.approx(SPECTRUM_CUTOFF_FRACTION * 0.5)
    assert bundle["band"].sum() > 0
    assert bundle["f"][bundle["band"]].max() < bundle["f_cutoff"]

    # The premultiplication scale is the truth's own resolved variance.
    total = sum(float(truth[name].values.var(ddof=1, axis=0)[0]) for name in "uvw")
    assert float(bundle["variance"][0]) == pytest.approx(total, rel=1e-6)


def test_probe_spectra_reads_the_cadence_off_the_time_axis(
    records: tuple[xarray.Dataset, xarray.Dataset],
) -> None:
    """The achieved cadence is the time axis; a stale attribute must not win."""
    truth, posterior = records
    lying = truth.assign_attrs(output_frequency=10.0)  # the time axis says 1 s
    bundle = probe_spectra(lying, posterior)
    assert bundle is not None
    assert bundle["sample_frequency"]["truth"] == pytest.approx(1.0)


def test_probe_spectra_uses_the_attribute_when_there_is_no_time_axis(
    records: tuple[xarray.Dataset, xarray.Dataset],
) -> None:
    truth, posterior = records
    bundle = probe_spectra(truth.drop_vars("time"), posterior)
    assert bundle is not None
    assert bundle["sample_frequency"]["truth"] == pytest.approx(1.0)


def test_probe_spectra_pairs_sensors_by_location_not_by_order(
    records: tuple[xarray.Dataset, xarray.Dataset],
) -> None:
    """Truth at one point against a member at another renders as a correct figure."""
    truth, _ = records
    members = _copy_members(truth, 2)
    reversed_members = members.isel(sensor=slice(None, None, -1))

    bundle = probe_spectra(truth, reversed_members)
    assert bundle is not None
    # Every member is a copy of the truth, so a correctly paired bundle has
    # identical spectra -- and a positionally paired one does not.
    assert bundle["posterior"][0] == pytest.approx(bundle["truth"])
    assert float(
        np.max(
            log_spectral_distance(
                bundle["truth"][:, bundle["band"]],
                bundle["posterior"][0][:, bundle["band"]],
            )
        )
    ) == pytest.approx(0.0)


def test_probe_spectra_scores_only_the_sensors_both_records_hold(
    records: tuple[xarray.Dataset, xarray.Dataset],
) -> None:
    truth, posterior = records
    bundle = probe_spectra(truth, posterior.isel(sensor=[0, 2]))
    assert bundle is not None
    assert bundle["truth"].shape[0] == 2
    assert bundle["sensor_sets"] == ("assimilation", "validation")


def test_probe_spectra_restricts_to_the_common_band_when_cadences_differ() -> None:
    """A truth that could not be re-run: shared segment *duration*, lower Nyquist.

    No resampling anywhere -- interpolating the coarse record onto the fine axis
    would invent exactly the high-frequency content under test -- so the
    comparison keeps the band both records resolve.
    """
    rng = np.random.default_rng(9)
    fs_fine, fs_coarse = 1.0, 0.1
    n_fine = _N_LONG
    n_coarse = n_fine // 10
    truth = _probes(_components((n_coarse, 2), fs_coarse, -1.0, rng), fs_coarse)
    posterior = _probes(_components((3, n_fine, 2), fs_fine, -1.0, rng), fs_fine)

    bundle = probe_spectra(truth, posterior)
    assert bundle is not None
    # The cutoff follows the *coarser* record's Nyquist, and the segment duration
    # is the truth's -- so both grids step by the same df and line up exactly.
    assert bundle["f_cutoff"] == pytest.approx(
        SPECTRUM_CUTOFF_FRACTION * 0.5 * fs_coarse
    )
    assert bundle["segment_seconds"] == pytest.approx(
        (n_coarse // SPECTRUM_SEGMENTS) / fs_coarse
    )
    assert float(bundle["f"].max()) <= 0.5 * fs_coarse * (1.0 + 1e-9)
    assert bundle["truth"].shape[-1] == bundle["posterior"].shape[-1]


def test_probe_spectra_prior_cannot_move_the_posterior_number(
    records: tuple[xarray.Dataset, xarray.Dataset],
) -> None:
    """The optional record must not touch a mandatory one. Bit-identical, not close.

    A prior at half the cadence would drag ``min(fs)`` down if it took part in
    choosing the band: ``f_cutoff`` would halve, the bins under it with it, and
    ``lsd_posterior_median`` would then depend on whether an optional rerun
    happened. Same for a prior covering fewer sensors.
    """
    truth, posterior = records
    rng = np.random.default_rng(41)
    without = spectral_metric_summary(probe_spectra(truth, posterior) or {})

    fs_coarse = 0.5
    n_coarse = int(truth.sizes["time"] * fs_coarse)
    coarse_prior = _probes(
        _components((4, n_coarse, 3), fs_coarse, -1.0, rng), fs_coarse
    )
    sparse_prior = posterior.isel(sensor=[0, 1])

    for label, prior in (("coarse", coarse_prior), ("sparse", sparse_prior)):
        bundle = probe_spectra(truth, posterior, prior)
        assert bundle is not None
        block = spectral_metric_summary(bundle)
        assert block["f_cutoff"] == without["f_cutoff"], label
        assert block["n_band_bins"] == without["n_band_bins"], label
        assert block["n_sensors"] == without["n_sensors"], label
        assert (
            block["lsd_posterior_median"] == without["lsd_posterior_median"]
        ), f"the {label} prior moved the posterior distance"


def test_probe_spectra_keeps_a_coarser_prior_over_the_bins_it_resolves(
    records: tuple[xarray.Dataset, xarray.Dataset],
) -> None:
    """Padded with ``nan`` above its own Nyquist, not dropped and not extrapolated."""
    truth, posterior = records
    rng = np.random.default_rng(42)
    fs_coarse = 0.5
    n_coarse = int(truth.sizes["time"] * fs_coarse)
    prior = _probes(_components((4, n_coarse, 3), fs_coarse, -1.0, rng), fs_coarse)

    bundle = probe_spectra(truth, posterior, prior)
    assert bundle is not None and bundle["prior"] is not None
    resolved = bundle["f"] <= 0.5 * fs_coarse
    assert np.isfinite(bundle["prior"][:, :, resolved]).all()
    assert not np.isfinite(bundle["prior"][:, :, ~resolved]).any()
    # And it still produces a prior distance, over the bins it does have.
    assert np.isfinite(
        float(spectral_metric_summary(bundle)["lsd_prior_median"] or np.nan)
    )


def test_probe_spectra_drops_a_prior_it_cannot_align(
    records: tuple[xarray.Dataset, xarray.Dataset],
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A prior at other sensor locations costs the envelope, never the comparison."""
    truth, posterior = records
    elsewhere = posterior.assign_coords(
        sensor_x=("sensor", np.array([100.0, 101.0, 102.0]))
    )
    with caplog.at_level(logging.INFO, logger="evaluation.turbulence"):
        bundle = probe_spectra(truth, posterior, elsewhere)
    assert bundle is not None
    assert bundle["prior"] is None
    assert "loses its prior envelope" in caplog.text
    assert "lsd_prior_median" not in spectral_metric_summary(bundle)


def test_probe_spectra_deduplicates_a_repeated_probe_point(
    records: tuple[xarray.Dataset, xarray.Dataset],
) -> None:
    """A point listed twice would be two identical panels and double weight."""
    truth, posterior = records
    doubled = truth.isel(sensor=[0, 0, 1, 2])
    bundle = probe_spectra(doubled, posterior)
    assert bundle is not None
    assert bundle["truth"].shape[0] == 3


def test_probe_spectra_refuses_a_single_segment(
    records: tuple[xarray.Dataset, xarray.Dataset],
) -> None:
    """One segment leaves no halves for the floor, which is not an optional part."""
    with pytest.raises(ValueError, match="at least 2"):
        probe_spectra(*records, segments=1)


def test_probe_spectra_carries_a_prior_when_one_was_run(
    records: tuple[xarray.Dataset, xarray.Dataset],
) -> None:
    truth, posterior = records
    bundle = probe_spectra(truth, posterior, posterior.isel(ensemble=slice(0, 2)))
    assert bundle is not None
    assert bundle["prior"] is not None
    assert bundle["prior"].shape == (2, 3, bundle["f"].size)


def test_probe_spectra_treats_a_memberless_record_as_one_member(
    records: tuple[xarray.Dataset, xarray.Dataset],
) -> None:
    truth, posterior = records
    bundle = probe_spectra(truth, posterior.isel(ensemble=0))
    assert bundle is not None
    assert bundle["posterior"].shape == (1, 3, bundle["f"].size)


@pytest.mark.parametrize(  # type: ignore[misc]
    "damage, reason",
    [
        ("short", "not a spectrum"),
        ("component", "missing w"),
        ("no_sensors", "share no sensor"),
        ("no_cadence", "no usable cadence"),
    ],
)
def test_probe_spectra_no_ops_on_records_it_cannot_compare(
    records: tuple[xarray.Dataset, xarray.Dataset],
    caplog: pytest.LogCaptureFixture,
    damage: str,
    reason: str,
) -> None:
    """Invariant 3: an unusable record logs and returns ``None``, never raises."""
    truth, posterior = records
    if damage == "short":
        # The CI smoke shape: a handful of frames per window.
        truth = truth.isel(time=slice(0, 4))
        posterior = posterior.isel(time=slice(0, 4))
    elif damage == "component":
        truth = truth.drop_vars("w")
    elif damage == "no_sensors":
        posterior = posterior.assign_coords(
            sensor_x=("sensor", np.array([100.0, 101.0, 102.0]))
        )
    elif damage == "no_cadence":
        truth = truth.drop_vars("time").drop_attrs()

    with caplog.at_level(logging.INFO, logger="evaluation.turbulence"):
        assert probe_spectra(truth, posterior) is None
    assert reason in caplog.text


def test_probe_spectra_no_ops_when_too_few_bins_fall_under_the_cutoff(
    records: tuple[xarray.Dataset, xarray.Dataset],
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Long enough to segment, too short to leave a band worth comparing."""
    truth, posterior = records
    with caplog.at_level(logging.INFO, logger="evaluation.turbulence"):
        assert (
            probe_spectra(
                truth.isel(time=slice(0, 64)), posterior.isel(time=slice(0, 64))
            )
            is None
        )
    assert "frequency bins fall under" in caplog.text


def test_probe_spectra_pairs_by_position_when_a_record_has_no_coordinates(
    records: tuple[xarray.Dataset, xarray.Dataset],
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The fallback is allowed but never silent."""
    truth, posterior = records
    bare = posterior.drop_vars(["sensor_x", "sensor_y", "sensor_z", "sensor_set"])
    with caplog.at_level(logging.INFO, logger="evaluation.turbulence"):
        bundle = probe_spectra(truth, bare)
    assert bundle is not None
    assert "by position" in caplog.text


# ---------------------------------------------------------------------------
# median_spectrum and the metric block
# ---------------------------------------------------------------------------


def test_median_spectrum_drops_a_member_with_a_gap() -> None:
    """Otherwise one diverged member blanks the median the whole metric uses."""
    members = np.stack([np.full(4, 1.0), np.full(4, 3.0), np.full(4, np.nan)])
    assert median_spectrum(members) == pytest.approx(np.full(4, 2.0))
    assert not np.isfinite(median_spectrum(np.full((2, 3), np.nan))).any()


def test_spectral_metrics_are_zero_against_a_copied_truth(
    records: tuple[xarray.Dataset, xarray.Dataset],
) -> None:
    """A posterior that *is* the truth has no distance -- and still has a floor."""
    truth, _ = records
    bundle = probe_spectra(truth, _copy_members(truth, 4))
    assert bundle is not None
    block = spectral_metric_summary(bundle)
    assert block["lsd_posterior_median"] == pytest.approx(0.0)
    # The floor is the truth against itself over half-records: not zero, which is
    # the whole reason the distance is reported next to it.
    assert block["lsd_truth_floor"] > 0.0
    assert "lsd_prior_median" not in block


def test_spectral_metrics_report_the_prior_when_one_was_run(
    records: tuple[xarray.Dataset, xarray.Dataset],
) -> None:
    truth, posterior = records
    bundle = probe_spectra(truth, posterior, posterior)
    assert bundle is not None
    block = spectral_metric_summary(bundle)
    assert block["lsd_prior_median"] == pytest.approx(block["lsd_posterior_median"])


def test_spectral_metrics_rank_a_collapsed_member_worse_than_a_faithful_one(
    records: tuple[xarray.Dataset, xarray.Dataset],
) -> None:
    """The failure mode the check exists for: energy at the wrong frequencies.

    A member with the *same* total variance as the truth but a much steeper
    spectrum (over-smoothed: the small scales damped, the large ones inflated to
    compensate) must score worse than one that reproduces the truth's slope. No
    mean-field metric separates those two.
    """
    truth, _ = records
    rng = np.random.default_rng(21)
    fs, n_sensor = 1.0, 3
    variance = np.stack(
        [truth[name].values.var(ddof=1, axis=0) for name in ("u", "v", "w")]
    )

    def _members(slope: float) -> xarray.Dataset:
        fields = []
        for component in range(3):
            field = _field((4, _N_LONG, n_sensor), fs, slope, rng)
            # Matched total variance, so only the *shape* differs.
            field *= np.sqrt(variance[component] / field.var(axis=-2, ddof=1))[
                :, None, :
            ]
            fields.append(field)
        return _probes(fields, fs)

    faithful = spectral_metric_summary(
        probe_spectra(truth, _members(-5.0 / 3.0)) or {}
    )["lsd_posterior_median"]
    smoothed = spectral_metric_summary(probe_spectra(truth, _members(-3.0)) or {})[
        "lsd_posterior_median"
    ]
    assert smoothed > faithful


def test_spectral_metrics_are_finite_on_a_short_record() -> None:
    """Smoke-scale spectra are noise: only presence and finiteness are assertable."""
    rng = np.random.default_rng(13)
    fs = 1.0
    truth = _probes(_components((_N_SHORT, 2), fs, -1.0, rng), fs)
    posterior = _probes(_components((2, _N_SHORT, 2), fs, -1.0, rng), fs)
    bundle = probe_spectra(truth, posterior)
    assert bundle is not None

    block = spectral_metric_summary(bundle)
    for key in ("lsd_posterior_median", "lsd_truth_floor", "f_cutoff"):
        assert key in block, f"the plan's {key} is missing"
        assert np.isfinite(float(block[key])), f"{key} is {block[key]}"
    assert block["n_sensors"] == 2
    assert block["n_band_bins"] >= 4


def test_spectral_metrics_report_a_comparable_reference_and_the_member_count(
    records: tuple[xarray.Dataset, xarray.Dataset],
) -> None:
    """The halves distance is ~2x the like-for-like scatter, so both are reported.

    Reading ``lsd_posterior_median`` against ``lsd_truth_floor`` would call a
    posterior twice as far off as it is "indistinguishable from the truth". The
    halved entry is the one to read against, and ``n_members`` is what makes two
    runs' distances comparable at all -- the median of M spectra is smoother the
    larger M is.
    """
    truth, posterior = records
    bundle = probe_spectra(truth, posterior)
    assert bundle is not None
    block = spectral_metric_summary(bundle)

    assert block["n_members"] == posterior.sizes["ensemble"]
    assert block["lsd_truth_floor_comparable"] == pytest.approx(
        block["lsd_truth_floor"] / 2.0
    )
    # And the member count follows the record it was taken over.
    fewer = spectral_metric_summary(
        probe_spectra(truth, posterior.isel(ensemble=slice(0, 2))) or {}
    )
    assert fewer["n_members"] == 2


def test_spectral_metrics_are_yaml_safe(
    records: tuple[xarray.Dataset, xarray.Dataset],
) -> None:
    """``run_summary.yaml`` is written with ``safe_dump``, which refuses numpy."""
    truth, posterior = records
    bundle = probe_spectra(truth, posterior)
    assert bundle is not None
    round_tripped = yaml.safe_load(yaml.safe_dump(spectral_metric_summary(bundle)))
    assert round_tripped["n_sensors"] == 3


# ---------------------------------------------------------------------------
# Figure S4
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)  # type: ignore[misc]
def _close_figures() -> Iterator[None]:
    import matplotlib.pyplot as plt

    yield
    plt.close("all")


@pytest.fixture  # type: ignore[misc]
def captured_figures(monkeypatch: pytest.MonkeyPatch) -> list[Figure]:
    """Every ``Figure`` a plot function saves, in save order (see WP1.5's suite)."""
    saved: list[Figure] = []
    original = Figure.savefig

    def _spy(self: Figure, *args: object, **kwargs: object) -> object:
        saved.append(self)
        return original(self, *args, **kwargs)

    monkeypatch.setattr(Figure, "savefig", _spy)
    return saved


def _draw(
    bundle: dict, output_path: pathlib.Path, **kwargs: object
) -> pathlib.Path | None:
    """S4 from a bundle, the way ``make_esmda_figures.py`` calls it."""
    return plot_spectra(
        bundle["f"],
        bundle["truth"],
        bundle["posterior"],
        output_path,
        prior=bundle["prior"],
        truth_halves=bundle["truth_halves"],
        variance=bundle["variance"],
        f_cutoff=bundle["f_cutoff"],
        sensor_sets=bundle["sensor_sets"],
        **kwargs,  # type: ignore[arg-type]
    )


def test_plot_spectra_writes_a_png_and_returns_its_path(
    records: tuple[xarray.Dataset, xarray.Dataset], tmp_path: pathlib.Path
) -> None:
    bundle = probe_spectra(*records)
    assert bundle is not None
    output = tmp_path / "probe_spectra.png"
    assert _draw(bundle, output) == output
    payload = output.read_bytes()
    assert payload[:8] == b"\x89PNG\r\n\x1a\n"
    assert len(payload) > 1024


def test_plot_spectra_draws_log_log_axes_with_the_cutoff_marked(
    records: tuple[xarray.Dataset, xarray.Dataset],
    tmp_path: pathlib.Path,
    captured_figures: list[Figure],
) -> None:
    bundle = probe_spectra(*records)
    assert bundle is not None
    _draw(bundle, tmp_path / "s4.png")

    panels = captured_figures[0].axes
    ax = panels[0]
    assert (ax.get_xscale(), ax.get_yscale()) == ("log", "log")
    # One shared y scale across the panels: unshared scales across a comparison is
    # one of the three mistakes the metrics doc says sink the figure suite.
    assert {panel.get_ylim() for panel in panels} == {ax.get_ylim()}
    cutoff = float(bundle["f_cutoff"])
    verticals = [
        line
        for line in ax.lines
        if np.allclose(np.asarray(line.get_xdata(), dtype=float), cutoff)
    ]
    assert verticals, "the comparison cutoff is not marked"
    assert verticals[0].get_linestyle() in (":", "dotted")


def test_plot_spectra_guide_segment_sits_above_the_data(
    records: tuple[xarray.Dataset, xarray.Dataset],
    tmp_path: pathlib.Path,
    captured_figures: list[Figure],
) -> None:
    """§7.5: the reference slope must not be drawn *through* the curves."""
    bundle = probe_spectra(*records)
    assert bundle is not None
    _draw(bundle, tmp_path / "s4.png")

    ax = captured_figures[0].axes[0]
    guides = [
        line
        for line in ax.lines
        if line.get_linestyle() in ("--", "dashed")
        and np.asarray(line.get_xdata()).size == 2
    ]
    assert len(guides) == 1, "expected exactly one short guide segment"
    x_guide = np.asarray(guides[0].get_xdata(), dtype=float)
    y_guide = np.asarray(guides[0].get_ydata(), dtype=float)

    # The -2/3 slope, on the log-log axes it is drawn on.
    slope = np.diff(np.log10(y_guide))[0] / np.diff(np.log10(x_guide))[0]
    assert slope == pytest.approx(-2.0 / 3.0, abs=1e-6)

    # And above every curve over its own span.
    for line in ax.lines:
        if line is guides[0]:
            continue
        x = np.asarray(line.get_xdata(), dtype=float)
        y = np.asarray(line.get_ydata(), dtype=float)
        inside = (x >= x_guide[0]) & (x <= x_guide[-1]) & np.isfinite(y)
        if inside.any():
            assert y[inside].max() < y_guide.min()


def test_plot_spectra_annotates_the_same_distance_the_metric_reports(
    records: tuple[xarray.Dataset, xarray.Dataset],
    tmp_path: pathlib.Path,
    captured_figures: list[Figure],
) -> None:
    """One function, one band: the panel and ``run_summary.yaml`` cannot disagree."""
    bundle = probe_spectra(*records)
    assert bundle is not None
    _draw(bundle, tmp_path / "s4.png")

    band = bundle["band"]
    expected = sorted(
        round(
            float(
                log_spectral_distance(
                    bundle["truth"][s, band],
                    median_spectrum(bundle["posterior"])[s, band],
                )
            ),
            2,
        )
        for s in range(bundle["truth"].shape[0])
    )
    annotated = sorted(
        float(text.get_text().split("=")[1].split("dB")[0])
        for ax in captured_figures[0].axes
        for text in ax.texts
        if text.get_text().startswith("LSD =")
    )
    assert annotated == pytest.approx(expected, abs=5e-3)


def test_plot_spectra_puts_the_held_out_sensor_first(
    records: tuple[xarray.Dataset, xarray.Dataset],
    tmp_path: pathlib.Path,
    captured_figures: list[Figure],
) -> None:
    """The held-out spectrum is the informative one; it never loses its slot."""
    bundle = probe_spectra(*records)
    assert bundle is not None
    _draw(bundle, tmp_path / "s4.png", max_sensors=1)

    axes = captured_figures[0].axes
    assert len(axes) == 1
    assert "validation" in axes[0].get_title(loc="left")


def test_plot_spectra_drops_a_sensor_with_no_usable_normalization(
    records: tuple[xarray.Dataset, xarray.Dataset],
    tmp_path: pathlib.Path,
    caplog: pytest.LogCaptureFixture,
    captured_figures: list[Figure],
) -> None:
    """A zero-variance probe must not be drawn raw under an ``f*E/sigma^2`` label.

    The label and the shared y scale are per figure, so one unnormalized panel is
    mislabelled by construction -- and it happens for real: a probe inside a solid
    cell has exactly zero variance.
    """
    bundle = probe_spectra(*records)
    assert bundle is not None
    bundle["variance"] = np.array([bundle["variance"][0], 0.0, np.nan])

    with caplog.at_level(logging.INFO, logger="evaluation.figures"):
        assert _draw(bundle, tmp_path / "s4.png") is not None
    assert "no usable variance" in caplog.text
    # Only the one sensor with a usable scale is drawn, and it is not the dropped
    # validation sensor (index 2), so the held-out slot is *not* silently filled.
    assert len(captured_figures[0].axes) == 1
    assert "assimilation #0" in captured_figures[0].axes[0].get_title(loc="left")


def test_plot_spectra_no_ops_when_no_sensor_can_be_normalized(
    records: tuple[xarray.Dataset, xarray.Dataset], tmp_path: pathlib.Path
) -> None:
    bundle = probe_spectra(*records)
    assert bundle is not None
    bundle["variance"] = np.zeros(3)
    output = tmp_path / "s4.png"
    assert _draw(bundle, output) is None
    assert not output.exists()


def test_plot_spectra_draws_six_panels_by_default(
    tmp_path: pathlib.Path, captured_figures: list[Figure]
) -> None:
    """Six, as S1: the shipped case has 6 assimilation + 4 validation probes, and a
    budget of 4 would leave the figure with no in-sample panel at all."""
    rng = np.random.default_rng(17)
    fs, n_sensor = 1.0, 10
    sets = ["assimilation"] * 6 + ["validation"] * 4
    truth = _probes(
        _components((_N_LONG, n_sensor), fs, -5.0 / 3.0, rng), fs, sets=sets
    )
    posterior = _probes(_components((3, _N_LONG, n_sensor), fs, -5.0 / 3.0, rng), fs)
    bundle = probe_spectra(truth, posterior)
    assert bundle is not None

    _draw(bundle, tmp_path / "s4.png")
    titles = [ax.get_title(loc="left") for ax in captured_figures[0].axes]
    assert len(titles) == 6
    # The four held-out panels first, then in-sample references beside them.
    assert sum("validation" in t for t in titles) == 4
    assert sum("assimilation" in t for t in titles) == 2


def test_plot_spectra_draws_without_a_prior(
    records: tuple[xarray.Dataset, xarray.Dataset], tmp_path: pathlib.Path
) -> None:
    """Prior probe runs are optional -- S4 is then truth vs posterior, not a skip."""
    bundle = probe_spectra(*records)
    assert bundle is not None
    assert bundle["prior"] is None
    assert _draw(bundle, tmp_path / "s4.png") is not None


@pytest.mark.parametrize(  # type: ignore[misc]
    "case", ["all_nan", "no_sensors", "wrong_rank"]
)
def test_plot_spectra_no_ops_on_degenerate_input(
    records: tuple[xarray.Dataset, xarray.Dataset],
    tmp_path: pathlib.Path,
    caplog: pytest.LogCaptureFixture,
    case: str,
) -> None:
    """Invariant 3: nothing to draw returns ``None`` and leaves no half-figure."""
    bundle = probe_spectra(*records)
    assert bundle is not None
    if case == "all_nan":
        bundle["truth"] = np.full_like(bundle["truth"], np.nan)
        bundle["posterior"] = np.full_like(bundle["posterior"], np.nan)
    elif case == "no_sensors":
        bundle["truth"] = bundle["truth"][:0]
        bundle["posterior"] = bundle["posterior"][:, :0]
        bundle["sensor_sets"] = ()
    else:
        bundle["posterior"] = bundle["posterior"][0]  # (sensor, freq): no members

    output = tmp_path / "s4.png"
    with caplog.at_level(logging.INFO, logger="evaluation.figures"):
        assert _draw(bundle, output) is None
    assert not output.exists()
    assert caplog.text


def test_plot_spectra_normalizes_every_curve_by_the_truths_variance(
    records: tuple[xarray.Dataset, xarray.Dataset],
    tmp_path: pathlib.Path,
    captured_figures: list[Figure],
) -> None:
    """A member with half the energy must not overlay the truth exactly.

    Which is what per-curve normalization would do -- and it is the failure this
    whole figure exists to expose.
    """
    truth, _ = records
    halved = _copy_members(truth, 3)
    for name in ("u", "v", "w"):
        halved[name] = halved[name] * np.sqrt(0.5)
    bundle = probe_spectra(truth, halved)
    assert bundle is not None
    _draw(bundle, tmp_path / "s4.png")

    ax = captured_figures[0].axes[0]
    truth_line = next(line for line in ax.lines if line.get_label() == "Truth")
    median_line = next(
        line for line in ax.lines if line.get_label() == "Posterior median"
    )
    ratio = np.asarray(median_line.get_ydata(), dtype=float) / np.asarray(
        truth_line.get_ydata(), dtype=float
    )
    assert np.nanmedian(ratio) == pytest.approx(0.5, rel=1e-6)


# ---------------------------------------------------------------------------
# The stages' wiring: absent probe records cost the block and nothing else
# ---------------------------------------------------------------------------


def test_probe_records_are_absent_without_a_rerun(tmp_path: pathlib.Path) -> None:
    """Every run dir that never had a probe rerun -- which is most of them."""
    from scripts.esmda._esmda_common import probe_record_paths, probe_spectra_bundle

    (tmp_path / "windows").mkdir()
    assert probe_record_paths(tmp_path) is None
    assert probe_spectra_bundle(tmp_path) is None
    # Not even a windows/ directory: an old run dir laid out differently.
    assert probe_spectra_bundle(tmp_path / "missing") is None


def test_probe_records_are_located_by_window_and_prior_is_optional(
    records: tuple[xarray.Dataset, xarray.Dataset], tmp_path: pathlib.Path
) -> None:
    """The rerun covers one window; the truth's ``window_index`` says which."""
    from scripts.esmda._esmda_common import probe_record_paths, probe_spectra_bundle

    truth, posterior = records
    windows = tmp_path / "windows"
    windows.mkdir()
    truth.to_netcdf(tmp_path / "truth_probes.nc")
    posterior.to_netcdf(windows / "window_2_probes.nc")
    # A decoy from another window: the truth names window 2 (see ``_probes``).
    posterior.to_netcdf(windows / "window_5_probes.nc")

    located = probe_record_paths(tmp_path)
    assert located is not None
    assert located[1].name == "window_2_probes.nc"
    assert located[2] is None, "the prior record is optional"
    assert probe_spectra_bundle(tmp_path) is not None

    posterior.to_netcdf(windows / "window_2_probes_prior.nc")
    located = probe_record_paths(tmp_path)
    assert located is not None and located[2] is not None
    bundle = probe_spectra_bundle(tmp_path)
    assert bundle is not None and bundle["prior"] is not None


def _minimal_run_dir(run_dir: pathlib.Path) -> None:
    """The smallest ``skip_viz`` run dir ``compute_metrics`` will process.

    The same shape ``test_evaluation_fair_scores.py`` builds for the metric-stage
    wiring: a saved config, the run metadata and the parameter datasets. Under
    ``skip_viz`` the stage stops before every block that reads the truth, which is
    exactly the path the spectral block has to survive.
    """
    from scripts.esmda._esmda_common import write_yaml

    (run_dir / "windows").mkdir(parents=True)
    write_yaml({"run": {"skip_viz": True}}, run_dir / "config.yaml")
    write_yaml({"configuration": {"ensemble_size": 2}}, run_dir / "run_info.yaml")
    members = xarray.Dataset(
        {"inflow_angle": (("ensemble", "time"), [[0.1, 0.1], [-0.1, -0.1]])},
        coords={"ensemble": [0, 1], "time": [0.0, 1.0]},
    )
    members.to_netcdf(run_dir / "posterior_params.nc")
    members.to_netcdf(run_dir / "prior_params.nc")
    xarray.Dataset(
        {"inflow_angle": (("time",), [0.0, 0.0])}, coords={"time": [0.0, 1.0]}
    ).to_netcdf(run_dir / "true_params.nc")


def test_metric_stage_emits_spectral_metrics_only_with_probe_records(
    records: tuple[xarray.Dataset, xarray.Dataset], tmp_path: pathlib.Path
) -> None:
    """The block lands in ``run_summary.yaml`` -- and is absent, not null, without it.

    Also pins the one deliberate asymmetry: the spectral block is computed under
    ``run.skip_viz``, because it reads the small probe records an explicit rerun
    wrote rather than the multi-GB truth that flag exists to avoid.
    """
    from scripts.esmda._esmda_common import read_yaml
    from scripts.esmda.compute_esmda_metrics import compute_metrics

    run_dir = tmp_path / "run"
    _minimal_run_dir(run_dir)

    compute_metrics(run_dir)
    summary = read_yaml(run_dir / "run_summary.yaml")
    assert "spectral_metrics" not in summary
    assert summary["parameter_metrics"], "the rest of the summary must be intact"

    truth, posterior = records
    truth.to_netcdf(run_dir / "truth_probes.nc")
    posterior.to_netcdf(run_dir / "windows" / "window_2_probes.nc")

    compute_metrics(run_dir)
    block = read_yaml(run_dir / "run_summary.yaml")["spectral_metrics"]
    for key in ("lsd_posterior_median", "lsd_truth_floor", "f_cutoff"):
        assert np.isfinite(float(block[key])), f"{key} is {block[key]}"
    assert "lsd_prior_median" not in block


def test_figure_stage_draws_s4_only_with_probe_records(
    records: tuple[xarray.Dataset, xarray.Dataset], tmp_path: pathlib.Path
) -> None:
    """The wiring, on the WP1.5 suite's own complete run dir.

    ``_figure_run_dir`` is imported rather than re-fabricated: it is 100 lines of
    ``run_esmda.py``'s output layout, and a second copy of that layout in this file
    would be the thing that drifts. Its no-probe half is what
    ``test_evaluation_figures.py`` already runs on every complete run dir -- what is
    new here is that dropping the records in makes S4 appear beside the rest.
    """
    from scripts.esmda.make_esmda_figures import make_figures
    from tests.test_evaluation_figures import _figure_run_dir

    run_dir = _figure_run_dir(tmp_path)
    make_figures(run_dir)
    assert not (run_dir / "probe_spectra.png").exists()

    truth, posterior = records
    truth.to_netcdf(run_dir / "truth_probes.nc")
    posterior.to_netcdf(run_dir / "windows" / "window_2_probes.nc")

    make_figures(run_dir)
    assert (run_dir / "probe_spectra.png").is_file()
    # And the rest of the set is still there: a new figure must not cost an old one.
    assert (run_dir / "sensor_fans.png").is_file()


# ---------------------------------------------------------------------------
# Sample floor and slightly-offset cadences (found on a real pylbm run)
# ---------------------------------------------------------------------------


def test_minimum_spectral_samples_is_the_floor_probe_spectra_enforces() -> None:
    """The advertised sample floor must be the one the bundle actually applies.

    `run_probe_series.py` warns against this number BEFORE paying for a solver
    re-run, so a floor that drifted from the real behaviour would send a user off
    to spend ~20 minutes producing records that cannot be scored.
    """
    from evaluation.turbulence import minimum_spectral_samples

    floor = minimum_spectral_samples()
    rng = np.random.default_rng(0)

    def bundle_for(n_time: int) -> object:
        truth = _probes(
            [rng.normal(size=(n_time, 3)) for _ in range(3)], fs=1.0, window_index=0
        )
        posterior = _probes(
            [rng.normal(size=(4, n_time, 3)) for _ in range(3)], fs=1.0, window_index=0
        )
        return probe_spectra(truth, posterior)

    assert bundle_for(floor - 1) is None
    assert bundle_for(floor) is not None


@pytest.mark.parametrize("n_time", [264, 1024, 2048])  # type: ignore[misc]
def test_probe_spectra_accepts_the_cadence_mismatch_a_real_run_produces(
    n_time: int,
) -> None:
    """A real run's records never share an exact cadence, and must still compare.

    The LBM timestep is derived from each solve's own velocity scale, so truth and
    members land on slightly different achieved cadences -- 1.00079 s vs 0.99917 s
    on the run this was verified against. Parametrized over length on purpose: the
    first attempt at this guard budgeted the *accumulated* frequency offset, which
    grows as k*eps and therefore refused exactly the long records the diagnostic
    wants (it passed at 480 samples and failed at 1024 and 2048).
    """
    rng = np.random.default_rng(1)
    truth = _probes(
        [rng.normal(size=(n_time, 3)) for _ in range(3)],
        fs=1.0 / 1.00079,
        window_index=0,
    )
    posterior = _probes(
        [rng.normal(size=(4, n_time, 3)) for _ in range(3)],
        fs=1.0 / 0.99917,
        window_index=0,
    )

    bundle = probe_spectra(truth, posterior)
    assert bundle is not None
    assert bundle["f"].size > 0
    assert bundle["f_cutoff"] > 0.0


def test_probe_spectra_compares_a_coarser_record_over_the_band_it_resolves() -> None:
    """A genuinely coarser cadence is truncated to its own Nyquist, not refused.

    Sharing the segment *duration* is what makes the bin spacing identical, so a
    coarser record lands on the same bins and simply stops earlier. Refusing it
    would throw away a comparison the design supports -- and an earlier version of
    the grid guard did exactly that for a 1.3 s vs 1.0 s pair, purely as a
    rounding accident.
    """
    rng = np.random.default_rng(2)
    n_time = 1024
    truth = _probes(
        [rng.normal(size=(n_time, 3)) for _ in range(3)], fs=1.0, window_index=0
    )
    coarse = _probes(
        [rng.normal(size=(4, n_time, 3)) for _ in range(3)],
        fs=1.0 / 1.30,
        window_index=0,
    )

    matched = probe_spectra(truth, truth_as_posterior(truth))
    bundle = probe_spectra(truth, coarse)
    assert matched is not None
    assert bundle is not None
    # Same bins, fewer of them: the cutoff follows the coarser record's Nyquist.
    assert bundle["f_cutoff"] < matched["f_cutoff"]
    assert np.allclose(bundle["f"], matched["f"][: bundle["f"].size])


def truth_as_posterior(truth: xarray.Dataset) -> xarray.Dataset:
    """The truth record reshaped as a one-member posterior, for a like-for-like run."""
    return truth.expand_dims(ensemble=1).assign_coords(member=("ensemble", [0]))


# ---------------------------------------------------------------------------
# The SHIPPED probe cadence (conf/run_probe_series.yaml)
# ---------------------------------------------------------------------------

_PROBE_CONFIG = (
    pathlib.Path(__file__).resolve().parents[1] / "conf/run_probe_series.yaml"
)

# The window length figure S4 is specified against: `time.simulation_time` as it
# is COMMITTED in conf/case/xie_and_castro.yaml and conf/case/barcelona.yaml.
# Hard-coded rather than composed, deliberately -- the case files carry live run
# tuning between experiments, and what is under test here is the shipped cadence
# against the documented window, not whatever window someone is mid-run on.
_REFERENCE_WINDOW_SECONDS = 300.0


def _shipped_probe_cadence() -> float:
    """`probes.output_frequency` as an operator gets it, straight off the file."""
    with _PROBE_CONFIG.open() as handle:
        return float(yaml.safe_load(handle)["probes"]["output_frequency"])


def test_the_shipped_probe_cadence_scores_a_band_wider_than_the_refusal_floor() -> None:
    """The default an operator pays for a full re-run to get must be worth having.

    Nothing else in this suite runs the estimator at the length the SHIPPED config
    actually produces -- the synthetic records above pick their own -- which is
    exactly how `probes.output_frequency: 1.0` survived: over the documented 300 s
    window it yields 300 samples, which land on `_MIN_BAND_BINS` EXACTLY. Four
    bins is the point at which `probe_spectra` stops refusing, not a comparison:
    it is a factor-of-four frequency span, an `lsd_posterior_median` that is an
    RMS over four numbers, and (at 0.027-0.108 Hz, U~8 m/s over 14 m blocks) the
    energy-containing range rather than an inertial one -- so S4's `-5/3` guide
    would be drawn over a band that cannot contain a `-5/3`.

    The bar is a decade of scored band, `SPECTRAL_BAND_DECADE_BINS`: bins run
    1..B, so the span is exactly a factor of B, and a slope guide over less than
    a decade is decoration. This asserts the whole chain -- config file to
    `probe_spectra`'s own `band` -- rather than the arithmetic alone.
    """
    cadence = _shipped_probe_cadence()
    n_samples = int(round(_REFERENCE_WINDOW_SECONDS / cadence))
    assert n_samples >= minimum_spectral_samples(SPECTRAL_BAND_DECADE_BINS), (
        f"the shipped {cadence} s cadence yields {n_samples} samples over a "
        f"{_REFERENCE_WINDOW_SECONDS} s window"
    )

    rng = np.random.default_rng(17)
    fs = 1.0 / cadence
    truth = _probes(
        _components((n_samples, 3), fs, -5.0 / 3.0, rng), fs, window_index=0
    )
    posterior = _probes(
        _components((4, n_samples, 3), fs, -5.0 / 3.0, rng), fs, window_index=0
    )
    bundle = probe_spectra(truth, posterior)
    assert bundle is not None, "the shipped cadence does not even produce a bundle"

    scored = bundle["f"][bundle["band"]]
    assert scored.size == spectral_band_bins(n_samples), (
        "spectral_band_bins disagrees with the band probe_spectra applies, so "
        "run_probe_series.py's pre-flight would quote a number the run misses"
    )
    assert scored.size >= SPECTRAL_BAND_DECADE_BINS, (
        f"the shipped {cadence} s cadence scores {scored.size} bins over a "
        f"{_REFERENCE_WINDOW_SECONDS} s window, against the "
        f"{SPECTRAL_BAND_DECADE_BINS} a decade-wide band needs "
        f"(_MIN_BAND_BINS, the refusal floor, is {_MIN_BAND_BINS})"
    )
    # A decade in frequency, not merely a bin count -- the property S4 needs.
    assert float(scored[-1] / scored[0]) >= 10.0


def test_spectral_band_bins_inverts_the_sample_floor() -> None:
    """The pre-flight promises a bin count; it has to be the one that arrives.

    Checked at both edges of the two thresholds `run_probe_series.py` reports,
    because both are quoted to the operator as "use a cadence at least this fine"
    and an off-by-one there is a silently narrow band.
    """
    for bins in (_MIN_BAND_BINS, SPECTRAL_BAND_DECADE_BINS):
        floor = minimum_spectral_samples(bins)
        assert spectral_band_bins(floor) == bins
        assert spectral_band_bins(floor - 1) < bins
    # A record too short to carry one segment scores nothing rather than raising.
    assert spectral_band_bins(8) == 0
