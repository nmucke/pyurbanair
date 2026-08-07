"""Compute the metrics for a finished ESMDA run and write ``run_summary.yaml``.

Second stage of the three-script single-run ESMDA pipeline (see
``scripts/run_esmda_pipeline.sh``):

  1. scripts/esmda/run_esmda.py            -- runs the assimilation, saves the raw
                                       artifacts + truth_access.yaml + run_info.yaml.
  2. scripts/esmda/compute_esmda_metrics.py (THIS) -- reads those, computes the
                                       parameter / state / sensor metrics and
                                       writes run_summary.yaml.
  3. scripts/esmda/make_esmda_figures.py   -- reads those, draws the figures.

``run_summary.yaml`` is the run_info (configuration + timing) saved by
run_esmda.py, augmented with:

  * ``metrics_version``    -- estimator-semantics marker; 2 = fair (``M(M-1)``)
                              pairwise scores. Absent or 1 means the older
                              biased estimators, whose CRPS / energy-score
                              numbers sit ~O(1/M) higher and must not be
                              compared with version-2 ones.
  * ``parameter_metrics``  -- per-parameter RMSE/CRPS summary (+ skill vs prior)
                              and calibration: z-score, normalized error and
                              contraction ratio, plus a pooled z-score entry.
  * ``ensemble_health``    -- duplicate-member counts, run-wide and per window.
  * ``state_metrics``      -- |U| field RMSE summary (streamed over a few z-slices).
  * ``sensor_metrics``     -- per sensor set, the full-vector (u, v, w) RMSE and
                              energy score (multivariate CRPS). The comprehensive
                              per-component sweep series are computed separately by
                              scripts/figure_creation/compute_sweep_metrics.py.
  * ``sensor_statistics``  -- per sensor set, the per-window mean and variance of
                              u/v/w/|U| scored as the verification object (fair
                              CRPS, z-score, rank), prior and posterior, with the
                              identifiability guard. This is the statistics-space
                              counterpart of ``sensor_metrics``: a pointwise
                              time-series error mostly measures turbulent phase,
                              which no parameter estimate controls.
  * ``spectral_metrics``   -- the log-spectral distance between the truth's and
                              the posterior-median frequency spectrum at the
                              probes, beside the truth's own self-distance floor
                              (metrics doc §4.3). Only present when an explicit
                              ``run_probe_series.py`` rerun wrote the high-rate
                              probe records: the assimilation's own output cadence
                              is far too coarse for a Welch spectrum.
  * ``field_metrics``      -- the VDI 3783/9 hit rate ``q`` of the time-mean
                              velocity field over evenly spaced z-slabs, prior
                              and posterior, with the absolute tolerance ``W``
                              block-bootstrapped from the truth's own sampling
                              error. The reduced fields behind it (mean, TKE,
                              ``<u'w'>``, on the slabs and at the station
                              columns) are written beside it as
                              ``eval_fields.nc`` for the figure stage.

Usage::

    python scripts/esmda/compute_esmda_metrics.py --run-dir <esmda output dir>
"""

import argparse
import logging
import pathlib
import re
import sys
import warnings
from collections.abc import Callable
from typing import NamedTuple

import numpy as np
import xarray

import pyurbanair.quiet_jax  # noqa: F401  (suppress JAX CPU-fallback noise)

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from evaluation.scores import (
    METRICS_VERSION,
    data_mismatch_summary,
    ensemble_uniqueness,
    hit_rate,
    parameter_metric_summary,
    series_stats,
    vector_sensor_metrics,
    window_statistics_summary,
)
from evaluation.sensors import window_sampling_std, window_statistics
from evaluation.turbulence import (
    MomentAccumulator,
    block_bootstrap_std,
    colocate_components,
    evenly_spaced_levels,
    extrapolated_centre_dims,
    spectral_metric_summary,
    streaming_state_rmse,
)

from scripts.esmda._esmda_common import (
    build_sensor_sets,
    ensemble_sensor_series,
    load_run_config,
    obs_diagnostics_bundle,
    open_truth,
    probe_spectra_bundle,
    read_yaml,
    truth_sensor_series,
    write_yaml,
)

logger = logging.getLogger(__name__)

# The ``on_member`` callback the streaming extraction hands each materialised
# member slice: ``(window_index, member_index, member_state)``.
_OnMember = Callable[[int, int, xarray.Dataset], None] | None


def _flatten_parameter_members(params: xarray.Dataset) -> np.ndarray:
    """Flatten every ensemble parameter variable into one row per member."""
    arrays: list[np.ndarray] = []
    n_members: int | None = None
    for name in sorted(str(v) for v in params.data_vars):
        variable = params[name]
        if "ensemble" not in variable.dims:
            continue
        values = np.asarray(variable.transpose("ensemble", ...).values)
        if n_members is None:
            n_members = values.shape[0]
        elif values.shape[0] != n_members:
            raise ValueError(
                f"parameter {name!r} has {values.shape[0]} ensemble members; "
                f"expected {n_members}"
            )
        arrays.append(values.reshape(values.shape[0], -1))
    if not arrays:
        raise ValueError(
            "parameter dataset has no variables with an ensemble dimension"
        )
    return np.concatenate(arrays, axis=1)


def _ensemble_health(
    run_dir: pathlib.Path, posterior_params: xarray.Dataset
) -> dict[str, object] | None:
    """Duplicate-member counts for the assembled posterior and each window.

    The assembled posterior concatenates the windows, so two members that were
    cloned in one window but diverge in another are distinct there and only the
    per-window counts see it -- hence both.

    This is a diagnostic, not a metric: a parameter file it cannot read (an old
    layout, a window truncated by a killed job) costs its own count -- a
    ``null`` entry, so the list stays aligned with the window index -- and a log
    line, never the whole metric stage. Returns ``None`` when even the assembled
    posterior cannot be read.

    ``min_over_median_pairwise`` is the near-duplicate detector exact row
    matching cannot be: a ratio near 0 means two members sit far closer to each
    other than the ensemble typically does. It is an unweighted L2 over all
    parameters concatenated, so a parameter with a much larger numerical range
    (an inflow angle in degrees against an SGS constant) dominates it; read it
    as a coarse flag, not a calibrated distance.
    """
    try:
        health = ensemble_uniqueness(_flatten_parameter_members(posterior_params))
    except (OSError, ValueError, KeyError) as exc:
        logger.warning("Cannot assess ensemble health: %s", exc)
        return None

    if health["n_unique"] < health["n_members"]:
        logger.warning(
            "Duplicate posterior parameter members: %s/%s unique -- the fair "
            "scores' M(M-1) corrections overstate the ensemble's spread",
            health["n_unique"],
            health["n_members"],
        )

    per_window: list[int | None] = []

    # Sorted by window index, with anything that does not carry one last: a
    # stray or hand-renamed file in windows/ must not take the sort (and with
    # it the whole metric stage) down.
    def _window_index(path: pathlib.Path) -> tuple[int, int, str]:
        match = re.fullmatch(r"window_(\d+)_posterior_params", path.stem)
        return (0, int(match.group(1)), "") if match else (1, 0, path.stem)

    window_paths = sorted(
        (run_dir / "windows").glob("window_*_posterior_params.nc"), key=_window_index
    )
    for path in window_paths:
        try:
            with xarray.open_dataset(path) as params:
                window_health = ensemble_uniqueness(_flatten_parameter_members(params))
        except (OSError, ValueError, KeyError) as exc:
            logger.warning("Skipping ensemble health for %s: %s", path.name, exc)
            # Keep the slot: consumers index this list by window.
            per_window.append(None)
            continue
        per_window.append(int(window_health["n_unique"]))
        if window_health["n_unique"] < window_health["n_members"]:
            logger.warning(
                "Duplicate posterior parameter members in %s: %s/%s unique",
                path.name,
                window_health["n_unique"],
                window_health["n_members"],
            )

    minimum, median = health["min_pairwise"], health["median_pairwise"]
    return {
        "n_members": int(health["n_members"]),
        "n_unique": int(health["n_unique"]),
        "n_unique_per_window": per_window,
        "min_over_median_pairwise": (
            float(minimum / median)
            if minimum is not None and median is not None and median > 0
            else None
        ),
    }


def _prior_sensor_series(
    run_dir: pathlib.Path,
    sensor_sets: dict,
    num_windows: int,
    sim_time: float,
    solver_name: str,
    on_member: _OnMember = None,
) -> dict | None:
    """Sensor series from the saved prior states, or ``None`` with a log line.

    ``run.save_prior_state`` is off by default (the prior states double the
    state-file bytes a run writes), so the prior half of the statistics block is
    genuinely optional -- absence is the common case, not a fault. A partially
    written set is treated the same way: scoring a prior over some windows and a
    posterior over all of them would put two different horizons in one skill
    score.
    """
    paths = [
        run_dir / "windows" / f"window_{w}_prior_state.nc" for w in range(num_windows)
    ]
    missing = [p for p in paths if not p.exists()]
    if missing:
        logger.info(
            "No prior sensor statistics: %d of %d prior state files are absent "
            "(run.save_prior_state was off, or the run was interrupted)",
            len(missing),
            num_windows,
        )
        return None
    try:
        # ``dict(...)``: the extraction module is mypy-waived and returns ``Any``.
        return dict(
            ensemble_sensor_series(
                paths, sensor_sets, solver_name, sim_time, on_member=on_member
            )
        )
    except (OSError, ValueError, KeyError) as exc:
        # A prior state file that *exists* but cannot be read -- a job killed
        # mid-``to_netcdf``, which is the same interrupted run the absence
        # branch above names. Invariant 3: it costs the prior half, not the
        # whole summary.
        logger.warning("Cannot read the prior sensor series: %s", exc)
        return None


def _sensor_statistics(
    sensor_sets: dict,
    truth_series: dict,
    ensemble_series: dict,
    prior_series: dict | None,
    num_windows: int,
    sim_time: float,
) -> dict:
    """The ``sensor_statistics`` block, per sensor set (metrics doc §4.2).

    Reuses the series the vector sensor metrics already extracted, so the
    posterior half costs no extra pass over the state files; the prior series is
    read by the caller (``compute_metrics``), which is also what decides whether
    the *field* block may use its prior -- one owner for one all-or-nothing
    decision, rather than the same decision made in two places.

    One caveat on the last window: ``truth_sensor_series`` extracts exactly
    ``num_windows * n_per_window`` frames, so when the truth's frame count is
    not a multiple of the window count the trailing frames are dropped and the
    last window's truth statistic is computed over slightly fewer samples than
    its peers. The members are unaffected (each window state file is its own
    file), so this shows up as a marginally noisier truth in the final window.
    """
    # No try/except around the scoring. Everything below is pure computation on
    # series the stage already holds in memory -- the only I/O, the prior read,
    # happened in the caller and has its own guard -- so the exceptions it can
    # raise are bugs, and swallowing them would ship a green run with an empty
    # block.
    statistics = {}
    for name in sensor_sets:
        prior = prior_series[name] if prior_series is not None else None
        statistics[name] = window_statistics_summary(
            window_statistics(truth_series[name], sim_time, num_windows, label=name),
            window_statistics(ensemble_series[name], sim_time, num_windows, label=name),
            prior_stats=(
                window_statistics(prior, sim_time, num_windows, label=name)
                if prior is not None
                else None
            ),
            posterior_sampling_std=window_sampling_std(
                ensemble_series[name], sim_time, num_windows
            ),
            prior_sampling_std=(
                window_sampling_std(prior, sim_time, num_windows)
                if prior is not None
                else None
            ),
            label=name,
        )
    return statistics


# ---------------------------------------------------------------------------
# Mean fields (WP1.4, metrics doc section 4.1)
# ---------------------------------------------------------------------------

_COMPONENTS = ("u", "v", "w")

# Region -> the dims each accumulated quantity carries, once the member axis is
# reduced away. ``mean`` is a vector (one entry per velocity component); ``tke``
# and ``uw`` are scalars per cell.
#
# The variables this produces in ``eval_fields.nc``, so a reader can grep for
# them: ``{truth,prior,posterior}_{slab,station}_{mean,tke,uw}``, each ensemble
# source additionally carrying ``_spread`` and, at the stations, ``_quantile``.
_FIELD_DIMS: dict[str, dict[str, tuple[str, ...]]] = {
    "slab": {
        "mean": ("component", "zlev", "y", "x"),
        "tke": ("zlev", "y", "x"),
        "uw": ("zlev", "y", "x"),
    },
    "station": {
        "mean": ("component", "z", "station"),
        "tke": ("z", "station"),
        "uw": ("z", "station"),
    },
}

_LONG_NAMES = {
    "mean": "time-mean velocity component",
    "tke": "resolved turbulent kinetic energy 0.5*<u_i'u_i'> (resolved only)",
    "uw": "resolved Reynolds stress <u'w'> (resolved only)",
}

# Cell-centre dims, in (z, y, x) order, as they appear after colocation:
# ``zt/yt/xt`` for uDALES, ``z/y/x`` for PALM, pylbm and the surrogate.
_CENTRE_DIM_CANDIDATES = (("z", "zt"), ("y", "yt"), ("x", "xt"))

# Evenly spaced z-levels the horizontal slabs are accumulated on.
_MEAN_FIELD_Z_SLICES = 4

# Ceiling on the accumulator state held for one ensemble during the pass. One
# member costs 80 bytes per slab cell (an int64 count, three float64 means and
# six float64 co-moments), independent of how many frames go past, so at M=128
# and four 512x512 slabs the unstrided cost would be 10 GB. The horizontal
# stride derived from this budget bounds it without a config knob; it is applied
# to the slabs only -- never to the station columns, which are a handful of
# cells, and never to time. Note the posterior and prior collectors are alive at
# the same time, so the pass's persistent worst case is twice this.
_MEAN_FIELD_BUDGET_BYTES = 1 << 30
_MEAN_FIELD_BYTES_PER_CELL = 80

# Ceiling on the *transient* of one accumulation step, which is a different
# quantity from the accumulator state and has to be bounded separately: a chunk
# of frames is materialised as float64 and ``MomentAccumulator.update``
# allocates about ten arrays of that shape. Time is sub-chunked to this bound,
# which the accumulator is designed for -- combining chunks is exactly what it
# does -- so neither term is left unbounded. 128 bytes per (frame, slab cell) is
# measured, not derived: the selection, its float64 upcast, the transpose copy
# and update's own temporaries.
_MEAN_FIELD_TRANSIENT_BYTES = 256 << 20
_MEAN_FIELD_BYTES_PER_SAMPLE = 128

# The obstacle indicator pylbm writes beside the state (non-zero = solid).
# pypalm replaces PALM's NaN with 0.0 and keeps no mask; uDALES ships none.
_BLANKING_VAR = "blanking"

# Points at which the truth's series is kept during its pass so the hit rate's
# absolute tolerance ``W`` can be block-bootstrapped from it.
_W_SAMPLE_POINTS = 64
_W_SAMPLE_SEED = 0

# Quantiles stored at the station columns (figure S1's nested bands). The slabs
# get per-cell ensemble mean and spread only -- quantiles over a full 3-D field
# would multiply ``eval_fields.nc`` by the number of levels stored here.
STATION_QUANTILES = (0.05, 0.25, 0.5, 0.75, 0.95)


class _Target(NamedTuple):
    """The scored region: which cells every source is accumulated on.

    Derived once, from the first state the ensemble collector sees, and then
    handed to the prior and truth collectors so all three describe the same
    cells. Immutable on purpose -- it is shared by aliasing.
    """

    z: np.ndarray  # slab heights
    y: np.ndarray  # slab y coordinates (strided)
    x: np.ndarray  # slab x coordinates (strided)
    z_index: np.ndarray  # native indices of the slab levels
    stride: int
    station_z: np.ndarray  # full column heights
    station_x: np.ndarray
    station_y: np.ndarray
    station_set: tuple[str, ...]  # which sensor set each station came from
    native: tuple[np.ndarray, np.ndarray, np.ndarray]  # the grid it was cut from


def _centre_dims(field: xarray.DataArray) -> tuple[str, str, str]:
    """``(z, y, x)`` dim names of a co-located field."""
    dims = []
    for candidates in _CENTRE_DIM_CANDIDATES:
        found = [d for d in candidates if d in field.dims]
        if not found:
            raise ValueError(
                f"co-located field has none of {candidates} among its dims "
                f"{tuple(field.dims)}"
            )
        dims.append(found[0])
    return (dims[0], dims[1], dims[2])


def _drop_member_axis(field: xarray.DataArray) -> xarray.DataArray:
    """Drop the length-1 ``ensemble`` axis the per-member slices carry."""
    return field.isel(ensemble=0) if "ensemble" in field.dims else field


def _bracket(axis: np.ndarray, value: float) -> slice:
    """The two cells of ``axis`` that surround ``value``.

    Clipped to the ends, so a point outside the domain brackets to an edge pair
    and interpolates to ``nan`` rather than raising -- the same answer the
    whole-field interpolation gave, at a fraction of the memory.
    """
    index = int(np.clip(np.searchsorted(axis, value) - 1, 0, max(axis.size - 2, 0)))
    return slice(index, index + 2)


def _mean_field_stride(n_members: int, n_z: int, n_y: int, n_x: int) -> int:
    """Horizontal stride keeping one ensemble's accumulators inside the budget.

    A knob would be the obvious alternative, and WP1.3 deliberately shipped no
    metric knobs: nothing here needs to vary per run, and the only value a
    caller could sensibly pick is the one that fits in memory -- which is what
    this computes. Logged when it bites, so a strided run says so.
    """
    stride = 1
    while (
        n_members * n_z * -(-n_y // stride) * -(-n_x // stride)
    ) * _MEAN_FIELD_BYTES_PER_CELL > _MEAN_FIELD_BUDGET_BYTES and stride < min(
        n_y, n_x
    ):
        stride += 1
    if stride > 1:
        logger.info(
            "Mean fields: %d members x %d x %d x %d cells would exceed the "
            "%.1f GB accumulator budget -- scoring every %d-th cell horizontally",
            n_members,
            n_z,
            n_y,
            n_x,
            _MEAN_FIELD_BUDGET_BYTES / (1 << 30),
            stride,
        )
    return stride


def station_columns(
    sensor_sets: dict,
) -> tuple[np.ndarray, np.ndarray, tuple[str, ...]]:
    """Station ``(x, y)`` and the sensor set each came from.

    **Both** sets, when the obs config defines validation sensors. Figure S1's
    profiles are the place a held-out column is worth most -- a profile drawn
    only at the points the assimilation was fitted to is the least informative
    one available -- and the columns are a handful of cells either way.

    Empty arrays when there are no sensors at all: the slabs are the scored
    region and they do not depend on this, so a sensorless run loses its station
    columns and keeps its hit rate (invariant 3).
    """
    xs, ys, labels = [np.empty(0)], [np.empty(0)], []
    for name, (sx, sy, _) in sensor_sets.items():
        x = np.asarray(sx, dtype=float).ravel()
        xs.append(x)
        ys.append(np.asarray(sy, dtype=float).ravel())
        labels.extend([name] * int(x.size))
    return np.concatenate(xs), np.concatenate(ys), tuple(labels)


class MeanFieldCollector:
    """Per-member time-mean velocity, TKE and ``<u'w'>`` on a fixed region.

    One :class:`~evaluation.turbulence.MomentAccumulator` per member and region,
    fed from the member slices the sensor pass already materialises (see
    ``ensemble_sensor_series``'s ``on_member``), so the mean fields cost no extra
    read of the window files.

    Two regions, because the two consumers want different things. The **slabs**
    -- a few evenly spaced z-levels at full (or strided) horizontal resolution --
    are what the hit rate scores and what figure F1 draws. The **station
    columns** -- full-depth profiles at the sensor (x, y) -- are what figure S1
    plots, and they are small enough to carry per-member quantiles.

    Accumulation spans windows: a member's statistics cover every frame of every
    window it was fed, which is what makes them comparable with a truth pass over
    the same time range.

    Args:
        solver_name: The staggering the states follow, for colocation.
        station_x, station_y: Station coordinates, in domain coordinates.
        station_set: Which sensor set each station came from, carried through to
            ``eval_fields.nc`` so a figure can label the columns.
        n_members: Ensemble size, used only to derive the horizontal stride.
        target: ``None`` to define the scored region from the first state seen
            (what the ensemble collector does); another collector's
            :attr:`target` to sample onto *that* region instead (what the prior
            and truth collectors do -- see :meth:`_slab`).
        solid_state_path: A window state file to read the backend's obstacle
            indicator from (see :meth:`_record_solid`). ``None`` falls back to
            the indicator on the state itself, if it carries one, and then to the
            truth-variance rule in :func:`_fluid_cells`.
        keep_samples: Retain a fixed sample of slab cells' series so
            :meth:`sampling_tolerance` can bootstrap the hit rate's ``W`` from
            them. Only the truth collector needs it -- ``W`` is a property of the
            truth -- and it is off by default so the ensemble pass does not carry
            M copies of series nothing reads.

    A layout colocation refuses sets :attr:`reason`, stops accumulation and makes
    :meth:`result` return ``None`` rather than propagating: the sensor extraction
    shares the pass and must not be taken down with it. That guard is deliberately
    narrow (invariant 3 is about absent inputs, not absent correctness) -- a bug
    in the accumulation itself raises, because a broken scorer shipping a green
    run is worse than a crash.
    """

    def __init__(
        self,
        solver_name: str,
        station_x: np.ndarray,
        station_y: np.ndarray,
        station_set: tuple[str, ...] = (),
        n_members: int = 1,
        target: _Target | None = None,
        keep_samples: bool = False,
        solid_state_path: pathlib.Path | None = None,
    ) -> None:
        self.solver_name = solver_name
        self.station_x = np.asarray(station_x, dtype=float)
        self.station_y = np.asarray(station_y, dtype=float)
        self.station_set = station_set or tuple("" for _ in self.station_x)
        self.n_members = int(n_members)
        self.target = target
        self.keep_samples = bool(keep_samples)
        self.solid_state_path = solid_state_path
        self.reason: str | None = None
        self.failed = False
        # Which centre axes carry an extrapolated last index, and which slab
        # cells the backend marks as solid; both are static, both are recorded
        # from the states themselves (see :meth:`_prepare`).
        self.extrapolated: tuple[str, ...] = ()
        self.solid: np.ndarray | None = None
        self._moments: dict[int, dict[str, MomentAccumulator]] = {}
        self._frames: dict[int, int] = {}
        self._windows: set[int] = set()
        self._samples: list[np.ndarray] = []
        self._sample_index: np.ndarray | None = None

    # -- accumulation -------------------------------------------------------

    def add(self, window_index: int, member_index: int, state: xarray.Dataset) -> None:
        """Fold one chunk of frames into ``member_index``'s accumulators."""
        if self.failed:
            return
        try:
            dims = self._prepare(state)
        except (ValueError, KeyError) as error:
            # Only the layout inspection is guarded: colocation and the dim
            # lookup are the steps that legitimately refuse an input (a
            # staggering the table does not know, a state subset before
            # colocation, a missing centre coordinate, no time axis). Everything
            # after them is pure computation on arrays already in memory, so an
            # exception there is a bug and must not be swallowed into a green run
            # with a silently missing block.
            self.failed = True
            self.reason = str(error)
            logger.warning(
                "Mean fields disabled after window %s member %s: %s",
                window_index,
                member_index,
                error,
            )
            return
        self._accumulate(window_index, member_index, state, dims)

    def _prepare(self, state: xarray.Dataset) -> tuple[str, str, str]:
        """Inspect the layout on one frame: the dims, the target, the solid mask.

        One frame rather than the window because colocation *materialises* every
        axis it moves: probing keeps the layout check cheap and keeps the
        full-size copy inside the sub-chunked loop below, where the transient
        budget governs it. Everything read here (grid coordinates, staggering,
        obstacle geometry) is static, so one frame settles it.
        """
        if "time" not in state.dims:
            raise ValueError("state has no 'time' axis, so there is nothing to average")
        probe = self._colocated(state.isel(time=slice(0, 1)))
        dims = _centre_dims(probe[0])
        if self.target is None:
            self.target = self._derive_target(probe[0], dims)
        # Outside the branch above: a collector handed *another's* target (the
        # prior and truth passes) never enters it, and :meth:`_time_block` reads
        # this to decide whether a step materialises the full source grid --
        # which colocation does for every member on a staggered backend. Left
        # unset it sizes the sub-chunk source_cells/target_cells too large (64x
        # at Barcelona shape). Static per solver and layout, so recording it on
        # every state is a table lookup and cannot drift.
        self.extrapolated = extrapolated_centre_dims(state, self.solver_name)
        self._record_solid(state, dims)
        return dims

    def _colocated(self, state: xarray.Dataset) -> list[xarray.DataArray]:
        """The three velocity components on cell centres, member axis dropped."""
        return [
            _drop_member_axis(field)
            for field in colocate_components(state, self.solver_name)
        ]

    def _accumulate(
        self,
        window_index: int,
        member_index: int,
        state: xarray.Dataset,
        dims: tuple[str, str, str],
    ) -> None:
        """Fold the frames in, sub-chunked in time to bound the transient.

        Colocation happens *inside* the loop, on each piece: it copies every axis
        it moves, so colocating the whole window first would materialise three
        full-size float64 copies of it before the budget below ever applied.
        """
        moments = self._moments.setdefault(
            member_index, {"slab": MomentAccumulator(), "station": MomentAccumulator()}
        )
        n_time = int(state.sizes["time"])
        block = self._time_block(state, dims)
        for start in range(0, n_time, block):
            piece = self._colocated(state.isel(time=slice(start, start + block)))
            slabs = [self._slab(field, dims) for field in piece]
            moments["slab"].update(*slabs)
            moments["station"].update(*(self._columns(field, dims) for field in piece))
            self._keep_samples(slabs)

        self._frames[member_index] = self._frames.get(member_index, 0) + n_time
        self._windows.add(int(window_index))

    def _time_block(self, state: xarray.Dataset, dims: tuple[str, str, str]) -> int:
        """Frames per accumulation step, from the transient budget.

        Sized on the **source** grid whenever a step touches it -- colocation on
        a staggered backend, or the cross-grid interpolation of a truth -- and on
        the target slab otherwise. The distinction matters in exactly the case it
        is most needed: a truth finer than the assimilation grid (a PALM truth
        against a surrogate ensemble) has more source cells than target cells, so
        sizing on the target alone would leave the larger term unbounded.
        """
        assert self.target is not None  # _prepare sets it before this is called
        target_cells = int(self.target.z.size * self.target.y.size * self.target.x.size)
        source_cells = 1
        for dim in dims:
            source_cells *= int(state.sizes.get(dim, 1))
        touches_source = bool(self.extrapolated) or not self._is_native_grid(
            state, dims
        )
        cells = max(target_cells, source_cells if touches_source else 0)
        return max(
            1, _MEAN_FIELD_TRANSIENT_BYTES // (_MEAN_FIELD_BYTES_PER_SAMPLE * cells)
        )

    def _record_solid(self, state: xarray.Dataset, dims: tuple[str, str, str]) -> None:
        """The backend's own obstacle indicator on the slab cells, read once.

        Static geometry, so one frame of one member settles it; and only from a
        state on the target's own grid, since an interpolated indicator is not an
        indicator (a zero-filled solid pulls its neighbour halfway to zero).
        pylbm ships ``blanking``; pypalm and uDALES ship no mask at all, and
        :func:`_fluid_cells` falls back to the truth for those.

        Read from ``solid_state_path`` rather than from the member slice, because
        the sensor pass hands out ``ds[["u", "v", "w"]]`` and the indicator is not
        in it. One metadata-cheap open of one window file, once per run -- against
        pulling the variable through every member's slice, which is a third more
        bytes per window for one frame's worth of static geometry.
        """
        if self.solid is not None:
            return
        indicator = self._solid_indicator(state)
        if indicator is None:
            return
        # Collapse every axis that is not one of the three centre dims, rather
        # than a fixed ("ensemble", "time") pair: the geometry is static, so any
        # of them is settled by index 0, and the leading axis is not always
        # called ``time``. The filtering pipeline's ``state_history.nc`` stacks
        # its analyzed frames on ``cycle``, and leaving that one in place used to
        # make the transpose below raise -- taking the whole mean-field block
        # down on a run whose obstacle mask was right there in the file. An axis
        # that IS spatial but under another name (a staggered indicator) is
        # collapsed too, and then fails the centre-dim check just below, which is
        # the same "no usable mask" outcome as before.
        extra = {dim: 0 for dim in indicator.dims if dim not in dims}
        if extra:
            indicator = indicator.isel(extra)
        if not all(dim in indicator.dims for dim in dims):
            return
        if not self._is_native_grid(indicator, dims):
            return
        assert self.target is not None
        zdim, ydim, xdim = dims
        selected = indicator.isel(
            {
                zdim: self.target.z_index,
                ydim: slice(None, None, self.target.stride),
                xdim: slice(None, None, self.target.stride),
            }
        )
        # Convention (pylbm's own, see warm_start_utils): non-zero is solid.
        self.solid = (
            np.asarray(selected.transpose(zdim, ydim, xdim).values, dtype=float) >= 0.5
        )

    def _solid_indicator(self, state: xarray.Dataset) -> xarray.DataArray | None:
        """The obstacle indicator, from the state itself or from the state file."""
        if _BLANKING_VAR in state:
            return state[_BLANKING_VAR]
        if self.solid_state_path is None:
            return None
        try:
            with xarray.open_dataset(self.solid_state_path) as ds:
                if _BLANKING_VAR not in ds:
                    return None
                return ds[_BLANKING_VAR].load()
        except OSError as error:
            # Unreadable is not fatal: the truth-variance fallback still runs.
            logger.info(
                "No solid-cell indicator (%s): %s", self.solid_state_path, error
            )
            return None

    def _derive_target(
        self, field: xarray.DataArray, dims: tuple[str, str, str]
    ) -> _Target:
        """Fix the scored region from the first state: the collector owns the grid."""
        zdim, ydim, xdim = dims
        z_index = evenly_spaced_levels(int(field.sizes[zdim]), _MEAN_FIELD_Z_SLICES)
        stride = _mean_field_stride(
            self.n_members,
            int(z_index.size),
            int(field.sizes[ydim]),
            int(field.sizes[xdim]),
        )
        native_z = np.asarray(field[zdim].values, dtype=float)
        native_y = np.asarray(field[ydim].values, dtype=float)
        native_x = np.asarray(field[xdim].values, dtype=float)
        return _Target(
            z=native_z[z_index],
            y=native_y[::stride],
            x=native_x[::stride],
            z_index=z_index,
            stride=int(stride),
            station_z=native_z,
            station_x=self.station_x,
            station_y=self.station_y,
            station_set=self.station_set,
            native=(native_z, native_y, native_x),
        )

    def _slab(self, field: xarray.DataArray, dims: tuple[str, str, str]) -> np.ndarray:
        """``(time, zlev, y, x)`` on the target slab cells.

        Interpolation when the field does not live on the grid the target was cut
        from: that is the truth collector, whose grid is a *different* grid in the
        general case (a PALM truth against a surrogate ensemble is the shipped
        default), and the metrics doc's rule is to interpolate the truth onto the
        assimilation grid before scoring. Target cells outside the truth's domain
        come back ``nan`` and are dropped by every consumer.
        """
        zdim, ydim, xdim = dims
        target = self.target
        assert target is not None
        if self._is_native_grid(field, dims):
            selected = field.isel(
                {
                    zdim: target.z_index,
                    ydim: slice(None, None, target.stride),
                    xdim: slice(None, None, target.stride),
                }
            )
        else:
            selected = field.interp({zdim: target.z, ydim: target.y, xdim: target.x})
        return np.asarray(
            selected.transpose("time", zdim, ydim, xdim).values, dtype=float
        )

    def _columns(
        self, field: xarray.DataArray, dims: tuple[str, str, str]
    ) -> np.ndarray:
        """``(time, z, station)`` profiles at the station points.

        Always an interpolation: a station is an arbitrary (x, y) point, not a
        cell centre. For a foreign grid the vertical is interpolated too, so the
        truth's profiles land on the assimilation grid's levels.

        **The horizontal neighbourhood is selected first**, one station at a
        time, and this is what makes the columns cheap rather than the most
        expensive thing in the pass. Handing the whole 3-D field to ``.interp``
        costs a float64 upcast of every cell in it -- measured at 8.5 bytes per
        source sample, 223 MB for a 100-frame 64^3 member, against 46 MB for the
        slab selection that is the actual point of this pass. Bracketing each
        station between the two cells that surround it reduces the source to
        ``(time, z, 2, 2)`` per station, and a station is a handful of them.
        """
        zdim, ydim, xdim = dims
        target = self.target
        assert target is not None
        native = self._is_native_grid(field, dims)
        y_axis = np.asarray(field[ydim].values, dtype=float)
        x_axis = np.asarray(field[xdim].values, dtype=float)

        columns = []
        for station_x, station_y in zip(target.station_x, target.station_y):
            near = field.isel(
                {
                    ydim: _bracket(y_axis, station_y),
                    xdim: _bracket(x_axis, station_x),
                }
            )
            points: dict[str, object] = {xdim: station_x, ydim: station_y}
            if not native:
                # A foreign grid's levels are not the target's, so the vertical
                # is interpolated too -- orthogonally, since x and y are scalars
                # here, which is why this needs no broadcast index arrays.
                points[zdim] = target.station_z
            columns.append(near.interp(points))
        if not columns:
            # No sensors at all. The slabs do not depend on the stations, so
            # this costs the columns and nothing else (invariant 3); an empty
            # concat would take the whole pass down instead.
            return np.empty(
                (int(field.sizes["time"]), int(target.station_z.size), 0), dtype=float
            )
        # A station outside the domain brackets to an edge cell pair and
        # interpolates to nan, which is what every consumer already expects of
        # a cell the source cannot reach.
        stacked = xarray.concat(columns, dim="station")
        return np.asarray(
            stacked.transpose("time", zdim, "station").values, dtype=float
        )

    def _is_native_grid(
        self, field: xarray.DataArray, dims: tuple[str, str, str]
    ) -> bool:
        """Whether ``field`` lives on the very grid the target was cut from.

        Compared against the *full* stored axes rather than the strided target
        ones, so a grid that merely happens to share every fourth coordinate
        cannot be mistaken for the native one and sliced.
        """
        assert self.target is not None
        return all(
            int(field.sizes[dim]) == axis.size
            and np.allclose(np.asarray(field[dim].values, dtype=float), axis)
            for dim, axis in zip(dims, self.target.native)
        )

    def _keep_samples(self, slabs: list[np.ndarray]) -> None:
        """Retain a fixed sample of slab cells' series for the ``W`` bootstrap.

        Inside the streaming pass because that is the only place the series
        exist; a fixed cell sample rather than the whole slab because ``W`` is a
        median over cells and 64 of them settle it.
        """
        if not self.keep_samples:
            return
        flat = [component.reshape(component.shape[0], -1) for component in slabs]
        if self._sample_index is None:
            n_cells = flat[0].shape[1]
            rng = np.random.default_rng(_W_SAMPLE_SEED)
            size = min(_W_SAMPLE_POINTS, n_cells)
            self._sample_index = np.sort(rng.choice(n_cells, size=size, replace=False))
        self._samples.append(
            np.stack([component[:, self._sample_index] for component in flat])
        )

    # -- reduction ----------------------------------------------------------

    def result(self) -> dict | None:
        """Per-member fields, or ``None`` if nothing usable was accumulated.

        Returns:
            ``{"n_members", "n_windows", "frames_per_member", "target",
            "slab_mean" (M, 3, zlev, y, x), "slab_tke", "slab_uw",
            "station_mean" (M, 3, z, station), "station_tke", "station_uw"}``.
            ``slab_uw`` is ``<u'w'>``, the off-diagonal the metrics doc singles
            out because it discriminates anisotropy that TKE alone cannot.
        """
        if self.failed or not self._moments:
            return None
        members = sorted(self._moments)
        stacked: dict[str, np.ndarray] = {}
        for region in ("slab", "station"):
            accumulators = [self._moments[m][region] for m in members]
            stacked[f"{region}_mean"] = np.stack([a.mean() for a in accumulators])
            stacked[f"{region}_tke"] = np.stack([a.tke() for a in accumulators])
            stacked[f"{region}_uw"] = np.stack(
                [a.reynolds_stress()[0, 2] for a in accumulators]
            )
        return {
            "n_members": len(members),
            "n_windows": len(self._windows),
            "frames_per_member": [int(self._frames[m]) for m in members],
            "target": self.target,
            **stacked,
        }

    def truth_pass(
        self, truth_access: dict, n_chunks: int, chunk_frames: int
    ) -> "MeanFieldCollector":
        """Stream the truth through this collector, one window's frames at a time.

        A second read of the truth, which the window-state pass avoids for the
        ensemble: the truth has to be sampled onto the target, and the target is
        only fixed once the ensemble pass has run. It is one member's worth of
        frames against the ensemble's M, and it keeps the same memory discipline
        -- one chunk at a time, sub-chunked again inside :meth:`_accumulate`.
        """
        for chunk in range(n_chunks):
            ds = open_truth(
                truth_access["true_state_path"],
                truth_access["n_total"],
                truth_access["x_offset"],
                truth_access["start_idx"],
                truth_access["t_offset"],
            ).isel(time=slice(chunk * chunk_frames, (chunk + 1) * chunk_frames))
            if ds.sizes.get("time", 0):
                self.add(chunk, 0, ds)
            ds.close()
        return self

    def sampling_tolerance(self, fluid: np.ndarray | None = None) -> np.ndarray | None:
        """Per-component ``W``: the truth's own sampling error on its time-mean.

        The block-bootstrap standard error of the time-mean at each sampled slab
        cell (which is ``sigma_u/sqrt(N_eff)`` with the correlation kept),
        reduced by the **median** over cells: a single cell inside a
        recirculation can carry a wild floor, and ``W`` is meant to describe a
        typical one. ``nan`` per component where no floor could be measured --
        short runs cannot be blocked, which includes the CI smoke shape -- and
        :func:`~evaluation.scores.hit_rate` reads that as "relative test only".

        Args:
            fluid: Flat boolean over the slab cells. Solid cells are dropped from
                the sample before the median, and they have to be: a cell inside
                a building holds a constant, so its sampling floor is ~0, and in
                a slab that is more than half solid the median would collapse to
                zero and silently switch the absolute criterion off. Ignored when
                no sampled cell survives it.
        """
        if not self._samples:
            return None
        series = np.concatenate(self._samples, axis=1)  # (component, time, point)
        floors = block_bootstrap_std(np.moveaxis(series, 1, -1))  # (component, point)
        if fluid is not None and self._sample_index is not None:
            keep = np.asarray(fluid, dtype=bool).ravel()[self._sample_index]
            if keep.any():
                floors = floors[:, keep]
        with warnings.catch_warnings():
            # Every sampled cell unmeasurable (a run too short to block) is the
            # common case in the smoke shape, not a fault.
            warnings.simplefilter("ignore", RuntimeWarning)
            return np.asarray(np.nanmedian(floors, axis=1))


def _ensemble_size(state_paths: list[pathlib.Path]) -> int:
    """Members in the window state files, from metadata alone.

    Read before the pass because the accumulators' horizontal stride has to be
    fixed at the first member, which is too late to count them. A metadata-only
    open, so it costs no data read; 1 when the files are unreadable or carry no
    ensemble axis, which is the conservative direction (no stride).
    """
    for path in state_paths:
        try:
            with xarray.open_dataset(path) as ds:
                return int(ds.sizes.get("ensemble", 1))
        except OSError:
            continue
    return 1


def _ensemble_reduction(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """``(mean, ddof=1 std)`` over the leading member axis.

    NaN propagates rather than being skipped: a member whose field is not
    finite is not a sample from the posterior, so a mean that silently drops it
    would describe an ensemble that does not exist (WP1.2 settled the same
    question the same way for the parameter calibration block). Cells that are
    NaN in *every* member -- masked solids, the edges an interpolated truth
    cannot reach -- come out NaN either way.
    """
    mean = values.mean(axis=0)
    if values.shape[0] < 2:
        return mean, np.full_like(mean, np.nan)
    return mean, values.std(axis=0, ddof=1)


def _source_variables(prefix: str, result: dict, ensemble: bool) -> dict:
    """``{variable name: (dims, array)}`` for one source (truth / prior / posterior)."""
    variables: dict[str, tuple[tuple[str, ...], np.ndarray]] = {}
    for region, quantities in _FIELD_DIMS.items():
        for quantity, dims in quantities.items():
            members = result[f"{region}_{quantity}"]
            name = f"{prefix}_{region}_{quantity}"
            if not ensemble:
                # The truth is a single "member"; keep the same layout minus the
                # spread, so every consumer indexes the two the same way.
                variables[name] = (dims, members[0])
                continue
            mean, spread = _ensemble_reduction(members)
            variables[name] = (dims, mean)
            variables[f"{name}_spread"] = (dims, spread)
            if region == "station":
                # Nested quantile bands, at the station columns only: the same
                # thing over a 3-D slab would multiply the file by the number of
                # quantiles for a figure nobody draws.
                variables[f"{name}_quantile"] = (
                    ("quantile",) + dims,
                    np.quantile(members, STATION_QUANTILES, axis=0),
                )
    return variables


def _long_name(variable: str) -> str:
    """A readable label for a ``{source}_{region}_{quantity}[_{reduction}]`` name."""
    parts = variable.split("_")
    label = f"{parts[0]} {_LONG_NAMES[parts[2]]}"
    reduction = {
        "spread": " -- ensemble spread (ddof=1)",
        "quantile": " -- ensemble quantiles",
    }
    return label + (reduction.get(parts[3], "") if len(parts) > 3 else "")


# What ``eval_fields.nc`` records when nobody says otherwise, which is every
# ESMDA run: one per-window posterior rollout per member, written at the output
# cadence, so the moments are within-window time averages and the figures need
# no qualification. Only a caller whose frames are sparser than that has to pass
# its own line (see ``moment_sampling``).
_ESMDA_MOMENT_SAMPLING = (
    "every output frame of each window's posterior rollout, so the moments are "
    "within-window time averages"
)


def _eval_fields_dataset(
    target: _Target,
    sources: dict,
    time_span: tuple[float, float] | None = None,
    fluid: np.ndarray | None = None,
    extrapolated: tuple[str, ...] = (),
    moment_sampling: str | None = None,
) -> xarray.Dataset:
    """``eval_fields.nc``: the reduced fields the WP1.5 figures read.

    Reductions only -- never the per-member fields, which are what the window
    state files already hold and what would make this file grow with the
    ensemble. float32 throughout: these are plotted and differenced, not
    accumulated, and the accumulation that needed float64 already happened.

    Self-contained on purpose: the averaging window figure F1 has to annotate,
    the stride, which sensor set each station column came from, which cells are
    fluid and which axes carry colocation's extrapolated edge are all here, so a
    figure never has to re-open the run's other artifacts -- or re-derive a mask
    -- to draw an honest plot.

    ``moment_sampling`` is the last piece of that: WHICH frames the moments were
    reduced over. The ESMDA pipeline has one answer and it is the default, so
    this file's writer never has to name it; the filter has two, and one of them
    ("one analyzed frame per cycle") makes ``*_tke`` / ``*_uw`` an across-cycle
    variance rather than resolved turbulence. That is invisible in the numbers
    and would otherwise be invisible on the figures too, which read this file
    and nothing else -- so the caller's sampling line travels with the data it
    qualifies.
    """
    variables = {}
    for prefix, (result, ensemble) in sources.items():
        for name, (dims, values) in _source_variables(prefix, result, ensemble).items():
            variables[name] = (
                dims,
                np.asarray(values, dtype=np.float32),
                {"long_name": _long_name(name)},
            )
    if fluid is not None:
        variables["slab_fluid"] = (
            _FIELD_DIMS["slab"]["tke"],
            np.asarray(fluid, dtype=np.int8),
            {
                "long_name": (
                    "1 where the slab cell was scored as fluid, 0 where it was "
                    "excluded as solid"
                )
            },
        )
    attrs: dict[str, object] = {
        "description": (
            "Time-mean velocity, resolved TKE and <u'w'> on evenly spaced "
            "z-slabs and at the sensor station columns, reduced across the "
            "ensemble. Stresses are resolved-only: the subgrid contribution "
            "is not included and is not negligible inside a canopy."
        ),
        # Read by S1 and F1 (via their ``sampling_note``) to qualify the labels
        # they would otherwise put on a continuous time average.
        "moment_sampling": moment_sampling or _ESMDA_MOMENT_SAMPLING,
        "horizontal_stride": target.stride,
        # Every axis colocation moved has its LAST index extrapolated from the
        # two faces below it rather than interpolated between two, so those cells
        # carry inflated second moments -- ~20 % for a well-resolved field, up to
        # 5x for face-to-face white noise. It is not only the vertical: uDALES
        # moves x, y and z, PALM moves x and y. An evenly spaced selection always
        # includes the last index, so a figure that does not want the artefact
        # has to exclude those cells itself -- hence recording them here.
        "extrapolated_edges": ",".join(extrapolated),
    }
    if time_span is not None:
        attrs["t_start"], attrs["t_end"] = time_span
    return xarray.Dataset(
        variables,
        coords={
            "component": list(_COMPONENTS),
            "zlev": target.z,
            "y": target.y,
            "x": target.x,
            "z": target.station_z,
            "station": np.arange(target.station_x.size),
            "station_x": ("station", target.station_x),
            "station_y": ("station", target.station_y),
            "station_set": ("station", list(target.station_set)),
            "quantile": list(STATION_QUANTILES),
        },
        attrs=attrs,
    )


def _hit_rates(predicted: np.ndarray, observed: np.ndarray, w: np.ndarray) -> dict:
    """Hit rate over the slab cells: pooled over the components, and per component.

    Pooled is the headline number the metrics doc asks for (one scalar for the
    mean field); the per-component entries are what says *which* component a
    poor one came from, and each is scored against its own sampling floor.
    """
    pooled = hit_rate(predicted, observed, tolerance_w=w[:, None, None, None])
    entry = {"q": pooled["q"], "n_points": pooled["n_points"]}
    for i, name in enumerate(_COMPONENTS):
        entry[name] = hit_rate(predicted[i], observed[i], tolerance_w=w[i])["q"]
    return entry


def _comparable_prior(prior_fields: dict, posterior_fields: dict) -> bool:
    """Whether the prior covers the same cells *and* the same horizon.

    The shape test alone is not enough, and the gap is not hypothetical: a prior
    read that fails partway (a job killed mid-``to_netcdf``) leaves the
    accumulators holding the windows that were read, and a per-member mean over
    three windows has exactly the same shape as one over ten. Scoring those
    against each other is the "two different horizons in one skill score" that
    WP1.3's all-or-nothing prior rule exists to prevent -- and neither the YAML
    nor ``eval_fields.nc`` records the prior's frame count, so a reader could not
    detect it afterwards.
    """
    if prior_fields["slab_mean"].shape != posterior_fields["slab_mean"].shape:
        logger.warning(
            "Prior mean fields cover %s cells and the posterior %s -- the prior "
            "half is dropped rather than scored on another grid",
            prior_fields["slab_mean"].shape,
            posterior_fields["slab_mean"].shape,
        )
        return False
    if prior_fields["frames_per_member"] != posterior_fields["frames_per_member"]:
        logger.warning(
            "Prior mean fields cover %s frames per member and the posterior %s "
            "-- the prior half is dropped rather than compared over a different "
            "horizon",
            prior_fields["frames_per_member"],
            posterior_fields["frames_per_member"],
        )
        return False
    return True


def _mean_field_block(
    run_dir: pathlib.Path,
    posterior: MeanFieldCollector,
    truth: MeanFieldCollector,
    prior: MeanFieldCollector | None,
    time_span: tuple[float, float] | None = None,
    moment_sampling: str | None = None,
) -> dict | None:
    """Write ``eval_fields.nc`` and return the ``field_metrics`` summary block.

    ``None`` (with a log line) whenever the fields could not be built --
    invariant 3: a state layout colocation refuses, or a truth that shares no
    cell with the assimilation grid, costs this block and nothing else.

    ``moment_sampling`` is passed straight through to
    :func:`_eval_fields_dataset`; it defaults to the ESMDA sampling, so the
    ESMDA call site never mentions it.
    """
    posterior_fields = posterior.result()
    truth_fields = truth.result()
    if posterior_fields is None or truth_fields is None:
        logger.warning(
            "No mean-field metrics: %s",
            posterior.reason or truth.reason or "no frames were accumulated",
        )
        return None

    target = posterior_fields["target"]
    sources = {"truth": (truth_fields, False), "posterior": (posterior_fields, True)}
    prior_fields = prior.result() if prior is not None else None
    if prior_fields is not None and not _comparable_prior(
        prior_fields, posterior_fields
    ):
        prior_fields = None
    if prior_fields is not None:
        sources["prior"] = (prior_fields, True)

    fluid, fluid_source = _fluid_cells(posterior, truth_fields)
    _eval_fields_dataset(
        target,
        sources,
        time_span,
        fluid=fluid,
        extrapolated=posterior.extrapolated,
        moment_sampling=moment_sampling,
    ).to_netcdf(run_dir / "eval_fields.nc")

    tolerance = truth.sampling_tolerance(fluid)
    if tolerance is None:
        tolerance = np.full(len(_COMPONENTS), np.nan)
    if not np.any(np.isfinite(tolerance)):
        logger.info(
            "Mean fields: the truth's window is too short to block-bootstrap a "
            "sampling floor, so the hit rate runs on its relative criterion "
            "alone (W = 0)"
        )

    # Fluid cells only, per the metrics doc. A solid cell holds ~0 in the truth
    # *and* in every member, so it is a hit whatever the flow does, and counting
    # them dilutes q toward the built-up fraction: at 30 % solid a fluid hit rate
    # of 0.52 would report as 0.66 and cross the acceptance threshold on a field
    # that fails it. Masking the truth is enough -- ``hit_rate`` drops a point
    # where either side is non-finite.
    observed = np.where(fluid, truth_fields["slab_mean"][0], np.nan)
    block: dict[str, object] = {
        "n_windows": posterior_fields["n_windows"],
        "frames_per_member": posterior_fields["frames_per_member"],
        "truth_frames": truth_fields["frames_per_member"][0],
        "z_levels": [float(v) for v in target.z],
        "horizontal_stride": target.stride,
        "n_fluid_cells": int(np.count_nonzero(fluid)),
        "solid_fraction": float(1.0 - np.count_nonzero(fluid) / fluid.size),
        "solid_cell_source": fluid_source,
        "hit_rate_tolerance_w": {
            name: (float(v) if np.isfinite(v) else None)
            for name, v in zip(_COMPONENTS, tolerance)
        },
        "hit_rate_posterior": _hit_rates(
            _ensemble_reduction(posterior_fields["slab_mean"])[0], observed, tolerance
        ),
    }
    if prior_fields is not None:
        block["hit_rate_prior"] = _hit_rates(
            _ensemble_reduction(prior_fields["slab_mean"])[0], observed, tolerance
        )
    return block


def _fluid_cells(
    posterior: MeanFieldCollector, truth_fields: dict
) -> tuple[np.ndarray, str]:
    """Which slab cells to score, and where the answer came from.

    The metrics doc scores the hit rate over **fluid cells**, and no backend
    makes that free: pylbm ships a ``blanking`` indicator, pypalm replaces PALM's
    NaN inside obstacles with 0.0 and keeps no mask, uDALES ships neither. So
    two rules, in order of authority:

    1. The backend's own indicator, when the state carried one.
    2. Otherwise the truth's own resolved TKE. A cell the solver held at a
       constant (a zero-filled obstacle interior) has *exactly* zero variance in
       every component, while a fluid cell in a turbulent flow does not -- so
       ``tke > 0`` separates them without a threshold to tune. It needs at least
       two frames to mean anything (a one-frame ``ddof=1`` variance is ``nan``),
       and it cannot see an obstacle a backend filled with time-varying junk
       rather than zeros, which is why it is the fallback and not the rule.

    Returns the mask and a label for it, so ``run_summary.yaml`` records which
    rule ran -- ``"none"`` means every cell was scored and a reader should treat
    ``q`` as diluted by whatever solids the domain holds.
    """
    if posterior.solid is not None:
        return ~posterior.solid, _BLANKING_VAR
    tke = truth_fields["slab_tke"][0]
    fluid = np.asarray(np.isfinite(tke) & (tke > 0.0))
    if fluid.any() and not fluid.all():
        return fluid, "truth-zero-variance"
    if not fluid.any():
        # Nothing in the slab fluctuates at all: a single-frame truth (``ddof=1``
        # leaves no variance to test) or a laminar/synthetic field. The rule
        # cannot separate solid from fluid there -- it would mask everything --
        # so it stands down rather than reporting a null hit rate.
        logger.info(
            "Mean fields: no solid-cell indicator and no resolved variance "
            "anywhere in the truth, so the hit rate scores every cell -- read q "
            "as diluted by any obstacles the domain holds"
        )
    return np.ones_like(tke, dtype=bool), "none"


def compute_metrics(run_dir: pathlib.Path) -> None:
    """Compute every run metric from the saved artifacts and write run_summary.yaml."""
    cfg = load_run_config(run_dir)
    ta = read_yaml(run_dir / "truth_access.yaml")

    # Seed the summary from the run metadata/timing saved by run_esmda.py.
    summary = read_yaml(run_dir / "run_info.yaml")
    summary["metrics_version"] = METRICS_VERSION

    # --- Parameters (always available) --------------------------------------
    posterior_params = xarray.open_dataset(run_dir / "posterior_params.nc")
    prior_params = xarray.open_dataset(run_dir / "prior_params.nc")
    true_params = xarray.open_dataset(run_dir / "true_params.nc")
    summary["parameter_metrics"] = parameter_metric_summary(
        posterior_params, true_params, prior_params
    )
    health = _ensemble_health(run_dir, posterior_params)
    if health is not None:
        summary["ensemble_health"] = health

    # --- Probe spectra: Welch + LSD (§4.3) -----------------------------------
    # Above the ``skip_viz`` gate on purpose. Every block below reads the
    # (multi-GB) truth, which is what that flag exists to avoid; this one reads
    # only the small probe records an explicit rerun wrote, and a rerun is far too
    # deliberate an act to have its one metric dropped because the assimilation
    # itself was run on the fast path. Absent records -- every run dir that never
    # had a rerun -- log and skip (invariant 3).
    spectra = probe_spectra_bundle(run_dir)
    if spectra is not None:
        summary["spectral_metrics"] = spectral_metric_summary(spectra)

    # --- ESMDA health: normalized data mismatch O_N (§5) ----------------------
    # Above the ``skip_viz`` gate for the same reason as the block above: it
    # reads only the KB-scale observation-space files, never the truth. Absent
    # on every run dir written before WP2.1 or with
    # ``esmda.save_obs_diagnostics=false`` -- logged and skipped (invariant 3).
    mismatch = obs_diagnostics_bundle(run_dir)
    if mismatch is not None:
        block = data_mismatch_summary(
            mismatch["per_step"],
            mismatch["num_observations"],
            per_window=mismatch["per_window"],
        )
        if block is not None:
            summary["esmda_diagnostics"] = {"data_mismatch": block}

    # The parameter metrics are always available. The state and sensor metrics
    # both open the (potentially multi-GB) truth, so -- matching run_esmda.py's
    # old in-script behaviour -- they are skipped under run.skip_viz (the fast
    # path for large sweeps), exactly as the figures are.
    if cfg.run.skip_viz:
        write_yaml(summary, run_dir / "run_summary.yaml")
        print(f"Saved run summary in {run_dir / 'run_summary.yaml'}")
        return

    # --- State field |U| RMSE -----------------------------------------------
    # Stream over a few z-slices and all time steps instead of the whole 4-D
    # field (interpolating onto the assim grid if coords differ).
    posterior_state = xarray.open_dataset(run_dir / "posterior_state_mean.nc")
    true_state = open_truth(
        ta["true_state_path"],
        ta["n_total"],
        ta["x_offset"],
        ta["start_idx"],
        ta["t_offset"],
    )
    rmse = streaming_state_rmse(true_state, posterior_state)
    summary["state_metrics"] = {"vel_magnitude_rmse": series_stats(rmse)}

    # --- Sensors: full-vector (u, v, w) RMSE + energy score ------------------
    # The truth and the ensemble are interpolated at the same physical points on
    # their own grids (one window at a time), so the metric is grid-independent
    # and never holds the multi-GB truth / full ensemble in memory in full.
    sensor_sets = build_sensor_sets(cfg)
    num_windows = int(ta["num_windows"])
    truth_series = truth_sensor_series(
        ta["true_state_path"],
        ta["n_total"],
        ta["x_offset"],
        ta["start_idx"],
        ta["t_offset"],
        sensor_sets,
        ta["truth_solver_name"],
        num_windows,
        int(ta["n_per_window"]),
    )
    state_paths = [
        run_dir / "windows" / f"window_{w}_posterior_state.nc"
        for w in range(num_windows)
    ]
    # The mean fields ride along on that same pass: the window files total tens
    # of GB at Barcelona scale, and the accumulators are fed from the member
    # slices the sensor extraction already materialises, so they cost no extra
    # read (metrics doc §4.1, master-plan invariant 2).
    station_x, station_y, station_set = station_columns(sensor_sets)
    n_members = _ensemble_size(state_paths)
    field_collector = MeanFieldCollector(
        ta["assim_solver_name"],
        station_x,
        station_y,
        station_set,
        n_members=n_members,
        solid_state_path=state_paths[0] if state_paths else None,
    )
    ensemble_series = ensemble_sensor_series(
        state_paths,
        sensor_sets,
        ta["assim_solver_name"],
        float(ta["sim_time"]),
        on_member=field_collector.add,
    )

    # Invariant 3, once for both sensor blocks: every score below is a
    # *probabilistic* one and needs the members. A state file written without an
    # ensemble axis (an old ensemble-mean-only artifact) has nothing to score,
    # and must cost its own sensor set rather than the whole metric stage --
    # which would take the parameter, health and state blocks with it.
    scorable = {
        name: coords
        for name, coords in sensor_sets.items()
        if "ensemble" in ensemble_series[name].dims
    }
    for name in sensor_sets:
        if name not in scorable:
            logger.warning(
                "No ensemble dimension in the %s sensor series -- the sensor "
                "metrics and statistics both need the members, so that set is "
                "omitted from both blocks",
                name,
            )

    sensor_metrics = {}
    for name, (sx, sy, sz) in scorable.items():
        m = vector_sensor_metrics(truth_series[name], ensemble_series[name])
        sensor_metrics[name] = {
            "num_sensors": int(np.asarray(sx).size),
            "velocity_vector_rmse": series_stats(m["rmse"]),
            "velocity_vector_energy_score": series_stats(m["energy_score"]),
        }
    summary["sensor_metrics"] = sensor_metrics

    # --- Sensors: window statistics as the verification object (§4.2) --------
    # One read of the prior states, feeding both prior blocks -- and one place
    # that decides whether they may be used at all. ``None`` means the set was
    # absent or unreadable, and *neither* the sensor prior nor the field prior
    # may be scored then: the field accumulators may already hold the windows
    # that were read before the failure, and a per-member mean over three
    # windows has the same shape as one over ten.
    prior_collector = MeanFieldCollector(
        ta["assim_solver_name"],
        station_x,
        station_y,
        station_set,
        n_members=n_members,
        target=field_collector.target,
    )
    prior_series = _prior_sensor_series(
        run_dir,
        scorable,
        num_windows,
        float(ta["sim_time"]),
        ta["assim_solver_name"],
        prior_collector.add,
    )
    summary["sensor_statistics"] = _sensor_statistics(
        scorable,
        truth_series,
        ensemble_series,
        prior_series,
        num_windows,
        float(ta["sim_time"]),
    )

    # --- Mean fields: hit rate + eval_fields.nc (§4.1) ------------------------
    truth_collector = MeanFieldCollector(
        ta["truth_solver_name"],
        station_x,
        station_y,
        station_set,
        target=field_collector.target,
        keep_samples=True,
    )
    if field_collector.target is not None and not field_collector.failed:
        # Only once the ensemble pass has fixed the region: the truth is scored
        # *on the assimilation grid*, so without a target there is nothing to
        # interpolate onto -- and re-reading the truth for a posterior that
        # failed mid-pass would buy nothing either. The block is dropped below
        # in both cases.
        truth_collector.truth_pass(ta, num_windows, int(ta["n_per_window"]))
    sim_time = float(ta["sim_time"])
    field_metrics = _mean_field_block(
        run_dir,
        field_collector,
        truth_collector,
        prior_collector if prior_series is not None else None,
        time_span=(0.0, num_windows * sim_time),
    )
    if field_metrics is not None:
        summary["field_metrics"] = field_metrics

    write_yaml(summary, run_dir / "run_summary.yaml")
    print(f"Saved run summary in {run_dir / 'run_summary.yaml'}")


def main() -> None:
    # Every reason a block degraded or was skipped is a ``logger`` call in this
    # module and in ``evaluation``; with no handler on the root logger, a stage run
    # standalone printed only its final line and none of them. Configured here, at
    # the entry point (never in ``compute_metrics``) so importing this module -- the
    # tests do -- cannot reconfigure anyone's logging. Mirrors
    # ``make_esmda_figures.main``.
    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s %(name)s: %(message)s"
    )
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "--run-dir",
        type=pathlib.Path,
        required=True,
        help="The ESMDA run output directory written by scripts/esmda/run_esmda.py.",
    )
    args = ap.parse_args()
    if not args.run_dir.exists():
        raise SystemExit(f"run dir not found: {args.run_dir}")
    compute_metrics(args.run_dir)


if __name__ == "__main__":
    main()
