"""WP1.3's shared per-member read of the full-ensemble window state files.

``stream_window_members`` replaced ``ensemble_sensor_series``' full-ensemble
``xarray.open_dataset(path).load()``. Two things therefore need pinning, and
neither is covered anywhere else:

**That it is actually streaming.** The file it reads is ~1 GB at smoke scale and
tens of GB at Barcelona scale, so "never materialise the ensemble" is a
correctness constraint, not a performance note -- but nothing about the *return
value* changes when a refactor quietly reads the whole file again. The read-count
tests below therefore assert on the reads themselves, by counting calls through
``NetCDF4ArrayWrapper._getitem``: the single funnel every lazily-indexed
netCDF4-backed read passes through. What that proves: no single read pulled more
than one member's worth of elements, and the whole pass read each member's bytes
exactly once. What it does *not* prove: anything about process RSS (the OS page
cache and NumPy's allocator are outside its view), or that a consumer downstream
of the generator does not itself accumulate the members it is handed.

**That the sensor series did not move.** WP1.2's ``sensor_statistic_scores``
numbers are calibrated on the old function's output -- the phase-1 deviation log
pins them to the last digit (``window_mean``'s identifiability ``ratio_median``
at 1.0421934131499127) -- so the refactor has to be bit-identical, not merely
close. ``_reference_series`` below is a verbatim copy of the pre-WP1.3
implementation (``git show d7a169b:scripts/esmda/_esmda_common.py``, lines
304-323); it calls the same ``_sensor_component_timeseries`` /
``_concat_sensor_pieces`` the refactor does, which the refactor left untouched.
The same comparison was additionally run once against the *whole* HEAD module
extracted from git, and once end to end by diffing all 587 leaves of
``run_summary.yaml`` computed both ways (all equal); the in-repo copy is what
stays green from here on.

The fixtures cover the shapes the two implementations could disagree on: the
multi-window fixture ``test_esmda_metrics_wiring`` already builds (which is what
WP1.2's own numbers come from), per-window time cadences that differ in frame
count *and* offset, a window file with no ``time`` coordinate at all (the
un-rebased branch), one with no ``ensemble`` coordinate variable (the layout
every disk-backed run writes), and a **staggered udales grid** -- the last
because the reason the generator yields raw rather than co-located members is
precisely that the sensor path resolves each component against its own staggered
dims.
"""

from __future__ import annotations

import ast
import inspect
import pathlib
import textwrap
from typing import Any

import numpy as np
import pytest
import xarray
from xarray.backends.file_manager import FILE_CACHE
from xarray.backends.netCDF4_ import NetCDF4ArrayWrapper

from scripts.esmda._esmda_common import (
    SensorSeriesAccumulator,
    _concat_sensor_pieces,
    _sensor_component_timeseries,
    ensemble_sensor_series,
    member_sensor_series,
    stream_window_members,
)
from tests.test_esmda_metrics_wiring import (
    ASSIM_POINTS,
    GRID_X,
    N_MEMBERS,
    N_WINDOWS,
    SIM_TIME,
    VALIDATION_POINTS,
    _state_dataset,
    _velocity_arrays,
    _write_state_artifacts,
)

SENSOR_SETS = {"assimilation": ASSIM_POINTS, "validation": VALIDATION_POINTS}
VELOCITY_VARS = ("u", "v", "w")


# --- the pre-WP1.3 implementation, verbatim ---------------------------------


def _reference_series(
    state_paths: list[pathlib.Path],
    sensor_sets: dict[str, Any],
    solver_name: str,
    sim_time: float,
) -> Any:
    """``ensemble_sensor_series`` as it was before WP1.3. Do not modernise."""
    pieces: dict[str, list[Any]] = {name: [] for name in sensor_sets}
    for w, path in enumerate(state_paths):
        ds = xarray.open_dataset(path).load()
        t = np.asarray(ds["time"].values, dtype=float) if "time" in ds.coords else None
        for name, (ox, oy, oz) in sensor_sets.items():
            vel = _sensor_component_timeseries(ds, ox, oy, oz, solver_name)
            if t is not None and "time" in vel.dims:
                vel = vel.assign_coords(time=(t - t[0]) + w * sim_time)
            pieces[name].append(vel)
        ds.close()
    return _concat_sensor_pieces(pieces)


# --- fixtures ---------------------------------------------------------------


def _write_windows(
    run_dir: pathlib.Path, datasets: list[xarray.Dataset]
) -> list[pathlib.Path]:
    """Write ``window_{w}_posterior_state.nc`` for each dataset, return the paths."""
    windows = run_dir / "windows"
    windows.mkdir(parents=True, exist_ok=True)
    paths = []
    for w, ds in enumerate(datasets):
        path = windows / f"window_{w}_posterior_state.nc"
        ds.to_netcdf(path)
        paths.append(path)
    return paths


@pytest.fixture  # type: ignore[misc]
def wiring_fixture_paths(tmp_path: pathlib.Path) -> list[pathlib.Path]:
    """The exact multi-window artifacts WP1.2's own wiring test scores."""
    run_dir = tmp_path / "wiring"
    run_dir.mkdir()
    _write_state_artifacts(run_dir, np.random.default_rng(11))
    return [
        run_dir / "windows" / f"window_{w}_posterior_state.nc" for w in range(N_WINDOWS)
    ]


def _varied_cadence_datasets(seed: int = 3) -> list[xarray.Dataset]:
    """Windows whose local time axes differ in frame count *and* start offset.

    ``(t - t[0]) + w*sim_time`` has to do real work here: a fixture whose windows
    all start at 0.0 with the same dt would pass with the subtraction dropped.
    """
    rng = np.random.default_rng(seed)
    gains = 1.0 + rng.normal(0.0, 0.12, size=N_MEMBERS)
    phases = rng.normal(0.0, 0.25, size=N_MEMBERS)
    datasets = []
    for n_frames, t0 in ((5, 0.0), (9, 1.7), (4, -0.4)):
        local = t0 + np.arange(n_frames) * (SIM_TIME / n_frames)
        datasets.append(
            _state_dataset(_velocity_arrays(local, gains, phases, rng), local, GRID_X)
        )
    return datasets


def _udales_window(
    n_members: int = 4, n_time: int = 5, seed: int = 17
) -> xarray.Dataset:
    """A staggered udales-shaped window file: u on xm, v on ym, w on zm.

    ``ObservationOperator("udales")`` maps the three components onto three
    different dim triples, so this is the fixture that fails if anything
    co-locates or otherwise rewrites the member before the sensor interpolation.
    """
    rng = np.random.default_rng(seed)
    xt = np.linspace(1.0, 21.0, 6)
    yt = np.linspace(1.0, 17.0, 5)
    zt = np.linspace(1.0, 13.0, 4)
    # Faces sit half a cell below the centres, the udales convention.
    xm = xt - 0.5 * (xt[1] - xt[0])
    ym = yt - 0.5 * (yt[1] - yt[0])
    zm = zt - 0.5 * (zt[1] - zt[0])
    time = np.arange(n_time) * 1.5
    dims = {
        "u": ("ensemble", "time", "zt", "yt", "xm"),
        "v": ("ensemble", "time", "zt", "ym", "xt"),
        "w": ("ensemble", "time", "zm", "yt", "xt"),
    }
    coords = {
        "ensemble": np.arange(n_members),
        "time": time,
        "xt": xt,
        "yt": yt,
        "zt": zt,
        "xm": xm,
        "ym": ym,
        "zm": zm,
    }
    data = {}
    for name, dim_names in dims.items():
        shape = tuple(len(coords[d]) for d in dim_names)
        data[name] = (dim_names, rng.normal(size=shape))
    return xarray.Dataset(data, coords=coords)


# Sensors strictly inside the *intersection* of the three staggered grids, so
# no component needs the half-cell extrapolation margin.
UDALES_SENSOR_SETS = {
    "assimilation": ([6.0, 15.0], [5.0, 12.0], [4.0, 9.0]),
    "validation": ([10.0], [8.0], [6.0]),
}


# --- read accounting --------------------------------------------------------


class ReadCounter:
    """Records every netCDF4-backed slab read: variable name and element count."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self.reads: list[tuple[str, int]] = []
        original = NetCDF4ArrayWrapper._getitem

        def counting_getitem(wrapper: Any, key: Any) -> Any:
            out = original(wrapper, key)
            self.reads.append((wrapper.variable_name, int(np.size(out))))
            return out

        monkeypatch.setattr(NetCDF4ArrayWrapper, "_getitem", counting_getitem)

    def velocity_reads(self) -> list[tuple[str, int]]:
        return [(n, s) for n, s in self.reads if n in VELOCITY_VARS]


def _open_handles(paths: list[pathlib.Path]) -> list[pathlib.Path]:
    """Paths among ``paths`` that xarray still holds an open file handle for.

    ``FILE_CACHE`` is xarray's global open-file LRU; a manager drops its entry
    and closes the underlying netCDF4 handle when the Dataset is closed, so key
    absence is a statement about the OS handle. It is the only sound observable
    here: reading through a *closed* xarray Dataset silently reopens the file
    via the caching manager, so "a read raises" would test nothing.
    """
    keys = [repr(k) for k in FILE_CACHE.keys()]
    return [p for p in paths if any(str(p) in k for k in keys)]


# --- the generator ---------------------------------------------------------


def test_stream_yields_every_member_of_every_window_in_order(
    wiring_fixture_paths: list[pathlib.Path],
) -> None:
    seen = [(w, m) for w, m, _ in stream_window_members(wiring_fixture_paths)]
    assert seen == [(w, m) for w in range(N_WINDOWS) for m in range(N_MEMBERS)]
    assert len(seen) == N_WINDOWS * N_MEMBERS


def test_yielded_member_is_one_member_on_the_files_own_dims(
    wiring_fixture_paths: list[pathlib.Path],
) -> None:
    """No ``ensemble`` dim, every velocity component, coordinates intact."""
    with xarray.open_dataset(wiring_fixture_paths[0]) as ds:
        expected_member_dims = {d: n for d, n in ds.sizes.items() if d != "ensemble"}
    for w, m, member in stream_window_members(wiring_fixture_paths):
        assert "ensemble" not in member.dims
        assert dict(member.sizes) == expected_member_dims
        assert set(member.data_vars) == set(VELOCITY_VARS)
        # The member label survives as a scalar coord when the file carries one;
        # `SensorSeriesAccumulator` needs it to rebuild the `ensemble` coord.
        assert int(member["ensemble"]) == m


def test_stream_never_reads_more_than_one_member(
    monkeypatch: pytest.MonkeyPatch, wiring_fixture_paths: list[pathlib.Path]
) -> None:
    """Read accounting: nothing bigger than a member, each member's bytes once.

    See the module docstring for what this does and does not establish.
    """
    with xarray.open_dataset(wiring_fixture_paths[0]) as ds:
        per_member = int(
            np.prod([n for d, n in ds["u"].sizes.items() if d != "ensemble"])
        )
    counter = ReadCounter(monkeypatch)

    for _w, _m, _member in stream_window_members(wiring_fixture_paths):
        pass

    reads = counter.velocity_reads()
    assert reads, "no velocity reads were recorded -- the counter is not wired"
    assert max(size for _, size in reads) <= per_member
    # One read per (window, member, component), no repeats and no leftovers.
    assert len(reads) == N_WINDOWS * N_MEMBERS * len(VELOCITY_VARS)
    assert sum(size for _, size in reads) == (
        N_WINDOWS * N_MEMBERS * len(VELOCITY_VARS) * per_member
    )


def test_two_consumers_share_one_read_per_member(
    monkeypatch: pytest.MonkeyPatch, wiring_fixture_paths: list[pathlib.Path]
) -> None:
    """The point of the whole refactor: N consumers, still one read per member.

    This is what a lazily-yielded member would fail -- xarray does not cache
    reads taken through ``.isel``, so each consumer would pay for the bytes
    again. Both sensor sets here stand in for the two real consumers (WP1.2's
    sensor extraction and WP1.3's moment accumulators).
    """
    with xarray.open_dataset(wiring_fixture_paths[0]) as ds:
        per_member = int(
            np.prod([n for d, n in ds["u"].sizes.items() if d != "ensemble"])
        )
    counter = ReadCounter(monkeypatch)

    ensemble_sensor_series(wiring_fixture_paths, SENSOR_SETS, "pylbm", SIM_TIME)

    reads = counter.velocity_reads()
    assert len(SENSOR_SETS) == 2  # otherwise this test is not testing sharing
    assert len(reads) == N_WINDOWS * N_MEMBERS * len(VELOCITY_VARS)
    assert sum(size for _, size in reads) == (
        N_WINDOWS * N_MEMBERS * len(VELOCITY_VARS) * per_member
    )


def test_stream_closes_its_file_handles_after_normal_completion(
    wiring_fixture_paths: list[pathlib.Path],
) -> None:
    assert not _open_handles(wiring_fixture_paths)
    for _w, _m, _member in stream_window_members(wiring_fixture_paths):
        pass
    assert _open_handles(wiring_fixture_paths) == []


def test_stream_closes_its_file_handles_when_abandoned_early(
    wiring_fixture_paths: list[pathlib.Path],
) -> None:
    """Abandonment mid-window is the case a trailing ``ds.close()`` would leak."""
    stream = stream_window_members(wiring_fixture_paths)
    next(stream)
    assert _open_handles(wiring_fixture_paths) == [wiring_fixture_paths[0]]
    stream.close()  # GeneratorExit at the yield, unwinding the `with`
    assert _open_handles(wiring_fixture_paths) == []


def test_stream_closes_its_file_handles_when_a_consumer_raises(
    wiring_fixture_paths: list[pathlib.Path],
) -> None:
    with pytest.raises(RuntimeError, match="consumer blew up"):
        for _w, _m, _member in stream_window_members(wiring_fixture_paths):
            raise RuntimeError("consumer blew up")
    assert _open_handles(wiring_fixture_paths) == []


def test_stream_rejects_a_file_that_is_not_a_full_ensemble(
    tmp_path: pathlib.Path,
) -> None:
    """A degenerate input fails where it is cheap to diagnose, naming the file."""
    local = np.arange(4) * 1.0
    single = _state_dataset(
        _velocity_arrays(
            local, np.array([1.0]), np.array([0.0]), np.random.default_rng(1)
        ),
        local,
        GRID_X,
    ).isel(ensemble=0, drop=True)
    paths = _write_windows(tmp_path, [single])
    with pytest.raises(ValueError, match="no 'ensemble' dimension"):
        list(stream_window_members(paths))
    assert _open_handles(paths) == []


def test_stream_names_a_missing_variable(
    wiring_fixture_paths: list[pathlib.Path],
) -> None:
    with pytest.raises(KeyError, match="tke"):
        list(stream_window_members(wiring_fixture_paths, variables=("u", "tke")))
    assert _open_handles(wiring_fixture_paths) == []


def test_stream_can_read_every_variable_when_asked(tmp_path: pathlib.Path) -> None:
    """``variables=None`` widens the read; the default is the velocity triple."""
    rng = np.random.default_rng(4)
    local = np.arange(4) * 1.0
    fields = _velocity_arrays(local, np.array([1.0, 1.1]), np.array([0.0, 0.3]), rng)
    fields["p"] = fields["u"] * 2.0
    paths = _write_windows(tmp_path, [_state_dataset(fields, local, GRID_X)])
    default = [m for _, _, m in stream_window_members(paths)]
    widened = [m for _, _, m in stream_window_members(paths, variables=None)]
    assert set(default[0].data_vars) == set(VELOCITY_VARS)
    assert set(widened[0].data_vars) == {"u", "v", "w", "p"}


# --- bit-identity of the sensor series -------------------------------------


def _assert_identical_series(new: dict[str, Any], old: dict[str, Any]) -> None:
    assert set(new) == set(old)
    for name in new:
        a, b = new[name], old[name]
        assert a.dims == b.dims, (name, a.dims, b.dims)
        assert a.shape == b.shape, (name, a.shape, b.shape)
        assert np.array_equal(a.values, b.values, equal_nan=True), name
        # Coordinates too: WP1.2 slices the ensemble series by *time value*, so
        # a series with the right numbers on a shifted axis is still wrong.
        assert a.identical(b), name


def test_sensor_series_is_bit_identical_on_the_wiring_fixture(
    wiring_fixture_paths: list[pathlib.Path],
) -> None:
    new = ensemble_sensor_series(wiring_fixture_paths, SENSOR_SETS, "pylbm", SIM_TIME)
    old = _reference_series(wiring_fixture_paths, SENSOR_SETS, "pylbm", SIM_TIME)
    _assert_identical_series(new, old)
    # Guard the guard: a fixture whose members were all equal would make the
    # comparison vacuous.
    series = new["assimilation"]
    assert series.std("ensemble").max() > 0.0
    assert series.dims == ("component", "ensemble", "time", "sensor")


def test_sensor_series_reproduces_the_reference_memory_layout(
    wiring_fixture_paths: list[pathlib.Path],
) -> None:
    """Identical contents are not enough: the layout has to match too.

    Every consumer reduces over some of these axes, numpy's pairwise summation
    walks a reduction in memory order, and floating-point addition is not
    associative -- so the same numbers in a different layout move
    ``run_summary.yaml`` in its 16th digit. Measured while building
    ``_stack_window_members``: stacking members first and transposing moved 4 of
    587 summary leaves, forcing C-contiguity moved 20, matching the layout moved
    none. The strides assertion pins the mechanism; the reductions pin the
    consequence, which is what actually matters.
    """
    new = ensemble_sensor_series(wiring_fixture_paths, SENSOR_SETS, "pylbm", SIM_TIME)
    old = _reference_series(wiring_fixture_paths, SENSOR_SETS, "pylbm", SIM_TIME)
    for name in new:
        a, b = new[name].values, old[name].values
        assert a.strides == b.strides, (name, a.strides, b.strides)
        for axes in (("component",), ("time",), ("component", "time", "sensor")):
            assert np.array_equal(
                new[name].sum(axes).values, old[name].sum(axes).values
            ), (name, axes)


def test_sensor_series_is_bit_identical_with_differing_window_cadences(
    tmp_path: pathlib.Path,
) -> None:
    paths = _write_windows(tmp_path, _varied_cadence_datasets())
    new = ensemble_sensor_series(paths, SENSOR_SETS, "pylbm", SIM_TIME)
    old = _reference_series(paths, SENSOR_SETS, "pylbm", SIM_TIME)
    _assert_identical_series(new, old)
    # And the rebasing itself: each window lands in [w*sim_time, (w+1)*sim_time).
    times = np.asarray(new["assimilation"]["time"].values)
    assert times.size == 5 + 9 + 4
    for w, (n_frames, _t0) in enumerate(((5, 0.0), (9, 1.7), (4, -0.4))):
        start = sum(n for n, _ in ((5, 0.0), (9, 1.7), (4, -0.4))[:w])
        block = times[start : start + n_frames]
        assert block[0] == pytest.approx(w * SIM_TIME)
        assert block.max() < (w + 1) * SIM_TIME


def test_sensor_series_is_bit_identical_without_a_time_coordinate(
    tmp_path: pathlib.Path,
) -> None:
    """No ``time`` coord -> no rebasing, in both implementations."""
    datasets = [ds.drop_vars("time") for ds in _varied_cadence_datasets(seed=6)[:2]]
    paths = _write_windows(tmp_path, datasets)
    new = ensemble_sensor_series(paths, SENSOR_SETS, "pylbm", SIM_TIME)
    old = _reference_series(paths, SENSOR_SETS, "pylbm", SIM_TIME)
    _assert_identical_series(new, old)
    assert "time" not in new["assimilation"].coords


def test_sensor_series_is_bit_identical_on_a_staggered_udales_grid(
    tmp_path: pathlib.Path,
) -> None:
    """The reason the generator yields raw, un-co-located members."""
    paths = _write_windows(tmp_path, [_udales_window(seed=17), _udales_window(seed=18)])
    new = ensemble_sensor_series(paths, UDALES_SENSOR_SETS, "udales", SIM_TIME)
    old = _reference_series(paths, UDALES_SENSOR_SETS, "udales", SIM_TIME)
    _assert_identical_series(new, old)
    # The three components really did come off three different staggered grids.
    with xarray.open_dataset(paths[0]) as ds:
        assert len({ds["u"].dims, ds["v"].dims, ds["w"].dims}) == 3


def test_sensor_series_is_bit_identical_without_an_ensemble_coordinate(
    tmp_path: pathlib.Path,
) -> None:
    """The disk-backed run layout: ``ensemble`` dimension, no coordinate variable.

    ``run_esmda._stream_concat_members`` creates the dimension only, so this is
    what every non-in-memory run writes -- and it is the branch where the
    accumulator has no member label to rebuild the coordinate from.
    """
    datasets = [ds.drop_vars("ensemble") for ds in _varied_cadence_datasets(seed=8)]
    paths = _write_windows(tmp_path, datasets)
    new = ensemble_sensor_series(paths, SENSOR_SETS, "pylbm", SIM_TIME)
    old = _reference_series(paths, SENSOR_SETS, "pylbm", SIM_TIME)
    _assert_identical_series(new, old)
    assert "ensemble" not in new["assimilation"].coords


# --- the shared-pass entry points -------------------------------------------


def test_accumulator_reproduces_ensemble_sensor_series(
    wiring_fixture_paths: list[pathlib.Path],
) -> None:
    """What the WP1.3 driver does: one stream, its own loop, same answer."""
    accumulator = SensorSeriesAccumulator(SENSOR_SETS, "pylbm", SIM_TIME)
    for w, m, member in stream_window_members(wiring_fixture_paths):
        accumulator.add_member(w, m, member)
    _assert_identical_series(
        accumulator.result(),
        ensemble_sensor_series(wiring_fixture_paths, SENSOR_SETS, "pylbm", SIM_TIME),
    )


def test_accumulator_is_order_independent(
    wiring_fixture_paths: list[pathlib.Path],
) -> None:
    """Members may arrive in any order -- pieces are keyed, then sorted."""
    members = list(stream_window_members(wiring_fixture_paths))
    shuffled = SensorSeriesAccumulator(SENSOR_SETS, "pylbm", SIM_TIME)
    for w, m, member in reversed(members):
        shuffled.add_member(w, m, member)
    _assert_identical_series(
        shuffled.result(),
        ensemble_sensor_series(wiring_fixture_paths, SENSOR_SETS, "pylbm", SIM_TIME),
    )


def test_member_sensor_series_rebases_by_window_index(
    wiring_fixture_paths: list[pathlib.Path],
) -> None:
    """The load-bearing arithmetic, on one member, without the accumulator."""
    _w, _m, member = next(iter(stream_window_members(wiring_fixture_paths)))
    local = np.asarray(member["time"].values, dtype=float)
    for window_index in (0, 1, 7):
        series = member_sensor_series(
            member,
            SENSOR_SETS,
            "pylbm",
            window_index=window_index,
            sim_time=SIM_TIME,
        )
        got = np.asarray(series["assimilation"]["time"].values)
        assert np.array_equal(got, (local - local[0]) + window_index * SIM_TIME)


def test_the_sensor_pass_no_longer_loads_a_window_state_file() -> None:
    """Pins phase 1's acceptance grep to the functions it is about.

    Checked on the parsed AST rather than the text, so the ``.load()`` mentions
    in these functions' own docstrings and comments (they exist to explain what
    was removed) cannot make it pass or fail for the wrong reason. Scoped to
    these four rather than the whole module so unrelated additions elsewhere in
    ``_esmda_common`` cannot trip it.

    ``.compute()`` on a single member is a load *of a member*, which is the
    granularity the invariant prescribes; what must not come back is a load of
    the full-ensemble Dataset.
    """
    # `Any` because the tuple mixes plain functions with a class; mypy otherwise
    # joins them to `object`, which carries neither `__name__` nor a source.
    targets: tuple[Any, ...] = (
        stream_window_members,
        member_sensor_series,
        ensemble_sensor_series,
        SensorSeriesAccumulator,
    )
    for func in targets:
        tree = ast.parse(textwrap.dedent(inspect.getsource(func)))
        calls = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "load"
        ]
        assert calls == [], f"{func.__name__} calls .load()"
