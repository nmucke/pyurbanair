"""Tests for the high-rate probe re-runs (scripts/esmda/run_probe_series.py, WP3.2a).

The integration test runs the ESMDA smoke shape (conftest ``_SMOKE_OVERRIDES``)
and then re-probes its single window, checking the output schema *and* that the
high-rate snapshots the re-run produced were deleted again -- the whole point of
the script is that the full fields are transient.

The fallback (a truth that cannot be re-run) is covered by a unit test on a
synthetic truth file: it needs no solver, and the attrs it asserts are what the
downstream spectrum uses to truncate to the common resolved band.
"""

import pathlib
from collections.abc import Callable

import numpy as np
import pytest
import xarray
from omegaconf import DictConfig

# Same shape as tests/test_run_esmda.py's cheapest mode: pylbm truth + pylbm
# assimilation, static parameters, one window, sensors inside the smoke domain
# (the conftest smoke overrides shrink the domain but keep the case's sensors,
# which fall outside it).
_ESMDA_MODE = [
    "model@truth_model=pylbm",
    "model@assim_model=pylbm",
    "esmda/smoother=static",
    "params@prior_params=static",
    "params@truth_params=static_truth",
    "esmda.localization=null",
    "esmda.num_steps=1",
    "esmda.num_assimilation_windows=1",
    "ensemble.ensemble_size=2",
    "run.skip_viz=true",
    "run.truth_dir=null",
    "truth_model.forward_model.cuda=false",
    "assim_model.forward_model.cuda=false",
    "obs.x_points=[2.5,2.5,18.0,18.0]",
    "obs.y_points=[5.0,15.0,5.0,15.0]",
    "obs.z_points=[3.0,3.0,3.0,3.0]",
    "obs.interval_seconds=3.0",
]

# 4 assimilation sensors above + the single held-out sensor the smoke overrides
# pin inside the domain: the probes are the union of both sets.
_NUM_SENSORS = 5
_PROBE_FREQUENCY = 0.5
_PROBE_SPINUP = 1.0


def _assert_probe_schema(
    ds: xarray.Dataset, *, member_source: str, n_frames: int
) -> None:
    """The WP3.2 probe contract the spectrum metric (WP3.2b) reads."""
    assert set(ds.data_vars) == {"u", "v", "w"}
    assert ds.sizes["time"] == n_frames
    assert ds.sizes["sensor"] == _NUM_SENSORS
    assert set(ds["sensor_set"].values) == {"assimilation", "validation"}
    for coord in ("sensor_x", "sensor_y", "sensor_z"):
        assert ds[coord].dims == ("sensor",)
    assert ds.attrs["member_source"] == member_source
    assert ds.attrs["window_index"] == 0
    # The ACHIEVED cadence, not the requested one: `iout = round(cadence/dt)` is an
    # integer and `dt` comes from the member's own velocity scale, so the request is
    # almost never hit exactly (0.5 s asked, 0.5067 s delivered here). What must
    # hold exactly is that the attr equals the axis it labels; the request is only
    # approached.
    achieved = float(ds.attrs["output_frequency"])
    assert achieved == pytest.approx(_PROBE_FREQUENCY, rel=0.2)
    assert ds.attrs["spinup_time"] == pytest.approx(_PROBE_SPINUP)
    assert ds.attrs["cadence_fallback"] == 0
    # Time is seconds from the window start, uniformly sampled at the probe
    # cadence (the discarded lead-in is not in it).
    assert ds["time"].values[0] == pytest.approx(0.0)
    assert np.allclose(np.diff(ds["time"].values), achieved)
    assert np.isfinite(ds["u"].values).all()


def test_run_probe_series_smoke(
    compose_test_cfg: Callable[..., DictConfig],
) -> None:
    """A smoke ESMDA run, re-probed: correct dims, and no snapshots left behind."""
    from scripts.esmda.run_esmda import run as run_esmda
    from scripts.esmda.run_probe_series import run as run_probe_series

    esmda_cfg = compose_test_cfg(_ESMDA_MODE, config_name="run_esmda")
    run_esmda(esmda_cfg)
    run_dir = pathlib.Path(esmda_cfg.paths.results_dir)

    probe_cfg = compose_test_cfg(
        [
            # The probe run must be composed like the run it re-probes (the
            # script cross-checks the saved config), plus the probes knobs.
            #
            # It must COMPILE, even though the ESMDA run above just built a binary
            # for this exact grid into the shared test build tree: the experiment
            # name is compiled in (mod_dimensions + the geometry case), and the
            # probe models are constructed under `_PROBE_EXPERIMENT` precisely so
            # they cannot touch the assimilation run's experiment dir. So that
            # binary is stamped `runcase`, `compile=false` sees a stale stamp and
            # refuses, and the extra build is the price of that isolation. (On a
            # developer machine whose build cache already holds a probe-stamped
            # binary this passes either way, which is exactly why it took CI on a
            # clean cache to catch.)
            "case=xie_and_castro",
            *_ESMDA_MODE,
            f"probes.run_dir={run_dir}",
            "probes.window_index=0",
            f"probes.output_frequency={_PROBE_FREQUENCY}",
            f"probes.spinup_time={_PROBE_SPINUP}",
            "probes.include_prior=true",
            # Exercise the parallel member path (two tiny solver runs).
            "ensemble.num_parallel_processes=2",
        ],
        config_name="run_probe_series",
    )
    run_probe_series(probe_cfg)

    n_frames = int(round(esmda_cfg.time.simulation_time / _PROBE_FREQUENCY))

    truth = xarray.open_dataset(run_dir / "truth_probes.nc")
    assert truth["u"].dims == ("time", "sensor")
    _assert_probe_schema(truth, member_source="truth", n_frames=n_frames)

    for source, name in (
        ("posterior", "window_0_probes.nc"),
        ("prior", "window_0_probes_prior.nc"),
    ):
        members = xarray.open_dataset(run_dir / "windows" / name)
        assert members["u"].dims == ("ensemble", "time", "sensor")
        assert members.sizes["ensemble"] == 2
        _assert_probe_schema(members, member_source=source, n_frames=n_frames)

    # The high-rate snapshots (GBs per member at production resolution) must not
    # survive the run: every re-run's output dir is swept once its probes are out.
    leftovers = list(pathlib.Path(probe_cfg.paths.experiment_dir).rglob("out_*.nc"))
    assert leftovers == []


def _write_pylbm_truth(path: pathlib.Path, n_frames: int, cadence: float) -> None:
    """A tiny pylbm-shaped truth state file (cell-centred x/y/z, u/v/w)."""
    nx, ny, nz = 6, 6, 4
    rng = np.random.default_rng(0)
    dims = ("time", "z", "y", "x")
    shape = (n_frames, nz, ny, nx)
    xarray.Dataset(
        {name: (dims, rng.standard_normal(shape)) for name in ("u", "v", "w")},
        coords={
            "time": np.arange(n_frames) * cadence,
            "x": np.arange(nx) + 0.5,
            "y": np.arange(ny) + 0.5,
            "z": np.arange(nz) + 0.5,
        },
    ).to_netcdf(path)


def test_stored_truth_probes_record_the_fallback_cadence(
    tmp_path: pathlib.Path,
) -> None:
    """A truth that cannot be re-run is probed at its own cadence, and says so."""
    from scripts.esmda.run_probe_series import _probe_points, _stored_truth_probes

    truth_path = tmp_path / "true_state.nc"
    _write_pylbm_truth(truth_path, n_frames=8, cadence=2.5)
    truth_access = {
        "true_state_path": str(truth_path),
        "n_total": 8,
        "x_offset": 0.0,
        "start_idx": 0,
        "t_offset": 0.0,
        "n_per_window": 4,
        "truth_solver_name": "pylbm",
    }
    points, labels = _probe_points(
        {
            "assimilation": (
                np.array([1.5, 3.0]),
                np.array([2.5, 3.0]),
                np.array([1.5, 2.0]),
            ),
            "validation": (np.array([4.5]), np.array([4.5]), np.array([2.5])),
        }
    )

    probes = _stored_truth_probes(
        truth_access, 1, points, labels, "the truth was produced by 'pyudales'"
    )

    # Window 1 is the second half of the record, rebased to start at zero.
    assert probes["u"].dims == ("time", "sensor")
    assert probes.sizes == {"time": 4, "sensor": 3}
    assert np.allclose(probes["time"].values, [0.0, 2.5, 5.0, 7.5])
    assert probes.attrs["output_frequency"] == pytest.approx(2.5)
    assert probes.attrs["spinup_time"] == pytest.approx(0.0)
    assert probes.attrs["member_source"] == "truth"
    assert probes.attrs["cadence_fallback"] == 1
    assert "pyudales" in probes.attrs["cadence_fallback_reason"]
    assert probes["sensor_set"].values.tolist() == [
        "assimilation",
        "assimilation",
        "validation",
    ]


# --- Round-1 review regressions -------------------------------------------------
#
# Three fixes that the integration test above cannot see, because it composes a
# fresh scratch dir per run and synthesizes its own time axis.


def test_snapshot_grid_drops_the_extra_nt1_frame() -> None:
    """The trailing off-grid dump must not survive into the kept window.

    ``main.F90`` dumps on ``mod(it,iout)==0 .or. it==nt1``, and a warm start makes
    ``nt1`` a non-multiple of ``iout`` whenever ``nt0`` is -- the normal case here.
    Keeping that frame would put two snapshots one timestep apart at the end of a
    series whose synthetic axis says they are a full interval apart.
    """
    from scripts.esmda.run_probe_series import _window_snapshots

    iout, nt0 = 38, 1
    on_grid = [38 * k for k in range(1, 9)]
    files = [
        pathlib.Path(f"out_0000_F{it:06d}.nc") for it in [*on_grid, nt0 + 8 * iout]
    ]

    keep, drop = _window_snapshots(files, iout=iout, n_window=6)

    assert [int(p.stem.split("F")[1]) for p in keep] == on_grid[-6:]
    assert drop  # the off-grid nt1 frame plus the lead-in
    assert files[-1] in drop


# The probe re-run used to carry its own shim (`_link_restart_for_solver`) that
# re-exposed the wrapper's 9-digit restart under the 6-digit name the solver
# actually opens, plus its own overflow ceiling. Both are gone: the wrapper now
# writes the solver-readable name directly, so there is nothing to re-expose and
# the ceiling belongs to whoever spells the filename. That coverage lives in
# tests/test_pylbm_restart_filenames.py -- which additionally pins the width
# against the Fortran sources, so it cannot drift back.


def test_short_members_are_dropped_rather_than_stacked_ragged() -> None:
    """A member whose solver stopped early is dropped, and `member` records that.

    The real run this WP verified lost a member that exited 0 with 3 of 48 frames;
    stacking that is either a bare ValueError or a ragged comparison.
    """
    from scripts.esmda.run_probe_series import _stack_members

    full = [np.zeros((8, 3, 2)), np.zeros((3, 3, 2)), np.zeros((8, 3, 2))]

    stacked, cadences, members = _stack_members(full, [1.0, 1.0, 1.0], [0, 1, 2])

    assert stacked.shape == (2, 8, 3, 2)
    assert members == [0, 2]
    assert cadences == [1.0, 1.0]


def test_probe_models_are_built_under_their_own_experiment_name(
    monkeypatch: pytest.MonkeyPatch,
    compose_test_cfg: Callable[..., DictConfig],
) -> None:
    """Both re-run models must be constructed away from the run's `runcase` dir.

    Sharing it made the probe's `finally` clean output and prune restarts inside a
    live run's directory, and could leave a legacy-named restart holding the
    probe's own field where a later warm start would silently read it. Asserted on
    what reaches the forward-model constructor, because the config values alone are
    satisfied by an implementation that passes no `experiment_name` at all.
    """
    from scripts.esmda import run_probe_series

    seen: list[object] = []

    def _spy(target: object, **kwargs: object) -> object:
        seen.append(kwargs.get("experiment_name"))

        class _Stub:
            dirs = None

        return _Stub()

    monkeypatch.setattr(run_probe_series, "instantiate", _spy)
    cfg = compose_test_cfg(
        [*_ESMDA_MODE, "probes.max_members=1"], config_name="run_probe_series"
    )
    try:
        run_probe_series._member_models(
            cfg, 1, sim_time=3.0, output_frequency=1.0, spinup_time=0.0
        )
    except Exception:
        # The stub cannot be cloned; the constructor call is what is under test.
        pass

    # `prepare` is instantiated too and carries no experiment name; what matters is
    # that every name that IS set is the probe one. Deleting `experiment_name=` from
    # the forward-model construction leaves `named` empty and fails here.
    named = [name for name in seen if name is not None]
    assert named, f"no forward model was constructed with an experiment name: {seen}"
    assert all(name == run_probe_series._PROBE_EXPERIMENT for name in named), named
    assert run_probe_series._PROBE_EXPERIMENT != "runcase"


def test_window_snapshots_rejects_a_run_that_died_in_the_lead_in() -> None:
    """A stop inside the lead-in still leaves enough on-grid frames to slice.

    The trailing slice would then return a window shifted back into the discarded
    lead-in -- evenly spaced, so the hole check passes too. Only the exact expected
    count catches it.
    """
    from scripts.esmda.run_probe_series import _window_snapshots

    iout, n_window, lead_in = 10, 6, 4
    full = [
        pathlib.Path(f"out_0000_F{iout * k:06d}.nc")
        for k in range(1, n_window + lead_in + 1)
    ]

    keep, _ = _window_snapshots(
        full, iout=iout, n_window=n_window, expected_total=n_window + lead_in
    )
    assert len(keep) == n_window

    died_in_lead_in = full[: n_window + 1]
    with pytest.raises(RuntimeError, match="lead-in"):
        _window_snapshots(
            died_in_lead_in,
            iout=iout,
            n_window=n_window,
            expected_total=n_window + lead_in,
        )


def test_window_snapshots_rejects_a_hole_in_the_kept_window() -> None:
    """A missing snapshot must raise, not be papered over by the uniform axis."""
    from scripts.esmda.run_probe_series import _window_snapshots

    iout, n_window = 10, 4
    holed = [pathlib.Path(f"out_0000_F{iout * k:06d}.nc") for k in (1, 2, 3, 5, 6)]

    with pytest.raises(RuntimeError, match="evenly spaced"):
        _window_snapshots(
            holed, iout=iout, n_window=n_window, expected_total=len(holed)
        )
