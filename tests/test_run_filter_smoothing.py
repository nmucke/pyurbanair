"""Tests for scripts/filter_smoothing/run_filter_smoothing.py (the ESMDA/filter hybrid).

Two tiers, following tests/test_run_filtering.py:

* compose-only tests (no solver, no script import) that pin the config wiring —
  that the EXISTING ``esmda/*`` and ``filtering/*`` groups still mount into a
  third entry point and that every ``${esmda.*}`` / ``${filtering.*}``
  interpolation they carry resolves against ``conf/run_filter_smoothing.yaml``'s
  own blocks. These run anywhere;
* end-to-end smoke tests under the tiny smoke config (conftest
  ``_SMOKE_OVERRIDES``) with the global (unlocalized) update — the correlation
  localization the ESMDA entry point defaults to is degenerate at this 2-member
  ensemble size. Three of them, one per hybrid path: state + dynamic
  (per-segment trajectory restriction), joint + static (the exact-reduction
  path, where the hybrid is a plain joint EnKF on the MDA posterior) and
  joint + dynamic (the correction-on-the-ESMDA-schedule path).

The e2e tests build and run pylbm and are verified in CI, serially; they
SIGABRT on the maintainer's Mac (libomp / pylbm ``prepare_compile``), which is
the same limitation test_run_esmda.py and test_run_filtering.py live with.
"""

import pathlib
from typing import Any, Optional

import pytest


def _overrides(
    smoother: str,
    mode: str,
    num_windows: int,
    extra: Optional[list[str]] = None,
    model: str = "pylbm",
) -> list[str]:
    """The hybrid's smoke override set: test_run_esmda's plus test_run_filtering's.

    ``smoother`` picks the MDA half (``static`` | ``dynamic``) and, with it, the
    matching prior/truth params mounts — the anti-inverse-crime pairing both
    siblings use. ``mode`` picks the filter half (``state`` | ``joint``).
    """
    prior = "static" if smoother == "static" else "dynamic"
    truth = "static_truth" if prior == "static" else "dynamic_truth"
    ov = [
        # The barcelona case's STL has no solid cells on the tiny smoke domain
        # (the same environment limitation both siblings' tests hit); the
        # xie_and_castro cube array voxelizes fine there.
        "case=xie_and_castro",
        f"model@truth_model={model}",
        f"model@assim_model={model}",
        f"esmda/smoother={smoother}",
        f"params@prior_params={prior}",
        f"params@truth_params={truth}",
        f"filtering.mode={mode}",
        f"filter_smoothing.num_assimilation_windows={num_windows}",
        # One MDA iteration: the hybrid's cost is num_steps window rollouts PLUS
        # a full filter pass over the same window, so the smoke runs keep the
        # smoother at its minimum.
        "esmda.num_steps=1",
        # The smoke window is 3 s, so this is one aggregation interval — the
        # minimum the aggregator can score (see conftest's note on
        # simulation_time being pinned to 3.0 by exactly this).
        "esmda.interval_seconds=3.0",
        # The global (unlocalized) update on BOTH halves; correlation
        # localization is degenerate at 2 members.
        "esmda/localization=none",
        "filtering/localization=none",
        "ensemble.ensemble_size=2",
        "ensemble.num_parallel_processes=2",
        # The conftest smoke overrides pin a tiny [0,20]^2 domain but supply no
        # matching sensor coordinates; place the assimilation sensors in the
        # open N-S lanes of that domain (the same four points both siblings use).
        "obs.x_points=[2.5,2.5,18.0,18.0]",
        "obs.y_points=[5.0,15.0,5.0,15.0]",
        "obs.z_points=[3.0,3.0,3.0,3.0]",
        "run.skip_viz=true",
        # Simulate the truth inline (the shipped default, pinned so a retuned
        # config cannot point the suite at a scratch path).
        "run.truth_dir=null",
    ]
    if model == "pylbm":
        ov += [
            "truth_model.forward_model.cuda=false",
            "assim_model.forward_model.cuda=false",
        ]
    # conf/model/pyudales.yaml's nudging depth is scaled to the smoke domain by
    # the conftest (`_fit_nudging_to_smoke_domain`), so no mount-specific
    # override is needed here.
    ov += extra or []
    return ov


# ---------------------------------------------------------------------------
# Compose-only: the config wiring (no solver, no script import)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(  # type: ignore[misc]
    "smoother,target",
    [
        ("static", "data_assimilation.smoothing.esmda.ParameterESMDA"),
        ("dynamic", "data_assimilation.smoothing.esmda.TimeVaryingParameterESMDA"),
    ],
)
@pytest.mark.parametrize("mode", ["state", "joint"])  # type: ignore[misc]
def test_filter_smoothing_composes(
    smoother: str, target: str, mode: str, compose_test_cfg: Any
) -> None:
    """Both halves mount, for every supported (smoother, mode) combination.

    The point of keeping the config's node names ``esmda:`` / ``filtering:`` is
    that the existing groups compose here UNCHANGED, so this asserts the two
    ``_target_``s and — more importantly — that each group's ``${esmda.*}`` /
    ``${filtering.*}`` interpolations actually resolve against this entry
    point's own blocks rather than raising at access time.
    """
    cfg = compose_test_cfg(
        _overrides(smoother, mode, 1), config_name="run_filter_smoothing"
    )

    # The smoother half, and its interpolations into the shared `esmda:` block.
    assert cfg.esmda.smoother._target_ == target
    assert cfg.esmda.smoother.num_steps == cfg.esmda.num_steps == 1
    assert cfg.esmda.smoother.alpha == cfg.esmda.num_steps
    assert cfg.esmda.smoother.localization is None
    if smoother == "dynamic":
        # Present but a placeholder: the script overrides it at instantiate time
        # from the sampled prior's knot count.
        assert "num_time_points" in cfg.esmda.smoother
        assert cfg.esmda.smoother.pin_initial_time_point is True

    # The filter half, and its interpolations into the shared `filtering:` block.
    assert cfg.filtering.filter._target_ == (
        "data_assimilation.filtering.EnsembleKalmanFilter"
    )
    assert cfg.filtering.filter.mode == mode
    assert cfg.filtering.filter.analysis._target_ == (
        "data_assimilation.filtering.analysis.StochasticEnKFAnalysis"
    )
    # The shipped defaults: rtps inflation (the parameter-updating mode needs
    # spread maintenance), no evolution, no localization, no reduction.
    assert cfg.filtering.filter.inflation._target_ == "data_assimilation.inflation.RTPS"
    assert cfg.filtering.filter.parameter_evolution is None
    assert cfg.filtering.filter.localization is None
    assert cfg.filtering.filter.state_reduction is None

    # The shared knobs live on their own node, not duplicated onto either half.
    assert cfg.filter_smoothing.num_assimilation_windows == 1
    assert cfg.filter_smoothing.seed == 42
    assert cfg.filter_smoothing.obs_error_std > 0.0
    assert "num_assimilation_windows" not in cfg.esmda
    assert "num_assimilation_windows" not in cfg.filtering
    assert "obs_error_std" not in cfg.esmda
    assert "obs_error_std" not in cfg.filtering

    # v1 pins the filter's analysis stride, so run_filtering.yaml's knob is
    # deliberately NOT exposed here (a strided filter alongside a whole-window
    # smoother would need two separate observation products).
    assert "assimilate_every_n_step" not in cfg.filtering

    # Aggregation is a SMOOTHER-only knob; the filter assimilates raw frames.
    assert cfg.esmda.interval_seconds == 3.0
    assert cfg.esmda.aggregation_mode == "mean"
    assert "interval_seconds" not in cfg.filtering


@pytest.mark.parametrize(  # type: ignore[misc]
    "group,option,path,target",
    [
        (
            "filtering/inflation",
            "none",
            "inflation",
            None,
        ),
        (
            "filtering/evolution",
            "random_walk",
            "parameter_evolution",
            "data_assimilation.filtering.parameter_evolution.RandomWalkEvolution",
        ),
        (
            "filtering/state_reduction",
            "svd_current",
            "state_reduction",
            "data_assimilation.reduction.OnlineStateReduction",
        ),
        (
            "filtering/localization",
            "distance",
            "localization",
            "data_assimilation.localization.distance.DistanceLocalization",
        ),
    ],
)
def test_filter_smoothing_filtering_groups_compose(
    group: str, option: str, path: str, target: Optional[str], compose_test_cfg: Any
) -> None:
    """The filtering/* groups mount into the hybrid's filter target unchanged."""
    cfg = compose_test_cfg(
        _overrides("dynamic", "joint", 1, [f"{group}={option}"]),
        config_name="run_filter_smoothing",
    )
    node = cfg.filtering.filter[path]
    if target is None:
        assert node is None
    else:
        assert node._target_ == target


@pytest.mark.parametrize(  # type: ignore[misc]
    "option,target",
    [
        ("none", None),
        (
            "correlation",
            "data_assimilation.localization.correlation.CorrelationLocalization",
        ),
    ],
)
def test_filter_smoothing_esmda_localization_composes(
    option: str, target: Optional[str], compose_test_cfg: Any
) -> None:
    """The esmda/localization group still reaches the smoother target.

    The smoother's localization is a SEPARATE axis from the filter's — both
    groups write into their own node and both interpolate into their own
    ``_target_``, which is the whole reason the two node names were kept.
    """
    cfg = compose_test_cfg(
        _overrides("dynamic", "joint", 1, [f"esmda/localization={option}"]),
        config_name="run_filter_smoothing",
    )
    if target is None:
        assert cfg.esmda.smoother.localization is None
    else:
        assert cfg.esmda.smoother.localization._target_ == target
    # Independent of the filter's, which the override must not have touched.
    assert cfg.filtering.filter.localization is None


def test_filter_smoothing_default_config_is_the_documented_one(
    compose_test_cfg: Any,
) -> None:
    """The shipped defaults list, composed without any mode overrides.

    Pins the defaults the config's header documents (and the plan fixed), so a
    later edit to one of the shared groups cannot quietly change what a bare
    ``python scripts/filter_smoothing/run_filter_smoothing.py`` runs.
    """
    cfg = compose_test_cfg([], config_name="run_filter_smoothing")
    assert cfg.esmda.smoother._target_ == (
        "data_assimilation.smoothing.esmda.TimeVaryingParameterESMDA"
    )
    assert cfg.esmda.localization is None
    assert cfg.esmda.state_reduction is None
    assert cfg.filtering.mode == "joint"
    assert cfg.filtering.analysis._target_ == (
        "data_assimilation.filtering.analysis.StochasticEnKFAnalysis"
    )
    assert cfg.filtering.inflation._target_ == "data_assimilation.inflation.RTPS"
    assert cfg.filtering.parameter_evolution is None
    assert cfg.filtering.localization is None
    assert cfg.filtering.state_reduction is None
    # A dynamic prior is the default, so the default run takes the trajectory
    # (per-segment) path rather than the static one.
    assert "seconds_per_knot" in cfg.prior_params
    assert str(cfg.paths.results_dir)  # rewritten by the conftest's isolation


# ---------------------------------------------------------------------------
# Configuration rejections (no solver: every one of these fires before the
# truth is simulated)
# ---------------------------------------------------------------------------


def test_run_filter_smoothing_rejects_zero_windows(compose_test_cfg: Any) -> None:
    """num_assimilation_windows < 1 fails before any simulation is started."""
    from scripts.filter_smoothing.run_filter_smoothing import run

    cfg = compose_test_cfg(
        _overrides("dynamic", "joint", 0), config_name="run_filter_smoothing"
    )
    with pytest.raises(ValueError, match="num_assimilation_windows"):
        run(cfg)
    assert not (pathlib.Path(cfg.paths.results_dir) / "true_state.nc").exists()


def test_run_filter_smoothing_rejects_parameter_mode(compose_test_cfg: Any) -> None:
    """``filtering.mode=parameter`` is not a hybrid mode, and says why."""
    from scripts.filter_smoothing.run_filter_smoothing import run

    cfg = compose_test_cfg(
        _overrides("dynamic", "parameter", 1), config_name="run_filter_smoothing"
    )
    with pytest.raises(ValueError, match="not a hybrid mode"):
        run(cfg)
    assert not (pathlib.Path(cfg.paths.results_dir) / "true_state.nc").exists()


def test_run_filter_smoothing_rejects_state_bearing_smoother(
    compose_test_cfg: Any,
) -> None:
    """A state-bearing smoother would fight the filter for the state increment.

    FilterSmoothing's constructor rejects it too, but only after the truth has
    been simulated (its C_D comes from the truth's observations), so the script
    mirrors the check at config time — a mistyped smoother must not cost a full
    truth rollout first.
    """
    from scripts.filter_smoothing.run_filter_smoothing import run

    cfg = compose_test_cfg(
        _overrides("dynamic", "joint", 1, ["esmda/smoother=state_and_dynamic"]),
        config_name="run_filter_smoothing",
    )
    with pytest.raises(ValueError, match="state-bearing"):
        run(cfg)
    assert not (pathlib.Path(cfg.paths.results_dir) / "true_state.nc").exists()


def test_run_filter_smoothing_rejects_indivisible_cycles(
    compose_test_cfg: Any,
) -> None:
    """The filter's cycles must tile the smoother's window exactly.

    A partial cycle at the window boundary would leave the two halves scoring
    different observation sets, so the run refuses to start (the hybrid's
    counterpart of run_filtering.py's stride divisibility check).
    """
    from scripts.filter_smoothing.run_filter_smoothing import run

    cfg = compose_test_cfg(
        _overrides("dynamic", "joint", 1, ["time.output_frequency=0.4"]),
        config_name="run_filter_smoothing",
    )
    with pytest.raises(ValueError, match="does not divide"):
        run(cfg)
    assert not (pathlib.Path(cfg.paths.results_dir) / "true_state.nc").exists()


def test_nominal_window_clock_yields_exact_segment_bounds() -> None:
    """The PRODUCTION observation geometry composes with ``segment_bounds``.

    The truth's raw frame coordinates are 0-based and cadence-jittered (both
    backends rebase their post-spin-up frames to start at t = 0), which
    ``segment_bounds`` rightly rejects — the raw first frame of every window
    ends its segment at 0.0 or below the previous window's end. The script's
    ``_nominal_window_clock`` re-labels each cycle's batch with the nominal END
    of its forecast segment, and this test pins that composition at the unit
    level with realistic (jittered) truth times, so the e2e solver runs are
    not the only guard on it.
    """
    import numpy as np
    import xarray
    from data_assimilation.filter_smoothing import segment_bounds

    from scripts.filter_smoothing.run_filter_smoothing import _nominal_window_clock

    cycle_seconds = 2.5
    # Truth frames as the backends emit them: first at 0.0, later frames off
    # the nominal grid by cadence jitter (values from a real uDALES truth).
    raw_times = [0.0, 2.515311, 4.999622, 7.487478]
    batches = [
        _nominal_window_clock(
            xarray.DataArray(
                np.zeros((1, 3)),
                dims=("time", "obs"),
                coords={"time": [raw_times[k]], "obs": np.arange(3)},
            ),
            k,
            cycle_seconds,
        )
        for k in range(len(raw_times))
    ]
    assert segment_bounds(batches) == [
        (k * cycle_seconds, (k + 1) * cycle_seconds) for k in range(len(raw_times))
    ]

    # A multi-frame batch tiles its segment and still ends on the nominal grid.
    two_frames = _nominal_window_clock(
        xarray.DataArray(
            np.zeros((2, 3)),
            dims=("time", "obs"),
            coords={"time": [0.0, 1.2], "obs": np.arange(3)},
        ),
        3,
        cycle_seconds,
    )
    np.testing.assert_allclose(
        np.asarray(two_frames["time"].values), [3 * 2.5 + 1.25, 4 * 2.5]
    )


# ---------------------------------------------------------------------------
# End-to-end smoke runs (pylbm; CI, serially)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(  # type: ignore[misc]
    "smoother,mode,num_windows",
    [
        # The trajectory path with no parameter correction: the filter forecasts
        # each segment with the MDA trajectory restricted to it and updates the
        # state alone.
        pytest.param("dynamic", "state", 1, id="state_dynamic"),
        # The exact-reduction path: a static MDA posterior makes the filter
        # phase one plain joint-EnKF pass over the window's cycles. Two windows,
        # so the prior carry (posterior -> next prior) is exercised.
        pytest.param("static", "joint", 2, id="joint_static"),
        # The correction-on-the-ESMDA-schedule path: a trajectory posterior plus
        # a joint parameter update, per segment. Two windows, so the dynamic
        # carry (extrapolate -> next prior) and the per-window reset of the
        # correction are both exercised.
        pytest.param("dynamic", "joint", 2, id="joint_dynamic"),
    ],
)
def test_run_filter_smoothing(
    smoother: str, mode: str, num_windows: int, compose_test_cfg: Any
) -> None:
    import numpy as np
    import xarray

    from scripts.esmda._esmda_common import read_yaml
    from scripts.filter_smoothing.run_filter_smoothing import run

    cfg = compose_test_cfg(
        _overrides(smoother, mode, num_windows), config_name="run_filter_smoothing"
    )
    run(cfg)

    # Cycles are DERIVED from the observation cadence, not configured: the smoke
    # shape is a 3 s window written every 1 s, so a window holds three cycles.
    cycles_per_window = round(
        float(cfg.time.simulation_time) / float(cfg.time.output_frequency)
    )
    num_cycles = num_windows * cycles_per_window
    n_sensors = len(cfg.obs.x_points)
    n_obs_frame = len(cfg.obs.states) * n_sensors
    num_steps = int(cfg.esmda.num_steps)
    is_dynamic = smoother == "dynamic"

    out_dir = pathlib.Path(cfg.paths.results_dir)
    for name in (
        "posterior_state.nc",
        "cycle_diagnostics.yaml",
        "truth_access.yaml",
        "run_info.yaml",
        "config.yaml",
        "true_params.nc",
        "true_state.nc",
    ):
        assert (out_dir / name).exists(), name

    windows_dir = out_dir / "windows"
    truth_times = np.asarray(
        xarray.open_dataset(out_dir / "true_state.nc")["time"].values, dtype=float
    )
    for w in range(num_windows):
        # The window state is the FILTER's analyzed series — one frame per
        # cycle, on the window's own truth frame times — occupying the slots an
        # ESMDA window's posterior rollout would.
        posterior_state = xarray.open_dataset(
            windows_dir / f"window_{w}_posterior_state.nc"
        )
        assert posterior_state.sizes["time"] == cycles_per_window
        assert posterior_state.sizes["ensemble"] == 2
        np.testing.assert_allclose(
            np.asarray(posterior_state["time"].values, dtype=float),
            truth_times[w * cycles_per_window : (w + 1) * cycles_per_window],
        )
        # A hybrid has no window-long prior rollout, exactly as a filter has
        # none; run_info records the absence.
        assert not (windows_dir / f"window_{w}_prior_state.nc").exists()

        for name in ("prior_params", "posterior_params"):
            piece = xarray.open_dataset(windows_dir / f"window_{w}_{name}.nc")
            assert "inflow_angle" in piece.data_vars
            if is_dynamic:
                # A trajectory: the window-LOCAL knot axis, which the assembled
                # file rebases onto the global one.
                assert "time" in piece["inflow_angle"].dims
                assert float(piece["time"].min()) == pytest.approx(0.0)
            else:
                # A static scalar, stamped with the physical window time so the
                # assembled file carries seconds.
                assert "time" in piece.coords and piece["time"].shape == ()

        # The filter's corrected parameters are a different quantity from the
        # MDA posterior and get their own file — in joint mode only.
        filter_params_path = windows_dir / f"window_{w}_filter_params.nc"
        assert filter_params_path.exists() is (mode == "joint")

        # Observation space, the FILTERING schema: the raw per-cycle frames.
        obs = xarray.open_dataset(windows_dir / f"window_{w}_obs.nc")
        n_d = obs.sizes["obs_index"]
        assert n_d == cycles_per_window * n_obs_frame
        assert obs["obs_error_std"].shape == (n_d,)
        assert np.allclose(
            obs["obs_error_std"].values, float(cfg.filter_smoothing.obs_error_std)
        )
        # obs_interval reads as the cycle index within the window.
        assert set(np.unique(obs["obs_interval"].values)) == set(
            range(cycles_per_window)
        )
        pred = xarray.open_dataset(windows_dir / f"window_{w}_pred_obs.nc")
        # Two steps: 0 = the stacked per-cycle prior H(x_f), 1 = the posterior.
        assert pred["pred_obs"].shape == (2, n_d, 2)

        # Observation space, the SMOOTHER half: its own file, on the AGGREGATED
        # axis, with exactly num_steps entries and NO posterior entry — the
        # visible consequence of final_forecast=False, and the reason this is
        # not window_{w}_pred_obs.nc.
        esmda_pred = xarray.open_dataset(windows_dir / f"window_{w}_esmda_pred_obs.nc")
        assert esmda_pred.sizes["esmda_step"] == num_steps
        # esmda.interval_seconds == the whole smoke window, so the window's
        # frames aggregate into a single interval.
        assert esmda_pred.sizes["obs_index"] == n_obs_frame
        assert esmda_pred["pred_obs"].shape == (num_steps, n_obs_frame, 2)
        for var in ("obs", "obs_clean", "obs_error_std"):
            assert esmda_pred[var].shape == (n_obs_frame,)
        assert "final_forecast=False" in esmda_pred.attrs["esmda_step"]

    # Assembled, ESMDA schema — the shared metric/figure stages' inputs.
    posterior_params = xarray.open_dataset(out_dir / "posterior_params.nc")
    prior_params = xarray.open_dataset(out_dir / "prior_params.nc")
    assert "inflow_angle" in posterior_params.data_vars
    assert "inflow_angle" in prior_params.data_vars
    if not is_dynamic and num_windows > 1:
        # One scalar-time entry per window, on the physical axis.
        assert posterior_params.sizes["time"] == num_windows
    state_mean = xarray.open_dataset(out_dir / "posterior_state_mean.nc")
    assert state_mean.sizes["time"] == num_cycles
    assert "vel_mean" in state_mean.data_vars and "vel_std" in state_mean.data_vars
    assert "ensemble" not in state_mean.dims

    truth_access = read_yaml(out_dir / "truth_access.yaml")
    # The ESMDA keys keep their ESMDA meanings; the filtering stages read the
    # cycle geometry from its own keys. One frame per cycle, so the two agree.
    assert truth_access["num_windows"] == num_windows
    assert truth_access["n_per_window"] == cycles_per_window
    assert truth_access["sim_time"] == float(cfg.time.simulation_time)
    assert truth_access["cycle_seconds"] == float(cfg.time.output_frequency)
    assert truth_access["n_per_cycle"] == 1
    assert truth_access["num_cycles"] == num_cycles
    assert truth_access["moment_sampling_is_sparse"] is False

    configuration = read_yaml(out_dir / "run_info.yaml")["configuration"]
    assert configuration["hybrid"] == "FilterSmoothing"
    assert configuration["smoother"] == (
        "TimeVaryingParameterESMDA" if is_dynamic else "ParameterESMDA"
    )
    assert configuration["filter"] == "EnsembleKalmanFilter"
    assert configuration["mode"] == mode
    assert configuration["time_varying_parameters"] is is_dynamic
    assert configuration["num_assimilation_windows"] == num_windows
    assert configuration["cycles_per_window"] == cycles_per_window
    assert configuration["num_cycles"] == num_cycles
    assert configuration["num_esmda_steps"] == num_steps
    assert configuration["assimilate_every_n_step"] == 1
    assert configuration["save_obs_diagnostics"] is True
    assert configuration["save_prior_state"] is False
    assert configuration["num_observations_per_frame"] == n_obs_frame
    assert configuration["num_observations_per_window_aggregated"] == n_obs_frame
    assert configuration["analysis"]["_target_"] == (
        "data_assimilation.filtering.analysis.StochasticEnKFAnalysis"
    )
    assert configuration["localization"] is None
    assert configuration["esmda_localization"] is None

    # One diagnostics row per cycle over the WHOLE horizon, numbered globally:
    # the window boundary is invisible to the filtering stages.
    diagnostics = read_yaml(out_dir / "cycle_diagnostics.yaml")
    assert [row["cycle"] for row in diagnostics] == list(range(num_cycles))
    assert diagnostics[0]["analysis_time"] is not None

    # Histories (run.save_history defaults true).
    state_history = xarray.open_dataset(out_dir / "state_history.nc")
    assert state_history.sizes["cycle"] == num_cycles
    esmda_history = xarray.open_dataset(out_dir / "esmda_params_history.nc")
    # entry 0 is the prior, then one per MDA iteration.
    assert esmda_history.sizes["esmda_step"] == num_steps + 1
    assert esmda_history.sizes["window"] == num_windows

    params_history_path = out_dir / "params_history.nc"
    applied_history_path = out_dir / "applied_params_history.nc"
    if mode == "joint":
        # ONE entry per cycle: unlike a plain filter run's, this history carries
        # no prepended prior, so it indexes the same cycles as state_history and
        # cycle_diagnostics.
        params_history = xarray.open_dataset(params_history_path)
        assert params_history.sizes["cycle"] == num_cycles
        # The parameters each segment was actually forecast with exist only on
        # the joint + DYNAMIC path (on the static path the schedule is constant,
        # so there is nothing to record beyond the analyzed values).
        assert applied_history_path.exists() is is_dynamic
        if is_dynamic:
            applied = xarray.open_dataset(applied_history_path)
            assert applied.sizes["cycle"] == num_cycles
    else:
        # State mode: the filter never touches the parameters, so it produces
        # neither history.
        assert not params_history_path.exists()
        assert not applied_history_path.exists()

    posterior_state = xarray.open_dataset(out_dir / "posterior_state.nc")
    assert posterior_state.sizes["ensemble"] == 2
