"""End-to-end smoke tests for scripts/filtering/run_filtering.py (sequential EnKF).

Mirrors tests/test_run_esmda.py: everything runs under the tiny smoke config
(conftest ``_SMOKE_OVERRIDES``) with the global (unlocalized) update — the
default correlation localization is degenerate at this 2-member ensemble size.
One test per filter mode (state / parameter / joint), plus a multi-cycle run,
distance localization (purely geometric, so meaningful at 2 members), the
deterministic ensemble-transform analyses, and the mixed drift-tracking path (a
dynamic truth tracked by a static prior).

The compose-only tests (state reduction, analysis scheme) need no solver and
cover every option of their group; the solver-running tests stay deliberately
few.
"""

import pathlib
from typing import Any, Optional

import pytest


def _overrides(
    mode: str,
    num_windows: int,
    extra: Optional[list[str]] = None,
    model: str = "pylbm",
) -> list[str]:
    ov = [
        # The barcelona default's STL has no solid cells on the tiny smoke
        # domain (the same environment limitation that blocks test_run_esmda);
        # the xie_and_castro cube array voxelizes fine there.
        "case=xie_and_castro",
        f"model@truth_model={model}",
        f"model@assim_model={model}",
        # Pin a static truth so these stay deterministic static-parameter smoke
        # tests independent of the config's default truth sampler (which is a
        # dynamic drift-tracking truth); the mixed dynamic-truth path has its own
        # test below.
        "params@truth_params=static_truth",
        f"filtering.mode={mode}",
        f"filtering.num_assimilation_windows={num_windows}",
        # The global (unlocalized) update; correlation localization is
        # degenerate at 2 members.
        "filtering/localization=none",
        "ensemble.ensemble_size=2",
        "ensemble.num_parallel_processes=2",
        # The conftest smoke overrides pin a tiny [0,20]^2 domain but do not
        # supply matching sensor coordinates; place the assimilation sensors in
        # the open N-S lanes of that domain (same points as test_run_esmda).
        "obs.x_points=[2.5,2.5,18.0,18.0]",
        "obs.y_points=[5.0,15.0,5.0,15.0]",
        "obs.z_points=[3.0,3.0,3.0,3.0]",
        # No aggregation override here: the sequential filter assimilates every
        # observation frame of a segment serially, so `filtering.interval_seconds`
        # does not exist (Hydra would reject it). Thinning the ANALYSES (not
        # averaging the observations) is `filtering.assimilate_every_n_step`,
        # which has its own tests at the end of this file.
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


@pytest.mark.parametrize(  # type: ignore[misc]
    "option,target",
    [
        ("none", None),
        ("svd_current", "data_assimilation.reduction.OnlineStateReduction"),
        ("svd_streaming", "data_assimilation.reduction.StreamingStateReduction"),
    ],
)
def test_filtering_state_reduction_config_composes(
    option: str, target: Optional[str], compose_test_cfg: Any
) -> None:
    """Every reduction-group option mounts into the filter target."""
    cfg = compose_test_cfg(
        _overrides("state", 1, [f"filtering/state_reduction={option}"]),
        config_name="run_filtering",
    )
    reduction = cfg.filtering.filter.state_reduction
    if target is None:
        assert reduction is None
    else:
        assert reduction._target_ == target
    if option == "svd_streaming":
        # The exponentially weighted accumulator keeps absorbing directions, so
        # an energy criterion alone would let the basis grow every cycle.
        assert reduction.max_rank is not None


@pytest.mark.parametrize(  # type: ignore[misc]
    "option,target,localization,truncated",
    [
        (
            "stochastic",
            "data_assimilation.filtering.analysis.StochasticEnKFAnalysis",
            "none",
            None,
        ),
        ("etkf", "data_assimilation.filtering.etkf.ETKFAnalysis", "none", False),
        ("etkf_tsvd", "data_assimilation.filtering.etkf.ETKFAnalysis", "none", True),
        ("letkf", "data_assimilation.filtering.etkf.LETKFAnalysis", "distance", False),
        (
            "letkf_tsvd",
            "data_assimilation.filtering.etkf.LETKFAnalysis",
            "distance",
            True,
        ),
    ],
)
def test_filtering_analysis_config_composes(
    option: str,
    target: str,
    localization: str,
    truncated: Optional[bool],
    compose_test_cfg: Any,
) -> None:
    """Every analysis-group option mounts into the filter target.

    Each option is composed with the localization its ``localization_policy``
    allows (``etkf*`` forbids localization, ``letkf*`` requires it), so the
    composed config is one BaseFilter would accept. The localization override
    comes last and therefore wins over the ``none`` pinned in ``_overrides``.
    """
    cfg = compose_test_cfg(
        _overrides(
            "state",
            1,
            [
                f"filtering/analysis={option}",
                f"filtering/localization={localization}",
            ],
        ),
        config_name="run_filtering",
    )
    analysis = cfg.filtering.filter.analysis
    assert analysis._target_ == target
    if truncated is None:
        # The stochastic analysis takes no TSVD settings at all.
        assert "tsvd" not in analysis
    elif not truncated:
        assert analysis.tsvd is None
    else:
        # A nested `_target_` under a `_target_`: recursive instantiation turns
        # this node into an ObservationTSVD before the analysis is constructed.
        assert (
            analysis.tsvd._target_ == "data_assimilation.filtering.etkf.ObservationTSVD"
        )
        assert analysis.tsvd.enabled is True


@pytest.mark.parametrize(  # type: ignore[misc]
    "mode,num_windows,extra",
    [
        pytest.param("joint", 1, None, id="joint"),
        pytest.param("joint", 2, None, id="joint_two_windows"),
        pytest.param("state", 1, None, id="state"),
        pytest.param(
            "state",
            1,
            ["filtering/state_reduction=svd_current"],
            id="state_svd_current",
        ),
        # Parameter mode needs spread maintenance; the random-walk evolution
        # exercises the parameter forecast model too.
        pytest.param(
            "parameter", 2, ["filtering/evolution=random_walk"], id="parameter"
        ),
    ],
)
def test_run_filtering(
    mode: str, num_windows: int, extra: Optional[list[str]], compose_test_cfg: Any
) -> None:
    import numpy as np
    import xarray

    from scripts.filtering.run_filtering import run

    cfg = compose_test_cfg(
        _overrides(mode, num_windows, extra), config_name="run_filtering"
    )
    run(cfg)

    # Cycles are DERIVED from the observation cadence, not configured: the smoke
    # shape is a 3 s window written every 1 s, so a window holds three cycles.
    cycles_per_window = round(
        float(cfg.time.simulation_time) / float(cfg.time.output_frequency)
    )
    num_cycles = num_windows * cycles_per_window

    out_dir = pathlib.Path(cfg.paths.results_dir)
    assert (out_dir / "posterior_state.nc").exists()
    assert (out_dir / "cycle_diagnostics.yaml").exists()
    assert (out_dir / "run_info.yaml").exists()

    # The ESMDA-schema per-window artifacts the shared metric/figure stages read
    # (see the artifact contract). No prior STATE file: a filter has no
    # window-long prior rollout, and its absence is what run_info explains.
    windows_dir = out_dir / "windows"
    truth_times = np.asarray(
        xarray.open_dataset(out_dir / "true_state.nc")["time"].values, dtype=float
    )
    for w in range(num_windows):
        posterior_state = xarray.open_dataset(
            windows_dir / f"window_{w}_posterior_state.nc"
        )
        # One analyzed frame per cycle, carrying the window's own truth frame
        # times (the solver's cadence, not a nominal multiple of the window).
        assert posterior_state.sizes["time"] == cycles_per_window
        assert posterior_state.sizes["ensemble"] == 2
        np.testing.assert_allclose(
            np.asarray(posterior_state["time"].values, dtype=float),
            truth_times[w * cycles_per_window : (w + 1) * cycles_per_window],
        )
        for name in ("prior_params", "posterior_params"):
            piece = xarray.open_dataset(windows_dir / f"window_{w}_{name}.nc")
            # A scalar PHYSICAL time coord, so the assembled files carry seconds.
            assert "time" in piece.coords and piece["time"].shape == ()
        assert not (windows_dir / f"window_{w}_prior_state.nc").exists()

        obs = xarray.open_dataset(windows_dir / f"window_{w}_obs.nc")
        n_d = obs.sizes["obs_index"]
        assert n_d % cycles_per_window == 0
        # obs_error_std is TILED to the window's full length, not the frame's.
        assert obs["obs_error_std"].shape == (n_d,)
        assert np.allclose(
            obs["obs_error_std"].values, float(cfg.filtering.obs_error_std)
        )
        # obs_interval reads as the cycle index within the window.
        assert set(np.unique(obs["obs_interval"].values)) == set(
            range(cycles_per_window)
        )
        pred = xarray.open_dataset(windows_dir / f"window_{w}_pred_obs.nc")
        # Two steps: 0 = the stacked per-cycle prior H(x_f), 1 = the posterior.
        assert pred["pred_obs"].shape == (2, n_d, 2)

    # Assembled, ESMDA-schema: one entry per window on a physical time axis.
    posterior_params = xarray.open_dataset(out_dir / "posterior_params.nc")
    if num_windows > 1:
        assert posterior_params.sizes["time"] == num_windows
    state_mean = xarray.open_dataset(out_dir / "posterior_state_mean.nc")
    assert state_mean.sizes["time"] == num_cycles
    assert "vel_mean" in state_mean.data_vars and "vel_std" in state_mean.data_vars
    assert "ensemble" not in state_mean.dims

    from scripts.esmda._esmda_common import read_yaml

    truth_access = read_yaml(out_dir / "truth_access.yaml")
    # ESMDA keys keep their ESMDA meanings; the cycle geometry has its own.
    assert truth_access["num_windows"] == num_windows
    assert truth_access["n_per_window"] == cycles_per_window
    assert truth_access["sim_time"] == float(cfg.time.simulation_time)
    assert truth_access["cycle_seconds"] == float(cfg.time.output_frequency)
    assert truth_access["n_per_cycle"] == 1
    assert truth_access["num_cycles"] == num_cycles
    posterior_state = xarray.open_dataset(out_dir / "posterior_state.nc")
    assert posterior_state.sizes["ensemble"] == 2
    from scripts.esmda._esmda_common import read_yaml

    configuration = read_yaml(out_dir / "run_info.yaml")["configuration"]
    # `filter` stays EnsembleKalmanFilter for every update flavor, so the
    # analysis subtree is what identifies the update math that ran.
    assert configuration["filter"] == "EnsembleKalmanFilter"
    assert configuration["analysis"]["_target_"] == (
        "data_assimilation.filtering.analysis.StochasticEnKFAnalysis"
    )
    assert configuration["num_assimilation_windows"] == num_windows
    assert configuration["cycles_per_window"] == cycles_per_window
    assert configuration["num_cycles"] == num_cycles
    assert configuration["save_obs_diagnostics"] is True
    assert configuration["save_prior_state"] is False
    diagnostics = read_yaml(out_dir / "cycle_diagnostics.yaml")
    # One row per cycle over the WHOLE horizon, numbered globally: the window
    # boundary is invisible to the filtering stages.
    assert [row["cycle"] for row in diagnostics] == list(range(num_cycles))
    reduction_enabled = bool(extra) and any(
        override.startswith("filtering/state_reduction=")
        and not override.endswith("=none")
        for override in (extra or [])
    )
    if not reduction_enabled:
        # The unreduced default must still write a well-formed run_info and a
        # complete (all-None) reduction block in the diagnostics schema.
        assert configuration["state_reduction"] is None
        assert configuration["state_reduction_resolved_variable_scales"] is None
        assert diagnostics[0]["reduction_rank"] is None
        assert diagnostics[0]["reduction_basis_time"] is None
        assert diagnostics[0]["obs_posterior_rmse_kind"] == "exact"
        assert diagnostics[0]["analysis_time"] is not None
    else:
        reduction = configuration["state_reduction"]
        assert reduction["_target_"] == (
            "data_assimilation.reduction.OnlineStateReduction"
        )
        assert reduction["whiten"] is False
        resolved_scales = configuration["state_reduction_resolved_variable_scales"]
        assert resolved_scales
        assert set(resolved_scales.values()) == {1.0}

        # Two members leave exactly one statistical mode.
        assert diagnostics[0]["reduction_rank"] == 1
        assert diagnostics[0]["reduction_available_rank"] == 1
        assert diagnostics[0]["analysis_time"] is not None
        assert diagnostics[0]["reduction_basis_time"] is not None
        assert diagnostics[0]["reduction_basis_updated"] is True
        assert diagnostics[0]["reduction_discarded_increment_fraction"] is not None
        # The observation-space posterior rides along on the full-space update,
        # so it must be labelled rather than compared with an unreduced run.
        assert diagnostics[0]["obs_posterior_rmse_kind"] == "unreduced_ride_along"
    if mode != "state":
        assert "inflow_angle" in posterior_params.data_vars
        # History: EXACTLY one leading prior plus one entry per cycle over the
        # horizon — every window's run() prepends the params it was handed, and
        # the repeats at the window boundaries are dropped.
        history = xarray.open_dataset(out_dir / "params_history.nc")
        assert history.sizes["cycle"] == num_cycles + 1
        # State history: one analyzed frame per cycle, accumulated over windows.
        state_history = xarray.open_dataset(out_dir / "state_history.nc")
        assert state_history.sizes["cycle"] == num_cycles


def test_run_filtering_distance_localization(compose_test_cfg: Any) -> None:
    """State rows localized by physical distance (params stay global)."""
    from scripts.filtering.run_filtering import run

    cfg = compose_test_cfg(
        _overrides(
            "joint",
            1,
            [
                "filtering/localization=distance",
                "filtering.localization.localization_radius=10.0",
            ],
        ),
        config_name="run_filtering",
    )
    run(cfg)


@pytest.mark.parametrize(  # type: ignore[misc]
    "option,localization,target",
    [
        pytest.param(
            "etkf",
            "none",
            "data_assimilation.filtering.etkf.ETKFAnalysis",
            id="etkf",
        ),
        pytest.param(
            "etkf_tsvd",
            "none",
            "data_assimilation.filtering.etkf.ETKFAnalysis",
            id="etkf_tsvd",
        ),
        pytest.param(
            "letkf",
            "distance",
            "data_assimilation.filtering.etkf.LETKFAnalysis",
            id="letkf",
        ),
        pytest.param(
            "letkf_tsvd",
            "distance",
            "data_assimilation.filtering.etkf.LETKFAnalysis",
            id="letkf_tsvd",
        ),
    ],
)
def test_run_filtering_ensemble_transform(
    option: str, localization: str, target: str, compose_test_cfg: Any
) -> None:
    """Deterministic ensemble-transform analyses end to end.

    All four deterministic options run the solver, covering the full
    (analysis class) x (TSVD on/off) grid rather than only its diagonal. Each
    case builds and runs pylbm, so this is the most expensive coverage in the
    file; it is kept because a nested-``_target_`` TSVD node reaching a real
    run is exactly the wiring the compose-only tests above cannot prove.

    At the smoke ensemble size (``ensemble.ensemble_size=2``) the forecast
    anomalies span a single direction, so the transform is well defined — the
    kernel needs `N_e >= 2`, the zero singular values are damped rather than
    divided by, and mean preservation is structural — but it is a rank-1
    posterior covariance. These assert plumbing and provenance only; nothing
    about filter skill can be read off a 2-member deterministic transform.
    """
    from scripts.esmda._esmda_common import read_yaml
    from scripts.filtering.run_filtering import run

    cfg = compose_test_cfg(
        _overrides(
            "joint",
            1,
            [
                f"filtering/analysis={option}",
                f"filtering/localization={localization}",
                *(
                    ["filtering.localization.localization_radius=10.0"]
                    if localization == "distance"
                    else []
                ),
            ],
        ),
        config_name="run_filtering",
    )
    run(cfg)

    out_dir = pathlib.Path(cfg.paths.results_dir)
    configuration = read_yaml(out_dir / "run_info.yaml")["configuration"]
    # The composed filter class is EnsembleKalmanFilter either way; only the
    # recorded analysis subtree distinguishes the update math.
    assert configuration["filter"] == "EnsembleKalmanFilter"
    assert configuration["analysis"]["_target_"] == target
    if option.endswith("_tsvd"):
        assert configuration["analysis"]["tsvd"]["enabled"] is True
    else:
        assert configuration["analysis"]["tsvd"] is None
    # The analysis and the localization are a coupled pair (localization_policy),
    # and the per-block diagnostics below mean nothing without the strategy that
    # produced them, so run_info records the resolved localization subtree too.
    if localization == "none":
        assert configuration["localization"] is None
    else:
        assert configuration["localization"]["localization_radius"] == 10.0

    # The transform diagnostics survive the dataclass -> YAML round trip (they
    # would not if any reached write_yaml as a JAX scalar: its `_to_native`
    # converts NumPy, not JAX), and each group is populated exactly on the path
    # where it exists.
    cycle = read_yaml(out_dir / "cycle_diagnostics.yaml")[0]
    transform_fields = [k for k in cycle if k.startswith("transform_")]
    local_fields = [k for k in cycle if k.startswith("local_")]
    assert len(transform_fields) == 4 and len(local_fields) == 14
    if option.startswith("etkf"):
        assert cycle["transform_retained_rank"] >= 1
        assert cycle["transform_available_rank"] is not None
        assert all(cycle[k] is None for k in local_fields)
    else:
        assert cycle["local_num_blocks"] >= 1
        assert cycle["local_num_active_blocks"] >= 1
        assert cycle["local_chunk_size"] >= 1
        # The per-block energy readouts are the other half of plan step 5's
        # four mandated diagnostics; assert them explicitly so they cannot
        # regress to None while the rank fields keep the test green.
        assert 0.0 <= cycle["local_retained_energy_min"] <= 1.0
        assert 0.0 <= cycle["local_retained_energy_mean"] <= 1.0
        assert cycle["local_discarded_spectrum_max"] >= 0.0
        assert all(cycle[k] is None for k in transform_fields)


def test_run_filtering_tracks_dynamic_truth(compose_test_cfg: Any) -> None:
    """Mixed mode: a time-varying (dynamic) TRUTH tracked by a static prior.

    The filter's scalar estimate tracks a drifting truth. The dynamic truth is
    sampled over the full horizon (num_assimilation_windows windows, so it
    varies across cycles), and the static prior is analysed/evolved each cycle.
    Exercises the truth-sampling horizon override and the parameter-block update
    against a time-varying truth.
    """
    import xarray

    from scripts.filtering.run_filtering import run

    cfg = compose_test_cfg(
        _overrides(
            "joint",
            2,
            [
                # Override the pinned static truth back to the dynamic one; the
                # prior stays static (the default), which is the mixed setup.
                "params@truth_params=dynamic_truth",
                "filtering/evolution=random_walk",
            ],
        ),
        config_name="run_filtering",
    )
    run(cfg)

    out_dir = pathlib.Path(cfg.paths.results_dir)
    # The truth params are time-varying and span the full horizon
    # (num_assimilation_windows * simulation_time), so the truth drifts across
    # every cycle.
    true_params = xarray.open_dataset(out_dir / "true_params.nc")
    assert "time" in true_params["inflow_angle"].dims
    assert (
        float(true_params["time"].max()) >= 2 * float(cfg.time.simulation_time) - 1e-6
    )
    # The per-window params stay static scalars the filter tracks; the assembled
    # file's only `time` axis is the one window boundary the pieces were stamped
    # with, so the drifting truth is interpolated at real seconds.
    posterior = xarray.open_dataset(out_dir / "posterior_params.nc")
    assert set(posterior["inflow_angle"].dims) == {"ensemble", "time"}
    assert list(posterior["time"].values) == [
        float(cfg.time.simulation_time),
        2 * float(cfg.time.simulation_time),
    ]


def test_run_filtering_rejects_zero_windows(compose_test_cfg: Any) -> None:
    """num_assimilation_windows < 1 fails before any simulation is started."""
    from scripts.filtering.run_filtering import run

    cfg = compose_test_cfg(_overrides("joint", 0), config_name="run_filtering")
    with pytest.raises(ValueError, match="num_assimilation_windows"):
        run(cfg)


def test_run_filtering_rejects_dynamic_prior(compose_test_cfg: Any) -> None:
    """The filter is static-parameters-only; a dynamic prior fails loudly."""
    from scripts.filtering.run_filtering import run

    cfg = compose_test_cfg(
        _overrides(
            "parameter",
            1,
            [
                "params@prior_params=dynamic",
                "params@truth_params=dynamic_truth",
                "filtering/evolution=random_walk",
            ],
        ),
        config_name="run_filtering",
    )
    with pytest.raises(ValueError, match="time-varying"):
        run(cfg)


# ---------------------------------------------------------------------------
# filtering.assimilate_every_n_step — the analysis stride
# ---------------------------------------------------------------------------


def test_filtering_assimilate_every_n_step_defaults_to_one(
    compose_test_cfg: Any,
) -> None:
    """The stride key ships defaulted to 1, and setting it to 1 changes nothing.

    First half of the n=1 identity guarantee: an unstrided run is configured
    exactly as it was before the knob existed. (The second half — that the n=1
    CODE path is the old one — is the tripwire in
    ``test_run_filtering_assimilate_every_n_step``.)
    """
    default_cfg = compose_test_cfg(_overrides("joint", 1), config_name="run_filtering")
    assert default_cfg.filtering.assimilate_every_n_step == 1

    explicit_cfg = compose_test_cfg(
        _overrides("joint", 1, ["filtering.assimilate_every_n_step=1"]),
        config_name="run_filtering",
    )
    # The composer hands every call its own isolated output roots, so those are
    # the one legitimate difference; everything else must match.
    for cfg in (default_cfg, explicit_cfg):
        cfg.paths.results_dir = ""
        cfg.paths.experiment_dir = ""
        cfg.paths.base_results_dir = ""
    assert default_cfg == explicit_cfg


def test_strided_observation_operator_subsets_the_assimilated_frames() -> None:
    """The predicted-observation stride: identity at 1, last-of-block above.

    The wrapper is what keeps the predicted observations aligned row for row
    with the truth-side subset. At stride 1 it is the identity, which is why
    ``run()`` can simply not build it (see the tripwire below) rather than rely
    on it being harmless.
    """
    import numpy as np
    import xarray

    from scripts.filtering.run_filtering import _StridedObservationOperator

    frames = xarray.DataArray(
        np.arange(12.0).reshape(4, 3),
        dims=("time", "obs"),
        coords={"time": [1.0, 2.0, 3.0, 4.0]},
    )

    class _TemporalOp:
        num_obs = 3

        def __call__(self, state: Any) -> xarray.DataArray:
            return frames

    # The stub operators ignore it; the wrapper only ever forwards it.
    unused = xarray.Dataset()

    strided = _StridedObservationOperator(_TemporalOp(), 2)
    out = strided(unused)
    # Every 2nd frame, ending on the block's LAST frame — the analysis time.
    np.testing.assert_array_equal(np.asarray(out["time"].values), [2.0, 4.0])
    np.testing.assert_array_equal(out.values, frames.values[1::2])
    # Attribute access is delegated, so the wrapper stays transparent to the
    # library's operator introspection (sensor_observation_coords et al).
    assert strided.num_obs == 3

    np.testing.assert_array_equal(
        _StridedObservationOperator(_TemporalOp(), 1)(unused).values, frames.values
    )

    # A segment whose frame count the stride does not divide would be subset to
    # a frame that is NOT the one the analysis updates (the segment's last), and
    # would still line up with the truth side row for row -- a plausible, wrong
    # analysis rather than an error. Unstrided, the library's own row check
    # raises on exactly this, so this one must too.
    with pytest.raises(ValueError, match="does not divide"):
        _StridedObservationOperator(_TemporalOp(), 3)(unused)

    class _BareOp:
        def __call__(self, state: Any) -> Any:
            return np.arange(3.0)

    # A bare (unlabelled) operator already reduces a segment to its last frame,
    # so there is nothing to stride and its output is passed through.
    np.testing.assert_array_equal(
        _StridedObservationOperator(_BareOp(), 2)(unused), np.arange(3.0)
    )


def test_unstrided_truth_subset_is_the_single_frame_slice() -> None:
    """At n=1 the cycle's block-plus-stride slice IS the old one-frame slice.

    The truth-side half of the n=1 identity guarantee: ``run()`` now takes a
    cycle's ``every_n``-frame block and strides it, and at ``every_n = 1`` that
    composition must be the very same frame the every-observation filter took —
    so the observation values it perturbs, and therefore the noise draws made
    against their shape, are unchanged.
    """
    import numpy as np
    import xarray

    truth = xarray.Dataset(
        {"u": (("time", "x"), np.arange(20.0).reshape(5, 4))},
        coords={"time": np.arange(5.0)},
    )
    for cycle in range(truth.sizes["time"]):
        block = truth.isel(time=slice(cycle * 1, (cycle + 1) * 1))
        xarray.testing.assert_identical(
            block.isel(time=slice(0, None, 1)),
            truth.isel(time=slice(cycle, cycle + 1)),
        )


@pytest.mark.parametrize(  # type: ignore[misc]
    "every_n,match",
    [
        pytest.param(0, "assimilate_every_n_step must be >= 1", id="zero"),
        pytest.param(2, "does not divide", id="indivisible"),
    ],
)
def test_run_filtering_rejects_bad_stride(
    every_n: int, match: str, compose_test_cfg: Any
) -> None:
    """A bad stride fails at CONFIG time, before any solver is started.

    The smoke shape is a 3 s window written every 1 s, i.e. 3 observation
    frames per window, which a stride of 2 does not divide: the cycles would
    not tile the window and the odd frame would be dropped silently.
    """
    from scripts.filtering.run_filtering import run

    cfg = compose_test_cfg(
        _overrides("joint", 1, [f"filtering.assimilate_every_n_step={every_n}"]),
        config_name="run_filtering",
    )
    with pytest.raises(ValueError, match=match):
        run(cfg)
    # Nothing ran: the truth is the first thing the script would simulate.
    assert not (pathlib.Path(cfg.paths.results_dir) / "true_state.nc").exists()


@pytest.mark.parametrize(  # type: ignore[misc]
    "every_n,num_windows",
    [
        pytest.param(1, 1, id="unstrided"),
        pytest.param(2, 2, id="stride2"),
    ],
)
def test_run_filtering_assimilate_every_n_step(
    every_n: int,
    num_windows: int,
    compose_test_cfg: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """End to end on uDALES: a stride thins the ANALYSES, not the output.

    The ``unstrided`` case additionally trips a wire on the stride wrapper: at
    ``assimilate_every_n_step = 1`` the filter must be handed the observation
    operator object itself, not a wrapper around it, which is what makes a
    default run bit-identical to the every-observation filter.

    The window is stretched to 4 s (4 observation frames) so that a stride of 2
    still leaves two cycles per window — enough to read the analysis cadence
    off the per-window state file rather than infer it.
    """
    import numpy as np
    import xarray

    import scripts.filtering.run_filtering as run_filtering
    from scripts.esmda._esmda_common import read_yaml

    if every_n == 1:

        def _tripwire(*args: Any, **kwargs: Any) -> Any:
            raise AssertionError(
                "assimilate_every_n_step=1 must hand the filter the "
                "observation operator UNWRAPPED (n=1 bit-identity)."
            )

        monkeypatch.setattr(
            run_filtering, "_StridedObservationOperator", _tripwire, raising=True
        )

    cfg = compose_test_cfg(
        _overrides(
            "joint",
            num_windows,
            [
                f"filtering.assimilate_every_n_step={every_n}",
                "time.simulation_time=4.0",
                # The full-resolution frames the analyses skip live here.
                "run.ensemble_save_on_disk=true",
            ],
            model="pyudales",
        ),
        config_name="run_filtering",
    )
    run_filtering.run(cfg)

    dt_obs = float(cfg.time.output_frequency)
    frames_per_window = round(float(cfg.time.simulation_time) / dt_obs)
    # What an unstrided run would have done, halved by the stride.
    cycles_per_window = frames_per_window // every_n
    assert cycles_per_window * every_n == frames_per_window
    num_cycles = num_windows * cycles_per_window
    n_obs = len(cfg.obs.states) * len(cfg.obs.x_points)

    out_dir = pathlib.Path(cfg.paths.results_dir)
    windows_dir = out_dir / "windows"
    truth_times = np.asarray(
        xarray.open_dataset(out_dir / "true_state.nc")["time"].values, dtype=float
    )

    for w in range(num_windows):
        posterior_state = xarray.open_dataset(
            windows_dir / f"window_{w}_posterior_state.nc"
        )
        # One analyzed frame per CYCLE (not per output frame), at the analysis
        # times: the last frame of each of the window's every_n-frame blocks.
        assert posterior_state.sizes["time"] == cycles_per_window
        analysis_idx = [
            w * frames_per_window + (c + 1) * every_n - 1
            for c in range(cycles_per_window)
        ]
        np.testing.assert_allclose(
            np.asarray(posterior_state["time"].values, dtype=float),
            truth_times[analysis_idx],
        )
        if cycles_per_window > 1:
            # ... i.e. spaced every_n * output_frequency apart. Loose rtol: the
            # solver lands its output frames on its own timestep grid, so the
            # frame times are only nominally multiples of output_frequency.
            np.testing.assert_allclose(
                np.diff(np.asarray(posterior_state["time"].values, dtype=float)),
                every_n * dt_obs,
                rtol=0.05,
            )

        # The window's flat observation vector covers the ASSIMILATED frames
        # only, so the stride shortens it by exactly that factor.
        obs = xarray.open_dataset(windows_dir / f"window_{w}_obs.nc")
        assert obs.sizes["obs_index"] == cycles_per_window * n_obs
        assert obs.sizes["obs_index"] * every_n == frames_per_window * n_obs
        pred = xarray.open_dataset(windows_dir / f"window_{w}_pred_obs.nc")
        assert pred["pred_obs"].shape == (2, cycles_per_window * n_obs, 2)

    # The forward model still OUTPUTS every output_frequency step: each cycle's
    # on-disk segment holds the full every_n frames, only the last of which saw
    # a Kalman step.
    for cycle in range(num_cycles):
        for member in range(2):
            segment = xarray.open_dataset(
                out_dir / "_ensemble_states" / f"cycle_{cycle}" / f"state_{member}.nc"
            )
            assert segment.sizes["time"] == every_n

    diagnostics = read_yaml(out_dir / "cycle_diagnostics.yaml")
    assert [row["cycle"] for row in diagnostics] == list(range(num_cycles))

    truth_access = read_yaml(out_dir / "truth_access.yaml")
    # A cycle covers every_n truth frames and analyzes the last of them, which
    # is exactly what `end_of_cycle_indices` reconstructs from `n_per_cycle`.
    assert truth_access["n_per_cycle"] == every_n
    assert truth_access["cycle_seconds"] == every_n * dt_obs
    # The ESMDA keys keep their ESMDA (truth-block) meanings.
    assert truth_access["n_per_window"] == frames_per_window
    assert truth_access["n_total"] == num_windows * frames_per_window
    assert truth_access["num_cycles"] == num_cycles
    assert truth_access["sim_time"] == float(cfg.time.simulation_time)
    # Strided analyses no longer tile the window, so the mean fields taken over
    # them are a strided sample rather than a time average.
    assert truth_access["moment_sampling_is_sparse"] is (every_n > 1)
    if every_n == 1:
        assert truth_access["moment_sampling"] == run_filtering._MOMENT_SAMPLING

    configuration = read_yaml(out_dir / "run_info.yaml")["configuration"]
    assert configuration["assimilate_every_n_step"] == every_n
    assert configuration["cycles_per_window"] == cycles_per_window
    assert configuration["num_cycles"] == num_cycles
    assert configuration["seconds_per_cycle"] == every_n * dt_obs
