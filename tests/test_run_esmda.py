"""End-to-end smoke tests for the unified scripts/run_esmda.py.

Covers the modes the single script replaces (the old
run_{parameter,state_and_parameter,rollout,time_varying_parameter,
time_varying_parameters_rollout}_esmda.py family) plus the joint
state+time-varying-parameter mode, a cross-model case and a disk-loaded-truth
case. Everything runs under the tiny smoke config (conftest `_SMOKE_OVERRIDES`) with the global
(unlocalized) update — the default correlation localization is degenerate at
this 2-member ensemble size and has its own test.

The smoother group options are static | state_and_parameter | dynamic |
state_and_dynamic (the old `parameter`/`time_varying` names mapped to
`static`/`dynamic`).
"""

import pathlib

import numpy as np
import pytest


def _overrides(
    truth_model, assim_model, smoother, prior, num_windows, localization=None
):
    truth = "static_truth" if prior == "static" else "dynamic_truth"
    # Localization selection. Default: the global (unlocalized) update — the
    # default correlation localization is degenerate at this 2-member ensemble
    # size. ``localization`` may instead be a list of overrides selecting a
    # config-group option (e.g. ["esmda/localization=distance", ...]); distance
    # localization is purely geometric, so it is meaningful even at 2 members.
    localization = (
        localization if localization is not None else ["esmda.localization=null"]
    )
    ov = [
        f"model@truth_model={truth_model}",
        f"model@assim_model={assim_model}",
        f"esmda/smoother={smoother}",
        f"params@prior_params={prior}",
        f"params@truth_params={truth}",
        *localization,
        "ensemble.ensemble_size=2",
        "ensemble.num_parallel_processes=2",
        "esmda.num_steps=1",
        f"esmda.num_assimilation_windows={num_windows}",
        "run.skip_viz=true",
        # Simulate the truth inline (the default config points truth_dir at a
        # disk path that does not exist in the test environment).
        "run.truth_dir=null",
        # The conftest smoke overrides pin a tiny [0,20]^2 domain but do not
        # supply matching sensor coordinates, so the case's full-domain obs points
        # fall outside the test grid. Place the assimilation sensors in the open
        # N-S lanes of the tiny [0,20]^2 domain. Validation sensors are only used
        # by the plots (skipped here via run.skip_viz=true).
        "obs.x_points=[2.5,2.5,18.0,18.0]",
        "obs.y_points=[5.0,15.0,5.0,15.0]",
        "obs.z_points=[3.0,3.0,3.0,3.0]",
        "obs.interval_seconds=3.0",
    ]
    if prior == "dynamic":
        # Keep all the time grids small. `time.seconds_per_knot` is the single
        # knot-spacing source: the prior/truth samplers space their knots by it,
        # and run_esmda.py derives the dynamic smoothers' `num_time_points` from
        # the resulting knot count, so the flattened knot grids stay in agreement.
        ov += ["time.seconds_per_knot=1.5"]
    if truth_model == "pylbm":
        ov.append("truth_model.forward_model.cuda=false")
    if assim_model == "pylbm":
        ov.append("assim_model.forward_model.cuda=false")
    # conf/model/pyudales.yaml's default nnudge_meters=16 m is tuned for a
    # full-size domain; on the smoke grid (zsize=10 m, ktot=4) it would
    # leave no nudged levels and raise. Scale it down to the test domain so a
    # couple of bottom cells stay un-nudged while the rest are nudged.
    if truth_model == "pyudales":
        ov.append("truth_model.forward_model.nudging_config.nnudge_meters=4.0")
    if assim_model == "pyudales":
        ov.append("assim_model.forward_model.nudging_config.nnudge_meters=4.0")
    return ov


@pytest.mark.parametrize(  # type: ignore[misc]
    "truth_model,assim_model,smoother,prior,num_windows",
    [
        # The parameter/state modes the single script unifies (pylbm/pylbm).
        pytest.param("pylbm", "pylbm", "static", "static", 1, id="parameter"),
        pytest.param(
            "pylbm", "pylbm", "state_and_parameter", "static", 1, id="state_and_param"
        ),
        pytest.param(
            "pylbm", "pylbm", "state_and_parameter", "static", 2, id="rollout"
        ),
        pytest.param("pylbm", "pylbm", "dynamic", "dynamic", 1, id="tv_param"),
        pytest.param("pylbm", "pylbm", "dynamic", "dynamic", 2, id="tv_rollout"),
        # Joint state + time-varying-parameter mode (single window + rollout).
        pytest.param(
            "pylbm", "pylbm", "state_and_dynamic", "dynamic", 1, id="state_and_tv_param"
        ),
        pytest.param(
            "pylbm",
            "pylbm",
            "state_and_dynamic",
            "dynamic",
            2,
            id="state_and_tv_rollout",
        ),
        # Extra backend coverage on the cheapest mode.
        pytest.param("pyudales", "pyudales", "static", "static", 1, id="udales"),
        pytest.param("pylbm", "pyudales", "static", "static", 1, id="cross"),
    ],
)
def test_run_esmda(
    truth_model: str,
    assim_model: str,
    smoother: str,
    prior: str,
    num_windows: int,
    compose_test_cfg,
) -> None:
    from scripts.run_esmda import run

    overrides = _overrides(truth_model, assim_model, smoother, prior, num_windows)
    run(compose_test_cfg(overrides, config_name="run_esmda"))


@pytest.mark.parametrize(  # type: ignore[misc]
    "smoother,prior",
    [
        pytest.param("static", "static", id="static_model_error"),
        pytest.param("dynamic", "dynamic", id="dynamic_model_error"),
    ],
)
def test_run_esmda_with_model_error_parameters(
    smoother: str, prior: str, compose_test_cfg
) -> None:
    """ESMDA estimates the model-error knobs when `params_to_estimate` includes
    them: the static path adds two scalar knobs; the dynamic path additionally
    rides the constant-in-time statics alongside the time-varying inflow
    (docs/esmda_model_error_parameters.md). The augmented posterior must carry
    both new parameters."""
    import xarray

    from scripts.run_esmda import run

    overrides = _overrides("pyudales", "pyudales", smoother, prior, 1)
    overrides.append(
        "params_to_estimate=[inflow_angle,velocity_magnitude,"
        "sgs_constant,vertical_inflow_exponent]"
    )
    cfg = compose_test_cfg(overrides, config_name="run_esmda")
    run(cfg)

    posterior = xarray.open_dataset(
        pathlib.Path(cfg.paths.results_dir) / "posterior_params.nc"
    )
    assert "sgs_constant" in posterior.data_vars
    assert "vertical_inflow_exponent" in posterior.data_vars


@pytest.mark.parametrize(  # type: ignore[misc]
    "smoother,prior,num_windows",
    [
        # Distance-based localization on the STATE (params stay global), selected
        # via the `esmda/localization` config group. Geometric -> fine at N_e=2.
        pytest.param("state_and_parameter", "static", 1, id="state_static_distance"),
        pytest.param("state_and_dynamic", "dynamic", 1, id="state_tv_distance"),
    ],
)
def test_run_esmda_distance_localization(
    smoother: str, prior: str, num_windows: int, compose_test_cfg
) -> None:
    """The state-bearing smoothers run with distance-based state localization."""
    from scripts.run_esmda import run

    overrides = _overrides(
        "pylbm",
        "pylbm",
        smoother,
        prior,
        num_windows,
        localization=[
            "esmda/localization=distance",
            "esmda.localization.localization_radius=10.0",
        ],
    )
    run(compose_test_cfg(overrides, config_name="run_esmda"))


@pytest.mark.parametrize(  # type: ignore[misc]
    "smoother,prior,reduction_overrides",
    [
        # Online reduced SVD/KL state update (esmda/state_reduction group),
        # state-bearing smoothers only. IC-source basis on the static case;
        # the dynamic case additionally exercises the window-snapshot basis
        # and the optional post-loop full-trajectory smoothing step.
        pytest.param(
            "state_and_parameter",
            "static",
            ["esmda/state_reduction=svd"],
            id="state_static_svd",
        ),
        pytest.param(
            "state_and_dynamic",
            "dynamic",
            [
                "esmda/state_reduction=svd",
                "esmda.state_reduction.basis_source=window_snapshots",
                "esmda.final_time_smoothing=true",
            ],
            id="state_tv_svd_snapshots_final_smoothing",
        ),
    ],
)
def test_run_esmda_state_reduction(
    smoother: str, prior: str, reduction_overrides: list, compose_test_cfg
) -> None:
    """The state-bearing smoothers run with the reduced SVD state update."""
    from scripts.run_esmda import run

    overrides = _overrides("pylbm", "pylbm", smoother, prior, 1) + reduction_overrides
    run(compose_test_cfg(overrides, config_name="run_esmda"))


def test_run_esmda_loads_ground_truth_from_disk(
    tmp_path: pathlib.Path, compose_test_cfg
) -> None:
    """run_forward_model.py writes a time-varying ground-truth artifact; run_esmda
    consumes it via run.truth_dir instead of simulating the truth."""
    from scripts.run_forward_model import run as run_forward

    gt_dir = tmp_path / "ground_truth"
    run_forward(
        compose_test_cfg(
            [
                "model=pylbm",
                "model.forward_model.cuda=false",
                # A dynamic (time-varying) params sampler is what makes
                # run_forward_model write the ground-truth state.nc/params.nc
                # artifact (it keys off a `time` coord on the sampled params, not
                # a `run.time_varying` flag — that flag no longer exists in the
                # flattened config).
                "params=dynamic",
                "time.seconds_per_knot=1.5",
                "run.rollout_steps=1",
                "run.skip_viz=true",
                f"run.results_dir={gt_dir}",
                f"paths.base_results_dir={gt_dir}",
                # The tiny test domain ([0,20]^2) needs in-bounds sensors, as the
                # smoke overrides supply none (see _overrides).
                "obs.x_points=[2.5,2.5,18.0,18.0]",
                "obs.y_points=[5.0,15.0,5.0,15.0]",
                "obs.z_points=[3.0,3.0,3.0,3.0]",
            ]
        )
    )
    # run_forward_model writes the truth artifact to
    # <base>/forward_model/<model>..._time_varying/{state,params}.nc. Select that
    # folder explicitly: run.results_dir also leaves a scratch state.nc directly
    # under gt_dir, so a bare rglob("state.nc") is order-dependent and may pick
    # the scratch file (which has no sibling params.nc).
    truth_dir = next(
        p.parent for p in gt_dir.rglob("state.nc") if (p.parent / "params.nc").exists()
    )

    from scripts.run_esmda import run

    # The code reads run.truth_dir (not run.ground_truth_dir). _overrides appends
    # `run.truth_dir=null` to force inline truth; override it here with the real
    # on-disk truth so this run loads from disk instead of simulating.
    overrides = _overrides("pylbm", "pylbm", "dynamic", "dynamic", 1)
    overrides.append(f"run.truth_dir={truth_dir}")
    run(compose_test_cfg(overrides, config_name="run_esmda"))


# ---------------------------------------------------------------------------
# Unit test for the per-window ensemble reduction (the finalize bottleneck).
# ---------------------------------------------------------------------------


def _write_ensemble_window(path, n_ens=6, seed=0):
    """Write a tiny ensemble window state file (leading ``ensemble`` axis).

    Mirrors a real window_*_posterior_state.nc: u/v/w plus a stored
    ``vel_magnitude`` == |vel|, a ``time`` coordinate and a global attr.
    """
    import netCDF4

    nt, nz, ny, nx = 2, 3, 4, 4
    rng = np.random.default_rng(seed)
    arrs = {n: rng.standard_normal((n_ens, nt, nz, ny, nx)) for n in ("u", "v", "w")}
    arrs["vel_magnitude"] = np.sqrt(arrs["u"] ** 2 + arrs["v"] ** 2 + arrs["w"] ** 2)
    dtypes = {"u": "f4", "v": "f4", "w": "f8", "vel_magnitude": "f8"}
    with netCDF4.Dataset(path, "w") as ds:
        for d, s in [
            ("ensemble", n_ens),
            ("time", nt),
            ("z", nz),
            ("y", ny),
            ("x", nx),
        ]:
            ds.createDimension(d, s)
        tv = ds.createVariable("time", "f8", ("time",))
        tv[:] = np.arange(nt)
        ds.setncattr("solver", "test")
        for name, a in arrs.items():
            var = ds.createVariable(
                name, dtypes[name], ("ensemble", "time", "z", "y", "x")
            )
            var[:] = a.astype(dtypes[name])
    return arrs


def test_streaming_state_summary_drops_redundant_velmag(tmp_path):
    """The window reduction returns u/v/w means + vel_mean/vel_std and, when a
    stored ``vel_magnitude`` is present, deliberately DROPS its (redundant with
    ``vel_mean``) ensemble mean -- the largest variable is never read."""
    from scripts.run_esmda import _streaming_state_summary

    path = tmp_path / "window_0_posterior_state.nc"
    arrs = _write_ensemble_window(path)
    out = _streaming_state_summary(path)

    # vel_magnitude mean is intentionally absent; vel_mean/vel_std take its place.
    assert "vel_magnitude" not in out.data_vars
    assert {"u", "v", "w", "vel_mean", "vel_std"} <= set(out.data_vars)
    # Coordinate + global attrs pass through unchanged.
    assert np.array_equal(out["time"].values, np.arange(2))
    assert out.attrs.get("solver") == "test"

    for name in ("u", "v", "w"):
        ref = arrs[name].mean(axis=0).astype(out[name].dtype)
        assert np.allclose(out[name].values, ref, rtol=1e-5, atol=1e-6)

    vm = arrs["vel_magnitude"]
    vmean = vm.mean(axis=0)
    vstd = np.sqrt(np.maximum((vm**2).mean(axis=0) - vmean**2, 0.0))
    assert np.allclose(out["vel_mean"].values, vmean.astype("f4"), rtol=1e-5, atol=1e-5)
    assert np.allclose(out["vel_std"].values, vstd.astype("f4"), rtol=1e-5, atol=1e-5)
    # The dropped quantity (mean vel_magnitude) equals vel_mean -- consumers that
    # used it are unaffected.
    assert np.allclose(out["vel_mean"].values, vmean.astype("f4"), rtol=1e-5, atol=1e-5)


def test_streaming_state_summary_threading_matches_serial(tmp_path, monkeypatch):
    """The multi-worker reduction agrees with the single-worker path."""
    import scripts.run_esmda as run_esmda

    path = tmp_path / "window_0_posterior_state.nc"
    _write_ensemble_window(path, n_ens=7)  # odd count -> uneven round-robin split

    monkeypatch.setattr(run_esmda, "_SUMMARY_WORKERS", 1)
    serial = run_esmda._streaming_state_summary(path)
    monkeypatch.setattr(run_esmda, "_SUMMARY_WORKERS", 3)
    threaded = run_esmda._streaming_state_summary(path)

    assert set(serial.data_vars) == set(threaded.data_vars)
    for name in serial.data_vars:
        assert np.allclose(
            serial[name].values, threaded[name].values, rtol=1e-6, atol=1e-8
        ), name
