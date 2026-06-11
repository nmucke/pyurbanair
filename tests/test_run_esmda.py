"""End-to-end smoke tests for the unified scripts/run_esmda.py.

Covers the modes the single script replaces (the old
run_{parameter,state_and_parameter,rollout,time_varying_parameter,
time_varying_parameters_rollout}_esmda.py family) plus the joint
state+time-varying-parameter mode, a cross-model case and a disk-loaded-truth
case. Everything runs under the tiny `+size=test` overlay with the global
(unlocalized) update — the default correlation localization is degenerate at
this 2-member ensemble size and has its own test.

The smoother group options are static | state_and_parameter | dynamic |
state_and_dynamic (the old `parameter`/`time_varying` names mapped to
`static`/`dynamic`).
"""

import pathlib

import pytest


def _overrides(truth_model, assim_model, smoother, prior, num_windows, localization=None):
    truth = "static_truth" if prior == "static" else "dynamic_truth"
    # Localization selection. Default: the global (unlocalized) update — the
    # default correlation localization is degenerate at this 2-member ensemble
    # size. ``localization`` may instead be a list of overrides selecting a
    # config-group option (e.g. ["esmda/localization=distance", ...]); distance
    # localization is purely geometric, so it is meaningful even at 2 members.
    localization = localization if localization is not None else ["esmda.localization=null"]
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
        # The `+size=test` overlay shrinks the domain to [0,20]^2 but (unlike the
        # other size overlays) does not supply matching sensor coordinates, so
        # the case's full-domain obs points fall outside the test grid. Place the
        # assimilation sensors in the open N-S lanes of the tiny domain, mirroring
        # conf/size/tiny.yaml. Validation sensors are only used by the plots
        # (skipped here via run.skip_viz=true).
        "obs.x_points=[2.5,2.5,18.0,18.0]",
        "obs.y_points=[5.0,15.0,5.0,15.0]",
        "obs.z_points=[3.0,3.0,3.0,3.0]",
        "obs.interval_seconds=3.0",
    ]
    if prior == "dynamic":
        # Keep all three time grids at the same small N. The state_and_dynamic
        # smoother flattens `num_time_points` (= params.time_coords.num, the
        # `params` mount from config.yaml) knots and isel(time=t_idx) into the
        # sampled prior, so the `params`, prior and truth grids must agree.
        ov += [
            "params.time_coords.num=3",
            "prior_params.time_coords.num=3",
            "truth_params.time_coords.num=3",
        ]
    if truth_model == "pylbm":
        ov.append("truth_model.forward_model.cuda=false")
    if assim_model == "pylbm":
        ov.append("assim_model.forward_model.cuda=false")
    # conf/model/pyudales.yaml's default nnudge_meters=16 m is tuned for a
    # full-size domain; on the `+size=test` grid (zsize=10 m, ktot=4) it would
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
            "pylbm", "pylbm", "state_and_dynamic", "dynamic", 2, id="state_and_tv_rollout"
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
        "pylbm", "pylbm", smoother, prior, num_windows,
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
                "params.time_coords.num=3",
                "run.rollout_steps=1",
                "run.skip_viz=true",
                f"run.results_dir={gt_dir}",
                f"paths.base_results_dir={gt_dir}",
                # The tiny test domain ([0,20]^2) needs in-bounds sensors, as the
                # `+size=test` overlay supplies none (see _overrides).
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
        p.parent
        for p in gt_dir.rglob("state.nc")
        if (p.parent / "params.nc").exists()
    )

    from scripts.run_esmda import run

    # The code reads run.truth_dir (not run.ground_truth_dir). _overrides appends
    # `run.truth_dir=null` to force inline truth; override it here with the real
    # on-disk truth so this run loads from disk instead of simulating.
    overrides = _overrides("pylbm", "pylbm", "dynamic", "dynamic", 1)
    overrides.append(f"run.truth_dir={truth_dir}")
    run(compose_test_cfg(overrides, config_name="run_esmda"))
