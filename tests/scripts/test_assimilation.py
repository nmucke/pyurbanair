"""Assimilation runs (run_smoother/filtering/hybrid.py) and their metrics and
figures (compute_metrics.py, visualize_assimilation.py)."""

from __future__ import annotations

import pathlib
from typing import Any

import numpy as np
import pytest
import xarray
import yaml
from omegaconf import OmegaConf

from tests.conftest import compose, load_script, surrogate

# Per method: the overrides that make a valid run on top of test/assimilation.
METHODS: dict[str, list[str]] = {
    "smoother": [],
    "filtering": ["params@prior_params=static"],
    "hybrid": [],
}
# Per method: the observation-space stages metrics.yaml scores. A hybrid's
# smoother stage has no analysis, so it has O_N only (no desroziers).
STAGES: dict[str, dict[str, set[str]]] = {
    "smoother": {"smoother": {"forecast_rmse", "data_mismatch"}},
    "filtering": {"filter": {"forecast_rmse", "innovation_chi2_diag"}},
    "hybrid": {"smoother": {"data_mismatch"}, "filter": {"forecast_rmse"}},
}
SURROGATE = [
    "model@truth_model=neural_surrogate_tiny",
    "model@assim_model=neural_surrogate_tiny",
]


def _run(method: str, cfg: Any) -> pathlib.Path:
    """Run the method, then compute its metrics and draw its figures."""
    load_script(f"scripts/run_{method}.py").run(cfg)
    run_dir = pathlib.Path(cfg.paths.results_dir) / method
    load_script("scripts/compute_metrics.py").run(run_dir)
    load_script("scripts/visualize_assimilation.py").run(run_dir)
    return run_dir


def _check_outputs(run_dir: pathlib.Path, num_windows: int = 2) -> None:
    for w in range(num_windows):
        posterior = xarray.load_dataset(
            run_dir / "windows" / f"window_{w}_posterior_params.nc"
        )
        assert posterior.sizes["ensemble"] == 2
        assert (run_dir / "windows" / f"window_{w}_posterior_state.nc").exists()
    metrics = yaml.safe_load((run_dir / "metrics.yaml").read_text())
    assert {"parameters", "state", "sensors"} <= set(metrics)
    for block in ("spread_skill", "climatology"):
        assert set(metrics[block]) == {"assimilation", "validation"}, block
    for name in ("parameter_evolution.png", "mean_slices.png", "tke_slices.png"):
        assert (run_dir / "figures" / name).exists(), name


@pytest.mark.parametrize("method", METHODS)  # type: ignore[misc]
def test_method(
    method: str, tmp_path: pathlib.Path, session_root: pathlib.Path, trained: Any
) -> None:
    cfg = compose(
        "assimilation",
        "+test=assimilation",
        *SURROGATE,
        *surrogate(session_root),
        *METHODS[method],
        root=tmp_path,
    )
    run_dir = _run(method, cfg)
    _check_outputs(run_dir)
    metrics = yaml.safe_load((run_dir / "metrics.yaml").read_text())
    for stage, keys in STAGES[method].items():
        assert keys <= set(metrics["observation"][stage]), stage
    assert set(metrics["desroziers"]) == {
        stage for stage, keys in STAGES[method].items() if "forecast_rmse" in keys
    }
    assert {"ratio", "spread"} <= set(metrics["spread_skill"]["validation"])
    assert "rmse_skill_vs_climatology" in metrics["climatology"]["validation"]


# Per method: the flag that saves a reference ensemble and the block it adds to
# sensor_statistics.
SAVED: dict[str, tuple[str, str]] = {
    "smoother": ("assimilation.save_prior_state=true", "prior"),
    "filtering": ("assimilation.save_forecast_history=true", "forecast"),
    "hybrid": ("assimilation.save_forecast_history=true", "forecast"),
}


@pytest.mark.parametrize("method", METHODS)  # type: ignore[misc]
def test_saved_reference_states(
    method: str, tmp_path: pathlib.Path, session_root: pathlib.Path, trained: Any
) -> None:
    """Saving the prior/forecast states adds their scores, not other changes."""
    flag, reference = SAVED[method]
    run_dirs = {}
    runs: dict[str, list[str]] = {"default": [], "saved": [flag]}
    for name, extra in runs.items():
        cfg = compose(
            "assimilation",
            "+test=assimilation",
            *SURROGATE,
            *surrogate(session_root),
            *METHODS[method],
            *extra,
            root=tmp_path / name,
        )
        load_script(f"scripts/run_{method}.py").run(cfg)
        run_dirs[name] = pathlib.Path(cfg.paths.results_dir) / method
        load_script("scripts/compute_metrics.py").run(run_dirs[name])

    default, saved = (sorted((d / "windows").iterdir()) for d in run_dirs.values())
    assert {f.name for f in default} < {f.name for f in saved}
    for path in default:
        xarray.testing.assert_identical(
            xarray.load_dataset(path),
            xarray.load_dataset(run_dirs["saved"] / "windows" / path.name),
        )
    metrics = yaml.safe_load((run_dirs["saved"] / "metrics.yaml").read_text())
    for name in ("assimilation", "validation"):
        stats = metrics["sensor_statistics"][name]
        assert reference in stats
        assert all(
            f"crps_reduction_vs_{reference}" in entry
            for entry in stats["posterior"].values()
        )


def test_truth_from_a_forward_run(
    tmp_path: pathlib.Path, session_root: pathlib.Path, trained: Any
) -> None:
    """A forward run's state.nc + params.nc serve as the smoother's truth."""
    forward = compose(
        "forward",
        "+test=forward",
        "model=neural_surrogate_tiny",
        *surrogate(session_root),
        "params=dynamic_sine",
        "forward.rollout_steps=1",  # covers both assimilation windows
        root=tmp_path,
    )
    load_script("scripts/run_forward.py").run(forward)
    cfg = compose(
        "assimilation",
        "+test=assimilation",
        *SURROGATE,
        *surrogate(session_root),
        f"assimilation.truth_dir={forward.paths.results_dir}",
        root=tmp_path,
    )
    _check_outputs(_run("smoother", cfg))


@pytest.mark.integration  # type: ignore[misc]
def test_smoother_on_udales(tmp_path: pathlib.Path) -> None:
    cfg = compose("assimilation", "+test=assimilation", root=tmp_path)
    _check_outputs(_run("smoother", cfg))


# Per method: what run_info.yaml names, which decides the observation stages.
RUN_INFO: dict[str, dict[str, Any]] = {
    "smoother": {"smoother": "ESMDA"},
    "filtering": {"filter": "EnsembleKalmanFilter"},
    "hybrid": {"smoother": "ESMDA", "filter": "EnsembleKalmanFilter"},
}


def _window_run_dir(method: str, root: pathlib.Path) -> pathlib.Path:
    """A run dir whose window files are the truth's own frames, stamped in time
    the way run_<method>.py stamps them.

    The model's output frames sit in (0, sim_time] of their window (a truth
    read from a truth_dir is cut the same way), the truth on (0, horizon].
    Every quantity grows with time, so a frame put on the wrong time scores.
    """
    helpers = load_script("scripts/utils/helper_functions.py")
    cfg = compose(
        "assimilation",
        "+test=assimilation",
        *SURROGATE,
        "time.simulation_time=3.0",
        "time.output_frequency=1.0",
        "time.seconds_per_knot=1.5",
        "assimilation.num_windows=2",
        "assimilation.assimilate_every_n_step=1",
        "assimilation.truth_dir=null",
        root=root,
    )
    num_windows, sim_time, dt = 2, 3.0, 1.0
    frames = int(sim_time / dt)
    run_dir = pathlib.Path(cfg.paths.results_dir) / method
    windows = run_dir / "windows"
    windows.mkdir(parents=True)
    OmegaConf.save(cfg, run_dir / "config.yaml", resolve=True)
    info = {**RUN_INFO[method], "cycles_per_window": frames}
    (run_dir / "run_info.yaml").write_text(yaml.safe_dump(info))

    # Truth: u = t, v = 2t, w = -t everywhere; one parameter linear in time.
    times = dt * (np.arange(num_windows * frames) + 1)
    grid = {"z": np.arange(6.0), "y": np.arange(20.0), "x": np.arange(20.0)}
    ones = np.ones((len(times), 6, 20, 20))
    t = times[:, None, None, None]
    dims = ("time", "z", "y", "x")
    truth = xarray.Dataset(
        {"u": (dims, t * ones), "v": (dims, 2 * t * ones), "w": (dims, -t * ones)},
        coords={"time": times, **grid},
    )
    truth.to_netcdf(run_dir / "true_state.nc")
    knots = np.arange(0.0, num_windows * sim_time + 1e-9, 1.5)
    xarray.Dataset(
        {"velocity_magnitude": (("ensemble", "time"), [7.0 + knots])},
        coords={"time": knots},
    ).to_netcdf(run_dir / "true_params.nc")

    def ensemble(ds: xarray.Dataset) -> xarray.Dataset:
        return xarray.concat([ds, ds], dim="ensemble").transpose("ensemble", ...)

    def obs(path: pathlib.Path, n_obs: int) -> None:
        pred = np.ones((2, n_obs, 2))
        xarray.Dataset(
            {
                "obs": ("observation", np.ones(n_obs)),
                "obs_clean": ("observation", np.ones(n_obs)),
                "obs_std": ("observation", np.ones(n_obs)),
                "pred_obs": (("esmda_step", "observation", "ensemble"), pred),
            }
        ).to_netcdf(path)

    n_sensors = len(cfg.obs.x_points)
    for w in range(num_windows):
        window = truth.isel(time=slice(w * frames, (w + 1) * frames))
        cycle_times = window.time.values  # cycle_observations: one per frame
        if method == "filtering":
            # run_filtering.py: the analysis per cycle, on the cycles' truth times.
            cycles = ensemble(window).rename(time="cycle")
            params = ensemble(
                xarray.Dataset(
                    {"velocity_magnitude": ("cycle", 7.0 + cycle_times)},
                )
            )
            for name in ("prior", "posterior"):
                helpers.cycles_to_time(params, cycle_times).to_netcdf(
                    windows / f"window_{w}_{name}_params.nc"
                )
        else:
            # run_smoother.py / run_hybrid.py: the parameters on the sampler's
            # window-local knots, including the window start.
            local = np.array([0.0, 1.5, 3.0])
            params = ensemble(
                xarray.Dataset(
                    {"velocity_magnitude": ("time", 7.0 + w * sim_time + local)},
                    coords={"time": local},
                )
            )
            for name in ("prior", "posterior"):
                params.to_netcdf(windows / f"window_{w}_{name}_params.nc")
        if method == "smoother":
            # The model's own window-local output (0, sim_time].
            state = ensemble(window.assign_coords(time=window.time - w * sim_time))
            state.to_netcdf(windows / f"window_{w}_posterior_state.nc")
            state.to_netcdf(windows / f"window_{w}_prior_state.nc")
            obs(windows / f"window_{w}_obs.nc", n_sensors)
        else:
            cycles = ensemble(window).rename(time="cycle")
            helpers.cycles_to_time(cycles, cycle_times).to_netcdf(
                windows / f"window_{w}_posterior_state.nc"
            )
            # run_filtering.py / run_hybrid.py: every forecast frame.
            ensemble(window).assign_coords(
                time=w * sim_time + dt * (np.arange(frames) + 1)
            ).to_netcdf(windows / f"window_{w}_forecast_state.nc")
            filter_obs = "filter_obs" if method == "hybrid" else "obs"
            obs(windows / f"window_{w}_{filter_obs}.nc", frames * n_sensors)
            if method == "hybrid":
                obs(windows / f"window_{w}_obs.nc", frames * n_sensors)
    return run_dir


@pytest.mark.parametrize("method", METHODS)  # type: ignore[misc]
def test_window_files_on_the_truth_time(
    method: str, tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An ensemble that is the truth scores zero error, in metrics and figures."""
    run_dir = _window_run_dir(method, tmp_path)
    load_script("scripts/compute_metrics.py").run(run_dir)
    metrics = yaml.safe_load((run_dir / "metrics.yaml").read_text())
    for name, block in metrics["sensors"].items():
        rmse = block["velocity_vector_rmse"]
        assert rmse["max"] == pytest.approx(0.0, abs=1e-9), (name, rmse)
    rmse = metrics["parameters"]["velocity_magnitude"]["rmse"]
    assert rmse["max"] == pytest.approx(0.0, abs=1e-6), rmse

    # The sensor series visualize_assimilation.py draws.
    visualize = load_script("scripts/visualize_assimilation.py")
    drawn = []
    monkeypatch.setattr(
        visualize, "plot_sensor_timeseries", lambda **kw: drawn.append(kw)
    )
    visualize.run(run_dir)
    for kw in drawn:
        truth, members = kw["true_sensor"], kw["ensemble_sensor"]
        on_ensemble_time = truth.interp(time=members.time)
        np.testing.assert_allclose(members, on_ensemble_time.broadcast_like(members))


def test_a_truth_from_before_the_time_axis_change_is_refused(
    tmp_path: pathlib.Path,
) -> None:
    """A truth_dir state.nc with a frame at t=0 is on the old axis: it raises."""
    helpers = load_script("scripts/utils/helper_functions.py")
    xarray.Dataset(coords={"time": [0.0, 1.0, 2.0]}).to_netcdf(tmp_path / "state.nc")
    cfg = OmegaConf.create(
        {
            "assimilation": {
                "truth_dir": str(tmp_path),
                "truth_start_time": None,
                "num_windows": 1,
            },
            "time": {"simulation_time": 2.0},
        }
    )
    with pytest.raises(ValueError, match="frame at t=0"):
        helpers.open_truth(cfg, tmp_path)


def test_case_stl_path_is_from_the_repo_root(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    helpers = load_script("scripts/utils/helper_functions.py")
    monkeypatch.chdir(tmp_path)
    stl = "geometries/xie_and_castro/xie_castro_2008_STL.stl"
    cfg = OmegaConf.create({"geometry": {"stl_path": stl}})

    assert helpers.case_stl_path(cfg).is_file()

    cfg.geometry.stl_path = "geometries/xie_and_castro/missing.stl"
    with pytest.raises(FileNotFoundError, match="geometry.stl_path"):
        helpers.case_stl_path(cfg)
