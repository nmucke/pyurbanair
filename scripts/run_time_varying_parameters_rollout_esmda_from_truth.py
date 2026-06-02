"""Rollout ESMDA with time-varying parameters, against a LOADED ground truth.

This is a variant of ``run_time_varying_parameters_rollout_esmda.py``. Instead
of simulating the truth on the fly (running ``truth_model`` forward and drawing
the truth parameters from a ``ParameterTimeSeries``), it loads a pre-computed
ground truth from a directory:

  * ``state.nc``  — the simulated truth state (time + grid dims), and
  * ``params.nc`` — the time-varying inflow parameter profile that produced it,

both as written by ``scripts/run_time_varying_forward_model.py`` (i.e. the
``ground_truth`` folder produced by ``job_scripts/snellius/ground_truth.slurm``).
Point the script at that directory with ``run.ground_truth_dir=<dir>``.

The loaded truth is one continuous simulation; the rollout chops it into
consecutive assimilation windows of ``time.simulation_time`` each. The number of
windows is clamped so the rollout never exceeds the time length of the loaded
ground truth.

Window 0 starts cold (``state=None``) with spin-up enabled. Subsequent windows
warm-start from the previous window's final forecast state, with spin-up
disabled. Everything downstream of the truth (the assimilation prior, the
window loop, persistence, plotting) is unchanged from the sibling script.
"""

import pathlib
import sys
from typing import Any

import pyurbanair.quiet_jax  # noqa: F401  (suppress JAX CPU-fallback noise; must precede `import jax`)

import hydra
import jax
import jax.numpy as jnp
import numpy as np
import xarray
from hydra.utils import instantiate
from omegaconf import DictConfig
from tqdm import tqdm

from pyurbanair.plotting import plot_state_init_and_terminal
from pyurbanair.config.hydra_helpers import (
    configure_failure_policy,
    create_C_D,
    create_observation_operator,
    create_observation_points,
    make_rng_key,
    resolve_output_dir,
)
from pyurbanair.utils.animation_utils import animate_rollout_state

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

# Reuse the per-window persistence / stream-merge / plotting helpers from the
# sibling rollout script rather than duplicating them. ``python scripts/<x>.py``
# puts the scripts dir on ``sys.path`` so the sibling import resolves; be
# explicit in case the script is imported some other way.
_SCRIPTS_DIR = pathlib.Path(__file__).resolve().parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))
from run_time_varying_parameters_rollout_esmda import (  # noqa: E402
    _persist_window_dataset,
    _plot_time_varying_params_rollout,
    _stream_merge_along_time,
)


# ---------------------------------------------------------------------------
# Ground-truth loading
# ---------------------------------------------------------------------------


def _resolve_ground_truth_paths(
    gt_dir: pathlib.Path,
) -> tuple[pathlib.Path, pathlib.Path]:
    """Locate ``state.nc`` and ``params.nc`` under ``gt_dir``.

    Accepts either the directory that directly contains the two files, or a
    parent of it (the forward-model script nests its outputs under a
    ``<model>_time_varying/`` subfolder), in which case the shallowest match
    is used.
    """
    gt_dir = pathlib.Path(gt_dir)
    if not gt_dir.exists():
        raise FileNotFoundError(f"Ground-truth directory does not exist: {gt_dir}")

    direct_state = gt_dir / "state.nc"
    direct_params = gt_dir / "params.nc"
    if direct_state.exists() and direct_params.exists():
        return direct_state, direct_params

    state_hits = sorted(gt_dir.glob("**/state.nc"), key=lambda p: len(p.parts))
    params_hits = sorted(gt_dir.glob("**/params.nc"), key=lambda p: len(p.parts))
    if not state_hits or not params_hits:
        raise FileNotFoundError(
            f"Could not find both 'state.nc' and 'params.nc' under {gt_dir}. "
            "Point run.ground_truth_dir at a directory produced by "
            "scripts/run_time_varying_forward_model.py."
        )
    return state_hits[0], params_hits[0]


def _slice_truth_params_per_window(
    gt_params: xarray.Dataset,
    num_windows: int,
    num_time_points: int,
    sim_time: float,
) -> list[xarray.Dataset]:
    """Interpolate the loaded truth parameter profile onto each window's grid.

    The loaded profile is a function of absolute time over the full ground
    truth. For window ``w`` we sample it at absolute times
    ``w*sim_time + linspace(0, sim_time, num_time_points)`` and relabel to the
    LOCAL window grid ``[0, sim_time]`` so it matches the ESMDA window grid
    exactly (the same convention as the simulated-truth sibling script). Only
    time-varying parameters are carried.
    """
    tv_names = [n for n in gt_params.data_vars if "time" in gt_params[n].dims]
    if not tv_names:
        raise ValueError(
            "Loaded params.nc has no time-varying parameters (no variable with "
            "a 'time' dim) — nothing to compare the estimate against."
        )
    local_time = np.asarray(jnp.linspace(0.0, sim_time, num_time_points))
    datasets: list[xarray.Dataset] = []
    for w in range(num_windows):
        abs_grid = w * sim_time + local_time
        window = gt_params[tv_names].interp(time=abs_grid)
        window = window.assign_coords(time=local_time)
        datasets.append(window)
    return datasets


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def run(cfg: DictConfig) -> None:
    # ---- Config -----------------------------------------------------------
    num_windows_req = int(cfg.esmda.num_assimilation_windows)
    num_time_points = int(cfg.time_varying.num_time_points)
    sim_time = float(cfg.time.simulation_time)
    output_frequency = float(cfg.time.output_frequency)

    if cfg.run.ground_truth_dir is None:
        raise ValueError(
            "run.ground_truth_dir is not set. Pass the directory holding the "
            "pre-simulated ground truth, e.g. "
            "run.ground_truth_dir=/projects/.../ground_truth"
        )

    rng_key = make_rng_key(cfg.esmda.seed)

    # ---- Load ground truth (state + params) -------------------------------
    state_path, params_path = _resolve_ground_truth_paths(
        pathlib.Path(cfg.run.ground_truth_dir)
    )
    gt_state = xarray.open_dataset(state_path)
    gt_params = xarray.open_dataset(params_path)
    print(f"Loaded ground-truth state  from {state_path}  dims={dict(gt_state.sizes)}")
    print(f"Loaded ground-truth params from {params_path}")

    # ---- Clamp the rollout to the ground-truth time length ----------------
    # The assim model produces ``sim_time / output_frequency`` state snapshots
    # per window (the backend trims to that count — see codebase guide §7), so
    # the loaded truth is chopped into consecutive chunks of that size. The
    # number of windows is capped so the rollout never runs past the end of the
    # loaded truth; the temporal observation operator only aligns truth and
    # assimilation by per-window snapshot count, so the chunk size must match.
    n_total = int(gt_state.sizes["time"])
    n_per_window = max(int(round(sim_time / output_frequency)), 1)
    if n_per_window > n_total:
        raise ValueError(
            f"One window needs {n_per_window} state snapshots "
            f"(sim_time={sim_time} / output_frequency={output_frequency}) but the "
            f"ground truth only has {n_total}. Shorten time.simulation_time or "
            "regenerate a longer ground truth."
        )
    max_windows = n_total // n_per_window
    num_windows = min(num_windows_req, max_windows)
    if num_windows < 1:
        raise ValueError("Ground truth is too short for even a single window.")
    gt_total_time = float(np.asarray(gt_state["time"].values)[-1])
    if num_windows < num_windows_req:
        print(
            f"Requested {num_windows_req} windows ({num_windows_req * sim_time:.1f}s) "
            f"but the ground truth spans only {gt_total_time:.1f}s "
            f"({n_total} snapshots) — clamping to {num_windows} window(s)."
        )
    print(
        f"Rollout: {num_windows} window(s) x {sim_time:.1f}s "
        f"({n_per_window} snapshots each) within {gt_total_time:.1f}s of truth."
    )

    # ---- Truth parameters per window (interpolated from the loaded profile) -
    true_params_per_window = _slice_truth_params_per_window(
        gt_params=gt_params,
        num_windows=num_windows,
        num_time_points=num_time_points,
        sim_time=sim_time,
    )
    # Truth state per window: consecutive snapshot chunks of the loaded state.
    true_state_per_window = [
        gt_state.isel(time=slice(w * n_per_window, (w + 1) * n_per_window))
        for w in range(num_windows)
    ]

    # ---- Parameter time-series prior model (assimilation side only) -------
    ts_model = instantiate(cfg.time_varying.prior_model)

    # ---- Assimilation forward model ---------------------------------------
    assim_results_dir = (
        pathlib.Path(cfg.run.results_dir)
        if cfg.run.results_dir is not None
        else None
    )
    assim_model = instantiate(
        cfg.assim_model.forward_model,
        results_dir=assim_results_dir,
    )
    instantiate(cfg.assim_model.prepare, forward_model=assim_model)

    # ---- Initial parameter ensemble for window 0 -------------------------
    initial_time_coords = jnp.linspace(0.0, sim_time, num_time_points)
    rng_key, prior_key = jax.random.split(rng_key)
    params_ensemble = ts_model.sample_prior(initial_time_coords, prior_key)

    # ---- Observation setup ------------------------------------------------
    # Truth obs use the GT backend's grid mapping (truth_model.solver_name);
    # the truth forward model itself is never instantiated — we read its state.
    truth_obs_op = create_observation_operator(cfg.obs, cfg.truth_model.solver_name)
    ensemble_model = instantiate(
        cfg.assim_model.ensemble_model,
        forward_model=assim_model,
    )
    configure_failure_policy(ensemble_model, cfg.ensemble.failure)
    assim_obs_op = create_observation_operator(cfg.obs, cfg.assim_model.solver_name)

    # ---- Output paths -----------------------------------------------------
    out_dir = resolve_output_dir(
        cfg,
        f"time_varying_rollout_esmda_from_truth_"
        f"{cfg.truth_model.name}_{cfg.assim_model.name}",
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    windows_dir = out_dir / "windows"
    windows_dir.mkdir(parents=True, exist_ok=True)

    # ---- Storage ----------------------------------------------------------
    prior_params_list: list[xarray.Dataset] = []
    posterior_params_list: list[xarray.Dataset] = []
    esmda_state_paths: list[pathlib.Path] = []

    # ---- State tracking for handoff between windows -----------------------
    esmda_final_state: xarray.Dataset | None = None
    C_D: jnp.ndarray | None = None
    esmda: Any | None = None

    # ---- Window loop ------------------------------------------------------
    for w in tqdm(range(num_windows), desc="Assimilation windows"):
        true_params_w = true_params_per_window[w]
        # Record the prior used for this window before ESMDA updates it.
        prior_params_list.append(params_ensemble)

        # --- Truth state (loaded slice; no forward run) --------------------
        true_state = true_state_per_window[w]

        # --- Noisy observations --------------------------------------------
        true_obs = jnp.asarray(truth_obs_op(true_state))
        if C_D is None:
            C_D = create_C_D(true_obs.shape[0], cfg.esmda.obs_error_std)
        rng_key, subkey = jax.random.split(rng_key)
        noisy_obs = true_obs + jnp.sqrt(C_D) @ jax.random.normal(subkey, true_obs.shape)

        # --- Window time coords (local; each window's sim runs 0..sim_time) -
        local_time_coords = jnp.linspace(0.0, sim_time, num_time_points)

        # --- Construct ESMDA (local time coords are the same every window) -
        if esmda is None:
            esmda = instantiate(
                cfg.esmda.smoother,
                observation_operator=assim_obs_op,
                forward_model=ensemble_model,
                C_D=C_D,
                time_coords=local_time_coords,
                rng_key=rng_key,
            )
        # Pin t=0 from window 1 onward so the Kalman update preserves the
        # continuity the extrapolation established at each window boundary.
        esmda.pin_initial_time_point = w > 0

        # --- Determine initial state for ESMDA -----------------------------
        if esmda_final_state is not None and "time" in esmda_final_state.dims:
            state_input = esmda_final_state.isel(time=-1)
        else:
            state_input = esmda_final_state  # None for window 0

        # --- Run ESMDA -----------------------------------------------------
        output = esmda(
            state=state_input,
            params=params_ensemble,
            observations=noisy_obs,
            return_params_history=True,
            return_state_history=False,
        )
        if not isinstance(output, tuple):
            raise RuntimeError(
                "Rollout ESMDA requires an in-memory ensemble forward model "
                "so the final window state can seed the next window. Got a "
                "single return value, which means the forward model is "
                "configured with save_on_disk=True."
            )
        params_history, esmda_final_state = output

        # Extract final posterior (last ESMDA step)
        posterior_params = params_history.isel(esmda_step=-1)
        posterior_params_list.append(posterior_params)

        # --- Persist this window's heavy outputs to disk -------------------
        # The truth state was loaded whole and is saved once after the loop;
        # only the freshly-produced assim state is stream-merged per window.
        esmda_state_paths.append(
            _persist_window_dataset(
                esmda_final_state, w, sim_time, windows_dir, "esmda_state"
            )
        )
        _persist_window_dataset(
            true_params_w, w, sim_time, windows_dir, "true_params"
        )
        _persist_window_dataset(
            posterior_params, w, sim_time, windows_dir, "posterior_params"
        )
        _persist_window_dataset(
            prior_params_list[-1], w, sim_time, windows_dir, "prior_params"
        )

        # --- Build next window's prior ------------------------------------
        if w < num_windows - 1:
            prediction_times = jnp.linspace(sim_time, 2.0 * sim_time, num_time_points)
            rng_key, extrap_key = jax.random.split(rng_key)
            extrapolated = ts_model.extrapolate(
                posterior_params,
                prediction_times=prediction_times,
                rng_key=extrap_key,
            )
            extrapolated = extrapolated.assign_coords(
                time=np.asarray(local_time_coords)
            )
            params_ensemble = extrapolated

    # ---- Save outputs -----------------------------------------------------
    def _shift_to_abs(ds: xarray.Dataset, w: int) -> xarray.Dataset:
        return ds.assign_coords(time=ds["time"].values + w * sim_time)

    # Heavy assim state: stream-merge the per-window NetCDFs.
    _stream_merge_along_time(esmda_state_paths, out_dir / "esmda_state.nc")

    # Truth state: the used slice of the loaded ground truth (already on an
    # absolute time axis), written directly.
    true_state_all = gt_state.isel(time=slice(0, num_windows * n_per_window))
    true_state_all.to_netcdf(out_dir / "true_state.nc")
    esmda_state_all = xarray.open_dataset(out_dir / "esmda_state.nc")

    # Small params files: concat in memory and write.
    true_params_abs = [
        _shift_to_abs(ds, w) for w, ds in enumerate(true_params_per_window)
    ]
    posterior_params_abs = [
        _shift_to_abs(ds, w) for w, ds in enumerate(posterior_params_list)
    ]
    prior_params_abs = [
        _shift_to_abs(ds, w) for w, ds in enumerate(prior_params_list)
    ]

    true_params_all = xarray.concat(true_params_abs, dim="time", join="override")
    posterior_params_all = xarray.concat(
        posterior_params_abs, dim="time", join="override"
    )
    prior_params_all = xarray.concat(prior_params_abs, dim="time", join="override")

    true_params_all.to_netcdf(out_dir / "true_params.nc")
    posterior_params_all.to_netcdf(out_dir / "posterior_params.nc")
    prior_params_all.to_netcdf(out_dir / "prior_params.nc")

    # ---- Plotting ---------------------------------------------------------
    if not cfg.run.skip_viz:
        _plot_time_varying_params_rollout(
            true_params_list=true_params_abs,
            posterior_params_list=posterior_params_abs,
            prior_params_list=prior_params_abs,
            output_path=out_dir / "time_varying_parameters_rollout.png",
        )
        obs_x, obs_y, _ = create_observation_points(cfg.obs)
        esmda_mean_all = (
            esmda_state_all.mean(dim="ensemble")
            if "ensemble" in esmda_state_all.dims
            else esmda_state_all
        )
        plot_state_init_and_terminal(
            true_state=true_state_all,
            estimated_state=esmda_mean_all,
            output_path=out_dir / "state_init_and_terminal.png",
            obs_x=obs_x,
            obs_y=obs_y,
            z_level=0,
        )
        animate_rollout_state(
            true_state=true_state_all,
            esmda_state=esmda_state_all,
            output_path=out_dir / "rollout_animation.mp4",
            z_level=0,
        )

    print(f"Saved outputs in {pathlib.Path(out_dir)}")


@hydra.main(
    version_base=None,
    config_path="../conf",
    config_name="time_varying_rollout_esmda",
)
def main(cfg: DictConfig) -> None:
    run(cfg)


if __name__ == "__main__":
    # Print tracebacks to ``sys.__stderr__`` (the original fd) before
    # re-raising. tqdm wraps ``sys.stderr``; on shutdown its buffered progress
    # line can be the only thing reaching the SLURM .err file, hiding the
    # actual traceback. Bypassing the wrapper guarantees it lands on disk.
    import sys as _sys
    import traceback as _traceback

    try:
        main()
    except BaseException:
        _traceback.print_exc(file=_sys.__stderr__)
        _sys.__stderr__.flush()
        raise
