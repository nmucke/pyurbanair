"""Rollout ESMDA with time-varying parameters across multiple windows.

Each assimilation window uses :class:`TimeVaryingParameterESMDA` to estimate
time-varying inflow parameters. Between windows the next window's prior parameter
ensemble is produced by a
:class:`pyurbanair.parameter_time_series.ParameterTimeSeries` instance selected
via the Hydra ``time_varying.method`` config: that object both draws the
initial-window prior and propagates the posterior into the next window's prior.

Window 0 starts cold (``state=None``) with spin-up enabled. Subsequent windows
warm-start from the previous window's final forecast state, with spin-up
disabled.

Two truth sources are supported, selected by ``run.ground_truth_dir``:

  * **Simulated** (default, ``run.ground_truth_dir=null``) — the truth is
    generated on the fly: truth parameters are drawn from a separate
    ``ParameterTimeSeries`` (a different correlation length than the assimilation
    prior, to avoid the inverse crime) and the truth forward model is run forward
    one window at a time, carrying its state between windows.
  * **Loaded** (``run.ground_truth_dir=<dir>``) — a pre-computed ground truth
    (``state.nc`` + ``params.nc``, as written by
    ``scripts/run_time_varying_forward_model.py``) is read from disk and chopped
    into consecutive windows. The number of windows is clamped so the rollout
    never exceeds the time length of the loaded truth. The truth forward model is
    never instantiated.

Everything downstream of the truth (the assimilation prior, the window loop,
persistence, plotting) is identical between the two modes.
"""

import pathlib
import sys
from typing import Any

import pyurbanair.quiet_jax  # noqa: F401  (suppress JAX CPU-fallback noise; must precede `import jax`)

import hydra
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import netCDF4
import numpy as np
import xarray
from hydra.utils import instantiate
from omegaconf import DictConfig
from tqdm import tqdm

from pyurbanair.parameter_time_series import ParameterTimeSeries
from pyurbanair.plotting import plot_state_init_and_terminal
from pyurbanair.config.hydra_helpers import (
    build_truth_ts_model,
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

# ---------------------------------------------------------------------------
# Truth generation (simulated mode)
# ---------------------------------------------------------------------------


def _generate_truth_params_all_windows(
    num_windows: int,
    num_time_points: int,
    sim_time: float,
    truth_ts_model: ParameterTimeSeries,
    rng_key: jax.random.PRNGKey,
) -> list[xarray.Dataset]:
    """Generate time-varying true parameters across all windows.

    A single ``sample_prior`` draw from ``truth_ts_model`` (with
    ``ensemble_size=1``) spans the full time horizon ``[0,
    num_windows * sim_time]`` on the union of all windows' time grids
    (sharing boundary points between adjacent windows), then is sliced
    into per-window datasets so that each slice matches the ESMDA
    window grid exactly and the profile is continuous across boundaries.

    Using a separate model instance for the truth (typically with a
    different correlation length than the assimilation prior) avoids
    the inverse crime of identical generative processes for truth and
    prior.

    Args:
        num_windows: Number of assimilation windows.
        num_time_points: Discrete parameter time points *per window*.
        sim_time: Duration of one window in seconds.
        truth_ts_model: Single-member ParameterTimeSeries instance used
            to draw the truth trajectory.
        rng_key: JAX random key.

    Returns:
        List of ``num_windows`` :class:`xarray.Dataset` objects, each
        with dims ``(time,)`` and time coordinates matching the ESMDA
        window grid for that window.
    """
    # Shared-boundary grid: window w spans indices
    # [w*(N_t-1) : w*(N_t-1) + N_t], so window w's last point == window
    # w+1's first point (both at t = (w+1)*sim_time).
    step = max(num_time_points - 1, 1)
    n_unique = num_windows * step + 1
    full_time = jnp.linspace(0, num_windows * sim_time, n_unique)

    full_ds = truth_ts_model.sample_prior(full_time, rng_key)
    # ensemble_size=1 by construction; squeeze the ensemble dim.
    full_ds = full_ds.isel(ensemble=0, drop=True)

    # Each window's simulation runs with its own clock starting at t=0,
    # so assign LOCAL time coords [0, sim_time] to each window's dataset.
    # Absolute times are reconstructed at save/plot time from the window index.
    local_time = np.asarray(jnp.linspace(0.0, sim_time, num_time_points))

    datasets: list[xarray.Dataset] = []
    for w in range(num_windows):
        start = w * step
        end = start + num_time_points
        data_vars = {
            name: ("time", np.asarray(full_ds[name].values[start:end]))
            for name in truth_ts_model.param_names
        }
        datasets.append(
            xarray.Dataset(data_vars=data_vars, coords={"time": local_time})
        )
    return datasets


# ---------------------------------------------------------------------------
# Ground-truth loading (loaded mode)
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
    exactly (the same convention as the simulated-truth mode). Only
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
# Truth providers
# ---------------------------------------------------------------------------
#
# A truth provider supplies, for each window, the truth parameters and truth
# state, and owns how the truth state is persisted. The two modes differ only
# here; the window loop and everything downstream are mode-agnostic.


class _SimulatedTruth:
    """Truth generated on the fly by running ``truth_model`` forward."""

    def __init__(
        self,
        cfg: DictConfig,
        num_windows: int,
        num_time_points: int,
        sim_time: float,
        truth_key: jax.random.PRNGKey,
    ) -> None:
        self.sim_time = sim_time
        self.num_windows = num_windows
        self.out_dir_label = (
            f"time_varying_rollout_esmda_{cfg.truth_model.name}_{cfg.assim_model.name}"
        )

        self.truth_model = instantiate(cfg.truth_model.forward_model)
        instantiate(cfg.truth_model.prepare, forward_model=self.truth_model)

        truth_ts_model = build_truth_ts_model(
            tv_cfg=cfg.time_varying,
            external_cfg=cfg.params.external,
            ensemble_size=1,
        )
        self.params_per_window = _generate_truth_params_all_windows(
            num_windows=num_windows,
            num_time_points=num_time_points,
            sim_time=sim_time,
            truth_ts_model=truth_ts_model,
            rng_key=truth_key,
        )

        self._true_state: xarray.Dataset | None = None
        self._paths: list[pathlib.Path] = []

    def state_for_window(self, w: int) -> xarray.Dataset:
        # Warm-start each window from the previous truth forecast.
        self._true_state = self.truth_model(
            params=self.params_per_window[w], state=self._true_state
        )
        if self._true_state is None:
            raise RuntimeError("Expected in-memory truth rollout state.")
        return self._true_state

    def persist_window(self, w: int, windows_dir: pathlib.Path) -> None:
        # Persist per window (LOCAL->ABSOLUTE time) so a later crash still leaves
        # usable per-window truth artifacts; stream-merged in ``finalize``.
        self._paths.append(
            _persist_window_dataset(
                self._true_state, w, self.sim_time, windows_dir, "true_state"
            )
        )

    def finalize(self, out_dir: pathlib.Path) -> xarray.Dataset:
        _stream_merge_along_time(self._paths, out_dir / "true_state.nc")
        return xarray.open_dataset(out_dir / "true_state.nc")


class _LoadedTruth:
    """Truth read from a pre-computed ``state.nc`` + ``params.nc`` directory."""

    def __init__(
        self,
        cfg: DictConfig,
        num_windows_req: int,
        num_time_points: int,
        sim_time: float,
    ) -> None:
        self.sim_time = sim_time
        self.out_dir_label = (
            f"time_varying_rollout_esmda_from_truth_"
            f"{cfg.truth_model.name}_{cfg.assim_model.name}"
        )
        output_frequency = float(cfg.time.output_frequency)

        state_path, params_path = _resolve_ground_truth_paths(
            pathlib.Path(cfg.run.ground_truth_dir)
        )
        self.gt_state = xarray.open_dataset(state_path)
        gt_params = xarray.open_dataset(params_path)
        print(
            f"Loaded ground-truth state  from {state_path}  "
            f"dims={dict(self.gt_state.sizes)}"
        )
        print(f"Loaded ground-truth params from {params_path}")

        # Clamp the rollout to the ground-truth time length. The assim model
        # produces ``sim_time / output_frequency`` state snapshots per window
        # (the backend trims to that count — see codebase guide §7), so the
        # loaded truth is chopped into consecutive chunks of that size and the
        # number of windows is capped so the rollout never runs past the end.
        n_total = int(self.gt_state.sizes["time"])
        self.n_per_window = max(int(round(sim_time / output_frequency)), 1)
        if self.n_per_window > n_total:
            raise ValueError(
                f"One window needs {self.n_per_window} state snapshots "
                f"(sim_time={sim_time} / output_frequency={output_frequency}) but "
                f"the ground truth only has {n_total}. Shorten time.simulation_time "
                "or regenerate a longer ground truth."
            )
        max_windows = n_total // self.n_per_window
        self.num_windows = min(num_windows_req, max_windows)
        if self.num_windows < 1:
            raise ValueError("Ground truth is too short for even a single window.")
        gt_total_time = float(np.asarray(self.gt_state["time"].values)[-1])
        if self.num_windows < num_windows_req:
            print(
                f"Requested {num_windows_req} windows "
                f"({num_windows_req * sim_time:.1f}s) but the ground truth spans "
                f"only {gt_total_time:.1f}s ({n_total} snapshots) — clamping to "
                f"{self.num_windows} window(s)."
            )
        print(
            f"Rollout: {self.num_windows} window(s) x {sim_time:.1f}s "
            f"({self.n_per_window} snapshots each) within {gt_total_time:.1f}s of "
            "truth."
        )

        self.params_per_window = _slice_truth_params_per_window(
            gt_params=gt_params,
            num_windows=self.num_windows,
            num_time_points=num_time_points,
            sim_time=sim_time,
        )
        self._state_per_window = [
            self.gt_state.isel(
                time=slice(w * self.n_per_window, (w + 1) * self.n_per_window)
            )
            for w in range(self.num_windows)
        ]

    def state_for_window(self, w: int) -> xarray.Dataset:
        return self._state_per_window[w]

    def persist_window(self, w: int, windows_dir: pathlib.Path) -> None:
        # Truth was loaded whole; saved once in ``finalize``.
        pass

    def finalize(self, out_dir: pathlib.Path) -> xarray.Dataset:
        true_state_all = self.gt_state.isel(
            time=slice(0, self.num_windows * self.n_per_window)
        )
        true_state_all.to_netcdf(out_dir / "true_state.nc")
        return true_state_all


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def _plot_time_varying_params_rollout(
    true_params_list: list[xarray.Dataset],
    posterior_params_list: list[xarray.Dataset],
    prior_params_list: list[xarray.Dataset],
    output_path: pathlib.Path,
) -> None:
    """Plot true vs prior vs posterior parameters across all windows.

    For each parameter the true profile is shown as a solid line, the
    per-window prior ensemble (cold-start for window 0, extrapolated for
    later windows) is shown in dashed green, and the posterior ensemble in
    orange.  Window boundaries are marked with vertical dashed lines.
    """
    param_names = [
        name
        for name in true_params_list[0].data_vars
        if "time" in true_params_list[0][name].dims
    ]
    n_params = len(param_names)
    fig, axes = plt.subplots(n_params, 1, figsize=(12, 4 * n_params), squeeze=False)

    for ax, name in zip(axes[:, 0], param_names):
        # True profile (concatenated across windows)
        true_times = np.concatenate(
            [np.asarray(ds.coords["time"].values) for ds in true_params_list]
        )
        true_vals = np.concatenate(
            [np.asarray(ds[name].values) for ds in true_params_list]
        )
        ax.plot(true_times, true_vals, color="C0", linewidth=2, label="True")

        # Prior per window: individual members + ensemble mean.  Window 0
        # is the cold-start prior; windows 1+ are extrapolated from the
        # previous posterior.
        for w, prior_ds in enumerate(prior_params_list):
            t = np.asarray(prior_ds.coords["time"].values)
            members = np.asarray(prior_ds[name].transpose("time", "ensemble").values)
            ax.plot(t, members, color="C2", linewidth=0.8, linestyle="--", alpha=0.50)
            ens_mean = members.mean(axis=1)
            mean_label = "Prior mean" if w == 0 else None
            ax.plot(
                t,
                ens_mean,
                color="C2",
                linewidth=2,
                linestyle="--",
                label=mean_label,
            )

        # Posterior per window: individual members + ensemble mean
        for w, post_ds in enumerate(posterior_params_list):
            t = np.asarray(post_ds.coords["time"].values)
            members = np.asarray(post_ds[name].transpose("time", "ensemble").values)
            ax.plot(t, members, color="C1", linewidth=0.8, alpha=0.50)
            ens_mean = members.mean(axis=1)
            mean_label = "Posterior mean" if w == 0 else None
            ax.plot(t, ens_mean, color="C1", linewidth=4, label=mean_label)

        # Window boundaries
        for w in range(1, len(true_params_list)):
            boundary = float(true_params_list[w].coords["time"].values[0])
            ax.axvline(boundary, color="gray", linestyle=":", linewidth=1)

        ax.set_xlabel("Time [s]")
        ax.set_ylabel(name)
        ax.legend()
        ax.set_title(name)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Saved time-varying parameter rollout plot to {output_path}")


# ---------------------------------------------------------------------------
# Per-window persistence
# ---------------------------------------------------------------------------
#
# Persist heavy per-window state files to disk inside the loop and stream-merge
# them at the end so peak memory tracks one window's data instead of all of
# them, and a final-step crash still leaves usable per-window artifacts.


def _persist_window_dataset(
    ds: xarray.Dataset,
    w: int,
    sim_time: float,
    windows_dir: pathlib.Path,
    kind: str,
) -> pathlib.Path:
    """Shift LOCAL window time coords to ABSOLUTE and write a per-window file."""
    ds_abs = ds.assign_coords(time=ds["time"].values + w * sim_time)
    path = windows_dir / f"window_{w:03d}_{kind}.nc"
    ds_abs.to_netcdf(path)
    return path


def _stream_merge_along_time(
    paths: list[pathlib.Path],
    out_path: pathlib.Path,
) -> None:
    """Concatenate per-window NetCDFs along ``time`` without a full in-memory load.

    Used for the big ensemble-state files (``true_state``, ``esmda_state``).
    Peak memory ≈ one window's data, vs. ``xarray.concat`` which materializes
    the full concatenation plus the target write buffer.

    Time slices are appended verbatim — matching the existing ``join="override"``
    semantics, duplicate boundary times between adjacent windows are kept.
    """
    with netCDF4.Dataset(paths[0]) as template:
        time_size_first = template.dimensions["time"].size
        with netCDF4.Dataset(out_path, "w") as dst:
            dst.setncatts({k: template.getncattr(k) for k in template.ncattrs()})
            for name, dim in template.dimensions.items():
                dst.createDimension(name, None if name == "time" else len(dim))
            for name, var in template.variables.items():
                fill = getattr(var, "_FillValue", None)
                new_var = dst.createVariable(
                    name, var.datatype, var.dimensions, fill_value=fill
                )
                new_var.setncatts(
                    {k: var.getncattr(k) for k in var.ncattrs() if k != "_FillValue"}
                )
                if "time" in var.dimensions:
                    time_axis = var.dimensions.index("time")
                    idx = [slice(None)] * var.ndim
                    idx[time_axis] = slice(0, time_size_first)
                    new_var[tuple(idx)] = var[...]
                else:
                    new_var[...] = var[...]

    offset = time_size_first
    for p in paths[1:]:
        with netCDF4.Dataset(p) as src, netCDF4.Dataset(out_path, "a") as dst:
            n_time = src.dimensions["time"].size
            for name, var in src.variables.items():
                if "time" not in var.dimensions:
                    continue
                time_axis = var.dimensions.index("time")
                idx = [slice(None)] * var.ndim
                idx[time_axis] = slice(offset, offset + n_time)
                dst[name][tuple(idx)] = var[...]
            offset += n_time


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def run(cfg: DictConfig) -> None:
    # ---- Config -----------------------------------------------------------
    num_windows_req = int(cfg.esmda.num_assimilation_windows)
    num_time_points = int(cfg.time_varying.num_time_points)
    sim_time = float(cfg.time.simulation_time)

    rng_key = make_rng_key(cfg.esmda.seed)

    # ---- Assimilation prior time-series model ----------------------------
    ts_model = instantiate(cfg.time_varying.prior_model)

    # ---- Truth source (simulated on the fly, or loaded from disk) ---------
    if cfg.run.ground_truth_dir is None:
        rng_key, truth_key = jax.random.split(rng_key)
        truth: _SimulatedTruth | _LoadedTruth = _SimulatedTruth(
            cfg, num_windows_req, num_time_points, sim_time, truth_key
        )
    else:
        truth = _LoadedTruth(cfg, num_windows_req, num_time_points, sim_time)
    num_windows = truth.num_windows
    true_params_per_window = truth.params_per_window

    # ---- Assimilation forward model ---------------------------------------
    assim_results_dir = (
        pathlib.Path(cfg.run.results_dir) if cfg.run.results_dir is not None else None
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
    # Truth obs use the truth backend's grid mapping (truth_model.solver_name);
    # in loaded mode the truth forward model itself is never instantiated.
    truth_obs_op = create_observation_operator(cfg.obs, cfg.truth_model.solver_name)
    ensemble_model = instantiate(
        cfg.assim_model.ensemble_model,
        forward_model=assim_model,
    )
    configure_failure_policy(ensemble_model, cfg.ensemble.failure)
    assim_obs_op = create_observation_operator(cfg.obs, cfg.assim_model.solver_name)

    # ---- Output paths -----------------------------------------------------
    # Resolved BEFORE the window loop so each window can persist to
    # ``windows_dir`` as soon as it finishes.
    out_dir = resolve_output_dir(cfg, truth.out_dir_label)
    out_dir.mkdir(parents=True, exist_ok=True)
    windows_dir = out_dir / "windows"
    windows_dir.mkdir(parents=True, exist_ok=True)

    # ---- Storage ----------------------------------------------------------
    # ``prior_params_list[w]`` is the prior ensemble for window ``w``:
    # cold-start from ``ts_model.sample_prior`` for window 0, and
    # ``ts_model.extrapolate(posterior_{w-1})`` for windows 1+.
    #
    # Params lists are small (a few floats × ensemble × time points × windows)
    # so we keep them in memory for the final concat-and-save + plotting. The
    # heavy assim-state datasets are written to ``windows_dir`` inside the loop
    # and stream-merged at the end; the truth provider owns truth-state
    # persistence.
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

        # --- Truth state for this window -----------------------------------
        true_state = truth.state_for_window(w)

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
        # continuity that the extrapolation established at each window
        # boundary. Window 0's prior t=0 is a free parameter (just a prior
        # draw), so we let ESMDA fit it.
        esmda.pin_initial_time_point = w > 0

        # --- Determine initial state for ESMDA -----------------------------
        if esmda_final_state is not None and "time" in esmda_final_state.dims:
            state_input = esmda_final_state.isel(time=-1)
        else:
            state_input = esmda_final_state  # None for window 0

        # --- Run ESMDA -----------------------------------------------------
        # ``_analysis`` returns either ``(params_history, final_state)`` (in
        # memory) or just ``params_history`` (when the ensemble forward model
        # has save_on_disk=True). The rollout flow needs the in-memory final
        # state to seed the next window, so reject the on-disk contract loudly
        # instead of letting it crash later as an opaque AttributeError on a
        # string return value.
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
        # Writes the per-window NetCDFs immediately so a crash in a later
        # window — or in the final stream-merge — leaves usable artifacts in
        # ``windows_dir``. Params (small) are also written for symmetry; the
        # in-memory params lists are still used by plotting and by the
        # post-loop concat for the merged param NetCDFs.
        truth.persist_window(w, windows_dir)
        esmda_state_paths.append(
            _persist_window_dataset(
                esmda_final_state, w, sim_time, windows_dir, "esmda_state"
            )
        )
        _persist_window_dataset(true_params_w, w, sim_time, windows_dir, "true_params")
        _persist_window_dataset(
            posterior_params, w, sim_time, windows_dir, "posterior_params"
        )
        _persist_window_dataset(
            prior_params_list[-1], w, sim_time, windows_dir, "prior_params"
        )

        # --- Build next window's prior ------------------------------------
        # Methods that fit-and-roll-forward (gp_linear_trend, ar1, OU) want
        # prediction times that start at the posterior's last training
        # point so the rollout is continuous; we extrapolate over
        # [sim_time, 2*sim_time] and relabel back to local [0, sim_time]
        # for the next window's forward model.  AR(2) relaxation is
        # translation-invariant in local time, so this works there too.
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
    # Each window's datasets carry LOCAL time coords [0, sim_time]; shift to
    # absolute times before concatenating so the global time axis is monotonic.
    def _shift_to_abs(ds: xarray.Dataset, w: int) -> xarray.Dataset:
        return ds.assign_coords(time=ds["time"].values + w * sim_time)

    # Heavy assim state: the per-window NetCDFs were already written inside the
    # loop with absolute time coords. Stream-merge them rather than re-loading
    # all windows into memory. Truth state is finalized by its provider.
    _stream_merge_along_time(esmda_state_paths, out_dir / "esmda_state.nc")
    true_state_all = truth.finalize(out_dir)
    esmda_state_all = xarray.open_dataset(out_dir / "esmda_state.nc")

    # Small params files: concat in memory and write.
    true_params_abs = [
        _shift_to_abs(ds, w) for w, ds in enumerate(true_params_per_window)
    ]
    posterior_params_abs = [
        _shift_to_abs(ds, w) for w, ds in enumerate(posterior_params_list)
    ]
    prior_params_abs = [_shift_to_abs(ds, w) for w, ds in enumerate(prior_params_list)]

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
    # re-raising. tqdm wraps ``sys.stderr``; on shutdown its buffered ``\r``
    # progress line can be the only thing that ever reaches the SLURM .err
    # file, hiding the actual traceback. Bypassing the wrapper guarantees
    # the traceback lands on disk even if tqdm swallows it.
    import sys as _sys
    import traceback as _traceback

    try:
        main()
    except BaseException:
        _traceback.print_exc(file=_sys.__stderr__)
        _sys.__stderr__.flush()
        raise
