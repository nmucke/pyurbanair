"""Generate training/validation/test data for the neural-surrogate library.

Samples inflow parameters from the configured `params_sampler` (Hydra
`_target_` block), runs ALL train+val+test simulations in one parallel
ensemble call, partitions the resulting per-member NetCDFs into split
directories, and writes one figure + one animation per split.

Usage:

    python scripts/generate_training_data.py training_data=tiny
    python scripts/generate_training_data.py training_data=small model=pyudales
"""

from __future__ import annotations

import pathlib
import shutil
import sys
import time as _time

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import hydra
import jax
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from pyurbanair.animation import animate_state
from pyurbanair.config.hydra_helpers import (
    clean_outputs,
    resolve_output_dir,
    resolve_parameter_schema,
)
from pyurbanair.utils.run_utils import add_velocity_magnitude, extract_2d_slice


def _sample_params(
    params_sampler,
    *,
    num_time_points: int,
    simulation_time: float,
    seed: int,
) -> xr.Dataset:
    """Draw all parameter trajectories in one shot.

    `params_sampler.sample_prior` must return `(time, ensemble)` arrays
    per parameter — true for both `ParameterTimeSeries` subclasses and
    `pyurbanair.training_data.UniformParameterSampler`.
    """
    rng_key = jax.random.PRNGKey(seed)
    time_coords = np.linspace(0.0, float(simulation_time), num_time_points)
    sampled = params_sampler.sample_prior(time_coords, rng_key)
    return sampled.assign_coords(time=time_coords)


def _augment_params_for_backend(
    sampled: xr.Dataset,
    *,
    model_name: str,
    pressure_gradient_magnitude: float | None,
) -> xr.Dataset:
    """Add backend-specific extra params (e.g. pyudales pressure gradient).

    The ensemble model slices `params.isel(ensemble=i)` per member, so
    extra vars need an `ensemble` dim of the same size.
    """
    schema = resolve_parameter_schema(model_name)
    if "pressure_gradient_magnitude" in schema:
        if pressure_gradient_magnitude is None:
            raise ValueError(
                f"{model_name} requires pressure_gradient_magnitude; "
                "set params.true.pressure_gradient_magnitude."
            )
        n = sampled.sizes["ensemble"]
        sampled = sampled.assign(
            pressure_gradient_magnitude=(
                ("ensemble",),
                np.full(n, float(pressure_gradient_magnitude)),
            )
        )
    return sampled


def _copy_stl_if_present(cfg: DictConfig, output_dir: pathlib.Path) -> None:
    stl_path = OmegaConf.select(cfg, "model.forward_model.stl_path")
    if stl_path is None:
        return
    src = pathlib.Path(stl_path)
    if not src.is_absolute():
        src = pathlib.Path.cwd() / src
    if not src.exists():
        print(f"WARNING: STL path {src} does not exist; skipping copy.")
        return
    shutil.copy2(src, output_dir / src.name)
    print(f"Copied STL geometry to {output_dir / src.name}")


def _plot_sampled_params(
    sampled: xr.Dataset,
    split_offsets: list[tuple[str, int, int]],
    output_path: pathlib.Path,
) -> None:
    """Plot every sampled parameter trajectory, colored by split."""
    param_names = [n for n in sampled.data_vars if "time" in sampled[n].dims]
    n_params = len(param_names)
    fig, axes = plt.subplots(
        n_params, 1, figsize=(9, 3.0 * max(n_params, 1)), squeeze=False
    )
    time_vals = np.asarray(sampled["time"].values)
    split_colors = {"train": "C0", "val": "C1", "test": "C2"}
    for ax, name in zip(axes[:, 0], param_names):
        arr = np.asarray(sampled[name].transpose("ensemble", "time").values)
        for split, n, offset in split_offsets:
            color = split_colors.get(split, "C3")
            for i in range(n):
                label = split if i == 0 else None
                ax.plot(
                    time_vals,
                    arr[offset + i],
                    color=color,
                    alpha=0.6,
                    linewidth=1.0,
                    label=label,
                )
        ax.set_xlabel("time [s]")
        ax.set_ylabel(name)
        ax.set_title(f"{name}: every sampled trajectory")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def _plot_split_examples(
    examples: dict[str, xr.Dataset],
    output_path: pathlib.Path,
    z_level: int = 0,
) -> None:
    """Plot one mid-time velocity-magnitude slice per split, side-by-side."""
    splits = list(examples.keys())
    fig, axes = plt.subplots(1, len(splits), figsize=(5 * len(splits), 4.5), squeeze=False)
    for ax, split in zip(axes[0], splits):
        state = add_velocity_magnitude(examples[split])
        plot_var = "vel_magnitude" if "vel_magnitude" in state.data_vars else "u"
        mid_t = state.sizes["time"] // 2
        slice_2d = extract_2d_slice(state[plot_var].isel(time=mid_t), z_level=z_level)
        im = ax.imshow(slice_2d, origin="lower")
        ax.set_title(f"{split} — {plot_var} (t={mid_t})")
        fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def _interpolate_params_to_state_time(
    sampled: xr.Dataset,
    state_time: np.ndarray,
) -> xr.Dataset:
    """Linearly interpolate the sampler's control points onto `state_time`.

    The forward model already linearly interpolates between the sampled
    control points internally, but the saved metadata is stored at the
    sampler's coarse grid. Project it onto the state's output cadence so
    each state time step has a matching param value.

    Non-time-varying vars (e.g. pyudales `pressure_gradient_magnitude`,
    shape `(ensemble,)`) pass through unchanged.
    """
    xp = np.asarray(sampled["time"].values)
    new_vars: dict = {}
    for name in sampled.data_vars:
        da = sampled[name]
        if "time" in da.dims:
            arr = np.asarray(da.transpose("time", "ensemble").values)
            interp = np.stack(
                [np.interp(state_time, xp, arr[:, j]) for j in range(arr.shape[1])],
                axis=1,
            )
            new_vars[name] = (("time", "ensemble"), interp)
        else:
            new_vars[name] = da
    return xr.Dataset(
        new_vars,
        coords={"time": state_time, "ensemble": np.asarray(sampled["ensemble"].values)},
    )


def _partition_states_into_splits(
    raw_dir: pathlib.Path,
    output_dir: pathlib.Path,
    sampled: xr.Dataset,
    split_specs: list[tuple[str, int, int]],
    model_name: str,
) -> tuple[dict[str, xr.Dataset], xr.Dataset]:
    """Move ensemble outputs into the `state/{split}/` + `param/{split}/` layout.

    The ensemble writes `state_{i}.nc` for member `i` under `raw_dir`.
    For each member we:
      * re-open the raw state, collocate it to cell centers if the backend
        uses a staggered grid (pyudales), save it to
        `state/{split}/sample_{i:04d}.nc`, and delete the raw file;
      * save its interpolated parameter trajectory (one value per state
        time point) to `param/{split}/sample_{i:04d}.nc`.

    Returns the first state per split (for downstream viz) and the
    consolidated interpolated parameter dataset.
    """
    # pyudales solves on a C-grid (u@xm, v@ym, w@zm); the neural surrogate
    # stacks u/v/w channelwise, which assumes a common grid. Collocate to
    # cell centers (xt, yt, zt) before saving so the on-disk training data
    # is regular.
    regrid = None
    if model_name == "pyudales":
        from pyudales.utils.grid_utils import interpolate_grid as regrid

    first_path = raw_dir / "state_0.nc"
    if not first_path.exists():
        raise FileNotFoundError(
            f"Expected ensemble output {first_path} not found; "
            "did the ensemble run fail silently?"
        )
    with xr.open_dataset(first_path) as ds:
        state_time = np.asarray(ds["time"].values)

    interpolated = _interpolate_params_to_state_time(sampled, state_time)
    interpolated.to_netcdf(output_dir / "params.nc")
    print(f"Saved consolidated interpolated params -> {output_dir / 'params.nc'}")

    first_example: dict[str, xr.Dataset] = {}
    for split, n, offset in split_specs:
        state_split_dir = output_dir / "state" / split
        param_split_dir = output_dir / "param" / split
        state_split_dir.mkdir(parents=True, exist_ok=True)
        param_split_dir.mkdir(parents=True, exist_ok=True)
        for i in range(n):
            sample_idx = offset + i
            src = raw_dir / f"state_{sample_idx}.nc"
            if not src.exists():
                raise FileNotFoundError(
                    f"Expected ensemble output {src} not found; "
                    "did the ensemble run fail silently?"
                )
            with xr.open_dataset(src) as ds:
                state = ds.load()
            if regrid is not None:
                state = regrid(state)
            state_dst = state_split_dir / f"sample_{i:04d}.nc"
            state.to_netcdf(state_dst)
            src.unlink()

            member_params = interpolated.isel(ensemble=sample_idx).drop_vars("ensemble")
            param_dst = param_split_dir / f"sample_{i:04d}.nc"
            member_params.to_netcdf(param_dst)

            if i == 0:
                first_example[split] = state
            print(f"[{split}] sample {i + 1}/{n} -> {state_dst}")
    return first_example, interpolated


def run(cfg: DictConfig) -> None:
    model_name = cfg.model.name
    td = cfg.training_data

    num_train = int(td.num_train)
    num_val = int(td.num_val)
    num_test = int(td.num_test)
    n_total = num_train + num_val + num_test
    if n_total == 0:
        raise ValueError("training_data: num_train + num_val + num_test must be > 0")

    num_parallel_processes = int(td.num_parallel_processes)

    # Resolve output directory.
    if td.output_dir is not None:
        output_dir = pathlib.Path(td.output_dir)
    else:
        output_dir = resolve_output_dir(cfg, "training_data")
    output_dir.mkdir(parents=True, exist_ok=True)
    # state/{split}/ and param/{split}/ are created by the partition step.
    # Stale state files in `_raw_states` from a previous (possibly
    # failed) run would otherwise be picked up by the ensemble's
    # `get_member_state` and trigger a warm-start path — which on
    # pyudales can SIGILL if the leftover NetCDF is partial/corrupt.
    raw_states_dir = output_dir / "_raw_states"
    if raw_states_dir.exists():
        shutil.rmtree(raw_states_dir)
    raw_states_dir.mkdir()
    print(f"Writing training data to {output_dir}")

    OmegaConf.save(cfg, output_dir / "config.yaml", resolve=True)

    # --- Sample parameters ------------------------------------------------
    sampler_cfg = OmegaConf.to_container(td.params_sampler, resolve=True)
    sampler_cfg["ensemble_size"] = n_total
    params_sampler = hydra.utils.instantiate(sampler_cfg)

    sampled = _sample_params(
        params_sampler,
        num_time_points=int(td.num_time_points),
        simulation_time=float(td.simulation_time),
        seed=int(td.seed),
    )

    split_specs = [
        ("train", num_train, 0),
        ("val", num_val, num_train),
        ("test", num_test, num_train + num_val),
    ]

    # Up-front parameter trajectory visualization — independent of the
    # simulation, so you can sanity-check the prior box before paying
    # the forward-model cost.
    _plot_sampled_params(
        sampled=sampled,
        split_offsets=split_specs,
        output_path=output_dir / "sampled_params.png",
    )
    print(f"Saved parameter trajectories -> {output_dir / 'sampled_params.png'}")

    # Backend-specific params (e.g. pyudales pressure gradient) before
    # saving + handing off to the ensemble.
    pgm = None
    if "pressure_gradient_magnitude" in resolve_parameter_schema(model_name):
        pgm = float(cfg.params.true.pressure_gradient_magnitude)
    sampled = _augment_params_for_backend(
        sampled, model_name=model_name, pressure_gradient_magnitude=pgm
    )
    sampled.to_netcdf(output_dir / "sampled_params.nc")

    # --- Build template forward model + ensemble -------------------------
    forward_model = instantiate(
        cfg.model.forward_model,
        simulation_time=float(td.simulation_time),
        output_frequency=float(td.output_frequency),
        spinup_time=float(td.spinup_time),
        results_dir=None,
    )
    instantiate(cfg.model.prepare, forward_model=forward_model)
    clean_outputs(model_name=model_name, forward_model=forward_model)

    _copy_stl_if_present(cfg, output_dir)

    # The ensemble writes per-member NetCDFs to `raw_states_dir`. Override
    # `ensemble_size` to the total sample count so a single ensemble call
    # generates train + val + test in one parallel run; later we partition
    # the outputs into the split folders.
    ensemble_model = instantiate(
        cfg.model.ensemble_model,
        forward_model=forward_model,
        ensemble_size=n_total,
        num_parallel_processes=num_parallel_processes,
        results_dir=raw_states_dir,
    )
    # Parallel + on-disk does not support resample, so leave the default
    # "raise" policy: any member failure aborts the whole generation run.
    ensemble_model.configure_failure_policy(policy="raise")

    print(
        f"Running ensemble: {n_total} members ({num_train}/{num_val}/{num_test} "
        f"train/val/test) on {num_parallel_processes} parallel processes "
        f"(model={model_name})"
    )
    t0 = _time.time()
    ensemble_model.run_ensemble(params=sampled, sim_name="state")
    elapsed = _time.time() - t0
    print(
        f"Ensemble run finished in {elapsed:.1f}s "
        f"(~{elapsed / max(n_total, 1):.1f}s/member wall-clock-equivalent)"
    )

    # --- Partition outputs into split directories ------------------------
    first_example, interpolated = _partition_states_into_splits(
        raw_dir=raw_states_dir,
        output_dir=output_dir,
        sampled=sampled,
        split_specs=split_specs,
        model_name=model_name,
    )

    # Plot interpolated trajectories (what each sample_XXXX.nc actually
    # stores) — complements `sampled_params.png` (raw control points).
    _plot_sampled_params(
        sampled=interpolated,
        split_offsets=split_specs,
        output_path=output_dir / "params_interpolated.png",
    )
    print(f"Saved interpolated trajectories -> {output_dir / 'params_interpolated.png'}")
    # Best-effort cleanup of the staging dir.
    try:
        raw_states_dir.rmdir()
    except OSError:
        pass

    # --- Visualization ---------------------------------------------------
    if first_example:
        _plot_split_examples(first_example, output_dir / "split_examples.png")
        print(f"Saved figure -> {output_dir / 'split_examples.png'}")

        for split, state in first_example.items():
            anim_state = add_velocity_magnitude(state)
            anim_path = output_dir / f"{split}_animation.mp4"
            animate_state(state=anim_state, output_path=anim_path, z_level=0)
            print(f"Saved animation -> {anim_path}")

    print(f"Done. Training data root: {output_dir}")


@hydra.main(version_base=None, config_path="../conf", config_name="generate_training_data")
def main(cfg: DictConfig) -> None:
    run(cfg)


if __name__ == "__main__":
    main()
