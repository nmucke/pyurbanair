"""Finish a generate_training_data.py run that failed during partitioning.

The simulation ensemble completed and `_raw_states/state_{i}.nc` exist, but
`_partition_states_into_splits` crashed (pandas InvalidIndexError) because
some members carry corrupted z coordinates: a stale fielddump got merged in,
padding the dataset to extra z-levels that are all-NaN with garbage `zt`/`zm`
values. The real data lives on the leading levels of the canonical grid.

This script redoes ONLY the post-processing:
  * fixes oversized members by slicing z back to the canonical grid
    (validated against a clean reference member, NaN-checked);
  * collocates to cell centers (pyudales staggered grid) and writes
    `state/{split}/sample_XXXX.nc` + `param/{split}/sample_XXXX.nc`;
  * regenerates the figures and animations;
  * deletes each raw file only after its outputs are written, so the script
    is resumable (already-processed members are skipped).

Usage:

    python scripts/finish_training_data_processing.py [output_dir]

`output_dir` defaults to training_data/pyudales_medium.
"""

from __future__ import annotations

import pathlib
import sys

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import numpy as np
import xarray as xr
from omegaconf import OmegaConf

from generate_training_data import (
    _attach_blanking,
    _plot_sampled_params,
    _plot_split_examples,
)
from pyudales.utils.grid_utils import interpolate_grid
from pyurbanair.animation import animate_state
from pyurbanair.utils.run_utils import add_velocity_magnitude

Z_DIMS = ("zt", "zm")


def _canonical_z_coords(raw_dir: pathlib.Path, n_members: int) -> dict[str, np.ndarray]:
    """Find the clean z grid: the smallest z size seen across members."""
    best: dict[str, np.ndarray] | None = None
    for i in range(n_members):
        path = raw_dir / f"state_{i}.nc"
        if not path.exists():
            continue
        with xr.open_dataset(path) as ds:
            coords = {d: np.asarray(ds[d].values) for d in Z_DIMS}
        if best is None or coords["zt"].size < best["zt"].size:
            best = coords
    if best is None:
        raise FileNotFoundError(f"No unprocessed state files left under {raw_dir}")
    for d in Z_DIMS:
        v = best[d]
        if len(np.unique(v)) != v.size or not np.all(np.diff(v) > 0):
            raise ValueError(f"Reference {d} grid is itself non-monotone: {v}")
    return best


def _fix_z_grid(ds: xr.Dataset, canonical: dict[str, np.ndarray], name: str) -> xr.Dataset:
    """Slice corrupted members back to the canonical z grid and sanity-check."""
    nz = canonical["zt"].size
    if ds.sizes["zt"] != nz or ds.sizes["zm"] != nz:
        ds = ds.isel(zt=slice(0, nz), zm=slice(0, nz))
    for d in Z_DIMS:
        if not np.allclose(np.asarray(ds[d].values), canonical[d]):
            raise ValueError(
                f"{name}: {d} does not match the canonical grid after slicing; "
                f"got {np.asarray(ds[d].values)}"
            )
    for var in ds.data_vars:
        if not np.isfinite(ds[var].values).all():
            raise ValueError(f"{name}: {var} has non-finite values on the canonical grid")
    return ds


def run(output_dir: pathlib.Path) -> None:
    cfg = OmegaConf.load(output_dir / "config.yaml")
    td = cfg.training_data
    num_train, num_val, num_test = int(td.num_train), int(td.num_val), int(td.num_test)
    split_specs = [
        ("train", num_train, 0),
        ("val", num_val, num_train),
        ("test", num_test, num_train + num_val),
    ]
    n_total = num_train + num_val + num_test
    raw_dir = output_dir / "_raw_states"

    solid_c_path = (
        pathlib.Path(cfg.model.forward_model.temp_dir)
        / "experiment"
        / str(cfg.model.forward_model.experiment_name)
        / "solid_c.txt"
    )
    if not solid_c_path.exists():
        raise FileNotFoundError(
            f"{solid_c_path} not found; needed to ship the obstacle mask "
            "(blanking) with each sample."
        )

    # params.nc (interpolated to the state time grid) was already written
    # before the original run crashed — reuse it.
    interpolated = xr.load_dataset(output_dir / "params.nc")

    canonical = _canonical_z_coords(raw_dir, n_total)
    print(f"Canonical z grid: {canonical['zt'].size} levels, zt={canonical['zt']}")

    first_example: dict[str, xr.Dataset] = {}
    n_fixed = 0
    for split, n, offset in split_specs:
        state_split_dir = output_dir / "state" / split
        param_split_dir = output_dir / "param" / split
        state_split_dir.mkdir(parents=True, exist_ok=True)
        param_split_dir.mkdir(parents=True, exist_ok=True)
        for i in range(n):
            sample_idx = offset + i
            src = raw_dir / f"state_{sample_idx}.nc"
            state_dst = state_split_dir / f"sample_{i:04d}.nc"
            param_dst = param_split_dir / f"sample_{i:04d}.nc"

            if not src.exists():
                if state_dst.exists() and param_dst.exists():
                    if i == 0:
                        first_example[split] = xr.load_dataset(state_dst)
                    print(f"[{split}] sample {i + 1}/{n} already processed, skipping")
                    continue
                raise FileNotFoundError(
                    f"Raw state {src} is gone but outputs are incomplete."
                )

            with xr.open_dataset(src) as ds:
                state = ds.load()
            if state.sizes["zt"] != canonical["zt"].size:
                n_fixed += 1
            state = _fix_z_grid(state, canonical, src.name)
            state = interpolate_grid(state)
            state = _attach_blanking(state, solid_c_path)
            state.to_netcdf(state_dst)

            member_params = interpolated.isel(ensemble=sample_idx).drop_vars("ensemble")
            member_params.to_netcdf(param_dst)
            src.unlink()

            if i == 0:
                first_example[split] = state
            print(f"[{split}] sample {i + 1}/{n} -> {state_dst}")

    print(f"Repaired z grid on {n_fixed} members this run.")

    _plot_sampled_params(
        sampled=interpolated,
        split_offsets=split_specs,
        output_path=output_dir / "params_interpolated.png",
    )
    print(f"Saved interpolated trajectories -> {output_dir / 'params_interpolated.png'}")

    try:
        raw_dir.rmdir()
    except OSError:
        pass

    if first_example:
        _plot_split_examples(first_example, output_dir / "split_examples.png")
        print(f"Saved figure -> {output_dir / 'split_examples.png'}")
        for split, state in first_example.items():
            anim_state = add_velocity_magnitude(state)
            anim_path = output_dir / f"{split}_animation.mp4"
            animate_state(state=anim_state, output_path=anim_path, z_level=0)
            print(f"Saved animation -> {anim_path}")

    print(f"Done. Training data root: {output_dir}")


if __name__ == "__main__":
    default = pathlib.Path("training_data/pyudales_medium")
    run(pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else default)
