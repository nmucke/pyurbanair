"""Backfill the `blanking` obstacle mask into an existing pyudales dataset.

uDALES fielddumps carry tiny non-zero velocities inside buildings, so the
`TransitionDataset` fallback ("fluid = non-zero state") classifies every
cell as fluid and the geometry channel degenerates to all ones. The STL
cannot be re-voxelised reliably either (`xie_castro_2008_STL.stl` is not
watertight, making `trimesh.contains` non-deterministic). The authoritative
mask is the solver's own IBM cell-centre classification, `solid_c.txt`,
written by the uDALES preprocessing into the experiment directory.

This script appends that mask to every `state/{split}/sample_*.nc` as an
int8 `blanking(zt, yt, xt)` variable (1 = solid/building, 0 = fluid) — the
convention `TransitionDataset._load_geometry` inverts into its fluid mask.
The variable is added in place via netCDF4 append mode (no file rewrite),
each file is validated first (the time-mean speed inside the mask must be
negligible compared to the fluid), and files that already have `blanking`
are skipped, so the script is idempotent.

Usage:

    python scripts/add_geometry_to_training_data.py \
        [output_dir] [solid_c_path] [splits]

`output_dir` defaults to training_data/pyudales_medium, `solid_c_path` to
`<temp_dir from config.yaml>/experiment/<experiment_name>/solid_c.txt`,
and `splits` to "train,val,test".
"""

from __future__ import annotations

import pathlib
import sys

import numpy as np
from netCDF4 import Dataset
from omegaconf import OmegaConf

SPLITS = ("train", "val", "test")
# A solid cell's time-mean speed must be below this fraction of the
# fluid mean. Validated values on pyudales_medium are ~1e-4 vs ~2-7 m/s.
MAX_SOLID_SPEED_FRACTION = 0.01


def load_obstacle_mask(
    solid_c_path: pathlib.Path, shape_zyx: tuple[int, int, int]
) -> np.ndarray:
    """Build the (zt, yt, xt) obstacle mask from uDALES's solid_c.txt.

    solid_c.txt lists 1-based Fortran (i, j, k) = (x, y, z) cell-centre
    indices of solid cells, one per line after a header.
    """
    idx = np.loadtxt(solid_c_path, skiprows=1, dtype=int)
    if idx.ndim != 2 or idx.shape[1] != 3:
        raise ValueError(f"{solid_c_path}: expected (n, 3) indices, got {idx.shape}")
    nz, ny, nx = shape_zyx
    if (
        idx.min() < 1
        or idx[:, 0].max() > nx
        or idx[:, 1].max() > ny
        or idx[:, 2].max() > nz
    ):
        raise ValueError(
            f"{solid_c_path}: indices out of range for grid "
            f"(nx={nx}, ny={ny}, nz={nz})"
        )
    obstacle = np.zeros(shape_zyx, dtype=np.int8)
    obstacle[idx[:, 2] - 1, idx[:, 1] - 1, idx[:, 0] - 1] = 1
    return obstacle


def validate_mask(nc: Dataset, obstacle: np.ndarray, name: str) -> None:
    """Reject the mask if the flow inside it is not (near-)stagnant."""
    u, v, w = (np.asarray(nc[var][:]) for var in ("u", "v", "w"))
    speed = np.sqrt(u**2 + v**2 + w**2).mean(axis=0)
    solid = obstacle.astype(bool)
    solid_mean = float(speed[solid].mean())
    fluid_mean = float(speed[~solid].mean())
    if solid_mean > MAX_SOLID_SPEED_FRACTION * fluid_mean:
        raise ValueError(
            f"{name}: mean speed inside the obstacle mask ({solid_mean:.3g}) is "
            f"not negligible vs fluid ({fluid_mean:.3g}); the mask does not "
            "match this dataset's geometry."
        )


def append_blanking(nc: Dataset, obstacle: np.ndarray) -> None:
    var = nc.createVariable("blanking", "i1", ("zt", "yt", "xt"))
    var.long_name = "obstacle indicator (1 = solid/building, 0 = fluid)"
    var[:] = obstacle


def run(
    output_dir: pathlib.Path,
    solid_c_path: pathlib.Path | None,
    splits: tuple[str, ...] = SPLITS,
) -> None:
    if solid_c_path is None:
        cfg = OmegaConf.load(output_dir / "config.yaml")
        solid_c_path = (
            pathlib.Path(cfg.model.forward_model.temp_dir)
            / "experiment"
            / str(cfg.model.forward_model.experiment_name)
            / "solid_c.txt"
        )
    print(f"Obstacle source: {solid_c_path}")

    obstacle: np.ndarray | None = None
    n_added = n_skipped = 0
    locked: list[str] = []
    for split in splits:
        for path in sorted((output_dir / "state" / split).glob("sample_*.nc")):
            # Validate on a read-only handle so the exclusive (write) lock is
            # only held for the brief append below — another process reading
            # the dataset (e.g. an active training run) keeps these files
            # open intermittently, and HDF5 refuses concurrent writers.
            with Dataset(path, "r") as nc:
                if obstacle is None:
                    shape = tuple(
                        nc.dimensions[d].size for d in ("zt", "yt", "xt")
                    )
                    obstacle = load_obstacle_mask(solid_c_path, shape)
                    print(
                        f"Mask: {int(obstacle.sum())} solid cells on grid "
                        f"(zt, yt, xt)={shape}"
                    )
                if "blanking" in nc.variables:
                    n_skipped += 1
                    continue
                validate_mask(nc, obstacle, path.name)
            try:
                with Dataset(path, "a") as nc:
                    append_blanking(nc, obstacle)
            except OSError:
                locked.append(f"{split}/{path.name}")
                print(f"[{split}] {path.name}: LOCKED by another process, skipped")
                continue
            n_added += 1
            print(f"[{split}] {path.name}: blanking added")

    if n_added + n_skipped + len(locked) == 0:
        raise FileNotFoundError(f"No sample_*.nc found under {output_dir}/state/")
    print(f"Done: {n_added} files patched, {n_skipped} already had blanking.")
    if locked:
        print(
            f"{len(locked)} files were locked by another process and skipped; "
            "rerun this script once that process exits to patch them:\n  "
            + "\n  ".join(locked)
        )
        sys.exit(1)


if __name__ == "__main__":
    out = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "training_data/pyudales_medium")
    src = pathlib.Path(sys.argv[2]) if len(sys.argv) > 2 and sys.argv[2] else None
    chosen = tuple(sys.argv[3].split(",")) if len(sys.argv) > 3 else SPLITS
    run(out, src, chosen)
