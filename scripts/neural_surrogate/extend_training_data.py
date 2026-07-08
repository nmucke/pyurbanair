"""Extend an existing random-geometry training-data folder with more TRAIN sims.

Companion to generate_random_geometries_training_data.py. Point it at a folder
that script produced (idealized | realistic geometry pools) and it appends
``--num-new`` fresh TRAINING simulations in place, leaving val/test untouched.
It reuses the folder's saved ``config.yaml`` (the same backend, generation
horizon and parameter sampler), so the new samples are physically consistent
with the originals.

Three guarantees, matching the requirements this exists for:

  1. **Train only.** Only ``state/train`` + ``param/train`` grow; val/test
     sample files and their geometry rows are never touched.
  2. **No held-out geometries.** New simulations draw geometries from the pool
     with every val/test STL (read from the folder's ``geometries.csv``)
     excluded, so training never sees a validation/test geometry.
  3. **No exact re-runs.** A simulation is identified by (geometry STL, full
     parameter trajectory). New trajectories are drawn from an unused RNG seed
     (recorded in ``extension_log.json``; reused seeds are refused), which
     makes every new trajectory distinct from every existing one. A direct
     on-disk (STL, trajectory) comparison against the current data is run as a
     belt-and-suspenders check and aborts on any collision.

New samples are numbered contiguously after the existing train samples
(``sample_0250.nc`` … for a 250-sample folder) in both ``state/train`` and
``param/train``, so the surrogate dataloader's ``sorted(glob("sample_*.nc"))``
picks them up automatically. The normalization-stats cache keys on the train
file list, so training recomputes it on next run — this script leaves it alone.

Unlike the generators this is a plain argparse script, not a Hydra entry point:
it consumes the *already-resolved* ``config.yaml`` saved in the data folder
rather than composing a fresh config tree.

Usage:

    python scripts/neural_surrogate/extend_training_data.py \
        training_data/pyudales_idealized --num-new 250
    # pin the trajectory seed explicitly (must be unused):
    python scripts/neural_surrogate/extend_training_data.py \
        training_data/pyudales_idealized --num-new 250 --seed 7
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import pathlib
import shutil
import sys
import time as _time
from collections.abc import Callable

if __package__ is None or __package__ == "":
    _here = pathlib.Path(__file__).resolve().parent
    sys.path.insert(0, str(_here.parent))
    sys.path.insert(0, str(_here))

import hydra
import numpy as np
import xarray as xr
from generate_random_geometries_training_data import (
    _POOL_SOURCES,
    _SUPPORTED_MODELS,
    GeometrySpec,
    Sample,
    _build_geometry_pool,
    _resolve_nz,
    _resolve_path,
    _stage_udales_case,
    _validate_ncpu,
)
from generate_training_data import (
    _attach_blanking,
    _augment_params_for_backend,
    _clamp_palm_inflow_block,
    _interpolate_params_to_state_time,
    _plot_sampled_params,
    _plot_split_examples,
    _sample_params,
)
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from pyurbanair.animation import animate_state
from pyurbanair.config.hydra_helpers import (
    clean_outputs,
    resolve_parameter_schema,
)
from pyurbanair.utils.run_utils import add_velocity_magnitude

_MANIFEST_HEADER = [
    "split",
    "sample",
    "stl_file",
    "lx_stl_m",
    "ly_stl_m",
    "z_max_m",
    "nx",
    "ny",
    "nz",
    "lx_domain_m",
    "ly_domain_m",
    "lz_domain_m",
]


def _read_manifest(path: pathlib.Path) -> list[dict[str, str]]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def _append_manifest_rows(path: pathlib.Path, samples: list[Sample]) -> None:
    """Append new train rows to geometries.csv without rewriting the header."""
    with open(path, "a", newline="") as f:
        writer = csv.writer(f)
        for s in samples:
            g = s.geom
            writer.writerow(
                [
                    s.split,
                    f"{s.local_idx:04d}",
                    g.stl_path.name,
                    g.lx,
                    g.ly,
                    g.z_max,
                    g.nx,
                    g.ny,
                    g.nz,
                    g.bounds[0][1],
                    g.bounds[1][1],
                    g.bounds[2][1],
                ]
            )


def _assign_train_geometries(
    n_pool: int, num_new: int, rng: np.random.Generator
) -> np.ndarray:
    """Draw ``num_new`` pool indices, tiling the shuffled pool when it's short.

    Mirrors the train branch of ``_assign_split_geometries`` in the generator:
    each geometry is used either floor or ceil of ``num_new / n_pool`` times, so
    coverage stays as uniform as possible.
    """
    if n_pool == 0:
        raise ValueError("No eligible training geometries after excluding val/test.")
    pool_ids = rng.permutation(n_pool)
    if num_new <= pool_ids.size:
        return pool_ids[:num_new]
    reps, remainder = divmod(num_new, pool_ids.size)
    return rng.permutation(
        np.concatenate(
            [
                np.tile(pool_ids, reps),
                rng.choice(pool_ids, size=remainder, replace=False),
            ]
        )
    )


def _param_signature(member: xr.Dataset) -> str:
    """A stable fingerprint of one member's raw parameter trajectory.

    Rounds each variable to 6 decimals (absorbing float-save jitter) and hashes
    the concatenation; two members share a signature iff their trajectories are
    numerically identical. Used with the geometry name to key simulations for
    duplicate detection.
    """
    h = hashlib.sha256()
    for name in sorted(member.data_vars):
        arr = np.round(np.asarray(member[name].values, dtype=np.float64), 6)
        h.update(name.encode())
        h.update(np.ascontiguousarray(arr).tobytes())
    return h.hexdigest()


def _existing_sim_keys(
    train_raw: xr.Dataset, train_stls: list[str]
) -> set[tuple[str, str]]:
    """Build the (STL, trajectory) key set of every existing train simulation."""
    keys: set[tuple[str, str]] = set()
    for i, stl in enumerate(train_stls):
        keys.add((stl, _param_signature(train_raw.isel(ensemble=i))))
    return keys


def _load_used_seeds(log_path: pathlib.Path, base_seed: int) -> tuple[set[int], list]:
    """Return (used seeds, log entries). The base generation seed is reserved."""
    used = {base_seed}
    entries: list = []
    if log_path.exists():
        entries = json.loads(log_path.read_text())
        used.update(int(e["seed"]) for e in entries)
    return used, entries


def run(
    data_dir: pathlib.Path,
    *,
    num_new: int,
    seed: int | None,
) -> None:
    if num_new <= 0:
        raise ValueError(f"--num-new must be positive, got {num_new}")

    config_path = data_dir / "config.yaml"
    if not config_path.exists():
        raise FileNotFoundError(
            f"{config_path} not found; point --data-dir at a folder produced by "
            "generate_random_geometries_training_data.py."
        )
    cfg: DictConfig = OmegaConf.load(config_path)  # type: ignore[assignment]

    model_name = cfg.model.name
    if model_name not in _SUPPORTED_MODELS:
        raise ValueError(
            f"model={model_name} is not supported; choose one of {_SUPPORTED_MODELS}."
        )
    td = cfg.training_data
    geom_cfg = td.geometry
    source = str(geom_cfg.source)
    if source not in _POOL_SOURCES:
        raise ValueError(
            f"training_data.geometry.source={source!r} is a single-geometry case; "
            "this extender only handles the random-geometry pools "
            f"{_POOL_SOURCES}."
        )

    # --- Existing data: held-out geometries + current train count ----------
    manifest_path = data_dir / "geometries.csv"
    if not manifest_path.exists():
        raise FileNotFoundError(f"{manifest_path} not found; nothing to extend.")
    rows = _read_manifest(manifest_path)
    heldout_stls = {r["stl_file"] for r in rows if r["split"] in ("val", "test")}
    train_rows = [r for r in rows if r["split"] == "train"]
    # Trust the on-disk sample files as the source of truth for the current
    # train count (geometries.csv and the state/ dir must agree).
    state_train_dir = data_dir / "state" / "train"
    param_train_dir = data_dir / "param" / "train"
    n_state = len(sorted(state_train_dir.glob("sample_*.nc")))
    n_param = len(sorted(param_train_dir.glob("sample_*.nc")))
    if not (len(train_rows) == n_state == n_param):
        raise ValueError(
            f"Inconsistent train counts: geometries.csv={len(train_rows)}, "
            f"state/train={n_state}, param/train={n_param}. Fix the folder first."
        )
    num_existing_train = n_state
    train_stls = [r["stl_file"] for r in sorted(train_rows, key=lambda r: r["sample"])]
    print(
        f"Existing folder: {num_existing_train} train samples, "
        f"{len(heldout_stls)} held-out val/test geometries."
    )

    # --- Trajectory seed: pick an unused one, refuse reuse -----------------
    base_seed = int(td.seed)
    log_path = data_dir / "extension_log.json"
    used_seeds, log_entries = _load_used_seeds(log_path, base_seed)
    if seed is None:
        seed = max(used_seeds) + 1
        print(f"Auto-selected unused trajectory seed {seed}.")
    elif seed in used_seeds:
        raise ValueError(
            f"seed={seed} was already used (base or a prior extension): {sorted(used_seeds)}. "
            "Reusing it would reproduce identical parameter trajectories. Pick another."
        )

    # --- Geometry pool, minus held-out geometries --------------------------
    resolution = float(geom_cfg.resolution)
    z_size = float(geom_cfg.z_size)
    nz = _resolve_nz(z_size, resolution)
    stl_dir = _resolve_path(geom_cfg.stl_dir)
    print(f"Scanning geometry pool {stl_dir} (resolution={resolution:g} m)")
    pool = _build_geometry_pool(stl_dir, resolution=resolution, z_size=z_size, nz=nz)
    eligible: list[GeometrySpec] = [
        g for g in pool if g.stl_path.name not in heldout_stls
    ]
    print(
        f"Eligible training geometries: {len(eligible)}/{len(pool)} "
        f"(excluded {len(pool) - len(eligible)} val/test geometries)."
    )

    rng = np.random.default_rng(seed)
    geom_ids = _assign_train_geometries(len(eligible), num_new, rng)
    new_samples: list[Sample] = [
        Sample(
            global_idx=k,  # index into the freshly-sampled param ensemble
            split="train",
            local_idx=num_existing_train + k,  # on-disk sample number
            geom=eligible[int(pool_id)],
        )
        for k, pool_id in enumerate(geom_ids)
    ]
    _validate_ncpu(cfg, new_samples)
    print(
        f"Planned {num_new} new train sims over "
        f"{len({s.geom.name for s in new_samples})} geometries "
        f"(samples {num_existing_train}..{num_existing_train + num_new - 1})."
    )

    # --- Sample fresh parameter trajectories -------------------------------
    sampler_cfg = OmegaConf.to_container(td.params_sampler, resolve=True)
    seconds_per_knot = float(sampler_cfg.pop("seconds_per_knot"))
    sampler_cfg["ensemble_size"] = num_new
    params_sampler = hydra.utils.instantiate(sampler_cfg)
    sampled = _sample_params(
        params_sampler,
        seconds_per_knot=seconds_per_knot,
        simulation_time=float(td.simulation_time),
        seed=seed,
    )
    pgm = None
    if "pressure_gradient_magnitude" in resolve_parameter_schema(model_name):
        pgm = OmegaConf.select(cfg, "training_data.pressure_gradient_magnitude")
        if pgm is not None:
            pgm = float(pgm)
    sampled = _augment_params_for_backend(
        sampled, model_name=model_name, pressure_gradient_magnitude=pgm
    )

    # --- Duplicate guard: no new (geometry, trajectory) matches an existing --
    # The raw control points of every existing train sim live in
    # sampled_params.nc (ensemble slice [0:num_existing_train], aligned with the
    # train rows). A fresh unused seed already guarantees distinct trajectories;
    # this compares the actual on-disk data as a hard backstop.
    existing_raw = xr.open_dataset(data_dir / "sampled_params.nc")
    if existing_raw.sizes["ensemble"] < num_existing_train:
        raise ValueError(
            f"sampled_params.nc has {existing_raw.sizes['ensemble']} members but "
            f"{num_existing_train} train samples exist; cannot verify duplicates."
        )
    existing_keys = _existing_sim_keys(
        existing_raw.isel(ensemble=slice(0, num_existing_train)), train_stls
    )
    new_keys = [
        (s.geom.stl_path.name, _param_signature(sampled.isel(ensemble=s.global_idx)))
        for s in new_samples
    ]
    collisions = [k for k in new_keys if k in existing_keys]
    if collisions:
        raise RuntimeError(
            f"{len(collisions)} planned simulation(s) exactly match existing ones "
            f"(same geometry + trajectory), e.g. {collisions[0]}. Choose a "
            "different --seed."
        )
    print(f"Duplicate check passed: {num_new} simulations are all new.")

    # --- Run each geometry's simulations sequentially ----------------------
    # (Same per-geometry grouping / prep / partition as the generator, but the
    # outputs land at the appended train indices and only geometries.csv grows.)
    raw_root = data_dir / "_raw_states_extend"
    if raw_root.exists():
        shutil.rmtree(raw_root)
    raw_root.mkdir()

    regrid: Callable[[xr.Dataset], xr.Dataset] | None = None
    if model_name == "pyudales":
        from pyudales.utils.grid_utils import interpolate_grid

        regrid = interpolate_grid

    # Ship any newly-used STLs alongside the originals.
    stl_out_dir = data_dir / "geometries"
    stl_out_dir.mkdir(exist_ok=True)
    for geom in {s.geom.stl_path: s.geom for s in new_samples}.values():
        dst = stl_out_dir / geom.stl_path.name
        if not dst.exists():
            shutil.copy2(geom.stl_path, dst)

    groups: dict[pathlib.Path, list[Sample]] = {}
    for s in new_samples:
        groups.setdefault(s.geom.stl_path, []).append(s)

    base_temp_dir = pathlib.Path(cfg.paths.experiment_dir)
    first_example: dict[str, xr.Dataset] = {}
    interp_ref: xr.Dataset | None = None
    t0 = _time.time()
    for group_num, (stl_path, group) in enumerate(groups.items()):
        geom = group[0].geom
        print(
            f"[{group_num + 1}/{len(groups)}] {geom.name}: {len(group)} sim(s), "
            f"grid {geom.nx}x{geom.ny}x{geom.nz}, domain "
            f"{geom.bounds[0][1]:g}x{geom.bounds[1][1]:g}x{geom.bounds[2][1]:g} m"
        )
        group_temp_dir = base_temp_dir / f"geom_{geom.name}"
        if group_temp_dir.exists():
            shutil.rmtree(group_temp_dir)
        group_temp_dir.mkdir(parents=True)
        group_raw_dir = raw_root / geom.name

        fm_overrides: dict = {
            "nx": geom.nx,
            "ny": geom.ny,
            "nz": geom.nz,
            "bounds": [list(b) for b in geom.bounds],
            "simulation_time": float(td.simulation_time),
            "output_frequency": float(td.output_frequency),
            "spinup_time": float(td.spinup_time),
            "results_dir": group_raw_dir,
            "temp_dir": str(group_temp_dir),
        }
        if model_name == "pyudales":
            template_dir = _resolve_path(geom_cfg.udales_case_dir)
            fm_overrides["case_dir"] = str(
                _stage_udales_case(template_dir, stl_path, group_temp_dir)
            )
            fm_overrides["precomputed_geom_dir"] = None
        else:
            fm_overrides["stl_path"] = str(stl_path)
            if model_name == "pypalm":
                fm_overrides["case_dir"] = str(_resolve_path(geom_cfg.palm_case_dir))

        forward_model = instantiate(cfg.model.forward_model, **fm_overrides)
        instantiate(cfg.model.prepare, forward_model=forward_model)
        clean_outputs(model_name=model_name, forward_model=forward_model)
        if model_name == "pypalm":
            _clamp_palm_inflow_block(forward_model)

        solid_c_path = None
        if model_name == "pyudales":
            solid_c_path = forward_model.dirs.experiment_dir / "solid_c.txt"
            if not solid_c_path.exists():
                raise FileNotFoundError(
                    f"{solid_c_path} not found; the geometry preprocessing should "
                    "have written it (needed for the obstacle mask)."
                )

        for j, sample in enumerate(group):
            print(f"  run {j + 1}/{len(group)}: train sample {sample.local_idx}")
            forward_model(
                state=None,
                params=sampled.isel(ensemble=sample.global_idx),
                sim_name=f"state_{j}",
            )

        for j, sample in enumerate(group):
            src = group_raw_dir / f"state_{j}.nc"
            if not src.exists():
                raise FileNotFoundError(
                    f"Expected output {src} not found; did the run fail silently?"
                )
            with xr.open_dataset(src) as ds:
                state = ds.load()
            if regrid is not None:
                state = regrid(state)
            if solid_c_path is not None:
                state = _attach_blanking(state, solid_c_path)
            state.attrs["geometry_stl"] = geom.stl_path.name
            state.attrs["geometry_source"] = source
            state.attrs["resolution_m"] = resolution

            state_time = np.asarray(state["time"].values)
            if interp_ref is None:
                interp_ref = _interpolate_params_to_state_time(sampled, state_time)
            elif state_time.shape[0] != interp_ref.sizes["time"]:
                print(
                    f"WARNING: {src} ({geom.name}) produced {state_time.shape[0]} "
                    f"frames vs {interp_ref.sizes['time']} for the first new sample; "
                    "the run may have been truncated."
                )

            state_dst = state_train_dir / f"sample_{sample.local_idx:04d}.nc"
            state.to_netcdf(state_dst)
            src.unlink()

            member_params = (
                _interpolate_params_to_state_time(
                    sampled.isel(ensemble=[sample.global_idx]), state_time
                )
                .isel(ensemble=0)
                .drop_vars("ensemble")
            )
            member_params.to_netcdf(param_train_dir / f"sample_{sample.local_idx:04d}.nc")

            if "train" not in first_example:
                first_example["train"] = state
            print(f"[train] sample {sample.local_idx} ({geom.name}) -> {state_dst}")

        shutil.rmtree(group_temp_dir, ignore_errors=True)
        shutil.rmtree(group_raw_dir, ignore_errors=True)

    elapsed = _time.time() - t0
    print(
        f"All {num_new} new simulations finished in {elapsed:.1f}s "
        f"(~{elapsed / max(num_new, 1):.1f}s/sim)."
    )
    try:
        raw_root.rmdir()
    except OSError:
        pass

    # --- Persist provenance ------------------------------------------------
    _append_manifest_rows(manifest_path, new_samples)
    log_entries.append(
        {
            "seed": int(seed),
            "num_added": int(num_new),
            "train_index_start": int(num_existing_train),
            "train_index_end": int(num_existing_train + num_new - 1),
            "num_geometries": len({s.geom.name for s in new_samples}),
        }
    )
    log_path.write_text(json.dumps(log_entries, indent=2))
    total_train = num_existing_train + num_new
    OmegaConf.set_struct(cfg, False)
    cfg.training_data.num_train = total_train
    OmegaConf.save(cfg, config_path, resolve=True)
    print(
        f"Updated geometries.csv, extension_log.json, and config "
        f"(num_train {num_existing_train} -> {total_train})."
    )

    # --- Refresh the first-sample figure/animation for a quick eyeball ------
    if first_example.get("train") is not None:
        _plot_split_examples(first_example, data_dir / "extend_example.png")
        anim_state = add_velocity_magnitude(first_example["train"])
        anim_state = anim_state.drop_vars(
            [n for n in anim_state.data_vars if "time" not in anim_state[n].dims]
        )
        animate_state(
            state=anim_state,
            output_path=data_dir / "extend_train_animation.mp4",
            z_level=0,
        )

    print(
        "Done. Normalization stats key on the train file list, so training "
        "recomputes them on the next run (this script leaves the cache alone).\n"
        f"Training data root: {data_dir} ({total_train} train samples)."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "data_dir",
        type=pathlib.Path,
        help="Existing random-geometry training-data folder to extend.",
    )
    parser.add_argument(
        "--num-new",
        type=int,
        default=250,
        help="Number of new TRAIN simulations to append (default: 250).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Trajectory RNG seed for the new samples (default: next unused). "
        "Must not equal the base or any prior extension seed.",
    )
    args = parser.parse_args()
    run(args.data_dir, num_new=args.num_new, seed=args.seed)


if __name__ == "__main__":
    main()
