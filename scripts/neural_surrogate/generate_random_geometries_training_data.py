"""Generate training/validation/test data over randomly sampled STL geometries.

Companion to generate_training_data.py for the geometry POOLS
(`training_data.geometry.source: idealized | realistic`, the UrbanTALES STLs
under examples/geometries/processed/). Each simulation gets a randomly drawn
geometry; the grid is derived per geometry from the STL's physical extent at
the configured resolution (nx/ny rounded up to a multiple of 16 by extending
the domain) with a fixed, shared vertical extent `z_size`.

Splits are geometry-disjoint: `num_val` + `num_test` geometries are held out
of the training pool (one simulation each); the remaining geometries serve
`num_train` simulations, re-drawn (with fresh parameter trajectories) when
`num_train` exceeds the pool.

Simulations run strictly sequentially as direct single-model runs: one
forward model per geometry (voxelized/prepared once), then one call per
simulation — no ensemble machinery is involved. The backend, generation
horizon and parameter sampler are all set in the training_data config.

Usage:

    python scripts/neural_surrogate/generate_random_geometries_training_data.py
    python scripts/neural_surrogate/generate_random_geometries_training_data.py \
        model=pylbm training_data.geometry.source=realistic \
        training_data.geometry.resolution=2.0 training_data.geometry.z_size=64.0
"""

from __future__ import annotations

import csv
import dataclasses
import math
import pathlib
import re
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
import trimesh
import xarray as xr
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
    resolve_output_dir,
    resolve_parameter_schema,
)
from pyurbanair.utils.run_utils import add_velocity_magnitude

_POOL_SOURCES = ("idealized", "realistic")
_SUPPORTED_MODELS = ("pylbm", "pyudales", "pypalm")


@dataclasses.dataclass(frozen=True)
class GeometrySpec:
    """One pool STL with its per-geometry grid at the configured resolution."""

    stl_path: pathlib.Path
    lx: float  # physical STL extent [m]
    ly: float
    z_max: float  # tallest structure in the STL [m]
    nx: int
    ny: int
    nz: int
    bounds: tuple[tuple[float, float], ...]  # ((0, nx*r), (0, ny*r), (0, z_size))

    @property
    def name(self) -> str:
        return self.stl_path.stem


@dataclasses.dataclass(frozen=True)
class Sample:
    """One planned simulation: a (split, index) slot bound to a geometry."""

    global_idx: int  # position in the concatenated train/val/test ordering
    split: str
    local_idx: int  # sample number within the split
    geom: GeometrySpec


def _resolve_path(path_str: str) -> pathlib.Path:
    path = pathlib.Path(path_str)
    return path if path.is_absolute() else pathlib.Path.cwd() / path


def _grid_cells(extent_m: float, resolution_m: float) -> int:
    """Smallest multiple of 16 whose span covers `extent_m` at `resolution_m`.

    The inner `round` absorbs float artifacts (640.0 / 2.0 -> 320.0000...) so
    an exact fit is not bumped a whole 16-cell block up.
    """
    n_cover = math.ceil(round(extent_m / resolution_m, 6))
    return max(16, int(math.ceil(n_cover / 16)) * 16)


def _resolve_nz(z_size: float, resolution: float) -> int:
    nz_exact = z_size / resolution
    nz = int(round(nz_exact))
    if not math.isclose(nz_exact, nz, abs_tol=1e-9) or nz <= 0 or nz % 16 != 0:
        raise ValueError(
            f"training_data.geometry.z_size={z_size} at resolution={resolution} "
            f"gives nz={nz_exact:g}; nz must be a positive multiple of 16. "
            f"Choose z_size as a multiple of {16 * resolution:g} m."
        )
    return nz


def _load_pool_manifest(
    stl_dir: pathlib.Path,
) -> dict[str, tuple[float, float, float]]:
    """Read (Lx_m, Ly_m, z_max_m) per geometry from the pool's manifest.csv.

    Written by examples/geometries/rasters_to_stl.py alongside the STLs; it
    records the source raster's domain size. Returns {} when the file is
    absent or lacks the expected columns (out-of-tree pools, with a warning);
    a malformed row in an otherwise-valid manifest raises — silently falling
    back to mesh bounds would change every grid, since the mesh under-spans
    the raster domain when the outermost buildings sit inset from its edges.
    """
    manifest_path = stl_dir / "manifest.csv"
    if not manifest_path.exists():
        return {}
    entries: dict[str, tuple[float, float, float]] = {}
    with open(manifest_path, newline="") as f:
        reader = csv.DictReader(f)
        required = {"name", "Lx_m", "Ly_m", "z_max_m"}
        if not required <= set(reader.fieldnames or ()):
            print(
                f"WARNING: {manifest_path} lacks columns {sorted(required)}; "
                "falling back to mesh bounds for the domain sizes."
            )
            return {}
        for row in reader:
            try:
                entries[row["name"]] = (
                    float(row["Lx_m"]),
                    float(row["Ly_m"]),
                    float(row["z_max_m"]),
                )
            except (TypeError, ValueError) as err:
                raise ValueError(f"Malformed row in {manifest_path}: {row!r}") from err
    return entries


def _build_geometry_pool(
    stl_dir: pathlib.Path,
    *,
    resolution: float,
    z_size: float,
    nz: int,
) -> list[GeometrySpec]:
    """Scan the pool dir, derive per-geometry grids, drop too-tall geometries.

    The physical domain size comes from the pool's manifest.csv when present
    — the mesh bounds under-span the raster domain when the outermost
    buildings sit inset from the domain edges. Without a manifest entry it
    falls back to the mesh's far bounds corner (the pool contract anchors
    the domain frame at the origin).
    """
    stl_paths = sorted(stl_dir.glob("*.stl"))
    if not stl_paths:
        raise FileNotFoundError(
            f"No .stl files in {stl_dir}; is the raster->STL conversion done "
            "(examples/geometries/rasters_to_stl.py)?"
        )
    manifest = _load_pool_manifest(stl_dir)
    if manifest:
        print(f"Domain sizes from {stl_dir / 'manifest.csv'} ({len(manifest)} entries)")

    pool: list[GeometrySpec] = []
    excluded: list[tuple[str, float]] = []
    for i, stl_path in enumerate(stl_paths):
        entry = manifest.get(stl_path.stem)
        if entry is not None:
            lx, ly, z_max = entry
        else:
            mesh = trimesh.load_mesh(str(stl_path), process=False)
            _, (xmax, ymax, zmax) = np.asarray(mesh.bounds)
            lx, ly, z_max = float(xmax), float(ymax), float(zmax)
            if (i + 1) % 100 == 0:
                print(f"Scanned {i + 1}/{len(stl_paths)} pool STLs...")
        if z_max >= z_size:
            excluded.append((stl_path.stem, z_max))
            continue
        nx = _grid_cells(lx, resolution)
        ny = _grid_cells(ly, resolution)
        pool.append(
            GeometrySpec(
                stl_path=stl_path,
                lx=lx,
                ly=ly,
                z_max=z_max,
                nx=nx,
                ny=ny,
                nz=nz,
                bounds=(
                    (0.0, nx * resolution),
                    (0.0, ny * resolution),
                    (0.0, z_size),
                ),
            )
        )

    if excluded:
        listing = ", ".join(f"{name} (z_max={z:g} m)" for name, z in excluded)
        print(
            f"WARNING: excluded {len(excluded)}/{len(stl_paths)} geometries "
            f"taller than z_size={z_size:g} m: {listing}"
        )
    if not pool:
        raise ValueError(
            f"Every geometry in {stl_dir} exceeds z_size={z_size:g} m; "
            "raise training_data.geometry.z_size."
        )
    return pool


def _assign_split_geometries(
    n_pool: int,
    *,
    num_train: int,
    num_val: int,
    num_test: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Assign a pool index to every simulation, geometry-disjoint across splits.

    Val/test each get distinct held-out geometries (one simulation per
    geometry); train draws from the remainder, tiling the shuffled remainder
    when `num_train` exceeds it so coverage stays as uniform as possible.
    """
    if num_val + num_test > n_pool:
        raise ValueError(
            f"num_val + num_test = {num_val + num_test} exceeds the geometry "
            f"pool size ({n_pool}); val/test geometries must be distinct."
        )
    perm = rng.permutation(n_pool)
    val_ids = perm[:num_val]
    test_ids = perm[num_val : num_val + num_test]
    train_pool = perm[num_val + num_test :]
    if num_train > 0 and train_pool.size == 0:
        raise ValueError(
            f"No geometries left for training after holding out "
            f"{num_val + num_test} val/test geometries from a pool of {n_pool}."
        )
    if num_train <= train_pool.size:
        train_ids = train_pool[:num_train]
    else:
        # Whole tiles + a without-replacement remainder: every training
        # geometry is used either floor or ceil of num_train/pool times.
        reps, remainder = divmod(num_train, train_pool.size)
        train_ids = rng.permutation(
            np.concatenate(
                [
                    np.tile(train_pool, reps),
                    rng.choice(train_pool, size=remainder, replace=False),
                ]
            )
        )
    return train_ids, val_ids, test_ids


def _stage_udales_case(
    template_dir: pathlib.Path,
    stl_path: pathlib.Path,
    staging_dir: pathlib.Path,
) -> pathlib.Path:
    """Materialize a per-geometry uDALES case dir from the source template.

    uDALES has no stl_path knob — the forward model reads `stl_file` from the
    namoptions inside its case dir — so each geometry gets a disposable copy
    of the template with the STL dropped in and `stl_file` rewritten.
    """
    case_dir = staging_dir / "udales_case"
    if case_dir.exists():
        shutil.rmtree(case_dir)
    shutil.copytree(template_dir, case_dir)
    shutil.copy2(stl_path, case_dir / stl_path.name)
    namoptions = sorted(case_dir.glob("namoptions.*"))
    if not namoptions:
        raise FileNotFoundError(f"No namoptions.* in uDALES template {template_dir}.")

    def _point_at_stl(m: re.Match[str]) -> str:
        return f"{m.group(1)}{stl_path.name}"

    for nam in namoptions:
        text, n_sub = re.subn(
            r"(?m)^(\s*stl_file\s*=\s*).*$",
            _point_at_stl,
            nam.read_text(),
        )
        if n_sub == 0:
            raise ValueError(
                f"{nam} has no 'stl_file =' entry to point at {stl_path.name}."
            )
        nam.write_text(text)
    return case_dir


def _write_geometry_manifest(path: pathlib.Path, samples: list[Sample]) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
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
        )
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


def _validate_ncpu(cfg: DictConfig, samples: list[Sample]) -> None:
    """Fail before any simulation if ncpu can't decompose some sampled grid.

    pyudales/pypalm use an x-strip/slab decomposition, so ncpu must divide nx
    for EVERY sampled geometry. All pool nx are multiples of 16.
    """
    ncpu = OmegaConf.select(cfg, "model.forward_model.ncpu")
    if ncpu is None:
        return
    ncpu = int(ncpu)
    bad = sorted({s.geom.nx for s in samples if s.geom.nx % ncpu != 0})
    if bad:
        raise ValueError(
            f"model.forward_model.ncpu={ncpu} does not divide nx for every "
            f"sampled geometry (offending nx: {bad}). Every pool nx is a "
            "multiple of 16, so any ncpu in {1, 2, 4, 8, 16} always works."
        )


def run(cfg: DictConfig) -> None:
    model_name = cfg.model.name
    if model_name not in _SUPPORTED_MODELS:
        raise ValueError(
            f"model={model_name} is not supported for random-geometry data "
            f"generation; choose one of {_SUPPORTED_MODELS}."
        )

    td = cfg.training_data
    geom_cfg = td.geometry
    source = str(geom_cfg.source)
    if source not in _POOL_SOURCES:
        raise ValueError(
            f"training_data.geometry.source={source!r} is a single-geometry "
            "case; use scripts/neural_surrogate/generate_training_data.py for "
            f"it, or set the source to one of {_POOL_SOURCES}."
        )

    num_train = int(td.num_train)
    num_val = int(td.num_val)
    num_test = int(td.num_test)
    n_total = num_train + num_val + num_test
    if n_total == 0:
        raise ValueError("training_data: num_train + num_val + num_test must be > 0")

    resolution = float(geom_cfg.resolution)
    z_size = float(geom_cfg.z_size)
    nz = _resolve_nz(z_size, resolution)

    # --- Geometry pool + split plan ---------------------------------------
    stl_dir = _resolve_path(geom_cfg.stl_dir)
    print(f"Scanning geometry pool {stl_dir} (resolution={resolution:g} m)")
    pool = _build_geometry_pool(stl_dir, resolution=resolution, z_size=z_size, nz=nz)

    rng = np.random.default_rng(int(td.seed))
    train_ids, val_ids, test_ids = _assign_split_geometries(
        len(pool),
        num_train=num_train,
        num_val=num_val,
        num_test=num_test,
        rng=rng,
    )

    samples: list[Sample] = []
    for split, ids, offset in (
        ("train", train_ids, 0),
        ("val", val_ids, num_train),
        ("test", test_ids, num_train + num_val),
    ):
        for local_idx, pool_id in enumerate(ids):
            samples.append(
                Sample(
                    global_idx=offset + local_idx,
                    split=split,
                    local_idx=local_idx,
                    geom=pool[int(pool_id)],
                )
            )
    print(
        f"Planned {n_total} simulations over {len({s.geom.name for s in samples})} "
        f"geometries (pool: {len(pool)}): "
        f"train {num_train} sims / {len(set(map(int, train_ids)))} geoms, "
        f"val {num_val}, test {num_test} (val/test geometries held out of train)"
    )

    _validate_ncpu(cfg, samples)

    # --- Output layout -----------------------------------------------------
    if td.output_dir is not None:
        output_dir = pathlib.Path(td.output_dir)
    else:
        output_dir = resolve_output_dir(cfg, "training_data")
    output_dir.mkdir(parents=True, exist_ok=True)
    # Per-geometry ensemble staging. Stale state files from a previous run
    # would be picked up as warm starts (see generate_training_data.py).
    raw_root = output_dir / "_raw_states"
    if raw_root.exists():
        shutil.rmtree(raw_root)
    raw_root.mkdir()
    print(f"Writing training data to {output_dir}")

    OmegaConf.save(cfg, output_dir / "config.yaml", resolve=True)
    _write_geometry_manifest(output_dir / "geometries.csv", samples)
    stl_out_dir = output_dir / "geometries"
    stl_out_dir.mkdir(exist_ok=True)
    for geom in {s.geom.stl_path: s.geom for s in samples}.values():
        shutil.copy2(geom.stl_path, stl_out_dir / geom.stl_path.name)
    print(
        f"Saved geometry manifest -> {output_dir / 'geometries.csv'} "
        f"(STL copies in {stl_out_dir})"
    )

    # --- Sample parameters (identical to generate_training_data.py) --------
    sampler_cfg = OmegaConf.to_container(td.params_sampler, resolve=True)
    seconds_per_knot = float(sampler_cfg.pop("seconds_per_knot"))
    sampler_cfg["ensemble_size"] = n_total
    params_sampler = hydra.utils.instantiate(sampler_cfg)

    sampled = _sample_params(
        params_sampler,
        seconds_per_knot=seconds_per_knot,
        simulation_time=float(td.simulation_time),
        seed=int(td.seed),
    )

    split_specs = [
        ("train", num_train, 0),
        ("val", num_val, num_train),
        ("test", num_test, num_train + num_val),
    ]
    _plot_sampled_params(
        sampled=sampled,
        split_offsets=split_specs,
        output_path=output_dir / "sampled_params.png",
    )
    print(f"Saved parameter trajectories -> {output_dir / 'sampled_params.png'}")

    pgm = None
    if "pressure_gradient_magnitude" in resolve_parameter_schema(model_name):
        pgm = OmegaConf.select(cfg, "training_data.pressure_gradient_magnitude")
        if pgm is not None:
            pgm = float(pgm)
    sampled = _augment_params_for_backend(
        sampled, model_name=model_name, pressure_gradient_magnitude=pgm
    )
    sampled.to_netcdf(output_dir / "sampled_params.nc")

    # --- Run each geometry's simulations sequentially -----------------------
    # Grouping resampled duplicates means each geometry is voxelized/prepared
    # once, then reused for every simulation drawn on it.
    regrid: Callable[[xr.Dataset], xr.Dataset] | None = None
    if model_name == "pyudales":
        from pyudales.utils.grid_utils import interpolate_grid

        regrid = interpolate_grid

    groups: dict[pathlib.Path, list[Sample]] = {}
    for s in samples:
        groups.setdefault(s.geom.stl_path, []).append(s)

    for split in ("train", "val", "test"):
        (output_dir / "state" / split).mkdir(parents=True, exist_ok=True)
        (output_dir / "param" / split).mkdir(parents=True, exist_ok=True)

    interpolated: xr.Dataset | None = None
    first_example: dict[str, xr.Dataset] = {}
    base_temp_dir = pathlib.Path(cfg.paths.experiment_dir)
    t0 = _time.time()
    for group_num, (stl_path, group) in enumerate(groups.items()):
        geom = group[0].geom
        print(
            f"[{group_num + 1}/{len(groups)}] {geom.name}: {len(group)} sim(s), "
            f"grid {geom.nx}x{geom.ny}x{geom.nz}, domain "
            f"{geom.bounds[0][1]:g}x{geom.bounds[1][1]:g}x{geom.bounds[2][1]:g} m "
            f"(STL {geom.lx:g}x{geom.ly:g} m, z_max {geom.z_max:g} m)"
        )
        # A fresh scratch dir per geometry: sequential reuse of member
        # experiment dirs across different grids leaves stale solver outputs
        # behind (uDALES fielddumps especially). pylbm's build tree lives in
        # the shared LBM checkout, so this does not force full rebuilds.
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
            # save_on_disk mode: every run writes state_{j}.nc here.
            "results_dir": group_raw_dir,
            "temp_dir": str(group_temp_dir),
        }
        if model_name == "pyudales":
            template_dir = _resolve_path(geom_cfg.udales_case_dir)
            fm_overrides["case_dir"] = str(
                _stage_udales_case(template_dir, stl_path, group_temp_dir)
            )
            # Precomputed IBM bundles are grid-specific; always voxelize from
            # the STL for pool geometries.
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
                    f"{solid_c_path} not found; the geometry preprocessing "
                    "should have written it. Without it the training data "
                    "ships no obstacle mask."
                )

        # Direct sequential single-model runs — no ensemble machinery (member
        # experiment-dir clones, process pools) for a per-geometry batch that
        # runs one simulation at a time anyway. Any run failure aborts the
        # generation (already-written splits stay on disk).
        for j, sample in enumerate(group):
            print(
                f"  run {j + 1}/{len(group)}: {sample.split} sample {sample.local_idx}"
            )
            forward_model(
                state=None,
                params=sampled.isel(ensemble=sample.global_idx),
                sim_name=f"state_{j}",
            )

        # Partition this group's members straight into the split layout.
        for j, sample in enumerate(group):
            src = group_raw_dir / f"state_{j}.nc"
            if not src.exists():
                raise FileNotFoundError(
                    f"Expected ensemble output {src} not found; "
                    "did the ensemble run fail silently?"
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

            # Adaptive-timestep solvers (uDALES) stamp outputs at the actual
            # solver time nearest the requested cadence, so the time axis
            # jitters per run. Params are therefore interpolated per sample
            # onto ITS OWN axis (below); params.nc keeps the first sample's
            # axis as the consolidated reference. A differing frame COUNT
            # still signals a truncated run, so warn on it.
            state_time = np.asarray(state["time"].values)
            if interpolated is None:
                interpolated = _interpolate_params_to_state_time(sampled, state_time)
                interpolated.to_netcdf(output_dir / "params.nc")
                print(
                    "Saved consolidated interpolated params -> "
                    f"{output_dir / 'params.nc'}"
                )
            elif state_time.shape[0] != interpolated.sizes["time"]:
                print(
                    f"WARNING: {src} ({geom.name}) produced "
                    f"{state_time.shape[0]} output frames vs "
                    f"{interpolated.sizes['time']} for the first sample; "
                    "the run may have been truncated."
                )

            state_dst = (
                output_dir
                / "state"
                / sample.split
                / f"sample_{sample.local_idx:04d}.nc"
            )
            state.to_netcdf(state_dst)
            src.unlink()

            member_params = (
                _interpolate_params_to_state_time(
                    sampled.isel(ensemble=[sample.global_idx]), state_time
                )
                .isel(ensemble=0)
                .drop_vars("ensemble")
            )
            member_params.to_netcdf(
                output_dir
                / "param"
                / sample.split
                / f"sample_{sample.local_idx:04d}.nc"
            )

            if sample.split not in first_example:
                first_example[sample.split] = state
            print(
                f"[{sample.split}] sample {sample.local_idx + 1} ({geom.name}) "
                f"-> {state_dst}"
            )

        shutil.rmtree(group_temp_dir, ignore_errors=True)
        shutil.rmtree(group_raw_dir, ignore_errors=True)

    elapsed = _time.time() - t0
    print(
        f"All ensembles finished in {elapsed:.1f}s "
        f"(~{elapsed / max(n_total, 1):.1f}s/simulation wall-clock-equivalent)"
    )
    try:
        raw_root.rmdir()
    except OSError:
        pass

    # --- Visualization -----------------------------------------------------
    assert interpolated is not None
    _plot_sampled_params(
        sampled=interpolated,
        split_offsets=split_specs,
        output_path=output_dir / "params_interpolated.png",
    )
    print(
        f"Saved interpolated trajectories -> {output_dir / 'params_interpolated.png'}"
    )

    if first_example:
        _plot_split_examples(first_example, output_dir / "split_examples.png")
        print(f"Saved figure -> {output_dir / 'split_examples.png'}")

        for split, state in first_example.items():
            anim_state = add_velocity_magnitude(state)
            # animate_state slices every data var along time; static vars
            # (the pyudales blanking mask) would crash it.
            anim_state = anim_state.drop_vars(
                [n for n in anim_state.data_vars if "time" not in anim_state[n].dims]
            )
            anim_path = output_dir / f"{split}_animation.mp4"
            animate_state(state=anim_state, output_path=anim_path, z_level=0)
            print(f"Saved animation -> {anim_path}")

    print(f"Done. Training data root: {output_dir}")


@hydra.main(  # type: ignore[misc]
    version_base=None,
    config_path="../../conf",
    config_name="neural_surrogate/training_data",
)
def main(cfg: DictConfig) -> None:
    run(cfg)


if __name__ == "__main__":
    main()
