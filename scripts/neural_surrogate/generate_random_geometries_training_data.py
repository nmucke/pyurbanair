"""Generate training/validation/test data over randomly sampled STL geometries.

Companion to generate_training_data.py for the geometry POOLS
(`training_data.geometry.source: idealized | realistic`, the UrbanTALES STLs
under examples/geometries/processed/). Each simulation gets a randomly drawn
geometry; the grid is derived per geometry from the STL's physical extent at
the configured resolution (nx/ny rounded up to a multiple of 16 by extending
the domain) with a fixed, shared vertical extent `z_size`.

The extra span is placed around the mesh rather than only behind it. In x the
configurable fetch and wake (`training_data.geometry.upstream_padding` /
`downstream_padding`, m) go in front of and behind the geometry, with the
rounding slack added to the wake; in y `lateral_padding` is applied to BOTH
sides and the slack is split evenly over them. The mesh is never moved — the
domain window is, via non-zero lower `bounds` (all three backends shift the
geometry and the output coordinates accordingly).

Splits are geometry-disjoint: `num_val` + `num_test` geometries are held out
of the training pool (one simulation each); the remaining geometries serve
`num_train` simulations, re-drawn (with fresh parameter trajectories) when
`num_train` exceeds the pool.

Simulations run strictly sequentially as direct single-model runs: one
forward model per geometry (voxelized/prepared once), then one call per
simulation — no ensemble machinery is involved. The backend, generation
horizon and parameter sampler are all set in the training_data config.

A generation too long for one job is split with `training_data.sharding`
(`stage: plan | simulate | finalize`), producing the same corpus as the
one-process `stage: all`: `plan` draws every geometry assignment and parameter
trajectory once and freezes them (with the resolved config) in the output dir;
each `simulate` shard runs a cost-balanced, disjoint set of geometry groups
against that frozen plan and skips samples already on disk, so a shard killed
at the time limit is resumed by resubmitting it; `finalize` checks every sample
exists and writes params.nc and the figures. Rerunning `all` on a finished or
partial output dir resumes it the same way.

Usage:

    python scripts/neural_surrogate/generate_random_geometries_training_data.py
    python scripts/neural_surrogate/generate_random_geometries_training_data.py \
        model=pylbm training_data.geometry.source=realistic \
        training_data.geometry.resolution=2.0 training_data.geometry.z_size=64.0

    # Sharded (see job_scripts/delftblue/submit_random_geometries_training_data.sh):
    python ... training_data.output_dir=/data/run training_data.sharding.stage=plan \
        training_data.sharding.num_shards=16
    python ... training_data.output_dir=/data/run training_data.sharding.stage=simulate \
        training_data.sharding.num_shards=16 training_data.sharding.shard_index=3
    python ... training_data.output_dir=/data/run training_data.sharding.stage=finalize
"""

from __future__ import annotations

import csv
import dataclasses
import math
import os
import pathlib
import re
import shutil
import sys
import time as _time
import traceback
from collections.abc import Callable

# Headless plotting. On a workstation with DISPLAY set, matplotlib picks a GUI
# backend (qtagg), which opens an X11/ICE connection; when that connection tears
# down at interpreter exit the ICE library's default IO error handler calls
# exit() itself, so the process returns nonzero ("ICE default IO error handler
# doing an exit(), errno = 32") AFTER a fully successful run. A batch generator
# only ever writes figures to disk, so force Agg. Set via the environment rather
# than matplotlib.use() so it cannot be defeated by import ordering (isort moves
# imports; this block stays put).
os.environ.setdefault("MPLBACKEND", "Agg")

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
    # ((x0, x1), (y0, y1), (0, z_size)) [m]. x0/y0 are <= 0: the mesh frame is
    # anchored at the origin and the domain is widened around it.
    bounds: tuple[tuple[float, float], ...]

    @property
    def name(self) -> str:
        return self.stl_path.stem

    @property
    def domain_size(self) -> tuple[float, float, float]:
        """Physical domain extent (lx, ly, lz) in metres."""
        (x0, x1), (y0, y1), (z0, z1) = self.bounds
        return (x1 - x0, y1 - y0, z1 - z0)


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


def _cover_cells(extent_m: float, resolution_m: float) -> int:
    """Cells needed to span `extent_m` at `resolution_m`.

    The inner `round` absorbs float artifacts (640.0 / 2.0 -> 320.0000...) so
    an exact fit is not bumped a whole cell up.
    """
    return int(math.ceil(round(extent_m / resolution_m, 6)))


def _grid_cells(extent_m: float, resolution_m: float, *, extra_cells: int = 0) -> int:
    """Smallest multiple of 16 covering `extent_m` plus `extra_cells` of padding."""
    n_cover = _cover_cells(extent_m, resolution_m) + extra_cells
    return max(16, int(math.ceil(n_cover / 16)) * 16)


def _axis_window(
    resolution_m: float, *, n_cells: int, front_cells: int
) -> tuple[float, float]:
    """Domain interval for one axis, with `front_cells` cells before the mesh.

    The STL frame is anchored at 0, so the domain runs from `-front_cells * r`
    over `n_cells` cells and whatever padding is not in front ends up behind
    the mesh. Offsets are whole cells on purpose: a fractional shift would move
    every cell centre off the STL's own raster frame and change how the
    buildings voxelize.
    """
    lo = -front_cells * resolution_m
    return (lo, lo + n_cells * resolution_m)


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
    upstream_padding: float = 0.0,
    downstream_padding: float = 0.0,
    lateral_padding: float = 0.0,
) -> list[GeometrySpec]:
    """Scan the pool dir, derive per-geometry grids, drop too-tall geometries.

    The physical domain size comes from the pool's manifest.csv when present
    — the mesh bounds under-span the raster domain when the outermost
    buildings sit inset from the domain edges. Without a manifest entry it
    falls back to the mesh's far bounds corner (the pool contract anchors
    the domain frame at the origin).

    `upstream_padding` / `downstream_padding` (m, each rounded up to whole
    cells) are open fluid inserted in front of and behind the mesh in x —
    inflow fetch, so the boundary profile is not imposed on the first row of
    buildings, and wake, so the last row does not sit on the outlet.
    `lateral_padding` is the same thing on BOTH y sides (the flow crosses
    laterally either way as `inflow_angle` changes sign, so the two sides are
    not distinguishable and share one knob). All three are minima: the slack
    from rounding up to a multiple of 16 is added to the wake in x and split
    evenly over the two sides in y. The mesh stays where it is; the domain's
    lower bound goes negative and the backend shifts the geometry.
    """
    for knob, value in (
        ("upstream_padding", upstream_padding),
        ("downstream_padding", downstream_padding),
        ("lateral_padding", lateral_padding),
    ):
        if value < 0:
            raise ValueError(
                f"training_data.geometry.{knob} must be >= 0, got {value}."
            )
    front_x = _cover_cells(upstream_padding, resolution)
    back_x = _cover_cells(downstream_padding, resolution)
    side_y = _cover_cells(lateral_padding, resolution)
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
        # x: fetch in front, wake behind, and the slack the 16-multiple
        # rounding adds on top of both goes to the wake — which is where a
        # longer domain is actually wanted.
        nx = _grid_cells(lx, resolution, extra_cells=front_x + back_x)
        # y: `lateral_padding` on both sides, then the rounding slack on top
        # of it split evenly over the two; an odd cell count leaves the extra
        # cell at the far side.
        ny = _grid_cells(ly, resolution, extra_cells=2 * side_y)
        slack_y = ny - _cover_cells(ly, resolution) - 2 * side_y
        front_y = side_y + slack_y // 2
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
                    _axis_window(resolution, n_cells=nx, front_cells=front_x),
                    _axis_window(resolution, n_cells=ny, front_cells=front_y),
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


# `x0_domain_m` / `y0_domain_m` are the domain's lower bounds in the mesh frame
# (<= 0): together with the extents they say where the geometry sits inside the
# padded domain.
MANIFEST_COLUMNS = (
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
    "x0_domain_m",
    "y0_domain_m",
)


def manifest_row(sample: Sample) -> list:
    """One `geometries.csv` row.

    Shared with extend_training_data.py, which appends rows under the header
    written here, so the two cannot drift apart.
    """
    g = sample.geom
    lx_domain, ly_domain, lz_domain = g.domain_size
    return [
        sample.split,
        f"{sample.local_idx:04d}",
        g.stl_path.name,
        g.lx,
        g.ly,
        g.z_max,
        g.nx,
        g.ny,
        g.nz,
        lx_domain,
        ly_domain,
        lz_domain,
        g.bounds[0][0],
        g.bounds[1][0],
    ]


def _write_geometry_manifest(path: pathlib.Path, samples: list[Sample]) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(list(MANIFEST_COLUMNS))
        for s in samples:
            writer.writerow(manifest_row(s))


def _resolve_padding(geom_cfg: DictConfig) -> tuple[float, float, float]:
    """Read the `(upstream, downstream, lateral)` padding knobs, in metres.

    Absent/null means no padding on that side; all three at 0.0 is the
    pre-knob behaviour of packing every spare cell behind the mesh in x and
    splitting only the rounding slack in y.
    """
    values = []
    for knob in ("upstream_padding", "downstream_padding", "lateral_padding"):
        value = OmegaConf.select(geom_cfg, knob)
        values.append(0.0 if value is None else float(value))
    return (values[0], values[1], values[2])


# Bulk fill speed as a fraction of the reference speed. The inlet mean profile
# is (z/z_ref)^alpha, whose vertical average over [0, zsize] is
# (zsize/z_ref)^alpha / (1 + alpha) -- with the default z_ref = zsize that is
# 1/(1+alpha) = 0.8 at alpha = 0.25. This is the speed at which the domain
# actually fills, which is what sizes the spinup.
def _bulk_fill_fraction(alpha: float, *, z_size: float, z_ref: float) -> float:
    return float((z_size / z_ref) ** alpha / (1.0 + alpha))


def _resolve_spinup_time(
    group: list[Sample],
    sampled: xr.Dataset,
    *,
    default_spinup: float,
    output_frequency: float,
    adaptive_cfg: DictConfig | None,
    alpha: float,
    z_size: float,
    z_ref: float,
) -> float:
    """Spinup for one geometry group, sized from its slowest streamwise fill.

    On the turbulent-inlet path uDALES starts the interior from REST with
    interior nudging off (`inlet_turbulence_utils._write_initial_condition_files`),
    so the domain fills by advection from the inlet face alone -- the spinup is
    a flow-through-time problem, not a `tnudge` one, and a value sized from
    nudging-path experience is far too short (docs/pyudales.md 6.1). Frames
    saved before the domain has filled carry a streamwise velocity deficit that
    is pure startup artefact: measured at 27% inlet->outlet on a 764 m domain
    given 2 flow-throughs.

    The fill speed is the profile's bulk mean `U * _bulk_fill_fraction()`
    projected onto x by `cos(inflow_angle)`, both taken from the group's
    INITIAL parameter values because `apply_time_varying_inflow` holds the
    parameters on a constant plateau for the whole spinup. The group's most
    demanding sample wins (groups are 1-2 samples, so this is effectively
    per-simulation).

    Returns `default_spinup` unchanged when adaptive sizing is disabled, and
    never returns less than it -- so the value recorded as
    `training_data.spinup_time` in the saved config stays a valid lower bound
    for the corpus (train_latent_generator's `constant_prehistory` check reads
    that scalar).
    """
    if adaptive_cfg is None or not bool(OmegaConf.select(adaptive_cfg, "enabled")):
        return default_spinup

    fill_times = float(OmegaConf.select(adaptive_cfg, "fill_times") or 6.0)
    max_spinup = float(OmegaConf.select(adaptive_cfg, "max_spinup_time") or 1500.0)
    if max_spinup < default_spinup:
        raise ValueError(
            f"training_data.adaptive_spinup.max_spinup_time={max_spinup} is below "
            f"training_data.spinup_time={default_spinup}, which is the floor."
        )

    lx_domain = group[0].geom.domain_size[0]
    bulk = _bulk_fill_fraction(alpha, z_size=z_size, z_ref=z_ref)

    slowest = math.inf
    for sample in group:
        member = sampled.isel(ensemble=sample.global_idx)
        speed = float(np.asarray(member["velocity_magnitude"].values).ravel()[0])
        angle = float(np.asarray(member["inflow_angle"].values).ravel()[0])
        # cos is clamped away from 0: a hypothetical 90-degree inflow has no
        # streamwise component at all and would divide by zero.
        u_x = bulk * speed * max(math.cos(math.radians(angle)), 0.1)
        slowest = min(slowest, u_x)

    needed = fill_times * lx_domain / slowest
    # Round UP to a whole number of output frames: the trim at
    # forward_model.py:1230 uses int(spinup/output_frequency), so a spinup that
    # is not a multiple of the cadence would leave a partial frame behind. A
    # multiple of output_frequency also keeps `spinup + simulation_time`
    # divisible by inlet_turbulence.time_step.
    quantized = math.ceil(needed / output_frequency) * output_frequency
    return float(min(max(quantized, default_spinup), max_spinup))


def _select_saved_vars(state: xr.Dataset, save_vars: list[str] | None) -> xr.Dataset:
    """Drop state variables the corpus does not need.

    `save_vars` is a whitelist of TIME-VARYING variables; anything without a
    time dimension (the pyudales blanking mask) is always kept, since it is
    geometry, not state. None keeps everything -- a strict no-op.

    uDALES returns `pres` alongside u/v/w and nothing downstream reads it:
    every conf/neural_surrogate/*.yaml uses `state_vars: [u, v, w]`. On this
    pool it is a quarter of the corpus on disk.
    """
    if save_vars is None:
        return state
    keep = set(save_vars)
    drop = [
        name
        for name, da in state.data_vars.items()
        if "time" in da.dims and name not in keep
    ]
    missing = keep - set(state.data_vars)
    if missing:
        raise KeyError(
            f"training_data.save_vars requests {sorted(missing)}, which the "
            f"{'/'.join(sorted(state.data_vars))} state does not carry."
        )
    return state.drop_vars(drop) if drop else state


def _netcdf_encoding(state: xr.Dataset, encoding_cfg: DictConfig | None) -> dict:
    """Per-variable NetCDF encoding, or {} when unconfigured (no-op).

    `least_significant_digit` is a LOSSY bit-rounding that makes zlib far more
    effective on these fields (~2.5x total vs ~1.3x lossless). It is applied to
    float variables only -- on the int8 blanking mask it would be meaningless.
    At 3 digits the quantisation is 5e-4 m/s on a 5-10 m/s field, orders of
    magnitude below the solver's own error.
    """
    if encoding_cfg is None:
        return {}
    base = OmegaConf.to_container(encoding_cfg, resolve=True)
    lsd = base.pop("least_significant_digit", None)
    encoding = {}
    for name, da in state.data_vars.items():
        spec = dict(base)
        if lsd is not None and da.dtype.kind == "f":
            spec["least_significant_digit"] = int(lsd)
        encoding[name] = spec
    return encoding


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


# --- Sharding -----------------------------------------------------------------
# One generation can be split over independent jobs (a SLURM array) without
# changing what it produces. `plan` draws everything random -- geometry/split
# assignment and every parameter trajectory -- exactly once and freezes it in
# the output dir; each `simulate` shard then runs a disjoint subset of geometry
# groups against that frozen plan; `finalize` writes the corpus-level outputs
# once every sample is on disk. `all` does the three in one process.
_STAGES = ("all", "plan", "simulate", "finalize")

# Keys a simulate/finalize job takes from its OWN command line instead of the
# plan's frozen config.yaml: where its scratch lives, how many cores it has and
# which shard it is. Everything else -- physics, horizon, sampler, pool -- comes
# from the plan, so a shard that starts hours later still runs the configuration
# the plan was drawn under even if the checkout's conf/ was edited meanwhile.
_RUNTIME_KEYS = (
    "paths",
    "model.forward_model.ncpu",
    "model.forward_model.temp_dir",
    "model.forward_model.output_dir",
    "model.forward_model.matlab_bin",
    "model.forward_model.verbose",
    "training_data.sharding",
)


@dataclasses.dataclass(frozen=True)
class ShardingSpec:
    stage: str
    num_shards: int
    shard_index: int


def _resolve_sharding(cfg: DictConfig) -> ShardingSpec:
    """Read `training_data.sharding`; absent means the one-process `all` run."""
    stage = str(OmegaConf.select(cfg, "training_data.sharding.stage") or "all")
    num_shards = int(OmegaConf.select(cfg, "training_data.sharding.num_shards") or 1)
    shard_index = int(OmegaConf.select(cfg, "training_data.sharding.shard_index") or 0)
    if stage not in _STAGES:
        raise ValueError(
            f"training_data.sharding.stage={stage!r}; choose one of {_STAGES}."
        )
    if num_shards < 1 or not 0 <= shard_index < num_shards:
        raise ValueError(
            f"training_data.sharding: need 0 <= shard_index ({shard_index}) < "
            f"num_shards ({num_shards})."
        )
    if stage == "all" and num_shards != 1:
        raise ValueError(
            "training_data.sharding.stage=all runs every geometry in one process; "
            "use stage=plan/simulate/finalize to shard."
        )
    return ShardingSpec(stage=stage, num_shards=num_shards, shard_index=shard_index)


def _without_runtime_keys(cfg: DictConfig) -> dict:
    """Resolved config as a plain dict with the `_RUNTIME_KEYS` removed."""
    container = OmegaConf.to_container(cfg, resolve=True)
    assert isinstance(container, dict)
    for key in _RUNTIME_KEYS:
        *parents, leaf = key.split(".")
        node: object = container
        for part in parents:
            node = node.get(part) if isinstance(node, dict) else None
        if isinstance(node, dict):
            node.pop(leaf, None)
    return container


def _diff_keys(a: object, b: object, prefix: str = "") -> list[str]:
    """Dotted keys at which two plain config containers differ."""
    if isinstance(a, dict) and isinstance(b, dict):
        out: list[str] = []
        for key in sorted(set(a) | set(b), key=str):
            out += _diff_keys(a.get(key), b.get(key), f"{prefix}{key}.")
        return out
    return [] if a == b else [prefix.rstrip(".") or "<root>"]


def _load_plan_config(cfg: DictConfig, output_dir: pathlib.Path) -> DictConfig:
    """The plan's frozen config with this job's runtime keys laid over it."""
    path = output_dir / "config.yaml"
    if not (path.exists() and (output_dir / "sampled_params.nc").exists()):
        raise FileNotFoundError(
            f"No complete plan in {output_dir} (config.yaml + sampled_params.nc); "
            "run training_data.sharding.stage=plan first."
        )
    frozen = OmegaConf.load(path)
    assert isinstance(frozen, DictConfig)
    ignored = _diff_keys(_without_runtime_keys(frozen), _without_runtime_keys(cfg))
    if ignored:
        print(
            f"NOTE: running the plan's frozen {path}; this job's config differs "
            f"at {ignored} and those values are IGNORED."
        )
    for key in _RUNTIME_KEYS:
        value = OmegaConf.select(cfg, key)
        if value is None:
            continue
        if isinstance(value, DictConfig):
            value = OmegaConf.to_container(value, resolve=True)
        OmegaConf.update(frozen, key, value, merge=False, force_add=True)
    return frozen


def _plan_samples(cfg: DictConfig) -> list[Sample]:
    """Every planned simulation, in the one-process order (train, val, test).

    Deterministic in the config and the pool dir's contents only, so the plan
    stage and every shard derive the same list.
    """
    td = cfg.training_data
    geom_cfg = td.geometry
    num_train = int(td.num_train)
    num_val = int(td.num_val)
    num_test = int(td.num_test)

    resolution = float(geom_cfg.resolution)
    z_size = float(geom_cfg.z_size)
    nz = _resolve_nz(z_size, resolution)
    upstream_padding, downstream_padding, lateral_padding = _resolve_padding(geom_cfg)

    stl_dir = _resolve_path(geom_cfg.stl_dir)
    print(
        f"Scanning geometry pool {stl_dir} (resolution={resolution:g} m, "
        f"fetch {upstream_padding:g} m = "
        f"{_cover_cells(upstream_padding, resolution)} cells, "
        f"wake >= {downstream_padding:g} m = "
        f"{_cover_cells(downstream_padding, resolution)} cells, "
        f"lateral >= {lateral_padding:g} m = "
        f"{_cover_cells(lateral_padding, resolution)} cells per side)"
    )
    pool = _build_geometry_pool(
        stl_dir,
        resolution=resolution,
        z_size=z_size,
        nz=nz,
        upstream_padding=upstream_padding,
        downstream_padding=downstream_padding,
        lateral_padding=lateral_padding,
    )

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
        f"Planned {len(samples)} simulations over "
        f"{len({s.geom.name for s in samples})} geometries (pool: {len(pool)}): "
        f"train {num_train} sims / {len(set(map(int, train_ids)))} geoms, "
        f"val {num_val}, test {num_test} (val/test geometries held out of train)"
    )
    return samples


def _group_samples(samples: list[Sample]) -> list[list[Sample]]:
    """Samples grouped by geometry, groups in first-occurrence order.

    Grouping resampled duplicates means each geometry is voxelized/prepared
    once, then reused for every simulation drawn on it.
    """
    groups: dict[pathlib.Path, list[Sample]] = {}
    for s in samples:
        groups.setdefault(s.geom.stl_path, []).append(s)
    return list(groups.values())


def _check_geometry_manifest(path: pathlib.Path, samples: list[Sample]) -> None:
    """Refuse to run if the pool no longer reproduces the plan's geometries.csv.

    The config is frozen, but the pool dir is read live: a changed STL set or
    manifest would silently re-deal geometries to sample slots.
    """
    with open(path, newline="") as f:
        saved = list(csv.reader(f))
    expected = [list(MANIFEST_COLUMNS)] + [
        [str(v) for v in manifest_row(s)] for s in samples
    ]
    if saved != expected:
        raise ValueError(
            f"The geometry pool no longer reproduces {path}: the STL dir or its "
            "manifest.csv (or the file itself) changed since the plan was made. "
            "Restore the pool, or start a new plan in a fresh "
            "training_data.output_dir."
        )


def _sample_paths(
    output_dir: pathlib.Path, sample: Sample
) -> tuple[pathlib.Path, pathlib.Path]:
    name = f"sample_{sample.local_idx:04d}.nc"
    return (
        output_dir / "state" / sample.split / name,
        output_dir / "param" / sample.split / name,
    )


def _sample_done(output_dir: pathlib.Path, sample: Sample) -> bool:
    # Both files are written atomically, the param file last, so their joint
    # presence means the sample is complete -- a shard killed at the SLURM time
    # limit never leaves a truncated file under the final name.
    state_path, param_path = _sample_paths(output_dir, sample)
    return state_path.exists() and param_path.exists()


def _to_netcdf_atomic(
    ds: xr.Dataset, dst: pathlib.Path, *, encoding: dict | None = None
) -> None:
    """Write via a hidden temp name + rename, so `dst` is complete or absent.

    The temp name does not match the dataloader's `sample_*.nc` glob.
    """
    tmp = dst.with_name(f".{dst.name}.tmp")
    ds.to_netcdf(tmp, encoding=encoding)
    os.replace(tmp, dst)


def _group_spinups(
    cfg: DictConfig, groups: list[list[Sample]], sampled: xr.Dataset
) -> list[float]:
    """Spinup per geometry group, from the group's FULL sample list.

    Always sized over the whole group, never just its still-pending samples, so
    a resumed group gets the same spinup it would have had in one process.
    """
    td = cfg.training_data
    z_size = float(td.geometry.z_size)
    # The vertical profile lives under the model's nudging_config even on the
    # turbulent-inlet path, which reuses the same `build_profile_shape` for the
    # driver-plane means (inlet_turbulence_utils:594). `z_ref` defaults to the
    # domain top when unset, exactly as vertical_profile._power_law does.
    adaptive_cfg = OmegaConf.select(cfg, "training_data.adaptive_spinup")
    profile_key = "model.forward_model.nudging_config.profile_config"
    alpha = float(OmegaConf.select(cfg, f"{profile_key}.alpha") or 0.0)
    z_ref = float(OmegaConf.select(cfg, f"{profile_key}.z_ref") or z_size)
    return [
        _resolve_spinup_time(
            group,
            sampled,
            default_spinup=float(td.spinup_time),
            output_frequency=float(td.output_frequency),
            adaptive_cfg=adaptive_cfg,
            alpha=alpha,
            z_size=z_size,
            z_ref=z_ref,
        )
        for group in groups
    ]


def _group_cost(
    group: list[Sample],
    sampled: xr.Dataset,
    *,
    spinup: float,
    simulation_time: float,
) -> float:
    """Relative compute cost of one geometry group, for balancing shards.

    Cells x simulated seconds x mean inflow speed: the adaptive (CFL) time step
    shrinks as the flow speeds up, so the step count scales with the speed as
    well as with the horizon. Only ratios between groups are meaningful.
    """
    g = group[0].geom
    cells = g.nx * g.ny * g.nz
    cost = 0.0
    for s in group:
        speed = 1.0
        if "velocity_magnitude" in sampled:
            member = sampled["velocity_magnitude"].isel(ensemble=s.global_idx)
            speed = float(member.mean())
        cost += cells * (spinup + simulation_time) * speed
    return cost


def _assign_shards(costs: list[float], num_shards: int) -> list[int]:
    """Shard index per group: greedy longest-processing-time balancing.

    Deterministic (ties break on group order, then shard index), so every job
    recomputes the same assignment from the same plan. Groups are never split:
    each geometry is prepared once, in exactly one shard.
    """
    loads = [0.0] * num_shards
    shard_of = [0] * len(costs)
    for i in sorted(range(len(costs)), key=lambda i: (-costs[i], i)):
        k = min(range(num_shards), key=lambda k: (loads[k], k))
        shard_of[i] = k
        loads[k] += costs[i]
    return shard_of


def _print_shard_summary(
    groups: list[list[Sample]],
    costs: list[float],
    shard_of: list[int],
    num_shards: int,
) -> None:
    total = sum(costs) or 1.0
    print(f"Shard assignment ({num_shards} shards, share of total compute):")
    for k in range(num_shards):
        members = [i for i, s in enumerate(shard_of) if s == k]
        n_sims = sum(len(groups[i]) for i in members)
        share = sum(costs[i] for i in members) / total
        print(
            f"  shard {k:3d}: {len(members):4d} geometries, {n_sims:4d} sims, "
            f"{100 * share:5.1f}%"
        )
    print(
        f"Largest single geometry group: {100 * max(costs) / total:.1f}% of the "
        "total -- no shard can finish faster than that."
    )


def _shard_layout(
    cfg: DictConfig,
    groups: list[list[Sample]],
    sampled: xr.Dataset,
    num_shards: int,
) -> tuple[list[float], list[float], list[int]]:
    """(spinup, cost, shard index) per group."""
    spinups = _group_spinups(cfg, groups, sampled)
    simulation_time = float(cfg.training_data.simulation_time)
    costs = [
        _group_cost(g, sampled, spinup=sp, simulation_time=simulation_time)
        for g, sp in zip(groups, spinups)
    ]
    return spinups, costs, _assign_shards(costs, num_shards)


def _split_specs(td: DictConfig) -> list[tuple[str, int, int]]:
    num_train, num_val, num_test = int(td.num_train), int(td.num_val), int(td.num_test)
    return [
        ("train", num_train, 0),
        ("val", num_val, num_train),
        ("test", num_test, num_train + num_val),
    ]


def _write_plan(
    cfg: DictConfig, samples: list[Sample], output_dir: pathlib.Path
) -> None:
    """Freeze everything random into `output_dir`, or verify an existing plan.

    An existing complete plan (sampled_params.nc present) is reused when it was
    made under the same config and still matches the pool -- rerunning a
    finished or partial generation then resumes it instead of redrawing. A plan
    made under a different config is refused rather than overwritten, since its
    sample files would silently mix with the new ones.
    """
    config_path = output_dir / "config.yaml"
    manifest_path = output_dir / "geometries.csv"
    params_path = output_dir / "sampled_params.nc"
    if params_path.exists() and config_path.exists():
        saved = OmegaConf.load(config_path)
        assert isinstance(saved, DictConfig)
        differing = _diff_keys(_without_runtime_keys(saved), _without_runtime_keys(cfg))
        if differing:
            raise ValueError(
                f"{output_dir} already holds a plan made under a different config "
                f"(differing keys: {differing}). Use a fresh "
                "training_data.output_dir, or delete that one to start over."
            )
        _check_geometry_manifest(manifest_path, samples)
        print(f"Reusing the existing plan in {output_dir} (same config and pool)")
        return

    frozen = OmegaConf.to_container(cfg, resolve=True)
    assert isinstance(frozen, dict)
    frozen.get("training_data", {}).pop("sharding", None)
    OmegaConf.save(OmegaConf.create(frozen), config_path)
    _write_geometry_manifest(manifest_path, samples)
    stl_out_dir = output_dir / "geometries"
    stl_out_dir.mkdir(exist_ok=True)
    for geom in {s.geom.stl_path: s.geom for s in samples}.values():
        shutil.copy2(geom.stl_path, stl_out_dir / geom.stl_path.name)
    print(f"Saved geometry manifest -> {manifest_path} (STL copies in {stl_out_dir})")

    # --- Sample parameters (identical to generate_training_data.py) --------
    td = cfg.training_data
    model_name = cfg.model.name
    sampler_cfg = OmegaConf.to_container(td.params_sampler, resolve=True)
    seconds_per_knot = float(sampler_cfg.pop("seconds_per_knot"))
    sampler_cfg["ensemble_size"] = len(samples)
    params_sampler = hydra.utils.instantiate(sampler_cfg)

    sampled = _sample_params(
        params_sampler,
        seconds_per_knot=seconds_per_knot,
        simulation_time=float(td.simulation_time),
        seed=int(td.seed),
    )
    _plot_sampled_params(
        sampled=sampled,
        split_offsets=_split_specs(td),
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
    # Written last: its presence is what marks the plan complete.
    _to_netcdf_atomic(sampled, params_path)


def _simulate(
    cfg: DictConfig,
    groups: list[list[Sample]],
    spinups: list[float],
    sampled: xr.Dataset,
    output_dir: pathlib.Path,
) -> None:
    """Run the given geometry groups sequentially, skipping finished samples.

    A failing geometry is logged and skipped so it does not cost the rest of
    the shard's allocation; the stage still raises at the end, and a rerun
    retries exactly the samples that are missing.
    """
    model_name = cfg.model.name
    td = cfg.training_data
    geom_cfg = td.geometry
    source = str(geom_cfg.source)
    resolution = float(geom_cfg.resolution)

    regrid: Callable[[xr.Dataset], xr.Dataset] | None = None
    if model_name == "pyudales":
        from pyudales.utils.grid_utils import interpolate_grid

        regrid = interpolate_grid

    for split in ("train", "val", "test"):
        (output_dir / "state" / split).mkdir(parents=True, exist_ok=True)
        (output_dir / "param" / split).mkdir(parents=True, exist_ok=True)
    raw_root = output_dir / "_raw_states"
    raw_root.mkdir(exist_ok=True)
    base_temp_dir = pathlib.Path(cfg.paths.experiment_dir)

    save_vars = OmegaConf.select(cfg, "training_data.save_vars")
    save_vars = list(save_vars) if save_vars is not None else None
    encoding_cfg = OmegaConf.select(cfg, "training_data.state_encoding")
    failures: list[tuple[str, str]] = []
    n_run = 0
    t0 = _time.time()
    for group_num, (group, group_spinup) in enumerate(zip(groups, spinups)):
        geom = group[0].geom
        stl_path = geom.stl_path
        pending = [
            (j, s) for j, s in enumerate(group) if not _sample_done(output_dir, s)
        ]
        if not pending:
            print(
                f"[{group_num + 1}/{len(groups)}] {geom.name}: all "
                f"{len(group)} sim(s) already on disk, skipping"
            )
            continue
        dom_x, dom_y, dom_z = geom.domain_size
        print(
            f"[{group_num + 1}/{len(groups)}] {geom.name}: {len(pending)}/"
            f"{len(group)} sim(s) to run, "
            f"grid {geom.nx}x{geom.ny}x{geom.nz}, domain "
            f"{dom_x:g}x{dom_y:g}x{dom_z:g} m, fetch {-geom.bounds[0][0]:g} m / "
            f"wake {geom.bounds[0][1] - geom.lx:g} m, lateral "
            f"{-geom.bounds[1][0]:g}/{geom.bounds[1][1] - geom.ly:g} m "
            f"(STL {geom.lx:g}x{geom.ly:g} m, z_max {geom.z_max:g} m)"
        )
        if group_spinup != float(td.spinup_time):
            print(
                f"  spinup {group_spinup:g} s "
                f"({group_spinup / (dom_x / 10.0):.1f}x the 10 m/s transit)"
            )
        # A fresh scratch dir per geometry: sequential reuse of member
        # experiment dirs across different grids leaves stale solver outputs
        # behind (uDALES fielddumps especially). pylbm's build tree lives in
        # the shared LBM checkout, so this does not force full rebuilds.
        group_temp_dir = base_temp_dir / f"geom_{geom.name}"
        if group_temp_dir.exists():
            shutil.rmtree(group_temp_dir)
        group_temp_dir.mkdir(parents=True)
        # Per-group staging, cleaned here rather than wiping the whole
        # _raw_states root: other shards stage into it concurrently, and stale
        # state files would be picked up as warm starts.
        group_raw_dir = raw_root / geom.name
        if group_raw_dir.exists():
            shutil.rmtree(group_raw_dir)

        try:
            fm_overrides: dict = {
                "nx": geom.nx,
                "ny": geom.ny,
                "nz": geom.nz,
                "bounds": [list(b) for b in geom.bounds],
                "simulation_time": float(td.simulation_time),
                "output_frequency": float(td.output_frequency),
                "spinup_time": group_spinup,
                # save_on_disk mode: every run writes state_{j}.nc here.
                "results_dir": group_raw_dir,
                "temp_dir": str(group_temp_dir),
            }
            if model_name == "pyudales":
                template_dir = _resolve_path(geom_cfg.udales_case_dir)
                fm_overrides["case_dir"] = str(
                    _stage_udales_case(template_dir, stl_path, group_temp_dir)
                )
                # Precomputed IBM bundles are grid-specific; always voxelize
                # from the STL for pool geometries.
                fm_overrides["precomputed_geom_dir"] = None
            else:
                fm_overrides["stl_path"] = str(stl_path)
                if model_name == "pypalm":
                    fm_overrides["case_dir"] = str(
                        _resolve_path(geom_cfg.palm_case_dir)
                    )

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

            # Direct sequential single-model runs — no ensemble machinery
            # (member experiment-dir clones, process pools) for a per-geometry
            # batch that runs one simulation at a time anyway.
            for j, sample in pending:
                print(
                    f"  run {j + 1}/{len(group)}: {sample.split} sample "
                    f"{sample.local_idx}"
                )
                forward_model(
                    state=None,
                    params=sampled.isel(ensemble=sample.global_idx),
                    sim_name=f"state_{j}",
                )

            # Partition this group's members straight into the split layout.
            for j, sample in pending:
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
                # Per-sample, since adaptive sizing makes it vary across the
                # corpus.
                state.attrs["spinup_time_s"] = group_spinup

                # Adaptive-timestep solvers (uDALES) stamp outputs at the
                # actual solver time nearest the requested cadence, so the
                # time axis jitters per run: params are interpolated per
                # sample onto ITS OWN axis.
                state_time = np.asarray(state["time"].values)
                state_dst, param_dst = _sample_paths(output_dir, sample)
                saved = _select_saved_vars(state, save_vars)
                _to_netcdf_atomic(
                    saved,
                    state_dst,
                    encoding=_netcdf_encoding(saved, encoding_cfg),
                )
                src.unlink()

                member_params = (
                    _interpolate_params_to_state_time(
                        sampled.isel(ensemble=[sample.global_idx]), state_time
                    )
                    .isel(ensemble=0)
                    .drop_vars("ensemble")
                )
                _to_netcdf_atomic(member_params, param_dst)
                n_run += 1
                print(
                    f"[{sample.split}] sample {sample.local_idx + 1} "
                    f"({geom.name}) -> {state_dst}"
                )
        except Exception as err:
            traceback.print_exc()
            failures.append((geom.name, f"{type(err).__name__}: {err}"))
            print(f"FAILED geometry {geom.name}; continuing with the next one.")
            continue

        shutil.rmtree(group_temp_dir, ignore_errors=True)
        shutil.rmtree(group_raw_dir, ignore_errors=True)

    elapsed = _time.time() - t0
    print(
        f"Ran {n_run} simulation(s) in {elapsed:.1f}s "
        f"(~{elapsed / max(n_run, 1):.1f}s/simulation)"
    )
    if failures:
        listing = "\n".join(f"  {name}: {msg}" for name, msg in failures)
        raise RuntimeError(
            f"{len(failures)} geometry group(s) failed (samples already written "
            f"are kept; rerunning retries only the missing ones):\n{listing}"
        )


def _finalize(
    cfg: DictConfig,
    samples: list[Sample],
    sampled: xr.Dataset,
    output_dir: pathlib.Path,
    *,
    missing_hint: Callable[[list[Sample]], str],
) -> None:
    """Corpus-level outputs, once every planned sample is on disk.

    Everything here is recomputed from the saved files, which is what makes
    the sharded corpus identical to a one-process one: params.nc takes the time
    axis of the FIRST planned sample, as the in-process run always did.
    """
    missing = [s for s in samples if not _sample_done(output_dir, s)]
    if missing:
        listing = ", ".join(f"{s.split}/{s.local_idx:04d}" for s in missing[:20])
        more = f" (+{len(missing) - 20} more)" if len(missing) > 20 else ""
        raise RuntimeError(
            f"{len(missing)}/{len(samples)} samples are not on disk yet: "
            f"{listing}{more}. {missing_hint(missing)}"
        )

    first_state, _ = _sample_paths(output_dir, samples[0])
    with xr.open_dataset(first_state) as ds:
        reference_time = np.asarray(ds["time"].values)
    interpolated = _interpolate_params_to_state_time(sampled, reference_time)
    interpolated.to_netcdf(output_dir / "params.nc")
    print(f"Saved consolidated interpolated params -> {output_dir / 'params.nc'}")

    # A differing frame COUNT signals a truncated run.
    for s in samples[1:]:
        state_path, _ = _sample_paths(output_dir, s)
        with xr.open_dataset(state_path) as ds:
            n_frames = ds.sizes["time"]
        if n_frames != reference_time.shape[0]:
            print(
                f"WARNING: {state_path} ({s.geom.name}) has {n_frames} output "
                f"frames vs {reference_time.shape[0]} for the first sample; "
                "the run may have been truncated."
            )

    raw_root = output_dir / "_raw_states"
    if raw_root.exists():
        shutil.rmtree(raw_root, ignore_errors=True)

    # --- Visualization -----------------------------------------------------
    _plot_sampled_params(
        sampled=interpolated,
        split_offsets=_split_specs(cfg.training_data),
        output_path=output_dir / "params_interpolated.png",
    )
    print(
        f"Saved interpolated trajectories -> {output_dir / 'params_interpolated.png'}"
    )

    # The first sample of each split, as stored (save_vars + encoding applied).
    first_example: dict[str, xr.Dataset] = {}
    for s in samples:
        if s.split not in first_example:
            first_example[s.split] = xr.load_dataset(_sample_paths(output_dir, s)[0])
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


def run(cfg: DictConfig) -> None:
    model_name = cfg.model.name
    if model_name not in _SUPPORTED_MODELS:
        raise ValueError(
            f"model={model_name} is not supported for random-geometry data "
            f"generation; choose one of {_SUPPORTED_MODELS}."
        )

    td = cfg.training_data
    source = str(td.geometry.source)
    if source not in _POOL_SOURCES:
        raise ValueError(
            f"training_data.geometry.source={source!r} is a single-geometry "
            "case; use scripts/neural_surrogate/generate_training_data.py for "
            f"it, or set the source to one of {_POOL_SOURCES}."
        )
    if int(td.num_train) + int(td.num_val) + int(td.num_test) == 0:
        raise ValueError("training_data: num_train + num_val + num_test must be > 0")

    sharding = _resolve_sharding(cfg)
    # --- Output layout -----------------------------------------------------
    # Every stage must find the same dir, so a sharded run cannot fall back to
    # the per-job Hydra run dir.
    if td.output_dir is not None:
        output_dir = pathlib.Path(td.output_dir)
    elif sharding.stage == "all":
        output_dir = resolve_output_dir(cfg, "training_data")
    else:
        raise ValueError(
            "training_data.output_dir must be set explicitly for "
            f"training_data.sharding.stage={sharding.stage}: every stage and "
            "shard has to find the same plan."
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Training data dir: {output_dir} (stage: {sharding.stage})")

    if sharding.stage in ("all", "plan"):
        samples = _plan_samples(cfg)
        _write_plan(cfg, samples, output_dir)
    else:
        # Shards run the plan's frozen config, not whatever conf/ says now.
        cfg = _load_plan_config(cfg, output_dir)
        samples = _plan_samples(cfg)
        _check_geometry_manifest(output_dir / "geometries.csv", samples)

    groups = _group_samples(samples)
    sampled = xr.load_dataset(output_dir / "sampled_params.nc")
    spinups, costs, shard_of = _shard_layout(cfg, groups, sampled, sharding.num_shards)

    if sharding.stage == "plan":
        if sharding.num_shards > 1:
            _print_shard_summary(groups, costs, shard_of, sharding.num_shards)
        print(f"Plan written to {output_dir}")
        return

    if sharding.stage in ("all", "simulate"):
        mine = [i for i, k in enumerate(shard_of) if k == sharding.shard_index]
        print(
            f"Shard {sharding.shard_index}/{sharding.num_shards}: "
            f"{len(mine)}/{len(groups)} geometries, "
            f"{sum(len(groups[i]) for i in mine)} sims, "
            f"{100 * sum(costs[i] for i in mine) / (sum(costs) or 1.0):.1f}% of "
            "the total compute"
        )
        my_groups = [groups[i] for i in mine]
        _validate_ncpu(cfg, [s for g in my_groups for s in g])
        _simulate(cfg, my_groups, [spinups[i] for i in mine], sampled, output_dir)

    if sharding.stage in ("all", "finalize"):

        def _missing_hint(missing: list[Sample]) -> str:
            if sharding.num_shards == 1:
                return "Rerun the generation to resume."
            group_of = {g[0].geom.stl_path: i for i, g in enumerate(groups)}
            shards = sorted({shard_of[group_of[s.geom.stl_path]] for s in missing})
            return (
                f"With num_shards={sharding.num_shards} they belong to shard(s) "
                f"{shards}; resubmit those with stage=simulate, then finalize."
            )

        _finalize(cfg, samples, sampled, output_dir, missing_hint=_missing_hint)


@hydra.main(  # type: ignore[misc]
    version_base=None,
    config_path="../../conf",
    config_name="neural_surrogate/training_data",
)
def main(cfg: DictConfig) -> None:
    run(cfg)


if __name__ == "__main__":
    main()
