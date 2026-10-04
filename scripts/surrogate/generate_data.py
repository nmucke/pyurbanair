"""Generate training data for the neural surrogates.

Config: configs/surrogate/generate_data.yaml.

    python scripts/surrogate/generate_data.py
    python scripts/surrogate/generate_data.py data.geometry.mode=fixed \
        'data.geometry.name=${case_name}'
    # one of four independent jobs:
    python scripts/surrogate/generate_data.py data.sharding.num_shards=4 \
        data.sharding.shard_index=0

One simulation per sample, each with its own inflow trajectory
(params/surrogate_training_data.yaml), on either
  * the case geometry (`data.geometry.mode=fixed`), or
  * a building layout drawn from the STL pool in `data.geometry.stl_dir`
    (`mode=random`), with a grid fitted around it. Val and test geometries are
    held out of train.
Samples on the same geometry share one prepared model and run one after the
other; shards split the geometries between independent jobs.

Resumable: a sample already on disk is skipped, so rerunning after a crash, or
with a larger `num_train`, only runs what is missing.

Outputs, in `<paths.results_dir>/`:
    config.yaml
    state/{train,val,test}/sample_XXXX.nc   the field on cell centres, plus the
                                            `blanking` mask (1 = building)
    param/{train,val,test}/sample_XXXX.nc   the parameters at the state times
"""

from __future__ import annotations

import pyurbanair.quiet_jax  # noqa: F401  (silences JAX CPU-fallback noise)

import csv
import dataclasses
import math
import os
import pathlib
import re
import shutil
import traceback
from typing import Any

import hydra
import jax
import numpy as np
import xarray
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from pyurbanair.config.hydra_helpers import clean_outputs, resolve_parameter_schema
from pyurbanair.dynamic_parameters import build_knot_times

SPLITS = ("train", "val", "test")


@dataclasses.dataclass(frozen=True)
class Geometry:
    """One pool STL with the grid fitted around it."""

    stl: pathlib.Path
    nx: int
    ny: int
    nz: int
    bounds: tuple[tuple[float, float], ...]  # ((x0, x1), (y0, y1), (0, z_size))


@dataclasses.dataclass(frozen=True)
class Sample:
    split: str
    index: int  # within the split
    member: int  # row in the sampled parameters
    geometry: Geometry | None  # None: the case geometry


def run(cfg: DictConfig) -> None:
    data = cfg.data
    out_dir = pathlib.Path(cfg.paths.results_dir)
    for split in SPLITS:
        (out_dir / "state" / split).mkdir(parents=True, exist_ok=True)
        (out_dir / "param" / split).mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, out_dir / "config.yaml", resolve=True)

    samples = _plan(cfg)
    params = _sample_params(cfg, len(samples))

    # Samples sharing a geometry run on one prepared model.
    groups: dict[Any, list[Sample]] = {}
    for sample in samples:
        groups.setdefault(sample.geometry, []).append(sample)
    shards, shard = int(data.sharding.num_shards), int(data.sharding.shard_index)
    failed = []
    for k, (geometry, group) in enumerate(groups.items()):
        if k % shards != shard:
            continue
        pending = [s for s in group if not _done(out_dir, s)]
        if not pending:
            continue
        name = geometry.stl.stem if geometry else cfg.case_name
        print(f"[{k + 1}/{len(groups)}] {name}: {len(pending)} sample(s)")
        try:
            spinup = _spinup(cfg, geometry, params, pending)
            model = _make_model(cfg, geometry, spinup)
            for sample in pending:
                member = params.isel(ensemble=sample.member)
                # Samples on one model would otherwise share uDALES's synthetic
                # inlet turbulence (its seed follows the experiment name).
                if hasattr(model, "inlet_turbulence"):
                    model.inlet_turbulence["seed"] = (
                        int(data.seed) * 100_003 + sample.member
                    )
                state = model(state=None, params=member)
                state.attrs.update(geometry=name, spinup_time_s=spinup)
                _save(cfg, model, state, member, out_dir, sample)
                print(f"  {sample.split} sample {sample.index}")
        except Exception:  # one bad geometry must not cost the rest of the job
            traceback.print_exc()
            failed.append(name)
    if failed:
        raise RuntimeError(
            f"{len(failed)} geometries failed (rerun to retry): {failed}"
        )
    print(f"Done. Training data in {out_dir}")


# ---------------------------------------------------------------------------
# Planning: which sample runs on which geometry, with which parameters
# ---------------------------------------------------------------------------


def _plan(cfg: DictConfig) -> list[Sample]:
    """All samples in train, val, test order. Deterministic in the config."""
    data = cfg.data
    counts = {
        "train": int(data.num_train),
        "val": int(data.num_val),
        "test": int(data.num_test),
    }
    if data.geometry.mode == "fixed":
        geometries: dict[str, list] = {split: [None] * n for split, n in counts.items()}
    else:
        pool = _geometry_pool(data.geometry)
        geometries = _split_geometries(
            pool, counts, np.random.default_rng(int(data.seed))
        )
    samples, member = [], 0
    for split in SPLITS:
        for index, geometry in enumerate(geometries[split]):
            samples.append(Sample(split, index, member, geometry))
            member += 1
    return samples


def _sample_params(cfg: DictConfig, n: int) -> xarray.Dataset:
    """Every sample's parameter trajectory, knots `seconds_per_knot` apart."""
    knots = np.asarray(
        build_knot_times(
            0.0, float(cfg.time.simulation_time), float(cfg.time.seconds_per_knot)
        )
    )
    sampler = instantiate(cfg.params, ensemble_size=n)
    sampled = sampler.sample_prior(knots, jax.random.PRNGKey(int(cfg.data.seed)))
    params: xarray.Dataset = sampled.assign_coords(time=knots)
    if "pressure_gradient_magnitude" in resolve_parameter_schema(cfg.model.name):
        params["pressure_gradient_magnitude"] = (
            "ensemble",
            np.full(n, float(cfg.data.pressure_gradient_magnitude)),
        )
    return params


def _geometry_pool(geom: DictConfig) -> list[Geometry]:
    """The pool's STLs, each with a grid at `resolution` fitted around it.

    The STL frame starts at the origin; the domain adds `upstream_padding` in
    front, `downstream_padding` behind and `lateral_padding` on both sides, and
    rounds nx and ny up to multiples of 16 (the slack goes behind and to the
    sides). Geometries taller than `z_size` are skipped.
    """
    stl_dir, r, z_size = (
        pathlib.Path(geom.stl_dir),
        float(geom.resolution),
        float(geom.z_size),
    )
    nz = round(z_size / r)
    # The pool's manifest.csv holds each layout's true domain size; the mesh
    # bounds under-span it when the outer buildings sit inset from the edge.
    sizes = {}
    if (stl_dir / "manifest.csv").exists():
        with open(stl_dir / "manifest.csv", newline="") as f:
            for row in csv.DictReader(f):
                sizes[row["name"]] = (
                    float(row["Lx_m"]),
                    float(row["Ly_m"]),
                    float(row["z_max_m"]),
                )

    def cells(length: float) -> int:
        return math.ceil(round(length / r, 6))

    front, back, side = (
        cells(float(geom[k]))
        for k in ("upstream_padding", "downstream_padding", "lateral_padding")
    )
    pool = []
    for stl in sorted(stl_dir.glob("*.stl")):
        if stl.stem in sizes:
            lx, ly, z_max = sizes[stl.stem]
        else:
            import trimesh

            lx, ly, z_max = (
                float(v) for v in trimesh.load_mesh(str(stl), process=False).bounds[1]
            )
        if z_max >= z_size:
            continue
        nx = math.ceil((cells(lx) + front + back) / 16) * 16
        ny = math.ceil((cells(ly) + 2 * side) / 16) * 16
        y_front = side + (ny - cells(ly) - 2 * side) // 2
        bounds = (
            (-front * r, (nx - front) * r),
            (-y_front * r, (ny - y_front) * r),
            (0.0, z_size),
        )
        pool.append(Geometry(stl, nx, ny, nz, bounds))
    if not pool:
        raise ValueError(f"No usable geometry in {stl_dir} (all taller than z_size?).")
    return pool


def _split_geometries(
    pool: list[Geometry], counts: dict[str, int], rng: np.random.Generator
) -> dict[str, list[Geometry]]:
    """Val and test each get distinct held-out geometries; train the rest,
    cycling through them when it needs more samples than geometries."""
    held_out = counts["val"] + counts["test"]
    if held_out >= len(pool):
        raise ValueError(
            f"num_val + num_test = {held_out} needs a pool larger than {len(pool)}."
        )
    order = [pool[i] for i in rng.permutation(len(pool))]
    val, test, train_pool = (
        order[: counts["val"]],
        order[counts["val"] : held_out],
        order[held_out:],
    )
    train = [train_pool[i % len(train_pool)] for i in range(counts["train"])]
    return {"train": train, "val": val, "test": test}


# ---------------------------------------------------------------------------
# Running
# ---------------------------------------------------------------------------


def _spinup(
    cfg: DictConfig,
    geometry: Geometry | None,
    params: xarray.Dataset,
    group: list[Sample],
) -> float:
    """Spin-up long enough for the inflow to cross the domain `fill_times` times.

    The interior starts at rest and fills from the inlet, so the spin-up scales
    with the domain length over the slowest streamwise inflow of the group
    (initial values, ~0.8 x speed for a 1/4 power-law profile). Never shorter
    than `time.spinup_time`, never longer than `max_spinup_time`.
    """
    base = float(cfg.time.spinup_time)
    adaptive = cfg.data.adaptive_spinup
    if not adaptive.enabled:
        return base
    (x0, x1) = geometry.bounds[0] if geometry else tuple(cfg.domain.bounds[0])
    slowest = min(
        0.8
        * float(params.velocity_magnitude.isel(ensemble=s.member, time=0))
        * max(
            math.cos(
                math.radians(float(params.inflow_angle.isel(ensemble=s.member, time=0)))
            ),
            0.1,
        )
        for s in group
    )
    needed = float(adaptive.fill_times) * (x1 - x0) / slowest
    step = float(cfg.time.output_frequency)
    return float(
        min(max(math.ceil(needed / step) * step, base), float(adaptive.max_spinup_time))
    )


def _make_model(cfg: DictConfig, geometry: Geometry | None, spinup: float) -> Any:
    """The forward model for one geometry (None: the case geometry)."""
    name = geometry.stl.stem if geometry else cfg.case_name
    scratch = pathlib.Path(cfg.paths.experiment_dir) / f"data_{name}"
    shutil.rmtree(scratch, ignore_errors=True)
    scratch.mkdir(parents=True)
    overrides: dict[str, Any] = {
        "results_dir": None,
        "temp_dir": str(scratch),
        "spinup_time": spinup,
    }
    if geometry is not None:
        overrides.update(
            nx=geometry.nx,
            ny=geometry.ny,
            nz=geometry.nz,
            bounds=[list(b) for b in geometry.bounds],
        )
        if cfg.model.name == "pyudales":
            overrides["case_dir"] = str(
                _udales_case(
                    pathlib.Path(cfg.data.geometry.case_dir),
                    geometry.stl,
                    scratch,
                )
            )
            overrides["precomputed_geom_dir"] = None
        else:
            overrides["stl_path"] = str(geometry.stl)
            if cfg.model.name == "pypalm":
                overrides["case_dir"] = str(cfg.data.geometry.case_dir)
    model = instantiate(cfg.model.forward_model, **overrides)
    instantiate(cfg.model.prepare, forward_model=model)
    clean_outputs(cfg.model.name, model)
    return model


def _udales_case(
    template: pathlib.Path, stl: pathlib.Path, scratch: pathlib.Path
) -> pathlib.Path:
    """A copy of the uDALES case template pointing at `stl`.

    uDALES reads its STL name from `stl_file` in namoptions, not from a model
    argument, so each geometry gets its own case copy.
    """
    case = scratch / "udales_case"
    shutil.copytree(template, case)
    shutil.copy2(stl, case / stl.name)
    for namoptions in case.glob("namoptions.*"):
        text = re.sub(
            r"(?m)^(\s*stl_file\s*=\s*).*$", rf"\g<1>{stl.name}", namoptions.read_text()
        )
        namoptions.write_text(text)
    return case


# ---------------------------------------------------------------------------
# Saving
# ---------------------------------------------------------------------------


def _paths(out_dir: pathlib.Path, sample: Sample) -> tuple[pathlib.Path, pathlib.Path]:
    name = f"sample_{sample.index:04d}.nc"
    return (
        out_dir / "state" / sample.split / name,
        out_dir / "param" / sample.split / name,
    )


def _done(out_dir: pathlib.Path, sample: Sample) -> bool:
    # Both files are written atomically, the parameters last.
    return all(p.exists() for p in _paths(out_dir, sample))


def _save(
    cfg: DictConfig,
    model: Any,
    state: xarray.Dataset,
    params: xarray.Dataset,
    out_dir: pathlib.Path,
    sample: Sample,
) -> None:
    """Write one sample: the state on cell centres and its parameters."""
    if cfg.model.name == "pyudales":
        from pyudales.utils.grid_utils import interpolate_grid

        state = interpolate_grid(state)  # staggered -> cell centres
        state["blanking"] = (("zt", "yt", "xt"), _udales_blanking(model, state))
    keep = set(cfg.data.save_vars)
    state = state.drop_vars(
        [v for v in state.data_vars if "time" in state[v].dims and v not in keep]
    )

    # Parameters at the state's times (the model interpolates between knots).
    times = state.time.values.astype(float)
    params = xarray.Dataset(
        {
            name: (
                ("time", np.interp(times, params.time.values, da.values))
                if "time" in da.dims
                else da
            )
            for name, da in params.drop_vars(
                "ensemble", errors="ignore"
            ).data_vars.items()
        },
        coords={"time": times},
    )

    state_path, param_path = _paths(out_dir, sample)
    _write_atomic(state, state_path, _encoding(state, cfg.data.state_encoding))
    _write_atomic(params, param_path, {})


def _udales_blanking(model: Any, state: xarray.Dataset) -> np.ndarray:
    """Building mask from uDALES's solid_c.txt (1-based i, j, k of solid cells).

    uDALES leaves small non-zero velocities inside buildings, so the mask
    cannot be recovered from the field.
    """
    idx = np.loadtxt(model.dirs.experiment_dir / "solid_c.txt", skiprows=1, dtype=int)
    mask = np.zeros(
        (state.sizes["zt"], state.sizes["yt"], state.sizes["xt"]), dtype=np.int8
    )
    mask[idx[:, 2] - 1, idx[:, 1] - 1, idx[:, 0] - 1] = 1
    return mask


def _encoding(state: xarray.Dataset, cfg: DictConfig) -> dict:
    """zlib compression; `least_significant_digit` (lossy) on float fields only."""
    base = dict(OmegaConf.to_container(cfg, resolve=True))  # type: ignore[arg-type, unused-ignore]
    digits = base.pop("least_significant_digit", None)
    return {
        name: {
            **base,
            **(
                {"least_significant_digit": digits}
                if digits is not None and da.dtype.kind == "f"
                else {}
            ),
        }
        for name, da in state.data_vars.items()
    }


def _write_atomic(ds: xarray.Dataset, path: pathlib.Path, encoding: dict) -> None:
    """Write to a hidden temporary name, then rename: `path` is complete or absent."""
    tmp = path.with_name(f".{path.name}.tmp")
    ds.to_netcdf(tmp, encoding=encoding)
    os.replace(tmp, path)


@hydra.main(  # type: ignore[misc, unused-ignore]
    version_base=None,
    config_path="../../configs",
    config_name="surrogate/generate_data",
)
def main(cfg: DictConfig) -> None:
    run(cfg)


if __name__ == "__main__":
    main()
