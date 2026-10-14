"""Run several CFD backends on the same case, each a few times, for a comparison.

Config: configs/model_comparison.yaml (forward.yaml plus the comparison keys).
The first step of workflows/compare_models.sh; compute_metrics.py and
visualize_comparison.py (next to this script) score and draw the run dir.

    python scripts/comparison/run_models.py 'models=[pyudales,pypalm]' replicas=2
    python scripts/comparison/run_models.py 'models=[pyudales,pypalm]' \
        +overrides.pypalm.model.forward_model.boundary_condition=inflow_outflow

Every model runs the same forward config (case, params, time, forward.*) with
its own `model` group, then its `replica_seed` keys for replica r (seed
`seed + r`), then `overrides.<label>` (any key, e.g. `time.spinup_time` or
`model.forward_model.ncpu`). Each run is `scripts/run_forward.py`'s `run`.
There is no truth: compute_metrics.py measures how far the models are apart
against how far a model's own replicas are apart.

Outputs, in `paths.results_dir` (the run dir):
  config.yaml                       the resolved comparison config
  models.yaml                       per label: model, solver_name, overrides
                                    (the per-label ones), and per replica its
                                    seed (null: the backend seeds itself), wall
                                    time in s and run dir
  runs/<label>/replica_<r>/         run_forward.py's output (config.yaml, state.nc,
                                    params.nc), as the backend wrote it
  fields/<label>/replica_<r>.nc     the state on the case grid: u, v, w (time, z,
                                    y, x) on the cell centres of `domain` (each
                                    backend's components colocated, then
                                    interpolated where its grid differs),
                                    `discard_time` dropped, NaN in the building
                                    cells (`solid`, the case STL voxelised on
                                    the grid); `fluid` (z, y, x), the cells the
                                    metrics use (the STL's building cells and
                                    their neighbours left out, the same for every
                                    model); `sensor_<u|v|w>` (time, sensor) at the
                                    case's obs and validation points, each
                                    component interpolated on its native grid
A rerun into the same run dir first removes the earlier runs, fields and
models.yaml.
"""

from __future__ import annotations

import pyurbanair.quiet_jax  # noqa: F401  (silences JAX CPU-fallback noise)

import copy
import pathlib
import shutil
import sys
import time
from typing import Any

import hydra
import numpy as np
import xarray
from evaluation.turbulence import (
    colocate_components,
    fluid_mask,
    on_grid,
    stl_solid_mask,
)
from hydra.core.global_hydra import GlobalHydra
from omegaconf import DictConfig, OmegaConf, open_dict

SCRIPTS = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPTS / "utils"))

from helper_functions import case_stl_path, sensor_series, sensor_sets  # noqa: E402
from inconsistency_check import check_config  # noqa: E402

CONFIGS = SCRIPTS.parent / "configs"
DIMS = ("z", "y", "x")


def run(cfg: DictConfig) -> None:
    labels = model_labels(cfg.models)
    if int(cfg.replicas) < 1:
        raise ValueError(f"replicas must be at least 1, got {cfg.replicas}")
    unknown = set(cfg.overrides or {}) - set(labels)
    if unknown:
        raise ValueError(f"overrides for {sorted(unknown)}, not in models {labels}")
    out = pathlib.Path(cfg.paths.results_dir)
    with open_dict(cfg):
        cfg.pop("model", None)  # each model brings its own
    runs = {
        label: [
            _model_config(cfg, label, name, r, out) for r in range(int(cfg.replicas))
        ]
        for label, name in labels.items()
    }
    for configs in runs.values():
        for model_cfg in configs:
            check_config(model_cfg, "forward")

    out.mkdir(parents=True, exist_ok=True)
    for name in ("runs", "fields"):
        shutil.rmtree(out / name, ignore_errors=True)
    (out / "models.yaml").unlink(missing_ok=True)
    OmegaConf.save(cfg, out / "config.yaml", resolve=True)

    forward = _load_run_forward()
    grid = case_grid(cfg)
    centres = [grid[d].values for d in DIMS]
    solid = stl_solid_mask(case_stl_path(cfg), *centres)
    fluid = fluid_mask(case_stl_path(cfg), *centres)
    sets = sensor_sets(cfg)
    models: dict[str, Any] = {}
    for label, configs in runs.items():
        first = configs[0]
        models[label] = {
            "model": labels[label],
            "solver_name": str(first.model.solver_name),
            "overrides": _flat(cfg.overrides.get(label, {})),
            "replicas": [],
        }
        for r, model_cfg in enumerate(configs):
            print(f"{label} replica {r}")
            tic = time.perf_counter()
            forward.run(model_cfg)
            seconds = time.perf_counter() - tic
            run_dir = pathlib.Path(model_cfg.paths.results_dir)
            with xarray.open_dataset(run_dir / "state.nc") as state:
                fields = to_case_grid(
                    state,
                    str(first.model.solver_name),
                    grid,
                    solid,
                    fluid,
                    float(cfg.discard_time),
                    sets,
                )
            path = out / "fields" / label / f"replica_{r}.nc"
            path.parent.mkdir(parents=True, exist_ok=True)
            fields.to_netcdf(path)
            models[label]["replicas"].append(
                {
                    "seed": _seed(cfg, model_cfg, r),
                    "wall_time_s": round(seconds, 1),
                    "run_dir": str(run_dir),
                }
            )
            # Written after every run, so a failed later run leaves a record.
            OmegaConf.save(OmegaConf.create(models), out / "models.yaml")
    print(f"Saved the comparison runs in {out}")


def model_labels(models: Any) -> dict[str, str]:
    """{label: model config name} from entries `name` or `label:name`."""
    pairs = [str(m).split(":", 1) if ":" in str(m) else [str(m)] * 2 for m in models]
    labels = dict(pairs)
    if len(labels) != len(pairs):
        raise ValueError(
            f"a label is listed twice in {list(models)}: run one backend twice as "
            "models=[label_a:name,label_b:name]"
        )
    if not labels:
        raise ValueError("list at least one model in `models`")
    return labels


def _model_config(
    cfg: DictConfig, label: str, name: str, replica: int, out: pathlib.Path
) -> DictConfig:
    """The forward config of one run: `cfg` with model `name`, the replica's
    seed keys and the label's overrides, writing to runs/<label>/replica_<r>."""
    model_cfg = copy.deepcopy(cfg)
    with open_dict(model_cfg):
        model_cfg.model = _model_node(name)
        backend = str(model_cfg.model.name)
        seed = int(cfg.seed) + replica
        for key, value in (cfg.replica_seed or {}).get(backend, {}).items():
            OmegaConf.update(model_cfg, key, _with_seed(value, seed), merge=True)
        for key, value in _flat(cfg.overrides.get(label, {})).items():
            OmegaConf.update(model_cfg, key, value, merge=True)
        model_cfg.paths.results_dir = str(out / "runs" / label / f"replica_{replica}")
    return model_cfg


def _model_node(name: str) -> DictConfig:
    """configs/model/<name>.yaml (interpolations unresolved), composed by Hydra.

    Inside the Hydra app the app's search path applies (so `--config-dir
    tests/configs` reaches the test models); called directly, configs/ alone.
    """
    overrides = [f"model={name}"]
    if GlobalHydra.instance().is_initialized():
        return hydra.compose("forward", overrides=overrides).model
    with hydra.initialize_config_dir(config_dir=str(CONFIGS), version_base=None):
        return hydra.compose("forward", overrides=overrides).model


def _with_seed(value: Any, seed: int) -> Any:
    """`value` with `{seed}` filled in; a bare "{seed}" becomes the int."""
    if not isinstance(value, str):
        return value
    return seed if value == "{seed}" else value.format(seed=seed)


def _seed(cfg: DictConfig, model_cfg: DictConfig, replica: int) -> int | None:
    """The replica's seed, or None when the backend has no seed keys."""
    keys = (cfg.replica_seed or {}).get(str(model_cfg.model.name))
    return int(cfg.seed) + replica if keys else None


def _flat(node: Any, prefix: str = "") -> dict[str, Any]:
    """A nested mapping as {dotted key: leaf value}."""
    if not hasattr(node, "items"):
        return {
            prefix: OmegaConf.to_object(node) if OmegaConf.is_config(node) else node
        }
    out: dict[str, Any] = {}
    for key, value in node.items():
        out |= _flat(value, f"{prefix}.{key}" if prefix else str(key))
    return out


def _load_run_forward() -> Any:
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "comparison_run_forward", SCRIPTS / "run_forward.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def case_grid(cfg: DictConfig) -> xarray.DataArray:
    """An empty (z, y, x) field on the cell centres of `domain`."""
    n = {"x": cfg.domain.nx, "y": cfg.domain.ny, "z": cfg.domain.nz}
    bounds = dict(zip(("x", "y", "z"), cfg.domain.bounds))
    coords = {
        d: bounds[d][0] + (np.arange(n[d]) + 0.5) * (bounds[d][1] - bounds[d][0]) / n[d]
        for d in DIMS
    }
    return xarray.DataArray(
        np.zeros([n[d] for d in DIMS], dtype=np.float32), dims=DIMS, coords=coords
    )


def to_case_grid(
    state: xarray.Dataset,
    solver_name: str,
    grid: xarray.DataArray,
    solid: np.ndarray,
    fluid: np.ndarray,
    discard_time: float,
    sets: dict[str, tuple],
) -> xarray.Dataset:
    """One run's state as compute_metrics.py reads it (see the module docstring)."""
    if "ensemble" in state.dims:
        state = state.isel(ensemble=0)
    state = state[["u", "v", "w"]]
    keep = state.time.values > discard_time + 1e-6
    state = state.isel(time=np.flatnonzero(keep)).load()
    if state.sizes["time"] < 2:
        raise ValueError(
            f"{state.sizes['time']} output frame(s) left after discard_time="
            f"{discard_time}: nothing to compare"
        )
    fields = {}
    for name, field in zip(("u", "v", "w"), colocate_components(state, solver_name)):
        # (time, z, y, x) by the first letter of each backend's dim names.
        spatial = [next(d for d in field.dims if d[0] == a) for a in DIMS]
        field = field.transpose("time", *spatial).rename(dict(zip(spatial, DIMS)))
        field = on_grid(field, grid)
        values = field.values.astype(np.float32)
        values[:, solid] = np.nan
        fields[name] = (("time", *DIMS), values)
    out = xarray.Dataset(
        fields
        | {
            "solid": (DIMS, solid.astype(np.int8)),
            "fluid": (DIMS, fluid.astype(np.int8)),
        },
        coords={"time": state.time.values.astype(float)}
        | {d: grid[d].values for d in DIMS},
    )
    series, names, xyz = [], [], []
    for name, points in sets.items():
        series.append(sensor_series(state, points, solver_name))
        names += [name] * len(points[0])
        xyz += list(zip(*points))
    sensors = xarray.concat(series, dim="sensor").transpose("component", "time", ...)
    for c, component in enumerate(("u", "v", "w")):
        out[f"sensor_{component}"] = (("time", "sensor"), sensors.values[c])
    xyz_array = np.array(xyz, dtype=float)
    return out.assign_coords(
        sensor_set=("sensor", names),
        sensor_x=("sensor", xyz_array[:, 0]),
        sensor_y=("sensor", xyz_array[:, 1]),
        sensor_z=("sensor", xyz_array[:, 2]),
    )


@hydra.main(  # type: ignore[misc, unused-ignore]
    version_base=None, config_path="../../configs", config_name="model_comparison"
)
def main(cfg: DictConfig) -> None:
    run(cfg)


if __name__ == "__main__":
    main()
