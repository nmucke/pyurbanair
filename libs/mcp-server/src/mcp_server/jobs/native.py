"""Explicit native-input coverage; edits apply only to private staged templates.

Only independently editable fields proven to survive wrapper construction and
member/window cloning are writable. Wrapper-managed fields direct callers to the
canonical Hydra setting. Unknown fields are rejected rather than silently lost.
"""

from __future__ import annotations

import importlib.util
import math
from pathlib import Path
from typing import Any

_MANAGED: dict[str, dict[str, str]] = {
    "pyudales": {
        "domain.itot": "domain.nx",
        "domain.jtot": "domain.ny",
        "domain.ktot": "domain.nz",
        "domain.xlen": "domain.bounds",
        "domain.ylen": "domain.bounds",
        "inps.zsize": "domain.bounds",
        "run.runtime": "time.simulation_time",
        "output.tfielddump": "time.output_frequency",
        "run.nprocx": "model.forward_model.ncpu",
        "run.nprocy": "model.forward_model.ncpu",
        "namsubgrid.cs": "model.forward_model.sgs_constant",
        "namsubgrid.c_vreman": "model.forward_model.sgs_constant",
        "inps.u0": "params",
        "inps.v0": "params",
        "inps.dpdx": "params",
        "inps.dpdy": "params",
        "inps.stl_file": "model.forward_model.case_dir",
        "run.iexpnr": "model.forward_model.experiment_name",
    },
    "pypalm": {
        "initialization_parameters.nx": "domain.nx",
        "initialization_parameters.ny": "domain.ny",
        "initialization_parameters.nz": "domain.nz",
        "initialization_parameters.dx": "domain.bounds",
        "initialization_parameters.dy": "domain.bounds",
        "initialization_parameters.dz": "domain.bounds",
        "runtime_parameters.end_time": "time.simulation_time",
        "runtime_parameters.dt_data_output": "time.output_frequency",
        "initialization_parameters.ug_surface": "params",
        "initialization_parameters.vg_surface": "params",
        "initialization_parameters.km_constant": "model.forward_model.sgs_constant",
        "initialization_parameters.bc_lr": "model.forward_model.boundary_condition",
        "initialization_parameters.bc_ns": "model.forward_model.boundary_condition",
    },
    "pylbm": {
        "nx": "domain.nx",
        "ny": "domain.ny",
        "nz": "domain.nz",
        "nt1": "time.simulation_time",
        "iout": "time.output_frequency",
        "smagor": "model.forward_model.sgs_constant",
        "lturb": "model.forward_model.inlet_turbulence",
        "uvel": "params",
    },
    "neural_surrogate": {},
}
_EDITABLE: dict[str, dict[str, str]] = {
    "pyudales": {"run.dtmax": "s", "run.courant": "dimensionless"},
    "pypalm": {
        "runtime_parameters.dt_max": "s",
        "initialization_parameters.cfl_factor": "dimensionless",
    },
    "pylbm": {},
    "neural_surrogate": {},
}


def _editor(repo_root: Path, backend: str, path: Path) -> Any:
    # Loading these standalone stdlib-only helpers by filename avoids package
    # __init__ hooks that fetch and build native solver sources.
    if backend == "pyudales":
        filename, class_name = "namoptions_utils.py", "NamoptionsFile"
    elif backend == "pypalm":
        filename, class_name = "p3d_utils.py", "P3DFile"
    else:
        raise ValueError(f"{backend}: no editable native template")
    source = repo_root / "libs" / backend / "src" / backend / "utils" / filename
    spec = importlib.util.spec_from_file_location(f"_forward_native_{backend}", source)
    if spec is None or spec.loader is None:
        raise ValueError(f"native editor unavailable: {source}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return getattr(module, class_name)(path)


def native_coverage(backend: str) -> dict[str, Any]:
    if backend not in _MANAGED:
        raise ValueError(f"unknown backend {backend!r}")
    entries = [
        {"field": key, "classification": "wrapper-managed", "hydra_field": value}
        for key, value in _MANAGED[backend].items()
    ]
    entries += [
        {
            "field": key,
            "classification": "independently-editable",
            "type": "positive number",
            "units": units,
        }
        for key, units in _EDITABLE[backend].items()
    ]
    return {
        "backend": backend,
        "fields": entries,
        "unlisted_fields": "unsupported",
        "format": "positional infile.in" if backend == "pylbm" else "namelist",
    }


def apply_native_overrides(
    config: dict[str, Any], overrides: dict[str, Any] | None, repo_root: str | Path
) -> list[dict[str, Any]]:
    if not overrides:
        return []
    backend = config["model"]["name"]
    flat: dict[str, Any] = {}
    for key, value in overrides.items():
        if isinstance(value, dict):
            flat.update(
                {f"{key}.{child}".lower(): item for child, item in value.items()}
            )
        else:
            flat[key.lower()] = value
    for key, value in flat.items():
        if key in _MANAGED.get(backend, {}):
            canonical = _MANAGED[backend][key]
            raise ValueError(
                f"native_overrides.{key}: wrapper-managed; use {canonical}"
            )
        if key not in _EDITABLE.get(backend, {}):
            raise ValueError(
                f"native_overrides.{key}: unsupported by the {backend} wrapper"
            )
        if (
            isinstance(value, bool)
            or not isinstance(value, (float, int))
            or not math.isfinite(value)
            or value <= 0
        ):
            raise ValueError(
                f"native_overrides.{key}: requires a finite positive number"
            )
        if key.endswith(("courant", "cfl_factor")) and value > 1:
            raise ValueError(f"native_overrides.{key}: must be at most 1")
    directory = Path(config["model"]["forward_model"]["case_dir"])
    pattern = "namoptions*" if backend == "pyudales" else "*_p3d"
    templates = sorted(directory.glob(pattern))
    if not templates:
        raise ValueError(f"native_overrides: no {pattern} template in {directory}")
    edits = []
    for path in templates:
        editor = _editor(Path(repo_root), backend, path)
        for key, value in flat.items():
            section, field = key.split(".", 1)
            section = section.upper() if backend == "pyudales" else section
            editor.set_value(section, field, value)
            edits.append({"path": str(path), "field": key, "value": value})
        editor.write()
    return edits
