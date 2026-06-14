"""Run discovery and data loading for the figure pipeline.

Everything that touches disk lives here so the driver scripts stay declarative.

Key conventions (see ``docs/figure_specs.md`` and ``figures/NOTES.md``):

* **Truth** is always the canonical wide uDALES run -- the per-run
  ``truth_access.yaml`` paths are stale (runs were produced on several systems),
  so :func:`open_truth` ignores them and uses :data:`TRUTH_STATE`.
* Each model stores the ensemble-mean speed as ``vel_mean`` (truth: precomputed
  ``vel_magnitude``). We compare those scalar |U| fields, interpolated onto the
  truth grid, which sidesteps the per-model staggered-grid differences.
* PALM runs use ``nz32`` (33 vertical levels, ``z`` 0.5..32.5) while the others
  use ``nz16``. Interpolating PALM fields onto the truth's 16 physical z-levels
  makes the discrepancy vanish -- so we treat PALM as nz16, as requested.
"""
from __future__ import annotations

import dataclasses
import pathlib
import re
from functools import lru_cache

import numpy as np
import xarray as xr
import yaml

# ---------------------------------------------------------------------------
# Canonical locations
# ---------------------------------------------------------------------------
DATA_ROOT = pathlib.Path("/projects/prjs2075/urbanair")
BLOCK_A_ROOT = DATA_ROOT / "assim_from_ground_truth"
BLOCK_B_ROOT = DATA_ROOT / "assim_with_state"
TRUTH_DIR = DATA_ROOT / "ground_truth_pyudales_wide" / "pyudales_time_varying"
TRUTH_STATE = TRUTH_DIR / "state.nc"
TRUTH_PARAMS = TRUTH_DIR / "params.nc"

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
STL_PATH = REPO_ROOT / "examples" / "xie_and_castro" / "xie_castro_2008_STL.stl"

PARAMS = ("inflow_angle", "velocity_magnitude")  # pressure_gradient absent on disk

# model prefix -> short key
_MODEL_KEY = {"pyudales": "udales", "pypalm": "palm", "pylbm": "lbm"}

_RUN_RE = re.compile(
    r"^(py\w+?)_nx(\d+)_ny(\d+)_nz(\d+)_ens(\d+)_steps(\d+)_int([\d.]+)(?:_(.+))?$"
)


# ---------------------------------------------------------------------------
# Run descriptor
# ---------------------------------------------------------------------------
@dataclasses.dataclass
class Run:
    path: pathlib.Path
    model: str            # udales | palm | lbm
    nx: int
    ny: int
    nz: int               # raw on-disk nz (palm=32); see nz_eff
    ens: int
    steps: int
    interval: float
    suffix: str | None    # method suffix for Block B, else None
    method: str | None    # canonical method key (Block B) or "param_only"

    # ---- identity helpers -------------------------------------------------
    @property
    def name(self) -> str:
        return self.path.name

    @property
    def nz_eff(self) -> int:
        """PALM nz32 is treated as nz16 per the data convention."""
        return 16 if self.model == "palm" else self.nz

    @property
    def dx(self) -> float:
        # domain x-extent is 60 m (=[-20,40]); cells = nx
        return 60.0 / self.nx

    @property
    def res_label(self) -> str:
        return f"{self.nx}x{self.ny}x{self.nz_eff}"

    @property
    def res_label_dx(self) -> str:
        return f"Δx={self.dx:.2g} m"

    # ---- artifact presence ------------------------------------------------
    def has(self, fname: str) -> bool:
        return (self.path / fname).exists()

    @property
    def complete(self) -> bool:
        """A run is plottable once it has its summary + posterior mean state."""
        return self.has("run_summary.yaml") and self.has("posterior_state_mean.nc")


def parse_run_name(name: str) -> dict | None:
    m = _RUN_RE.match(name)
    if not m:
        return None
    prefix, nx, ny, nz, ens, steps, interval, suffix = m.groups()
    model = _MODEL_KEY.get(prefix)
    if model is None:
        return None
    return dict(
        model=model, nx=int(nx), ny=int(ny), nz=int(nz), ens=int(ens),
        steps=int(steps), interval=float(interval), suffix=suffix,
    )


# Block B suffix -> canonical method key (spec §1).
_BLOCK_B_METHOD = {
    "localization_corr_ic": "ic_corr",
    "localization_dist_dist_ic": "ic_dist",
    "svd_ic": "ic_red",
    "svd_all": "icstate_red",
}


def discover_block_a(root: pathlib.Path = BLOCK_A_ROOT) -> list[Run]:
    """All parameter-only model x resolution runs (one method: param_only)."""
    runs = []
    for d in sorted(root.glob("py*_nx*")):
        if not d.is_dir():
            continue
        info = parse_run_name(d.name)
        if info is None:
            continue
        runs.append(Run(path=d, suffix=info["suffix"], method="param_only",
                        **{k: info[k] for k in
                           ("model", "nx", "ny", "nz", "ens", "steps", "interval")}))
    return runs


def discover_block_b(root: pathlib.Path = BLOCK_B_ROOT,
                     min_ens: int = 16) -> list[Run]:
    """State-estimation uDALES runs (ens128 production; skips ens4 smoke tests)."""
    runs = []
    for d in sorted(root.glob("py*_nx*")):
        if not d.is_dir():
            continue
        info = parse_run_name(d.name)
        if info is None or info["ens"] < min_ens:
            continue
        method = _BLOCK_B_METHOD.get(info["suffix"] or "")
        runs.append(Run(path=d, suffix=info["suffix"], method=method,
                        **{k: info[k] for k in
                           ("model", "nx", "ny", "nz", "ens", "steps", "interval")}))
    return runs


# ---------------------------------------------------------------------------
# Small artifact loaders
# ---------------------------------------------------------------------------
def load_yaml(path: pathlib.Path) -> dict:
    path = pathlib.Path(path)
    if not path.exists():
        return {}
    with open(path) as f:
        return yaml.safe_load(f) or {}


def load_summary(run: Run) -> dict:
    return load_yaml(run.path / "run_summary.yaml")


def load_config(run: Run) -> dict:
    return load_yaml(run.path / "config.yaml")


def load_params(run: Run, kind: str) -> xr.Dataset | None:
    """kind in {prior, posterior, true}."""
    fname = {"prior": "prior_params.nc", "posterior": "posterior_params.nc",
             "true": "true_params.nc"}[kind]
    p = run.path / fname
    return xr.open_dataset(p) if p.exists() else None


def load_state_mean(run: Run) -> xr.Dataset | None:
    p = run.path / "posterior_state_mean.nc"
    return xr.open_dataset(p) if p.exists() else None


# ---------------------------------------------------------------------------
# Truth (canonical wide uDALES run)
# ---------------------------------------------------------------------------
@lru_cache(maxsize=1)
def open_truth() -> xr.Dataset:
    return xr.open_dataset(TRUTH_STATE)


@lru_cache(maxsize=1)
def open_truth_params() -> xr.Dataset:
    return xr.open_dataset(TRUTH_PARAMS)


# ---------------------------------------------------------------------------
# Standardized |U| field: dims (time, z, y, x) with physical coords
# ---------------------------------------------------------------------------
_RENAME = {
    # any model's speed field lives on the u-grid; unify the dim names
    "zt": "z", "zm": "z", "z": "z",
    "yt": "y", "ym": "y", "y": "y",
    "xm": "x", "xt": "x", "xu": "x", "x": "x",
}


def velmag_field(ds: xr.Dataset, *, prefer: str = "vel_mean") -> xr.DataArray:
    """Return |U| as a DataArray with dims (time, z, y, x) and physical coords.

    Uses the precomputed ``vel_mean`` (model runs) or ``vel_magnitude`` (truth)
    when available, else builds it from u/v/w element-wise (matching how the
    pipeline forms speed elsewhere).
    """
    var = None
    for cand in (prefer, "vel_mean", "vel_magnitude"):
        if cand in ds.data_vars:
            var = cand
            break
    if var is not None:
        da = ds[var]
    else:
        if not all(v in ds.data_vars for v in ("u", "v", "w")):
            raise KeyError("dataset has neither a speed field nor u/v/w")
        da = np.sqrt(ds["u"] ** 2 + ds["v"] ** 2 + ds["w"] ** 2)
        da = da.rename("vel_magnitude")

    rename = {d: _RENAME[d] for d in da.dims if d in _RENAME}
    da = da.rename(rename)
    # drop any stray non-dim coords that would confuse interp
    keep = {"time", "z", "y", "x"}
    da = da.reset_coords([c for c in da.coords if c not in keep and c not in da.dims],
                         drop=True)
    return da


@lru_cache(maxsize=1)
def truth_velmag() -> xr.DataArray:
    """Truth |U| field, standardized dims (time, z, y, x)."""
    return velmag_field(open_truth(), prefer="vel_magnitude")


def truth_grid() -> dict:
    """The common evaluation grid (truth coords) as 1-D arrays."""
    tv = truth_velmag()
    return {k: np.asarray(tv[k].values) for k in ("z", "y", "x")}


def interp_to_truth(da: xr.DataArray, *, with_time: xr.DataArray | None = None
                    ) -> xr.DataArray:
    """Interpolate a standardized model field onto the truth (z, y, x) grid.

    PALM's 33 z-levels collapse onto the truth's 16 here (treat-as-nz16).
    If ``with_time`` is given, also interpolate the truth onto that time axis.
    """
    g = truth_grid()
    # Linear interp; points outside a model's grid become NaN (excluded from
    # metrics). The interior covers the full domain for every resolution.
    out = da.interp(z=g["z"], y=g["y"], x=g["x"])
    return out


def align_truth_time(truth_da: xr.DataArray, model_da: xr.DataArray) -> xr.DataArray:
    """Interpolate truth |U| onto a model field's time axis (nearest physical t)."""
    if "time" not in truth_da.dims or "time" not in model_da.dims:
        return truth_da
    return truth_da.interp(time=model_da["time"].values,
                           kwargs={"fill_value": "extrapolate"})


# ---------------------------------------------------------------------------
# Time / window helpers
# ---------------------------------------------------------------------------
def window_edges(run: Run) -> np.ndarray | None:
    cfg = load_summary(run).get("configuration", {}) or {}
    nwin = cfg.get("num_assimilation_windows")
    final = cfg.get("final_time")
    if not nwin or final is None:
        return None
    return np.linspace(0.0, float(final), int(nwin) + 1)


def param_window_edges(time: np.ndarray, nwin: int | None) -> np.ndarray | None:
    if not nwin or nwin < 2:
        return None
    return np.linspace(float(np.min(time)), float(np.max(time)), int(nwin) + 1)


# ---------------------------------------------------------------------------
# Sensors (from config.yaml; spec §3 / §10.4)
# ---------------------------------------------------------------------------
def sensor_coords(cfg: dict, kind: str = "assimilation") -> np.ndarray | None:
    """Return (n, 3) array of (x, y, z) sensor coordinates, or None.

    kind in {assimilation, validation}.
    """
    obs = cfg.get("obs", {}) or {}
    if kind == "assimilation":
        keys = ("x_points", "y_points", "z_points")
    else:
        keys = ("validation_x_points", "validation_y_points", "validation_z_points")
    if not all(k in obs and obs[k] is not None for k in keys):
        return None
    x, y, z = (np.asarray(obs[k], dtype=float) for k in keys)
    if len(x) == 0:
        return None
    return np.column_stack([x, y, z])


def sensor_timeseries(da: xr.DataArray, xyz: np.ndarray) -> np.ndarray:
    """Sample a standardized (time, z, y, x) |U| field at sensor points.

    Returns ``(n_sensors, n_time)`` via trilinear interpolation on the field grid.
    """
    xs = xr.DataArray(xyz[:, 0], dims="sensor")
    ys = xr.DataArray(xyz[:, 1], dims="sensor")
    zs = xr.DataArray(xyz[:, 2], dims="sensor")
    # fill_value=None => RegularGridInterpolator extrapolates (sensors are
    # inside the domain, but z=2 m can sit just below the lowest cell centre).
    sampled = da.interp(x=xs, y=ys, z=zs, kwargs={"fill_value": None})
    # -> dims (time, sensor); return (sensor, time)
    if "time" in sampled.dims:
        sampled = sampled.transpose("sensor", "time")
    return np.asarray(sampled.values, dtype=float)
