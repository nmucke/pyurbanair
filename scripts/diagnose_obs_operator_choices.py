"""Diagnostic: how observation-operator choices affect cross-model representation error.

Runs uDALES and LBM at the SAME steady parameters, extracts the full
(time, sensor) series for u/v/w at each sensor, and measures the cross-model
discrepancy under different operator choices:
  * temporal aggregation window (instantaneous -> longer mean)
  * component subset (uvw / uv / u / horizontal speed)
  * sensor subset (upstream free-stream vs in-canopy)

Goal: find observation functionals that depend on the inflow parameters but are
robust to the structural (wake/turbulence) differences between the two solvers.
"""

import os
import pathlib
import sys

import numpy as np
import xarray

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from data_assimilation.interpolation import interpolate_dataarray_at_points
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate

from pyurbanair.config.hydra_helpers import (
    create_observation_points,
    create_true_params,
    resolve_parameter_schema,
)

DIMMAP = {
    "udales": {
        "u": {"z": "zt", "y": "yt", "x": "xm"},
        "v": {"z": "zt", "y": "ym", "x": "xt"},
        "w": {"z": "zm", "y": "yt", "x": "xt"},
    },
    "pylbm": {c: {"z": "z", "y": "y", "x": "x"} for c in ("u", "v", "w")},
}


def run_model(cfg, role):
    model = instantiate(cfg[role].forward_model)
    instantiate(cfg[role].prepare, forward_model=model)
    params = create_true_params(
        cfg[role].name, cfg.params.true, resolve_parameter_schema(cfg[role].name)
    )
    state = model(params=params)
    if state is None:
        raise RuntimeError("expected in-memory state")
    return state


def series(state, solver, comp, ox, oy, oz):
    """(time, sensor) interpolated series for one component."""
    m = DIMMAP[solver][comp]
    da = interpolate_dataarray_at_points(
        state[comp],
        x_dim=m["x"],
        y_dim=m["y"],
        z_dim=m["z"],
        obs_x=ox,
        obs_y=oy,
        obs_z=oz,
    )
    return np.asarray(da.transpose("time", "sensor").values)  # (T, S)


def block_mean(a, w):
    """Mean over consecutive time blocks of width w. a: (T, S) -> (T//w, S)."""
    T = a.shape[0]
    nb = T // w
    return a[: nb * w].reshape(nb, w, a.shape[1]).mean(axis=1)


def rms(x):
    return float(np.sqrt(np.mean(np.square(x))))


def main():
    with initialize_config_dir(version_base=None, config_dir=os.path.abspath("conf")):
        cfg = compose(
            config_name="config",
            overrides=[
                "size=small",
                "model@truth_model=pyudales",
                "model@assim_model=pylbm",
                "params.true.inflow_angle=41.0",
                "params.true.velocity_magnitude=8.2",
            ],
        )

    ud_state = run_model(cfg, "truth_model")
    lb_state = run_model(cfg, "assim_model")
    ox, oy, oz = create_observation_points(cfg.obs)

    ud = {c: series(ud_state, "udales", c, ox, oy, oz) for c in ("u", "v", "w")}
    lb = {c: series(lb_state, "pylbm", c, ox, oy, oz) for c in ("u", "v", "w")}
    T = ud["u"].shape[0]
    print(f"\nsensors: {[(round(x,1),round(y,1)) for x,y in zip(ox,oy)]}")
    print(f"timesteps: {T}\n")

    UPSTREAM = [0]  # x=-10, open inflow region
    ALL = list(range(len(ox)))

    def build(comps, sensors, w, derived=None):
        """Aggregated vectors (uDALES, LBM) for a given operator choice."""
        uu, ll = [], []
        if derived == "speed":
            su = block_mean(np.hypot(ud["u"], ud["v"])[:, sensors], w)
            sl = block_mean(np.hypot(lb["u"], lb["v"])[:, sensors], w)
            return su.ravel(), sl.ravel()
        for c in comps:
            uu.append(block_mean(ud[c][:, sensors], w).ravel())
            ll.append(block_mean(lb[c][:, sensors], w).ravel())
        return np.concatenate(uu), np.concatenate(ll)

    def report(label, comps, sensors, w, derived=None):
        u, l = build(comps, sensors, w, derived)
        disc = rms(u - l)
        rel = disc / max(rms(u), 1e-9)
        print(
            f"  {label:42s} absRMSdisc={disc:6.3f}  signalRMS={rms(u):6.3f}  rel={rel:5.1%}"
        )

    print("=== A. Temporal aggregation (components uvw, all sensors) ===")
    for w in [1, 3, 6, T]:
        report(f"window={w} step(s)", ("u", "v", "w"), ALL, w)

    print("\n=== B. Component subset (full-record time mean, all sensors) ===")
    report("u,v,w", ("u", "v", "w"), ALL, T)
    report("u,v (drop vertical w)", ("u", "v"), ALL, T)
    report("u only (streamwise)", ("u",), ALL, T)
    report("horizontal speed sqrt(u^2+v^2)", ("u", "v"), ALL, T, derived="speed")

    print("\n=== C. Sensor subset (full-record time mean) ===")
    report("u,v ALL sensors", ("u", "v"), ALL, T)
    report("u,v UPSTREAM (free-stream) only", ("u", "v"), UPSTREAM, T)
    report("speed UPSTREAM only", ("u", "v"), UPSTREAM, T, derived="speed")

    print("\n=== D. Best-case combo vs current operator ===")
    report("CURRENT: uvw, all, window=3", ("u", "v", "w"), ALL, 3)
    report("ROBUST: speed+uv, upstream, full-mean", ("u", "v"), UPSTREAM, T)


if __name__ == "__main__":
    main()
