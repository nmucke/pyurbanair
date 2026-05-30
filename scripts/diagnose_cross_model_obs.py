"""Diagnostic: compare uDALES vs LBM observations at identical parameters.

Runs both backends at the SAME steady parameters and diffs their observation
vectors (same obs config / sensor coords). This isolates the model-to-model
mismatch that makes cross-model ESMDA diverge.
"""

import os
import pathlib
import sys

import numpy as np
import xarray

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from data_assimilation.observation_operator import ObservationOperator
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate

from pyurbanair.config.hydra_helpers import (
    create_observation_points,
    create_true_params,
    resolve_parameter_schema,
)


def run_model(cfg, role: str) -> xarray.Dataset:
    model = instantiate(cfg[role].forward_model)
    instantiate(cfg[role].prepare, forward_model=model)
    params = create_true_params(
        cfg[role].name, cfg.params.true, resolve_parameter_schema(cfg[role].name)
    )
    print(f"[{role}] {cfg[role].name} params: {dict(params.data_vars.items())}")
    state = model(params=params)
    if state is None:
        raise RuntimeError("expected in-memory state")
    return state


def raw_obs(cfg, state: xarray.Dataset, solver_name: str) -> np.ndarray:
    """Per-(state, sensor) interpolated values at the last timestep, shape (n_states, n_sensors).

    The base ObservationOperator reduces to the last timestep and returns a
    flat state-major vector [u_s0..u_sN, v_s0..v_sN, w_s0..w_sN].
    """
    ox, oy, oz = create_observation_points(cfg.obs)
    op = ObservationOperator(
        obs_x=ox.tolist(),
        obs_y=oy.tolist(),
        obs_z=oz.tolist(),
        obs_states=list(cfg.obs.states),
        solver_name=solver_name,
    )
    vec = np.asarray(op(state))
    return vec.reshape(len(cfg.obs.states), len(ox))


def main() -> None:
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

    ud_state = run_model(cfg, "truth_model")  # pyudales
    lb_state = run_model(cfg, "assim_model")  # pylbm

    ud = raw_obs(cfg, ud_state, cfg.truth_model.solver_name)
    lb = raw_obs(cfg, lb_state, cfg.assim_model.solver_name)
    states = list(cfg.obs.states)
    ox, oy, oz = create_observation_points(cfg.obs)

    print("\nobs array shapes (state, sensor):", ud.shape, lb.shape)
    print(f"sensors (x,y,z): {list(zip(ox.tolist(), oy.tolist(), oz.tolist()))}\n")

    for si, s in enumerate(states):
        print(f"== {s} at each sensor (last timestep) ==")
        print("  uDALES:", np.array2string(ud[si], precision=3, floatmode="fixed"))
        print("  LBM   :", np.array2string(lb[si], precision=3, floatmode="fixed"))
        print(
            f"  uDALES |mean|={np.abs(ud[si]).mean():.3f}  "
            f"LBM |mean|={np.abs(lb[si]).mean():.3f}  "
            f"ratio(LBM/uDALES)={np.abs(lb[si]).mean()/max(np.abs(ud[si]).mean(),1e-9):.2f}"
        )
    # Overall
    print("\n== overall obs vector ==")
    print(
        f"  uDALES RMS={np.sqrt((ud**2).mean()):.3f}  LBM RMS={np.sqrt((lb**2).mean()):.3f}"
    )
    print(f"  RMS(uDALES-LBM)={np.sqrt(((ud-lb)**2).mean()):.3f}")
    flat_u, flat_l = ud.ravel(), lb.ravel()
    print(f"  correlation={np.corrcoef(flat_u, flat_l)[0,1]:.3f}")

    # Domain-mean horizontal speed profile (sanity on bulk flow)
    print("\n== domain-mean speed by height (bulk flow) ==")
    for name, st in [("uDALES", ud_state), ("LBM", lb_state)]:
        s = st.isel(time=-1)
        u = s["u"]
        v = s["v"]
        spd = np.sqrt(u.values**2 + v.values**2)
        zdim = [d for d in u.dims if d.startswith("z")][0]
        prof = spd.mean(axis=tuple(i for i, d in enumerate(u.dims) if d != zdim))
        zc = st[zdim].values
        print(f"  {name:7s} z={np.array2string(zc, precision=1, floatmode='fixed')}")
        print(
            f"  {name:7s} |U|={np.array2string(prof, precision=2, floatmode='fixed')}"
        )


if __name__ == "__main__":
    main()
