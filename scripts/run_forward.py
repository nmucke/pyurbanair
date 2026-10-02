"""Run the forward model (one member or an ensemble) over consecutive windows.

Config: configs/forward.yaml (the `forward` block, `params`, `model`).

    python scripts/run_forward.py
    python scripts/run_forward.py model=pylbm params=static_truth
    python scripts/run_forward.py forward.ensemble=true forward.rollout_steps=2

Runs 1 + `forward.rollout_steps` windows of `time.simulation_time` seconds; each
window starts from the previous one's last state, and time-varying parameters
are extrapolated into the next window.

Outputs, in `<paths.results_dir>/`:
    config.yaml
    state.nc    the field over all windows (an `ensemble` dim only for ensembles)
    params.nc   the sampled parameters (`time` dim if time-varying)
`state.nc` + `params.nc` are what `assimilation.truth_dir` expects.
"""

from __future__ import annotations

import pathlib
import sys

import hydra
import jax
import jax.numpy as jnp
import numpy as np
import xarray
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from inconsistency_check import check_config  # noqa: E402

from pyurbanair.config.hydra_helpers import clean_outputs  # noqa: E402


def run(cfg: DictConfig) -> None:
    check_config(cfg, "forward")
    is_ensemble = bool(cfg.forward.ensemble)
    num_members = int(cfg.ensemble.ensemble_size) if is_ensemble else 1
    num_windows = 1 + int(cfg.forward.rollout_steps)
    sim_time = float(cfg.time.simulation_time)

    out_dir = pathlib.Path(cfg.paths.results_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, out_dir / "config.yaml", resolve=True)

    sampler = instantiate(cfg.params)
    params = sampler.sample(num_members)
    rng_key = jax.random.PRNGKey(int(cfg.params.get("seed", 0)))

    model = instantiate(cfg.model.forward_model, results_dir=None)
    instantiate(cfg.model.prepare, forward_model=model)
    clean_outputs(cfg.model.name, model)
    if is_ensemble:
        model = instantiate(cfg.model.ensemble_model, forward_model=model)

    state = _initial_state(cfg.forward.initial_state, is_ensemble)
    states, all_params = [], []
    for w in tqdm(range(num_windows), desc="windows"):
        if w > 0 and "time" in params.dims:
            knot_times = np.asarray(sampler.time_coords)
            rng_key, key = jax.random.split(rng_key)
            params = sampler.extrapolate(
                params, jnp.asarray(knot_times) + sim_time, key
            ).assign_coords(time=knot_times)
        if is_ensemble:
            out = model.run_ensemble(params=params, state=state, sim_name="state")
            out = model.get_states() if out is None else out
        else:
            out = model(params=params.isel(ensemble=0), state=state)
        states.append(_shift_time(out, w * sim_time))
        all_params.append(_shift_time(params, w * sim_time))
        state = out.isel(time=-1)

    _concat(states).to_netcdf(out_dir / "state.nc")
    if "time" in params.dims:
        _concat(all_params).to_netcdf(out_dir / "params.nc")
    else:
        params.to_netcdf(out_dir / "params.nc")
    print(f"Saved outputs in {out_dir}")


def _initial_state(path: str | None, is_ensemble: bool) -> xarray.Dataset | None:
    """The last frame of a NetCDF state (a run's state.nc), or None to cold-start.

    A single run takes the file's first member; an ensemble run needs the file
    to hold one member per ensemble member.
    """
    if path is None:
        return None
    state = xarray.load_dataset(path)
    if "time" in state.dims:
        state = state.isel(time=-1)
    if not is_ensemble and "ensemble" in state.dims:
        state = state.isel(ensemble=0)
    return state


def _shift_time(ds: xarray.Dataset, offset: float) -> xarray.Dataset:
    """Move a window's local time axis onto the global one."""
    if "time" not in ds.dims:
        return ds
    return ds.assign_coords(time=ds.time.values.astype(float) + offset)


def _concat(pieces: list[xarray.Dataset]) -> xarray.Dataset:
    """Join windows along time; where two windows share an instant keep the later."""
    combined = xarray.concat(pieces, dim="time", join="override")
    _, last = np.unique(combined.time.values[::-1], return_index=True)
    return combined.isel(time=np.sort(combined.sizes["time"] - 1 - last))


@hydra.main(version_base=None, config_path="../configs", config_name="forward")  # type: ignore[misc, unused-ignore]
def main(cfg: DictConfig) -> None:
    run(cfg)


if __name__ == "__main__":
    main()
