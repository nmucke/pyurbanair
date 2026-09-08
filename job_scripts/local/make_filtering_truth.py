"""Generate a shared ground-truth artifact for the filtering benchmark campaign.

``scripts/run_forward_model.py`` only persists ``state.nc``/``params.nc`` when the
params sampler is time-varying (``is_dynamic_params = "time" in params.coords``),
so it cannot produce a STATIC truth. This script fills that gap: it mirrors the
inline-truth branch of ``scripts/filtering/run_filtering.py`` exactly -- same
config tree, same sampler filtering, same ``simulation_time=final_time`` -- and
writes the result as a ``state.nc``/``params.nc`` pair that every run of the
campaign then shares via ``run.truth_dir``.

``params.nc`` is saved as ``truth_sampler.sample(1)`` returns it (``ensemble``
dim retained), so the disk branch's ``xarray.load_dataset(params.nc)`` yields the
same object the inline branch builds -- the two truth paths stay equivalent.

The horizon is ``time.simulation_time * filtering.num_cycles``, taken from the
same overrides the campaign runs use, so the truth is long enough by
construction. Pass the campaign's overrides verbatim, e.g.::

    pixi run -e cuda python job_scripts/local/make_filtering_truth.py \
        case=xie_and_castro model@truth_model=pyudales \
        params@truth_params=static_truth params@prior_params=static \
        domain.nz=24 time.simulation_time=60.0 time.output_frequency=5.0 \
        filtering.num_cycles=60 \
        truth.out_dir=.temp/filtering_state_reduction_benchmark/_truth

Writes ``state.nc``, ``params.nc`` and ``truth_info.yaml`` (horizon, frame count,
resolved domain/time subtree and SHA-256 checksums of both files -- the campaign
record's "shared truth artifact path and checksum" row).
"""

import hashlib
import pathlib
import sys
import time

import hydra
import xarray
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from pyurbanair.config.hydra_helpers import clean_outputs, filter_parameter_config

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from scripts.esmda._esmda_common import write_yaml


def _sha256(path: pathlib.Path) -> str:
    """Checksum a (possibly multi-GB) NetCDF file without loading it."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def run(cfg: DictConfig) -> None:
    num_cycles = int(cfg.filtering.num_cycles)
    sim_time = float(cfg.time.simulation_time)
    final_time = sim_time * num_cycles

    # `truth.out_dir` is not part of run_filtering.yaml; it is appended on the
    # CLI (`+truth.out_dir=...`), so fail with a usable message when it is absent
    # rather than with Hydra's struct-mode KeyError.
    out_dir_cfg = cfg.get("truth", None)
    out_dir_cfg = out_dir_cfg.get("out_dir", None) if out_dir_cfg is not None else None
    if out_dir_cfg is None:
        raise ValueError(
            "No output directory: pass `+truth.out_dir=<path>` (the directory "
            "that will hold state.nc / params.nc for run.truth_dir)."
        )
    out_dir = pathlib.Path(out_dir_cfg)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Mirror run_filtering.py: the same `params_to_estimate` contract applies to
    # the truth sampler, and a dynamic truth is sampled over the FULL horizon.
    selected = cfg.get("params_to_estimate", None)
    selected = list(selected) if selected is not None else None
    truth_params_cfg = filter_parameter_config(cfg.truth_params, selected)
    is_dynamic_truth = "seconds_per_knot" in list(cfg.truth_params.keys())

    print(f"Truth horizon: {final_time:g}s ({num_cycles} cycles x {sim_time:g}s)")
    print(f"Output dir:    {out_dir}")

    true_forward_model = instantiate(
        cfg.truth_model.forward_model,
        results_dir=None,
        simulation_time=final_time,
    )
    if is_dynamic_truth:
        truth_sampler = instantiate(truth_params_cfg, simulation_time=final_time)
    else:
        truth_sampler = instantiate(truth_params_cfg)
    true_params = truth_sampler.sample(1)

    instantiate(cfg.truth_model.prepare, forward_model=true_forward_model)
    clean_outputs(model_name=cfg.truth_model.name, forward_model=true_forward_model)

    started = time.time()
    true_state = true_forward_model(params=true_params.isel(ensemble=0))
    elapsed = time.time() - started

    state_path = out_dir / "state.nc"
    params_path = out_dir / "params.nc"
    true_state.to_netcdf(state_path)
    true_params.to_netcdf(params_path)

    n_frames = int(true_state.sizes["time"])
    if n_frames < num_cycles:
        raise ValueError(
            f"The truth produced {n_frames} frame(s) for {num_cycles} cycles; "
            "each cycle needs at least one. Lower time.output_frequency."
        )

    write_yaml(
        {
            "final_time": final_time,
            "simulation_time_per_cycle": sim_time,
            "num_cycles": num_cycles,
            "num_frames": n_frames,
            "frames_per_cycle": n_frames // num_cycles,
            "truth_model": cfg.truth_model.name,
            "is_dynamic_truth": is_dynamic_truth,
            "wall_seconds": elapsed,
            "domain": OmegaConf.to_container(cfg.domain, resolve=True),
            "time": OmegaConf.to_container(cfg.time, resolve=True),
            "truth_params": OmegaConf.to_container(truth_params_cfg, resolve=True),
            "sha256": {
                "state.nc": _sha256(state_path),
                "params.nc": _sha256(params_path),
            },
        },
        out_dir / "truth_info.yaml",
    )
    print(f"Saved truth -> {state_path} ({n_frames} frames, {elapsed:.1f}s)")


@hydra.main(version_base=None, config_path="../../conf", config_name="run_filtering")
def main(cfg: DictConfig) -> None:
    run(cfg)


if __name__ == "__main__":
    main()
