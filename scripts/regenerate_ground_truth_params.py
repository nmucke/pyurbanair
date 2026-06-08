"""Regenerate the time-varying ground-truth params.nc without rerunning the solver.

The inflow profile produced by scripts/run_time_varying_forward_model.py is a
deterministic function of the config + seed (esmda.seed), so we can reproduce
the exact params.nc from a past run by composing the same Hydra overrides and
calling create_time_varying_true_params directly. Writes params.nc to the repo
root.
"""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))

from hydra import compose, initialize_config_dir

from pyurbanair.config.hydra_helpers import create_time_varying_true_params

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent

# Mirror the CONFIG block of job_scripts/snellius/ground_truth.slurm.
OVERRIDES = [
    "model=pyudales",
    "case=xie_and_castro",
    "domain.nx=100",
    "domain.ny=80",
    "domain.nz=32",
    "domain.bounds=[[-20.0, 80.0], [0.0, 80.0], [0.0, 32.0]]",
    "time.simulation_time=3600.0",
    "time.output_frequency=1.0",
    "time.spinup_time=50.0",
    "time_varying.num_time_points=120",
    "esmda.seed=0",
    "run.use_true_params=false",
]


def main() -> None:
    with initialize_config_dir(version_base=None, config_dir=str(REPO_ROOT / "conf")):
        cfg = compose(config_name="config", overrides=OVERRIDES)

    params = create_time_varying_true_params(
        model_name=cfg.model.name,
        tv_cfg=cfg.time_varying,
        true_cfg=cfg.params.true,
        external_cfg=cfg.params.external,
        simulation_time=cfg.time.simulation_time,
        num_time_points=int(cfg.time_varying.num_time_points),
        seed=cfg.esmda.seed,
    )

    out = REPO_ROOT / "params.nc"
    params.to_netcdf(out)
    print(f"model={cfg.model.name} seed={cfg.esmda.seed} "
          f"num_time_points={int(cfg.time_varying.num_time_points)}")
    print(
        f"inflow_angle: {float(params['inflow_angle'].values[0]):.2f} -> "
        f"{float(params['inflow_angle'].values[-1]):.2f} deg"
    )
    print(
        f"velocity_magnitude: {float(params['velocity_magnitude'].values[0]):.2f} -> "
        f"{float(params['velocity_magnitude'].values[-1]):.2f} m/s"
    )
    print(f"Saved -> {out}")


if __name__ == "__main__":
    main()
