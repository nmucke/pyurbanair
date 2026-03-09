"""Generate ensemble of developed flows for ESMDA init conditions."""

import argparse
import pathlib
import shutil
import sys

import jax
import jax.numpy as jnp
import xarray

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from scripts import config


def _create_random_params(
    model_name: str,
    n: int,
    seed: int = 42,
) -> xarray.Dataset:
    """Create random parameter ensemble for init condition generation."""
    cfg = config.PARAM_PRIORS
    rng_key = jax.random.PRNGKey(seed)

    rng_key, subkey = jax.random.split(rng_key)
    inflow = (
        jax.random.normal(subkey, (n,)) * cfg["inflow_angle_std"]
        + cfg["inflow_angle_mean"]
    )

    rng_key, subkey = jax.random.split(rng_key)
    vel = jax.random.normal(subkey, (n,)) * cfg["velocity_std"] + cfg["velocity_mean"]
    vel = jnp.maximum(vel, 0.1)

    data_vars = {
        "inflow_angle": ("ensemble", inflow),
        "velocity_magnitude": ("ensemble", vel),
    }
    if model_name == "pyudales":
        rng_key, subkey = jax.random.split(rng_key)
        pressure = (
            jax.random.normal(subkey, (n,)) * cfg["pressure_std"] + cfg["pressure_mean"]
        )
        pressure = jnp.maximum(pressure, 1e-6)
        data_vars["pressure_gradient_magnitude"] = ("ensemble", pressure)

    return xarray.Dataset(data_vars=data_vars, coords={"ensemble": jnp.arange(n)})


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate ensemble of developed flows for ESMDA init conditions.",
    )
    parser.add_argument(
        "--model",
        choices=["pylbm", "pyudales"],
        default="pylbm",
        help="Forward model backend.",
    )
    parser.add_argument(
        "--num-simulations",
        type=int,
        default=500,
        help="Number of simulations to generate.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for parameter generation.",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Remove .temp directory before running.",
    )
    parser.add_argument(
        "--output-dir",
        type=pathlib.Path,
        default=None,
        help="Override output directory (default: esmda_init_conditions/{lbm|udales}/).",
    )
    args = parser.parse_args()

    model_name = args.model
    n = args.num_simulations
    subdir = "lbm" if model_name == "pylbm" else "udales"

    if args.clean and pathlib.Path(".temp").exists():
        shutil.rmtree(".temp")

    output_dir = args.output_dir
    if output_dir is None:
        base = pathlib.Path(
            config.ESMDA.get("init_conditions_dir", "esmda_init_conditions")  # type: ignore[arg-type]
        )
        output_dir = base / subdir
    output_dir.mkdir(parents=True, exist_ok=True)

    forward_model = config.create_forward_model(
        model_name=model_name,
        rollout=False,
        results_dir=None,
    )
    config.prepare_forward_model(model_name=model_name, forward_model=forward_model)
    config.clean_forward_model_outputs(
        model_name=model_name, forward_model=forward_model
    )

    params = _create_random_params(model_name, n, seed=args.seed)
    params.to_netcdf(output_dir / "params.nc")

    for i in range(n):
        print(f"Simulation {i + 1} / {n}")
        state = forward_model(params=params.isel(ensemble=i))
        if state is None:
            raise RuntimeError(f"Expected in-memory state from simulation {i}.")
        if "time" in state.dims:
            state = state.isel(time=-1)
        state.to_netcdf(output_dir / f"state_{i}.nc")

    print(f"Saved {n} init conditions to {output_dir}")


if __name__ == "__main__":
    main()
