"""Sweep tnudge for member 0 params at runtime=50."""

import argparse
import pathlib
import sys

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import xarray as xr

from scripts import config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--angle", type=float, default=4.85)
    parser.add_argument("--vel", type=float, default=2.79)
    parser.add_argument("--tnudge", type=float, default=10.0)
    args = parser.parse_args()

    fm = config.create_forward_model(model_name="pyudales")
    config.prepare_forward_model(model_name="pyudales", forward_model=fm)
    config.clean_forward_model_outputs(model_name="pyudales", forward_model=fm)

    fm._nudging_config = {
        "tnudge": args.tnudge,
        "nnudge": 0,
        "profile_config": {"type": "power_law", "alpha": 0.25},
    }

    params = xr.Dataset(
        data_vars={
            "inflow_angle": args.angle,
            "velocity_magnitude": args.vel,
            "pressure_gradient_magnitude": config.TRUE_PARAMS[
                "pressure_gradient_magnitude"
            ],
        }
    )
    try:
        state = fm(params=params)
        if state is None:
            state = fm.get_states()
        print(f"OK angle={args.angle} vel={args.vel} tnudge={args.tnudge}")
    except Exception as e:
        print(f"FAIL angle={args.angle} vel={args.vel} tnudge={args.tnudge}: {type(e).__name__}")


if __name__ == "__main__":
    main()
