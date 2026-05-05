"""Smoke test: tiny rollout-ESMDA run for DelftBlue env validation.

Mutates ``scripts.config`` in place with minimal values so the
time-varying parameter rollout-ESMDA pipeline completes in a few minutes
on a single compute node, then delegates to the real entry point in
``scripts/run_time_varying_parameters_rollout_esmda.py``.
"""

import pathlib
import sys

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from scripts import config

config.TIME["simulation_time"] = 60.0
config.TIME["spinup_time"] = 5.0
config.TIME["output_frequency"] = 5.0

config.ENSEMBLE["ensemble_size"] = 4
config.ENSEMBLE["num_parallel_processes"] = 4
config.ENSEMBLE["failure_policy"] = "raise"

config.ESMDA["num_assimilation_windows"] = 1
config.ESMDA["num_steps"] = 1

config.TIME_VARYING_PARAMS["num_time_points"] = 3

from scripts.run_time_varying_parameters_rollout_esmda import main  # noqa: E402

if __name__ == "__main__":
    main()
