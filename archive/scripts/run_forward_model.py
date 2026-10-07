"""Run a single forward window or rollout through the shared workflow.

Examples::

    python scripts/run_forward_model.py model=pylbm params=static
    python scripts/run_forward_model.py model=pyudales run.ensemble=true
    python scripts/run_forward_model.py run.rollout_steps=2  # three windows
    python scripts/run_forward_model.py params=dynamic run.ensemble=true
"""

import pathlib
import sys

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import hydra
from omegaconf import DictConfig

from pyurbanair.workflows.forward import _concat_windows, get_stepper, run  # noqa: F401


@hydra.main(version_base=None, config_path="../conf", config_name="run_forward_model")  # type: ignore[misc, unused-ignore]
def main(cfg: DictConfig) -> None:
    run(cfg)


if __name__ == "__main__":
    main()
