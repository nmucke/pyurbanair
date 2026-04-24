"""Clone a ForwardModel onto a fresh experiment directory for ensemble members."""

import copy
import logging
import pathlib
import shutil
from typing import TYPE_CHECKING

from .dir_utils import create_dir, get_palm_directory_paths

if TYPE_CHECKING:
    from pypalm.forward_model import ForwardModel

logger = logging.getLogger(__name__)


def create_new_forward_model(
    forward_model: "ForwardModel",
    experiment_base_dir: pathlib.Path,
    experiment_name: str,
) -> "ForwardModel":
    """Deep-copy ``forward_model`` and retarget its dirs to a new experiment.

    The source ``experiment_dir`` (containing INPUT/_p3d and _topo written by
    ``__init__``) is recursively copied. PALM job files are unnamed generically,
    but the filename prefix matches the experiment_name, so we rename any
    ``<old_name>_p3d`` / ``<old_name>_topo`` files to ``<experiment_name>_p3d``
    / ``<experiment_name>_topo`` after copy.
    """
    old_experiment_dir = forward_model.dirs.experiment_dir
    old_experiment_name = forward_model.dirs.experiment_name
    cwd = forward_model.dirs.cwd

    new_base = create_dir(cwd / experiment_base_dir)
    new_experiment = create_dir(new_base / experiment_name)

    if old_experiment_dir.exists():
        for item in old_experiment_dir.iterdir():
            dest = new_experiment / item.name
            if item.is_dir():
                shutil.copytree(item, dest, dirs_exist_ok=True)
            else:
                shutil.copy2(item, dest)

    new_input = new_experiment / "INPUT"
    if new_input.exists() and old_experiment_name != experiment_name:
        for item in new_input.iterdir():
            if not item.is_file():
                continue
            if item.name.startswith(f"{old_experiment_name}_"):
                suffix = item.name[len(old_experiment_name) :]
                item.rename(new_input / f"{experiment_name}{suffix}")

    new_model = copy.deepcopy(forward_model)
    new_model.dirs = get_palm_directory_paths(
        case_dir=forward_model.dirs.case_dir,
        experiment_name=experiment_name,
        temp_dir=forward_model.dirs.temp_dir,
        experiment_base_dir=new_base,
        cwd=cwd,
        results_dir=forward_model.dirs.results_dir,
    )
    new_model.experiment_name = experiment_name

    logger.info(
        "Created new PALM ForwardModel: experiment_dir=%s, experiment_name=%s",
        new_model.dirs.experiment_dir,
        experiment_name,
    )
    return new_model
