"""Compose and validate a workflow without importing or starting a backend.

Usage: python scripts/preview_config.py run_esmda experiment=esmda/barcelona_dynamic
"""

from __future__ import annotations

import argparse
import pathlib
import sys

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from pyurbanair.config.run_record import validate_run_config

WORKFLOWS = {
    "run_forward_model": "forward",
    "run_esmda": "esmda",
    "run_filtering": "filtering",
    "run_filter_smoothing": "filter_smoothing",
    "compare_models": "comparison",
    "run_probe_series": "probe",
    "neural_surrogate/training_data": "surrogate_random_data",
    "fixed_training_data": "surrogate_data",
    "neural_surrogate/training": "surrogate_training",
    "neural_surrogate/finetuning": "surrogate_finetuning",
    "neural_surrogate/testing": "surrogate_testing",
    "neural_surrogate/comparison": "surrogate_comparison",
    "neural_surrogate/pretrain_autoencoder": "surrogate_autoencoder_training",
    "neural_surrogate/testing_autoencoder": "surrogate_autoencoder_testing",
    "neural_surrogate/train_latent_generator": "surrogate_latent_training",
    "neural_surrogate/testing_latent_generator": "surrogate_latent_testing",
    "render_les": "render",
}

_INPUT_PATHS = {
    "surrogate_training": ("dataset.root_dir",),
    "surrogate_finetuning": ("pretrained_model_dir", "dataset.root_dir"),
    "surrogate_testing": ("model_dir",),
    "surrogate_comparison": ("data.root_dir",),
    "surrogate_autoencoder_training": ("dataset.root_dir",),
    "surrogate_autoencoder_testing": ("model_dir",),
    "surrogate_latent_training": ("pretrained_ae_dir", "dataset.root_dir"),
    "surrogate_latent_testing": ("model_dir",),
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", choices=WORKFLOWS)
    parser.add_argument("overrides", nargs="*", help="Hydra key=value overrides")
    args = parser.parse_args()
    conf_dir = pathlib.Path(__file__).resolve().parent.parent / "conf"
    config_name = (
        "neural_surrogate/training_data"
        if args.config == "fixed_training_data"
        else args.config
    )
    if args.config == "fixed_training_data":
        args.overrides.insert(0, "training_data/geometry_mode=fixed")
    with initialize_config_dir(version_base=None, config_dir=str(conf_dir)):
        cfg = compose(
            config_name=config_name,
            overrides=args.overrides,
            return_hydra_config=True,
        )
    choices = OmegaConf.to_container(cfg.hydra.runtime.choices, resolve=True)
    del cfg.hydra
    validate_run_config(cfg, WORKFLOWS[args.config])
    print("Selected configuration files:")
    print(OmegaConf.to_yaml(choices, resolve=True))
    print("Resolved user configuration:")
    print(OmegaConf.to_yaml(cfg, resolve=True))
    print("Deferred runtime arguments:")
    for key in _INPUT_PATHS.get(WORKFLOWS[args.config], ()):
        value = OmegaConf.select(cfg, key)
        if value in (None, "???"):
            print(f"  {key}: required at launch")
        elif not pathlib.Path(str(value)).exists():
            print(f"  {key}: path is absent here ({value})")
    if WORKFLOWS[args.config] in {"esmda", "filter_smoothing"}:
        print("  observation covariance size: observed data after aggregation")
        if "TimeVaryingParameter" in str(
            OmegaConf.select(cfg, "esmda.smoother._target_")
        ):
            print("  smoother.num_time_points: sampled prior knot count")
    if WORKFLOWS[args.config] in {"filtering", "filter_smoothing"}:
        print("  filter cycle count: truth observation timeline")
    if WORKFLOWS[args.config] == "forward":
        print("  sampled parameter shape: parameter sampler")


if __name__ == "__main__":
    main()
