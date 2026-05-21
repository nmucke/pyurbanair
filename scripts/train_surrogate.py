"""Train a neural surrogate (docs/neural_surrogate_plan.md §6, P2/P4).

Architecture-agnostic: the architecture is whatever ``arch.name`` selects
(``unet3d`` now, ``upt`` later) — no script changes to add a new one.

    pixi run -e cuda python scripts/train_surrogate.py \
        corpus_path=.temp/neural_surrogate/corpus run_id=lbm_xie_castro_unet3d
"""

import hydra
from omegaconf import DictConfig


def run(cfg: DictConfig) -> dict:
    # Heavy NN imports stay function-local so importing this module is cheap.
    from neural_surrogates.training.train import run as train_run

    return train_run(cfg)


@hydra.main(version_base=None, config_path="../conf/neural_surrogate", config_name="train")
def main(cfg: DictConfig) -> None:
    run(cfg)


if __name__ == "__main__":
    main()
