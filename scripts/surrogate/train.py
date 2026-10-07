"""Train a neural surrogate. One script for every training config:

    python scripts/surrogate/train.py --config-name surrogate/train_stepper
    python scripts/surrogate/train.py --config-name surrogate/train_autoencoder
    python scripts/surrogate/train.py --config-name surrogate/train_latent_generator
    python scripts/surrogate/train.py --config-name surrogate/train_dft
    python scripts/surrogate/train.py --config-name surrogate/finetune_stepper

Every run does the same: build the train/val datasets and the model for the
config's `task` (tasks.py), then fit with the configured `trainer`,
`optimizer` and `loss`.

Outputs, in `<paths.weights_dir>/<name>/`: config.yaml (everything needed to
rebuild the model), weights.pt (best validation weights) and the trainer's
checkpoint; plus encoder.pt/decoder.pt for an autoencoder and adapter/ for LoRA.
An autoencoder with `prechunk.output_root` trains on a re-chunked copy of the
data; `prechunk.prepare_only=true` only makes that copy.
"""

from __future__ import annotations

import pathlib
import sys

import hydra
from hydra.utils import instantiate
from neural_surrogates.training.data_utils import build_loader
from omegaconf import DictConfig, OmegaConf

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "utils"))

from tasks import TASKS, prechunked_root  # noqa: E402


def run(cfg: DictConfig) -> None:
    OmegaConf.set_struct(cfg, False)  # tasks record derived settings in cfg
    if cfg.get("prechunk") is not None and cfg.prechunk.prepare_only:
        if cfg.prechunk.output_root is None:
            raise ValueError("prechunk.prepare_only needs prechunk.output_root")
        print(f"Prepared {prechunked_root(cfg)}")
        return
    out_dir = pathlib.Path(cfg.paths.weights_dir) / cfg.name
    out_dir.mkdir(parents=True, exist_ok=True)
    if int(cfg.dataloader.num_workers) == 0:
        cfg.dataloader.persistent_workers = False  # needs worker processes

    setup = TASKS[cfg.task](cfg, out_dir)
    OmegaConf.save(cfg, out_dir / "config.yaml")

    trainable = [p for p in setup.model.parameters() if p.requires_grad]
    total = sum(p.numel() for p in setup.model.parameters())
    print(
        f"{cfg.task}: {len(setup.train)} train / {len(setup.val)} val samples, "
        f"{sum(p.numel() for p in trainable):,} of {total:,} parameters trained"
    )
    trainer = instantiate(
        cfg.trainer,
        model=setup.model,
        train_loader=build_loader(cfg, setup.train, train=True),
        val_loader=build_loader(cfg, setup.val, train=False),
        optimizer=instantiate(cfg.optimizer, params=trainable),
        loss_fn=instantiate(cfg.loss),
        weights_path=out_dir / "weights.pt",
        **setup.trainer_kwargs,
    )
    if setup.before_fit is not None:
        setup.before_fit(trainer)
    trainer.fit()
    if setup.after_fit is not None:
        setup.after_fit(trainer)
    print(f"Saved model in {out_dir}")


@hydra.main(  # type: ignore[misc, unused-ignore]
    version_base=None,
    config_path="../../configs",
    config_name="surrogate/train_stepper",
)
def main(cfg: DictConfig) -> None:
    run(cfg)


if __name__ == "__main__":
    main()
