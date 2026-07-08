"""LoRA fine-tuning of an already-trained next-step surrogate (plan 01).

Takes a ``model_dir`` produced by ``train_neural_surrogate.py``, rebuilds its
architecture, loads its weights, injects LoRA adapters (PEFT), trains **only**
the adapter weights on new data with the existing ``Trainer``, and writes a new
``model_dir`` whose merged ``weights.pt`` drops into
``NeuralSurrogateForwardModel`` unchanged (plus an ``adapter/`` subdir holding the
portable PEFT adapter).

The flow deliberately mirrors ``train_neural_surrogate.py``; the differences are:
the architecture + state/param/SDF spec come from the **pretrained** config
(single source of truth, cross-checked against the fine-tune dataset), the base
weights are frozen and only LoRA trains, and the trainer persists an always-valid
*merged* ``weights.pt`` via the ``weights_transform`` hook.

    pixi run -e dev python scripts/neural_surrogate/finetune_neural_surrogate.py \
        pretrained_model_dir=model_weights/p3d_xie_and_castro \
        model_name=p3d_xie_and_castro_ft_barcelona \
        dataset.root_dir=training_data/pylbm_barcelona
"""

from __future__ import annotations

from pathlib import Path

import hydra
import numpy as np
import torch
from hydra.utils import instantiate
from neural_surrogates.finetuning import (
    inject_lora,
    merge_to_state_dict,
    resolve_target_modules,
    save_adapter,
)
from neural_surrogates.training.data_utils import build_loader, get_normalization_stats
from omegaconf import DictConfig, OmegaConf


def _as_list(value) -> list | None:
    """OmegaConf list / None -> plain python list / None."""
    if value is None:
        return None
    return list(OmegaConf.to_container(value) if OmegaConf.is_config(value) else value)


def run(cfg: DictConfig) -> None:
    OmegaConf.set_struct(cfg, False)

    if cfg.get("pretrained_model_dir") in (None, "???"):
        raise ValueError("pretrained_model_dir must be set (a trained model_dir).")
    if cfg.get("model_name") in (None, "???"):
        raise ValueError("model_name must be set (the fine-tuned model_dir name).")

    pretrained_dir = Path(cfg.pretrained_model_dir)
    pre_cfg_path = pretrained_dir / "config.yaml"
    if not pre_cfg_path.exists():
        raise FileNotFoundError(f"pretrained config not found: {pre_cfg_path}")
    pre_cfg = OmegaConf.load(pre_cfg_path)

    # --- 1. Pull the architecture + channel/param/SDF spec from the pretrained
    # config: it is the single source of truth so the fine-tuned net can never
    # diverge from the weights it loads. The fine-tune dataset only supplies the
    # NEW data root (and dataloader / pushforward knobs).
    arch_node = pre_cfg.architecture
    pre_state_vars = _as_list(pre_cfg.dataset.state_vars)
    pre_param_vars = _as_list(pre_cfg.dataset.get("param_vars"))
    pre_sdf_features = pre_cfg.dataset.get("sdf_features", "none")
    pre_sdf_clamp = pre_cfg.dataset.get("sdf_clamp_cells", 32.0)
    if pre_state_vars is None:
        raise ValueError(
            f"pretrained config {pre_cfg_path} has no dataset.state_vars; cannot "
            "determine the channel ordering to fine-tune."
        )

    # Stamp the pretrained spec onto the fine-tune dataset config before building
    # it. This guarantees the dataset ships exactly the channels / SDF features
    # the pretrained stem expects.
    cfg.dataset.state_vars = pre_state_vars
    cfg.dataset.sdf_features = pre_sdf_features
    cfg.dataset.sdf_clamp_cells = pre_sdf_clamp
    if cfg.dataset.get("param_vars") is None and pre_param_vars is not None:
        cfg.dataset.param_vars = pre_param_vars

    if cfg.get("dataset") is None or cfg.dataset.get("root_dir") in (None, "???"):
        raise ValueError("dataset.root_dir must point at the fine-tune data.")

    dtype = getattr(torch, cfg.dataset.dtype)
    train_ds = instantiate(cfg.dataset, split="train", dtype=dtype)
    val_ds = instantiate(cfg.dataset, split="val", dtype=dtype)
    if int(cfg.dataloader.get("num_workers", 0)) == 0:
        cfg.dataloader.persistent_workers = False
    train_loader = build_loader(cfg, train_ds, train=True)
    val_loader = build_loader(cfg, val_ds, train=False)

    # --- Cross-check the fine-tune dataset against the pretrained spec. Fail loud
    # on any mismatch (same spirit as the SDF cross-check in the train script) so
    # a channel/param reorder can never silently train on the wrong stem.
    if list(train_ds.state_vars) != list(pre_state_vars):
        raise ValueError(
            "state_vars mismatch between fine-tune dataset "
            f"({list(train_ds.state_vars)}) and pretrained model "
            f"({list(pre_state_vars)})."
        )
    if pre_param_vars is not None and list(train_ds.param_names) != list(
        pre_param_vars
    ):
        raise ValueError(
            "param order mismatch between fine-tune dataset "
            f"({list(train_ds.param_names)}) and pretrained model "
            f"({list(pre_param_vars)})."
        )

    # --- 2. Rebuild the architecture from the pretrained node and load its
    # weights. n_state_channels / n_params are derived from the (cross-checked)
    # dataset, exactly as in train_neural_surrogate.py.
    model = instantiate(
        arch_node,
        n_state_channels=len(pre_state_vars),
        n_params=len(train_ds.param_names),
    ).to(dtype=dtype)

    weights_path = pretrained_dir / "weights.pt"
    if not weights_path.exists():
        raise FileNotFoundError(f"pretrained weights not found: {weights_path}")
    # Load onto CPU then move the whole (peft-wrapped) model to the device once,
    # below -- no pointless weights->device->CPU round-trip into a CPU-resident
    # model.
    model.load_state_dict(torch.load(weights_path, map_location="cpu"))
    print(f"loaded pretrained weights from {weights_path}")

    # --- 3. Optionally recompute normalization on the fine-tune split. Off by
    # default so val metrics stay comparable to the base model; when on, the
    # architecture MUST expose set_normalization -- fail loud otherwise rather
    # than silently ignoring the flag (a UNetConvNeXt has no such hook).
    if cfg.get("recompute_normalization", False):
        if not hasattr(model, "set_normalization"):
            raise ValueError(
                "recompute_normalization=true but architecture "
                f"{type(model).__name__!r} has no set_normalization hook "
                "(only UPT / P3D do). Set recompute_normalization=false."
            )
        s_mean, s_std, p_mean, p_std = get_normalization_stats(train_ds)
        model.set_normalization(s_mean, s_std, p_mean, p_std)
        print(
            "recomputed normalization on the fine-tune split:\n"
            f"  state_mean={np.round(s_mean, 4)} state_std={np.round(s_std, 4)}\n"
            f"  param_mean={np.round(p_mean, 4)} param_std={np.round(p_std, 4)}"
        )

    # --- 4. Freeze the base weights and inject LoRA. get_peft_model freezes the
    # base itself, but freeze explicitly first so the intent is unambiguous.
    model.requires_grad_(False)
    # An explicit target_modules may be a regex string or a list of names; a list
    # comes off OmegaConf as a config object, so normalise it. None => use preset.
    explicit_targets = cfg.lora.get("target_modules")
    if OmegaConf.is_config(explicit_targets):
        explicit_targets = _as_list(explicit_targets)
    target_modules = resolve_target_modules(
        model,
        preset=cfg.lora.get("target_preset"),
        target_modules=explicit_targets,
    )
    peft_model = inject_lora(
        model,
        rank=cfg.lora.rank,
        alpha=cfg.lora.alpha,
        dropout=cfg.lora.get("dropout", 0.0),
        target_modules=target_modules,
        variant=cfg.lora.get("variant", "standard"),
        modules_to_save=_as_list(cfg.lora.get("modules_to_save")) or None,
    )
    peft_model = peft_model.to(cfg.trainer.device)
    peft_model.print_trainable_parameters()

    print(
        f"train pairs={len(train_ds)}  param_names={train_ds.param_names}  "
        f"n_state_channels={len(pre_state_vars)}"
    )

    # --- 5. Save the resolved config to the new model_dir. It records the
    # pretrained `architecture` node (so ESMDA rebuilds the net without chasing
    # the original dir), the fine-tune `dataset` (root/state_vars/param_vars),
    # and a `pretrained:` provenance block.
    out_dir = Path("model_weights") / cfg.model_name
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg.architecture = arch_node
    cfg.dataset.param_vars = list(train_ds.param_names)
    cfg.pretrained = {
        "model_dir": str(pretrained_dir),
        "architecture": arch_node,
    }
    OmegaConf.save(cfg, out_dir / "config.yaml")

    # --- 6. Train only the adapter weights with the existing Trainer. The
    # weights_transform makes the best-val weights.pt an always-valid *merged*
    # plain state dict (see BaseTraining), so a killed run still leaves an
    # ESMDA-loadable file behind.
    trainable = [p for p in peft_model.parameters() if p.requires_grad]
    trainer = instantiate(
        cfg.trainer,
        model=peft_model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=instantiate(cfg.optimizer, params=trainable),
        loss_fn=instantiate(cfg.loss),
        weights_path=out_dir / "weights.pt",
        # merge_to_state_dict runs a deepcopy+merge on every val improvement (see
        # BaseTraining) -- cheap per epoch, non-trivial repeated allocation over a
        # long run; acceptable for short fine-tunes. The trainer restores the
        # best-val model at the end of fit() (surviving resume via checkpoint.pt).
        weights_transform=merge_to_state_dict,
    )
    trainer.fit()

    # --- 7. Persist the adapter, and (only when fit() actually holds the best-val
    # model) overwrite weights.pt with the final merged plain dict. If a resumed
    # run could not recover the best weights, DON'T overwrite -- the trainer's
    # mid-training best-val weights.pt is already on disk and must not be clobbered
    # with a worse (last-epoch) model. weights.pt stays indistinguishable from a
    # fully trained model -> zero changes to forward_model.py.
    save_adapter(peft_model, out_dir / "adapter")
    if trainer.restored_best_weights:
        torch.save(merge_to_state_dict(peft_model), out_dir / "weights.pt")
        print(f"fine-tuned model (adapter + merged best weights) saved to {out_dir}")
    else:
        print(
            f"WARNING: kept the trainer's on-disk best-val weights.pt in {out_dir} "
            "(fit() could not restore the best model in memory -- e.g. a resume "
            "with no improvement from a pre-upgrade checkpoint); the adapter/ "
            "reflects the last epoch, not the best."
        )


@hydra.main(
    version_base=None,
    config_path="../../conf",
    config_name="neural_surrogate/finetuning",
)
def main(cfg: DictConfig) -> None:
    run(cfg)


if __name__ == "__main__":
    main()
