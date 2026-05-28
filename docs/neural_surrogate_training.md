# Neural-surrogate training

A minimal end-to-end training stack for a one-step neural surrogate over
the [`training_data/`](training_data.md) splits. The model is intentionally
trivial — its only job is to exercise the dataloader → model → optimizer →
trainer wiring on real data before any architecture work begins.

## 1. Components

| Piece | File |
|---|---|
| `SimpleConv` baseline | [libs/neural-surrogates/src/neural_surrogates/architectures/simple_conv.py](../libs/neural-surrogates/src/neural_surrogates/architectures/simple_conv.py) |
| `Trainer` (train/val loop) | [libs/neural-surrogates/src/neural_surrogates/training.py](../libs/neural-surrogates/src/neural_surrogates/training.py) |
| `TransitionDataset` | [libs/neural-surrogates/src/neural_surrogates/data.py](../libs/neural-surrogates/src/neural_surrogates/data.py) (pre-existing) |
| Run script | [scripts/train_neural_surrogate.py](../scripts/train_neural_surrogate.py) |
| Config | [conf/neural_surrogate_training/train.yaml](../conf/neural_surrogate_training/train.yaml) |

## 2. `SimpleConv`

Single `Conv3d` layer over `(state ⊕ geometry)` along the channel dim.

- **Input channels**: `n_state_channels + 1` — the state channels stacked
  in `state_vars` order, with the binary geometry mask appended.
- **Output channels**: `n_state_channels` — one channel per state var.
- **Parameter injection**: each inflow parameter is broadcast-added to a
  distinct output channel (param `i` → channel `i`). If
  `n_params < n_state_channels` the extra channels receive zero bias. If
  `n_params > n_state_channels` construction raises.

The model predicts `state_next` directly; there is no residual /
delta-state structure yet.

## 3. `Trainer`

Generic loop. Constructor takes `model`, `train_loader`, `val_loader`,
`optimizer`, `loss_fn`, `num_epochs`, `device`. `fit()` runs the loop;
each epoch calls `_train_epoch` then `_validate` and prints the mean
losses. Batch unpacking assumes the `TransitionDataset` dict layout
(`state_n`, `state_next`, `params_n`, `geometry`).

The model and dataloaders are deliberately **constructed outside** the
trainer and passed in — this keeps `Trainer` agnostic to backend choice,
augmentation, and config structure.

## 4. Script flow

[scripts/train_neural_surrogate.py](../scripts/train_neural_surrogate.py):

1. Pull `dtype` from `cfg.dataset.dtype` (string → `torch.dtype`).
2. `instantiate(cfg.dataset, split="train"|"val", dtype=...)` → two
   `TransitionDataset`s.
3. `instantiate(cfg.dataloader, dataset=...)` for each, forcing
   `shuffle=False` on val.
4. Build `SimpleConv` with `n_state_channels=len(cfg.dataset.state_vars)`
   and `n_params=len(train_ds.param_names)`.
5. `instantiate(cfg.trainer, model=..., train_loader=..., val_loader=...,
   optimizer=instantiate(cfg.optimizer, params=model.parameters()),
   loss_fn=instantiate(cfg.loss))`.
6. `trainer.fit()`.

Every runtime object except the model — dataset, dataloader, optimizer,
loss, trainer — is constructed via `hydra.utils.instantiate` against a
`_target_` block. The model stays explicit because its `n_state_channels`
and `n_params` depend on the dataset.

## 5. Config

[conf/neural_surrogate_training/train.yaml](../conf/neural_surrogate_training/train.yaml)
holds five `_target_` blocks (`trainer`, `optimizer`, `loss`, `dataset`,
`dataloader`) plus the model's `kernel_size`. Defaults point at
`training_data/pylbm_tiny`, 5 epochs, CPU.

Override anything on the CLI:

```bash
pixi run -e dev python scripts/train_neural_surrogate.py
pixi run -e dev python scripts/train_neural_surrogate.py \
  dataset.root_dir=training_data/pylbm_small \
  dataloader.batch_size=16 \
  trainer.num_epochs=20 \
  optimizer.lr=5e-4
```

## 6. Extending

- **New architecture**: add a module under
  [libs/neural-surrogates/src/neural_surrogates/architectures/](../libs/neural-surrogates/src/neural_surrogates/architectures/),
  re-export from
  [architectures/__init__.py](../libs/neural-surrogates/src/neural_surrogates/architectures/__init__.py),
  and swap the explicit `SimpleConv(...)` call in the script. The
  `Trainer` does not need to change as long as the new model accepts
  `(state, params, geometry)`.
- **New optimizer / loss / loader**: change the `_target_` (and kwargs)
  in [train.yaml](../conf/neural_surrogate_training/train.yaml). No code
  edits required.
- **New trainer behavior** (schedulers, checkpointing, logging): extend
  `Trainer` and bump the `_target_` in the `trainer:` block.
