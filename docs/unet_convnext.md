# UNet-ConvNeXt architecture

A 3D UNet whose every stage is a stack of ConvNeXt blocks. Drop-in
alternative to the `SimpleConv` baseline for the neural-surrogate
training stack ([neural_surrogate_training.md](neural_surrogate_training.md)).
External contract is unchanged: `forward(state, params, geometry) →
state_next`, with `geometry` concatenated to `state` along the channel
dimension at the stem.

## 1. Model — `UNetConvNeXt`

File:
[libs/neural-surrogates/src/neural_surrogates/architectures/unet_convnext.py](../libs/neural-surrogates/src/neural_surrogates/architectures/unet_convnext.py)

### `_ConvNeXtBlock3d`

- **Depthwise conv** `Conv3d(C, C, k, groups=C)` — large-kernel spatial
  mixing per channel.
- **GroupNorm(1, C)** — channel-wise normalization (LayerNorm-equivalent
  for conv tensors).
- **Pointwise expand** `Conv3d(C → C·expansion, 1)` → **GELU** →
  **Pointwise project** `Conv3d(C·expansion → C, 1)` — the inverted
  bottleneck MLP.
- **Parameter bias injection**: a `Linear(n_params, C)` projects the
  per-sample inflow vector to one bias per channel; that bias is
  broadcast-added over all spatial positions inside *every* block, so
  params modulate every layer of the network.
- Residual connection wraps the whole block.

### `UNetConvNeXt`

- **Stem**: `Conv3d(n_state_channels + 1, base_channels, 3)`.
- **Encoder**: for each level `i`, a `_Stage` of `depths[i]` ConvNeXt
  blocks at `base_channels · channel_mults[i]`, then a stride-2 `Conv3d`
  to the next stage's channel count. Each pre-downsample activation is
  stashed as a skip.
- **Bottleneck**: one `_Stage` at the deepest channel count.
- **Decoder** (mirror): `ConvTranspose3d` upsamples, a 1×1 `Conv3d`
  fuses the upsampled tensor concatenated with its skip, then another
  `_Stage` of ConvNeXt blocks.
- **Head**: `Conv3d(base_channels, n_state_channels, 1)` — predicts
  `state_next` directly (no residual / delta-state structure yet).
- **Arbitrary input shapes**: `_pad_to_multiple` pads `(D, H, W)` up to
  a multiple of `2^n_levels` before the stem, then the head output is
  cropped back to the original spatial shape. Lets odd grid sizes
  (e.g. `5×7×11`) round-trip cleanly.

## 2. Config group

Directory:
[conf/neural_surrogate_architectures/unet_convnext/](../conf/neural_surrogate_architectures/unet_convnext/)

Five presets scale `base_channels`, `channel_mults`, `depths`,
`kernel_size`, `expansion`. Each file is a single
`_target_: neural_surrogates.UNetConvNeXt` block:

| Preset | base | mults | depths | kernel | expansion |
|---|---|---|---|---|---|
| tiny | 8 | [1, 2] | [1, 1] | 3 | 2 |
| small | 16 | [1, 2, 4] | [1, 1, 1] | 5 | 4 |
| medium | 24 | [1, 2, 4] | [2, 2, 2] | 7 | 4 |
| large | 32 | [1, 2, 4, 8] | [2, 2, 2, 2] | 7 | 4 |
| xlarge | 48 | [1, 2, 4, 8] | [3, 3, 3, 3] | 7 | 4 |

## 3. Wiring

- [neural_surrogates/__init__.py](../libs/neural-surrogates/src/neural_surrogates/__init__.py)
  and
  [architectures/__init__.py](../libs/neural-surrogates/src/neural_surrogates/architectures/__init__.py)
  re-export `UNetConvNeXt`, so `_target_: neural_surrogates.UNetConvNeXt`
  resolves directly.
- [conf/neural_surrogate_training/train.yaml](../conf/neural_surrogate_training/train.yaml)
  is `# @package _global_` and pulls
  `/neural_surrogate_architectures/unet_convnext@architecture: small`
  into its defaults list, so the preset lives at `cfg.architecture`.
- [scripts/train_neural_surrogate.py](../scripts/train_neural_surrogate.py)
  calls `instantiate(cfg.architecture, n_state_channels=...,
  n_params=...)`; `n_state_channels` and `n_params` stay explicit
  because they're derived from the dataset, not the architecture
  preset. `@hydra.main` is pointed at the top-level `conf/` so the
  cross-group defaults entry resolves.

## 4. Usage

Default preset (`small`):

```bash
pixi run -e dev python scripts/train_neural_surrogate.py
```

Swap presets — the override key is the full group path:

```bash
pixi run -e dev python scripts/train_neural_surrogate.py \
    'neural_surrogate_architectures/unet_convnext@architecture=medium'
```

Override individual fields:

```bash
pixi run -e dev python scripts/train_neural_surrogate.py \
    architecture.kernel_size=5 \
    architecture.expansion=2
```
