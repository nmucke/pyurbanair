# Plan 02 — Autoencoder (foundation-model) pre-training, Tadpole-style

**Status: implemented (2026-07-09).** Delivered: the vendored autoencoder subtree
`neural_surrogates.architectures._tadpole` (Tadpole @ 4232e698, Apache-2.0),
`TadpoleAE` (`architectures/tadpole_ae.py`), `SnapshotDataset` + `snapshot_collate`
(`datasets/snapshot.py`), `AutoencoderTrainer` (`training/autoencoder.py`),
`conf/neural_surrogate/pretrain_autoencoder.yaml`,
`scripts/neural_surrogate/pretrain_autoencoder.py`,
`tests/test_autoencoder_pretraining.py`. See `docs/neural_surrogates.md` Part G.
**Dependency decision baked in (§1):** the upstream `tadpole` package is *not* a
dependency — only the autoencoder subtree is vendored (like `_upt/`), because
upstream's `requirements.txt` pulls in `torchfsm` → `vape4d` (its unused online
data-gen chain). The vendored files are byte-for-byte upstream except that the
`GIFt` import (its integer-rank LoRA path, which we never take — LoRA goes through
PEFT) is made optional. **Crop-size constraint found in implementation:**
`encoder_crop_size` must be a multiple of 16 (the encoder's total downsampling);
smaller values make the upstream decoder over-upsample (output ≠ input shape), so
the wrapper validates it. The adversarial (GAN) loss remains the scoped optional
extension (the discriminator is vendored but not wired).

**Goal:** a new training submodule + config for pre-training a Tadpole
autoencoder on our flow data — representation learning only, no next-step
objective. Faithful to the paper's single-channel-crop design. Staged scope:
**VAE core is the deliverable; the adversarial (GAN) loss is an optional
extension** with its own section.

We do **not** modify our `P3D` wrapper or the `p3d_surrogate` package. The
autoencoder uses the `tadpole` package's own P3D encoder/decoder
(`tadpole.architecture.p3d`), which already ships the KL/VAE head and the
skip-connection machinery.

## 0. What the `tadpole` package gives us (verified against the repo)

- `tadpole.model.autoencoder.TadpoleAutoencoder(size, weight_encoder=None,
  weight_decoder=None, encoder_ft_state="FPFT", decoder_ft_state="FPFT",
  latent_type="sample"|"mode", encoder_crop_size=64, max_internal_batchsize)`
  — builds `_KLP3DEncoder` + `_P3DDecoder` (no skips), forward:
  `(B, C, X, Y, Z) → (B, C, X, Y, Z)` with channels folded into batch and the
  field tiled into `encoder_crop_size³` crops internally (einops rearrange).
  `forward(x, return_kl_element=True)` also returns the per-crop KL element
  for the VAE loss. `save_separate_weights(enc_path, dec_path)`.
- Latent: `DiagonalGaussianDistribution` (mean/log-var head); compression 16/8/4
  for sizes S/B/L (8.8M/38.1M/152.1M params).
- Pretrained weights on HF `thuerey-group/Tadpole` (per-size encoder/decoder
  state dicts, loadable via the `weight_encoder`/`weight_decoder` args —
  verify exact HF file layout at implementation time; the
  `tutorials/load_and_run.ipynb` in the repo shows the intended loading path).
- Deps: `einops`, `torchfsm`, `diffusers`, `timm`, `GIFt` (git). License
  Apache-2.0 (LICENSE file; setup.py classifier says MIT — LICENSE wins).

## 1. Dependency wiring

- Add to `libs/neural-surrogates/pyproject.toml` as an optional extra,
  mirroring `[p3d]`:
  `tadpole = ["tadpole @ git+https://github.com/tum-pbs/Tadpole@<pinned-commit>"]`
  and add the feature to the pixi dev env. Lazy-import `tadpole` inside the
  wrapper module (same pattern as `architectures/p3d.py`) so the base install
  stays light.
- Risk check first: `torchfsm` (their online data-generation dep) must
  pip-resolve on both linux-64 and osx-arm64 in the dev env. If it's the only
  blocker, vendor-free workaround: depend on tadpole with
  `--no-deps`-equivalent is not expressible in pyproject — instead open with a
  quick `pixi add --pypi` trial; if unresolvable, fall back to vendoring
  `tadpole/architecture/p3d` + `tadpole/model` (Apache-2.0 permits it, ~1k
  lines) under `neural_surrogates/architectures/_tadpole/` like `_upt/`.
  Decide in the first hour of implementation, not later.

## 2. Wrapper architecture: `architectures/tadpole_ae.py`

`class TadpoleAE(nn.Module)` — our conventions around `TadpoleAutoencoder`:

- **Constructor** (all Hydra-instantiable):
  `size, n_state_channels, latent_type="sample", encoder_crop_size=64,
  max_internal_batchsize=None, pretrained="none"|"hf"|{enc,dec paths},
  encode_geometry=true, sdf_features=false, sdf_clamp_cells=32.0,
  normalize=true`.
- **Normalization**: `set_normalization(state_mean, state_std, ...)` buffers,
  identical contract to `P3D`/`UPT` so the train script's existing
  `_compute_normalization_stats` + `set_normalization` path just works.
  Per-channel z-score *before* the channel fold (each folded single-channel
  crop is then ~N(0,1), matching Tadpole's pre-training statistics — important
  when starting from HF weights).
- **Geometry handling** (urban flow has obstacles; Tadpole's pre-training data
  doesn't): mask input (`state * geometry`) like `P3D` does; and with
  `encode_geometry=true`, append the geometry mask (and, when
  `sdf_features=true`, the clamped SDF channel) as **extra folded channels**
  through the same encoder. They are reconstructed alongside the state
  (trivially — they're crop-wise near-constant) and, crucially, their latents
  are what the DFT sub-network (plan 03) will attend over to see geometry.
  With `encode_geometry=false` the input is state channels only — the knob
  exists so we can A/B it.
- **Padding**: pad each spatial dim up to a multiple of `encoder_crop_size`
  (reflect-pad fluid regions / zero-pad like `P3D._pad_to_multiple`), crop the
  reconstruction back. Alternative to padding: choose `encoder_crop_size`
  dividing the grid (48/32 work per the paper's translation-equivariance
  argument) — expose it in config, keep padding as the general fallback.
- **Forward**: `forward(state, geometry, geom_features=None, *,
  return_kl_element=False)` → reconstruction (+ KL element). Note: NOT the
  time-stepper contract; the AE is never an ESMDA forward model.
- Expose `encode(...)`/`decode(...)` passthroughs (used by plan 03 and by
  analysis notebooks).

Params (`params_n`) are deliberately **not** an AE input: physical parameters
condition dynamics, not single-snapshot appearance; they enter in plan 03.

## 3. Dataset: `datasets/snapshot.py`

`class SnapshotDataset` — sibling of `TransitionDataset` (reuse its file
discovery, geometry/SDF loading, lazy `xr.open_dataset` pattern):

- Item: `{"state": (C, *grid), "geometry": (*grid), "geom_features": ...}` —
  single time slices, every `(traj, t)` pair is a sample (optionally strided
  via `time_stride` to decorrelate).
- **Random-crop augmentation** (paper's "intermediate pre-cropping"): optional
  `random_crop_size` (e.g. 96) — the dataset returns a random spatial crop of
  the full field; the model then tiles it into `encoder_crop_size³` internally.
  More crop diversity per epoch, smaller batches in memory. Default `null`
  (full field) so the smoke tests stay trivial.
- Reuse `transition_collate`'s shared-geometry trick only when not
  random-cropping (crops break the "identical geometry per batch" assumption —
  ship per-sample geometry crops in that case).

## 4. Trainer: `training/autoencoder.py`

`class AutoencoderTrainer(BaseTraining)` — reuse the infrastructure, replace
the rollout-shaped parts:

- Inherits: device/amp/GradScaler, warmup+cosine LR, grad clipping, early
  stopping, checkpoint/resume, `metrics.csv`, best-weights saving.
- Overrides/neutralizes: the pushforward curriculum (no-op:
  `pushforward_epochs_per_step: null`, `grad_unroll_steps` unused) and
  `_forward`/`_final_loss`:

  ```python
  loss = masked_mse(recon, state, fluid_mask) + kl_weight * kl_elem.mean()
  ```

  - `masked_mse` restricted to fluid cells (same `mask_loss` convention as
    `Trainer`); reconstruction of the geometry/SDF channels (if
    `encode_geometry`) gets a separate small weight `geometry_recon_weight`
    (default keeps total loss dominated by state recon).
  - `kl_weight` (β): default tiny (Tadpole/latent-diffusion convention,
    ~1e-6); `kl_weight: 0` + `latent_type: mode` degrades gracefully to a
    plain deterministic AE — that's the "AE core" of the staged scope, one
    config knob away.
- If `BaseTraining`'s structure makes the override awkward (e.g. `_forward`'s
  `(state_n, params_n, ...)` batch unpacking is hard-wired), refactor
  minimally: extract the batch-unpack + loss into the two overridable methods
  and keep `Trainer`/`PatchTrainer` byte-identical (existing tests guard this).

### Optional extension (separate PR, only if VAE recon is too smooth):
adversarial loss. `tadpole.architecture.discriminator` ships the
discriminator. Adds: second optimizer, `adv_weight` warm-up schedule, and a
`discriminator:` config block (default `null` → exactly the VAE path). Scoped
here so the core lands without GAN-tuning risk.

## 5. Config + script

`conf/neural_surrogate/pretrain_autoencoder.yaml` (`# @package _global_`):

```yaml
model_name: tadpole_ae_s
architecture:
  _target_: neural_surrogates.TadpoleAE
  size: S
  encoder_crop_size: 64
  latent_type: sample
  encode_geometry: true
  pretrained: hf            # none | hf | {encoder: path, decoder: path}
loss:
  kl_weight: 1.0e-6
  geometry_recon_weight: 0.1
trainer: { _target_: neural_surrogates.AutoencoderTrainer, ...training.yaml-like... }
optimizer: { _target_: torch.optim.AdamW, lr: 1.0e-4, ... }
dataset:  { _target_: neural_surrogates.SnapshotDataset, root_dir: ..., time_stride: 1, random_crop_size: null, ... }
dataloader: { ... }
```

`scripts/neural_surrogate/pretrain_autoencoder.py` — same skeleton as
`train_neural_surrogate.py` (`run(cfg)` + `@hydra.main`): datasets →
normalization stats → `set_normalization` → save resolved `config.yaml` →
`AutoencoderTrainer.fit()`.

**Artifacts** (`model_weights/<name>/`): `weights.pt` (full `TadpoleAE` state
dict — our standard), plus `encoder.pt`/`decoder.pt` via
`save_separate_weights` — the natural handoff format for plan 03 and for
sharing/HF-style reuse. `config.yaml`, `checkpoint.pt`, `metrics.csv` as usual.

## 6. Tests

- Unit (CPU, tiny): `TadpoleAE(size="S", encoder_crop_size=8)` on an
  `(1, 3, 16, 16, 16)` field — recon shape, KL element finite, padding
  round-trip on a non-divisible grid, `encode_geometry` on/off shapes.
  Gate with `pytest.importorskip("tadpole")` so envs without the extra skip.
- e2e smoke: compose `pretrain_autoencoder.yaml` with smoke overrides, 2
  epochs on the fixture data, assert artifacts exist and `weights.pt` reloads.

## 7. Open questions (to resolve during implementation, defaults chosen)

- **HF-pretrained vs from-scratch on our data**: support both (§2
  `pretrained`), evaluate recon RMSE on a held-out uDALES split. Their S-model
  reports recon RMSE ~5.5e-3 on their PDE mix; treat that as an order-of-
  magnitude sanity bar, not a target.
- **`latent_type` during pre-training**: `sample` (VAE-proper) default;
  `mode` available for the deterministic-AE ablation.
- **Crop size for our grids**: pick per-dataset so crops tile the domain with
  minimal padding; expose in config, note the choice in the run's config.yaml.
