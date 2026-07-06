# Fine-Net-Only & Patch-Based Training — Implementation Plan

Status: **design approved, not yet implemented.** This is a working note in
`docs/plans/` (not a maintained reference). Verify every line number / signature
against the code before relying on it — the codebase moves.

Audience: an agent implementing this from scratch. Read
[docs/neural_surrogates.md](../neural_surrogates.md) and the existing
[docs/plans/dd_implementation_plan.md](dd_implementation_plan.md) first for the
domain-decomposition (DD) background.

---

## 1. Motivation

Training the current DD surrogate (`DomainDecomposed` + `PatchTrainer` +
`DomainDecompositionLoss`) on the Barcelona PALM case (grid `z=33, y=224, x=224`,
batch 4) measured **~20 s per training step** on an RTX 3090 — ~8 h/epoch, ×200
epochs = unusable. The cost is **fine-net compute**: the medium 4.8M-param 3-D
UNet runs `M=98` times per sample (`B·M = 392` forward+backward passes per step,
each on a 48³ block, plus gradient-checkpoint recompute). Increasing
`fine_chunk_size` does not help (compute-bound, not launch-bound); disabling
`fine_checkpoint` OOMs. Dataloading (~118 ms/sample) is fully hidden behind
compute and is **not** the bottleneck.

The fix the user wants is **architectural flexibility**, not just tuning:

1. **`coarse_net: null | <architecture>`** (surrogate definition). When `null`,
   the fine net predicts each patch's next state from only: the patch state at
   the previous step, the positional embedding, the parameters, and the patch
   geometry. When present, behaviour is exactly as today (coarse context field).
2. **`train_on_patches_only: bool`** (training). When `true`, the dataloader
   yields **individual patches** and the fine net trains on them directly (no
   full-grid op, no PoU merge) — extremely scalable, at the slight risk of
   interface mismatch. **Validation is still done on full states** (predict each
   patch, combine via partition-of-unity, score on the merged field).

This enables a **two-stage, separately-launched** fine-only workflow:

- **Stage 1** — train the fine net on patches with plain masked MSE. Maximally
  scalable (patches are i.i.d., trivially shardable).
- **Stage 2** — predict full states one patch at a time, merge via PoU, and train
  with the interface-aware loss. Warm-started from a path to the Stage-1 fine net.

A future **three-stage** path (train coarse alone → train fine on patches →
combine and fine-tune together) is explicitly **out of scope now** but must stay
easy to add later (see §8).

---

## 2. Approved design decisions

These were confirmed with the user; do not relitigate them:

1. **Smallest fine net.** With `coarse_net=null`, the fine stem ingests only the
   positional + geometry + params channels (`extra_in_channels = n_pos`). Do
   **not** reserve zero-fed coarse-context channels. (Consequence: warm-starting a
   future *combined* coarse+fine model from Stage-1 weights will need a deliberate
   stem-surgery step — that is accepted and deferred. See §8.)
2. **Mode bundles for the CLI.** Ship `patch_only` and `patch_interface`
   `mode` files that wire `coarse_net` + trainer + loss + datasets together as a
   one-line switch. The literal `coarse_net` and `train_on_patches_only` fields
   still exist and remain overridable on the CLI.
3. **Both stages share one model contract:** `DomainDecomposed` with
   `coarse_net=None`. Stage 1 differs only in *where* the loss is computed
   (per-patch, no merge) and *that validation still merges*. Inference/rollout
   (`NeuralSurrogateForwardModel`) is unchanged.
4. **Stage 2 needs no new trainer** — reuse `PatchTrainer` +
   `DomainDecompositionLoss` with `lambda_coarse=null` (and
   `lambda_divergence=null`). The loss already computes a term only when its
   lambda is set, and a coarse-less `info` dict simply omits `coarse_pred`.
5. **Stage 1 needs exactly one new trainer**, `FinePatchTrainer(BaseTraining)`,
   overriding only `_forward`. All other machinery (schedulers, AMP,
   channels-last, `torch.compile`, checkpoint/resume, early stopping, metrics,
   `fit()`) is inherited unchanged — this satisfies the "same capabilities"
   requirement.

---

## 3. Current code — what exists and is reused

| File | Role | Reuse |
|---|---|---|
| [architectures/domain_decomposed.py](../../libs/neural-surrogates/src/neural_surrogates/architectures/domain_decomposed.py) | `DomainDecomposed` wrapper | **modify** for `coarse_net=None` |
| [decomposition.py](../../libs/neural-surrogates/src/neural_surrogates/decomposition.py) | `DomainDecomposition` tiling/merge ops | reuse as-is |
| [architectures/unet_convnext.py](../../libs/neural-surrogates/src/neural_surrogates/architectures/unet_convnext.py) | fine/coarse sub-net; `forward(state, params, geometry, extra=None)` | reuse as-is |
| [datasets/transition.py](../../libs/neural-surrogates/src/neural_surrogates/datasets/transition.py) | `TransitionDataset` (full-field) | reuse for **validation** + Stage 2 |
| [datasets/patch.py](../../libs/neural-surrogates/src/neural_surrogates/datasets/patch.py) | `PatchTransitionDataset` (per-patch) | reuse for Stage 1 (add `include_coarse` flag) |
| [training/base.py](../../libs/neural-surrogates/src/neural_surrogates/training/base.py) | `BaseTraining` (fit loop, all machinery) | subclass |
| [training/patch.py](../../libs/neural-surrogates/src/neural_surrogates/training/patch.py) | `PatchTrainer` (full-field DD loss) | reuse for Stage 2 |
| [training/standard.py](../../libs/neural-surrogates/src/neural_surrogates/training/standard.py) | `Trainer._final_loss` (masked MSE) | pattern to copy for val path |
| [dd_loss.py](../../libs/neural-surrogates/src/neural_surrogates/dd_loss.py) | `DomainDecompositionLoss` (4-term) | reuse for Stage 2 |
| [scripts/neural_surrogate/train_neural_surrogate.py](../../scripts/neural_surrogate/train_neural_surrogate.py) | training entry point | **modify** for `val_dataset` + knob |

### 3.1 `PatchTransitionDataset` is already correct (verified)

`__getitem__` returns (see [datasets/patch.py:161-207](../../libs/neural-surrogates/src/neural_surrogates/datasets/patch.py#L161-L207)):

```
state_n_patch  (C, n+2h, n+2h, n+2h)   R_p S_t      extended block
delta_target   (C, n,    n,    n)       interior(S_{t+K} - S_t)
geometry_patch (1, n+2h, n+2h, n+2h)    R_p m
positional     (n_pos, n+2h, n+2h, n+2h) e_p
coarse_input   (C, *coarse_grid)        R_H S_t     (skippable, see §4.3)
coarse_geom    (1, *coarse_grid)        R_H m       (skippable)
coarse_target  (C, *coarse_grid)        R_H S_{t+K} (skippable)
neighbors      (6,) long
params_n       (K, P)
patch_index    () long
traj_index     () long
```

The interior is the centre crop `[h : h+n]` of the extended block. **K=1 only**
is correct/tested (a K>1 patch pushforward needs the merge, which only the model
owns — the docstring is explicit). Stage 1 uses K=1.

### 3.2 Fine-net contract (`UNetConvNeXt.forward`)

`forward(state, params, geometry, extra=None)`
([unet_convnext.py:410-492](../../libs/neural-surrogates/src/neural_surrogates/architectures/unet_convnext.py#L410-L492)):
`extra` must be provided **iff** `extra_in_channels > 0`, is concatenated **raw**
(no normalize, no geometry mask) after `[state, geometry]`, and with
`residual=True` the net returns the next state (`state + Δ`), geometry-masked.

### 3.3 How `DomainDecomposed.forward` uses the coarse net today

[domain_decomposed.py:288-398](../../libs/neural-surrogates/src/neural_surrogates/architectures/domain_decomposed.py#L288-L398):
coarse step builds `context = prolong(coarse_net(restrict_coarse(state)))`; the
fine step sets `extra = cat([context_blocks, pos_blocks], dim=1)` so
`extra_in_channels = n_state_channels (context C) + n_pos`. The `info` dict
carries `coarse_pred`, `patch_pred`, `context`, `num_patches`, `dd`.

### 3.4 `BaseTraining` hooks the new trainer relies on

- `fit()` drives `_train_epoch()` → `_validate()`; both call `self._forward(batch)`
  ([base.py:315-354](../../libs/neural-surrogates/src/neural_surrogates/training/base.py#L315-L354)).
  **Override `_forward` only** and everything else is inherited.
- `_prepare_batch` ([base.py:238-255](../../libs/neural-surrogates/src/neural_surrogates/training/base.py#L238-L255))
  caches `self._geometry`/`self._fluid_mask` lazily from the first **full-grid**
  batch's `batch["geometry"][0]`. The patch path must **not** use it; the val path
  sets it. Keep the two paths independent.
- AMP via `self._autocast()`; masked-MSE pattern in `Trainer._final_loss`.

---

## 4. File-by-file changes

### 4.1 `architectures/domain_decomposed.py` — support `coarse_net=None`

- `__init__`:
  - `has_coarse = coarse_net is not None`.
  - `extra_in = (n_state_channels if has_coarse else 0) + self.dd.n_pos`.
  - Build `self.fine_net` with that `extra_in` (unchanged call).
  - Build `self.coarse_net` only when `has_coarse`; else `self.coarse_net = None`.
  - Expose `self.has_coarse_net = has_coarse` for callers/asserts.
- `forward`:
  - When `self.coarse_net is None`: skip `restrict_coarse`/`coarse_net`/`prolong`;
    set `extra = pos_blocks` (positional only); build `info` **without**
    `coarse_pred` (omit the key — the Stage-2 loss with `lambda_coarse=null`
    never reads it). Keep `patch_pred`, `num_patches`, `dd`. `context` may be
    omitted or `None`.
  - When `self.coarse_net is not None`: unchanged.
- `set_normalization` ([domain_decomposed.py:222-234](../../libs/neural-surrogates/src/neural_surrogates/architectures/domain_decomposed.py#L222-L234)):
  iterate only over non-`None` nets (skip `None` coarse).
- **New helper** to share the "run fine net on blocks" contract between the
  wrapper and `FinePatchTrainer` (avoids drift in residual/crop semantics):

  ```python
  def fine_patch_forward(self, state_blocks, params_blocks, geom_blocks, pos_blocks):
      """Run fine_net on an extended-block batch (coarse-less extra = positional).
      Returns the extended (n+2h) output; caller crops the interior [h:h+n]."""
      extra = pos_blocks  # coarse-less: positional only
      return self._run_fine(state_blocks, params_blocks, geom_blocks, extra)
  ```

  (`_run_fine` already handles chunking + checkpointing.)

- **Guard:** if `coarse_net is None` *and* the model is asked to produce coarse
  intermediates, raise a clear error rather than `KeyError` later.

### 4.2 `training/patch_fine.py` — NEW `FinePatchTrainer(BaseTraining)`

Export from `training/__init__.py` and package `__init__.py` (add to `__all__`).

Override **only** `_forward(batch)` to branch on batch schema:

```python
def _forward(self, batch):
    if "state_n_patch" in batch:           # Stage-1 patch batch
        return self._patch_loss(batch)
    return self._full_state_loss(batch)    # validation (and any full-grid batch)
```

- `_patch_loss(batch)` (train):
  - Move `state_n_patch`, `geometry_patch`, `positional`, `params_n`,
    `delta_target` to device (channels-last for `state_n_patch` if
    `self.channels_last`).
  - `K = params_n.shape[1]`; `params_last = params_n[:, K-1]`.
  - `n, h = model.dd.interior_size, model.dd.halo` (use `self._eager_model` to
    bypass any `torch.compile` wrapper).
  - Under `self._autocast()`: `pred_ext = self._eager_model.fine_patch_forward(
    state_n_patch, params_last, geometry_patch, positional)`.
  - Crop interior: `pred_int = pred_ext[:, :, h:h+n, h:h+n, h:h+n]`.
  - `target_int = state_n_patch[:, :, h:h+n, h:h+n, h:h+n] + delta_target`.
  - Mask by `geometry_patch[:, :, h:h+n, ...] > 0` and apply `self.loss_fn`
    (MSELoss) on selected cells (mirror `Trainer._final_loss`). Respect
    `self.mask_loss`.
- `_full_state_loss(batch)` (validation / full-grid):
  - `state, state_next, params, geometry = self._prepare_batch(batch)`.
  - `merged = self.model(state, params[:, -1], geometry)` (no intermediates →
    PoU-merged full field).
  - Masked MSE on `self._fluid_mask` exactly like
    [standard.py:19-33](../../libs/neural-surrogates/src/neural_surrogates/training/standard.py#L19-L33).

Notes:
- **Pushforward curriculum off** for Stage 1: set `pushforward_epochs_per_step:
  null`, `pushforward_start_steps: 1`, `dataset.pushforward_steps: 1`. With null,
  `_pushforward_steps_for_epoch` returns the max (1); the one-time
  `set_pushforward_steps(1)` + refork on epoch 0 is harmless.
- Assert at init that `self.model` is a `DomainDecomposed` with
  `has_coarse_net is False` and `model.dd.n_pos` equals the dataset's positional
  channel count (consistency guardrail — see §7).
- No `_aux_terms` (single MSE term); base logging handles it.

### 4.3 `datasets/patch.py` — add `include_coarse: bool = True`

When `False`, skip computing/returning `coarse_input`, `coarse_geom`,
`coarse_target` (pure efficiency for the coarse-less path; avoids
`restrict_coarse` per item). No correctness change. Default `True` keeps existing
callers byte-identical.

### 4.4 New architecture presets

`conf/neural_surrogate/architectures/domain_decomposed/fine_only_medium.yaml`
(and a `fine_only_small.yaml`): same as
[domain_decomposed/medium.yaml](../../conf/neural_surrogate/architectures/domain_decomposed/medium.yaml)
but the `defaults` list references **only** `@fine_net` (no `@coarse_net`), and
the body sets `coarse_net: null`. Keep `_recursive_: false`, `_convert_: all`,
the `decomposition:` block, and `fine_chunk_size`/`fine_checkpoint`. Example:

```yaml
defaults:
  - /neural_surrogate/architectures/unet_convnext@fine_net: medium
  - _self_
_target_: neural_surrogates.DomainDecomposed
_recursive_: false
_convert_: all
coarse_net: null
decomposition:
  interior_size: 32
  halo: 8
  taper: 4
  coarsen_factor: 4   # unused when coarse_net is null but harmless
  n_pos: 3
  boundary_mode: replicate
  periodic_axes: [false, true, false]
  geometry_coarsen: any_fluid
fine_chunk_size: 8
fine_checkpoint: true
```

### 4.5 New `mode` files

`conf/neural_surrogate/mode/patch_only.yaml` (`# @package _global_`):
- `defaults: [/neural_surrogate/architectures@architecture: domain_decomposed/fine_only_medium]`
- `trainer._target_: neural_surrogates.FinePatchTrainer`
- `train_on_patches_only: true`
- `loss: {_target_: torch.nn.MSELoss}`
- `dataset: {_target_: neural_surrogates.PatchTransitionDataset,
  include_coarse: false, decomposition: ${architecture.decomposition},
  pushforward_steps: 1}` (other fields inherited from `training.yaml`'s `dataset`)
- `val_dataset: {_target_: neural_surrogates.TransitionDataset}` (full-field)
- `model_name: dd_fine_only_patch_barcelona`
- Override curriculum/horizon: `trainer.pushforward_epochs_per_step: null`.

`conf/neural_surrogate/mode/patch_interface.yaml` (Stage 2):
- Same coarse-less architecture default.
- `trainer._target_: neural_surrogates.PatchTrainer`
- `train_on_patches_only: false`
- `loss: {_target_: neural_surrogates.DomainDecompositionLoss,
  lambda_interface: 0.1, lambda_coarse: null, lambda_divergence: null,
  velocity_channels: [0,1,2], mask_loss: true}`
- `dataset: {_target_: neural_surrogates.TransitionDataset}` (full-field)
- `model_name: dd_fine_only_interface_barcelona`
- User supplies `init_weights_path=<stage1 weights>` on the CLI.

Using `${architecture.decomposition}` interpolation keeps the dataset's tiling
**identical** to the model's (single source of truth). Note: this picks up
`decomposition.periodic_axes` directly; if anyone later drives periodicity via the
letter-form `architecture.periodic_axes` override, replicate it into the dataset
too (or assert equality at init).

### 4.6 Training script `train_neural_surrogate.py`

- **`val_dataset` support:** if `cfg.get("val_dataset")` is not `None`,
  instantiate it with `split="val"` for the val loader; else reuse `cfg.dataset`
  with `split="val"` (current behaviour). Stage 1: train = `PatchTransitionDataset`,
  val = `TransitionDataset`.
- **`train_on_patches_only`:** read `cfg.get("train_on_patches_only", False)`. When
  true, assert the trainer target is `FinePatchTrainer` and the dataset target is
  `PatchTransitionDataset` (guardrail). When false, current behaviour.
- **Normalization:** `_compute_normalization_stats` already works on
  `PatchTransitionDataset` (it inherits `_geometry`, `_state_files`, `_params`),
  and `DomainDecomposed.set_normalization` forwards only to the fine net when
  coarse is `None`. No change needed beyond confirming it runs.
- Keep the existing `def run(cfg)` + thin `@hydra.main` shape and
  `resolve_output_dir`/`model_weights/<model_name>` layout.

---

## 5. Two-stage workflow (separate launches)

```bash
# Stage 1 — fine net on patches, masked MSE, full-state validation
pixi run -e cuda python scripts/neural_surrogate/train_neural_surrogate.py \
    neural_surrogate/mode@_global_=patch_only

# Stage 2 — full-state predict-per-patch + PoU merge, interface loss, warm-started
pixi run -e cuda python scripts/neural_surrogate/train_neural_surrogate.py \
    neural_surrogate/mode@_global_=patch_interface \
    init_weights_path=model_weights/dd_fine_only_patch_barcelona/weights.pt
```

**Run with `-e cuda`, not `-e dev`.** The `dev` env is CPU-only torch
(`+cpu`, no bf16, no triton); `cuda` is the GPU env (`+cu126`, bf16 + triton). CPU
`dev` is for smoke tests only.

---

## 6. Why this scales (the priority)

- Stage 1 items are independent 48³ blocks: **no full-grid tensor, no merge, no
  gradient checkpointing.** Batch = many patches; the GPU saturates; patches are
  i.i.d. so the split shards trivially across workers/GPUs/nodes.
- Per-step cost is **constant per patch**, independent of total domain size;
  dataset length scales as `(T−1)·M`. Larger cases add patches, not per-step cost.
- The only full-grid op is **validation** (PoU merge), once per epoch over the
  small val split.
- Stage 2 is the existing full-field DD path (heavier) but runs from a good
  Stage-1 init and only needs to repair interface seams.

---

## 7. Risks / watch-items

- **Decomposition consistency** between dataset and model — enforced by the
  `${architecture.decomposition}` interpolation; additionally assert `n_pos` and
  tiling match at `FinePatchTrainer.__init__`.
- **Geometry cache** in `BaseTraining._prepare_batch` is full-grid only; keep the
  patch path from touching `self._geometry`/`self._fluid_mask`.
- **`extra_in_channels` mismatch:** coarse-less fine net must be built with
  `extra_in = n_pos` (positional only). The patch dataset returns `positional`
  with exactly `n_pos` channels — they must agree.
- **Interface seams** (the accepted "slight risk of mismatch"): Stage 1 alone may
  show seam artifacts in autoregressive rollout; Stage 2 is what repairs them.
- **`torch.compile`:** if enabled later, compile the **inner `fine_net`** (static
  block shape, grid-size-independent), never the wrapper — compiling the wrapper
  (Python chunk loop, `checkpoint()`, dict return, `id()`-keyed plan cache) is the
  known slow/ineffective path.

---

## 8. Future three-stage path (coarse → fine → combined) — NOT now

Keep easy to add later; do not build:

- A `coarse_only` trainer/mode: train `coarse_net` alone with MSE on the coarse
  grid (`restrict_coarse(target)`), validating on the merged full field via a
  coarse-only forward.
- **Combining** a patch-trained fine net with a coarse net requires the fine stem
  to ingest the C context channels (`extra_in = n_pos + C`). Because we chose the
  *smallest* Stage-1 net (`extra_in = n_pos`), warm-starting the combined model
  needs a deliberate **stem-surgery** step: load all fine-net weights except the
  stem input conv, and initialise the new context input channels to zero. Provide
  this as an explicit `init_weights` adapter when the combined mode is added.
- The wrapper already supports the combined model unchanged (it is today's
  `coarse_net != None` path); only the warm-start adapter is new.

---

## 9. Testing (smoke-shaped, fast — see `tests/conftest.py`)

Add to the neural-surrogate test suite (e.g.
[tests/test_neural_surrogate_forward_model.py](../../tests/test_neural_surrogate_forward_model.py)
or a sibling trainer test):

1. `DomainDecomposed(coarse_net=None)` — forward returns a correctly shaped merged
   field, `set_normalization` is a no-op-safe, gradients flow, `info` (with
   `return_intermediates=True`) omits `coarse_pred` but carries `patch_pred`.
2. `FinePatchTrainer` — one epoch on a tiny `PatchTransitionDataset` (train) +
   `TransitionDataset` (val); asserts train uses the patch path, val uses the
   merge path, checkpoint/resume + metrics.csv produced.
3. Stage-2 `PatchTrainer` on a coarse-less model with `lambda_coarse=null` runs an
   epoch (loss has `one_step` + `interface`, no `coarse`).
4. Warm-start round-trip: Stage-1 `weights.pt` loads into the Stage-2 model
   (`init_weights_path`) without shape errors.

Run: `pixi run -e dev py.test` (CPU smoke). Then `pixi run -e dev pre-commit`
before committing (black + isort + mypy). Branch first
(`feat/fine-only-patch-training`) — never commit to `main`.

---

## 10. Implementation order

1. `DomainDecomposed`: `coarse_net=None` path + `fine_patch_forward` helper.
2. `FinePatchTrainer` + exports.
3. `PatchTransitionDataset.include_coarse`.
4. `fine_only_medium` / `fine_only_small` presets.
5. `patch_only` / `patch_interface` mode files.
6. Train script: `val_dataset` + `train_on_patches_only` wiring.
7. Smoke tests; `pre-commit`; CPU smoke run of `patch_only`.
8. (User runs the real GPU training on `-e cuda`.)

Keep [docs/neural_surrogates.md](../neural_surrogates.md) in sync if any
documented contract changes (new modes, the `coarse_net=null` option, the new
trainer).
