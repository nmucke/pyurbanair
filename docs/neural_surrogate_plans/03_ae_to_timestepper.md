# Plan 03 — Pre-trained autoencoder → time-stepping predictor (Tadpole DFT)

**Status: implemented (2026-07-09).** Delivered: the vendored `TadpoleDFT` +
downstream sub-network subtree (`architectures/_tadpole/model/dft.py`,
`architecture/downstream/`, optional-dep-shimmed for the no-triton/no-mamba/no-flash
env), the `TadpoleTimeStepper` + `ParamConditionedSubnetwork` wrapper
(`architectures/tadpole_stepper.py`) sharing field IO with `TadpoleAE` via the
`_TadpoleFieldIO` mixin (`architectures/_tadpole_field_io.py`), the `tadpole_encdec`
LoRA preset (`finetuning/targets.py`), the `dft` finetune mode
(`conf/neural_surrogate/finetune_mode/dft.yaml` + the DFT dispatch in
`scripts/neural_surrogate/finetune_neural_surrogate.py`), and
`tests/test_ae_to_timestepper.py`. See `docs/neural_surrogates.md` Part H. **Wiring
decisions baked in:** the residual is *intrinsic* — `state_next = dft_state * mask`
(the DFT morphs its own reconstruction toward `u_{t+Δt}`; `predict_residual` is a
no-op framing). The honest init invariant is `stepper(state) ==
stepper._ae_reference_recon(state, geometry)` **exactly** (zero-init sub-network + γ
skips ⇒ the DFT output equals the plain-AE reconstruction), *not* `state_next ==
state` (a perfectly-reconstructing-AE / training outcome). Deploy stamps
`skip_pretrained_load: true` + `pretrained_ae_dir: null` into the saved config so
ESMDA rebuilds the net from the merged `weights.pt` without the AE dir.

**Goal:** load a pre-trained autoencoder (plan 02) and fine-tune it into a
next-step predictor with Tadpole's DFT recipe — latent sub-network +
γ-gated reintroduced skip connections + LoRA on encoder/decoder — trained
through the fine-tuning pipeline from plan 01, and deployable as an ESMDA
forward model.

Depends on plans 01 (fine-tune config/script, PEFT helpers) and 02
(`TadpoleAE`, `encoder.pt`/`decoder.pt` artifacts, tadpole dependency).

## 0. What `tadpole.model.dft.TadpoleDFT` gives us (verified)

- Builds `_KLP3DEncoder` + `_P3DDecoder`, loads `weight_encoder`/
  `weight_decoder`, sets fine-tune state (frozen / FPFT / GIFt-LoRA-rank),
  then wraps both in `KLP3DEncoderSkip`/`P3DDecoderSkip` — the skip wrappers
  that expose encoder residuals to the decoder, gated by zero-initialized
  trainable scales (paper's γ; verify the init lives in `skip_wrapper.py` at
  implementation time).
- Latent sub-network: `subnetwork="default"` → `SequentialModel` (encoder-only
  transformer over flattened latent tokens, hyper-attention;
  `in_dim = input_channels × latent_channels` per size), applied as
  `x = latent_residual_scale * x + subnetwork(x, *args, **kwargs)` — note the
  pass-through `*args/**kwargs`: **our conditioning hook**.
- Forward: `(B, C, X, Y, Z) → (B, C, X, Y, Z)`; channels folded to batch,
  crops handled as in the AE. Zero-init of sub-network/LoRA/γ ⇒ the model
  starts as the identity autoencoder (predicts u_t), and training moves it
  toward u_{t+Δt}.

Per the master plan's unified LoRA strategy: construct with
`encoder_ft_state="frozen"`, `decoder_ft_state="frozen"` and apply **our PEFT
injection** (plan 01) on `dft.encoder`/`dft.decoder` — not the bundled GIFt
path — so `lora.variant: standard|balora` works here too. Keep
`hyper-attention` sub-network only if its triton kernels resolve in our env;
otherwise pass a custom sub-network (§1) — it's a constructor arg, no fork
needed.

## 1. Wrapper: `architectures/tadpole_stepper.py`

`class TadpoleTimeStepper(nn.Module)` — adapts `TadpoleDFT` to our forward
contract `forward(state, params, geometry, geom_features=None) -> state_next`
(what `Trainer._forward` and `NeuralSurrogateForwardModel._rollout_chunk`
call):

- **Constructor**: `size, n_state_channels, n_params,
  pretrained_ae_dir (→ encoder.pt/decoder.pt + the AE's config.yaml),
  subnetwork cfg (n_layers/heads/hidden override or "default"),
  lora handled externally (plan 01), latent_type="mode" (deterministic
  rollouts by default; "sample" is a knob), encoder_crop_size,
  predict_residual=true, normalize=true, encode_geometry (inherited from the
  AE config — must match, cross-check like the SDF flags)`.
- **Normalization**: copy the `state_mean/std, param_mean/std` buffers from
  the pre-trained AE (they were baked into its crops' statistics); expose
  `set_normalization` but default to inheriting.
- **Geometry**: mask input/output like `P3D` (`state * geometry`); with
  `encode_geometry=true` the mask/SDF channels ride through the frozen
  encoder exactly as in pre-training, so the latent token sequence the
  sub-network sees includes geometry tokens. The geometry channels of the
  *output* are discarded (geometry is static).
- **Parameter conditioning** (Tadpole has none; our key addition): a
  `ParamConditionedSubnetwork` wrapping/replacing `SequentialModel` —
  `params (B, P)` → z-scored → small MLP → either
  (a) **FiLM/adaLN** scale-shift on the latent tokens (zero-init out layer,
  preferred: matches both Tadpole's zero-init philosophy and P3D's native
  adaLN conditioning), or
  (b) a prepended parameter token.
  Plan (a) as the default, (b) behind a config enum. Absent params
  (`n_params == 0`) ⇒ no conditioning module at all (repo's no-op rule).
  Wiring: the wrapper's forward passes `params` down via the
  `subnetwork(x, params=...)` kwargs pass-through noted in §0.
- **Residual prediction**: `state_next = state + out` (matches our `P3D`
  convention and the identity-init story; the zero-init DFT then starts at
  exactly `state_next = state`... note the base DFT already outputs ~u_t, so
  with `predict_residual=true` we must subtract: out = dft(state) - state +
  state — resolve this carefully at implementation time so the init is
  identity exactly once, and cover it with the parity unit test).
- Expose `n_geom_feature_channels`/`sdf_clamp_cells` attrs so the existing
  train/fine-tune script cross-checks keep working.

## 2. Training path: `finetune_mode: dft` (extends plan 01)

New group file `conf/neural_surrogate/finetune_mode/dft.yaml`:

```yaml
# @package _global_
architecture:
  _target_: neural_surrogates.TadpoleTimeStepper
  size: S                      # must match the AE; cross-checked
  pretrained_ae_dir: ???       # model_weights/tadpole_ae_s
  subnetwork: default
  param_conditioning: film     # film | token | none
  latent_type: mode
trainer: { _target_: neural_surrogates.Trainer }   # the EXISTING next-step trainer
loss:    { _target_: torch.nn.MSELoss }
lora:
  rank: 32
  alpha: 64
  target_preset: tadpole_encdec   # targets.py preset: Linear+Conv3d inside dft.encoder/dft.decoder
trainable_modules:                # fully-trained NEW modules (not LoRA):
  - subnetwork                    # latent transformer (+ param conditioning)
  - skip_scales                   # the γ gates
  - latent_residual_scale
```

Script changes in `finetune_neural_surrogate.py` (small, mode-dispatched):

- `pretrained_model_dir` for this mode is the **AE** dir; the architecture is
  instantiated fresh (encoder/decoder weights loaded inside the wrapper), so
  step 1–2 of the plan-01 flow become "instantiate TadpoleTimeStepper".
- Freezing: freeze everything, then unfreeze `trainable_modules` + injected
  LoRA params. (Plan 01's `inject_lora` handles the LoRA half;
  `modules_to_save` can carry `trainable_modules` if PEFT wrapping is
  cleaner than manual unfreezing — decide by whichever keeps the merged
  export simple.)
- Everything else is identical: dataset = `TransitionDataset` on the target
  data, pushforward curriculum, masked MSE, merged `weights.pt` +
  `adapter/` export. **The DFT stage is "just" a plan-01 fine-tune with a
  different architecture + extra trainable modules** — that's the design
  point of doing 01 first.
- Rollout stability: start with `grad_unroll_steps`/pushforward settings as in
  `training.yaml`; the paper leans on uncropped-encoding translation
  equivariance for rollouts — our wrapper always encodes the full (padded)
  field, so no crop-boundary accumulation issue.

## 3. ESMDA integration

- Export shape is standard (`config.yaml` + merged `weights.pt`), and the
  forward contract matches, so `NeuralSurrogateForwardModel` mostly works.
  Two loader touch-ups in `forward_model.py`:
  1. Architecture instantiation: `TadpoleTimeStepper` needs
     `pretrained_ae_dir` at build time only to fetch encoder/decoder weights —
     but the merged `weights.pt` already contains them. Add a
     `skip_pretrained_load: true`-style constructor flag (or make
     `pretrained_ae_dir` nullable) so deployment doesn't depend on the AE dir
     still existing. The saved `config.yaml`'s architecture node is written
     with that flag set.
  2. Domain check: fixed-grid, same as P3D — no `domain_flexible` claims.
- Ensemble path (`clone_for_member` shallow-shares the net) is unaffected.
- e2e smoke: AE-pretrain (2 epochs) → DFT fine-tune (2 epochs) → `run_esmda`
  on the fixture shapes, serially (memory: serial e2e only).

## 4. Tests

- Identity-at-init parity: freshly built `TadpoleTimeStepper` (zero-init
  sub-network/γ/LoRA) returns `state_next == state` (up to
  latent sampling — use `latent_type: mode`) — this is the single most
  informative unit test of the whole DFT wiring.
- Param-conditioning no-op: `param_conditioning: none` and `n_params=0` paths
  produce identical module trees to a param-free build.
- Cross-checks fire: size mismatch AE↔stepper, `encode_geometry` mismatch.
- e2e as in §3.

## 5. Open questions (defaults chosen, revisit with results)

- **Freeze-only vs LoRA on encoder/decoder**: paper default is LoRA rank 32 on
  both; `encoder/decoder_ft_state`-style config lets us ablate frozen
  (cheapest) vs LoRA vs full FT. Config: `lora.rank: 0` ⇒ frozen.
- **`latent_type` at fine-tune/rollout**: `mode` default for deterministic
  ESMDA rollouts; `sample` becomes interesting again with BaLoRA-style
  stochastic ensembles (plan 04).
- **Sub-network attention backend**: their `hyper` attention needs triton;
  fall back to standard `nn.MultiheadAttention` in `ParamConditionedSubnetwork`
  if triton is unavailable on target machines (DelftBlue/Snellius nodes vary).
