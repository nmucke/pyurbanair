# Master plan: foundation-model pre-training + LoRA fine-tuning for neural surrogates

**Status: plan (2026-07-06). Not yet implemented.**

This folder holds the implementation plan for extending `libs/neural-surrogates`
with (a) a Tadpole-style foundation-model setup (autoencoder pre-training →
time-stepping fine-tuning) and (b) a LoRA-based fine-tuning pipeline that works
both for the existing next-step models (focus: P3D) and for the new
autoencoder-derived predictors. Two LoRA variants are in scope: standard LoRA
(via HF PEFT) and Bayesian LoRA (BaLoRA, implemented as a PEFT-compatible
custom layer).

## Source material

- **Tadpole** — Liu, Koehler, Holzschuh, Thuerey,
  [arXiv:2605.15284](https://arxiv.org/abs/2605.15284),
  code: <https://github.com/tum-pbs/Tadpole> (Apache-2.0, pip-installable),
  pretrained weights: HF `thuerey-group/Tadpole` (sizes S/B/L; 8.8M/38.1M/152.1M
  params; latent compression 16/8/4).
  Recipe: pre-train the P3D backbone as a **VAE** (optionally + adversarial
  loss) on **single-channel spatial crops** (channels folded into batch, crop
  size 64³), with encoder↔decoder skip connections removed. Fine-tune to a
  time-stepper ("DFT") by adding (i) a latent transformer sub-network with a
  residual connection, (ii) reintroduced skip connections gated by
  zero-initialized trainable scales, (iii) LoRA (default rank 32) on the frozen
  encoder/decoder. Everything zero-initialized so the fine-tuned model starts as
  the identity autoencoder.
- **BaLoRA** — Coscia, Löwe, Welling,
  [arXiv:2605.08110](https://arxiv.org/abs/2605.08110).
  Only the LoRA down-projection `θ_A` is stochastic:
  `ω_A = θ_A + sqrt(α(x)·θ_A²)·ε`, with `α(x)` produced per layer by a small
  inference network on pooled input features. Trained with an ELBO (single MC
  sample, closed-form KL against a dropout-informed prior). Inference is either
  deterministic (merge mean weights, zero overhead) or sampling (S forward
  passes with rank-r noise, aggregate mean/variance).

## Key facts about the current codebase these plans build on

- `libs/neural-surrogates` is pure **PyTorch**. Our `P3D`
  (`architectures/p3d.py`) is a thin wrapper around the external
  `p3d_surrogate` package and **stays untouched** as the straightforward
  next-step forward model. The Tadpole path uses the **`tadpole` package's own
  P3D implementation** (`tadpole/architecture/p3d/`) instead of doing surgery
  on `p3d_surrogate`.
- Training: `BaseTraining` (`training/base.py`) + `Trainer` (`training/standard.py`),
  driven by `scripts/neural_surrogate/train_neural_surrogate.py` with the Hydra
  config `conf/neural_surrogate/training.yaml` (+ `mode/` group). Best weights =
  bare `state_dict` at `model_weights/<model_name>/weights.pt`, resolved config
  saved alongside as `config.yaml`, resume checkpoint `checkpoint.pt`, CSV
  metrics.
- Deployment contract (ESMDA): `NeuralSurrogateForwardModel`
  (`forward_model.py`) rebuilds the architecture by `instantiate(config.yaml →
  architecture)` and does a plain `load_state_dict(torch.load(weights.pt))`,
  then rolls out via `model(state, params, geometry)`. **Every new model kind
  must end up as a `model_dir` with that shape** (or extend the loader — plan 03
  covers the extension).
- There is already a crude full-weight fine-tune hook (`init_weights_path` in
  the train script); the LoRA pipeline supersedes it for parameter-efficient
  fine-tuning but does not remove it.
- Relevant repo conventions: no-op defaults (new knobs must leave existing runs
  byte-identical), tests compose Hydra configs and call `run(cfg)` on smoke
  shapes (`tests/conftest.py`), pre-commit before committing, docs updated in
  the same PR.

## Deliverables and sub-plans

| # | Plan | Delivers | Depends on |
|---|------|----------|------------|
| 1 | [01_lora_finetuning.md](01_lora_finetuning.md) | `neural_surrogates.finetuning` module (PEFT injection, adapter save/merge/export), `conf/neural_surrogate/finetuning.yaml`, `scripts/neural_surrogate/finetune_neural_surrogate.py`. Fine-tunes an existing trained model (focus P3D) on new data, training only LoRA weights, exporting an ESMDA-compatible `model_dir`. | — |
| 2 | [02_autoencoder_pretraining.md](02_autoencoder_pretraining.md) | `tadpole` optional dependency, `TadpoleAE` wrapper architecture, `SnapshotDataset`, `AutoencoderTrainer` (new `training/autoencoder.py`), `conf/neural_surrogate/pretrain_autoencoder.yaml`, `scripts/neural_surrogate/pretrain_autoencoder.py`. VAE core; adversarial loss is a scoped optional extension. | — |
| 3 | [03_ae_to_timestepper.md](03_ae_to_timestepper.md) | `TadpoleTimeStepper` wrapper around `TadpoleDFT` (params/geometry/SDF conditioning, our forward contract), fine-tune path from a pre-trained autoencoder to a next-step predictor, ESMDA integration. | 1, 2 |
| 4 | [04_bayesian_lora.md](04_bayesian_lora.md) | `BaLoRALinear`/`BaLoRAConv3d` as PEFT custom layers, ELBO trainer hook, deterministic + sampling inference modes, ensemble/DA weight-sampling design. | 1 (and 3 for the Tadpole path) |
| 5 | [05_latent_space_pretraining_extensions.md](05_latent_space_pretraining_extensions.md) | Research report + plan for extended `TadpoleAE` trainings: JEPA-inspired geometry-conditioned latent prediction, masked "flow given geometry" inpainting, denoising/free-bits/near-wall upgrades, temporal latent regularizers, and the latent-evaluation harness (geometry-held-out splits, probes, RankMe). Extends 2, feeds 3; phases E0–E5. | 2 (and 3 for the warm-start payoff) |

## Implementation order and rationale

Implement in the numbered order:

1. **Standard LoRA fine-tuning (plan 01) first.** It is immediately useful on
   the current P3D models, needs no new heavy dependencies beyond `peft`, and
   builds the machinery (injection helpers, adapter checkpoints, merged
   export, fine-tune config/script shape) that phases 3 and 4 reuse. It also
   forces the two make-or-break verifications early (PEFT×Conv3d,
   PEFT×`torch.compile`×`BaseTraining` checkpointing).
2. **Autoencoder pre-training (plan 02).** Independent of 01; brings in the
   `tadpole` dependency, the snapshot dataset and the AE trainer. Can start in
   parallel with 01 if desired — nothing in 02 imports 01.
3. **AE → time-stepper (plan 03).** Needs both: the Tadpole wrappers from 02
   and the fine-tune pipeline from 01 (the DFT stage *is* a LoRA fine-tune with
   extra new modules).
4. **BaLoRA (plan 04) last.** It slots into the layer abstraction from 01 and
   is the most experimental piece (stochastic training, ELBO, sampling
   inference). Doing it after 03 means it lands on both fine-tuning targets at
   once.

Each phase is a separate branch/PR, each ending green on
`pixi run -e dev py.test` with new smoke tests.

## Unified LoRA strategy (the one decision that spans all plans)

Use **HF PEFT as the single LoRA stack everywhere**:

- Plain models (our P3D wrapper etc.): `peft.LoraConfig(target_modules=...)` +
  `get_peft_model` directly on the architecture.
- Tadpole encoder/decoder: construct `TadpoleAutoencoder`/`TadpoleDFT` with
  `encoder_ft_state="frozen"` / `decoder_ft_state="frozen"` and apply **our**
  PEFT injection on top, instead of the integer-rank path that would use the
  bundled `GIFt` library. One code path means the BaLoRA variant (a PEFT
  custom-dispatch layer) works on both tracks, and switching
  `lora.variant: standard ↔ balora` is a config change.
- `GIFt` remains available as a fallback (it ships with `tadpole` anyway) if
  PEFT turns out to fight Tadpole's module structure; note it in the code but
  don't build on it.

**Verify at the start of phase 1** (cheap, in a scratch script):
`peft` version pinned in the dev env must provide `lora.Conv3d` (added ~v0.11)
and regex `target_modules` on non-transformers models; and a
`get_peft_model`-wrapped module must survive our `BaseTraining` flow
(`torch.compile` off first, `_eager_model` state-dict save/load). If Conv3d
support is missing or broken, the fallback is the custom-dispatch mechanism
from plan 04 with a plain (deterministic) low-rank layer — same public API.

## Artifact layout (shared convention across plans)

Everything continues to live under `model_weights/<model_name>/`:

```
model_weights/<name>/
  config.yaml           # resolved Hydra config (already the convention)
  weights.pt            # full state dict — ALWAYS present, ESMDA loads this
  adapter/              # NEW, fine-tuned models only
    adapter_model.safetensors   # PEFT adapter weights (LoRA/BaLoRA only)
    adapter_config.json         # PEFT config
  checkpoint.pt, metrics.csv    # training-loop artifacts (unchanged)
```

Rule: `weights.pt` is always a **full, merged, plain state dict** loadable into
the freshly instantiated architecture, so `NeuralSurrogateForwardModel`
continues to work with zero changes for deterministic models. The `adapter/`
dir is the parameter-efficient record (small, portable, re-appliable to the
base weights). BaLoRA models additionally need the adapter at inference time
when sampling is on (the stochastic parameters can't be merged) — plan 04.

## Config entry points after all phases

```
conf/neural_surrogate/training.yaml              # existing next-step training (unchanged)
conf/neural_surrogate/pretrain_autoencoder.yaml  # NEW: Tadpole-style (V)AE pre-training
conf/neural_surrogate/finetuning.yaml            # NEW: all fine-tuning; mode group selects
                                                 #   lora_nextstep | dft (AE→predictor)
                                                 #   and lora.variant: standard | balora
```

Both new entry points follow the `def run(cfg)` + thin `@hydra.main` wrapper
shape so tests can compose them (`compose_test_cfg`).

## Risks / open items tracked across plans

- **PEFT ↔ Conv3d ↔ torch.compile**: verified first thing in phase 1 (above).
- **Tadpole package maturity**: v0.0.1, training tutorials still "to be
  added" upstream — we write our own trainers (which we want anyway); pin the
  dependency to a commit hash. Its deps (`torchfsm`, `diffusers`, `timm`,
  `GIFt` from git) must resolve in the pixi dev env; make it an optional extra
  like `[p3d]`.
- **HF pretrained weights vs from-scratch pre-training**: plan 02 supports
  both (start from `thuerey-group/Tadpole` weights and continue pre-training on
  our flow data, or pre-train from scratch). Which works better on urban-flow
  data is an experiment, not a design decision.
- **`BaseTraining` assumptions**: it is transition/pushforward-shaped; the AE
  trainer reuses its infrastructure (amp, LR schedule, checkpointing, CSV) but
  not the rollout machinery — plan 02 details the split so we don't fork the
  whole class.
- **BaLoRA in the DA ensemble**: sampling weights per ensemble member is a
  natural model-error representation for ESMDA; plan 04 designs it behind a
  config knob without touching default behavior.
