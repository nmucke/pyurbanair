# Plan 01 — LoRA fine-tuning of existing next-step models (PEFT)

**Status: implemented (2026-07-08).** Delivered:
`neural_surrogates.finetuning` (`inject.py`, `targets.py`),
`BaseTraining.weights_transform`, `conf/neural_surrogate/finetuning.yaml` +
`finetune_mode/lora_nextstep`, `scripts/neural_surrogate/finetune_neural_surrogate.py`,
`tests/test_lora_finetuning.py`. See `docs/neural_surrogates.md` Part E. Pre-flight
findings baked in: peft 0.19 crashes merging a **1×1×1** Conv3d LoRA and rejects
**grouped** convs at inject — both are excluded from the auto presets so they
always inject *and* merge cleanly.

**Goal:** take an already-trained surrogate (focus: the `P3D` wrapper in
`architectures/p3d.py`, but architecture-agnostic), inject LoRA adapters,
train **only** the adapter weights on new data with the existing `Trainer`
machinery, and export a fine-tuned `model_dir` that drops into
`NeuralSurrogateForwardModel` unchanged.

Driven by a **separate config file** (`conf/neural_surrogate/finetuning.yaml`)
that points at the pre-trained weights and at the new save location, per the
session brief.

## 0. Pre-flight verification (do this before writing any real code)

In a scratch script (not committed), on the dev env:

1. Pin/install `peft` (latest; must be ≥ the version that ships
   `peft.tuners.lora.Conv3d`, ~v0.11) and confirm
   `get_peft_model(P3D(...), LoraConfig(target_modules=r"..."))` wraps both
   `nn.Linear` and `nn.Conv3d` modules inside `p3d.net`
   (print `peft_model.targeted_module_names`).
2. Confirm a wrapped model runs our forward contract
   `model(state, params, geometry, geom_features)` untouched (PEFT wraps
   leaf modules; the outer forward signature is preserved via
   `peft_model.base_model.model` — decide whether the trainer receives the
   `PeftModel` or the still-wrapped base module, see §3).
3. Confirm `state_dict` round-trip and `merge_and_unload()` produce a state
   dict loadable into a fresh `P3D` instance with byte-identical outputs.
4. Check interaction with `BaseTraining`: `compile_model: false` first;
   then verify `torch.compile` on the PEFT-wrapped model (known to work in
   recent versions, but our recompile-limit + P3D dynamic-shape settings need
   an A/B). If compile breaks, document `compile_model: false` as required for
   fine-tuning runs — acceptable, runs are short.

Outcome gates the approach; fallback is a custom dispatch layer (plan 04's
mechanism) with the same public helpers.

## 1. New module: `neural_surrogates/finetuning/`

```
libs/neural-surrogates/src/neural_surrogates/finetuning/
  __init__.py        # re-exports: inject_lora, save_adapter, load_adapter,
                     #             merge_to_state_dict, TargetSpec presets
  inject.py          # PEFT wiring
  targets.py         # per-architecture default target_modules
```

### `inject.py`

```python
def inject_lora(model, *, rank, alpha, dropout=0.0, target_modules,
                variant="standard", balora_cfg=None) -> PeftModel: ...
def merge_to_state_dict(peft_model) -> dict[str, Tensor]:
    """merge_and_unload() on a deepcopy → plain base-architecture state dict."""
def save_adapter(peft_model, adapter_dir): ...      # peft save_pretrained
def load_adapter(base_model, adapter_dir) -> PeftModel: ...  # PeftModel.from_pretrained
```

- `variant="balora"` routes through the custom-dispatch registration from
  plan 04; in this phase it raises `NotImplementedError`.
- Normalization buffers (`state_mean/std`, `param_mean/std`) are plain buffers
  on the base model — untouched by PEFT, and included in the merged export.

### `targets.py`

Default `target_modules` per architecture family, as regexes over
`named_modules()` names:

- **P3D**: the windowed-attention QKV/out `nn.Linear`s and the adaLN
  modulation `nn.Linear`s inside `p3d.net`; optionally the `nn.Conv3d`
  blocks (config-selectable tiers: `attention` | `attention+conv` | `all`).
  The concrete name patterns must be derived by enumerating
  `P3D(...).net.named_modules()` at implementation time (the classes live in
  the external `p3d_surrogate` package) — same walk the wrapper already does
  to zero dropout (`p3d.py` ~line 271). Record the chosen regexes in
  `targets.py` with a comment giving the module-tree snippet they matched.
- **Fallback for any architecture**: "every `nn.Linear`/`nn.Conv3d` whose name
  matches a user-supplied regex" — the config always wins over presets.
- Exclusions: `param_to_scalar` (tiny, train fully via
  `modules_to_save` instead), any `Norm` layers.

`modules_to_save` (PEFT's full-training escape hatch) is exposed in config for
small head/stem modules that should adapt fully.

## 2. Config: `conf/neural_surrogate/finetuning.yaml`

`# @package _global_`, same shape as `training.yaml` (so `BaseTraining`,
dataset and dataloader blocks are reused verbatim), plus:

```yaml
defaults:
  - _self_
  - /neural_surrogate/finetune_mode@_global_: lora_nextstep   # plan 03 adds: dft

# Pre-trained source: a model_dir produced by train_neural_surrogate.py.
# architecture + state_vars/param_vars are read from ITS config.yaml, not
# re-declared here (single source of truth; mismatches impossible).
pretrained_model_dir: ???          # e.g. model_weights/p3d_xie_and_castro

# Where the fine-tuned model lands (standard model_dir layout + adapter/).
model_name: ???                    # e.g. p3d_xie_and_castro_ft_barcelona

lora:
  variant: standard                # standard | balora  (balora: plan 04)
  rank: 32                         # tadpole's default; sweep 8/16/32/64
  alpha: 64                        # 2·rank as the usual starting point
  dropout: 0.0
  target_preset: attention         # from targets.py; or set target_modules
  target_modules: null             # explicit regex overrides the preset
  modules_to_save: []              # e.g. [param_to_scalar]

recompute_normalization: false     # keep the pre-trained stats by default;
                                   # true = recompute on the fine-tune split
                                   # (then val metrics aren't comparable to base)

trainer:      # same keys as training.yaml; fine-tune-shaped defaults:
  num_epochs: 50
  compile_model: false             # pending pre-flight §0.4
  ...
optimizer:
  _target_: torch.optim.AdamW
  lr: 1.0e-3                       # LoRA tolerates ~10x the full-FT LR
  weight_decay: 0.0                # no decay on adapter weights
dataset: { ... }                   # points at the NEW (fine-tune) data root
dataloader: { ... }
```

A `finetune_mode` group mirrors the existing `mode` group: `lora_nextstep`
bundles `Trainer` + `MSELoss` + this flow; plan 03 adds `dft`.

## 3. Script: `scripts/neural_surrogate/finetune_neural_surrogate.py`

`def run(cfg)` + thin `@hydra.main(config_name="neural_surrogate/finetuning")`
wrapper. Flow (mirrors `train_neural_surrogate.py` deliberately):

1. Load `<pretrained_model_dir>/config.yaml`; instantiate the architecture
   from **its** `architecture` node with `n_state_channels`/`n_params` derived
   from its `dataset.state_vars`/`param_vars`. Cross-check the fine-tune
   dataset's `state_vars`/`param_vars`/`sdf_features` against the pre-trained
   ones — fail loud on mismatch (same spirit as the SDF cross-check in the
   train script).
2. `load_state_dict(torch.load(weights.pt))`.
3. Optionally recompute + `set_normalization` (config flag, default off).
4. Freeze all params; `inject_lora(...)`; print trainable-parameter fraction
   (`print_trainable_parameters`).
5. Save resolved config to the new `model_dir` (config.yaml must record both
   the fine-tune cfg and, under a `pretrained:` key, the source architecture
   node so ESMDA can rebuild the net without chasing the original dir).
6. Instantiate the existing `Trainer` with
   `optimizer(params=[p for p in model.parameters() if p.requires_grad])`.
   **Trainer receives the `PeftModel`** — `BaseTraining` only needs
   `forward`/`state_dict`/`load_state_dict`, which `PeftModel` provides; its
   in-loop best-weights file then contains the wrapped (base+adapter) state
   dict, which is fine as an intermediate.
7. After `fit()`: `save_adapter(...)` → `model_dir/adapter/`, then
   `merge_to_state_dict(...)` → overwrite `model_dir/weights.pt` with the
   plain merged dict. Final `weights.pt` is indistinguishable from a fully
   trained model → **zero changes to `forward_model.py`**.

One wrinkle to handle in `BaseTraining`, not around it: `weights.pt` is
written *during* training on every val improvement (`base.py` ~line 549) with
the wrapped keys, and step 7 overwrites it at the end. But a run killed
mid-training would leave a wrapped-format `weights.pt` behind. Fix: add an
optional `weights_transform` callable (default `None` → no-op, byte-identical
behavior for existing runs) that `BaseTraining` applies to the state dict when
saving best weights; the fine-tune script passes `merge_to_state_dict`. Cheap
(a deepcopy+merge per improvement) and keeps `weights.pt` always-valid.

## 4. ESMDA / rollout compatibility

- Merged export ⇒ `NeuralSurrogateForwardModel._load_weights` works unchanged.
- `config.yaml` in the fine-tuned dir must present the same top-level keys the
  loader reads (`architecture`, `dataset.state_vars`, `dataset.param_vars`,
  `dataset.root_dir` for domain/step-size): step 5 writes the fine-tune config
  with the `architecture` node copied from the source, and `dataset` pointing
  at the fine-tune data root (that *is* the domain the fine-tuned model
  targets). Verify with an end-to-end smoke: fine-tune on the test fixture
  data → `run_esmda` with the fine-tuned dir.

## 5. Tests (smoke-shaped, `tests/`)

- Unit: inject → forward parity at init (LoRA B=0 ⇒ outputs byte-identical to
  base model); merge round-trip parity; only `lora_*` params have
  `requires_grad`.
- Config/e2e: `compose_test_cfg` on `finetuning.yaml` with the smoke
  overrides; run `finetune.run(cfg)` for 2 epochs on the tiny fixture
  dataset starting from a 1-epoch base model trained in the same test; assert
  the exported dir loads in `NeuralSurrogateForwardModel` and rolls out.
  (Respect the serial-e2e constraint from memory: no parallel e2e sessions.)

## 6. Dependency & docs

- Add `peft` to `libs/neural-surrogates` deps (hard dep is fine — it's light;
  `transformers` is *not* required by peft for custom models. Verify import
  footprint; if it drags in heavy extras, make it an extra like `[p3d]` and
  lazy-import inside `finetuning/`).
- Update `docs/neural_surrogates.md` (new module + artifact layout) and
  `docs/scripts_and_configs.md` (new entry point) in the same PR.
