# Implementation review — plans 01–03 (LoRA fine-tuning, AE pre-training, AE→time-stepper)

**Status: review findings (2026-07-09). This is a work order, not a plan.**

Four independent review passes (plan 01, plan 02, plan 03, shared plumbing) were
run over the implemented code behind
[01_lora_finetuning.md](01_lora_finetuning.md),
[02_autoencoder_pretraining.md](02_autoencoder_pretraining.md) and
[03_ae_to_timestepper.md](03_ae_to_timestepper.md), checking correctness,
computational optimality, and unnecessary complexity. The two highest-severity
findings (H1, H2) plus the M1 guard gap were re-verified by direct inspection
before this file was written.

**Overall verdict:** the implementations are faithful to the plans and the
load-bearing invariants hold (identity-at-init is exact, LoRA presets exclude
the peft-0.19 Conv3d crashers, merged exports are strict-loadable by ESMDA,
`weights_transform` is a true no-op by default, `forward_model.py` has zero
diff vs. baseline). Two high-severity defects and a handful of medium items
need fixing; the rest is cheap cleanup or explicitly-accepted trade-offs.

## Instructions for the fixing agent

- **Work per section**: §1 items are required and independent — one branch/PR
  per item (or batch H1 with §2's config items). §2 items are cheap, batchable.
  §3 is optional perf work — do NOT start it unless explicitly asked. §4 lists
  things that look like defects but are deliberate — do not "fix" them.
- **Line numbers are anchors, not gospel.** Verify each by reading the file
  before editing; the code may have drifted since 2026-07-09.
- **Repo rules apply**: branch first, `pixi run -e dev pre-commit` before
  committing, new knobs no-op by default, update `docs/neural_surrogates.md`
  (Parts E/F/G) in the same PR when behavior or contracts change.
- **Vendored code** (`libs/neural-surrogates/src/neural_surrogates/architectures/_tadpole/`)
  is byte-for-byte upstream (Tadpole @ 4232e698) except documented shims
  (optional GIFt import, optional-dep guards) and is excluded from formatters.
  H2 below is the one sanctioned modification — mark it with a clear
  `# LOCAL MODIFICATION` comment like the existing shims. Touch nothing else
  in that subtree.
- **Test baseline**: there are ~28 pre-existing unrelated test failures
  (param_time_series imports, pyudales precomputed-geom, dd_wiring sdf
  mismatch). Fixing H1 is expected to clear the sdf-mismatch subset. For
  everything else, verify a failure exists before your change before blaming
  your change (`git stash` trick).
- Per-item **verify** lines name the targeted test file(s); run those, not the
  whole suite, while iterating. Finish with one full
  `pixi run -e dev py.test` before the PR.

---

## 1. Required fixes

### H1 — `training.yaml` sdf_features regression breaks stock training runs

- **Severity: HIGH (regression).**
- **Where:** `conf/neural_surrogate/training.yaml:154`
  (`dataset.sdf_features: sdf`) vs.
  `conf/neural_surrogate/architectures/p3d/medium.yaml:10`
  (`sdf_features: both`).
- **Problem:** commit `2255742` (the autoencoder PR, #86) changed the dataset
  default `both → sdf` inside the *existing* training entry point. The default
  mode (`mode/standard.yaml` → `architectures/p3d/medium.yaml`) still declares
  `both`, so a stock `python scripts/neural_surrogate/train_neural_surrogate.py`
  now raises the SDF-mismatch `ValueError`
  (`scripts/neural_surrogate/train_neural_surrogate.py:49–64`) at startup.
  Direct violation of the no-op-default rule; almost certainly the "dd_wiring
  sdf mismatch" entry in the known failure baseline.
- **Fix:** restore `sdf_features: both` in `training.yaml` (the
  preset-consistent, pre-#86 value). Do not instead change the p3d preset —
  that would silently alter the channel set of the default architecture.
- **Verify:** compose the default training config in a test / smoke run and
  confirm no startup ValueError; `tests/test_dd_training_wiring.py` and the
  neural-surrogate training smoke tests; confirm the sdf-mismatch entries
  disappear from the failure baseline.

### H2 — vendored naive attention runs with head_dim = 1 (num_heads knob silently dead)

- **Severity: HIGH (correctness — cripples the DFT sub-network).**
- **Where:**
  `libs/neural-surrogates/src/neural_surrogates/architectures/_tadpole/architecture/downstream/llm.py:259`.
- **Problem:** `LLMLayer` calls `AttentionBlock(dim, dim, num_heads,
  causal=causal)`, but the signature is `AttentionBlock(dim, num_heads=8,
  qkv_bias=False, ...)` — so the second positional `dim` binds to `num_heads`
  and the intended `num_heads` (8) binds to `qkv_bias`. Verified at runtime:
  `LLMLayer(144, 576, 8, attention_method="naive").attn.num_heads == 144`,
  i.e. 144 heads of dimension 1; attention scores are products of scalars and
  the `num_heads` values in `_SUBNET_SIZES` / `subnetwork_cfg` never take
  effect. Upstream's default backend is `"hyper"` (triton), which this repo
  cannot build — `"naive"` is the **only** path `TadpoleTimeStepper` uses, so
  every DFT fine-tune trains a degenerate latent transformer. No crash, and
  the identity-at-init test is insensitive to attention internals, so nothing
  catches it.
- **Fix:** change the call to bind correctly, e.g.
  `AttentionBlock(dim, num_heads, causal=causal)` (prefer keyword args:
  `AttentionBlock(dim, num_heads=num_heads, causal=causal)`), with a
  `# LOCAL MODIFICATION` comment referencing this finding. Check whether the
  same mis-binding pattern occurs elsewhere in `downstream/` (grep for
  `AttentionBlock(` call sites) and fix those too.
- **Caveat:** this changes what a checkpoint means — any DFT checkpoint
  trained before the fix has weights shaped by head_dim-1 attention. Land this
  before real DFT training runs; flag existing experimental checkpoints as
  suspect.
- **Verify:** add a unit test (outside `_tadpole/`, e.g. in
  `tests/test_ae_to_timestepper.py`) asserting the built subnetwork's
  attention blocks report the configured `num_heads` (and `qkv_bias` is a
  bool, not an int). Re-run the identity-at-init parity test — it must still
  pass exactly.

### M1 — resumed fine-tune can silently clobber a better on-disk `weights.pt`

- **Severity: MEDIUM (silent data loss, needs crash + resume + no re-improvement).**
- **Where:** `scripts/neural_surrogate/finetune_neural_surrogate.py:349–365`
  and `libs/neural-surrogates/src/neural_surrogates/training/base.py` (best-state
  persistence, ~lines 626–697).
- **Problem:** `weights.pt` is written (merged) on *every* val improvement,
  but `best_model_state`/`best_val` are persisted only every
  `checkpoint_every` epochs. Scenario: improvement at epoch 12 (merged
  weights.pt on disk), crash at 13, last checkpoint at 9. Resume restores the
  epoch-9 best; if the resumed run never re-beats epoch-9's `best_val`,
  `restored_best_weights` is still `True` and the final overwrite at
  `finetune_neural_surrogate.py:356–357` replaces the better epoch-12
  `weights.pt` with epoch-9 weights, silently. The existing guard only covers
  the `best_state is None` case, not staleness. (Plain training self-heals
  because it restores best *from disk*.)
- **Fix (pick one, prefer the first):**
  1. Persist `best_val` alongside the transformed save (e.g. a small
     `best_val.json` or inside `checkpoint.pt` written at improvement time,
     tiny — no full-model write), and make the end-of-run overwrite
     conditional on `trainer.best_val <=` the persisted value.
  2. Persist an adapter-only best snapshot on every improvement (base is
     frozen, so adapter + `modules_to_save` fully determines the model) and
     rebuild from it on resume.
- **Constraint:** keep `weights_transform=None` behavior byte-identical
  (no-op rule); keep `weights.pt` always ESMDA-loadable mid-run.
- **Verify:** extend `tests/test_base_training_weights_transform.py` with the
  staleness scenario (improve → "crash" after a non-checkpoint epoch → resume
  → no further improvement → assert `weights.pt` still holds the newer
  weights).

### M2 — missing AE normalization stats only warn; fine-tune proceeds with identity stats

- **Severity: MEDIUM (wasted full training runs, repo convention is fail-loud).**
- **Where:**
  `libs/neural-surrogates/src/neural_surrogates/architectures/tadpole_stepper.py:374–395`
  (`_load_ae_state_stats`), and `_check_ae_stepper_match` in
  `scripts/neural_surrogate/finetune_neural_surrogate.py:47–91`.
- **Problem:** if the AE dir's `weights.pt` is absent or lacks
  `state_mean/state_std`, the stepper `warnings.warn`s and continues with
  zeros/ones — feeding the frozen encoder out-of-distribution input; the
  warning scrolls past and a full wasted DFT run follows. Also,
  `_check_ae_stepper_match` cross-checks size/geometry/SDF but not the AE's
  `normalize` flag, so an AE trained with `normalize: false` silently pairs
  with a `normalize: true` stepper through exactly this path.
- **Fix:** raise instead of warn in `_load_ae_state_stats` (allow an explicit
  opt-out only where `recompute_normalization: true` will overwrite the stats
  anyway — thread that intent through rather than guessing); add the
  `normalize` flag to `_check_ae_stepper_match`'s fail-loud comparisons.
- **Verify:** unit tests asserting (a) missing stats raise with an actionable
  message, (b) normalize-flag mismatch raises; existing
  `tests/test_ae_to_timestepper.py` cross-check tests stay green.

### M3 — `pretrain_autoencoder.yaml` ships an *active* batch_sampler that its own comment says is off

- **Severity: MEDIUM (undocumented default behavior + dead knobs).**
- **Where:** `conf/neural_surrogate/pretrain_autoencoder.yaml:111–129`.
- **Problem:** the comment block says "the default (null) plain shuffled
  DataLoader… it stays null", `# batch_sampler: null` is commented out, and an
  **active** `TrajectoryBatchSampler` block follows. Every default run
  therefore uses trajectory-bucketed batching: `dataloader.batch_size /
  shuffle / drop_last` become dead knobs (neutralized in
  `training/data_utils.py:50–64`), batches never mix trajectories (lower
  snapshot diversity than the documented default), and on grids ≥ ~2.1M cells
  the `cell_budget: 4194304` silently drops batch size to 1. Docs Part F does
  not mention this default.
- **Fix:** make the shipped default match the documented one — comment the
  sampler block out (leaving it as the ready-to-enable example) so
  `batch_sampler: null` is the effective default. If the sampler default was
  actually intended, instead fix the comment *and* document the behavior
  (including the cell-budget batch-size-1 effect and the dead dataloader
  knobs) in the yaml and in `docs/neural_surrogates.md` Part F.
- **Verify:** `tests/test_autoencoder_pretraining.py` (composes this config);
  confirm which path the e2e smoke actually exercised before/after.

### M4 — `AutoencoderTrainer` re-uploads geometry/SDF host→GPU every step

- **Severity: MEDIUM (easy win on DRAM-bandwidth-bound hardware).**
- **Where:**
  `libs/neural-surrogates/src/neural_surrogates/training/autoencoder.py:66–76`
  vs. the device-side cache in `BaseTraining._prepare_batch`
  (`training/base.py:290–314`).
- **Problem:** `_prepare_ae_batch` calls `.to(self.device)` on `geometry` and
  `geom_features` per batch. `BaseTraining._prepare_batch` maintains a
  device-side cache keyed on the host tensor precisely to avoid this (a
  documented optimization). Cost ≈ `(1 + n_sdf) · grid · 4` bytes per step —
  ~17 MB/step at 128³ with `sdf`, ~130 MB/step at 256³ with `both`.
- **Fix:** reuse the base class's cache pattern (identity check +
  `torch.equal` revalidation, same as `_prepare_batch`) in
  `_prepare_ae_batch`. Note the random-crop path ships per-sample geometry —
  the cache must not serve stale crops; key/validate exactly as the base does
  and accept that the cache only pays off on the shared-geometry path.
- **Verify:** `tests/test_autoencoder_pretraining.py`; add a small unit test
  that a changed geometry tensor busts the cache (mirror the base trainer's
  existing test if one exists).

### M5 — random-crop dataset path reads the full snapshot then crops (~20× I/O amplification)

- **Severity: MEDIUM (only when `random_crop_size` is set; default null unaffected).**
- **Where:**
  `libs/neural-surrogates/src/neural_surrogates/datasets/snapshot.py:274–283`.
- **Problem:** `__getitem__` materializes all `state_vars` on the full grid
  (`np.asarray(snap[v].values)`) before `_random_crop` slices. A 96³ crop from
  a 256³ / 3-var f4 dataset reads ~200 MB to produce a ~10 MB sample, per
  item, per worker.
- **Fix:** choose the crop origin first (from the trajectory's grid shape),
  then read lazily with
  `ds[v].isel(time=t, x=slice(...), y=slice(...), z=slice(...))` so only the
  crop is read. Apply the same slicing to geometry/SDF. Keep the
  full-field (no-crop) path byte-identical.
- **Verify:** existing SnapshotDataset tests (item shapes, both collate
  paths, stride) stay green; add a test that a cropped item equals the
  corresponding slice of the full-field item for a fixed seed/origin.

### M6 — SDF features recomputed per member per internal step in ESMDA rollouts

- **Severity: MEDIUM, conditional (only SDF-enabled steppers in deployment; default `none` unaffected).**
- **Where:**
  `libs/neural-surrogates/src/neural_surrogates/architectures/_tadpole_field_io.py:79–81`
  (fallback `_sdf_features` compute) +
  `libs/neural-surrogates/src/neural_surrogates/forward_model.py` `_rollout_chunk`
  (~line 801), which calls `self.model(current, param_k, geom)` without
  `geom_features`.
- **Problem:** with SDF enabled, every internal rollout step recomputes the
  signed-distance transform (scipy `distance_transform_edt` on CPU, GPU→CPU→GPU
  round trip) for a geometry that never changes within a rollout —
  O(0.05–0.5 s × members × internal steps), potentially dominating wall-clock.
  Training is unaffected (the dataset ships precomputed features).
- **Fix:** compute `geom_features` once per rollout in `_rollout_chunk`
  (alongside the existing `geom` handling) when the model exposes
  `n_geom_feature_channels > 0`, and pass it down every step. Must be a no-op
  for models without SDF features (P3D `none`, UPT, existing dirs).
- **Verify:** e2e stepper→ESMDA smoke in `tests/test_ae_to_timestepper.py`
  still passes; outputs byte-identical for a non-SDF model; for an SDF model,
  assert the per-step output is unchanged vs. the recompute path (same values,
  fewer computations).

### M7 — LoRA merge parity is never numerically asserted at nonzero B

- **Severity: MEDIUM-LOW (test gap on the repo-specific peft×Conv3d path).**
- **Where:** `tests/test_lora_finetuning.py:97–113`
  (`test_merge_round_trips_to_plain_state_dict`).
- **Problem:** merge parity is proven only at init (LoRA B = 0). The actual
  delta math — peft's alpha/r scaling and the Conv3d weight fold — is never
  checked; if peft 0.19's Conv3d merge were subtly wrong for our shapes, all
  tests pass while ESMDA gets wrong weights. This repo has already caught two
  peft-0.19 conv bugs, so the residual risk is not theoretical.
- **Fix:** in the round-trip test, perturb a `lora_B` tensor (nonzero), then
  assert wrapped-model forward == fresh-base-model-loaded-with-merged-dict
  forward on a nontrivial input (tolerance ~1e-5 for the conv fold).
- **Verify:** the new assertion itself.

---

## 2. Cheap cleanups (batchable into one PR)

Each is LOW severity; fix opportunistically, don't over-engineer.

- **C1 — validation loss stochastic under `latent_type: sample`.**
  `_tadpole/model/autoencoder.py:94–100` samples unconditionally, so
  `AutoencoderTrainer._validate` scores a sampled latent — best-weights
  selection and early stopping (patience 20) ride on sampling noise. Fix
  *without touching vendored code*: in the trainer's validation path, switch
  the wrapper to mode-latents (e.g. temporarily set the AE's `latent_type` to
  `"mode"` during `_validate`, restoring after). Document in Part F.
- **C2 — parity-test tolerance weakens the headline invariant.**
  `tests/test_ae_to_timestepper.py:130,146,266` assert `< 1e-5` where docs
  promise exact equality (and it *is* bitwise at init). Tighten to
  `torch.equal`.
- **C3 — `_assemble_working_input` silently accepts mis-shaped geometry.**
  `architectures/_tadpole_field_io.py:140–150`: a `(1, *grid)` mask with
  batched state isn't expanded (fails later or broadcasts); a
  `(B, 1, *grid)` mask falls through both branches. Handle `(1, *grid)`
  explicitly (expand to B) and `else: raise` on anything unrecognized.
- **C4 — duplicated stat-coercion helper.**
  `architectures/tadpole_ae.py:263–272` inline `_to` duplicates
  `_TadpoleFieldIO._to_buffer` (`_tadpole_field_io.py:163–172`), which the
  stepper already uses. Use the mixin method.
- **C5 — dead `cfg.get("dataset") is None` guard fires too late.**
  `scripts/neural_surrogate/finetune_neural_surrogate.py:163` — the dataset
  node is dereferenced at 157–159 first. Move the guard above the first use.
- **C6 — `is_dft` dispatch keys on mere presence of `architecture`.**
  `finetune_neural_surrogate.py:137` — a user adding `+architecture=...` to a
  `lora_nextstep` run silently flips into the dft path. Dispatch on an
  explicit marker key set by the `finetune_mode` group files (e.g.
  `finetune_mode: lora_nextstep|dft`) instead.
- **C7 — `restored_best_weights` only exists after `fit()`.**
  `training/base.py:548` — initialize it in `__init__` so pre-`fit()` access
  is `False`, not `AttributeError`.
- **C8 — resume-recovered `best_model_state` stays GPU-resident.**
  `training/base.py:~568` — `torch.load(..., map_location=self.device)` pins
  the recovered snapshot on the GPU for the whole resumed run (fresh snapshots
  are CPU). `.cpu()` the recovered one.
- **C9 — stale scaffolding in stepper tests.**
  `tests/test_ae_to_timestepper.py:57–61,385–387,533–535` — the
  `_blocked_on_2a` skipif is permanently satisfied and `TODO(phase-2a)`
  comments reference a shipped phase. Delete.
- **C10 — duplicated "Part E" heading in docs.**
  `docs/neural_surrogates.md:985` and `:1250` are both titled Part E
  (domain decomposition and LoRA); renumber the LoRA section onward (E→F→G→H
  or similar) and fix internal references, including in the plan docs' status
  headers.
- **C11 — script should warn on full-field padding waste.**
  `scripts/neural_surrogate/pretrain_autoencoder.py:68–81` warns only for
  `random_crop_size < encoder_crop_size`; also warn when a full-field dim is
  not a multiple of `encoder_crop_size` (e.g. 64-crop on z=48 spends 25% of
  every forward on padding tiles, which also pollute the KL metric —
  see D-list note on KL below).
- **C12 — VRAM caveat missing from `merge_to_state_dict` docstring.**
  `finetuning/inject.py:86–92` documents the time cost of the per-improvement
  deepcopy but not the transient ~1× model VRAM spike (it deepcopies on the
  training device right after validation). One sentence.

---

## 3. Optional / deferred (do only if asked, or if profiling shows the cost)

- **D1 — static geometry channels re-encoded and decoded-then-discarded every
  rollout step.** `tadpole_stepper.py:478–487` + vendored `dft.py:151–213`:
  geometry latents + skip residuals are constant per rollout and cacheable;
  their decode is pure waste. ~20–25% of ESMDA rollout wall-clock recoverable
  at defaults (~57% of enc/dec FLOPs with `sdf_features=all`). Feasible
  without touching the vendored file (slice the folded batch before decode,
  cache geometry latents), but invasive — do only if DFT rollouts become a
  measured bottleneck.
- **D2 — adapter-sized best-state snapshots instead of full-model copies.**
  `training/base.py:632–635` (RAM best = full CPU clone) and `:526`
  (checkpoint.pt stores model + best = 2× on disk): in a LoRA run ~95% of
  both copies is frozen base. Generic `weights_transform` can't know what's
  frozen, so this needs a design decision. Note: M1-fix option 2 subsumes
  this.
- **D3 — `encode()` ignores `max_internal_batchsize`.**
  `tadpole_ae.py:282–302` runs all crops in one encoder call, bypassing the
  chunking that bounds memory in the vendored `forward`. Matters only for
  analysis/plan-03 use on large fields; add chunking if it OOMs in practice.
- **D4 — KL element includes padded all-zero crops.** `tadpole_ae.py:331–334`:
  the logged `kl` metric isn't comparable across grids with different padding
  fractions (loss effect negligible at β=1e-6). Fix only if the metric is
  actually being compared across domains; C11's warning covers awareness.
- **D5 — dead API surface to reconsider after plan 04.**
  `predict_residual` (accepted only as `True`, no caller passes it —
  `tadpole_stepper.py:228,273–279`); the `"token"` conditioning variant (a
  strict subset of `"film"`, not the plan's parameter-token idea —
  `tadpole_stepper.py:145,166–167`); stepper `encode()/decode()` passthroughs
  (no callers); `subnetwork_cfg`'s `"hidden"` alias (nothing sets it);
  `load_adapter` (exported, zero call sites — plan-04 forward-compat);
  `pretrained.architecture` duplication in exported configs
  (`finetune_neural_surrogate.py:319–323` — byte-identical to the top-level
  node in lora_nextstep mode). Trim in one pass **after plan 04 lands** and
  it's clear what plan 04 actually uses.

---

## 4. Accepted trade-offs — do NOT change

- **`merge_to_state_dict` deepcopy per val improvement** (`inject.py:92`):
  deliberate; the in-place `merge_adapter()/unmerge_adapter()` alternative was
  correctly rejected (base + Δ − Δ leaves fp roundoff in training weights).
  Only C12 (docstring) and optionally D2 apply.
- **`balora_cfg` accepted-and-ignored / `variant` with one legal value**
  (`inject.py:44–63`): plan-04 forward-compat, mandated by the master plan.
- **Normalization-cache signature now includes the dataset class**
  (`training/data_utils.py:123`): one-time re-stream of pre-existing
  `normalization_stats/*.npz` caches; values identical; deliberate isolation.
- **`checkpoint.pt` gaining a `best_model_state` key**: resume of pre-change
  checkpoints is handled via `.get(...)`; no fixed-key readers exist.
- **The near-tautological state_vars cross-check**
  (`finetune_neural_surrogate.py:177–182`): cheap insurance, keep.
- **Vendored subtree fidelity**: only sanctioned local modifications are the
  GIFt import shim, the optional-dep guards, and (after this review) H2.

## 5. Verified clean (don't re-litigate)

- Identity-at-init is exact across every zero-init path (FiLM out layer, γ
  skip scales, `latent_residual_scale=1.0`, subnetwork `out_proj`, LoRA B) —
  empirically confirmed; the honest invariant (`stepper == _ae_reference_recon`)
  is tested with `latent_type="mode"`.
- P3D LoRA preset regexes match exactly the documented modules (44 Linears for
  `attention`; +2 3×3×3 groups=1 Convs for `attention+conv`; no `cpb_mlp`
  leakage); the 1×1×1 / grouped-Conv3d exclusions catch the peft-0.19 crashers
  and are tested. `tadpole_encdec` cannot target the new fully-trained modules.
- Merged `weights.pt` exports are plain, strict-loadable, include normalization
  buffers and the non-LoRA trained modules; `forward_model.py` needed zero
  changes (deploy is pure config stamping: `skip_pretrained_load: true`,
  `pretrained_ae_dir: null`); no filesystem access at ESMDA instantiation.
- `weights_transform=None` leaves `BaseTraining` byte-identical; existing
  Hydra entry points (`training.yaml` defaults list, `run_esmda.yaml`,
  `run_forward_model.yaml`, `conf/model/*`) untouched except H1.
- `AutoencoderTrainer` is a clean `_forward`/`_prepare_ae_batch` subclass (no
  BaseTraining fork); pushforward machinery genuinely inert.
- Normalize-before-fold, mask→z-score→re-mask, padding round-trip,
  multiple-of-16 validation, masked-MSE denominator, KL formula/finiteness,
  chunked-decode parity, SnapshotDataset handle caching + worker
  `__getstate__` + collate paths, packaging (peft in pypi tables, lazy
  imports, `_tadpole/` formatter exclusion) — all checked and correct.
- Actual `pretrained` default in `pretrain_autoencoder.yaml` is `none` — no
  default run touches HuggingFace.
