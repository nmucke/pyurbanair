# Domain-Decomposition Neural Surrogate — Implementation Plan (Recommendation A)

Two-level overlapping domain decomposition (companion PDF §2 + §5), implemented so
that **the model is still called as `forward(state, params, geometry) -> state_next`**
on full-grid inputs — the decomposition is embedded *inside* the new model and the
forward model. A **new** patch dataset (separate file) reads the existing
`training_data/` layout for the advanced patch-based objective.

## 0. Corrected design overview (after reading the real code)

1. **`extra` concat is raw, unmasked, after geometry.** In `UNetConvNeXt.forward`,
   after `x = torch.cat([x, geometry], dim=1)`, append `extra` (context `R_iC_t` +
   positional `e_i`) **without** normalization or geometry masking. Stem in-channels
   become `n_state_channels + 1 + extra_in_channels`. `residual=True` already gives
   the delta-state.
2. **`periodic_axes` is GLOBAL — force it OFF in the inner nets.** Patch interiors are
   not periodic; global periodicity is handled by the DD halo fill (`mode='circular'`
   per axis), configured under `decomposition.periodic_axes`, never on the inner net.
3. **`_pad_to_multiple` already makes patches size-robust** — `n+2h` need not be a
   multiple of `2^n_levels` (just `>= 2^n_levels`).
4. **Trainer drop-in path needs NO change.** `DomainDecomposed.forward(state,params,
   geometry)` consumes/returns full grids, so the existing `Trainer` +
   `TransitionDataset` + `MSELoss` train it directly (milestone 1, primary validation).
5. **`set_normalization` forwards to BOTH inner nets** (no-op when `normalize=False`).
6. **Forward-model relaxation needs a cell-spacing invariant**, not merely disabling the
   check: require trained vs requested `(dx,dy,dz)` equal; free `nx/ny/nz` and bounds.
   Detect via `getattr(self.model, "domain_flexible", False)`.
7. **`share_coarse_weights` has a stem in-channel conflict** (fine stem includes
   context+positional channels; coarse takes none) → default to a small dedicated
   `coarse_net`; sharing only via zero-filled `extra` (documented, not default).

## 1. Per-file changes

### Phase 1 — Foundation (must land first)
- **NEW `libs/neural-surrogates/src/neural_surrogates/decomposition.py`** — pure torch.
  `DomainDecomposition(interior_size=n, halo=h, taper=δ, coarsen_factor=r, n_pos=3,
  boundary_mode='replicate', periodic_axes=(F,F,F), geometry_coarsen='any_fluid')`.
  Caches a `_Plan` per global spatial shape. Operators: `restrict` (→ `(B*M,C,n+2h³)`
  via pad + strided `unfold`, stride `n`), `extend_merge` (windowed scatter-add of
  `n+2δ` blocks, crop to grid), `restrict_coarse` (`avg_pool3d(r)`), `prolong`
  (`interpolate(..., 'trilinear')`), `positional` (`(M,n_pos,n+2h³)` signed distance to
  global boundary), `neighbor_indices`.
  - **Uniform tiling via padding** (chosen over per-patch clipping): pad each axis up to
    a multiple of `n`, then `halo` on every side (`mode=boundary_mode`, `'circular'` on
    periodic axes). Every extended block is exactly `(n+2h)³` → one batched forward.
    Crop back at the end of the merge.
  - **PoU windows (CRITICAL):** scatter **`n+2δ`-sized** Hann-tapered windows (stride
    `n`, overlap `2δ`) — NOT strict-`n` interiors (those are disjoint → no blend).
    Normalize by the overlap-sum `S(x)=Σ_i E_i(w_i)` so `Σ_i w_i ≡ 1` everywhere.
    Eq (6) crop becomes crop-to-(interior+δ band) before the windowed scatter.
  - **Geometry coarsen `any_fluid`:** `(avg_pool3d(m,r) > 0).float()` (keeps thin fluid
    corridors visible to the coarse pressure field).
- **EDIT `architectures/unet_convnext.py`** — add `extra_in_channels: int = 0`
  (stem widened) and `forward(..., extra=None)` concatenated raw after geometry.
  Default 0/None ⇒ **byte-identical, same state-dict keys** (existing ckpts load).
- **NEW `architectures/domain_decomposed.py`** — `class DomainDecomposed(nn.Module)`
  with `domain_flexible = True`. Holds `fine_net` (`UNetConvNeXt`,
  `extra_in_channels=C+n_pos`, `residual=True`, `periodic_axes=()`) and `coarse_net`
  (small dedicated `UNetConvNeXt`, `extra_in_channels=0`). `forward` = one step of
  Algorithm 1: coarse pool→coarse_net→prolong=C; tile state/geom/context(+pos); fine_net
  residual per patch; crop to PoU block; window-merge; crop to grid; mask by geometry;
  optional `divergence_projection` (default OFF). `set_normalization` forwards to both.
- **EDIT exports** in `architectures/__init__.py` and `neural_surrogates/__init__.py`
  (`DomainDecomposed`, `DomainDecomposition`).

### Phase 2 — Forward model (parallel with Phase 3)
- **EDIT `forward_model.py::_check_domain`** — when `domain_flexible`, replace `nx/ny/nz`
  + bounds equality with a spacing-equality check; else keep current behavior verbatim.
  `rollout_batched` unchanged.

### Phase 3 — Dataset (parallel with Phase 2)
- **NEW `libs/neural-surrogates/src/neural_surrogates/data_patch.py`** —
  `class PatchTransitionDataset(TransitionDataset)` (subclass for DRY, own file/class).
  Shares the `DomainDecomposition` spec. Index flattens `(traj, t, patch)`. `__getitem__`
  → `state_n_patch (C,n+2h³)`, `delta_target (C,n³)=R°_i S_{t+K}-R°_i S_t`,
  `geometry_patch`, `coarse_input=R_H S_t` (+ `R_H m`), `positional`, `neighbors`,
  `params_n`. Ship `K=1` first (patch pushforward needs the full merge).

### Phase 4 — Training objective Eq (9) — second milestone
- **NEW `dd_loss.py`** — `DomainDecompositionLoss(λ_if=0.1, λ_div=0.01, λ_coarse=1.0)`:
  one-step + interface (neighbor-overlap) + divergence (finite-diff on velocity) + coarse
  terms. Thin `PatchTrainer` unpacking `PatchTransitionDataset`. `Trainer` stays untouched.

### Phase 5 — Configs (parallel after Phase 1)
- **NEW `conf/neural_surrogate/architectures/domain_decomposed/{tiny,small,medium}.yaml`**
  `_target_: neural_surrogates.DomainDecomposed`, nested `decomposition`/`fine_net`/
  `coarse_net`. Inner-net `periodic_axes` empty; global periodicity under
  `decomposition.periodic_axes`. Selected via
  `'neural_surrogate/architectures/domain_decomposed@architecture=small'`.

### Phase 6 — Tests
- `test_decomposition.py` (Σw≡1, restrict∘extend round-trip, positional, coarse geom,
  cache), `test_domain_decomposed.py` (shape, grid-flexibility, determinism, geometry
  zero, gradients, divergence), `test_unet_convnext_extra_channels.py` (state-dict
  identity at `extra_in_channels=0`), `test_patch_transition_dataset.py` (shapes,
  delta-target, neighbors; synthetic pylbm-layout fixture), `test_dd_forward_model_flexible.py`
  (larger grid OK, wrong spacing raises).

### Phase 7 — Docs
- Append **Part E — Domain decomposition** to `docs/neural_surrogates.md`.

## 2. Phase order / parallelism
Phase 1 first (decomposition → unet widening → domain_decomposed → exports). Then
Phase 2 and Phase 3 in parallel (disjoint files). Phase 4 after 3. Phase 5 after 1.
Tests land with their phase. Docs last.

## 3. Top risks
1. **PoU `n+2δ` overlap-blend** (strict-`n` tiles give no blend); normalize by overlap-sum.
2. Three pad/crop bookkeepings (interior→mult-of-`n`, halo `h`, coarse→mult-of-`r`) must
   round-trip to the exact original grid.
3. Global periodicity in the halo fill, OFF in inner nets.
4. `extra_in_channels=0` keeps the UNet state-dict byte-identical.
5. `extra` bypasses normalization + geometry mask.
6. Geometry coarsen `any_fluid` (not `all_fluid`).
7. Grid-flexibility cache: rebuild buffers on shape change, right device/dtype.
8. `torch.compile dynamic=False` for fixed training grid; default off.
9. Divergence projection strictly behind the flag.
10. Patch pushforward infeasible alone → `K=1` first.

## Validation recipe
```
cd /export/scratch1/ntm/pyurbanair && \
PYTHONPATH=/export/scratch1/ntm/pyurbanair-domain-decomposition/libs/neural-surrogates/src:/export/scratch1/ntm/pyurbanair-domain-decomposition/src \
/export/scratch1/ntm/pyurbanair/.pixi/envs/dev/bin/python -m pytest \
  /export/scratch1/ntm/pyurbanair-domain-decomposition/tests/<file> -x -q
```
