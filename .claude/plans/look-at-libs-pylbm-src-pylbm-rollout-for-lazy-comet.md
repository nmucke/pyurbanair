# Integrate RolloutForwardModel into pylbm ForwardModel

## Context

`pylbm` currently exposes two orchestration classes: `ForwardModel` (cold-start single run) and `RolloutForwardModel` (thin wrapper that writes a restart file from an xarray state, flips `nt0`, and cleans up between rollout steps). The wrapper is mostly glue — the actual warmstart work lives in `pylbm.utils.warm_start_utils`. We want to collapse this so `ForwardModel.run_single(state=xarray, ...)` alone handles warmstart: callers pass the previous step's state and ForwardModel prepares the restart files, flips `nt0`, runs, and cleans up. The wrapper and the `create_rollout_forward_model` factory go away. Scope is pylbm only — pyudales stays untouched for now.

The "warmstart template" is the restart file written by LBM at the end of a prior run (`restart/restart_0000_<iter>.uf` plus optional auxiliary files). `write_restart_file_from_xarray` already prefers the latest on-disk template when one exists (for ghost cells and non-equilibrium content) and falls back to pure equilibrium when no template exists. In the integrated flow, the template is guaranteed by the fact that every LBM run writes a restart file at `nt1`, so the second-and-later warm calls always see a template left by the previous call. ForwardModel prepares the warmstart in `run_single` before invoking the solver.

## Design

### Semantics

- `run_single(state=None, params=...)` — cold start. Unchanged.
- `run_single(state=xarray, params=...)` — warmstart:
  1. Build restart file from `state` using the latest on-disk restart as template.
  2. Set `_nt0_override = restart_iteration` (consumed by `_set_scaling_factors` → writes `nt0`/`nt1` in `infile.in`).
  3. Run the solver.
  4. Clean stale output files and prune old restart iterations.
- `spinup_first_step_only` is self-managed: after a successful `run_single` call, if `spinup_time > 0`, ForwardModel calls `disable_spinup()` so subsequent calls run only `simulation_time`.

### Changes to `libs/pylbm/src/pylbm/forward_model.py`

Add constructor arg:
- `spinup_first_step_only: bool = True` (stored as attribute).

Add private helper method `_prepare_warmstart(state: xarray.Dataset) -> None`:
- Calls `identify_latest_restart_iteration(self.dirs)` → `latest`.
- Calls `write_restart_file_from_xarray(state=state, dirs=self.dirs, restart_iteration=latest)` → `restart_iteration`.
- Sets `self._nt0_override = restart_iteration`.

Modify `run_single`:
- Before `_apply_inflow_settings`: if `state is not None`, call `self._prepare_warmstart(state)`.
- After the run and successful result collection, add cleanup:
  - `remove_old_restart_files(self.dirs)` — prune superseded restart iterations (LBM writes a new one at `nt1`, so we keep only the latest).
  - If `self.spinup_first_step_only and self.spinup_time > 0`: `self.disable_spinup()`.
- Do NOT call `clean_output_files` inside `run_single`; the existing `_clean_output()` hook is already invoked by `BaseForwardModel.__call__` after the run returns. (This matches what the current `RolloutForwardModel._post_run_rollout_step` was trying to do, minus the no-longer-needed `save_on_disk` branching.)

### Files/modules to delete or rewrite

| File | Action |
|---|---|
| `libs/pylbm/src/pylbm/rollout_forward_model.py` | **Delete.** All behavior moves into `ForwardModel`. |
| `libs/pylbm/src/pylbm/utils/rollout_utils.py` | **Delete.** `collect_rollout_results` is unused (only commented-out reference in the file being deleted). Confirmed no LBM caller after deletion. |
| `libs/pylbm/src/pylbm/ensemble_forward_model.py` | Remove `RolloutForwardModel` import; drop the `isinstance(forward_model, RolloutForwardModel)` branch in `_create_new_forward_model`; narrow the `forward_model` type to `ForwardModel`. |
| `src/pyurbanair/utils/config_utils.py` | Remove `LBMRolloutForwardModel` import. In `create_rollout_forward_model`, pylbm branch returns the `forward_model` unchanged (no wrapping); pyudales branch keeps wrapping with `UDALESRolloutForwardModel`. Alternatively, to match the user's preference of deleting the factory entirely, **delete** `create_rollout_forward_model` and update pyudales callers to wrap inline. Since scope is pylbm-only, we keep the factory but make pylbm a pass-through. *(Revisit during implementation if this bothers us.)* |

### Scripts to update (pylbm paths only)

Scripts call `config.create_rollout_forward_model(model_name, forward_model)`. For `model_name == "pylbm"`, after the change this returns the plain `ForwardModel`, which now accepts `state=` in `run_single`. Behavior is preserved without script edits. The scripts below already pass `state=` on subsequent calls (e.g., `scripts/run_rollout_forward_model.py:63`), which is exactly the new API — no script edits needed for pylbm:

- `scripts/run_rollout_forward_model.py`
- `scripts/run_ensemble_rollout_forward_model.py`
- `scripts/run_state_and_parameter_esmda.py`
- `scripts/run_rollout_esmda.py`
- `scripts/run_time_varying_parameters_rollout_esmda.py`

### Ensemble impact

`BaseEnsembleForwardModel` currently detects rollout via `hasattr(forward_model, "rollout_step")`. After this change, a pylbm ensemble built from the plain `ForwardModel` will lose that detection path. Before implementation, read `src/pyurbanair/base_ensemble_forward_model.py` to confirm what the `rollout_step` check gates; the integration likely needs an equivalent signal on `ForwardModel` (e.g., rely on the fact that `state` is passed through `__call__` at each step — which is already the case).

## Critical files

- `libs/pylbm/src/pylbm/forward_model.py` — main target (add `_prepare_warmstart`, wire into `run_single`, add `spinup_first_step_only`).
- `libs/pylbm/src/pylbm/rollout_forward_model.py` — delete.
- `libs/pylbm/src/pylbm/utils/rollout_utils.py` — delete.
- `libs/pylbm/src/pylbm/ensemble_forward_model.py` — drop rollout branch.
- `src/pyurbanair/utils/config_utils.py` — pylbm branch of `create_rollout_forward_model` becomes pass-through.
- `src/pyurbanair/base_ensemble_forward_model.py` — verify `rollout_step` gating still works (read-only check first).

## Functions to reuse (no new code)

- `pylbm.utils.warm_start_utils.identify_latest_restart_iteration`
- `pylbm.utils.warm_start_utils.write_restart_file_from_xarray`
- `pylbm.utils.warm_start_utils.remove_old_restart_files`
- `pylbm.forward_model.ForwardModel.disable_spinup` (already exists)
- `ForwardModel._nt0_override` (already exists; consumed in `_set_scaling_factors`)

No new modules are introduced. The existing split between `forward_model.py` (orchestration) and `utils/warm_start_utils.py` (restart-file I/O) is kept — it's already the right seam.

## Verification

Run via `pixi`:

1. `pixi run pytest tests/test_run_rollout_forward_model.py -k pylbm -x` — two-step rollout for pylbm (with and without `results_dir`). Should pass without code edits to the script.
2. `pixi run pytest tests/test_run_ensemble_rollout_forward_model.py -k pylbm -x` — ensemble rollout still works after dropping the `RolloutForwardModel` branch in `EnsembleForwardModel._create_new_forward_model`.
3. `pixi run python scripts/run_rollout_forward_model.py --model pylbm --skip-viz` — end-to-end smoke.
4. Sanity-check cold start is untouched: `pixi run python scripts/run_forward_model.py --model pylbm --skip-viz`.
5. Confirm deletion didn't leave orphan imports: `rg "rollout_forward_model|RolloutForwardModel" libs/pylbm src/pyurbanair scripts tests` should return only pyudales hits.
