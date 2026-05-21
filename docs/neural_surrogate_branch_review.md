# Neural Surrogate Branch Review

Review scope: implementation currently present on `feat/neural-surrogates`, including tracked working-tree diffs and new untracked surrogate package/config/script/test files.

Verification run:

```bash
.pixi/envs/dev/bin/pytest -q libs/neural_surrogates/tests tests/test_neural_surrogate_forward.py tests/test_hydra_config.py
```

Result: 47 passed, 1 warning.

## Findings

### P1: uDALES-trained surrogate params are not wired at runtime

`resolve_parameter_schema()` can read `schema.json`, but callers still invoke `create_true_params(model_name, ...)` without a checkpoint path, so `neural_surrogate` falls back to only `inflow_angle` and `velocity_magnitude` in `src/pyurbanair/config/hydra_helpers.py`. `scripts/run_forward_model.py` does this at the forward-model entry point, and the ESMDA scripts follow the same pattern. `create_parameter_ensemble()` also always emits only those two variables.

Any uDALES checkpoint whose schema requires `pressure_gradient_magnitude` will then fail in `libs/neural_surrogates/src/neural_surrogates/utils/params_io.py` when conditioning is built.

Suggested fix: thread `cfg.model.checkpoint_path` / `cfg.assim_model.checkpoint_path` into parameter creation for neural surrogates, and extend `create_parameter_ensemble()` to accept resolved parameter names or checkpoint path.

### P1: corpus generation cannot run for `model=pyudales` as written

`scripts/generate_neural_surrogate_data.py` assumes `cfg.model.forward_model.stl_path`, but `conf/model/pyudales.yaml` only provides `case_dir`. This blocks uDALES corpus generation before simulation starts.

Suggested fix: resolve geometry per backend. For uDALES, derive the STL from `case_dir` or add an explicit `stl_path` field to the model config and use that in the generator.

### P2: uDALES collocation output dims do not match surrogate tensor IO

The generator calls `pyudales.utils.grid_utils.interpolate_grid()` and then immediately calls `state_io.state_to_tensor()`. The interpolation helper returns center-grid dims/coords named `zt`, `yt`, `xt`, while `state_to_tensor()` requires `z`, `y`, `x`.

Suggested fix: add a surrogate-specific collocation/renaming step, or teach `state_io.state_to_tensor()` a solver/grid-dim mapping. The checkpoint output should still remain collocated `z/y/x`.

### P2: UNet ignores `hist_mask`

`extract_history()` left-pads short histories with zero frames and returns `hist_mask`, but `UNet3D.init_carry()` returns only `hist_fields`. For `history_len > available frames`, cold starts and early windows are conditioned on artificial zero-flow frames despite the plan’s mask contract.

Suggested fix: have the UNet use `hist_mask`, for example by replacing padded slots with the first real frame, adding the mask as extra channels, or carrying both fields and mask through `init_carry()`.

### P2: solid-cell masking zeroes pressure too

`ForwardModel._reapply_mask()` multiplies every predicted channel by the fluid mask. That is appropriate for velocity no-penetration, but corrupts `pres` for checkpoints trained with `include_pressure=true`.

Suggested fix: apply the mask only to velocity channels (`u`, `v`, `w`) and leave pressure channels unchanged unless a checkpoint/schema explicitly says otherwise.

## Notes

The branch tip currently matches `main`; this implementation is in working-tree changes and untracked files rather than commits.

`libs/pylbm/LBM/` and `libs/pyudales/u-dales/` are untracked simulator trees. They appear to be generated/vendor checkouts and should be intentionally ignored or intentionally vendored before staging.
