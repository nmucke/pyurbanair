# Surrogate `training_data` spin-up (deferred)

Not scheduled yet. `configs/model/neural_surrogate.yaml` defaults to
`spinup_source: training_data` and documents a `training_data_spinup:` block
"consumed by run_esmda". No current script reads that block, so an assimilation
with `model=neural_surrogate` and the defaults fails at the cold start
(`neural_surrogates/ensemble_forward_model.py` raises).

The helpers exist in `neural_surrogates/training_spinup.py`
(`resolve_training_root`, `list_split_samples`, `write_initial_state_files`,
`anchor_prior_params`); `archive/scripts/esmda/run_esmda.py` shows how they were
called. The MCP server already requires an explicit `initial_state` for this
mode (`mcp_server/jobs/preparation.py`).

Two options, to decide with the user:

- **Wire it up:** one helper in `scripts/utils/` that, when
  `spinup_source == "training_data"`, writes the initial states and anchors the
  prior; call it from `run_smoother`, `run_filtering` and `run_hybrid`. Test
  with the tiny surrogate overlay in `tests/configs/`.
- **Drop it:** default to `spinup_source: forward_model`, delete the
  `training_data_spinup` block and the `training_data` branch if nothing else
  uses it.

Either way, make `check_config` reject a `training_data` spin-up that nothing
loads, and fix the stale `run_esmda` comments in the files above.
