# Handover: keep building cells out of the state metrics

**Outcome:** implemented with the STL mask (option 2), but with the
highest-crossing rule rather than parity: the backends write no zeros inside
buildings (only PALM does), and on Barcelona's mesh, whose internal faces give
even crossing counts inside buildings, parity opens fluid holes. The mask equals
pylbm's voxelisation on Xie & Castro (1368 cells on the canonical grid, not the
"392 at z = 2 m" below, which came from the flawed parity count). The same PR
also takes the truth at the ensemble's heights. See `docs/evaluation.md`.

State metrics average over every grid cell, building interiors included. Inside
a building both truth and ensemble are (near) zero, so those cells add zero error
and dilute the RMSE. This PR excludes solid cells, and fixes or deletes the
broken STL mask helper. It **changes default results** (the state block of
`metrics.yaml` and the state-RMSE figures), so it is its own PR. Read `AGENTS.md`
and `docs/evaluation.md` first.

Continue on the branch `fix/state-metrics-solid-cells`, which holds the updated
version of this file, and open the PR into `main`. Don't merge it.

## What is there now

- `evaluation.turbulence.streaming_state_rmse` (`libs/evaluation`) takes |U|
  on `n_z_slices=4` evenly spaced z-levels (`_vel_field_4z`, components
  combined by index), interpolates the truth onto the assimilation grid when the
  grids differ, and returns `sqrt(nanmean(diff**2))` per time step. No mask.
  Callers: `scripts/compute_metrics.py` (the `state.vel_magnitude_rmse` block)
  and `scripts/visualize_assimilation.py` (the state-RMSE figure). Since #163
  both pass the truth already on the mean state's times
  (`sel(..., method="nearest", tolerance=0.5 * output_frequency)`), so the mask
  only has to handle space. The lowest selected z-level is where the buildings
  are, so it carries most of the dilution.
- `evaluation.style.stl_solid_mask` has no caller (`docs/evaluation.md`, "Who
  calls it"), and is broken on Xie & Castro: the `if cz.size < 2: continue`
  guard skips every column with a single z-crossing. That STL has no ground
  triangles under the cubes, so every building column has just one crossing
  (the roof) and nothing is masked. Measured: 0 instead of 392 solid cells at
  z = 2 m. `read_binary_stl` is used only by it.
- Each backend already knows its solid cells: pylbm voxelises the STL
  (`pylbm/stl_to_lbm.py`), uDALES has its IBM solid points, PALM its topography,
  `neural_surrogates.geometry.stl_to_fluid_mask` voxelises for the networks.

## Step 1: measure, then decide (report to the user before coding)

For one tiny run per backend (`--config-dir tests/configs +test=forward`
with `model=pyudales|pylbm|pypalm`), report what the output holds inside
buildings: exactly 0, NaN, a fill value or something else, at every time step,
for both the truth and the ensemble states that `compute_metrics.py` reads.
Also check what interpolating the truth onto the assimilation grid does at
building edges.

Then propose one mask source. In order of preference:
1. **From the data:** solid = cells where the truth |U| is exactly 0 (or NaN)
   at every time step, on the grid where the difference is taken. No geometry
   or backend knowledge needed. Valid only if step 1 shows every backend writes
   exact zeros or NaN inside buildings and nowhere else.
2. **From the STL:** fix `stl_solid_mask` (parity rule:
   `count(z_crossings > z) % 2 == 1` per column, no size guard) and pass the
   case's STL path from the run's `config.yaml`.

With option 1, delete `stl_solid_mask` and `read_binary_stl`; it is dead code.
With option 2, fix it and give it its first caller.

## Step 2: implement

- Apply the mask inside `streaming_state_rmse` (or one helper it calls), so
  both callers get it. No new config knob: excluding solid cells is the correct
  metric, not an option.
- Same treatment for any other state metric in `compute_metrics.py` that
  averages over the grid. List them in the PR.
- Tests in `tests/evaluation/`: a synthetic case with a solid block where the
  masked RMSE equals the fluid-only RMSE; with option 2, also the Xie & Castro
  STL masking 392 cells at z = 2 m.
- Update `docs/evaluation.md` (the `stl_solid_mask` note and the
  `streaming_state_rmse` description).

## Step 3: results that change

List in the PR which numbers change: the state block of `metrics.yaml`, the
state-RMSE figures, and any number quoted from them in
`docs/archive/experiments_report` or the decks in `latex/` (not tracked; ask the
user). Rerun one tiny assimilation workflow before and after
(`bash workflows/assimilation_workflow.sh smoother` with the test overlays) and
show the change in state RMSE.

## Done when

- [x] Step 1 findings and the chosen mask approved by the user.
- [x] State metrics exclude solid cells; tests pass; dead helpers fixed or
      deleted.
- [x] `docs/evaluation.md` updated; changed results listed in the PR.
- [x] `tests/evaluation`, `tests/scripts`, `pre-commit` pass; CI green on Linux
      and macOS.
- [x] This file moved to `docs/plans/implemented/`.
