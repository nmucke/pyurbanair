#!/usr/bin/env bash
#
# Run scripts/run_esmda.py once per mode on an otherwise identical joint
# state + time-varying-parameter ESMDA config, so the state-update strategies
# can be compared. Each run writes its outputs (incl. run_summary.yaml) to its
# own .temp/loc_<mode> dir.
#
# Localization is applied to the STATE only (parameters always get the global
# update); the three runs differ solely in the `esmda/localization=` group token.
#
# Usage:
#   scripts/compare_localization.sh            # tiny pylbm smoke comparison
#   SIZE=large scripts/compare_localization.sh # bigger domain
#   ENSEMBLE_SIZE=40 NUM_STEPS=4 scripts/compare_localization.sh
#   SIZE=large RADIUS_SMALL=8 RADIUS_LARGE=40 scripts/compare_localization.sh
#
# The truth (state + params) is LOADED FROM DISK (run.truth_dir, default the
# repo-root `ground_truth/` dir: a uDALES state.nc/params.nc artifact, 3600 s
# at 1 Hz on a 100x80x32 grid) instead of being simulated inline. The truth
# model is therefore never run — TRUTH_MODEL only selects the grid convention
# of the truth observation operator and must match the artifact (pyudales).
# Restore the inline-truth behavior with TRUTH_DIR=null.
#
# Currently active modes (the localization runs are kept but commented out) —
# two reduced-SVD state updates (esmda/state_reduction=svd,
# docs/reduced_state_da.md; incompatible with localization, so both run with
# esmda/localization=none):
#   svd_ic         basis trained on the time=0 IC ensemble, update applied to
#                  the IC only (the plain reduced update)
#   svd_snap_final basis trained on ALL window snapshots, plus the optional
#                  post-loop Kalman update of the state at every time step
#                  (esmda.final_time_smoothing=true)
# — plus a parameter-only baseline (the `dynamic` smoother: estimate the
# time-varying parameters only, no state estimation).
#
# Knobs (env vars, with defaults):
#   SIZE           domain size: 'test' (tiny smoke grid) | 'large'  (test)
#                    large = x[-20,40] y[0,40] z[0,32] (nx30,ny20); nz=4 in all sizes
#   TRUTH_DIR      truth artifact dir with state.nc/params.nc   (ground_truth)
#                    'null' -> simulate the truth inline instead
#   TRUTH_START    start the assimilation horizon this many seconds into the
#                    truth (skips its spin-up; rebases that time to t=0)  (0)
#   TRUTH_MODEL    solver grid convention of the truth artifact (must match it;
#                    only simulates the truth when TRUTH_DIR=null)  (pyudales)
#   ASSIM_MODEL    solver used in the assimilation ensemble                (pylbm)
#                    TRUTH_MODEL != ASSIM_MODEL injects model error (see note below)
#   NNUDGE_M       pyudales un-nudged height (m), used if a model is pyudales (4.0)
#   ENSEMBLE_SIZE  ensemble members            (16 — keep >~10 so correlation
#                                                isn't degenerate)
#   NPROC          parallel forward processes  (8)
#   NUM_STEPS      ESMDA iterations            (2)
#   NUM_WINDOWS    assimilation windows        (3)
#   RADIUS_SMALL   distance localization radius (small), domain units   (5.0)
#   RADIUS_LARGE   distance localization radius (large), domain units   (20.0)
#   RHO_T          correlation truncation threshold             (0.3)
#   MAX_INFL_C     correlation max error-variance inflation      (8.0)
#   MAX_INFL_D     distance max error-variance inflation         (4.0)
#   SVD_ENERGY     reduced-SVD retained-energy fraction          (0.99)
set -euo pipefail

# Run from the repository root regardless of where the script is invoked.
cd "$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)"

ENSEMBLE_SIZE=${ENSEMBLE_SIZE:-32}
NPROC=${NPROC:-8}
NUM_STEPS=${NUM_STEPS:-3}
NUM_WINDOWS=${NUM_WINDOWS:-4}
RADIUS_SMALL=${RADIUS_SMALL:-5.0}
RADIUS_LARGE=${RADIUS_LARGE:-20.0}
RHO_T=${RHO_T:-0.3}
MAX_INFL_C=${MAX_INFL_C:-8.0}
MAX_INFL_D=${MAX_INFL_D:-4.0}
SVD_ENERGY=${SVD_ENERGY:-0.99}
SIZE=${SIZE:-test}
TRUTH_DIR=${TRUTH_DIR:-ground_truth}
TRUTH_START=${TRUTH_START:-0}
TRUTH_MODEL=${TRUTH_MODEL:-pyudales}
ASSIM_MODEL=${ASSIM_MODEL:-pylbm}
NNUDGE_M=${NNUDGE_M:-4.0}

# Truth/assim solver selection (mounted twice via the model@ package syntax) plus
# the per-solver overrides each backend needs on these grids. Setting
# TRUTH_MODEL != ASSIM_MODEL introduces genuine model error between the truth and
# the assimilation ensemble.
model_args=( "model@truth_model=${TRUTH_MODEL}" "model@assim_model=${ASSIM_MODEL}" )
case "${TRUTH_MODEL}" in
  pylbm)    model_args+=("truth_model.forward_model.cuda=false") ;;
  pyudales) model_args+=("truth_model.forward_model.nudging_config.nnudge_meters=${NNUDGE_M}") ;;
  pypalm)   ;;
  *) echo "Unknown TRUTH_MODEL='${TRUTH_MODEL}' (use pylbm|pyudales|pypalm)." >&2; exit 1 ;;
esac
case "${ASSIM_MODEL}" in
  pylbm)    model_args+=("assim_model.forward_model.cuda=false") ;;
  pyudales) model_args+=("assim_model.forward_model.nudging_config.nnudge_meters=${NNUDGE_M}") ;;
  pypalm)   ;;
  *) echo "Unknown ASSIM_MODEL='${ASSIM_MODEL}' (use pylbm|pyudales|pypalm)." >&2; exit 1 ;;
esac

# Domain + matching sensors per SIZE. The assimilation and (held-out, scored
# under run.skip_viz=false) validation sensors must sit inside the chosen grid.
case "${SIZE}" in
  test)
    # conf/size/test.yaml: [0,20] x [0,20] x [0,10] grid (nz=4), 3 s spin-up.
    size_args=(
      +size=test
      'obs.x_points=[2.5,2.5,18.0,18.0]' 'obs.y_points=[5.0,15.0,5.0,15.0]'
      'obs.z_points=[3.0,3.0,3.0,3.0]'
      'obs.validation_x_points=[10.0,10.0]' 'obs.validation_y_points=[6.0,14.0]'
      'obs.validation_z_points=[3.0,3.0]'
    )
    ;;
  large)
    # Larger domain: x in [-20, 40], y in [0, 40], z in [0, 32] (~2 m/cell
    # horizontally; nz is forced to 4 below, as for every size). No +size
    # overlay, so it inherits the case time.yaml (300 s window, 50 s spin-up)
    # -- substantially heavier than the tiny test grid.
    size_args=(
      'domain.bounds=[[-20.0,40.0],[0.0,40.0],[0.0,32.0]]'
      domain.nx=30 domain.ny=20
      'obs.x_points=[10.0,20.0,20.0,30.0,30.0]' 'obs.y_points=[20.0,10.0,30.0,10.0,30.0]'
      'obs.z_points=[2.0,2.0,2.0,2.0,2.0]'
      'obs.validation_x_points=[10.0,10.0]' 'obs.validation_y_points=[15.0,25.0]'
      'obs.validation_z_points=[2.0,2.0]'
    )
    ;;
  *)
    echo "Unknown SIZE='${SIZE}' — use 'test' (tiny smoke) or 'large'." >&2
    exit 1
    ;;
esac

# Shared, size-independent base: joint state + time-varying-parameter smoother,
# truth loaded from TRUTH_DIR. Combined with "${size_args[@]}" (domain + sensors).
common=(
  esmda/smoother=state_and_dynamic
  params@prior_params=dynamic params@truth_params=dynamic_truth
  # 30 s assimilation window, with 2 time knots per window in the dynamic
  # parameters (params/prior/truth grids kept equal; the smoother's
  # num_time_points = params.time_coords.num). Total horizon = 30*NUM_WINDOWS s.
  time.simulation_time=30
  params.time_coords.num=3 prior_params.time_coords.num=3 truth_params.time_coords.num=3
  "ensemble.ensemble_size=${ENSEMBLE_SIZE}" "ensemble.num_parallel_processes=${NPROC}"
  "esmda.num_steps=${NUM_STEPS}" "esmda.num_assimilation_windows=${NUM_WINDOWS}"
  # Truth from disk (state.nc/params.nc; x is auto-shifted onto the domain
  # frame, frames sliced to 30*NUM_WINDOWS s). TRUTH_DIR=null simulates inline
  # instead — only then are the truth_params overrides below actually used.
  "run.truth_dir=${TRUTH_DIR}" "run.truth_start_time=${TRUTH_START}"
  run.skip_viz=false
  obs.interval_seconds=5.0
  # Always 4 vertical levels, regardless of SIZE (overrides the per-size domain).
  domain.nz=4
  truth_params.correlation_length=30
)

run_esmda() { pixi run --environment dev python scripts/run_esmda.py "$@"; }

# echo "== 1/7  none (global update)  [SIZE=${SIZE}] =="
# run_esmda "${common[@]}" "${model_args[@]}" "${size_args[@]}" \
#   esmda/localization=none \
#   hydra.run.dir=.temp/loc_none

# echo "== 2/7  correlation-based localization (state only)  [SIZE=${SIZE}] =="
# run_esmda "${common[@]}" "${model_args[@]}" "${size_args[@]}" \
#   esmda/localization=correlation \
#   "esmda.localization.truncation_correlation=${RHO_T}" \
#   "esmda.localization.max_inflation=${MAX_INFL_C}" \
#   hydra.run.dir=.temp/loc_correlation

# echo "== 3/7  distance localization, small radius R=${RADIUS_SMALL} (state only)  [SIZE=${SIZE}] =="
# run_esmda "${common[@]}" "${model_args[@]}" "${size_args[@]}" \
#   esmda/localization=distance \
#   "esmda.localization.localization_radius=${RADIUS_SMALL}" \
#   "esmda.localization.max_inflation=${MAX_INFL_D}" \
#   hydra.run.dir=.temp/loc_distance_small

# echo "== 4/7  distance localization, large radius R=${RADIUS_LARGE} (state only)  [SIZE=${SIZE}] =="
# run_esmda "${common[@]}" "${model_args[@]}" "${size_args[@]}" \
#   esmda/localization=distance \
#   "esmda.localization.localization_radius=${RADIUS_LARGE}" \
#   "esmda.localization.max_inflation=${MAX_INFL_D}" \
#   hydra.run.dir=.temp/loc_distance_large

# Reduced-SVD state updates (docs/reduced_state_da.md). Incompatible with
# state localization, so both run with the global update (esmda/localization=none).
echo "== 5/7  reduced SVD, IC basis (update applied to the IC only)  [SIZE=${SIZE}] =="
run_esmda "${common[@]}" "${model_args[@]}" "${size_args[@]}" \
  esmda/localization=none esmda/state_reduction=svd \
  esmda.state_reduction.basis_source=initial_condition \
  "esmda.state_reduction.energy_fraction=${SVD_ENERGY}" \
  hydra.run.dir=.temp/loc_svd_ic

echo "== 6/7  reduced SVD, all-window-snapshot basis + final all-time-steps state update  [SIZE=${SIZE}] =="
run_esmda "${common[@]}" "${model_args[@]}" "${size_args[@]}" \
  esmda/localization=none esmda/state_reduction=svd \
  esmda.state_reduction.basis_source=window_snapshots \
  "esmda.state_reduction.energy_fraction=${SVD_ENERGY}" \
  esmda.final_time_smoothing=true \
  hydra.run.dir=.temp/loc_svd_snap_final

# Parameter-only baseline: the `dynamic` smoother (TimeVaryingParameterESMDA)
# estimates the time-varying parameters only -- no state estimation, so no
# localization. The later esmda/smoother + esmda/localization overrides win over
# the state_and_dynamic / distance set in `common`.
echo "== 7/7  parameter-only (dynamic smoother, no state estimation)  [SIZE=${SIZE}] =="
run_esmda "${common[@]}" "${model_args[@]}" "${size_args[@]}" \
  esmda/smoother=dynamic esmda/localization=none \
  hydra.run.dir=.temp/loc_param_only

echo
# Build a comparison table (mean metrics, <=4 decimals) across the modes from
# each run's run_summary.yaml: the per-parameter RMSE/CRPS and the held-out
# validation-sensor RMSE/CRPS (run.skip_viz=false also leaves the figures in
# each .temp/loc_<mode>/ dir). svd_ic / svd_snap_final = the reduced-SVD state
# updates; param_only = the dynamic smoother (no state est.).
# all modes = ["none", "correlation", "distance_small", "distance_large",
#              "svd_ic", "svd_snap_final", "param_only"]
pixi run --environment dev python - <<'PY'
import os, yaml

modes = ["svd_ic", "svd_snap_final", "param_only"]
summary = {}
for m in modes:
    p = f".temp/loc_{m}/run_summary.yaml"
    summary[m] = yaml.safe_load(open(p)) if os.path.isfile(p) else None

# (row label, path into run_summary.yaml — all "mean" of the per-window series)
rows = [
    ("inflow_angle  RMSE",     ("parameter_metrics", "inflow_angle", "rmse", "mean")),
    ("inflow_angle  CRPS",     ("parameter_metrics", "inflow_angle", "crps", "mean")),
    ("velocity_mag  RMSE",     ("parameter_metrics", "velocity_magnitude", "rmse", "mean")),
    ("velocity_mag  CRPS",     ("parameter_metrics", "velocity_magnitude", "crps", "mean")),
    ("validation vec RMSE",    ("sensor_metrics", "validation", "velocity_vector_rmse", "mean")),
    ("validation vec ES",      ("sensor_metrics", "validation", "velocity_vector_energy_score", "mean")),
]

def dig(d, path):
    for k in path:
        if not isinstance(d, dict) or k not in d:
            return None
        d = d[k]
    return d

def fmt(v):
    return f"{v:.4f}" if isinstance(v, (int, float)) else "n/a"

header = ["metric"] + modes
table = [[label] + [fmt(dig(summary[m], path)) for m in modes] for label, path in rows]
widths = [max(len(header[i]), *(len(r[i]) for r in table)) for i in range(len(header))]

def render(cells):
    return "  ".join(
        c.ljust(widths[i]) if i == 0 else c.rjust(widths[i])
        for i, c in enumerate(cells)
    )

print("=" * len(render(header)))
print("  localization comparison  (mean metric, lower is better)")
print("=" * len(render(header)))
print(render(header))
print("  ".join("-" * w for w in widths))
for r in table:
    print(render(r))
PY
echo
echo "Figures + full results per mode: .temp/loc_{none,correlation,distance_small,distance_large,svd_ic,svd_snap_final,param_only}/"
