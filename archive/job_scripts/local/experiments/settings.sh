#!/usr/bin/env bash
# THE one place to set the arguments shared by all three DA-method campaigns.
#
# Sourced by run_{esmda,filtering,filter_smoothing}_experiments.sh, which each
# expand the same axes into their own entry point's override names. Everything
# here is `: "${VAR:=default}"`, so any value can be set by editing this file OR
# by exporting it for a single campaign:
#
#   NUM_WINDOWS_LIST="2 4" INFLOW_LIST=periodic \
#     bash job_scripts/local/experiments/run_esmda_experiments.sh
#
# CONSTANTS vs AXES. The `*_LIST` variables are the swept axes — every campaign
# runs their full cross product, one run per combination. Everything else is
# held constant across every run of every campaign, which is what makes the
# three methods comparable: same case and grid, same ensemble, same horizon per
# window, same observation error, same assimilation backend.
set -uo pipefail

# ---------------------------------------------------------------------------
# Constant across every run (the comparability contract)
# ---------------------------------------------------------------------------

# Geometry / case, and the assimilation backend. The TRUTH backend is an AXIS
# (TRUTH_MODEL_LIST, below): with truth == assim the campaigns measure the DA
# method alone, and with a pypalm truth against the pyudales assimilation model
# they additionally measure model error.
: "${CASE:=xie_and_castro}"
: "${ASSIM_MODEL:=pyudales}"

# Which parameters the DA estimates. Empty -> the entry point's own default
# (inflow_angle + velocity_magnitude). A CROSS-MODEL campaign (pypalm truth,
# pyudales assimilation) is exactly the case the model-error compensation knobs
# exist for (docs/esmda_model_error_parameters.md), so consider:
#   PARAMS_TO_ESTIMATE='params_to_estimate=[inflow_angle,velocity_magnitude,vertical_inflow_exponent]'
# Add `sgs_constant` only after checking the ranges in conf/params/: the three
# backends' sgs_constant are NOT the same quantity (uDALES/pylbm take a
# dimensionless Smagorinsky-family constant, PALM an eddy diffusivity in m^2/s,
# and setting it on PALM at all is a closure regime switch), so a shared prior
# range across a cross-model pair is meaningless at best and divergent at worst.
: "${PARAMS_TO_ESTIMATE:=}"

# Per-window horizon (seconds). ONE assimilation window is `SIMULATION_TIME`
# long in all three entry points, so the experiment horizon of a run is
# `SIMULATION_TIME * <num windows>` — the num-windows axis below is therefore
# also a horizon axis. `OUTPUT_FREQUENCY` is the observation cadence: the filter
# runs one analysis cycle per output frame, the smoothers aggregate frames into
# `OBS_INTERVAL_LIST`-wide bins.
: "${SIMULATION_TIME:=120.0}"
: "${OUTPUT_FREQUENCY:=2.0}"
# Spin-up plateau in front of the window, PER INFLOW SETTING (the one non-axis
# knob that is not one value for the whole campaign). An inflow_outflow run is
# fed by its inlet from the first step, whereas a periodic run has to build its
# entire momentum field from rest through the nudging relaxation, so it needs
# markedly longer before the window starts.
: "${SPINUP_TIME_INFLOW:=50.0}"
: "${SPINUP_TIME_PERIODIC:=150.0}"
# Time-varying parameter knot spacing (both the truth sampler's and the dynamic
# smoother's). All three entry points carry the key, so pinning it here keeps
# the campaigns identical on that axis too.
: "${SECONDS_PER_KNOT:=30.0}"

# Ensemble. WORKERS is the number of concurrent single-core uDALES members per
# run; this box is DRAM-bandwidth-bound past ~4-8 of them, and NUM_LANES (below)
# multiplies it, so raise WORKERS*NUM_LANES only after re-benchmarking.
# num_cpus_per_process stays 1: uDALES mis-stitches its output with ncpu>1.
: "${ENSEMBLE_SIZE:=50}"
: "${WORKERS:=10}"

# Shared DA knobs. `OBS_ERROR_STD` is deliberately one value for all three
# methods (the entry points ship different defaults, 0.25 / 0.1 / 0.25) — the
# observation error is a property of the experiment, not of the algorithm.
: "${SEED:=42}"
: "${OBS_ERROR_STD:=0.1}"

# Truth parameter sampler, shared by all three (the prior sampler is per-method:
# the filter only supports a static prior, the smoothers take a dynamic one).
: "${TRUTH_PARAMS:=dynamic_sine}"

# Truth source. Empty -> each run simulates its own truth inline (the default).
# A path to a dir holding state.nc + params.nc -> every run assimilates that
# artifact. NOTE a shared truth is only meaningful when the INFLOW axis holds a
# single value: the inflow setting changes the truth's forcing, so one artifact
# cannot serve two of them. The campaigns refuse to start on that combination.
: "${TRUTH_DIR:=}"
: "${TRUTH_START_TIME:=}"

# --- Inlet turbulence (the `inflow_turb` setting of the INFLOW axis) ---------
#
# The two backends do NOT share a mechanism: uDALES gets digital-filter (Xie &
# Castro) driver planes injected at the inlet plane and advected in, PALM gets
# instantaneous random kicks re-injected in a strip near the inlet every
# dt_disturb (its recycling and synthetic-turbulence generators are both
# unreachable from a config toggle — docs/pypalm.md §8). What CAN be matched is
# the injected turbulence's amplitude, time scale and spatial extent, so the two
# knobs below are shared and the PALM-specific ones are DERIVED from them by
# default. See README.md "How close are the two inflow_turb setups?".

# Turbulence intensity u'_rms / |U_ref|, the shared amplitude knob.
: "${INLET_INTENSITY:=0.05}"
# Reference speed the intensity is taken against (m/s): the params samplers'
# mean velocity_magnitude (7.5 in conf/params/dynamic*.yaml). uDALES needs no
# such constant — it scales by its own |U(z)| at run time — but PALM's
# disturbance_amplitude is an ABSOLUTE m/s, so the conversion needs one.
: "${INLET_REFERENCE_SPEED:=7.5}"

# uDALES: digital-filter length scales, as set in conf/model/pyudales.yaml.
# L_y = 7.5% of the 80 m span and L_z = 19% of the 32 m height, both well inside
# the regime where the generator's plane-mean subtraction is harmless (it starts
# discarding a large share of the fluctuation energy once L_y/L_z approach the
# plane's own dimensions, and warns when they do). L_x sets the AR(1) time scale
# via Taylor, T = L_x / U_ref = 0.8 s here.
: "${UDALES_INLET_LENGTH_SCALE_X:=6.0}"
: "${UDALES_INLET_LENGTH_SCALE_Y:=6.0}"
: "${UDALES_INLET_LENGTH_SCALE_Z:=6.0}"
# Driver-plane sampling step (&DRIVER dtdriver). The solver interpolates
# linearly between records, so the injected turbulence is band-limited to
# ~1/(2*dtdriver). This is a RESOLUTION, not an injection rate — do not equate
# it with PALM's dt_disturb (see below).
: "${UDALES_INLET_TIME_STEP:=0.5}"

# PALM, matched to the uDALES settings above as far as the mechanisms allow.
#   amplitude   DERIVED when empty. PALM draws uniform noise on [-1.5A, 1.5A]
#               (disturb_field.f90:119), whose rms is A*sqrt(3)/2, so
#               A = intensity*U_ref/(sqrt(3)/2) targets the same NOMINAL rms as
#               uDALES' intensity*|U|: 0.05*7.5 -> A = 0.433 m/s.
#   dt_disturb  NOT derived from L_x, deliberately. PALM ADDS each kick to the
#               existing field (disturb_field.f90:224) rather than prescribing a
#               boundary value the way uDALES' driver planes do, so the realised
#               variance grows roughly like tau_decay/dt_disturb: pacing kicks at
#               the eddy time T = L_x/U_ref = 0.8 s would inject ~6x more energy
#               than at 5 s and land far above the 5% target. Kept at the
#               conf/model/pypalm.yaml value, which is the same order as the
#               turbulence decay time. PALM's realised intensity is not
#               analytically predictable — measure it in a run and retune this
#               (or the amplitude) if it misses 5%.
: "${PALM_DISTURBANCE_AMPLITUDE:=}"
: "${PALM_DT_DISTURB:=5.0}"
# Where PALM's kicks land. The defaults below replace PALM's own auto values,
# which are a poor match for an inlet-turbulence experiment on this domain:
#   * begin/end (grid points from the inflow, x-range of the perturbed strip).
#     PALM auto-derives min(10, nx/2)=10 and min(100, 3nx/4)=29
#     (check_parameters.f90:2755-2768), i.e. x = -5..23 m here — straight
#     THROUGH the building array (which starts at x=5), whereas uDALES injects
#     only at the inlet plane. 2..14 keeps the strip at x = -17..1 m, upstream
#     of the array.
#   * level_b/level_t (m). PALM auto-derives zu(3)..zu(nzt/3) = 5..9 m here, so
#     the kicks never reach the 17 m rooftops; uDALES' inlet plane carries
#     fluctuations over the full height. level_t is capped by PALM at
#     zu(nzt-2) = 27 m on this 32 m / nz=16 grid (dz = 2 m) — RETUNE IT with the
#     grid; PALM aborts with PAC0155 if it is too high.
#     Empty level_b keeps PALM's floor (it rejects anything below zu(3) = 5 m).
# Export any of them EMPTY to fall back to PALM's own auto behaviour — which is
# why these three take the default only when UNSET (`:=` would treat an
# exported empty string as "no value" and re-apply the default).
: "${PALM_DISTURBANCE_BEGIN=2}"
: "${PALM_DISTURBANCE_END=14}"
: "${PALM_DISTURBANCE_LEVEL_T=26.0}"
: "${PALM_DISTURBANCE_LEVEL_B:=}"
# PALM's SEPARATE one-off random kick at cold-start init, which breaks the
# symmetry of its smooth analytic initial profile. It is not inlet turbulence,
# so the non-turbulent inflow settings leave it at this value (default off,
# matching conf/model/pypalm.yaml); `inflow_turb` forces it on. Set true if a
# `periodic` PALM truth is developing turbulence too slowly.
: "${PALM_INITIAL_SEED:=false}"
# PALM slab decomposition (npex=ncpu, npey=1). The inflow/outflow multigrid
# solver needs uniform subdomains, so ncpu must DIVIDE domain.nx (40 for
# xie_and_castro -> 1,2,4,5,8,10,20,40). Applied to pypalm mounts only; PALM is
# the slow half of a cross-model run, and the truth is a single simulation, so
# giving it cores costs nothing the ensemble needs.
: "${PALM_NCPU:=1}"

# ---------------------------------------------------------------------------
# The swept axes (space-separated lists; the cross product is the campaign)
# ---------------------------------------------------------------------------

# Truth backend(s). `pyudales` -> truth == assimilation model (no model error).
# `pypalm` -> a PALM truth assimilated by the uDALES ensemble, i.e. a genuine
# cross-model experiment; the run dirs are named <truth>_to_<assim>_... so both
# live side by side. PALM needs its binary built once (conf/model/pypalm.yaml
# ships `compile: false`), an even nx/ny and nz >= 16 — all true for the
# xie_and_castro case at its default grid.
: "${TRUTH_MODEL_LIST:=pyudales}"

# Number of assimilation windows. Mapped to esmda.num_assimilation_windows /
# filtering.num_assimilation_windows / filter_smoothing.num_assimilation_windows
# — the same unit in all three.
: "${NUM_WINDOWS_LIST:=3}"

# Localization: `correlation` (Vossepoel et al. adaptive correlation-based) or
# `none` (global update). Mapped to esmda/localization and/or
# filtering/localization per method.
: "${LOCALIZATION_LIST:=none correlation}"

# Observation aggregation bin width in seconds (esmda.interval_seconds).
# SMOOTHER-SIDE ONLY: it applies to the ESMDA and filter-smoothing campaigns.
# The pure filter never aggregates (one analysis per observation frame), so
# run_filtering_experiments.sh ignores this axis. Bins are absolute and start at
# the window's first frame, so a width that does not divide SIMULATION_TIME just
# leaves a short final bin.
: "${OBS_INTERVAL_LIST:=15.0}"

# Inflow / boundary condition, applied to BOTH model mounts:
#   inflow       inflow_outflow BCs, smooth mean profile, no synthetic
#                turbulence (nudged inlet + interior relaxation).
#   inflow_turb  inflow_outflow BCs + the digital-filter synthetic inlet
#                (BCxm=3 / idriver=2). Volume nudging is turned OFF by the
#                backend in this mode, by design.
#   periodic     periodic x BCs; the interior nudging is then the only momentum
#                source, so it stays on and no inlet turbulence is possible.
: "${INFLOW_LIST:=inflow inflow_turb periodic}"

# ---------------------------------------------------------------------------
# Campaign execution
# ---------------------------------------------------------------------------

# Output root: run dirs, figures, logs and the per-lane solver scratch all live
# under it. One subdir per method (esmda/ filtering/ filter_smoothing/), one run
# dir per axis combination inside it. Relative paths are resolved against the
# repo root.
#
# On scratch2 deliberately, NOT under the repo's .temp/: a campaign is ~225 GB
# (~450 GB with a pypalm truth as well), and the repo's own filesystem
# (/export/scratch1) does not have that. Re-check `df -h` before a launch —
# these are shared disks.
: "${RESULTS_ROOT:=/export/scratch2/ntm/experiments}"

# Concurrent runs. Each lane gets its OWN paths.experiment_dir — the uDALES
# scratch tree (experiment/999, ensemble_experiments/, outputs/) is the one
# piece of shared mutable state a run has, and two runs sharing it corrupt each
# other's namoptions and fielddumps. NUM_LANES=1 keeps wall times comparable
# between runs; >1 trades that for throughput.
: "${NUM_LANES:=1}"

# Restrict the queue to these run ids (space-separated), e.g.
# ONLY="pyudales_to_pyudales_w4_locnone_obs30_inflow".
: "${ONLY:=}"

# Print the resolved command for every run and exit without running anything.
: "${DRY_RUN:=0}"

# Extra Hydra overrides appended to EVERY run of every campaign (word-split).
: "${EXTRA_ARGS:=}"

# Per-run visualization inside the runner (the pipeline's figure stage runs
# either way; this gates the state/sensor blocks the shared stages read).
: "${SKIP_VIZ:=false}"

# Keep JAX on the CPU. The pipelines invoke the runners under `pixi run -e
# cuda`, so without this the parent AND every forkserver worker creates its own
# CUDA context; on a shared card that exhausts device memory and the run dies
# hours in with a bare BrokenProcessPool. Nothing here wants the GPU (uDALES is
# CPU Fortran, the analysis is a small SVD), and pinning the backend also keeps
# the runs numerically comparable.
export JAX_PLATFORMS=cpu
export XLA_PYTHON_CLIENT_PREALLOCATE=false
