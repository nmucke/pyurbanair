# Configuration overview

Runs are configured with [Hydra](https://hydra.cc). The setup is built from a few
**orthogonal axes** — each answers one question and they don't overlap, so you can
mix them freely:

| Axis (CLI) | Answers | Where it lives |
|------------|---------|----------------|
| `case=<name>` | **What** physical experiment: domain bounds + grid resolution, geometry/STL, sensors (assimilation **and** validation), per-window time | [`case/`](case/) — **one self-contained file per case** |
| `model@…=<backend>` | **Which** solver: `pylbm` / `pyudales` / `pypalm` / `neural_surrogate` | [`model/`](model/) |
| `params…=<kind>` | parameter prior: `static` / `dynamic` (+ `*_truth`) | [`params/`](params/) |
| `esmda/smoother=…` `esmda/localization=…` `esmda/state_reduction=…` | the DA method | [`esmda/`](esmda/) |

> **Set up a new experiment = edit (or copy) one file in [`case/`](case/).** Domain,
> grid, geometry and sensors all live there.

The **compute budget** (ensemble size, parallelism, ESMDA steps/windows, param
knots) is baked into the two entry points at medium-sized defaults — change it
with plain CLI overrides, e.g. `ensemble.ensemble_size=8`,
`esmda.num_assimilation_windows=10`.

## Entry points (primary configs)

There are exactly two run entry points, and each is **self-contained** — it
inlines the shared base (output `paths`, the `time.num_param_knots` knob,
`ensemble` defaults, the `run:` namespace and Hydra settings) rather than
pulling them from separate files:

| Script | `config_name` | What it adds |
|--------|---------------|--------------|
| `scripts/run_forward_model.py` | [`run_forward_model.yaml`](run_forward_model.yaml) | `case` + `params` + `model@model` + the inlined base |
| `scripts/run_esmda.py` | [`run_esmda.yaml`](run_esmda.yaml) | the inlined base + inlined `esmda:` scalars + the esmda axes; doubles the model mount (`@truth_model` / `@assim_model`) and the params mount (`@truth_params` / `@prior_params`) so truth and prior never share a generative process (anti-inverse-crime) |

(`scripts/neural_surrogate/generate_training_data.py` uses
[`neural_surrogate/training_data.yaml`](neural_surrogate/training_data.yaml), and
the surrogate train/test scripts use
[`neural_surrogate/training.yaml`](neural_surrogate/training.yaml) /
[`neural_surrogate/testing.yaml`](neural_surrogate/testing.yaml). All three base
off the run entry points / `case` for their physical setup.)

## Groups (selected from the entry points)

- [`case/`](case/) — the experiment (domain+grid+geometry+sensors+time).
- [`model/`](model/) — solver backend.
- [`params/`](params/) — parameter samplers (static/dynamic + `*_truth`).
- [`esmda/`](esmda/) — `smoother/`, `localization/`, `state_reduction/` (the base
  `esmda:` scalars are inlined in `run_esmda.yaml`; these groups override them).

## Smoke runs

For a fast smoke run, shrink the grid / window / ensemble on the CLI, e.g.
`domain.nx=20 domain.ny=20 domain.nz=4 time.simulation_time=5 ensemble.ensemble_size=4`.
The pytest suite applies exactly such a smoke shape (`_SMOKE_OVERRIDES` in
`tests/conftest.py`).

## Examples

```bash
# Forward model, default case, smaller ensemble
python scripts/run_forward_model.py model@model=pylbm ensemble.ensemble_size=8

# ESMDA: joint state + time-varying params, distance localization, Barcelona
python scripts/run_esmda.py case=barcelona \
  esmda/smoother=state_and_dynamic esmda/localization=distance \
  params@prior_params=dynamic params@truth_params=dynamic_truth

# One-off override: coarsen the grid on the CLI
python scripts/run_forward_model.py domain.nx=40 domain.ny=40
```
