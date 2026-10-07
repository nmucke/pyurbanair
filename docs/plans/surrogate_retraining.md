# Surrogate retraining (when you next retrain)

Not scheduled. Do it in one round with
[surrogate_training_data_spinup.md](surrogate_training_data_spinup.md), which
needs the same corpus.

## Why

Since #172, a surrogate step from frame `t` to `t+1` is driven by the
parameters at its **start** (the time of frame `t`), in training and inference
alike. That only holds if the frames carry their physical times. Corpora
generated before #163 by pylbm, PALM or spun-up uDALES don't: their frames are
labelled one output interval early (`0 … T−tf` instead of `tf … T`), so each
frame's parameter row is the value from one `tf` too early. The current
training corpora are spun-up uDALES (`configs/surrogate/generate_data.yaml`:
an adaptive spin-up of at least 300 s, `tf = 5 s`), so they are affected, and
so are the weights trained on them: inference now runs with a one-`tf` (5 s)
parameter lag.

The effect is small: the training parameters are AR(2) series with a 300 s
correlation length (`configs/params/surrogate_training_data.yaml`). It only
goes away by retraining.

**Check a corpus:** a state file whose `time` starts at 0 has the old labels;
one starting at `output_frequency` is fine (`docs/neural_surrogates.md`,
"Parameter time convention").

## Options

1. **Relabel the existing corpus (no solver runs).** The flow fields are
   correct; only the labels are off. Per sample: `state.time += tf`; parameter
   row `t` ← old row `t+1`; drop the last frame, whose correct parameter isn't
   stored (the datasets expect a parameter row for every frame). Scalar
   parameters are unchanged. A short script in `scripts/tools/`, run once per
   corpus, with a test on a tiny corpus that the relabelled pairs equal those of
   a freshly generated one.
2. **Regenerate** with `scripts/surrogate/generate_data.py`. Worth it if the
   corpus changes anyway (new geometries, parameter ranges).

Then retrain (`scripts/surrogate/train.py`) and re-run the surrogate
evaluations against the old weights to see what the lag cost.

## Done when

- The corpus starts at `output_frequency` (relabelled or regenerated).
- The surrogates are retrained on it and their evaluation is compared with the
  old weights.
- This file moves to `docs/plans/implemented/`.
