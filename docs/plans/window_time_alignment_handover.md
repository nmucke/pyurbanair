# Handover: put window files on the right global time

Found while reviewing PR #162. Post-processing maps each window file onto the
run's global time axis by anchoring its **first** frame on the window start:

```python
time - time[0] + w * sim_time
```

But a run's output frames sit in `(start, start + sim_time]`: the first frame
is one output interval *after* the window start (`_time_window` in
`scripts/utils/helper_functions.py` says so). A filter's posterior holds one
frame per cycle, so its first frame is a whole cycle in. If that holds, every
window's ensemble series is labelled one output interval (smoother) or one
cycle (filter, hybrid) too early. The truth, read on its own correct axis, is
then compared with the wrong ensemble frame. That would bias every existing
`sensors` and `sensor_statistics` score, the figures, and the new #162
diagnostics that reuse them.

This PR **changes results**. Read `AGENTS.md` and `docs/evaluation.md` first.
Branch: continue on `fix/window-time-alignment`, which holds this file, and
open the PR into `main`. Don't merge it.

## Where

The same expression appears in three places:
- `scripts/compute_metrics.py`, `_read_ensemble`:
  `global_time = time - time[0] + t_start`.
- `scripts/visualize_assimilation.py`, `add_window`:
  `time - time[0] + t_start`.
- `scripts/utils/helper_functions.py`, `concat_windows` (parameters):
  `ds.time - ds.time[0] + w * sim_time`. Smoother parameter knots include the
  window start (`keep_start=True`), so they may already be right. Filter
  parameters (`cycles_to_time`) start one cycle in.

The extra-file path in `_read_ensemble` (prior and forecast, matched to the
posterior's times "before the end") is consistent with the posterior, so it
inherits whatever the posterior gets.

## Step 1: prove it with a test (before any fix)

Write a test in `tests/scripts/` that builds a tiny run directory by hand:
- a truth;
- posterior (and prior, forecast) window files that are exactly the truth's
  frames, stamped the way `run_smoother.py`, `run_filtering.py` and
  `run_hybrid.py` actually stamp them (read each script; reuse their helpers);
- a varying truth signal, so a one-frame shift shows up.

Then run `compute_metrics.run` on it. Every sensor RMSE must be 0, and the
parameter metrics must match the truth's knots. Also check that the figure
inputs in `visualize_assimilation.py` line up (the series it plots).

Report the result per method before fixing. If the test passes on `main`, the
offset isn't real: stop, say so, and close this plan as rejected.

## Step 2: fix in one place

One helper in `scripts/utils/` maps a window file onto the global axis. Every
window file ends on the window's end, so anchor on the end:
`time - time[-1] + (w + 1) * sim_time`. Check that this holds for every file
type: posterior, prior, forecast, the parameter files of each method. Use the
helper in all three places above, and delete the old expressions. If some file
type does not end on the window's end, fix how it is stamped where it is
written, rather than special-casing the reader.

## Step 3: also in this PR

`docs/evaluation.md` needs one sentence on reading Desroziers and χ² (asked
for in the #162 review). Both assume the update used consistent error
statistics. With a collapsed spread the gain is about 0, so `d_a ≈ d_f`, and
Desroziers returns the innovation RMS, which says nothing about R. Read them
only where `spread_skill.ratio` and `innovation_chi2_diag` are near 1.

## Results that change

Rerun the three tiny workflows before and after (`--config-dir tests/configs
+test=assimilation`, smoother, filtering, hybrid; the same surrogate weights)
and list in the PR which `metrics.yaml` values change, and by how much. Name
anything already reported from older runs that used this code (ask the user).

## Done when

- [ ] The step 1 test fails on `main` and passes after the fix, for all three
      methods (or the plan is rejected because it passes on `main`).
- [ ] One helper, used everywhere a window file goes onto the global axis.
- [ ] The Desroziers / χ² sentence is in `docs/evaluation.md`.
- [ ] Changed values listed in the PR.
- [ ] `tests/scripts`, `tests/evaluation`, `pre-commit` pass; CI green on Linux
      and macOS.
- [ ] This file moved to `docs/plans/implemented/`.
