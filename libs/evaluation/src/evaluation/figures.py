"""Figure builders: one function per figure ID, plus the general plots.

The evaluation figure set (see ``docs/plans/esmda_turbulence_evaluation.md``
§7): P1 parameter marginals, S1 vertical profiles at stations, S5 sensor time
series with held-out sensors in their own column, F1 time-averaged slice
comparison, D1 rank histogram; then D3 (data-mismatch decay) and S4 (energy
spectra) in phases 2--3. The general state / parameter plots moved out of
``pyurbanair.plotting`` live here too.

Functions build and return a ``Figure``; callers decide where it is written.
Field figures operate on time averages or statistics only -- never on
instantaneous fields, which decorrelate after a Lyapunov horizon and measure
chaos rather than parameter quality.

Populated in WP0.2 (move) and extended in WP1.5.
"""
