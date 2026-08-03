"""Figure builders: one function per figure ID, plus the general plots.

P1, S1, S5, F1, D1 (then D3 and S4 in phases 2--3), listed in
``docs/plans/esmda_turbulence_evaluation.md`` §7, alongside the general state /
parameter plots moved out of ``pyurbanair.plotting``.

Functions build and return a ``Figure``; callers decide where it goes. Field
figures take time averages or statistics only -- never instantaneous fields,
which decorrelate after a Lyapunov horizon and measure chaos rather than
parameter quality.

Populated in WP0.2 (move), extended in WP1.5.
"""
