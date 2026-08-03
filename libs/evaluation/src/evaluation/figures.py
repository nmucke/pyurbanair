"""Figure builders: one function per figure ID, plus the general plots.

P1, S1, S5, F1, D1 (then D3 and S4 in phases 2--3), listed in
``docs/plans/esmda_turbulence_evaluation.md`` §7, alongside the general state /
parameter plots moved out of ``pyurbanair.plotting``.

The new figure set takes time averages or statistics only -- never
instantaneous fields, which decorrelate after a Lyapunov horizon and measure
chaos rather than parameter quality. The general plots moved from
``pyurbanair.plotting`` predate that rule: several of them do plot snapshots,
and they take an ``output_path`` and write the file themselves rather than
returning a ``Figure``. WP0.2 is a pure refactor and keeps both properties;
changing either is a later cleanup.

Populated in WP0.2 (move), extended in WP1.5.
"""
