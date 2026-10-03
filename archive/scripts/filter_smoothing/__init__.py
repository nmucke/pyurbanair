"""Filter-smoothing hybrid entry point: ESMDA parameter MDA + a sequential filter.

One stage only (``run_filter_smoothing``). It writes BOTH downstream schemas —
the ESMDA per-window one and the filtering-native one — so the existing
``scripts.esmda`` and ``scripts.filtering`` metric/figure stages read a hybrid
run directory unchanged, and it reuses their per-window artifact writers rather
than reimplementing them.
"""
