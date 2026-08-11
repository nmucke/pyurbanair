"""Filter-smoothing single-run pipeline: run_filter_smoothing /
compute_filter_smoothing_metrics / make_filter_smoothing_figures.

The inner state filter *is* the EnKF of ``scripts.filtering``, so the per-cycle
artifacts (``state_history.nc``, ``cycle_diagnostics.yaml``, the optional
``_ensemble_states/cycle_*/`` tree) are laid out identically and the metric and
figure stages reuse that pipeline's machinery unchanged
(``scripts.filtering._filtering_common``, and through it
``scripts.esmda._esmda_common``). What is filter-smoothing-specific -- the
moving window's cycle bookkeeping and the outer ESMDA loop's per-iteration
records -- lives in ``scripts.filter_smoothing._filter_smoothing_common``.
"""
