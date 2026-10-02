"""Sequential filtering (EnKF) single-run pipeline: run_filtering /
compute_filtering_metrics / make_filtering_figures.

The metric and figure stages reuse the ESMDA pipeline's truth-access and
sensor-series helpers (``scripts.esmda._esmda_common``) via the small
filtering-specific glue in ``scripts.filtering._filtering_common``.
"""
