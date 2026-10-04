"""ESMDA single-run pipeline: run_esmda / compute_esmda_metrics / make_esmda_figures.

The three stages share their lazy truth-access and sensor-series helpers via
``scripts.esmda._esmda_common`` (also reused by the filtering pipeline in
``scripts.filtering``).
"""
