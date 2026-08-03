"""Reductions of pre-extracted sensor and probe series.

Consumes ``(ensemble, time, sensor)`` arrays that a script has already pulled
out of the state files and turns them into the per-window statistics that
phase 1 scores: means, variances / TKE, velocity magnitudes.

Extraction deliberately stays in ``scripts/esmda/_esmda_common.py``: it needs
``data_assimilation``'s ``ObservationOperator`` and interpolation helpers,
which pull in jax, and it has to know the run-directory layout. Both are
forbidden here (leaf-library rule).

Populated in WP0.2 (move) and extended in WP1.3.
"""
