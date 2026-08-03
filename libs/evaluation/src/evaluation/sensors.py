"""Reductions of pre-extracted sensor and probe series to window statistics.

Consumes ``(ensemble, time, sensor)`` arrays a script has already pulled out of
the state files. Extraction itself stays in ``scripts/esmda/_esmda_common.py``:
it needs ``data_assimilation``'s observation operator (jax) and the run-dir
layout, both forbidden here.

Populated in WP0.2 (move), extended in WP1.3.
"""
