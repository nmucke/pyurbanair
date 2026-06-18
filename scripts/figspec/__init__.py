"""Shared library for generating the EnKF-2026 talk figures.

See ``docs/figure_specs.md`` for the full specification. The driver scripts
``scripts/make_figures_*.py`` build on the helpers here:

  * :mod:`figspec.style`   -- UrbanAIR palette, rcParams, save helpers, labels.
  * :mod:`figspec.dataio`  -- run discovery, loaders, common-grid interpolation.
  * :mod:`figspec.metrics` -- field / parameter / sensor metrics.
  * :mod:`figspec.mask`    -- building (solid-cell) masks from the case STL.
"""
