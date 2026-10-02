"""Shared library for generating the EnKF-2026 talk figures.

See ``docs/figure_specs.md`` for the full specification. The driver scripts
``scripts/make_figures_*.py`` build on the helpers here:

  * :mod:`figspec.dataio`  -- run discovery, loaders, common-grid interpolation.
  * :mod:`figspec.mask`    -- building (solid-cell) masks from the case STL.
  * :mod:`figspec.figcommon` -- shared panel builders on top of the two above.

The UrbanAIR palette / rcParams / save helpers and the metric definitions live
in the ``evaluation`` library: :mod:`evaluation.style` and
:mod:`evaluation.scores`.
"""
