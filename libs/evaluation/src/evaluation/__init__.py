"""Evaluation metrics and figures for ensemble data assimilation runs.

This is a **leaf library**: plain functions, arrays / ``xr.Dataset``s in →
floats / dicts / ``Figure``s out. It knows nothing about Hydra, run-directory
layout, or the forward-model backends, and it never imports ``pyurbanair``,
``data_assimilation`` or a backend package. Scripts
(``scripts/esmda/``, ``scripts/filtering/``, ``scripts/figure_creation/``)
own I/O and orchestration and call in here.

Five flat modules, no subpackages. This is the *target* shape from the master
plan, not an inventory -- phases 1--3 fill it in, and each module's own
docstring says what has actually landed there:

- :mod:`evaluation.scores` -- probabilistic ensemble scores (fair CRPS/CRPSS,
  energy score, z-score, rank, spread--skill, hit rate) and the parameter /
  sensor metric bundles built on them.
- :mod:`evaluation.turbulence` -- flow statistics: streaming moment
  accumulation over state files, block bootstrap, spectra.
- :mod:`evaluation.sensors` -- reductions of pre-extracted sensor / probe
  series to window statistics. Extraction itself stays in the scripts: it
  needs the observation-operator machinery from ``data_assimilation`` (jax).
- :mod:`evaluation.style` -- figure conventions: colors, quantile bands,
  shared norms, solid-cell masking.
- :mod:`evaluation.figures` -- one function per figure ID plus the general
  state / parameter plots.

Design rules (deliberate -- keep them): no base classes, no registries. The
only class the library may grow is the streaming moment accumulator
(:class:`evaluation.turbulence.MomentAccumulator`, WP1.4), which is genuinely
stateful; it is still the only one. Add abstraction only when a third caller
would otherwise copy-paste.

Nothing is re-exported here on purpose: ``style`` and ``figures`` import
matplotlib, and a root re-export would drag it into every consumer. Import
from the module you need (``from evaluation.scores import ...``).

See ``docs/plans/esmda_evaluation/master_plan.md``.
"""
