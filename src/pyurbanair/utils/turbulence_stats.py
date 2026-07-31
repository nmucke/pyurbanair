"""Flow-statistics helpers for the ESMDA evaluation stack (WP1.3).

Companion to :mod:`pyurbanair.utils.ensemble_scores`: that module scores
ensembles probabilistically, this one turns raw solver output into the
turbulence quantities worth scoring. Both are pure numpy so the metrics
stage, the figure stage and ``scripts/figspec`` share one implementation.

Deliberately empty for now. WP1.0 creates the module so the import path is
settled; WP1.3 fills it with

* ``StreamingMoments`` -- chunk-wise accumulation of ``n``, ``sum u_i`` and
  ``sum u_i u_j`` giving means, Reynolds stresses and TKE without ever
  holding a window state file in memory;
* ``colocate_components`` -- staggered-grid velocity components interpolated
  onto cell centres, per backend (uDALES/PALM staggering, pass-through for
  the uniform pylbm and surrogate grids);
* the mean-field scores (hit rate, FAC2, fractional bias, NMSE and its
  systematic/unsystematic split) plus ``block_bootstrap_std`` for their
  sampling floors.

Specification: ``docs/plans/esmda_evaluation/phase1_postprocessing_metrics.md``
(WP1.3). Adding stubs here ahead of that work would only create names that
have to be deleted, so there are none.
"""

from __future__ import annotations

__all__: list[str] = []
