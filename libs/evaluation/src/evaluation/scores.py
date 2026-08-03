"""Probabilistic ensemble scores and the metric bundles built on them.

Fair CRPS/CRPSS, energy score, z-scores, ranks, spread--skill, the hit rate
``q``, and the parameter / sensor bundles that assemble them for
``run_summary.yaml``.

Pairwise estimators here divide by ``M(M-1)``, not ``M**2``: the biased form's
optimum is a collapsed ensemble, the exact failure these scores exist to
detect. Formulas in ``docs/plans/esmda_turbulence_evaluation.md`` §3--§6.

Populated in WP0.2 (move), extended in phase 1.
"""
