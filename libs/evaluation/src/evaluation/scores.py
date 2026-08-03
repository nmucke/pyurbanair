"""Probabilistic ensemble scores and the metric bundles built on them.

Fair CRPS/CRPSS, energy score, z-scores, ranks, spread--skill, the hit rate
``q``, and the parameter / sensor bundles that assemble them for
``run_summary.yaml``.

From WP1.1 the pairwise estimators divide by ``M(M-1)`` rather than ``M**2``,
because the biased form's optimum is a collapsed ensemble -- the exact failure
these scores exist to detect. WP0.2 moves the biased forms in unchanged, so
until WP1.1 lands what is here is the old estimator. Formulas in
``docs/plans/esmda_turbulence_evaluation.md`` §3--§6; rollout in
``phase1_metrics_and_figures.md``.

Populated in WP0.2 (move), extended in phase 1.
"""
