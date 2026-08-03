"""Probabilistic ensemble scores and the metric bundles built on them.

Holds the scoring vocabulary shared by every evaluation stage: fair CRPS and
CRPSS, the energy score, z-scores, ranks, spread--skill ratios, the VDI 3783/9
hit rate ``q``, and the parameter / sensor metric bundles that assemble them
into dictionaries for ``run_summary.yaml``.

Fairness matters here: pairwise estimators divide by ``M(M-1)``, not ``M**2``.
The biased form's optimum is a collapsed ensemble, which is the exact failure
mode these scores exist to detect. See
``docs/plans/esmda_turbulence_evaluation.md`` §6 for the formulas and
``docs/plans/esmda_evaluation/phase1_metrics_and_figures.md`` for the rollout.

Populated in WP0.2 (move) and extended in phase 1.
"""
