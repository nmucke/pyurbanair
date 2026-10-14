"""The comparison diagnostics."""

from __future__ import annotations

import numpy as np
from neural_surrogate_baselines import diagnostics


def test_profiles() -> None:
    rng = np.random.default_rng(0)
    fields = rng.standard_normal((50, 3, 2, 4, 4))
    fields[:, 2] = fields[:, 0]  # w = u: -<u'w'> = -<u'^2>
    fluid = np.ones((2, 4, 4), dtype=bool)
    prof = diagnostics.profiles(fields, fluid)
    prime = diagnostics.fluctuations(fields, fluid)
    np.testing.assert_allclose(prof["mean_u"], fields[:, 0].mean(axis=(0, 2, 3)))
    np.testing.assert_allclose(prof["uw"], -(prime[:, 0] ** 2).mean(axis=(0, 2, 3)))
    np.testing.assert_allclose(
        prof["tke"], 0.5 * (prime**2).sum(axis=1).mean(axis=(0, 2, 3))
    )
