"""The online adaptive observation error (data_assimilation.adaptive_observation_error).

A linear-Gaussian toy, cycled like the parameter-only smoother over windows:
the true observation error is k times the configured one (k per height group),
and the adaptive option must bring the Desroziers ratio back to ~1 and the
posterior back to calibrated within a few windows.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from data_assimilation.adaptive_observation_error import AdaptiveObservationError
from data_assimilation.filtering.analysis import stochastic_enkf_update

HEIGHTS = np.repeat([2.0, 8.0, 20.0], 12)  # three groups of 12 observations
STD = 0.1  # configured std
TRUE_FACTOR = np.repeat([2.0, 4.0, 6.0], 12)  # true std / configured std
N_PARAMS, N_E, N_STEPS, N_WINDOWS = 3, 64, 4, 6


def _cycle(adaptive: AdaptiveObservationError | None, seed: int) -> dict:
    """One toy run: per window the ratio of the Desroziers std to the std used,
    and the posterior z-scores of the true parameters."""
    rng = np.random.default_rng(seed)
    key = jax.random.PRNGKey(seed)
    H = 0.5 * rng.standard_normal((HEIGHTS.size, N_PARAMS))
    truth = rng.standard_normal(N_PARAMS)
    members = rng.standard_normal((N_PARAMS, N_E))
    configured = np.full(HEIGHTS.size, STD**2)
    ratio, z = [], []
    for _ in range(N_WINDOWS):
        y = H @ truth + STD * TRUE_FACTOR * rng.standard_normal(HEIGHTS.size)
        used = (
            configured if adaptive is None else adaptive.variance(configured, HEIGHTS)
        )
        forecast = H @ members
        for _ in range(N_STEPS):
            key, sub = jax.random.split(key)
            members = np.asarray(
                stochastic_enkf_update(
                    jnp.asarray(members),
                    jnp.asarray(H @ members),
                    jnp.asarray(y),
                    jnp.asarray(used),
                    sub,
                    alpha=float(N_STEPS),
                )
            )
        analysis = H @ members
        if adaptive is not None:
            adaptive.record(y, forecast, analysis, configured, HEIGHTS)
        cross = (y - forecast.mean(1)) * (y - analysis.mean(1))
        ratio.append(np.sqrt(np.mean(cross / used)))
        z.append((members.mean(1) - truth) / members.std(1, ddof=1))
    return {"ratio": np.array(ratio), "z": np.array(z)}


def _trials(initial_inflation: float | None) -> dict:
    """12 toy runs; `None` keeps the configured error throughout."""
    runs = [
        _cycle(
            (
                None
                if initial_inflation is None
                else AdaptiveObservationError(initial_inflation=initial_inflation)
            ),
            seed,
        )
        for seed in range(12)
    ]
    return {key: np.stack([r[key] for r in runs]) for key in runs[0]}


@pytest.fixture(scope="module")  # type: ignore[misc]
def toy() -> dict:
    return {
        "fixed": _trials(None),
        # Window 0 with the configured (too small) error, or conservatively
        # inflated beyond the true error.
        "adaptive": _trials(1.0),
        "conservative": _trials(6.0),
    }


def test_adaptive_error_recovers_a_desroziers_ratio_of_one(toy: dict) -> None:
    # The configured error is 2-6x too small: the ratio says so, every window.
    assert toy["fixed"]["ratio"][:, 3:].mean() > 3.0
    # Adapted from window 1 on, so by windows 3-5 it is ~1 whatever window 0 used.
    for arm in ("adaptive", "conservative"):
        assert 0.85 < toy[arm]["ratio"][:, 3:].mean() < 1.15, arm


def test_adaptive_error_calibrates_the_posterior(toy: dict) -> None:
    def rms(z: np.ndarray) -> float:
        return float(np.sqrt(np.mean(z[:, 3:] ** 2)))

    # Fixed R: overconfident (true parameters many posterior stds away).
    assert rms(toy["fixed"]["z"]) > 2.0
    # A window 0 with the too-small error has already collapsed the ensemble,
    # which a parameter-only smoother never re-inflates: the right R later
    # comes too late. A conservative window 0 keeps the spread for it.
    assert rms(toy["adaptive"]["z"]) > 2.0
    assert 0.6 < rms(toy["conservative"]["z"]) < 1.5


def test_inflation_pools_shrinks_and_bounds() -> None:
    heights = np.array([2.0, 2.0, 20.0, 20.0])
    configured = np.ones(4)
    adaptive = AdaptiveObservationError(
        prior_weight=2.0, initial_inflation=2.0, max_inflation=5.0
    )
    # Window 0: the initial inflation.
    np.testing.assert_allclose(adaptive.variance(configured, heights), 4.0)
    # d_a d_f / R = 9 at z=2 (x3) and 100 at z=20 (x10, capped at 5).
    obs = np.zeros(4)
    forecast = np.array([[3.0], [3.0], [10.0], [10.0]])
    adaptive.record(obs, forecast, forecast, configured, heights)
    # Shrunk with 2 pseudo-observations of 4: (2*4 + 2*9) / 4 = 6.5.
    np.testing.assert_allclose(
        adaptive.variance(configured, heights), [6.5, 6.5, 25.0, 25.0]
    )
    assert adaptive.history[0]["window_estimate"] == pytest.approx([3.0, 10.0])
    assert adaptive.history[1]["groups"] == ["z=2", "z=20"]

    # memory=1 forgets window 0; calibration_windows=1 freezes after it.
    small = np.array([[1.0], [1.0], [1.0], [1.0]])
    for kwargs, expected in (
        ({"memory": 1}, 1.0),
        ({"calibration_windows": 1}, 9.0),
    ):
        a = AdaptiveObservationError(**kwargs)  # type: ignore[arg-type]
        a.record(obs, forecast, forecast, configured, heights)
        a.record(obs, small, small, configured, heights)
        assert a.variance(configured, heights)[0] == pytest.approx(expected)

    # Floors at the configured error: a negative sample cannot deflate it.
    a = AdaptiveObservationError()
    a.record(obs, forecast, -forecast, configured, heights)
    np.testing.assert_allclose(a.variance(configured, heights), 1.0)


def test_height_bands_group_and_must_cover_every_sensor() -> None:
    adaptive = AdaptiveObservationError(
        height_bands=[{"min_z": 0.0, "max_z": 10.0}, {"min_z": 10.0, "max_z": 40.0}]
    )
    ids, names = adaptive.groups(np.array([2.0, 8.0, 20.0]))
    assert ids.tolist() == [0, 0, 1] and names == ["z=[0,10)", "z=[10,40)"]
    with pytest.raises(ValueError, match="no adaptive height band"):
        adaptive.groups(np.array([50.0]))
    with pytest.raises(ValueError, match="initial_inflation"):
        AdaptiveObservationError(initial_inflation=0.5)
