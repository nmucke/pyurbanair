"""Re-implemented published urban-flow surrogates, trained and evaluated with
the neural_surrogates infrastructure for comparison with our steppers.

See docs/neural_surrogate_baselines.md.
"""

from neural_surrogate_baselines.datasets import RolloutTransitionDataset
from neural_surrogate_baselines.local_fno import LocalFNOStepper, LocalFNOTrainer
from neural_surrogate_baselines.ssrolling import RolloutTrainer, SSRollingUrbanNet

__all__ = [
    "LocalFNOStepper",
    "LocalFNOTrainer",
    "RolloutTrainer",
    "RolloutTransitionDataset",
    "SSRollingUrbanNet",
]
