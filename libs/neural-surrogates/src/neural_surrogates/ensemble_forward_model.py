"""Ensemble wrapper around :class:`NeuralSurrogateForwardModel`.

Each member shares the (stateless, read-only) trained network but owns an
isolated clone of the spin-up backend so per-member cold starts run in
their own experiment directories.
"""

from __future__ import annotations

import pathlib
from typing import Optional

from pyurbanair.base_ensemble_forward_model import BaseEnsembleForwardModel

from .forward_model import NeuralSurrogateForwardModel


class NeuralSurrogateEnsembleForwardModel(BaseEnsembleForwardModel):
    """Ensemble of neural-surrogate forward models."""

    def __init__(
        self,
        forward_model: NeuralSurrogateForwardModel,
        ensemble_size: int = 10,
        temp_dir: Optional[pathlib.Path] = None,
        results_dir: Optional[pathlib.Path] = None,
        num_parallel_processes: int = 1,
        num_cpus_per_process: int = 1,
    ) -> None:
        super().__init__(
            forward_model=forward_model,
            ensemble_size=ensemble_size,
            results_dir=results_dir,
            num_parallel_processes=num_parallel_processes,
            num_cpus_per_process=num_cpus_per_process,
            temp_dir=temp_dir,
        )

    def _create_new_forward_model(  # type: ignore[override]
        self,
        forward_model: NeuralSurrogateForwardModel,
        experiment_base_dir: pathlib.Path,
        experiment_name: str,
    ) -> NeuralSurrogateForwardModel:
        return forward_model.clone_for_member(experiment_base_dir, experiment_name)
