import jax.numpy as jnp
from data_assimilation.observation_operator import ObservationOperator

from pyurbanair.base_forward_model import BaseForwardModel


class BaseSmoothing:
    """Base class for smoothing."""

    def __init__(
        self,
        observation_operator: ObservationOperator,
        forward_model: BaseForwardModel,
    ) -> None:
        self.observation_operator = observation_operator
        self.forward_model = forward_model

    def smooth(self) -> jnp.ndarray:
        """Smooth the observations."""
        pass
