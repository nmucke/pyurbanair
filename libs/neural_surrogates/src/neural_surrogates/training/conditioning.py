"""Shared parameter-embedding + FiLM helpers (``docs/neural_surrogate_plan.md`` §6.2).

The framework hands each architecture **dense, already sin/cos-encoded** params
(§1.5). The *embedding* lives here and is shared, but is invoked **inside** the
architecture so each model fuses conditioning natively (FiLM levels for the
UNet; tokens for UPT). For the UNet, ``ParamEmbedding`` lifts the ``P`` encoded
scalars to an embedding vector, and a per-level ``FiLM`` produces per-channel
scale/shift from that embedding.
"""

from __future__ import annotations

from typing import Callable

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

ACTIVATIONS: dict[str, Callable[[Array], Array]] = {
    "gelu": jax.nn.gelu,
    "relu": jax.nn.relu,
    "silu": jax.nn.silu,
    "tanh": jnp.tanh,
}


def get_activation(name: str) -> Callable[[Array], Array]:
    if name not in ACTIVATIONS:
        raise ValueError(f"Unknown activation {name!r}; have {sorted(ACTIVATIONS)}.")
    return ACTIVATIONS[name]


class ParamEmbedding(eqx.Module):
    """MLP lifting the ``P`` encoded conditioning scalars to an embedding."""

    layer_in: eqx.nn.Linear
    layer_out: eqx.nn.Linear
    activation: Callable = eqx.field(static=True)

    def __init__(
        self,
        in_dim: int,
        embed_dim: int,
        *,
        activation: str = "silu",
        key: jax.Array,
    ) -> None:
        k1, k2 = jax.random.split(key)
        self.layer_in = eqx.nn.Linear(in_dim, embed_dim, key=k1)
        self.layer_out = eqx.nn.Linear(embed_dim, embed_dim, key=k2)
        self.activation = get_activation(activation)

    def __call__(self, params: Float[Array, "P"]) -> Float[Array, "E"]:
        h = self.activation(self.layer_in(params))
        return self.layer_out(h)


class FiLM(eqx.Module):
    """Feature-wise linear modulation: per-channel ``(1 + scale)`` and ``shift``.

    Produces ``2 * channels`` values from the conditioning embedding and applies
    them broadcast over the 3D spatial axes of a ``[C, Z, Y, X]`` feature map.
    """

    proj: eqx.nn.Linear
    channels: int = eqx.field(static=True)

    def __init__(self, embed_dim: int, channels: int, *, key: jax.Array) -> None:
        # zero-init so FiLM starts as identity (stable early training).
        self.proj = eqx.nn.Linear(embed_dim, 2 * channels, key=key)
        self.proj = eqx.tree_at(
            lambda m: (m.weight, m.bias),
            self.proj,
            (jnp.zeros_like(self.proj.weight), jnp.zeros_like(self.proj.bias)),
        )
        self.channels = channels

    def __call__(
        self, features: Float[Array, "C Z Y X"], embedding: Float[Array, "E"]
    ) -> Float[Array, "C Z Y X"]:
        params = self.proj(embedding)
        scale = params[: self.channels].reshape(self.channels, 1, 1, 1)
        shift = params[self.channels :].reshape(self.channels, 1, 1, 1)
        return features * (1.0 + scale) + shift
