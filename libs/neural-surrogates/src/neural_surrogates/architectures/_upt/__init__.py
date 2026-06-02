"""Vendored UPT building blocks (pure-torch).

These modules are derived from upt-tutorial and KappaModules (MIT, Benedikt
Alkin); see ``LICENSE`` in this directory. They are private helpers for the
public :class:`neural_surrogates.architectures.upt.UPT` wrapper.
"""

from neural_surrogates.architectures._upt.approximator import Approximator
from neural_surrogates.architectures._upt.decoder import DecoderPerceiver
from neural_surrogates.architectures._upt.encoder import EncoderSupernodes
from neural_surrogates.architectures._upt.supernode_pooling import SupernodePooling

__all__ = [
    "Approximator",
    "DecoderPerceiver",
    "EncoderSupernodes",
    "SupernodePooling",
]
