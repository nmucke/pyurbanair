"""Name → architecture resolution (``docs/neural_surrogate_plan.md`` §1.1, §7).

The architecture choice fixes the meaning of the trained weights, so it is
baked into the checkpoint manifest (``architecture.json``). The forward model
and training loop resolve the name through this registry; the loader asserts
the requested architecture matches the checkpoint.

UNet is the first (and currently only) architecture; ``upt`` is registered
later under ``architectures/upt/`` with **no framework changes**.
"""

from __future__ import annotations

from typing import Any, Callable

import jax

from .base import SurrogateArchitecture


def _build_unet3d(config: dict[str, Any], *, key: jax.Array) -> SurrogateArchitecture:
    # Imported lazily so registering the name does not import the UNet module
    # (and its heavy deps) until something actually builds one.
    from .unet3d import UNet3D

    return UNet3D(config, key=key)


# name -> factory(config, *, key) -> SurrogateArchitecture
_REGISTRY: dict[str, Callable[..., SurrogateArchitecture]] = {
    "unet3d": _build_unet3d,
}


def available_architectures() -> list[str]:
    """Return the registered architecture names."""
    return sorted(_REGISTRY)


def resolve_architecture(
    name: str,
    config: dict[str, Any],
    *,
    key: jax.Array,
) -> SurrogateArchitecture:
    """Instantiate the named architecture from its config dict.

    Args:
        name: Registered architecture name (e.g. ``"unet3d"``).
        config: Architecture hyperparameters (snapshotted in the checkpoint).
        key: PRNG key for weight initialization.

    Raises:
        KeyError: if ``name`` is not registered.
    """
    if name not in _REGISTRY:
        raise KeyError(
            f"Unknown architecture {name!r}. Available: {available_architectures()}."
        )
    return _REGISTRY[name](config, key=key)
