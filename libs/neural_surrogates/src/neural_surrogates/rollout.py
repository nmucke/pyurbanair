"""Architecture-agnostic autoregressive rollout (``docs/neural_surrogate_plan.md`` §1.1).

A single ``rollout`` helper, built on ``jax.lax.scan`` over ``arch.step``, is
used by **both** the forward model's inference autoregression (§4) and the
training pushforward loop (§6.1). Pushforward feeds the model its own output
automatically, because ``step`` returns the carry it needs next (a predicted
field for the UNet, a propagated latent for UPT) — the loop never knows which.

The optional ``warmup`` argument supports the pushforward curriculum
(``docs/neural_surrogate_plan.md`` §6.1): the first ``warmup`` steps are run
under ``jax.lax.stop_gradient`` on the carry so gradients only flow through the
final ``n_steps - warmup`` steps. At inference ``warmup=0`` (default) and the
``stop_gradient`` is a no-op.
"""

from __future__ import annotations

from typing import Optional

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

from .architectures.base import Carry, SurrogateArchitecture


def rollout(
    arch: SurrogateArchitecture,
    carry: Carry,
    future_params: Float[Array, "H P"],
    static: Float[Array, "S Z Y X"],
    n_steps: int,
    *,
    warmup: int = 0,
) -> tuple[Float[Array, "H C Z Y X"], Carry]:
    """Unroll ``arch`` for ``n_steps`` autoregressive steps.

    Args:
        arch: Any ``SurrogateArchitecture``.
        carry: Initial carry from ``arch.init_carry`` (single-member; ``vmap``
            over the ensemble axis for batched inference, D2).
        future_params: Per-step boundary conditions, ``[H, P]`` with
            ``H >= n_steps`` (already framework-encoded, §1.5).
        static: Baked geometry channels (D5), constant across the rollout.
        n_steps: Number of steps to unroll.
        warmup: Number of leading steps to run with ``stop_gradient`` on the
            carry (pushforward curriculum, §6.1). ``0`` at inference.

    Returns:
        ``(fields, final_carry)`` where ``fields`` is ``[n_steps, C, Z, Y, X]``
        stacked in time order, and ``final_carry`` is the carry after the last
        step (so a rollout can be continued across windows).
    """
    if n_steps <= 0:
        raise ValueError(f"n_steps must be positive, got {n_steps}.")

    params_seq = future_params[:n_steps]
    # Per-step boolean: should this step's carry be detached before stepping?
    detach = jnp.arange(n_steps) < warmup

    def scan_step(
        carry: Carry, step_inputs: tuple[Float[Array, "P"], Array]
    ) -> tuple[Carry, Float[Array, "C Z Y X"]]:
        next_param, detach_flag = step_inputs
        carry = jax.lax.cond(
            detach_flag,
            lambda c: jax.lax.stop_gradient(c),
            lambda c: c,
            carry,
        )
        field, new_carry = arch.step(carry, next_param, static)
        return new_carry, field

    final_carry, fields = jax.lax.scan(scan_step, carry, (params_seq, detach))
    return fields, final_carry


def rollout_from_history(
    arch: SurrogateArchitecture,
    hist_fields: Float[Array, "K C Z Y X"],
    hist_params: Float[Array, "K P"],
    hist_mask: Float[Array, "K"],
    future_params: Float[Array, "H P"],
    static: Float[Array, "S Z Y X"],
    n_steps: int,
    *,
    warmup: int = 0,
    final_carry: bool = False,
) -> Float[Array, "H C Z Y X"] | tuple[Float[Array, "H C Z Y X"], Carry]:
    """Convenience wrapper: ``init_carry`` then ``rollout``.

    Used by the training loop and the forward model so neither has to spell out
    the two-step dance. Returns just the fields by default; pass
    ``final_carry=True`` to also get the carry for cross-window continuation.
    """
    carry = arch.init_carry(hist_fields, hist_params, hist_mask, static)
    fields, carry = rollout(
        arch, carry, future_params, static, n_steps, warmup=warmup
    )
    if final_carry:
        return fields, carry
    return fields


__all__ = ["rollout", "rollout_from_history"]
