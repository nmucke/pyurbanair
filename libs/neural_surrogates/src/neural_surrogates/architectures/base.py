"""The architecture abstraction (``docs/neural_surrogate_plan.md`` §1.1).

Every neural architecture is a JAX/Equinox module implementing a *field-space
stepping interface with an opaque carry*. The carry is a PyTree the
architecture owns; the framework never inspects it. This is the single
load-bearing cut that keeps rollout, training, and the forward model
architecture-agnostic:

- the UNet's carry is a ring buffer of the last ``K`` decoded *fields*;
- UPT's carry (added later) is a ring buffer of the last ``K`` *latents*.

Both implement the same two methods, so ``rollout`` (``rollout.py``), the
pushforward training loop (``training/loop.py``), and the forward model's
autoregression (``forward_model.py``) all drive any architecture identically.

Shape conventions (channels-first, matching ``utils.state_io``):

- ``C`` velocity/pressure channels (``u, v, w[, pres]``),
- ``Z, Y, X`` collocated grid axes (``docs/neural_surrogate_plan.md`` D3),
- ``K`` history frames, ``P`` conditioning params per frame,
- ``S`` static geometry channels (SDF [+ mask], D5).
"""

from __future__ import annotations

from typing import Any, TypeAlias

import equinox as eqx
from jaxtyping import Array, Float

# The carry is architecture-owned and opaque to the framework. We only require
# it to be a JAX PyTree so ``jax.lax.scan`` in ``rollout`` can thread it.
Carry: TypeAlias = Any


class SurrogateArchitecture(eqx.Module):
    """Contract every neural architecture implements.

    Field-space in/out; the architecture owns whatever internal state it
    carries between steps. Implementations MUST be deterministic and
    stateless between calls (no batch norm / dropout / mutable layer state)
    unless this contract is explicitly extended with mode/RNG/state plumbing
    (``docs/neural_surrogate_plan.md`` §1.2, P2).

    Implementations SHOULD special-case ``K == 1`` to skip all multi-frame
    machinery so the simplest model carries no temporal overhead (the Markov
    fast path; also the UPT↔reference parity reduction, §1.1).
    """

    def init_carry(
        self,
        hist_fields: Float[Array, "K C Z Y X"],
        hist_params: Float[Array, "K P"],
        hist_mask: Float[Array, "K"],
        static: Float[Array, "S Z Y X"],
    ) -> Carry:
        """Build the initial autoregressive carry from the K-frame history.

        Args:
            hist_fields: Last ``K`` history frames, normalized, left-padded
                if fewer than ``K`` real frames are available.
            hist_params: Per-frame conditioning, dense (already sin/cos
                encoded by the framework, §1.5).
            hist_mask: ``1`` for a real frame, ``0`` for a left-pad slot.
            static: Baked SDF + mask channels (D5), constant across a rollout.

        Returns:
            The architecture-owned initial carry.
        """
        raise NotImplementedError

    def step(
        self,
        carry: Carry,
        next_param: Float[Array, "P"],
        static: Float[Array, "S Z Y X"],
    ) -> tuple[Float[Array, "C Z Y X"], Carry]:
        """Advance one step.

        Args:
            carry: The current architecture carry.
            next_param: Boundary condition for ``t -> t+1`` (ready to embed,
                §1.5); the architecture does the embedding internally.
            static: Baked SDF + mask channels (D5).

        Returns:
            ``(field_prediction, new_carry)`` — the decoded next field
            (channels-first ``[C, Z, Y, X]``) and the carry to feed the next
            ``step``. The loop never has to know whether the carry holds a
            field (UNet) or a propagated latent (UPT).
        """
        raise NotImplementedError
