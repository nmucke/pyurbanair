"""Architecture-agnostic pushforward training loop (§6.1).

Written **once** for any ``SurrogateArchitecture``: it only ever calls
``rollout_from_history`` and the field-space loss — it never knows whether the
carry holds fields (UNet) or latents (UPT).

Key behaviors:

- **Field-space pushforward**: ``H`` autoregressive steps feed the model its
  own output (via ``rollout``); the loss is masked per-variable MSE of decoded
  predictions vs the next ``H`` *true* frames. ``K=1, H=1`` is one-step.
- **Curriculum**: ``H`` grows over training; ``warmup`` leading steps run under
  ``stop_gradient`` so error doesn't blow up (handled in ``rollout``).
- **Off-manifold IC injection**: history frames are optionally perturbed before
  ``init_carry``, feeding the GATE B2 (analysis-OOD) bar (§6.1/§14).
- **Mask-aware loss**: solid cells excluded from the MSE (D5).
"""

from __future__ import annotations

from typing import Callable

import equinox as eqx
import jax
import jax.numpy as jnp
import optax
from jaxtyping import Array, Float

from ..architectures.base import SurrogateArchitecture
from ..rollout import rollout_from_history

Batch = dict[str, Array]


def masked_mse(
    pred: Float[Array, "B H C Z Y X"],
    target: Float[Array, "B H C Z Y X"],
    fluid: Float[Array, "Z Y X"],
    eps: float = 1e-8,
) -> Float[Array, ""]:
    """Mean squared error over **fluid** cells only (solid cells masked, D5)."""
    w = fluid.reshape(1, 1, 1, *fluid.shape)
    sq_err = jnp.square(pred - target) * w
    # normalize by number of fluid elements actually summed
    denom = w.sum() * pred.shape[0] * pred.shape[1] * pred.shape[2] + eps
    return sq_err.sum() / denom


def compute_loss(
    arch: SurrogateArchitecture,
    batch: Batch,
    static: Float[Array, "S Z Y X"],
    fluid: Float[Array, "Z Y X"],
    n_steps: int,
    warmup: int,
) -> Float[Array, ""]:
    """Pushforward loss for a batch (vmapped over the batch axis)."""

    def single(hist, hp, hm, fp):
        return rollout_from_history(
            arch, hist, hp, hm, fp, static, n_steps, warmup=warmup
        )

    preds = jax.vmap(single)(
        batch["hist_fields"],
        batch["hist_params"],
        batch["hist_mask"],
        batch["future_params"],
    )
    target = batch["target_fields"][:, :n_steps]
    return masked_mse(preds, target, fluid)


def inject_off_manifold(
    batch: Batch,
    key: jax.Array,
    noise_std: float,
) -> Batch:
    """Perturb real history frames with Gaussian noise (analysis-OOD, §6.1).

    Padded (mask-0) frames are left untouched. Cheap, architecture-agnostic
    stand-in for ensemble-mix / synthetic EnKF increments; richer schemes plug
    in here without touching the architecture.
    """
    if noise_std <= 0.0:
        return batch
    hist = batch["hist_fields"]
    noise = noise_std * jax.random.normal(key, hist.shape)
    frame_mask = batch["hist_mask"][..., None, None, None, None]
    return {**batch, "hist_fields": hist + noise * frame_mask}


def make_train_step(
    optimizer: optax.GradientTransformation,
) -> Callable:
    """Build a jitted ``train_step(arch, opt_state, batch, static, fluid, ...)``.

    ``n_steps`` and ``warmup`` are static (Python ints): changing them on a
    curriculum step triggers one recompile, which is intended.
    """

    @eqx.filter_jit
    def train_step(
        arch: SurrogateArchitecture,
        opt_state: optax.OptState,
        batch: Batch,
        static: Float[Array, "S Z Y X"],
        fluid: Float[Array, "Z Y X"],
        n_steps: int,
        warmup: int,
    ):
        loss, grads = eqx.filter_value_and_grad(compute_loss)(
            arch, batch, static, fluid, n_steps, warmup
        )
        params = eqx.filter(arch, eqx.is_inexact_array)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        arch = eqx.apply_updates(arch, updates)
        return arch, opt_state, loss

    return train_step


@eqx.filter_jit
def eval_loss(
    arch: SurrogateArchitecture,
    batch: Batch,
    static: Float[Array, "S Z Y X"],
    fluid: Float[Array, "Z Y X"],
    n_steps: int,
) -> Float[Array, ""]:
    """Validation loss (no gradient, no warm-up)."""
    return compute_loss(arch, batch, static, fluid, n_steps, warmup=0)


@eqx.filter_jit
def per_horizon_sq_error(
    arch: SurrogateArchitecture,
    batch: Batch,
    static: Float[Array, "S Z Y X"],
    fluid: Float[Array, "Z Y X"],
    horizon: int,
) -> Float[Array, "H"]:
    """Per-step masked MSE across a rollout — the metric that matters (§6.5).

    Returns one error per horizon step, so callers can log/plot
    rollout-error-vs-horizon (the GATE B1 bar, §13).
    """

    def single(hist, hp, hm, fp):
        return rollout_from_history(arch, hist, hp, hm, fp, static, horizon)

    preds = jax.vmap(single)(
        batch["hist_fields"],
        batch["hist_params"],
        batch["hist_mask"],
        batch["future_params"],
    )
    target = batch["target_fields"][:, :horizon]
    w = fluid.reshape(1, 1, 1, *fluid.shape)
    sq = jnp.square(preds - target) * w  # [B, H, C, Z, Y, X]
    denom = w.sum() * preds.shape[0] * preds.shape[2] + 1e-8
    return sq.sum(axis=(0, 2, 3, 4, 5)) / denom  # [H]
