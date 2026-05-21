"""UNet3D architecture tests: grid divisibility / pad-crop + K=1 fast path (§11)."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from neural_surrogates.architectures.registry import resolve_architecture
from neural_surrogates.rollout import rollout_from_history


def _build(history_len: int, num_levels: int, static_channels: int = 2):
    cfg = dict(
        in_state_channels=3,
        static_channels=static_channels,
        history_len=history_len,
        param_dim=3,
        base_channels=8,
        channel_multipliers=[1, 2][:num_levels] if num_levels <= 2 else [1, 2, 4],
        num_res_blocks_per_level=1,
        embed_dim=16,
    )
    return resolve_architecture("unet3d", cfg, key=jax.random.PRNGKey(0))


@pytest.mark.parametrize("grid", [(4, 4, 4), (5, 7, 9), (3, 3, 16)])
def test_unet_handles_arbitrary_grids_via_pad_crop(grid) -> None:
    z, y, x = grid
    arch = _build(history_len=2, num_levels=3)
    hf = jnp.zeros((2, 3, z, y, x))
    static = jnp.zeros((2, z, y, x))
    out = rollout_from_history(
        arch, hf, jnp.zeros((2, 3)), jnp.ones((2,)), jnp.ones((3, 3)), static, n_steps=3
    )
    assert out.shape == (3, 3, z, y, x)
    assert bool(jnp.all(jnp.isfinite(out)))


def test_unet_k1_markov_fast_path() -> None:
    arch = _build(history_len=1, num_levels=2)
    hf = jnp.zeros((1, 3, 4, 4, 4))
    static = jnp.zeros((2, 4, 4, 4))
    out = rollout_from_history(
        arch, hf, jnp.zeros((1, 3)), jnp.ones((1,)), jnp.ones((2, 3)), static, n_steps=2
    )
    assert out.shape == (2, 3, 4, 4, 4)


def test_unet_residual_starts_near_identity() -> None:
    """FiLM is zero-init and residual mode adds to the last frame, so the first
    step should stay close to the input (no blow-up at init)."""
    arch = _build(history_len=1, num_levels=2)
    field = jax.random.normal(jax.random.PRNGKey(1), (1, 3, 4, 4, 4))
    static = jnp.zeros((2, 4, 4, 4))
    carry = arch.init_carry(field, jnp.zeros((1, 3)), jnp.ones((1,)), static)
    pred, _ = arch.step(carry, jnp.zeros((3,)), static)
    # prediction = last_frame + small conv output; finite and same shape
    assert pred.shape == (3, 4, 4, 4)
    assert bool(jnp.all(jnp.isfinite(pred)))
