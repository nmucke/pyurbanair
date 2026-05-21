"""P0 interface/rollout conformance test (``docs/neural_surrogate_plan.md`` §11).

A tiny dummy ``SurrogateArchitecture`` exercises ``init_carry`` / ``step`` /
``rollout`` shapes and the K=1 fast path independently of any real network,
guarding the contract before UNet or UPT exist.
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from neural_surrogates.architectures.base import SurrogateArchitecture
from neural_surrogates.rollout import rollout, rollout_from_history


class _DummyArch(SurrogateArchitecture):
    """Carry = ring buffer of the last K fields; step nudges by next_param[0]."""

    n_channels: int = eqx.field(static=True)
    history_len: int = eqx.field(static=True)

    def init_carry(self, hist_fields, hist_params, hist_mask, static):
        # store the last K fields exactly as given (channels-first)
        return hist_fields

    def step(self, carry, next_param, static):
        # decode = last frame + scalar bump; append, drop oldest (ring buffer)
        last = carry[-1]
        field = last + next_param[0]
        new_carry = jnp.concatenate([carry[1:], field[None]], axis=0)
        return field, new_carry


def _shapes(k: int):
    c, z, y, x, s = 3, 4, 5, 6, 2
    arch = _DummyArch(n_channels=c, history_len=k)
    hist_fields = jnp.zeros((k, c, z, y, x))
    hist_params = jnp.zeros((k, 1))
    hist_mask = jnp.ones((k,))
    static = jnp.zeros((s, z, y, x))
    return arch, hist_fields, hist_params, hist_mask, static, (c, z, y, x)


def test_rollout_shapes_k3() -> None:
    arch, hf, hp, hm, static, (c, z, y, x) = _shapes(k=3)
    carry = arch.init_carry(hf, hp, hm, static)
    future_params = jnp.ones((4, 1))
    fields, final = rollout(arch, carry, future_params, static, n_steps=4)
    assert fields.shape == (4, c, z, y, x)
    # each step adds 1.0 (next_param[0]); 4 steps from zeros -> last == 4
    np.testing.assert_allclose(np.asarray(fields[-1]), 4.0)


def test_rollout_k1_fast_path() -> None:
    arch, hf, hp, hm, static, (c, z, y, x) = _shapes(k=1)
    fields = rollout_from_history(arch, hf, hp, hm, jnp.ones((2, 1)), static, n_steps=2)
    assert fields.shape == (2, c, z, y, x)


def test_rollout_is_vmappable_over_ensemble() -> None:
    """D2: batched inference is just vmap over the ensemble axis."""
    arch, hf, hp, hm, static, (c, z, y, x) = _shapes(k=2)
    n_members = 5
    batched_hist = jnp.broadcast_to(hf, (n_members,) + hf.shape)
    batched_params = jnp.arange(n_members, dtype=jnp.float32)[:, None, None] * jnp.ones((1, 3, 1))

    def run(hist, fp):
        return rollout_from_history(arch, hist, hp, hm, fp, static, n_steps=3)

    out = jax.vmap(run)(batched_hist, batched_params)
    assert out.shape == (n_members, 3, c, z, y, x)


def test_rollout_warmup_stop_gradient_runs() -> None:
    """Pushforward warm-up steps must run without error and stay finite."""
    arch, hf, hp, hm, static, _ = _shapes(k=2)
    carry = arch.init_carry(hf, hp, hm, static)
    fields, _ = rollout(arch, carry, jnp.ones((4, 1)), static, n_steps=4, warmup=2)
    assert jnp.all(jnp.isfinite(fields))
