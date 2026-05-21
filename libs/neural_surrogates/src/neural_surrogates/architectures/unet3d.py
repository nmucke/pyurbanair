"""First architecture: a config-driven 3D convolutional UNet (§1.2).

Implements ``SurrogateArchitecture`` with a standard encoder/decoder UNet over
the collocated 3D grid. Its job is to exercise and prove the whole framework
(data -> train -> checkpoint -> forward model -> ESMDA) end-to-end cheaply; it
is intentionally *not* claimed optimal.

Design choices (``docs/neural_surrogate_plan.md`` §1.2):

- **History** by channel-stacking: the carry holds the last ``K`` decoded
  *fields*; ``step`` concatenates ``[K*C fields, S static]`` as the conv input.
  At ``K = 1`` this is the plain Markov ``(field_t, params) -> field_{t+1}``
  with no temporal overhead (the K=1 fast path).
- **Conditioning** via FiLM: ``next_param`` is embedded by a shared MLP
  (``training/conditioning.py``) and injected as per-channel scale/shift at
  every residual block.
- **Output**: predicts the next field, optionally as a residual ``Δfield``
  added to the most recent input frame (recommended for stability).
- **Grid divisibility**: pooling needs each spatial dim divisible by
  ``2**(num_levels-1)``; the model pads (edge) on entry and crops on exit.
"""

from __future__ import annotations

import math
from typing import Any, Callable

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

from ..training.conditioning import FiLM, ParamEmbedding, get_activation
from .base import SurrogateArchitecture

# The carry for the UNet is simply the ring buffer of the last K fields.
UNetCarry = Float[Array, "K C Z Y X"]


def _group_norm(channels: int, groups: int) -> eqx.nn.GroupNorm:
    # gcd guarantees groups divides channels (GroupNorm requires it); falls
    # back gracefully for small/odd channel counts in tiny test models.
    groups_eff = math.gcd(channels, groups) or 1
    return eqx.nn.GroupNorm(groups=groups_eff, channels=channels)


class _ResBlock3d(eqx.Module):
    """Two conv layers + norm + FiLM + activation, with a residual skip."""

    conv1: eqx.nn.Conv3d
    conv2: eqx.nn.Conv3d
    norm1: eqx.nn.GroupNorm | None
    norm2: eqx.nn.GroupNorm | None
    film1: FiLM
    film2: FiLM
    skip: eqx.nn.Conv3d | None
    activation: Callable = eqx.field(static=True)

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        embed_dim: int,
        *,
        norm: str,
        groups: int,
        activation: str,
        key: jax.Array,
    ) -> None:
        k1, k2, k3, k4, k5 = jax.random.split(key, 5)
        self.conv1 = eqx.nn.Conv3d(in_channels, out_channels, 3, padding=1, key=k1)
        self.conv2 = eqx.nn.Conv3d(out_channels, out_channels, 3, padding=1, key=k2)
        self.norm1 = _group_norm(out_channels, groups) if norm == "group" else None
        self.norm2 = _group_norm(out_channels, groups) if norm == "group" else None
        self.film1 = FiLM(embed_dim, out_channels, key=k3)
        self.film2 = FiLM(embed_dim, out_channels, key=k4)
        self.skip = (
            eqx.nn.Conv3d(in_channels, out_channels, 1, key=k5)
            if in_channels != out_channels
            else None
        )
        self.activation = get_activation(activation)

    def __call__(
        self, x: Float[Array, "C Z Y X"], emb: Float[Array, "E"]
    ) -> Float[Array, "C Z Y X"]:
        h = self.conv1(x)
        if self.norm1 is not None:
            h = self.norm1(h)
        h = self.activation(self.film1(h, emb))
        h = self.conv2(h)
        if self.norm2 is not None:
            h = self.norm2(h)
        h = self.activation(self.film2(h, emb))
        res = x if self.skip is None else self.skip(x)
        return h + res


class UNet3D(SurrogateArchitecture):
    """3D conv UNet stepping in field space (the first ``SurrogateArchitecture``)."""

    stem: eqx.nn.Conv3d
    head: eqx.nn.Conv3d
    encoder: list[list[_ResBlock3d]]
    downs: list[eqx.nn.Conv3d]
    bottleneck: _ResBlock3d
    ups: list[eqx.nn.ConvTranspose3d]
    decoder: list[list[_ResBlock3d]]
    param_embedding: ParamEmbedding

    # static config (does not participate in gradient / PyTree leaves)
    n_state_channels: int = eqx.field(static=True)
    history_len: int = eqx.field(static=True)
    n_static_channels: int = eqx.field(static=True)
    num_levels: int = eqx.field(static=True)
    residual: bool = eqx.field(static=True)
    divisor: int = eqx.field(static=True)

    def __init__(self, config: dict[str, Any], *, key: jax.Array) -> None:
        c = self.n_state_channels = int(config["in_state_channels"])
        k = self.history_len = int(config.get("history_len", 1))
        s = self.n_static_channels = int(config.get("static_channels", 0))
        param_dim = int(config["param_dim"])

        base = int(config.get("base_channels", 16))
        mults = list(config.get("channel_multipliers", [1, 2, 4]))
        self.num_levels = len(mults)
        if self.num_levels < 1:
            raise ValueError("channel_multipliers must be non-empty.")
        n_blocks = int(config.get("num_res_blocks_per_level", 1))
        norm = str(config.get("norm", "group"))
        groups = int(config.get("groups", 8))
        activation = str(config.get("activation", "silu"))
        embed_dim = int(config.get("embed_dim", 64))
        self.residual = bool(config.get("residual", True))
        if config.get("attention_at_levels"):
            raise NotImplementedError(
                "attention_at_levels is reserved but not implemented yet."
            )

        widths = [base * m for m in mults]
        self.divisor = 2 ** (self.num_levels - 1)

        keys = iter(jax.random.split(key, 4 + 8 * (self.num_levels + 1) * (n_blocks + 1)))

        def nk() -> jax.Array:
            return next(keys)

        in_channels = k * c + s
        self.stem = eqx.nn.Conv3d(in_channels, widths[0], 3, padding=1, key=nk())
        self.param_embedding = ParamEmbedding(
            param_dim, embed_dim, activation=activation, key=nk()
        )

        def make_blocks(in_ch: int, out_ch: int) -> list[_ResBlock3d]:
            blocks = []
            ch = in_ch
            for _ in range(n_blocks):
                blocks.append(
                    _ResBlock3d(
                        ch, out_ch, embed_dim, norm=norm, groups=groups,
                        activation=activation, key=nk(),
                    )
                )
                ch = out_ch
            return blocks

        # Encoder: blocks at each level, downsample between levels.
        self.encoder = [make_blocks(widths[i], widths[i]) for i in range(self.num_levels)]
        self.downs = [
            eqx.nn.Conv3d(widths[i], widths[i + 1], 3, stride=2, padding=1, key=nk())
            for i in range(self.num_levels - 1)
        ]
        self.bottleneck = _ResBlock3d(
            widths[-1], widths[-1], embed_dim, norm=norm, groups=groups,
            activation=activation, key=nk(),
        )
        # Decoder: upsample then concat skip then blocks.
        self.ups = [
            eqx.nn.ConvTranspose3d(widths[i + 1], widths[i], 2, stride=2, key=nk())
            for i in reversed(range(self.num_levels - 1))
        ]
        self.decoder = [
            make_blocks(2 * widths[i], widths[i])
            for i in reversed(range(self.num_levels - 1))
        ]
        self.head = eqx.nn.Conv3d(widths[0], c, 1, key=nk())

    # ----- SurrogateArchitecture interface -------------------------------
    def init_carry(
        self,
        hist_fields: Float[Array, "K C Z Y X"],
        hist_params: Float[Array, "K P"],
        hist_mask: Float[Array, "K"],
        static: Float[Array, "S Z Y X"],
    ) -> UNetCarry:
        # The UNet's carry is the K-frame field ring buffer. Honor hist_mask
        # (D4): left-pad slots (mask 0) carry artificial zeros, so replace them
        # with the first real frame instead of conditioning on zero-flow. With
        # all-real history (or K=1) this is a no-op.
        if self.history_len == 1:
            return hist_fields
        first_real = jnp.argmax(hist_mask)  # first index where mask == 1
        is_real = hist_mask.reshape(-1, 1, 1, 1, 1) > 0
        return jnp.where(is_real, hist_fields, hist_fields[first_real][None])

    def step(
        self,
        carry: UNetCarry,
        next_param: Float[Array, "P"],
        static: Float[Array, "S Z Y X"],
    ) -> tuple[Float[Array, "C Z Y X"], UNetCarry]:
        k, c = self.history_len, self.n_state_channels
        stacked = carry.reshape(k * c, *carry.shape[2:])  # [K*C, Z, Y, X]
        if self.n_static_channels:
            stacked = jnp.concatenate([stacked, static], axis=0)

        emb = self.param_embedding(next_param)
        prediction = self._unet(stacked, emb)
        if self.residual:
            prediction = carry[-1] + prediction

        # ring buffer: drop oldest, append the new field
        new_carry = jnp.concatenate([carry[1:], prediction[None]], axis=0)
        return prediction, new_carry

    # ----- internals ------------------------------------------------------
    def _pad(self, x: Float[Array, "C Z Y X"]) -> tuple[Float[Array, "C Z Y X"], tuple[int, int, int]]:
        z, y, xx = x.shape[1:]
        orig = (z, y, xx)
        pads = [(0, 0)]
        for dim in (z, y, xx):
            target = math.ceil(dim / self.divisor) * self.divisor
            pads.append((0, target - dim))
        return jnp.pad(x, pads, mode="edge"), orig

    def _crop(
        self, x: Float[Array, "C Z Y X"], orig: tuple[int, int, int]
    ) -> Float[Array, "C Z Y X"]:
        z, y, xx = orig
        return x[:, :z, :y, :xx]

    def _unet(
        self, x: Float[Array, "Cin Z Y X"], emb: Float[Array, "E"]
    ) -> Float[Array, "C Z Y X"]:
        x, orig = self._pad(x)
        h = self.stem(x)

        skips: list[Float[Array, "C Z Y X"]] = []
        for level in range(self.num_levels):
            for block in self.encoder[level]:
                h = block(h, emb)
            if level < self.num_levels - 1:
                skips.append(h)
                h = self.downs[level](h)

        h = self.bottleneck(h, emb)

        for i in range(self.num_levels - 1):
            h = self.ups[i](h)
            skip = skips.pop()
            h = jnp.concatenate([h, skip], axis=0)
            for block in self.decoder[i]:
                h = block(h, emb)

        h = self.head(h)
        return self._crop(h, orig)
