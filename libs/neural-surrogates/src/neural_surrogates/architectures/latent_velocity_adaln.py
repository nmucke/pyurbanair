"""adaLN-Zero velocity network for the latent flow-matching generator (plan 07).

``TadpoleLatentGenerator(velocity_conditioning="adaln")`` builds
:class:`AdaLNVelocityNet` instead of a
:class:`~neural_surrogates.architectures.tadpole_stepper.ParamConditionedSubnetwork`.
Where the latter applies the conditioning vector (z-scored parameter history +
sinusoidal flow time ``tau``) once, as a FiLM on the input tokens, and leaves
every transformer block unconditioned, this is the DiT adaLN-Zero design: the
conditioning vector is embedded once, ``c = MLP(cond)``, and **every** block
(and the final norm) derives its own shift / scale / gate from ``c``. A flow
velocity must behave very differently near ``tau = 0`` (pure noise) and
``tau = 1`` (a developed field), and per-block modulation lets each layer act
on that.

Geometry is spatially varying, so it cannot go through the (global) adaLN
path: a ``Linear(G, h)`` of ``geom_cond`` is added to every token after the
input projection, which is the same as projecting ``concat(z, geom_cond)``
(channel concatenation, the usual way spatial conditions enter a diffusion
backbone). It is not zero-initialised, so geometry shapes the hidden tokens
from the start; adaLN-Zero already makes the whole network output ``0`` at
init.

Blocks reuse the vendored Tadpole pieces (``AttentionBlock`` with naive
attention, the gated ``LlamaMLP``, ``get_3d_sincos_pos_embed``); the vendored
code is not modified. Norms are affine-free ``LayerNorm`` (the modulation
supplies the affine part), as in DiT. No cls token: the conditioning reaches
every block directly. At init every modulation output layer and the output
projection are zero, so each block is the identity and the output is exactly
``0``; on the first step only the output projection receives gradients, and
the modulation layers from the second step on.
"""

from __future__ import annotations

import torch
import torch.utils.checkpoint as checkpoint
from einops import rearrange
from neural_surrogates.architectures._tadpole.architecture.downstream.llm import (
    AttentionBlock,
    LlamaMLP,
    _pos_embed_on,
    get_3d_sincos_pos_embed,
)
from torch import nn


def _modulate(
    x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor
) -> torch.Tensor:
    """``x * (1 + scale) + shift`` with per-sample ``(B, h)`` shift / scale."""
    return x * (1.0 + scale[:, None]) + shift[:, None]


def _zero_linear(in_dim: int, out_dim: int) -> nn.Linear:
    layer = nn.Linear(in_dim, out_dim)
    nn.init.zeros_(layer.weight)
    nn.init.zeros_(layer.bias)
    return layer


class _AdaLNBlock(nn.Module):
    """Pre-norm transformer block with adaLN-Zero modulation from ``c``."""

    def __init__(self, hidden: int, num_heads: int, mlp_ratio: int, cond: int):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden, elementwise_affine=False, eps=1e-6)
        self.attn = AttentionBlock(hidden, num_heads=num_heads)
        self.norm2 = nn.LayerNorm(hidden, elementwise_affine=False, eps=1e-6)
        self.mlp = LlamaMLP(hidden, hidden * mlp_ratio)
        # shift/scale/gate for attention and MLP; zero -> identity block.
        self.modulation = nn.Sequential(nn.SiLU(), _zero_linear(cond, 6 * hidden))

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        mods = self.modulation(c).chunk(6, dim=-1)
        shift_a, scale_a, gate_a, shift_m, scale_m, gate_m = mods
        x = x + gate_a[:, None] * self.attn(_modulate(self.norm1(x), shift_a, scale_a))
        x = x + gate_m[:, None] * self.mlp(_modulate(self.norm2(x), shift_m, scale_m))
        return x


class AdaLNVelocityNet(nn.Module):
    """Transformer over latent tokens with per-block adaLN-Zero conditioning.

    Parameters
    ----------
    in_dim:
        Token width ``D`` in and out.
    cond_dim:
        Width of the conditioning vector (``Hp*P + time_embed_dim``).
    geom_cond_dim:
        Channels ``G`` of the per-token geometry conditioning.
    n_layers, num_heads, hidden_size, mlp_ratio:
        Transformer sizing (``hidden_size`` a multiple of ``num_heads``).
    film_hidden:
        Width of the conditioning embedding ``c`` every block modulates from.
    use_checkpoint:
        Recompute each block during backward to save activation memory.
    """

    def __init__(
        self,
        in_dim: int,
        cond_dim: int,
        geom_cond_dim: int,
        n_layers: int = 4,
        num_heads: int = 8,
        hidden_size: int = 768,
        mlp_ratio: int = 4,
        film_hidden: int = 128,
        use_checkpoint: bool = False,
    ) -> None:
        super().__init__()
        if hidden_size % num_heads:
            raise ValueError(
                f"hidden_size={hidden_size} must be a multiple of num_heads={num_heads}"
            )
        self.hidden_size = int(hidden_size)
        self.use_checkpoint = bool(use_checkpoint)
        self.input_proj = nn.Linear(in_dim, hidden_size)
        self.geom_proj = nn.Linear(geom_cond_dim, hidden_size)
        self.cond_mlp = nn.Sequential(
            nn.Linear(cond_dim, film_hidden),
            nn.SiLU(),
            nn.Linear(film_hidden, film_hidden),
        )
        self.blocks = nn.ModuleList(
            _AdaLNBlock(hidden_size, num_heads, mlp_ratio, film_hidden)
            for _ in range(n_layers)
        )
        self.norm_out = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.out_modulation = nn.Sequential(
            nn.SiLU(), _zero_linear(film_hidden, 2 * hidden_size)
        )
        self.out_proj = _zero_linear(hidden_size, in_dim)

    def forward(
        self, x: torch.Tensor, cond: torch.Tensor, geom_cond: torch.Tensor
    ) -> torch.Tensor:
        # x: (B, D, Zl, Yl, Xl); cond: (B, cond_dim); geom_cond: (B, G, Zl, Yl, Xl)
        _, _, nz, ny, nx = x.shape
        pos_embed = get_3d_sincos_pos_embed(self.hidden_size, (nz, ny, nx))
        tokens = self.input_proj(rearrange(x, "b c z y x -> b (z y x) c"))
        tokens = tokens + self.geom_proj(
            rearrange(geom_cond, "b c z y x -> b (z y x) c")
        )
        tokens = tokens + _pos_embed_on(pos_embed, tokens)
        c = self.cond_mlp(cond)
        for block in self.blocks:
            if self.use_checkpoint and torch.is_grad_enabled():
                tokens = checkpoint.checkpoint(block, tokens, c, use_reentrant=False)
            else:
                tokens = block(tokens, c)
        shift, scale = self.out_modulation(c).chunk(2, dim=-1)
        out = self.out_proj(_modulate(self.norm_out(tokens), shift, scale))
        return rearrange(out, "b (z y x) c -> b c z y x", z=nz, y=ny, x=nx)
