"""UPT latent approximator.

Vendored verbatim from upt-tutorial:
    https://github.com/BenediktAlkin/upt-tutorial
    upt/models/approximator.py  (main branch)

upt-tutorial and KappaModules are MIT-licensed (Benedikt Alkin); see
``_upt/LICENSE``. Pure kappamodules with an optional ``condition`` kwarg; no
changes versus upstream.
"""

from __future__ import annotations

from functools import partial

from kappamodules.layers import LinearProjection, Sequential
from kappamodules.transformer import DitBlock, PrenormBlock
from torch import nn


class Approximator(nn.Module):
    def __init__(
        self,
        input_dim,
        depth,
        num_heads,
        dim=None,
        cond_dim=None,
        init_weights="truncnormal002",
        attn_ctor=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        dim = dim or input_dim
        self.dim = dim
        self.depth = depth
        self.num_heads = num_heads
        self.cond_dim = cond_dim
        self.init_weights = init_weights

        # project
        self.input_proj = LinearProjection(
            input_dim, dim, init_weights=init_weights, optional=True
        )

        # blocks. ``attn_ctor`` (when given) selects the self-attention impl;
        # both block types accept it, so it is simply forwarded.
        attn_kwargs = {} if attn_ctor is None else {"attn_ctor": attn_ctor}
        if cond_dim is None:
            block_ctor = partial(PrenormBlock, **attn_kwargs)
        else:
            block_ctor = partial(DitBlock, cond_dim=cond_dim, **attn_kwargs)
        self.blocks = Sequential(
            *[
                block_ctor(
                    dim=dim,
                    num_heads=num_heads,
                    init_weights=init_weights,
                )
                for _ in range(depth)
            ],
        )

    def forward(self, x, condition=None):
        # check inputs
        assert x.ndim == 3, "expected shape (batch_size, num_latent_tokens, dim)"
        if condition is not None:
            assert condition.ndim == 2, "expected shape (batch_size, cond_dim)"

        # pass condition to DiT blocks
        cond_kwargs = {}
        if condition is not None:
            cond_kwargs["cond"] = condition

        # project to decoder dim
        x = self.input_proj(x)

        # apply blocks
        x = self.blocks(x, **cond_kwargs)

        return x
