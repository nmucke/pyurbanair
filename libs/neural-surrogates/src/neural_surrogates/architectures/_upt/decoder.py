"""UPT perceiver decoder (dense-output adaptation).

Vendored from upt-tutorial:
    https://github.com/BenediktAlkin/upt-tutorial
    upt/models/decoder_perceiver.py  (main branch)

upt-tutorial and KappaModules are MIT-licensed (Benedikt Alkin); see
``_upt/LICENSE``.

Changes versus upstream: the image-unbatch branch (and the ``math``/``einops``
imports it needed) is dropped, and the default path returns the DENSE
``(B, N, output_dim)`` tensor instead of the ``(b n) c`` flatten -- the UPT
wrapper scatters predictions back to the grid per sample. The transformer
blocks + perceiver cross-attention tail are unchanged.
"""

from __future__ import annotations

from functools import partial

from kappamodules.layers import ContinuousSincosEmbed, LinearProjection, Sequential
from kappamodules.transformer import (
    DitBlock,
    DitPerceiverBlock,
    PerceiverBlock,
    PrenormBlock,
)
from kappamodules.vit import VitBlock
from torch import nn


class DecoderPerceiver(nn.Module):
    def __init__(
        self,
        input_dim,
        output_dim,
        ndim,
        dim,
        depth,
        num_heads,
        perc_dim=None,
        perc_num_heads=None,
        cond_dim=None,
        init_weights="truncnormal002",
        attn_ctor=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        perc_dim = perc_dim or dim
        perc_num_heads = perc_num_heads or num_heads
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.ndim = ndim
        self.dim = dim
        self.depth = depth
        self.num_heads = num_heads
        self.perc_dim = perc_dim
        self.perc_num_heads = perc_num_heads
        self.cond_dim = cond_dim
        self.init_weights = init_weights

        # input projection
        self.input_proj = LinearProjection(
            input_dim, dim, init_weights=init_weights, optional=True
        )

        # blocks. The default self-attention block here is ``VitBlock``, which
        # has no ``attn_ctor`` hook; when a custom attention is requested fall
        # back to the equivalent prenorm block that does (``PrenormBlock``).
        if cond_dim is None:
            block_ctor = VitBlock if attn_ctor is None else partial(
                PrenormBlock, attn_ctor=attn_ctor
            )
        else:
            attn_kwargs = {} if attn_ctor is None else {"attn_ctor": attn_ctor}
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

        # prepare perceiver
        self.pos_embed = ContinuousSincosEmbed(
            dim=perc_dim,
            ndim=ndim,
        )
        if cond_dim is None:
            block_ctor = PerceiverBlock
        else:
            block_ctor = partial(DitPerceiverBlock, cond_dim=cond_dim)

        # decoder
        self.query_proj = nn.Sequential(
            LinearProjection(perc_dim, perc_dim, init_weights=init_weights),
            nn.GELU(),
            LinearProjection(perc_dim, perc_dim, init_weights=init_weights),
        )
        self.perc = block_ctor(
            dim=perc_dim, kv_dim=dim, num_heads=perc_num_heads, init_weights=init_weights
        )
        self.pred = nn.Sequential(
            nn.LayerNorm(perc_dim, eps=1e-6),
            LinearProjection(perc_dim, output_dim, init_weights=init_weights),
        )

    def forward(self, x, output_pos, condition=None):
        # check inputs
        assert x.ndim == 3, "expected shape (batch_size, num_latent_tokens, dim)"
        assert output_pos.ndim == 3, "expected shape (batch_size, num_outputs, ndim)"
        if condition is not None:
            assert condition.ndim == 2, "expected shape (batch_size, cond_dim)"

        # pass condition to DiT blocks
        cond_kwargs = {}
        if condition is not None:
            cond_kwargs["cond"] = condition

        # input projection
        x = self.input_proj(x)

        # apply blocks
        x = self.blocks(x, **cond_kwargs)

        # create query
        query = self.pos_embed(output_pos)
        query = self.query_proj(query)

        x = self.perc(q=query, kv=x, **cond_kwargs)
        x = self.pred(x)

        # DENSE output (B, num_outputs, output_dim): the wrapper scatters per sample.
        return x
