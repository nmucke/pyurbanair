"""UPT supernode encoder (dense-layout adaptation).

Vendored from upt-tutorial:
    https://github.com/BenediktAlkin/upt-tutorial
    upt/models/encoder_supernodes.py  (main branch)

upt-tutorial and KappaModules are MIT-licensed (Benedikt Alkin); see
``_upt/LICENSE``.

The only change versus upstream is ``forward``: it consumes the dense
``(B, N, feat)`` features + precomputed ``supernode_idxs / nbr_idx / nbr_mask``
neighbour tensors and calls the pure-torch :class:`SupernodePooling`. The
``enc_proj`` / transformer ``blocks`` / perceiver-pooling tail are unchanged.
"""

from __future__ import annotations

from functools import partial

from kappamodules.layers import LinearProjection, Sequential
from kappamodules.transformer import (
    DitBlock,
    DitPerceiverPoolingBlock,
    PerceiverPoolingBlock,
    PrenormBlock,
)
from torch import nn

from neural_surrogates.architectures._upt.supernode_pooling import SupernodePooling


class EncoderSupernodes(nn.Module):
    def __init__(
        self,
        input_dim,
        ndim,
        radius,
        max_degree,
        gnn_dim,
        enc_dim,
        enc_depth,
        enc_num_heads,
        perc_dim=None,
        perc_num_heads=None,
        num_latent_tokens=None,
        cond_dim=None,
        init_weights="truncnormal",
        attn_ctor=None,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.ndim = ndim
        self.radius = radius
        self.max_degree = max_degree
        self.gnn_dim = gnn_dim
        self.enc_dim = enc_dim
        self.enc_depth = enc_depth
        self.enc_num_heads = enc_num_heads
        self.perc_dim = perc_dim
        self.perc_num_heads = perc_num_heads
        self.num_latent_tokens = num_latent_tokens
        self.condition_dim = cond_dim
        self.init_weights = init_weights

        # supernode pooling
        self.supernode_pooling = SupernodePooling(
            radius=radius,
            max_degree=max_degree,
            input_dim=input_dim,
            hidden_dim=gnn_dim,
            ndim=ndim,
        )

        # blocks
        self.enc_proj = LinearProjection(
            gnn_dim, enc_dim, init_weights=init_weights, optional=True
        )
        # ``attn_ctor`` (when given) selects the self-attention impl for the
        # transformer blocks; the perceiver-pooling tail below is left untouched.
        attn_kwargs = {} if attn_ctor is None else {"attn_ctor": attn_ctor}
        if cond_dim is None:
            block_ctor = partial(PrenormBlock, **attn_kwargs)
        else:
            block_ctor = partial(DitBlock, cond_dim=cond_dim, **attn_kwargs)
        self.blocks = Sequential(
            *[
                block_ctor(dim=enc_dim, num_heads=enc_num_heads, init_weights=init_weights)
                for _ in range(enc_depth)
            ],
        )

        # perceiver pooling
        if num_latent_tokens is None:
            self.perceiver = None
        else:
            if cond_dim is None:
                block_ctor = partial(
                    PerceiverPoolingBlock,
                    perceiver_kwargs=dict(
                        kv_dim=enc_dim,
                        init_weights=init_weights,
                    ),
                )
            else:
                block_ctor = partial(
                    DitPerceiverPoolingBlock,
                    perceiver_kwargs=dict(
                        kv_dim=enc_dim,
                        cond_dim=cond_dim,
                        init_weights=init_weights,
                    ),
                )
            self.perceiver = block_ctor(
                dim=perc_dim,
                num_heads=perc_num_heads,
                num_query_tokens=num_latent_tokens,
            )

    def forward(
        self,
        feat,
        pos,
        supernode_idxs,
        nbr_idx,
        nbr_mask,
        condition=None,
    ):
        # check inputs
        assert feat.ndim == 3, "expected dense tensor (batch_size, num_inputs, input_dim)"
        assert pos.ndim == 2, "expected shared positions (num_inputs, ndim)"
        assert supernode_idxs.ndim == 1
        if condition is not None:
            assert condition.ndim == 2, "expected shape (batch_size, cond_dim)"

        # pass condition to DiT blocks
        cond_kwargs = {}
        if condition is not None:
            cond_kwargs["cond"] = condition

        # supernode pooling -> (B, num_supernodes, gnn_dim)
        x = self.supernode_pooling(
            feat=feat,
            pos=pos,
            supernode_idxs=supernode_idxs,
            nbr_idx=nbr_idx,
            nbr_mask=nbr_mask,
        )

        # project to encoder dimension
        x = self.enc_proj(x)

        # transformer
        x = self.blocks(x, **cond_kwargs)

        # perceiver
        if self.perceiver is not None:
            x = self.perceiver(kv=x, **cond_kwargs)

        return x
