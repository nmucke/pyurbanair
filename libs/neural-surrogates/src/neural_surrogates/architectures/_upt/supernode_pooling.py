"""Pure-torch supernode pooling for the UPT encoder.

Vendored and REWRITTEN from upt-tutorial:
    https://github.com/BenediktAlkin/upt-tutorial
    upt/modules/supernode_pooling.py  (main branch)

upt-tutorial and KappaModules are MIT-licensed (Benedikt Alkin); see
``_upt/LICENSE``.

The upstream implementation relies on ``torch_geometric.nn.pool.radius_graph``
and ``torch_scatter.segment_csr`` over a sparse ``(B*N, .)`` layout. Those
native dependencies ship as ABI-pinned wheels and have no prebuilt wheel for
the torch version used here, so this module replaces the graph op with a dense,
pure-torch radius neighbourhood + masked-mean aggregation over a SHARED point
set (the fluid cells, identical across the batch). Only ``forward`` changes;
``input_proj``, ``pos_embed`` and the ``message`` MLP are kept identical to
upstream so the learned semantics match.
"""

from __future__ import annotations

import torch
from kappamodules.layers import ContinuousSincosEmbed, LinearProjection
from torch import nn


class SupernodePooling(nn.Module):
    def __init__(
        self,
        radius,
        max_degree,
        input_dim,
        hidden_dim,
        ndim,
        init_weights="torch",
    ):
        super().__init__()
        self.radius = radius
        self.max_degree = max_degree
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.ndim = ndim
        self.init_weights = init_weights

        self.input_proj = LinearProjection(
            input_dim, hidden_dim, init_weights=init_weights
        )
        self.pos_embed = ContinuousSincosEmbed(dim=hidden_dim, ndim=ndim)
        self.message = nn.Sequential(
            LinearProjection(hidden_dim * 2, hidden_dim, init_weights=init_weights),
            nn.GELU(),
            LinearProjection(hidden_dim, hidden_dim, init_weights=init_weights),
        )
        self.output_dim = hidden_dim

    @staticmethod
    def build_neighbors(pos, supernode_pos, radius, max_degree):
        """Precompute the padded neighbour index/mask for each supernode.

        Parameters
        ----------
        pos:           ``(N, ndim)`` positions of all input points (shared).
        supernode_pos: ``(S, ndim)`` positions of the selected supernodes.
        radius:        neighbourhood radius (in the same units as ``pos``).
        max_degree:    maximum number of neighbours kept per supernode.

        Returns
        -------
        nbr_idx:  ``(S, K)`` long tensor of neighbour indices into ``[0, N)``,
                  padded with 0 where ``nbr_mask`` is False (``K <= max_degree``).
        nbr_mask: ``(S, K)`` bool tensor marking valid neighbours.

        The selection is a constant function of the (shared) geometry, so no
        gradient flows through it -- matching upstream where the graph topology
        is not learned.
        """
        # dist[s, n] = ||supernode_pos[s] - pos[n]||
        dist = torch.cdist(supernode_pos, pos)  # (S, N)
        within = dist <= radius  # self-loop guaranteed (dist == 0 for the supernode)

        # Keep up to ``max_degree`` nearest within-radius neighbours per supernode.
        # Sort by distance (ascending) and take the first ``K`` columns; invalidate
        # entries that fall outside the radius via the gathered mask.
        n = pos.shape[0]
        k = min(int(max_degree), n)
        order = torch.argsort(dist, dim=1)[:, :k]  # (S, K) indices into [0, N)
        nbr_mask = torch.gather(within, 1, order)  # (S, K) bool
        nbr_idx = order
        return nbr_idx, nbr_mask

    def forward(self, feat, pos, supernode_idxs, nbr_idx, nbr_mask):
        """Dense supernode pooling over a shared point set.

        Parameters
        ----------
        feat:           ``(B, N, input_dim)`` per-sample point features.
        pos:            ``(N, ndim)`` shared point positions.
        supernode_idxs: ``(S,)`` long indices into ``[0, N)`` selecting supernodes.
        nbr_idx:        ``(S, K)`` long neighbour indices (see ``build_neighbors``).
        nbr_mask:       ``(S, K)`` bool neighbour validity mask.

        Returns
        -------
        ``(B, S, hidden_dim)`` pooled supernode features.
        """
        assert feat.ndim == 3, "expected dense tensor (batch_size, num_inputs, input_dim)"
        assert pos.ndim == 2, "expected shared positions (num_inputs, ndim)"
        assert supernode_idxs.ndim == 1

        b = feat.shape[0]

        # embed mesh: project features + add positional embedding (shared over batch)
        x = self.input_proj(feat) + self.pos_embed(pos).unsqueeze(0)  # (B, N, hidden)

        # gather destination (supernode) features: (B, S, hidden)
        dst = x[:, supernode_idxs, :]

        # gather source (neighbour) features: (B, S, K, hidden)
        s, k = nbr_idx.shape
        src = x[:, nbr_idx.reshape(-1), :].reshape(b, s, k, -1)

        # message input concat[src, dst] -> message MLP, matching upstream semantics
        dst_b = dst.unsqueeze(2).expand(b, s, k, dst.shape[-1])  # broadcast dst
        m = self.message(torch.cat([src, dst_b], dim=-1))  # (B, S, K, hidden)

        # masked mean over the neighbour (K) axis
        mask = nbr_mask.to(m.dtype).unsqueeze(0).unsqueeze(-1)  # (1, S, K, 1)
        summed = (m * mask).sum(dim=2)  # (B, S, hidden)
        counts = mask.sum(dim=2).clamp_min(1.0)  # (1, S, 1) -- self-loop => >= 1
        x = summed / counts

        return x
