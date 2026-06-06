"""Universal Physics Transformer (UPT) neural surrogate.

Wraps the vendored UPT encoder / approximator / decoder (see ``_upt/``) behind
the same ``forward(state, params, geometry) -> next_state`` contract used by the
other architectures (``unet_convnext.py``, ``simple_conv.py``).

The state lives on a regular ``(D, H, W)`` grid; UPT treats the fluid cells as
a point cloud, pools them onto a fixed set of supernodes, runs a latent
transformer, and decodes back at the fluid-cell positions. Obstacle cells stay
exactly zero.

Shared-geometry assumption
--------------------------
Within a single forward call the per-sample geometry masks are identical (every
batch member voxelises the same STL onto the same trained grid -- see
``forward_model.rollout_batched`` / ``data.TransitionDataset``). The fast path
builds ONE point set + supernode selection + neighbour graph from
``geometry[0]`` and batches only the features. A per-sample fallback loop covers
the rare non-shared case so correctness never depends on the assumption.

UPT building blocks are derived from upt-tutorial + KappaModules (MIT, Benedikt
Alkin); see ``_upt/LICENSE``.
"""

from __future__ import annotations

import torch
from torch import nn

from neural_surrogates.architectures._upt.approximator import Approximator
from neural_surrogates.architectures._upt.decoder import DecoderPerceiver
from neural_surrogates.architectures._upt.encoder import EncoderSupernodes
from neural_surrogates.architectures._upt.supernode_pooling import SupernodePooling


class UPT(nn.Module):
    """Universal Physics Transformer surrogate.

    Parameters
    ----------
    n_state_channels:
        Number of state channels ``C`` (Hydra-injected).
    n_params:
        Number of inflow parameters ``P`` (Hydra-injected).
    dim:
        Latent token dimension shared by encoder / approximator / decoder.
        Must be divisible by ``num_heads``.
    num_latent_tokens:
        Number of perceiver latent tokens produced by the encoder.
    num_supernodes:
        Number of supernodes the input fluid points are pooled onto. Clamped to
        ``min(num_supernodes, N)`` when there are fewer fluid cells.
    radius:
        Supernode neighbourhood radius, in integer cell units (spacing 1.0), so
        ``radius=2.5`` reaches roughly a 5-cell-wide neighbourhood.
    max_degree:
        Maximum neighbours kept per supernode.
    gnn_dim:
        Hidden dimension of the supernode-pooling message MLP.
    enc_depth, approx_depth, dec_depth:
        Transformer depths of the encoder, approximator and decoder.
    num_heads:
        Attention heads (``dim % num_heads == 0`` required).
    cond_dim:
        If ``None`` (default) the inflow ``params`` are concatenated to every
        point's input features and the model runs without DiT conditioning. If
        set, ``params`` are projected to a ``(B, cond_dim)`` condition and passed
        as DiT conditioning to encoder/approximator/decoder.
    ndim:
        Spatial dimensionality of the grid (3 for these volumes).
    """

    def __init__(
        self,
        n_state_channels: int,
        n_params: int,
        # --- latent / token sizing ---
        dim: int = 192,
        num_latent_tokens: int = 64,
        # --- supernodes (input pooling) ---
        num_supernodes: int = 64,
        radius: float = 2.5,
        max_degree: int = 16,
        gnn_dim: int = 96,
        # --- depths / heads ---
        enc_depth: int = 2,
        approx_depth: int = 4,
        dec_depth: int = 2,
        num_heads: int = 3,
        # --- conditioning ---
        cond_dim: int | None = None,
        ndim: int = 3,
    ) -> None:
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(
                f"dim ({dim}) must be divisible by num_heads ({num_heads})"
            )

        self.n_state_channels = n_state_channels
        self.n_params = n_params
        self.dim = dim
        self.num_latent_tokens = num_latent_tokens
        self.num_supernodes = num_supernodes
        self.radius = radius
        self.max_degree = max_degree
        self.gnn_dim = gnn_dim
        self.num_heads = num_heads
        self.cond_dim = cond_dim
        self.ndim = ndim

        # default (cond_dim is None): inflow params concatenated to per-point feats
        feat_dim = n_state_channels + (n_params if cond_dim is None else 0)
        self.feat_dim = feat_dim

        if cond_dim is not None:
            # optional DiT path: project params -> condition vector
            self.param_proj = nn.Sequential(
                nn.Linear(n_params, cond_dim),
                nn.SiLU(),
                nn.Linear(cond_dim, cond_dim),
            )
        else:
            self.param_proj = None

        self.encoder = EncoderSupernodes(
            input_dim=feat_dim,
            ndim=ndim,
            radius=radius,
            max_degree=max_degree,
            gnn_dim=gnn_dim,
            enc_dim=dim,
            enc_depth=enc_depth,
            enc_num_heads=num_heads,
            perc_dim=dim,
            perc_num_heads=num_heads,
            num_latent_tokens=num_latent_tokens,
            cond_dim=cond_dim,
        )
        self.approximator = Approximator(
            input_dim=dim,
            depth=approx_depth,
            num_heads=num_heads,
            dim=dim,
            cond_dim=cond_dim,
        )
        self.decoder = DecoderPerceiver(
            input_dim=dim,
            output_dim=n_state_channels,
            ndim=ndim,
            dim=dim,
            depth=dec_depth,
            num_heads=num_heads,
            perc_dim=dim,
            perc_num_heads=num_heads,
            cond_dim=cond_dim,
        )

        # lazy per-(D,H,W,device,dtype) coordinate cache (no giant buffers)
        self._coords_cache: dict = {}

    # -- geometry-derived helpers ------------------------------------------

    def _grid_coords(
        self, d: int, h: int, w: int, device, dtype
    ) -> torch.Tensor:
        """Integer ``(z, y, x)`` cell coordinates, ``(D*H*W, ndim)``, cached."""
        key = (d, h, w, device, dtype)
        coords = self._coords_cache.get(key)
        if coords is None:
            zz, yy, xx = torch.meshgrid(
                torch.arange(d, device=device, dtype=dtype),
                torch.arange(h, device=device, dtype=dtype),
                torch.arange(w, device=device, dtype=dtype),
                indexing="ij",
            )
            coords = torch.stack(
                [zz.reshape(-1), yy.reshape(-1), xx.reshape(-1)], dim=-1
            )  # (D*H*W, 3) ordered (z, y, x)
            self._coords_cache[key] = coords
        return coords

    def _select_supernodes(self, n: int, device) -> torch.Tensor:
        """Deterministic strided supernode selection over ``[0, N)``.

        No randomness (required for ``test_rollout_batched_matches_per_member``),
        and independent of batch size -- derived purely from the shared geometry.
        """
        s = min(self.num_supernodes, n)
        stride = max(1, n // s)
        idx = torch.arange(0, n, stride, device=device)[:s]
        return idx

    def _build_point_geometry(self, mask: torch.Tensor, coords: torch.Tensor):
        """Build fluid indices, positions, supernodes and neighbour tensors.

        ``mask`` is a single ``(D, H, W)`` geometry; ``coords`` are the grid cell
        coordinates ``(D*H*W, ndim)``.
        """
        fluid_idx = mask.reshape(-1).nonzero(as_tuple=False).squeeze(1)  # (N,)
        input_pos = coords[fluid_idx]  # (N, ndim)
        n = input_pos.shape[0]
        supernode_local = self._select_supernodes(n, input_pos.device)
        nbr_idx, nbr_mask = SupernodePooling.build_neighbors(
            input_pos,
            input_pos[supernode_local],
            self.radius,
            self.max_degree,
        )
        return fluid_idx, input_pos, supernode_local, nbr_idx, nbr_mask

    def _run(
        self,
        state: torch.Tensor,
        params: torch.Tensor,
        mask: torch.Tensor,
        coords: torch.Tensor,
    ) -> torch.Tensor:
        """Encode/approximate/decode a batch that SHARES the geometry ``mask``."""
        b, c, d, h, w = state.shape
        (
            fluid_idx,
            input_pos,
            supernode_local,
            nbr_idx,
            nbr_mask,
        ) = self._build_point_geometry(mask, coords)

        # per-point input features (B, N, C) gathered at fluid cells
        flat = state.reshape(b, c, -1)[:, :, fluid_idx].transpose(1, 2)  # (B, N, C)
        if self.cond_dim is None:
            params_b = params[:, None, :].expand(b, flat.shape[1], -1)
            feats = torch.cat([flat, params_b], dim=-1)  # (B, N, C + P)
            condition = None
        else:
            feats = flat
            condition = self.param_proj(params)  # (B, cond_dim)

        latent = self.encoder(
            feats,
            input_pos,
            supernode_local,
            nbr_idx,
            nbr_mask,
            condition=condition,
        )  # (B, num_latent_tokens, dim)
        latent = self.approximator(latent, condition=condition)

        query_pos = input_pos.unsqueeze(0).expand(b, -1, -1)  # (B, N, ndim)
        pred = self.decoder(latent, query_pos, condition=condition)  # (B, N, C)

        # scatter back to the grid; obstacle cells stay 0
        out = state.new_zeros(b, c, d * h * w)
        # pred may be half under autocast while out follows the input dtype;
        # index_put won't cast, so match out's dtype explicitly (no-op otherwise).
        out[:, :, fluid_idx] = pred.transpose(1, 2).to(out.dtype)  # (B, C, N) placed
        out = out.reshape(b, c, d, h, w)
        return out

    # -- forward -----------------------------------------------------------

    def forward(
        self,
        state: torch.Tensor,
        params: torch.Tensor,
        geometry: torch.Tensor,
    ) -> torch.Tensor:
        # normalize geometry to (B, D, H, W)
        if geometry.dim() == state.dim():  # (B, 1, D, H, W) -> (B, D, H, W)
            geometry = geometry.squeeze(1)
        b, c, d, h, w = state.shape
        coords = self._grid_coords(d, h, w, state.device, state.dtype)

        # cheap shared-geometry guard: compare per-sample fluid-cell counts to
        # geometry[0]. Cheap (a reduction per step), not a per-element equal.
        mask0 = geometry[0]
        if b > 1:
            counts = geometry.reshape(b, -1).count_nonzero(dim=1)  # (B,)
            shared = bool(torch.all(counts == counts[0]).item())
        else:
            shared = True

        if shared:
            # fast path: one point set / supernode graph for the whole batch
            return self._run(state, params, mask0, coords) * geometry.unsqueeze(1)

        # documented fallback: rebuild per sample (rare; geometry not shared)
        outs = []
        for i in range(b):
            outs.append(
                self._run(
                    state[i : i + 1],
                    params[i : i + 1],
                    geometry[i],
                    coords,
                )
            )
        out = torch.cat(outs, dim=0)
        return out * geometry.unsqueeze(1)
