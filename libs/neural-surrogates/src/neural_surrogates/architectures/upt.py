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

import inspect

import torch
from kappamodules.attention import (
    DotProductAttention1d,
    DotProductAttentionSlow,
    EfficientAttention1d,
    LinformerAttention1d,
    TranssolverAttention,
)
from torch import nn

from neural_surrogates.architectures._upt.approximator import Approximator
from neural_surrogates.architectures._upt.decoder import DecoderPerceiver
from neural_surrogates.architectures._upt.encoder import EncoderSupernodes
from neural_surrogates.architectures._upt.supernode_pooling import SupernodePooling


# Selectable self-attention implementations for the encoder / approximator /
# decoder transformer stacks. All are the ``...1d`` (token-sequence) variants
# operating on ``(B, N, dim)`` tokens; the perceiver cross-attention tails are
# left untouched.
ATTENTION_TYPES = {
    "dot_product": DotProductAttention1d,
    "dot_product_slow": DotProductAttentionSlow,
    "efficient": EfficientAttention1d,
    "linformer": LinformerAttention1d,
    "transsolver": TranssolverAttention,
}


def _build_attn_ctor(attention_type, seqlen, extra):
    """Return an ``attn_ctor(**block_kwargs)`` for ``attention_type``, or ``None``.

    ``None`` is the sentinel for the default dot-product attention: the
    sub-modules then keep their original block types unchanged, so existing
    checkpoints/behaviour are preserved exactly.

    The transformer blocks call the attention constructor with a fixed set of
    kwargs (``dim``/``num_heads``/``qkv_bias``/``init_weights``/... and, for
    ``PrenormBlock``, ``proj_bias``). The attention classes accept different
    subsets and some need extra arguments, so the returned ctor (a) injects the
    per-attention extras (``seqlen`` for Linformer, ``num_slices`` for
    Transsolver, anything in ``extra``) and (b) drops kwargs the chosen class
    does not accept.
    """
    if attention_type is None or attention_type == "dot_product":
        return None
    if attention_type not in ATTENTION_TYPES:
        raise ValueError(
            f"unknown attention_type {attention_type!r}; "
            f"choose from {sorted(ATTENTION_TYPES)}"
        )
    base = ATTENTION_TYPES[attention_type]

    fixed = dict(extra or {})
    if attention_type == "linformer":
        # Linformer projects the key/value sequence to a fixed length, so it
        # needs the (compile-time) self-attention sequence length. kv_seqlen
        # defaults to no compression unless the caller overrides it.
        fixed.setdefault("input_seqlen", seqlen)
        fixed.setdefault("kv_seqlen", seqlen)
    if attention_type == "transsolver" and "num_slices" not in fixed:
        raise ValueError(
            "attention_type='transsolver' requires attention_kwargs.num_slices"
        )

    accepted = set(inspect.signature(base.__init__).parameters)

    def attn_ctor(**block_kwargs):
        merged = {**block_kwargs, **fixed}
        return base(**{k: v for k, v in merged.items() if k in accepted})

    return attn_ctor


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
    normalize:
        Z-score normalise the per-point state channels and the inflow params
        before they enter the network, and de-normalise the output. This
        mirrors upstream UPT, which standardises every field with dataset
        mean/std (``upt/datasets/simulation_dataset.py``). It is **load-bearing
        here**: the raw inflow ``inflow_angle`` is ~50x larger than the
        velocity channels, so without normalisation the encoder's single
        ``input_proj`` linear layer is swamped by the param offset and the
        latent stops carrying the flow -- the model then decodes a
        params/position-conditioned field that barely depends on the input
        state and the rollout collapses to a constant. The statistics live in
        buffers (``state_mean``/``state_std``/``param_mean``/``param_std``,
        identity by default) and are set from the training data via
        :meth:`set_normalization` (saved/restored with the checkpoint).
    predict_residual:
        Predict the *change* ``state_{t+1} - state_t`` instead of the absolute
        next state, i.e. ``out = state + decode(...)``. Upstream UPT predicts
        the absolute field, but it operates at far higher token-per-point
        capacity (512 supernodes / 128 latent tokens over an 8192-point
        subsampled mesh) than is affordable on these dense ~256k-cell grids.
        At this compression an absolute prediction collapses to the smooth mean
        field; the residual is ~14% of the field magnitude, so predicting it
        keeps the task well-scaled and -- crucially -- makes near-identity (the
        correct behaviour for a slow transient) the model's default, which is
        what lets the rollout actually advance in time rather than freeze on a
        constant.
    attention_type:
        Self-attention implementation used by the encoder / approximator /
        decoder transformer stacks (the perceiver cross-attention tails are
        unaffected). One of :data:`ATTENTION_TYPES`: ``"dot_product"`` (default,
        standard scaled-dot-product), ``"dot_product_slow"``, ``"efficient"``
        (linear-attention), ``"linformer"`` or ``"transsolver"``. The default
        leaves every block exactly as before.
    attention_kwargs:
        Extra keyword arguments forwarded to the attention constructor, e.g.
        ``{"num_slices": 32}`` for ``transsolver`` (required) or
        ``{"kv_seqlen": 32}`` for ``linformer``.
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
        normalize: bool = True,
        predict_residual: bool = True,
        # --- attention ---
        attention_type: str = "dot_product",
        attention_kwargs: dict | None = None,
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
        self.normalize = normalize
        self.predict_residual = predict_residual
        self.attention_type = attention_type

        # Build the self-attention constructors for the transformer stacks. The
        # encoder attends over the supernode tokens, the approximator/decoder
        # over the latent tokens, so each gets the sequence length it operates on
        # (needed by length-dependent attentions such as Linformer).
        attn_extra = {**attention_kwargs} if attention_kwargs else {}
        enc_attn_ctor = _build_attn_ctor(attention_type, num_supernodes, attn_extra)
        latent_attn_ctor = _build_attn_ctor(
            attention_type, num_latent_tokens, attn_extra
        )

        # Standardisation statistics (identity until set_normalization is called
        # from the training data). Registered as buffers so they travel with the
        # checkpoint -- every call site (training, rollout, test) then gets the
        # correct normalisation for free by simply loading the state dict.
        self.register_buffer("state_mean", torch.zeros(n_state_channels))
        self.register_buffer("state_std", torch.ones(n_state_channels))
        self.register_buffer("param_mean", torch.zeros(n_params))
        self.register_buffer("param_std", torch.ones(n_params))

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
            attn_ctor=enc_attn_ctor,
        )
        self.approximator = Approximator(
            input_dim=dim,
            depth=approx_depth,
            num_heads=num_heads,
            dim=dim,
            cond_dim=cond_dim,
            attn_ctor=latent_attn_ctor,
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
            attn_ctor=latent_attn_ctor,
        )

        # lazy per-(D,H,W,device,dtype) coordinate cache (no giant buffers)
        self._coords_cache: dict = {}
        # lazy cache of the (fluid_idx, positions, supernodes, neighbour graph)
        # derived from a geometry. The graph is a pure function of the (fixed)
        # geometry, so caching it turns the per-step ``cdist`` over every fluid
        # cell into a one-off cost -- important now that the supernode count is
        # large enough to resolve these dense grids.
        self._geom_cache: dict = {}

    # -- normalisation -----------------------------------------------------

    @torch.no_grad()
    def set_normalization(
        self,
        state_mean,
        state_std,
        param_mean=None,
        param_std=None,
        eps: float = 1e-6,
    ) -> None:
        """Install per-channel standardisation statistics (see ``normalize``).

        ``state_mean``/``state_std`` are length-``C`` and
        ``param_mean``/``param_std`` length-``P`` (params left untouched if
        omitted). Standard deviations are floored at ``eps`` so a constant
        channel/param (e.g. a fixed ``pressure_gradient_magnitude``) maps to 0
        rather than dividing by zero.
        """

        def _to(buf, value):
            t = torch.as_tensor(value, dtype=buf.dtype, device=buf.device)
            if t.shape != buf.shape:
                raise ValueError(
                    f"normalization stat shape {tuple(t.shape)} does not match "
                    f"buffer shape {tuple(buf.shape)}"
                )
            return t

        self.state_mean.copy_(_to(self.state_mean, state_mean))
        self.state_std.copy_(_to(self.state_std, state_std).clamp_min(eps))
        if param_mean is not None:
            self.param_mean.copy_(_to(self.param_mean, param_mean))
        if param_std is not None:
            self.param_std.copy_(_to(self.param_std, param_std).clamp_min(eps))

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
        coordinates ``(D*H*W, ndim)``. The result is cached on the geometry's
        fluid-cell identity (the graph never changes while the geometry does
        not), so the dense neighbour build runs once rather than every step.
        """
        fluid_idx = mask.reshape(-1).nonzero(as_tuple=False).squeeze(1)  # (N,)
        key = (coords.shape[0], int(fluid_idx.numel()), mask.device, coords.dtype)
        cached = self._geom_cache.get(key)
        if cached is not None and torch.equal(cached[0], fluid_idx):
            return cached

        input_pos = coords[fluid_idx]  # (N, ndim)
        n = input_pos.shape[0]
        supernode_local = self._select_supernodes(n, input_pos.device)
        nbr_idx, nbr_mask = SupernodePooling.build_neighbors(
            input_pos,
            input_pos[supernode_local],
            self.radius,
            self.max_degree,
        )
        result = (fluid_idx, input_pos, supernode_local, nbr_idx, nbr_mask)
        self._geom_cache[key] = result
        return result

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

        # Standardise the state channels and inflow params before the network
        # (see ``normalize``). With identity buffers this is a no-op.
        if self.normalize:
            feat_in = (flat - self.state_mean.view(1, 1, -1)) / self.state_std.view(
                1, 1, -1
            )
            params_in = (params - self.param_mean.view(1, -1)) / self.param_std.view(
                1, -1
            )
        else:
            feat_in = flat
            params_in = params

        if self.cond_dim is None:
            params_b = params_in[:, None, :].expand(b, feat_in.shape[1], -1)
            feats = torch.cat([feat_in, params_b], dim=-1)  # (B, N, C + P)
            condition = None
        else:
            feats = feat_in
            condition = self.param_proj(params_in)  # (B, cond_dim)

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

        # De-normalise the network output. In residual mode it is the
        # (normalised) change, so only the std scales back; in absolute mode the
        # mean is added too. The std/mean cancel cleanly because the input was
        # standardised with the same statistics.
        if self.normalize:
            pred = pred * self.state_std.view(1, 1, -1)
            if not self.predict_residual:
                pred = pred + self.state_mean.view(1, 1, -1)

        # scatter back to the grid; obstacle cells stay 0
        out = state.new_zeros(b, c, d * h * w)
        # pred may be half under autocast while out follows the input dtype;
        # index_put won't cast, so match out's dtype explicitly (no-op otherwise).
        out[:, :, fluid_idx] = pred.transpose(1, 2).to(out.dtype)  # (B, C, N) placed
        out = out.reshape(b, c, d, h, w)

        # Residual prediction: the network output is the change applied to the
        # current state. Obstacle cells stay 0 (the residual there is 0 and the
        # input state is masked), so the final geometry multiply is unaffected.
        if self.predict_residual:
            out = out + state
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
