"""Domain-decomposed neural surrogate (Recommendation A, Phase 1).

A two-level overlapping domain-decomposition wrapper around a *fine* and a
*coarse* sub-network. Each sub-net is **any** registered architecture
(``UNetConvNeXt``, ``SimpleConv``, ``UPT``, ...), instantiated via Hydra from a
plain ``_target_`` config node -- so the two nets are configured by referring to
the existing ``architectures/<family>/<size>`` presets and the wrapper builds
them itself. The external contract is unchanged --
``forward(state, params, geometry) -> state_next`` on full grids -- so the
existing :class:`~neural_surrogates.training.Trainer` and
``TransitionDataset`` train it as a drop-in replacement.

One ``forward`` is one step of Algorithm 1 (companion PDF §5):

1. **Coarse step.** Average-pool the state and geometry to the coarse level,
   run a small dedicated ``coarse_net``, and trilinearly prolong the result
   back to the fine grid -- this is the global *context* field ``C``.
2. **Fine step.** Tile the state, geometry and context into uniform
   overlapping extended blocks, append the per-patch positional encoding, and
   run the ``fine_net`` (``residual=True``) once on the whole patch batch. The
   context + positional channels enter the stem RAW via the widened
   ``extra_in_channels`` path.
3. **Merge.** Crop each patch output to the partition-of-unity footprint,
   window-merge (``Σ_i w_i ≡ 1``) back to the full grid, crop to the original
   shape, and zero out obstacle cells with the geometry mask.

``domain_flexible = True`` advertises to the forward model that a single
trained instance can run any grid with the same cell spacing (the per-shape
tiling plan is rebuilt lazily by :class:`DomainDecomposition`).
"""

from __future__ import annotations

import inspect
from typing import Sequence

import torch
from neural_surrogates.decomposition import DomainDecomposition
from torch import nn


class DomainDecomposed(nn.Module):
    """Domain-decomposed surrogate wrapping a fine and a coarse sub-network.

    Parameters
    ----------
    n_state_channels:
        Number of state channels (e.g. 3 for ``u, v, w``).
    n_params:
        Number of inflow parameters.
    decomposition:
        Keyword arguments forwarded to :class:`DomainDecomposition` (e.g.
        ``interior_size``, ``halo``, ``taper``, ``coarsen_factor``, ``n_pos``,
        ``boundary_mode``, ``periodic_axes``, ``geometry_coarsen``). Global
        periodicity lives here, never on the inner nets.
    fine_net, coarse_net:
        Hydra config nodes (a dict / ``DictConfig`` carrying a ``_target_``, or
        an already-built ``nn.Module``) for the per-patch fine net and the
        dedicated coarse net. Any registered architecture is allowed
        (``UNetConvNeXt``, ``SimpleConv``, ``UPT``, ...); the hyperparameters
        come straight from the existing ``architectures/<family>/<size>``
        presets. The wrapper instantiates each node via Hydra, injecting
        ``n_state_channels`` / ``n_params`` and overriding the few keys that the
        decomposition fixes: the fine net's stem is widened by
        ``extra_in_channels = n_state_channels + n_pos`` to ingest the prolonged
        coarse context plus the positional encoding (the coarse net takes none),
        and both nets are forced non-periodic (``periodic_axes=()``) because the
        wrap is handled by the DD halo. A fine-net architecture that does not
        accept ``extra_in_channels`` raises a clear error (it cannot ingest the
        context); everything else (residual mode, normalisation, channel widths)
        is taken from the chosen preset.
    fine_chunk_size:
        Maximum number of patch blocks to push through ``fine_net`` per call.
        The fine step tiles the field into ``M`` overlapping blocks and, with a
        batch of ``B`` samples, runs the fine net on ``B*M`` blocks. On a large
        grid (e.g. Barcelona) ``M`` is large, so a single ``B*M`` call's
        activations can exhaust GPU memory even at ``B=2``. When set, the blocks
        are streamed through ``fine_net`` in chunks of this many and the outputs
        concatenated -- bounding the per-call working set. ``None`` (default)
        keeps the original single-call behaviour (byte-identical). Combine with
        ``fine_checkpoint`` to also bound the saved-for-backward activations.
    fine_checkpoint:
        Gradient-checkpoint each ``fine_net`` (chunk) call during training so its
        intermediate activations are recomputed in the backward pass instead of
        all being held until ``backward()``. This is what actually frees the
        memory that accumulates across patches while training (chunking alone
        only bounds the transient forward working set, since autograd otherwise
        retains every chunk's activations). Off in eval / no-grad. Default
        ``False`` (no recompute cost). Pair with ``fine_chunk_size`` for the full
        memory reduction.
    divergence_projection:
        Enable the (coarse-grid) divergence-cleaning projection of the merged
        velocity field. Default ``False``; the body is a runnable identity stub
        pending Eq (8).
    periodic_axes:
        Optional data-level periodicity knob shared with the other
        architectures, given as axis letters (e.g. ``["y"]`` for a spanwise-
        periodic domain). When provided it is translated to the ``(z, y, x)``
        bool tuple and **overrides** ``decomposition['periodic_axes']`` -- this
        lets the single ``architecture.periodic_axes`` setting in
        ``training.yaml`` drive a DD model exactly as it drives a plain
        ``UNetConvNeXt``. ``None`` keeps the preset's own
        ``decomposition.periodic_axes``.
    """

    domain_flexible = True

    def __init__(
        self,
        n_state_channels: int,
        n_params: int,
        decomposition: dict,
        fine_net,
        coarse_net,
        divergence_projection: bool = False,
        periodic_axes: Sequence[str] | None = None,
        fine_chunk_size: int | None = None,
        fine_checkpoint: bool = False,
    ) -> None:
        super().__init__()
        self.n_state_channels = n_state_channels
        self.n_params = n_params
        self.divergence_projection = bool(divergence_projection)
        # Patch-batch streaming knobs for large grids: cap the per-call fine-net
        # block count (transient working set) and/or recompute its activations in
        # backward (saved-activation memory). See the class docstring.
        if fine_chunk_size is not None and fine_chunk_size <= 0:
            raise ValueError(
                f"fine_chunk_size must be a positive int or None, got {fine_chunk_size}"
            )
        self.fine_chunk_size = fine_chunk_size
        self.fine_checkpoint = bool(fine_checkpoint)

        # ``periodic_axes`` (axis letters, e.g. ``["y"]``) is the data-level
        # periodicity knob shared with the other architectures and set once in
        # training.yaml. When given it overrides the preset's
        # ``decomposition.periodic_axes`` (a (z, y, x) bool tuple); the inner
        # nets stay non-periodic regardless (the wrap lives in the halo fill).
        decomposition = dict(decomposition)
        if periodic_axes is not None:
            axes = tuple(periodic_axes)
            unknown = set(axes) - {"z", "y", "x"}
            if unknown:
                raise ValueError(
                    "periodic_axes entries must be in ('z', 'y', 'x'), got "
                    f"{sorted(unknown)}"
                )
            decomposition["periodic_axes"] = tuple(a in axes for a in ("z", "y", "x"))
        self.dd = DomainDecomposition(**decomposition)

        # The fine net ingests the prolonged coarse context (C channels) plus the
        # positional encoding (n_pos channels) as RAW extra input, so its stem is
        # widened by ``extra_in_channels``. The coarse net takes no extra input.
        extra_in = n_state_channels + self.dd.n_pos
        self.fine_net = self._build_subnet(
            fine_net,
            n_state_channels,
            n_params,
            extra_in_channels=extra_in,
            role="fine_net",
        )
        self.coarse_net = self._build_subnet(
            coarse_net,
            n_state_channels,
            n_params,
            extra_in_channels=0,
            role="coarse_net",
        )

        # The positional encoding is constant per grid (cached in the DD plan),
        # so its batch-tiled form ``pos_blocks`` is identical across every
        # forward at a fixed batch size. Cache the materialised expand+reshape to
        # skip that copy each step; rebuilt only when (batch, plan) change.
        self._pos_blocks: torch.Tensor | None = None
        self._pos_blocks_key: tuple | None = None

    # ------------------------------------------------------------------ #
    @staticmethod
    def _build_subnet(node, n_state_channels, n_params, *, extra_in_channels, role):
        """Instantiate a sub-net from a Hydra ``_target_`` node (or pass an
        already-built ``nn.Module`` through), injecting the channel/param counts
        and the few keys the decomposition fixes.

        Only overrides the target actually accepts are passed (so any
        architecture works): ``extra_in_channels`` for the context/positional
        channels (required of the fine net -- a clear error if unsupported), and
        ``periodic_axes=()`` so the inner net never wraps (the DD halo owns
        periodicity). Everything else -- residual mode, normalisation, channel
        widths -- comes from the chosen preset.
        """
        if isinstance(node, nn.Module):
            return node
        if node is None or "_target_" not in node:
            raise ValueError(
                f"{role} must be a config node carrying a `_target_` (any "
                "registered architecture, e.g. neural_surrogates.UNetConvNeXt), "
                f"got {type(node).__name__}"
            )
        from hydra.utils import get_class, instantiate

        target = node["_target_"]
        accepts = inspect.signature(get_class(target)).parameters
        overrides = {"n_state_channels": n_state_channels, "n_params": n_params}
        if extra_in_channels:
            if "extra_in_channels" not in accepts:
                raise ValueError(
                    f"{role} architecture '{target}' does not accept "
                    "`extra_in_channels`, so it cannot ingest the coarse-context "
                    "/ positional channels. Use an architecture that supports the "
                    "extra-channel contract (UNetConvNeXt, SimpleConv, UPT)."
                )
            overrides["extra_in_channels"] = extra_in_channels
        elif "extra_in_channels" in accepts:
            overrides["extra_in_channels"] = 0
        # Inner nets must NOT be periodic: global periodicity is realised by the
        # DD halo fill, so each per-patch net uses ordinary edge padding.
        if "periodic_axes" in accepts:
            overrides["periodic_axes"] = ()
        return instantiate(node, **overrides)

    # ------------------------------------------------------------------ #
    def set_normalization(
        self,
        state_mean,
        state_std,
        param_mean,
        param_std,
    ) -> None:
        """Forward training-split standardisation statistics to whichever inner
        nets support it (a no-op for nets without ``set_normalization`` or built
        with ``normalize=False``)."""
        for net in (self.fine_net, self.coarse_net):
            if hasattr(net, "set_normalization"):
                net.set_normalization(state_mean, state_std, param_mean, param_std)

    def _run_fine(
        self,
        state_blocks: torch.Tensor,
        params_blocks: torch.Tensor,
        geom_blocks: torch.Tensor,
        extra: torch.Tensor,
    ) -> torch.Tensor:
        """Run ``fine_net`` over the ``(B*M, ...)`` patch batch, optionally in
        chunks and/or with gradient checkpointing to bound peak GPU memory.

        Without ``fine_chunk_size`` and ``fine_checkpoint`` this is a single
        ``fine_net`` call -- byte-identical to the original code path. With
        chunking the blocks are streamed in groups of ``fine_chunk_size`` and the
        outputs concatenated; with checkpointing each call's activations are
        recomputed in backward (only while training under grad)."""
        n = state_blocks.shape[0]
        chunk = self.fine_chunk_size or n

        def call(s, p, g, e):
            if self.fine_checkpoint and self.training and torch.is_grad_enabled():
                from torch.utils.checkpoint import checkpoint

                return checkpoint(
                    lambda s, p, g, e: self.fine_net(s, p, g, extra=e),
                    s,
                    p,
                    g,
                    e,
                    use_reentrant=False,
                )
            return self.fine_net(s, p, g, extra=e)

        if chunk >= n:
            return call(state_blocks, params_blocks, geom_blocks, extra)

        outs = []
        for i in range(0, n, chunk):
            sl = slice(i, i + chunk)
            outs.append(
                call(state_blocks[sl], params_blocks[sl], geom_blocks[sl], extra[sl])
            )
        return torch.cat(outs, dim=0)

    def _divergence_projection(self, state: torch.Tensor) -> torch.Tensor:
        """Project the merged velocity field to be (approximately) divergence
        free. Identity stub for now.

        # TODO Eq(8): coarse-grid Jacobi/FFT pressure projection.
        """
        return state

    # ------------------------------------------------------------------ #
    def forward(
        self,
        state: torch.Tensor,
        params: torch.Tensor,
        geometry: torch.Tensor,
        return_intermediates: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict]:
        """One DD step. With ``return_intermediates=False`` (default) this is
        byte-identical to the original contract -- it returns the merged
        ``state_next`` and the existing ``Trainer`` / forward model never pass
        the flag.

        With ``return_intermediates=True`` it returns ``(state_next, info)``
        where ``info`` is a dict exposing the intermediates the Eq (9) patch
        loss needs (see :class:`neural_surrogates.dd_loss.DomainDecompositionLoss`):

        * ``coarse_pred`` -- the coarse next state ``S̄_{t+1}``
          (``coarse_net`` output on the coarse grid, shape
          ``(B, C, *coarse_grid)``), supervised by ``λ_c`` against
          ``dd.restrict_coarse(target)``.
        * ``patch_pred`` -- the per-patch interior prediction blocks ``Ŝ_i``
          cropped to the ``(n + 2*taper)`` partition-of-unity footprint
          BEFORE windowing / merge, shape ``(B*M, C, blk, blk, blk)`` with
          ``blk = interior_size + 2*taper``. Adjacent patches' blocks overlap
          by ``2*taper`` cells; the interface term penalises their
          disagreement on that overlap band.
        * ``context`` -- the prolonged coarse context field ``C`` on the fine
          grid (``(B, C, Nz, Ny, Nx)``).
        * ``num_patches`` -- the patch count ``M`` for this grid.
        * ``dd`` -- a reference to ``self.dd`` so the loss can recover the
          tiling plan (counts, taper, neighbour indices) via
          ``dd.plan(state)`` / ``dd.neighbor_indices(state)``.
        """
        # Normalise the geometry to (B, 1, Nz, Ny, Nx).
        if geometry.dim() == state.dim() - 1:
            geometry = geometry.unsqueeze(1)
        geometry = geometry.to(dtype=state.dtype)

        b = state.shape[0]

        # --- Coarse step: global context field C = prolong(coarse_net(...)).
        coarse_state = self.dd.restrict_coarse(state)
        coarse_geom = self.dd.restrict_coarse_geom(geometry)
        coarse_out = self.coarse_net(coarse_state, params, coarse_geom)
        context = self.dd.prolong(coarse_out, state)  # (B, C, Nz, Ny, Nx)

        # --- Fine step: tile into extended blocks and run the fine net once.
        m = self.dd.num_patches(state)
        state_blocks = self.dd.restrict(state)  # (B*M, C, e, e, e)
        geom_blocks = self.dd.restrict(geometry)  # (B*M, 1, e, e, e)
        context_blocks = self.dd.restrict(context)  # (B*M, C, e, e, e)

        # Positional encoding: (M, n_pos, e, e, e) -> tile over batch to (B*M).
        # ``pos`` is plan-cached (constant per grid), so the expand+reshape (a
        # materialising copy) is cached too and only rebuilt when (b, plan, dtype,
        # device) change -- keyed on id(pos) since the plan rebuilds it on a new
        # grid.
        pos = self.dd.positional(state)  # (M, n_pos, e, e, e)
        pos_key = (b, m, id(pos), pos.dtype, pos.device)
        if self._pos_blocks_key != pos_key:
            self._pos_blocks = (
                pos.unsqueeze(0)
                .expand(b, *pos.shape)
                .reshape(b * m, *pos.shape[1:])
                .contiguous()
            )
            self._pos_blocks_key = pos_key
        pos_blocks = self._pos_blocks

        extra = torch.cat([context_blocks, pos_blocks], dim=1)

        # params (B, P) -> (B*M, P): each patch of a sample shares its params.
        params_blocks = (
            params.unsqueeze(1).expand(b, m, params.shape[1]).reshape(b * m, -1)
        )

        fine_out = self._run_fine(
            state_blocks, params_blocks, geom_blocks, extra
        )  # (B*M, C, e, e, e)

        # --- Merge: window-blend the patch outputs back to the full grid.
        merged = self.dd.extend_merge(fine_out, state)  # (B, C, Nz, Ny, Nx)

        if self.divergence_projection:
            merged = self._divergence_projection(merged)

        merged = merged * geometry

        if not return_intermediates:
            return merged

        # Per-patch interior prediction blocks Ŝ_i on the (n+2t) PoU footprint,
        # BEFORE windowing/merge: crop the (n+2h) extended fine output to the
        # central (n+2t) sub-block (the same crop extend_merge applies before
        # the window). Adjacent patches overlap by 2*taper cells here.
        h = self.dd.halo
        t = self.dd.taper
        lo = h - t
        blk = self.dd.interior_size + 2 * t
        patch_pred = fine_out[
            :, :, lo : lo + blk, lo : lo + blk, lo : lo + blk
        ].contiguous()  # (B*M, C, blk, blk, blk)

        info = {
            "coarse_pred": coarse_out,  # (B, C, *coarse_grid)
            "patch_pred": patch_pred,  # (B*M, C, n+2t, n+2t, n+2t)
            "context": context,  # (B, C, Nz, Ny, Nx)
            "num_patches": m,
            "dd": self.dd,
        }
        return merged, info
