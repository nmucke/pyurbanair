"""Eq (9) patch-based training objective for the domain-decomposed surrogate.

This module exposes :class:`DomainDecompositionLoss`, the four-term loss of
companion PDF §2.7 / Eq (9):

.. math::

    \\mathcal{L} = \\underbrace{\\|\\Delta_i|_{\\Omega_i^\\circ} - d_i\\|^2}_{\\text{one-step}}
        + \\lambda_{if} \\sum_{j \\in \\mathcal{N}(i)}
            \\|\\hat S_i - \\hat S_j\\|^2_{\\Omega_i \\cap \\Omega_j}
        + \\lambda_{div} \\|\\nabla\\cdot\\hat u\\|^2
        + \\lambda_c \\|\\bar S_{t+1} - \\mathcal{R}_H S^{ref}_{t+1}\\|^2 .

It consumes the ``info`` dict returned by
``DomainDecomposed.forward(..., return_intermediates=True)`` together with the
merged prediction, the ground-truth next state, the input state, the geometry
mask and the decomposition object, and returns a scalar total loss plus a dict
of the individual (detached) term values for logging.

Design notes
------------
* **one-step** is computed on the *merged* next-state field rather than on
  per-patch interior deltas. The merged field is the windowed sum of the
  interior predictions, so penalising it on fluid cells is equivalent to (and
  simpler than) penalising every interior delta; it is also exactly the field
  the autoregressive rollout consumes.
* **interface** uses the model's own tiling: ``info['patch_pred']`` are the
  ``(n + 2*taper)`` partition-of-unity blocks ``Ŝ_i`` BEFORE windowing, which
  overlap their axis neighbours by ``2*taper`` cells. For each ``+axis`` face
  (so each shared overlap is visited once) the high ``2*taper`` band of patch
  ``i`` is MSE-matched to the low ``2*taper`` band of its neighbour ``j``.
  Periodic wrap is automatic: ``dd.neighbor_indices`` already returns the
  wrapped patch as the neighbour across a periodic seam.
* **divergence** is a soft regulariser: central finite differences of the
  velocity channels of the merged field, grid spacing assumed = 1 cell.
"""

from __future__ import annotations

from typing import Optional, Sequence

import torch
from torch import nn


class DomainDecompositionLoss(nn.Module):
    """Four-term Eq (9) loss for :class:`DomainDecomposed`.

    Parameters
    ----------
    lambda_interface:
        Weight ``λ_if`` of the neighbour-overlap interface term (Table 1
        default ``0.1``). Set to ``None`` to drop the interface term entirely
        -- it is then neither computed nor logged.
    lambda_divergence:
        Weight ``λ_div`` of the velocity-divergence regulariser (default
        ``0.01``). ``None`` drops the divergence term (not computed/logged).
    lambda_coarse:
        Weight ``λ_c`` of the coarse-supervision term (default ``1.0``).
        ``None`` drops the coarse term (not computed/logged).
    velocity_channels:
        State channels holding the velocity components ``(u, v, w)`` =
        ``(x-velocity, y-velocity, z-velocity)`` for the divergence term, in
        that order. Each component is differenced along its own physical axis.
        Defaults to ``(0, 1, 2)``.
    mask_loss:
        Restrict the one-step and divergence terms to fluid cells (geometry
        ``> 0``); obstacle cells carry junk solver values. Default ``True``.
    """

    def __init__(
        self,
        lambda_interface: Optional[float] = 0.1,
        lambda_divergence: Optional[float] = 0.01,
        lambda_coarse: Optional[float] = 1.0,
        velocity_channels: Sequence[int] = (0, 1, 2),
        mask_loss: bool = True,
    ) -> None:
        super().__init__()
        # ``None`` disables a term: it is neither computed nor logged. A bare
        # float keeps it active with that weight.
        self.lambda_interface = None if lambda_interface is None else float(lambda_interface)
        self.lambda_divergence = None if lambda_divergence is None else float(lambda_divergence)
        self.lambda_coarse = None if lambda_coarse is None else float(lambda_coarse)
        self.velocity_channels = tuple(int(c) for c in velocity_channels)
        self.mask_loss = bool(mask_loss)

    # ------------------------------------------------------------------ #
    def forward(
        self,
        info: dict,
        state_next: torch.Tensor,
        target_next: torch.Tensor,
        geometry: torch.Tensor,
        dd=None,
    ) -> tuple[torch.Tensor, dict]:
        """Return ``(total_loss, terms)``.

        Parameters
        ----------
        info:
            The dict from ``model(..., return_intermediates=True)`` -- needs at
            least ``patch_pred``, ``coarse_pred`` and (``dd`` or) ``num_patches``.
        state_next:
            Merged predicted next state ``(B, C, Nz, Ny, Nx)``.
        target_next:
            Ground-truth next state ``(B, C, Nz, Ny, Nx)``.
        geometry:
            Fluid mask ``(B, 1, Nz, Ny, Nx)`` or ``(B, Nz, Ny, Nx)``.
        dd:
            The :class:`DomainDecomposition`. Falls back to ``info['dd']``.

        ``terms`` always maps ``'one_step'`` and ``'total'`` to detached scalar
        tensors; ``'interface'``, ``'divergence'`` and ``'coarse'`` appear only
        when their weight is not ``None`` (a disabled term is never computed, so
        it is omitted from ``terms`` rather than logged as zero).
        """
        dd = dd if dd is not None else info["dd"]

        if geometry.dim() == state_next.dim() - 1:
            geometry = geometry.unsqueeze(1)
        geometry = geometry.to(dtype=state_next.dtype)
        fluid = geometry > 0  # (B, 1, Nz, Ny, Nx)

        # one-step is the base supervision (implicit weight 1) and always on;
        # each weighted term is computed only when its lambda is not None.
        one_step = self._one_step_term(state_next, target_next, fluid)
        total = one_step
        terms = {"one_step": one_step.detach()}

        if self.lambda_interface is not None:
            interface = self._interface_term(info, dd)
            total = total + self.lambda_interface * interface
            terms["interface"] = interface.detach()

        if self.lambda_divergence is not None:
            divergence = self._divergence_term(state_next, fluid)
            total = total + self.lambda_divergence * divergence
            terms["divergence"] = divergence.detach()

        if self.lambda_coarse is not None:
            coarse = self._coarse_term(info, target_next, dd)
            total = total + self.lambda_coarse * coarse
            terms["coarse"] = coarse.detach()

        terms["total"] = total.detach()
        return total, terms

    # ------------------------------------------------------------------ #
    def _one_step_term(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        fluid: torch.Tensor,
    ) -> torch.Tensor:
        """MSE between predicted and target next-state on fluid cells."""
        diff = pred - target
        if self.mask_loss:
            mask = fluid.expand_as(diff)
            sel = diff[mask]
            if sel.numel() == 0:
                return diff.new_zeros(())
            return sel.pow(2).mean()
        return diff.pow(2).mean()

    # ------------------------------------------------------------------ #
    def _interface_term(self, info: dict, dd) -> torch.Tensor:
        """Penalise disagreement between adjacent patches on their overlap.

        ``patch_pred`` is ``(B*M, C, blk, blk, blk)`` with ``blk = n + 2*taper``;
        adjacent patches overlap by ``2*taper`` cells. For every ``+axis`` face
        (slots 1, 3, 5 of ``neighbor_indices``; each shared band visited once)
        the high ``2*taper`` band of patch ``i`` is matched to the low
        ``2*taper`` band of neighbour ``j``. Periodic wrap is handled by
        ``neighbor_indices`` returning the wrapped patch.
        """
        patch_pred = info["patch_pred"]  # (B*M, C, blk, blk, blk)
        t = dd.taper
        if t == 0:
            return patch_pred.new_zeros(())

        if "num_patches" not in info:
            raise KeyError(
                "interface term needs info['num_patches'] to reshape "
                "patch_pred (B*M, C, ...) into (B, M, C, ...)"
            )
        m = int(info["num_patches"])
        bm, c = patch_pred.shape[0], patch_pred.shape[1]
        b = bm // m
        blocks = patch_pred.view(b, m, c, *patch_pred.shape[2:])

        # neighbor_indices needs a tensor with the right spatial shape to look
        # up the plan; rebuild it from the patch counts via a probe.
        neighbors = self._neighbor_indices(info, dd, blocks)  # (M, 6) long

        band = 2 * t
        blk = blocks.shape[-1]
        # (+z, +y, +x) live in slots 1, 3, 5; spatial axis a is dim 3+a of the
        # ``(B, M, C, z, y, x)`` block tensor.
        face_slots = ((0, 1), (1, 3), (2, 5))

        # Vectorised over patches: for each axis gather every patch's high band
        # and its neighbour's low band in one indexed op (no Python loop over M,
        # no per-pair ``int(neighbors[i])`` host sync, no M-long autograd chain).
        # Every valid pair contributes a band of identical element count, so the
        # element-summed MSE below is exactly the original mean-of-per-pair-means
        # (equal group sizes), just in a single reduction.
        total_sq = patch_pred.new_zeros(())
        total_elems = 0
        for axis, slot in face_slots:
            dim = 3 + axis  # spatial dim in the (B, M, C, z, y, x) block tensor
            nbr = neighbors[:, slot]  # (M,) flat neighbour index, -1 = boundary
            valid = nbr >= 0  # drop non-periodic global-boundary faces
            hi = blocks.narrow(dim, blk - band, band)[:, valid]  # (B, n_valid, ...)
            lo = blocks.narrow(dim, 0, band)[:, nbr[valid]]  # neighbour low bands
            diff = hi - lo
            total_sq = total_sq + diff.pow(2).sum()
            total_elems += diff.numel()
        if total_elems == 0:
            return total_sq
        return total_sq / total_elems

    @staticmethod
    def _neighbor_indices(info: dict, dd, blocks: torch.Tensor) -> torch.Tensor:
        """``(M, 6)`` neighbour table. Built from the model's grid via the
        cached plan; ``dd.plan`` was populated during the forward pass on the
        same grid, so a probe of that grid reuses the cache."""
        plan = dd._plan
        if plan is not None:
            gz, gy, gx = plan.grid
            probe = blocks.new_zeros(1, 1, gz, gy, gx)
        else:  # pragma: no cover - forward always builds the plan first
            m = blocks.shape[1]
            probe = blocks.new_zeros(1, 1, m * dd.interior_size, 1, 1)
        return dd.neighbor_indices(probe)

    # ------------------------------------------------------------------ #
    def _divergence_term(
        self, state: torch.Tensor, fluid: torch.Tensor
    ) -> torch.Tensor:
        """Squared velocity divergence (central differences, ``dx = 1``).

        ``∇·u = du/dx + dv/dy + dw/dz`` where ``velocity_channels = (cu, cv, cw)``
        name the channels holding the x-, y- and z-velocity components
        ``(u, v, w)``. Each component is differenced along its matching physical
        axis (x = dim 4, y = dim 3, z = dim 2). Central differences leave the two
        boundary planes per axis out; the divergence is evaluated and penalised on
        the interior intersection (and on fluid cells when ``mask_loss``).
        """
        cu, cv, cw = self.velocity_channels
        u = state[:, cu : cu + 1]  # x-velocity, (B, 1, Nz, Ny, Nx)
        v = state[:, cv : cv + 1]  # y-velocity
        w = state[:, cw : cw + 1]  # z-velocity

        # central diff each component along its physical axis (dims: z=2, y=3,
        # x=4); crop to the common interior [1:-1] on all axes so the terms align.
        du_dx = (u[:, :, 1:-1, 1:-1, 2:] - u[:, :, 1:-1, 1:-1, :-2]) * 0.5
        dv_dy = (v[:, :, 1:-1, 2:, 1:-1] - v[:, :, 1:-1, :-2, 1:-1]) * 0.5
        dw_dz = (w[:, :, 2:, 1:-1, 1:-1] - w[:, :, :-2, 1:-1, 1:-1]) * 0.5
        div = du_dx + dv_dy + dw_dz  # (B, 1, Nz-2, Ny-2, Nx-2)
        if div.numel() == 0:
            return state.new_zeros(())

        if self.mask_loss:
            inner = fluid[:, :, 1:-1, 1:-1, 1:-1]
            sel = div[inner]
            if sel.numel() == 0:
                return state.new_zeros(())
            return sel.pow(2).mean()
        return div.pow(2).mean()

    # ------------------------------------------------------------------ #
    def _coarse_term(
        self, info: dict, target_next: torch.Tensor, dd
    ) -> torch.Tensor:
        """MSE(coarse_pred, restrict_coarse(target_next))."""
        coarse_pred = info["coarse_pred"]
        coarse_target = dd.restrict_coarse(target_next)
        return (coarse_pred - coarse_target).pow(2).mean()
