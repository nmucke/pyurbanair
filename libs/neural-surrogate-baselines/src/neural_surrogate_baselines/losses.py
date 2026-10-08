"""Training losses of the re-implemented papers."""

from __future__ import annotations

import torch


def masked_rmse(
    pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    """One RMSE over every masked cell and channel (Qin et al. 2025, Eq. 8).

    ``mask`` is boolean and broadcasts against ``pred`` (e.g. ``(B, 1, *grid)``).
    """
    mask = mask.expand_as(pred)
    return torch.sqrt(torch.square(pred - target)[mask].mean())


def spectral_loss(
    pred: torch.Tensor, target: torch.Tensor, dim: int = -2, norm: str = "backward"
) -> torch.Tensor:
    """Mean squared error of the complex Fourier coefficients along ``dim``
    (Park & Lee 2026, Eq. 10; ``dim`` is the lateral axis).

    By Parseval this equals ``n * MSE`` (``norm="backward"``, ``n`` the axis
    length), ``MSE`` (``"ortho"``) or ``MSE / n`` (``"forward"``): the spectral
    term only rescales the MSE. Kept for fidelity to the paper.
    """
    diff = torch.fft.fft(pred.float(), dim=dim, norm=norm) - torch.fft.fft(
        target.float(), dim=dim, norm=norm
    )
    return torch.mean(diff.real.square() + diff.imag.square())
