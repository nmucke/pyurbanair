"""3D spectral convolution, the kernel operator ``K`` of an FNO layer.

Follows Li et al.'s reference ``SpectralConv3d``: a real FFT over
``(z, y, x)``, a learned complex linear map on the lowest ``modes`` of each
axis, and an inverse FFT. The x axis is the halved real-FFT axis, so its
modes are ``0 .. m_x - 1``; z and y keep ``±m`` (the four corner blocks). By
Hermitian symmetry that covers ``|k_j| <= m_j`` on every axis, Qin et al.'s
Eq. (7).
"""

from __future__ import annotations

import torch
from torch import nn


class SpectralConv3d(nn.Module):
    def __init__(
        self, in_channels: int, out_channels: int, modes: tuple[int, int, int]
    ) -> None:
        super().__init__()
        self.modes = tuple(int(m) for m in modes)
        scale = 1.0 / (in_channels * out_channels)
        shape = (in_channels, out_channels, *self.modes)
        # One weight per (z, y) corner: (+, +), (-, +), (+, -), (-, -). Stored
        # as real (..., 2) pairs, viewed as complex at use: ``module.to(dtype)``
        # (the forward model's cast) would drop the imaginary part of a
        # complex parameter.
        self.weights = nn.ParameterList(
            nn.Parameter(scale * torch.rand(*shape, 2)) for _ in range(4)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, _, nz, ny, nx = x.shape
        # Modes beyond the grid's Nyquist limit do not exist; small patches
        # (or the tiny test grids) use the lowest ones only.
        mz = min(self.modes[0], nz // 2)
        my = min(self.modes[1], ny // 2)
        mx = min(self.modes[2], nx // 2 + 1)
        # FFTs in fp32: half-precision FFTs need power-of-two sizes.
        with torch.autocast(device_type=x.device.type, enabled=False):
            x_ft = torch.fft.rfftn(x.float(), dim=(-3, -2, -1))
            out = torch.zeros(
                b,
                self.weights[0].shape[1],
                nz,
                ny,
                nx // 2 + 1,
                dtype=torch.cfloat,
                device=x.device,
            )
            zs = (slice(0, mz), slice(nz - mz, nz))
            ys = (slice(0, my), slice(ny - my, ny))
            for weight, (sz, sy) in zip(
                self.weights,
                ((zs[0], ys[0]), (zs[1], ys[0]), (zs[0], ys[1]), (zs[1], ys[1])),
            ):
                w = torch.view_as_complex(weight.float())
                out[:, :, sz, sy, :mx] = torch.einsum(
                    "bizyx,iozyx->bozyx",
                    x_ft[:, :, sz, sy, :mx],
                    w[:, :, :mz, :my, :mx],
                )
            return torch.fft.irfftn(out, s=(nz, ny, nx), dim=(-3, -2, -1))
