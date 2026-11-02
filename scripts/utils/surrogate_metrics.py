"""The metrics of the surrogate comparison (surrogate_comparison/compute_metrics.py).

One pure function per metric or reference quantity, its formula in the
docstring, so each can be checked on its own (tests/scripts/test_surrogate_metrics.py).
Definitions: docs/plans/implemented/surrogate_comparison_paper_metrics.md.

Conventions: `fields` are (T, C, nz, ny, nx) arrays with C = (u, v, w) in m/s,
`fluid` the (nz, ny, nx) bool mask, `spacing` (dz, dy, dx) in m, `dt` the
output interval in s. Fluctuations (a prime) are about ū, the truth's per-cell
time mean over the compared window, unless a docstring says otherwise. Sums run
in float64: float32 sums over ~1e8 values lose digits.
"""

from __future__ import annotations

import numpy as np

F8 = np.float64


def fluid_values(fields: np.ndarray, fluid: np.ndarray) -> np.ndarray:
    """(T, C, n_fluid) values of (T, C, nz, ny, nx) `fields` on the fluid cells.

    `take` on the flat cells is much faster than a boolean mask."""
    return np.take(fields.reshape(*fields.shape[:2], -1), np.flatnonzero(fluid), -1)


def interior_cells(fluid: np.ndarray) -> np.ndarray:
    """(nz, ny, nx) bool: fluid cells whose six face neighbours are fluid.

    False on the domain edge (no neighbour there), so a central difference on
    these cells only reads fluid values."""
    inner = (slice(1, -1),) * 3
    interior = np.zeros_like(fluid, dtype=bool)
    interior[inner] = fluid[inner]
    for axis in range(3):
        for shift in (-1, 1):
            interior[inner] &= np.roll(fluid, shift, axis=axis)[inner]
    return interior


def vorticity(fields: np.ndarray, spacing: tuple[float, float, float]) -> np.ndarray:
    """(T, 3, nz, ny, nx) vorticity (ω_x, ω_y, ω_z) [1/s] of (T, 3, ...) (u, v, w).

    ω_x = ∂w/∂y − ∂v/∂z, ω_y = ∂u/∂z − ∂w/∂x, ω_z = ∂v/∂x − ∂u/∂y by central
    differences, (f[i+1] − f[i−1]) / 2h; only meaningful on `interior_cells`
    (one-sided at the domain edge)."""
    dz, dy, dx = spacing
    u, v, w = fields[:, 0], fields[:, 1], fields[:, 2]
    return np.stack(
        [
            np.gradient(w, dy, axis=-2) - np.gradient(v, dz, axis=-3),
            np.gradient(u, dz, axis=-3) - np.gradient(w, dx, axis=-1),
            np.gradient(v, dx, axis=-1) - np.gradient(u, dy, axis=-2),
        ],
        axis=1,
    )


def divergence(fields: np.ndarray, spacing: tuple[float, float, float]) -> np.ndarray:
    """(T, nz, ny, nx) divergence ∂u/∂x + ∂v/∂y + ∂w/∂z [1/s] of (T, 3, ...) (u, v, w),
    central differences as in :func:`vorticity`."""
    dz, dy, dx = spacing
    result: np.ndarray = (
        np.gradient(fields[:, 0], dx, axis=-1)
        + np.gradient(fields[:, 1], dy, axis=-2)
        + np.gradient(fields[:, 2], dz, axis=-3)
    )
    return result


def mean_building_height(fluid: np.ndarray, dz: float) -> float:
    """H [m]: the mean over the columns holding a solid cell of (solid cells in
    the column) * dz."""
    heights = (~fluid).sum(axis=0)
    return float(heights[heights > 0].mean() * dz)


def canopy_level(fluid: np.ndarray) -> int:
    """k_c: the lowest level with no solid cell (every spanwise line fully fluid)."""
    open_levels = np.flatnonzero(fluid.all(axis=(1, 2)))
    if not open_levels.size:
        raise ValueError("no level is entirely fluid: no canopy-top level")
    return int(open_levels[0])


def integral_time_scale(series: np.ndarray, dt: float) -> np.ndarray:
    """(n_rows,) integral time scale [s] of (n_rows, T) fluctuation series.

    R(τ) = Σ_t x_t x_{t+τ} / Σ_t x_t² (biased autocorrelation; the series is
    taken as given, not re-centred), T = dt (Σ_{τ=0}^{τ0−1} R(τ) − 1/2) with τ0
    the first lag where R ≤ 0 (the whole series if none): the trapezoid rule for
    ∫ R dt from 0 to τ0 dt with R(τ0) taken as 0 (Local-FNO §3.1; the rectangle
    rule is biased high by dt/2). NaN for a constant-zero row."""
    rows = np.asarray(series, dtype=F8)
    n = rows.shape[-1]
    size = 1 << int(2 * n - 1).bit_length()  # zero padding: linear, not circular
    acov = np.fft.irfft(np.abs(np.fft.rfft(rows, size, axis=-1)) ** 2, size)[:, :n]
    with np.errstate(all="ignore"):
        r = acov / acov[:, :1]
    nonpositive = r <= 0
    first = np.where(nonpositive.any(axis=1), nonpositive.argmax(axis=1), n)
    kept = np.where(np.arange(n) < first[:, None], r, 0.0)
    return np.where(acov[:, 0] > 0, dt * (kept.sum(axis=1) - 0.5), np.nan)


def reference_scales(
    truth: np.ndarray,
    fluid: np.ndarray,
    spacing: tuple[float, float, float],
    dt: float,
    chunk: int = 8192,
) -> dict[str, float]:
    """The per-sample reference scales of a (T, 3, nz, ny, nx) truth window.

    H: :func:`mean_building_height` [m]; k_c: :func:`canopy_level`; z_c = (k_c
    + 1/2) dz [m], its centre height; u_ref: the plane mean of ū_u over the fluid
    cells at k_c [m/s]; T: :func:`integral_time_scale` of truth − ū, averaged
    over the fluid cells and the three components [s] (cells processed `chunk`
    at a time)."""
    dz = spacing[0]
    level = canopy_level(fluid)
    values = fluid_values(truth, fluid)  # (T, 3, n)
    mean = values.mean(axis=0, dtype=F8)
    scales = []
    for start in range(0, values.shape[-1], chunk):
        prime = values[:, :, start : start + chunk] - mean[:, start : start + chunk]
        scales.append(integral_time_scale(prime.reshape(len(values), -1).T, dt))
    return {
        "H": mean_building_height(fluid, dz),
        "u_ref": float(truth[:, 0, level].mean(dtype=F8)),
        "T": float(np.nanmean(np.concatenate(scales))),
        "k_c": level,
        "z_c": (level + 0.5) * dz,
    }


def rmse(pred: np.ndarray, truth: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """RMSE of (T, C, n) `pred` against `truth` over the n cells:
    ((T,) pooled sqrt(mean_{c,cells} e²), (T, C) per component sqrt(mean_cells e²))."""
    mse = ((pred - truth) ** 2).mean(axis=-1, dtype=F8)
    return np.sqrt(mse.mean(axis=1)), np.sqrt(mse)


def acc(
    pred: np.ndarray, truth: np.ndarray, mean: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Anomaly correlation of (T, C, n) `pred` and `truth` about the (C, n) `mean` ū:
    Σ p't' / sqrt(Σ p'² Σ t'²) with p' = pred − ū, t' = truth − ū, summed over
    the cells: ((T,) pooled over the components too, (T, C) per component)."""
    p, t = pred - mean, truth - mean
    products = np.stack(
        [
            (p * t).sum(-1, dtype=F8),
            (p * p).sum(-1, dtype=F8),
            (t * t).sum(-1, dtype=F8),
        ]
    )  # (3, T, C)
    pooled = products.sum(axis=-1)
    return (
        pooled[0] / np.sqrt(pooled[1] * pooled[2]),
        products[0] / np.sqrt(products[1] * products[2]),
    )


def vorticity_rmse(
    pred: np.ndarray,
    truth: np.ndarray,
    interior: np.ndarray,
    spacing: tuple[float, float, float],
) -> tuple[np.ndarray, np.ndarray]:
    """RMSE [1/s] of the :func:`vorticity` of (T, 3, ...) `pred` against `truth` on
    the `interior` cells: ((T,) pooled over (ω_x, ω_y, ω_z), (T, 3) per component).
    One frame at a time (the vorticity of a whole rollout is GBs)."""
    cells = np.flatnonzero(interior)
    mse = np.empty((len(pred), 3))
    for i in range(len(pred)):
        error = vorticity(pred[i : i + 1], spacing) - vorticity(
            truth[i : i + 1], spacing
        )
        mse[i] = (np.take(error[0].reshape(3, -1), cells, -1) ** 2).mean(-1, dtype=F8)
    return np.sqrt(mse.mean(axis=1)), np.sqrt(mse)


def divergence_rms(
    fields: np.ndarray, interior: np.ndarray, spacing: tuple[float, float, float]
) -> np.ndarray:
    """(T,) RMS [1/s] of the :func:`divergence` of (T, 3, ...) `fields` on the
    `interior` cells, one frame at a time."""
    cells = np.flatnonzero(interior)
    return np.array(
        [
            np.sqrt(
                (np.take(divergence(f[None], spacing)[0], cells) ** 2).mean(dtype=F8)
            )
            for f in fields
        ]
    )


def tke(values: np.ndarray) -> np.ndarray:
    """(T,) resolved TKE [m²/s²] of (T, 3, n) velocity values, about their **own**
    time mean m: 0.5 Σ_c mean_cells (v_c − m_c)². Pass the finite frames only."""
    mean = values.mean(axis=0, dtype=F8).astype(values.dtype)
    result: np.ndarray = 0.5 * ((values - mean) ** 2).mean(axis=-1, dtype=F8).sum(1)
    return result


def tke_field(fields: np.ndarray) -> np.ndarray:
    """(nz, ny, nx) resolved TKE [m²/s²] of (T, 3, ...) `fields` about their own
    time mean: 0.5 Σ_c var_t(u_c)."""
    result: np.ndarray = 0.5 * fields.var(axis=0, dtype=F8).sum(axis=0)
    return result


def spanwise_lines(plane: np.ndarray, fluid_level: np.ndarray) -> np.ndarray:
    """(T, ny, n_lines): the fully fluid y-lines (x columns with no solid cell)
    of a (T, ny, nx) plane, with the (ny, nx) fluid mask of its level."""
    return plane[:, :, fluid_level.all(axis=0)]


def spanwise_spectrum(lines: np.ndarray, dy: float) -> tuple[np.ndarray, np.ndarray]:
    """Per-frame spanwise spectrum of (T, ny, n_lines) fluctuation lines.

    y is periodic: each line is FFT'd as it is, no window or detrending. The
    one-sided density, averaged over the lines, E(κ) = w_κ |FFT_κ / ny|² / dκ
    (w_κ = 2 except at κ = 0 and the even-ny Nyquist bin), so that
    Σ E dκ = the variance along y about the line mean. The angular wavenumber
    κ = 2π / λ, as the papers; evaluation.turbulence.spanwise_spectra has the
    same density per cycle/m (2π times this; it also subtracts the time mean of
    what it is given and averages over frames). Returns `(κ, E)`: κ (n_k,) in
    rad/m without κ = 0, E (T, n_k) in (m/s)² m/rad."""
    n_y = lines.shape[1]
    k = 2 * np.pi * np.fft.rfftfreq(n_y, d=dy)
    weight = np.full(k.size, 2.0)
    weight[0] = 1.0
    if n_y % 2 == 0:
        weight[-1] = 1.0
    power = (np.abs(np.fft.rfft(lines.astype(F8), axis=1)) ** 2).mean(axis=-1)
    return k[1:], (power * weight / (n_y**2 * k[1]))[:, 1:]


def integral_length(lines: np.ndarray, dy: float) -> np.ndarray:
    """(T,) spanwise integral length Λ [m] of (T, ny, n_lines) fluctuation lines.

    R(Δy) = Σ_lines Σ_y u'(y) u'(y + Δy) / Σ_lines Σ_y u'(y)², y periodic, with
    each line's spanwise mean removed per frame (the κ = 0 bin, which the
    spectra drop too: a gust or inflow-angle swing over the whole line would
    give R a positive floor); Λ = dy (Σ_{m=0}^{m0−1} R(m dy) − 1/2) with m0 the
    first lag where R ≤ 0, searched up to ny/2 (all lags up to ny/2 if R never
    drops to 0): the trapezoid rule for ∫ R dy from 0 to m0 dy with R(m0) taken
    as 0, as :func:`integral_time_scale`."""
    n_y = lines.shape[1]
    spectrum = np.abs(np.fft.rfft(lines.astype(F8), axis=1)) ** 2
    spectrum[:, 0] = 0.0  # the line means
    acov = np.fft.irfft(spectrum.sum(axis=-1), n_y, axis=1)[:, : n_y // 2 + 1]
    with np.errstate(all="ignore"):
        r = acov / acov[:, :1]
    nonpositive = r <= 0
    first = np.where(nonpositive.any(axis=1), nonpositive.argmax(axis=1), r.shape[1])
    kept = np.where(np.arange(r.shape[1]) < first[:, None], r, 0.0)
    result: np.ndarray = dy * (kept.sum(axis=1) - 0.5)
    return result


def bin_spectrum(kh: np.ndarray, spectrum: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """(..., n_bins) of a (..., n_k) spectrum at the wavenumbers `kh` on bins `edges`.

    A bin holding wavenumbers gets their mean; an empty bin whose geometric
    centre lies inside [kh[0], kh[-1]] gets the spectrum interpolated linearly
    in log k there; a bin outside that range is NaN."""
    centres = np.sqrt(edges[1:] * edges[:-1])
    index = np.digitize(kh, edges) - 1
    out = np.full((*spectrum.shape[:-1], len(centres)), np.nan)
    for b, centre in enumerate(centres):
        inside = index == b
        if inside.any():
            out[..., b] = spectrum[..., inside].mean(axis=-1)
        elif kh[0] <= centre <= kh[-1]:
            j = np.searchsorted(kh, centre)  # kh[j - 1] < centre < kh[j]
            a = np.log(centre / kh[j - 1]) / np.log(kh[j] / kh[j - 1])
            out[..., b] = (1 - a) * spectrum[..., j - 1] + a * spectrum[..., j]
    return out


def density(values: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """(n_bins,) probability density of 1-D `values` on bins `edges`:
    counts / (len(values) * bin width). Values outside the edges fall out but
    still count in len(values), so tails stay comparable between sources."""
    counts = np.histogram(values, edges)[0]
    result: np.ndarray = counts / (len(values) * np.diff(edges))
    return result


def window_rms(curve: np.ndarray) -> float:
    """sqrt(mean_t curve²) over the window (NaN if any lead is)."""
    with np.errstate(all="ignore"):
        return float(np.sqrt(np.mean(np.asarray(curve, dtype=F8) ** 2)))
