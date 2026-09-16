"""Conditional statistics for accepting a latent generator (plan 07 phase 3).

Pure numpy functions over stacks of physical states ``(N, C, nz, ny, nx)`` and
a fluid mask ``(nz, ny, nx)`` (``1`` / ``True`` = fluid). Nothing here touches
Hydra, files, matplotlib or the model: ``scripts/neural_surrogate/
test_latent_generator.py`` owns generation, batching and I/O and calls in here,
so every metric can be unit-tested on tiny analytic fields.

Why a separate module rather than ``evaluation.turbulence``: that library scores
*time series* of ensemble runs (Welch spectra at probes, streaming time-moments
over window files). A generator is judged on *snapshot ensembles* under matched
conditioning -- moments across samples, spatial spectra along a row, a
divergence stencil. Its one directly reusable function,
``evaluation.turbulence.log_spectral_distance``, is deliberately NOT imported:
``libs/evaluation`` is a leaf the scripts depend on, while
``neural_surrogates`` declares only numpy / xarray / torch, and a backend
library reaching into the evaluation library would invert that layering for a
six-line formula. :func:`log_spectral_distance` here is the same definition
(RMS dB ratio over the bins both spectra resolve) restricted to ``k > 0``;
keep the two in step if either changes.

Conventions (every function):

* ``fields`` is ``(N, C, nz, ny, nx)``; ``fluid`` is ``(nz, ny, nx)`` and only
  fluid cells count. Obstacle cells are ignored, never zero-filled into a mean.
* Results are plain dicts of numpy arrays / floats so they serialise straight
  into ``metrics.csv`` / ``summary.json`` and merge across trajectory groups
  with :func:`merge_group_metrics` (count-weighted).
* ``nan`` marks "not estimable" (a z level with no fluid, fewer than two
  samples for a covariance, no fully-fluid row for a spectrum); callers keep
  the nan rather than substituting zero.
"""

from __future__ import annotations

import math
import warnings
from typing import Any, Sequence

import numpy as np

# Report order for the initial-state sources the acceptance study compares.
SOURCE_ORDER: tuple[str, ...] = (
    "real",
    "ae_recon",
    "generated",
    "generated_const_history",
    "generated_shuffled_history",
    "generated_omitted_history",
)

# ``i <= j`` component pairs of the Reynolds-stress tensor, in report order.
STRESS_PAIRS: tuple[tuple[int, int], ...] = (
    (0, 0),
    (1, 1),
    (2, 2),
    (0, 1),
    (0, 2),
    (1, 2),
)


def nanmean(values: Any) -> float:
    """Mean over the estimable (finite) entries, ``nan`` when there are none.

    ``np.nanmean`` is right except for its "Mean of empty slice" warning: an
    all-nan input is a normal outcome here (a single-sample group has no
    covariance, no bootstrap and no pairwise spread), and the nan it returns is
    the answer, not a numerical accident to warn about.
    """
    arr = np.asarray(values, dtype=np.float64).ravel()
    if arr.size == 0 or not np.isfinite(arr).any():
        return float("nan")
    return float(np.mean(arr[np.isfinite(arr)]))


def _check_fields(
    fields: np.ndarray, fluid: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    f = np.asarray(fields, dtype=np.float64)
    m = np.asarray(fluid).astype(bool)
    if f.ndim != 5:
        raise ValueError(f"fields must be (N, C, nz, ny, nx), got {f.shape}")
    if m.shape != f.shape[2:]:
        raise ValueError(
            f"fluid mask {m.shape} does not match the field grid {f.shape[2:]}"
        )
    return f, m


# --------------------------------------------------------------------------- #
# Masks
# --------------------------------------------------------------------------- #


def stencil_fluid_mask(fluid: np.ndarray) -> np.ndarray:
    """Cells whose 6 face neighbours are all fluid (and that are not on the
    domain boundary): where a central-difference stencil is fully valid.

    A cell touching an obstacle -- or the domain edge -- would difference across
    a zeroed / missing neighbour and report a spurious divergence, so those
    cells are excluded rather than one-sided.
    """
    f = np.asarray(fluid).astype(bool)
    if f.ndim != 3:
        raise ValueError(f"fluid must be (nz, ny, nx), got {f.shape}")
    out = np.zeros_like(f)
    if min(f.shape) < 3:
        return out
    c = f[1:-1, 1:-1, 1:-1]
    out[1:-1, 1:-1, 1:-1] = (
        c
        & f[:-2, 1:-1, 1:-1]
        & f[2:, 1:-1, 1:-1]
        & f[1:-1, :-2, 1:-1]
        & f[1:-1, 2:, 1:-1]
        & f[1:-1, 1:-1, :-2]
        & f[1:-1, 1:-1, 2:]
    )
    return out


# --------------------------------------------------------------------------- #
# Profiles
# --------------------------------------------------------------------------- #


def profiles(fields: np.ndarray, fluid: np.ndarray) -> dict[str, Any]:
    """Conditional mean and fluctuation-RMS vertical profiles.

    Per component and z level, averaged over every sample and every fluid cell
    of that level: ``mean[c, z] = <u_c>`` and ``rms[c, z] = sqrt(<(u_c -
    mean[c, z])^2>)`` (the level's own mean is removed, so ``rms`` is the
    horizontal + sample fluctuation intensity, not the second raw moment).

    Returns ``{"mean": (C, nz), "rms": (C, nz), "count": (nz,), "sample_mean":
    (N, C, nz)}``; ``count`` is samples x fluid cells per level (the merge
    weight) and ``sample_mean`` the per-sample level means the held-out
    bootstrap resamples.
    """
    f, m = _check_fields(fields, fluid)
    n, c, nz = f.shape[0], f.shape[1], f.shape[2]
    cells = m.sum(axis=(1, 2)).astype(np.float64)  # (nz,)
    safe = np.where(cells > 0, cells, 1.0)
    masked = f * m[None, None]
    sample_mean = masked.sum(axis=(3, 4)) / safe  # (N, C, nz)
    mean = sample_mean.mean(axis=0) if n > 0 else np.full((c, nz), np.nan)
    dev = (f - mean[None, :, :, None, None]) * m[None, None]
    rms = np.sqrt((dev**2).sum(axis=(0, 3, 4)) / (max(n, 1) * safe))
    empty = cells == 0
    mean[:, empty] = np.nan
    rms[:, empty] = np.nan
    sample_mean[:, :, empty] = np.nan
    return {
        "mean": mean,
        "rms": rms,
        "count": cells * n,
        "sample_mean": sample_mean,
    }


def profile_rmse(a: dict[str, Any], b: dict[str, Any], key: str = "mean") -> float:
    """Count-weighted RMSE between two profile dicts (``key`` = mean | rms)
    over the levels both estimate; nan when none."""
    pa, pb = np.asarray(a[key]), np.asarray(b[key])
    w = np.asarray(b["count"], dtype=np.float64)
    ok = np.isfinite(pa) & np.isfinite(pb) & (w[None, :] > 0)
    if not ok.any():
        return float("nan")
    wt = np.broadcast_to(w[None, :], pa.shape)[ok]
    return float(np.sqrt(np.sum(wt * (pa[ok] - pb[ok]) ** 2) / np.sum(wt)))


def bootstrap_profile_rmse(
    sample_mean: np.ndarray, n_resamples: int = 200, seed: int = 0
) -> float:
    """Held-out sampling variability of a mean profile: the RMSE between a
    bootstrap resample's profile and the full-set profile, averaged over
    resamples. The reference scale a generator's profile error is judged
    against (an ``N``-sample real set cannot pin its own mean better)."""
    s = np.asarray(sample_mean, dtype=np.float64)
    n = s.shape[0]
    if n < 2:
        return float("nan")
    rng = np.random.default_rng(seed)
    errs = []
    # A level with no fluid is nan in every sample; nanmean's "empty slice"
    # warning there is expected, not a problem.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        full = np.nanmean(s, axis=0)
        for _ in range(int(n_resamples)):
            idx = rng.integers(0, n, size=n)
            rep = np.nanmean(s[idx], axis=0)
            ok = np.isfinite(rep) & np.isfinite(full)
            errs.append(float(np.sqrt(np.mean((rep[ok] - full[ok]) ** 2))))
    return float(np.mean(errs))


# --------------------------------------------------------------------------- #
# Distributions
# --------------------------------------------------------------------------- #


def fluid_values(
    fields: np.ndarray,
    fluid: np.ndarray,
    max_values: int | None = None,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Fluid-cell values pooled over samples, ``(C, M)``, optionally
    subsampled (without replacement) to ``max_values`` per component."""
    f, m = _check_fields(fields, fluid)
    vals = f[:, :, m].transpose(1, 0, 2).reshape(f.shape[1], -1)
    if max_values is not None and vals.shape[1] > max_values:
        rng = np.random.default_rng(0) if rng is None else rng
        idx = rng.choice(vals.shape[1], size=int(max_values), replace=False)
        vals = vals[:, np.sort(idx)]
    return vals


def pool_values(
    chunks: Sequence[np.ndarray],
    max_values: int | None = None,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Concatenate ``(C, M_i)`` reservoirs along ``M`` and cap at ``max_values``."""
    if not chunks:
        return np.zeros((0, 0))
    vals = np.concatenate([np.asarray(c) for c in chunks], axis=1)
    if max_values is not None and vals.shape[1] > max_values:
        rng = np.random.default_rng(0) if rng is None else rng
        idx = rng.choice(vals.shape[1], size=int(max_values), replace=False)
        vals = vals[:, np.sort(idx)]
    return vals


def wasserstein_1(a: np.ndarray, b: np.ndarray) -> float:
    """1-Wasserstein distance between two 1-D empirical distributions:
    ``int |F_a(x) - F_b(x)| dx`` over the pooled support. A pure translation
    by ``d`` gives exactly ``|d|``."""
    a = np.sort(np.asarray(a, dtype=np.float64).ravel())
    b = np.sort(np.asarray(b, dtype=np.float64).ravel())
    if a.size == 0 or b.size == 0:
        return float("nan")
    support = np.sort(np.concatenate([a, b]))
    cdf_a = np.searchsorted(a, support[:-1], side="right") / a.size
    cdf_b = np.searchsorted(b, support[:-1], side="right") / b.size
    return float(np.sum(np.abs(cdf_a - cdf_b) * np.diff(support)))


def histogram_edges(reference: np.ndarray, n_bins: int = 64) -> np.ndarray:
    """Shared bin edges ``(C, n_bins + 1)`` spanning the reference values per
    component (1 % padding so the extremes fall inside)."""
    ref = np.asarray(reference, dtype=np.float64)
    edges = np.empty((ref.shape[0], int(n_bins) + 1))
    for c in range(ref.shape[0]):
        lo, hi = float(np.min(ref[c])), float(np.max(ref[c]))
        pad = 0.01 * (hi - lo) if hi > lo else 0.5
        edges[c] = np.linspace(lo - pad, hi + pad, int(n_bins) + 1)
    return edges


def histograms_and_w1(
    values: np.ndarray, reference: np.ndarray, edges: np.ndarray
) -> dict[str, Any]:
    """Per-component histogram densities on fixed ``edges`` plus the
    1-Wasserstein distance of ``values`` to ``reference`` (both ``(C, M)``).

    Returns ``{"density": (C, nb), "counts": (C, nb), "w1": (C,)}``.
    """
    vals = np.asarray(values, dtype=np.float64)
    ref = np.asarray(reference, dtype=np.float64)
    c = edges.shape[0]
    counts = np.zeros((c, edges.shape[1] - 1))
    density = np.zeros_like(counts)
    w1 = np.full(c, np.nan)
    for i in range(c):
        if vals.shape[0] <= i or vals.shape[1] == 0:
            continue
        counts[i], _ = np.histogram(vals[i], bins=edges[i])
        density[i], _ = np.histogram(vals[i], bins=edges[i], density=True)
        w1[i] = wasserstein_1(vals[i], ref[i])
    return {"density": density, "counts": counts, "w1": w1}


def bootstrap_w1(
    reference: np.ndarray, n_resamples: int = 50, seed: int = 0
) -> np.ndarray:
    """Per-component W1 between a bootstrap resample of ``reference`` and the
    full reference, averaged: the sampling floor a W1 comparison sits on."""
    ref = np.asarray(reference, dtype=np.float64)
    rng = np.random.default_rng(seed)
    out = np.full(ref.shape[0], np.nan)
    for c in range(ref.shape[0]):
        m = ref.shape[1]
        if m < 2:
            continue
        dists = [
            wasserstein_1(ref[c, rng.integers(0, m, size=m)], ref[c])
            for _ in range(int(n_resamples))
        ]
        out[c] = float(np.mean(dists))
    return out


# --------------------------------------------------------------------------- #
# Spectra
# --------------------------------------------------------------------------- #


def spectra(fields: np.ndarray, fluid: np.ndarray, dx: float = 1.0) -> dict[str, Any]:
    """1-D energy spectra along ``x`` averaged over fully fluid rows.

    A row ``(sample, component, z, y)`` contributes only when every one of its
    ``nx`` cells is fluid (an obstacle inside the row would inject a step). Per
    row ``E(k) = |rfft(u)|^2 / nx^2`` (the ``k = 0`` bin holds the row mean's
    energy and is kept but excluded from distances); ``k = 2*pi*rfftfreq(nx,
    dx)``. Returns ``{"k": (nk,), "energy": (C, nk), "rows": int}``; ``energy``
    is nan when no row qualifies.
    """
    f, m = _check_fields(fields, fluid)
    n, c, nz, ny, nx = f.shape
    rows = m.all(axis=2)  # (nz, ny)
    k = 2.0 * math.pi * np.fft.rfftfreq(nx, d=float(dx))
    n_rows = int(rows.sum()) * n
    if n_rows == 0:
        return {"k": k, "energy": np.full((c, k.size), np.nan), "rows": 0}
    sel = f[:, :, rows, :]  # (N, C, R, nx)
    coeff = np.fft.rfft(sel, axis=-1) / nx
    energy = (np.abs(coeff) ** 2).mean(axis=(0, 2))
    return {"k": k, "energy": energy, "rows": n_rows}


def log_spectral_distance(energy_a: np.ndarray, energy_b: np.ndarray) -> np.ndarray:
    """Per-component RMS of ``10*log10(E_a/E_b)`` (dB) over the bins ``k > 0``
    where both spectra are positive; nan when none."""
    a, b = np.asarray(energy_a, dtype=np.float64), np.asarray(
        energy_b, dtype=np.float64
    )
    out = np.full(a.shape[0], np.nan)
    for c in range(a.shape[0]):
        ea, eb = a[c, 1:], b[c, 1:]
        ok = np.isfinite(ea) & np.isfinite(eb) & (ea > 0) & (eb > 0)
        if ok.any():
            out[c] = float(np.sqrt(np.mean((10.0 * np.log10(ea[ok] / eb[ok])) ** 2)))
    return out


# --------------------------------------------------------------------------- #
# Reynolds stresses
# --------------------------------------------------------------------------- #


def reynolds_stresses(fields: np.ndarray, fluid: np.ndarray) -> dict[str, Any]:
    """Resolved second moments ``R_ij = <u_i' u_j'>`` about the per-cell
    sample mean of the stack (fluctuations across the ``N`` samples under the
    same conditioning), ``ddof = 1``, then averaged over fluid cells.

    Returns ``{"stress": (C, C), "profile": (C, C, nz), "count": int}`` --
    ``stress`` over all fluid cells, ``profile`` per z level; nan with fewer
    than two samples (a constant field gives exactly zero).
    """
    f, m = _check_fields(fields, fluid)
    n, c, nz = f.shape[0], f.shape[1], f.shape[2]
    if n < 2:
        return {
            "stress": np.full((c, c), np.nan),
            "profile": np.full((c, c, nz), np.nan),
            "count": int(m.sum()),
        }
    dev = f - f.mean(axis=0, keepdims=True)  # (N, C, ...)
    comoment = np.einsum("nizyx,njzyx->ijzyx", dev, dev) / (n - 1)
    cells = m.sum(axis=(1, 2)).astype(np.float64)
    safe = np.where(cells > 0, cells, 1.0)
    profile = (comoment * m[None, None]).sum(axis=(3, 4)) / safe
    profile[:, :, cells == 0] = np.nan
    total = float(m.sum())
    stress = (
        (comoment * m[None, None]).sum(axis=(2, 3, 4)) / total
        if total > 0
        else np.full((c, c), np.nan)
    )
    return {"stress": stress, "profile": profile, "count": int(total)}


# --------------------------------------------------------------------------- #
# Divergence
# --------------------------------------------------------------------------- #


def divergence(
    fields: np.ndarray, stencil: np.ndarray, spacing: Sequence[float]
) -> dict[str, Any]:
    """Central-difference divergence ``du/dx + dv/dy + dw/dz`` on the cells of
    ``stencil`` (see :func:`stencil_fluid_mask`), with ``spacing = (dz, dy,
    dx)`` matching the ``(z, y, x)`` axis order and ``fields[:, 0:3] = (u, v,
    w)`` the x / y / z components.

    Returns ``{"rms": float, "mean_abs": float, "per_sample_rms": (N,),
    "sum_sq": float, "count": int}`` (``sum_sq`` / ``count`` merge across
    groups); nan when the stencil is empty.
    """
    f, _ = _check_fields(fields, stencil)
    if f.shape[1] < 3:
        raise ValueError("divergence needs the three velocity components (u, v, w)")
    s = np.asarray(stencil).astype(bool)
    dz, dy, dx = (float(v) for v in spacing)
    u, v, w = f[:, 0], f[:, 1], f[:, 2]
    div = np.zeros_like(u)
    inner = (slice(None), slice(1, -1), slice(1, -1), slice(1, -1))
    div[inner] = (
        (u[:, 1:-1, 1:-1, 2:] - u[:, 1:-1, 1:-1, :-2]) / (2 * dx)
        + (v[:, 1:-1, 2:, 1:-1] - v[:, 1:-1, :-2, 1:-1]) / (2 * dy)
        + (w[:, 2:, 1:-1, 1:-1] - w[:, :-2, 1:-1, 1:-1]) / (2 * dz)
    )
    count = int(s.sum())
    if count == 0:
        n = f.shape[0]
        return {
            "rms": float("nan"),
            "mean_abs": float("nan"),
            "per_sample_rms": np.full(n, np.nan),
            "sum_sq": 0.0,
            "count": 0,
        }
    vals = div[:, s]  # (N, count)
    sum_sq = float(np.sum(vals**2))
    return {
        "rms": float(np.sqrt(sum_sq / vals.size)),
        "mean_abs": float(np.mean(np.abs(vals))),
        "per_sample_rms": np.sqrt(np.mean(vals**2, axis=1)),
        "sum_sq": sum_sq,
        "count": int(vals.size),
    }


# --------------------------------------------------------------------------- #
# Diversity / energy
# --------------------------------------------------------------------------- #


def pairwise_rms_distance(fields: np.ndarray, fluid: np.ndarray) -> float:
    """Mean over all sample pairs of the fluid-cell RMS distance (over every
    component); nan with fewer than two samples."""
    f, m = _check_fields(fields, fluid)
    n = f.shape[0]
    if n < 2 or m.sum() == 0:
        return float("nan")
    vals = f[:, :, m].reshape(n, -1)
    dists = [
        float(np.sqrt(np.mean((vals[i] - vals[j]) ** 2)))
        for i in range(n)
        for j in range(i + 1, n)
    ]
    return float(np.mean(dists))


def diversity(samples: np.ndarray, fluid: np.ndarray) -> dict[str, Any]:
    """Spread across noise seeds under fixed conditioning.

    ``samples`` is ``(S, B, C, nz, ny, nx)``: ``S`` draws for each of ``B``
    conditionings. Returns ``{"per_condition": (B,), "mean": float}`` -- the
    mean pairwise RMS distance between the ``S`` draws of each conditioning,
    and its mean over conditionings. Compare with
    :func:`pairwise_rms_distance` of the real held-out set (the scale of real
    variability); a ratio near zero is mode collapse.
    """
    s = np.asarray(samples, dtype=np.float64)
    if s.ndim != 6:
        raise ValueError(f"samples must be (S, B, C, nz, ny, nx), got {s.shape}")
    per = np.array([pairwise_rms_distance(s[:, b], fluid) for b in range(s.shape[1])])
    return {"per_condition": per, "mean": nanmean(per)}


def kinetic_energy(fields: np.ndarray, fluid: np.ndarray) -> np.ndarray:
    """Per-sample mean kinetic energy ``0.5 * <|u|^2>`` over fluid cells, ``(N,)``."""
    f, m = _check_fields(fields, fluid)
    if m.sum() == 0:
        return np.full(f.shape[0], np.nan)
    return np.asarray(0.5 * (f[:, :, m] ** 2).sum(axis=1).mean(axis=1))


# --------------------------------------------------------------------------- #
# Padding sensitivity
# --------------------------------------------------------------------------- #


def padding_sensitivity(
    fields: np.ndarray,
    fluid: np.ndarray,
    spacing: Sequence[float],
    padded_axes: Sequence[bool],
    edge: int = 16,
) -> dict[str, Any] | None:
    """Divergence RMS and velocity RMS on the last ``edge`` cells of each padded
    axis versus the interior (everything else), or ``None`` when no axis is
    padded (the grid is a multiple of the AE's padding multiple).

    The AE pads each axis up to its crop / stride multiple and crops back; the
    band next to the padding is where a decoder artefact would show. Returns
    ``{axis: {"edge": {...}, "interior": {...}}}`` with ``rms_velocity`` and
    ``divergence_rms`` per region (``nan`` when a region has no stencil cell).
    """
    f, m = _check_fields(fields, fluid)
    axes = [bool(a) for a in padded_axes]
    if len(axes) != 3:
        raise ValueError("padded_axes must have one flag per (z, y, x) axis")
    if not any(axes):
        return None
    stencil = stencil_fluid_mask(m)
    out: dict[str, Any] = {}
    for ax, name in enumerate(("z", "y", "x")):
        if not axes[ax]:
            continue
        n_ax = m.shape[ax]
        band = np.zeros(m.shape, dtype=bool)
        sl: list[Any] = [slice(None)] * 3
        sl[ax] = slice(max(n_ax - int(edge), 0), n_ax)
        band[tuple(sl)] = True
        regions = {"edge": band, "interior": ~band}
        out[name] = {}
        for label, region in regions.items():
            fm = m & region
            st = stencil & region
            vel = (
                float(np.sqrt(np.mean(f[:, :, fm] ** 2))) if fm.any() else float("nan")
            )
            div = divergence(f, st, spacing)["rms"] if st.any() else float("nan")
            out[name][label] = {"rms_velocity": vel, "divergence_rms": div}
    return out


# --------------------------------------------------------------------------- #
# Merging across trajectory groups and the report
# --------------------------------------------------------------------------- #


def _weighted_profile(
    entries: list[dict[str, Any]], key: str, quadratic: bool = False
) -> np.ndarray:
    """Count-weighted merge of one profile key over groups.

    ``quadratic`` merges in the square (``sqrt(sum(w * p^2) / sum(w))``), which
    is what an RMS needs: it is the exact pooled value when the groups share a
    level mean and an over-estimate by exactly the between-group variance of
    those means otherwise -- whereas averaging the RMS values themselves is
    biased low for no reason.
    """
    num = None
    den = None
    for e in entries:
        p = np.asarray(e[key], dtype=np.float64)
        w = np.asarray(e["count"], dtype=np.float64)[None, :]
        ok = np.isfinite(p)
        term = np.where(ok, p**2 if quadratic else p, 0.0) * w
        num = term if num is None else num + term
        den = w * ok if den is None else den + w * ok
    assert num is not None and den is not None
    merged = np.where(den > 0, num / np.where(den > 0, den, 1.0), np.nan)
    return np.sqrt(merged) if quadratic else merged


def merge_group_metrics(groups: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Count-weighted merge of per-group metric dicts of ONE grid shape.

    Each group dict holds the outputs of :func:`profiles` (``"profiles"``),
    :func:`spectra` (``"spectra"``), :func:`reynolds_stresses`
    (``"reynolds"``), :func:`divergence` (``"divergence"``) and the sample
    count ``"n"``; sample-level arrays (``sample_mean``, ``per_sample_rms``)
    are concatenated. Profile ``rms`` merges in the square (see
    :func:`_weighted_profile`).
    """
    if not groups:
        raise ValueError("no groups to merge")
    prof = [g["profiles"] for g in groups]
    merged_prof = {
        "mean": _weighted_profile(prof, "mean"),
        "rms": _weighted_profile(prof, "rms", quadratic=True),
        "count": np.sum([np.asarray(p["count"]) for p in prof], axis=0),
        "sample_mean": np.concatenate([p["sample_mean"] for p in prof], axis=0),
    }
    spec = [g["spectra"] for g in groups]
    rows = np.array([s["rows"] for s in spec], dtype=np.float64)
    if rows.sum() > 0:
        energy = sum(s["energy"] * r for s, r in zip(spec, rows) if r > 0) / rows.sum()
    else:
        energy = np.asarray(spec[0]["energy"])
    merged_spec = {"k": spec[0]["k"], "energy": energy, "rows": int(rows.sum())}
    rey = [g["reynolds"] for g in groups]
    wts = np.array(
        [float(r["count"]) * (0.0 if np.isnan(r["stress"]).all() else 1.0) for r in rey]
    )
    if wts.sum() > 0:
        stress = sum(r["stress"] * w for r, w in zip(rey, wts) if w > 0) / wts.sum()
        profile = (
            sum(np.nan_to_num(r["profile"]) * w for r, w in zip(rey, wts) if w > 0)
            / wts.sum()
        )
    else:
        stress, profile = rey[0]["stress"], rey[0]["profile"]
    merged_rey = {
        "stress": stress,
        "profile": profile,
        "count": int(sum(r["count"] for r in rey)),
    }
    div = [g["divergence"] for g in groups]
    sum_sq = float(sum(d["sum_sq"] for d in div))
    count = int(sum(d["count"] for d in div))
    merged_div = {
        "rms": float(np.sqrt(sum_sq / count)) if count > 0 else float("nan"),
        "mean_abs": (
            float(np.nansum([d["mean_abs"] * d["count"] for d in div]) / count)
            if count > 0
            else float("nan")
        ),
        "per_sample_rms": np.concatenate([d["per_sample_rms"] for d in div]),
        "sum_sq": sum_sq,
        "count": count,
    }
    return {
        "profiles": merged_prof,
        "spectra": merged_spec,
        "reynolds": merged_rey,
        "divergence": merged_div,
        "n": int(sum(int(g["n"]) for g in groups)),
    }


def compare_to_real(source: dict[str, Any], real: dict[str, Any]) -> dict[str, float]:
    """Scalar distances of one source's merged metrics to the real set's."""
    lsd = log_spectral_distance(source["spectra"]["energy"], real["spectra"]["energy"])
    s, r = source["reynolds"]["stress"], real["reynolds"]["stress"]
    errs = [
        abs(float(s[i, j] - r[i, j]))
        for i, j in STRESS_PAIRS
        if i < s.shape[0] and j < s.shape[0]
    ]
    return {
        "profile_rmse": profile_rmse(source["profiles"], real["profiles"], "mean"),
        "rms_profile_rmse": profile_rmse(source["profiles"], real["profiles"], "rms"),
        "spectra_lsd_db": nanmean(lsd),
        "reynolds_abs_err": nanmean(errs),
        "divergence_rms": float(source["divergence"]["rms"]),
    }


def _is_within(value: float, bound: float) -> bool:
    return bool(np.isfinite(value) and np.isfinite(bound) and value <= bound)


def _nanmax(values: Sequence[float]) -> float:
    """Largest finite entry, ``nan`` when there is none (see :func:`nanmean`)."""
    finite = [v for v in values if np.isfinite(v)]
    return max(finite) if finite else float("nan")


def aggregate_report(
    scalars: dict[str, dict[str, float]],
    reference: dict[str, Any],
    tolerances: dict[str, float],
) -> dict[str, Any]:
    """The declared tolerance block and the PASS / FAIL verdict.

    ``scalars[source]`` holds the per-source numbers (``profile_rmse``, ``w1``,
    ``divergence_rms``, ``diversity`` where defined); ``reference`` the
    held-out scales (``bootstrap_profile_rmse``, ``bootstrap_w1``,
    ``real_divergence_rms``, ``real_pairwise_spread``). Each criterion bounds
    the ``generated`` source by ``factor x max(AE baseline, held-out
    variability)`` -- the AE reconstruction is the best the decoder can do and
    the bootstrap is the best ``N`` samples can resolve, so a generator inside
    both is indistinguishable from real at this sample size. A non-finite
    number on either side is a failure, never a pass.
    """
    gen = scalars.get("generated", {})
    ae = scalars.get("ae_recon", {})
    pf = float(tolerances.get("profile_rmse_factor", 2.0))
    wf = float(tolerances.get("w1_factor", 2.0))
    df = float(tolerances.get("divergence_factor", 3.0))
    dr = float(tolerances.get("diversity_min_ratio", 0.25))

    def _nan(x: Any) -> float:
        return float("nan") if x is None else float(x)

    bounds = {
        "profile_rmse": pf
        * _nanmax(
            [
                _nan(ae.get("profile_rmse")),
                _nan(reference.get("bootstrap_profile_rmse")),
            ]
        ),
        "w1": wf * _nanmax([_nan(ae.get("w1")), _nan(reference.get("bootstrap_w1"))]),
        "divergence_rms": df
        * _nanmax(
            [_nan(ae.get("divergence_rms")), _nan(reference.get("real_divergence_rms"))]
        ),
    }
    spread = _nan(reference.get("real_pairwise_spread"))
    ratio = (
        _nan(gen.get("diversity")) / spread
        if spread and np.isfinite(spread)
        else float("nan")
    )
    checks: dict[str, dict[str, Any]] = {
        "profile_rmse": {
            "value": _nan(gen.get("profile_rmse")),
            "bound": float(bounds["profile_rmse"]),
            "rule": f"generated profile RMSE <= {pf} x max(AE profile RMSE, bootstrap profile RMSE)",
        },
        "w1": {
            "value": _nan(gen.get("w1")),
            "bound": float(bounds["w1"]),
            "rule": f"generated mean W1 <= {wf} x max(AE W1, bootstrap W1)",
        },
        "divergence_rms": {
            "value": _nan(gen.get("divergence_rms")),
            "bound": float(bounds["divergence_rms"]),
            "rule": f"generated divergence RMS <= {df} x max(AE, real divergence RMS)",
        },
        "diversity_ratio": {
            "value": float(ratio),
            "bound": dr,
            "rule": f"generated diversity / real pairwise spread >= {dr}",
        },
    }
    failures: list[str] = []
    for name, chk in checks.items():
        ok = (
            bool(np.isfinite(chk["value"]) and chk["value"] >= chk["bound"])
            if name == "diversity_ratio"
            else _is_within(chk["value"], chk["bound"])
        )
        chk["passed"] = ok
        if not ok:
            failures.append(
                f"{name}: {chk['value']:.4g} vs bound {chk['bound']:.4g} ({chk['rule']})"
            )
    return {
        "tolerances": {
            "profile_rmse_factor": pf,
            "w1_factor": wf,
            "divergence_factor": df,
            "diversity_min_ratio": dr,
        },
        "checks": checks,
        "acceptance": {"passed": not failures, "failures": failures},
    }
