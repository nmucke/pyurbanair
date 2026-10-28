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


# --------------------------------------------------------------------------- #
# Latent spectrum (the one metric on latents, not physical fields)
# --------------------------------------------------------------------------- #

# Upper ends (1-based, inclusive) of the eigen-bands of :func:`latent_spectrum`;
# the last band always ends at D. Chosen for the Tadpole latents (S: D = 768,
# B: D = 1536): the top components, the variance plateau, and the tail.
LATENT_BANDS: tuple[int, ...] = (32, 256, 768, 1024)


def latent_moments(z: np.ndarray) -> dict[str, Any]:
    """Float64 first and second moments of the tokens of latents ``(B, D, ...)``
    (every grid position of every sample is one ``D``-vector); add them up over
    batches with :func:`merge_latent_moments`."""
    x = np.asarray(z, dtype=np.float64)
    tokens = np.moveaxis(x, 1, -1).reshape(-1, x.shape[1])
    return {"n": len(tokens), "sum": tokens.sum(0), "outer": tokens.T @ tokens}


def merge_latent_moments(parts: Sequence[dict[str, Any]]) -> dict[str, Any]:
    return {k: sum(p[k] for p in parts) for k in ("n", "sum", "outer")}


def _covariance(moments: dict[str, Any]) -> np.ndarray:
    mean = moments["sum"] / moments["n"]
    return np.asarray(moments["outer"] / moments["n"] - np.outer(mean, mean))


def latent_spectrum(
    real: dict[str, Any],
    generated: dict[str, Any],
    bands: Sequence[int] = LATENT_BANDS,
) -> dict[str, Any]:
    """Variance of generated latents in the PCA basis of the real ones.

    ``real`` / ``generated`` are :func:`latent_moments`. The basis is the
    eigenvectors of the real (centred) token covariance, in descending
    eigenvalue order; each covariance is centred on its own mean. Returns the
    real eigenvalues ``"real"`` (D,), the generated variance along the same
    directions ``"generated"`` (D,), and per band (components ``lo``-``hi``,
    1-based) the geometric mean of the per-direction ratio generated / real
    (``"bands"``: ``"lo_hi"`` -> ratio; 1 = the right variance, > 1 = too
    noisy). The geometric mean keeps the near-empty tail directions (real
    variance down to ~1e-7) from being swamped by the band's largest ones.
    """
    eig, basis = np.linalg.eigh(_covariance(real))
    order = np.argsort(eig)[::-1]
    eig, basis = eig[order], basis[:, order]
    gen = np.einsum("dk,de,ek->k", basis, _covariance(generated), basis)
    d = len(eig)
    floor = max(float(eig[0]), 1e-30) * 1e-12  # round-off floor of the ratio
    log_ratio = np.log(np.maximum(gen, floor)) - np.log(np.maximum(eig, floor))
    edges = [0] + [b for b in bands if b < d] + [d]
    ratios = {
        f"{lo + 1}_{hi}": float(np.exp(log_ratio[lo:hi].mean()))
        for lo, hi in zip(edges[:-1], edges[1:])
    }
    return {"real": eig, "generated": gen, "bands": ratios}


# --------------------------------------------------------------------------- #
# Seed ensembles: many draws of ONE conditioning against the solver's own spread
# --------------------------------------------------------------------------- #
#
# ``scripts/surrogate/evaluate_latent_generator_ensemble.py`` compares, per
# geometry, a source ensemble ``S`` (generated, AE reconstruction, ...) with the
# solver's seed ensemble ``R`` at fixed geometry and forcing. Every estimator
# here is unbiased (expected 0 or 1 when ``S`` and ``R`` come from the same
# distribution) or meant to be read against a split-half floor of ``R``
# (:func:`split_half_floor`). A source whose member ``i`` is derived from real
# member ``i`` (the AE reconstruction) is *paired*: the functions take
# ``exclude_self`` / ``paired`` / ``diff_var`` so member ``i`` is never scored
# against its own truth.


def ensemble_moments(fields: np.ndarray, fluid: np.ndarray) -> dict[str, Any]:
    """Pointwise ensemble mean and unbiased (``ddof = 1``) variance over the
    ``N`` samples at every fluid cell.

    Returns ``{"mean": (C, M), "var": (C, M), "n": N, "z": (M,), "nz": nz}``
    with ``M`` the fluid cells in C order and ``z`` the level of each (for the
    per-height reductions); ``var`` is nan with fewer than two samples.
    Accumulates in float64 one component at a time, so a float32 stack is
    never copied whole.
    """
    f = np.asarray(fields)
    m = np.asarray(fluid).astype(bool)
    if f.ndim != 5 or m.shape != f.shape[2:]:
        raise ValueError(
            f"fields must be (N, C, *fluid.shape), got {f.shape} for {m.shape}"
        )
    n, c = f.shape[0], f.shape[1]
    cells = int(m.sum())
    mean = np.full((c, cells), np.nan)
    var = np.full((c, cells), np.nan)
    for i in range(c):
        x = f[:, i][:, m].astype(np.float64)  # (N, M)
        if n > 0:
            mean[i] = x.mean(axis=0)
        if n > 1:
            var[i] = x.var(axis=0, ddof=1)
    return {"mean": mean, "var": var, "n": n, "z": np.nonzero(m)[0], "nz": m.shape[0]}


def _level_sum(values: np.ndarray, z: np.ndarray, nz: int) -> np.ndarray:
    """Per-level sums ``(C, nz)`` of per-cell ``values`` ``(C, M)``."""
    return np.stack([np.bincount(z, weights=v, minlength=nz) for v in values])


def _level_mean(values: np.ndarray, z: np.ndarray, nz: int) -> np.ndarray:
    count = np.bincount(z, minlength=nz).astype(np.float64)
    out = _level_sum(values, z, nz) / np.where(count > 0, count, 1.0)
    out[:, count == 0] = np.nan
    return out


def mean_bias(
    source: dict[str, Any],
    real: dict[str, Any],
    diff_var: np.ndarray | None = None,
) -> dict[str, Any]:
    """Squared bias of the ensemble mean, raw and debiased.

    ``source`` / ``real`` are :func:`ensemble_moments`. ``raw = mean_cells
    (mu_S - mu_R)^2``; its sampling noise ``mean_cells (var_S / n_S + var_R /
    n_R)`` is subtracted for ``debiased`` (expected 0 when the means agree).
    A paired source passes ``diff_var``, the per-cell variance ``(C, M)`` of
    the member differences ``S_i - R_i``, whose mean has noise ``diff_var /
    n``. ``standardised`` divides by ``mean_cells var_R``: the bias in units of
    the solver's own spread. Returns ``(C,)`` arrays ``raw``, ``debiased``,
    ``standardised`` and their per-level ``(C, nz)`` ``profile`` (debiased)
    and ``profile_standardised``.
    """
    d2 = (source["mean"] - real["mean"]) ** 2
    if diff_var is None:
        noise = source["var"] / source["n"] + real["var"] / real["n"]
    else:
        noise = np.asarray(diff_var) / real["n"]
    z, nz = real["z"], real["nz"]
    scale = real["var"].mean(axis=1)
    debiased = d2.mean(axis=1) - noise.mean(axis=1)
    profile = _level_mean(d2 - noise, z, nz)
    with np.errstate(invalid="ignore", divide="ignore"):
        return {
            "raw": d2.mean(axis=1),
            "debiased": debiased,
            "standardised": debiased / scale,
            "profile": profile,
            "profile_standardised": profile / _level_mean(real["var"], z, nz),
        }


def spread(source: dict[str, Any], real: dict[str, Any]) -> dict[str, Any]:
    """Ensemble spread of ``source`` against ``real`` (both
    :func:`ensemble_moments`).

    ``ratio = sqrt(sum var_S / sum var_R)`` per component ``(C,)``, per level
    (``ratio_profile`` ``(C, nz)``) and over every component (``ratio_total``);
    1 = the solver's spread, < 1 under-dispersed. The spatial pattern: the
    Pearson correlation of the ``sigma_S`` and ``sigma_R`` maps over cells
    (``map_corr``) and the RMS of ``log(sigma_S / sigma_R)`` (``log_ratio_rms``;
    > 0 even for equal distributions at finite ``n``, so read it against the
    split-half floor). Both skip the cells where ``sigma_R = 0`` (a solver
    ensemble can be exactly constant in an enclosed courtyard; their count per
    component is ``excluded``), and the log ratio also those with ``sigma_S =
    0``.
    """
    vs, vr = source["var"], real["var"]
    z, nz = real["z"], real["nz"]
    ss, sr = np.sqrt(np.maximum(vs, 0.0)), np.sqrt(np.maximum(vr, 0.0))
    c = vs.shape[0]
    corr = np.full(c, np.nan)
    log_rms = np.full(c, np.nan)
    varying = sr > 0
    for i in range(c):
        a, b = ss[i][varying[i]], sr[i][varying[i]]
        if a.size > 1 and np.std(a) > 0 and np.std(b) > 0:
            corr[i] = float(np.corrcoef(a, b)[0, 1])
        ok = (ss[i] > 0) & varying[i]
        if ok.any():
            log_rms[i] = float(np.sqrt(np.mean(np.log(ss[i][ok] / sr[i][ok]) ** 2)))
    with np.errstate(invalid="ignore", divide="ignore"):
        prof = _level_sum(vs, z, nz) / _level_sum(vr, z, nz)
        prof[:, np.bincount(z, minlength=nz) == 0] = np.nan
        return {
            "ratio": np.sqrt(vs.sum(axis=1) / vr.sum(axis=1)),
            "ratio_total": float(np.sqrt(vs.sum() / vr.sum())),
            "ratio_profile": np.sqrt(prof),
            "map_corr": corr,
            "log_ratio_rms": log_rms,
            "excluded": (~varying).sum(axis=1),
        }


def select_points(fluid: np.ndarray, num_points: int, seed: int = 0) -> np.ndarray:
    """A fixed, deterministic set of ``num_points`` cells of the mask
    ``fluid`` (sorted flat indices into ``fluid.ravel()``, all of them when
    there are fewer): the same ``seed`` and mask give the same points for
    every source and model. Pass the fluid cells where the real ensemble
    varies, so no point has a constant truth."""
    cells = np.flatnonzero(np.asarray(fluid).astype(bool).ravel())
    if len(cells) <= int(num_points):
        return cells
    rng = np.random.default_rng(int(seed))
    return np.sort(rng.choice(cells, size=int(num_points), replace=False))


def point_values(fields: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Values ``(N, C, P)`` (float64) of ``fields`` at :func:`select_points`."""
    f = np.asarray(fields)
    return f.reshape(f.shape[0], f.shape[1], -1)[:, :, points].astype(np.float64)


def _check_self(ensemble: np.ndarray, truth: np.ndarray) -> None:
    if len(ensemble) != len(truth):
        raise ValueError(
            "exclude_self pairs ensemble member i with truth i: got "
            f"{len(ensemble)} members for {len(truth)} truths"
        )


def fair_crps(
    ensemble: np.ndarray, truth: np.ndarray, exclude_self: bool = False
) -> np.ndarray:
    """Fair (ensemble-size unbiased) CRPS of an ensemble ``(m, C, P)`` against
    each truth ``(n, C, P)``, averaged over truths and points: ``(C,)``.

    ``CRPS_fair(X, y) = mean_i |x_i - y| - sum_{i != j} |x_i - x_j| / (2 m (m
    - 1))``. Its expectation does not depend on ``m``, so ensembles of
    different sizes compare directly. ``exclude_self`` scores truth ``r``
    against the ensemble without member ``r`` (``m = n``): the leave-one-out
    oracle ``fair_crps(R, R, exclude_self=True)`` -- the score of the solver's
    own ensemble -- and the paired AE reconstruction.
    """
    x = np.asarray(ensemble, dtype=np.float64)
    y = np.asarray(truth, dtype=np.float64)
    if exclude_self:
        _check_self(x, y)
    m = len(x) - (1 if exclude_self else 0)
    if m < 2:
        return np.full(x.shape[1], np.nan)
    # |x_r - x_j| summed over j, per member r; their sum is the full pair sum.
    row = np.stack([np.abs(x - x[r]).sum(axis=0) for r in range(len(x))])
    pair_sum = row.sum(axis=0)  # (C, P), each unordered pair counted twice
    scores = []
    for r in range(len(y)):
        skill = np.abs(x - y[r]).sum(axis=0)
        pairs = pair_sum
        if exclude_self:
            skill = skill - np.abs(x[r] - y[r])
            pairs = pair_sum - 2.0 * row[r]
        scores.append(skill / m - pairs / (2.0 * m * (m - 1)))
    return np.asarray(np.mean(scores, axis=(0, 2)))


def rank_histogram(
    ensemble: np.ndarray,
    truth: np.ndarray,
    bins: int = 8,
    exclude_self: bool = False,
    seed: int = 0,
) -> dict[str, Any]:
    """Rank (Talagrand) histogram of each truth ``(n, C, P)`` within the
    ensemble ``(m, C, P)``, coarsened to ``bins`` equal bins per component.

    The rank is randomised (``u = (#{x < y} + V (#{x = y} + 1)) / (m + 1)``,
    ``V ~ U(0, 1)``), so a calibrated ensemble gives exactly uniform ``u`` and
    flat bins whatever ``m``. ``exclude_self`` as in :func:`fair_crps`
    (``rank_histogram(R, R, exclude_self=True)`` is the solver's own, flat in
    expectation). A U shape is under-dispersion, a hump over-dispersion, a
    slope a bias. Returns ``counts`` ``(C, bins)``, ``density`` (``counts``
    over the flat count: 1 = flat), ``chi2_per_bin`` ``(C,)`` (``sum (h - e)^2
    / e / bins``; correlated points inflate it, so read it against the
    real one) and ``outer_mass`` ``(C,)`` (the two outer bins' share; flat =
    ``2 / bins``).
    """
    x = np.asarray(ensemble, dtype=np.float64)
    y = np.asarray(truth, dtype=np.float64)
    if exclude_self:
        _check_self(x, y)
    rng = np.random.default_rng(int(seed))
    m = len(x) - (1 if exclude_self else 0)
    c = x.shape[1]
    counts = np.zeros((c, int(bins)))
    for r in range(len(y)):
        less = (x < y[r]).sum(axis=0)
        equal = (x == y[r]).sum(axis=0)
        if exclude_self:  # x[r] vs y[r]: one comparison to undo
            less = less - (x[r] < y[r])
            equal = equal - (x[r] == y[r])
        u = (less + rng.random(less.shape) * (equal + 1)) / (m + 1)
        b = np.minimum((u * bins).astype(int), int(bins) - 1)
        for i in range(c):
            counts[i] += np.bincount(b[i], minlength=int(bins))
    expected = counts.sum(axis=1, keepdims=True) / bins
    with np.errstate(invalid="ignore", divide="ignore"):
        return {
            "counts": counts,
            "density": counts / expected,
            "chi2_per_bin": ((counts - expected) ** 2 / expected).sum(axis=1) / bins,
            "outer_mass": (counts[:, 0] + counts[:, -1]) / counts.sum(axis=1),
        }


def rms_distances(
    stacks: Sequence[np.ndarray], fluid: np.ndarray, chunk_cells: int = 65536
) -> np.ndarray:
    """RMS distance (over every component and fluid cell) between every pair
    of samples of the concatenated ``stacks`` (each ``(N_k, C, nz, ny, nx)``):
    a symmetric ``(sum N_k, sum N_k)`` matrix with a zero diagonal.

    Built from a float64 Gram matrix accumulated over chunks of
    ``chunk_cells`` fluid cells, each centred on its column mean (distances
    are translation invariant; centring removes the cancellation of
    ``|a|^2 + |b|^2 - 2 a.b`` on fields with a large mean). One pass serves
    every energy distance and diversity of a geometry (:func:`energy_distance`).
    """
    m = np.asarray(fluid).astype(bool)
    cells = np.flatnonzero(m.ravel())
    flat = [np.asarray(s).reshape(len(s), s.shape[1], -1) for s in stacks]
    n = sum(len(s) for s in flat)
    c = flat[0].shape[1]
    gram = np.zeros((n, n))
    for start in range(0, len(cells), int(chunk_cells)):
        cols = cells[start : start + int(chunk_cells)]
        x = np.concatenate([s[:, :, cols].reshape(len(s), -1) for s in flat]).astype(
            np.float64
        )
        x -= x.mean(axis=0)
        gram += x @ x.T
    sq = np.diag(gram)
    d2 = np.maximum(sq[:, None] + sq[None, :] - 2.0 * gram, 0.0)
    np.fill_diagonal(d2, 0.0)
    return np.asarray(np.sqrt(d2 / max(c * len(cells), 1)))


def _mean_pairwise(d: np.ndarray) -> float:
    n = len(d)
    return float(d.sum() / (n * (n - 1))) if n > 1 else float("nan")


def energy_distance(
    distances: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    paired: bool = False,
) -> dict[str, float]:
    """Unbiased (U-statistic) energy distance between the samples ``x`` and
    ``y`` (index arrays into the :func:`rms_distances` matrix).

    ``ED = 2 E|X - Y| - E|X - X'| - E|Y - Y'|``, each within-set mean over the
    ``i != j`` pairs, so its expectation is exactly 0 when both sets come from
    one distribution (an estimate can be slightly negative). ``paired`` (``x_i``
    derived from ``y_i``, equal sizes) drops the ``i = j`` cross pairs, which
    are not independent draws. Returns ``energy_distance`` (in the field's
    units) and ``relative`` (over ``E|Y - Y'|``, the mean pairwise distance of
    ``y``, the reference set).
    """
    d = np.asarray(distances)
    x, y = np.asarray(x), np.asarray(y)
    cross = d[np.ix_(x, y)]
    if paired:
        if len(x) != len(y):
            raise ValueError("paired sets must have equal sizes")
        cross_mean = _mean_pairwise(cross - np.diag(np.diag(cross)))
    else:
        cross_mean = float(cross.mean())
    within_y = _mean_pairwise(d[np.ix_(y, y)])
    ed = 2.0 * cross_mean - _mean_pairwise(d[np.ix_(x, x)]) - within_y
    return {"energy_distance": ed, "relative": ed / within_y}


def pointwise_w1(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """1-Wasserstein distance between the marginals of two ensembles at each
    point (``(n_a, C, P)``, ``(n_b, C, P)``), averaged over points: ``(C,)``.
    Equal sizes use the sorted-sample identity ``mean |a_(k) - b_(k)|``."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if len(a) == len(b):
        return np.asarray(
            np.abs(np.sort(a, axis=0) - np.sort(b, axis=0)).mean(axis=(0, 2))
        )
    return np.array(
        [
            np.mean([wasserstein_1(a[:, c, p], b[:, c, p]) for p in range(a.shape[2])])
            for c in range(a.shape[1])
        ]
    )


def split_half_floor(
    metric: Any,
    n_real: int,
    num_splits: int = 16,
    seed: int = 0,
    n_source: int | None = None,
    paired: bool = False,
) -> dict[str, dict[str, Any]]:
    """The finite-sample floor of a metric: its value between disjoint random
    halves of the real ensemble, and a source's value at the same sizes.

    Per split, the ``n_real`` real indices are permuted into halves ``a`` and
    ``b`` of ``n_real // 2``. ``metric(from_source, x, b)`` returns a dict of
    numbers / arrays for the set ``x`` (real indices when ``from_source`` is
    False, source indices otherwise) against the real half ``b``. The floor is
    ``metric(False, a, b)``; with ``n_source`` the matched value is
    ``metric(True, s, b)``, ``s`` a random half-size subset of the source --
    or ``a`` itself when ``paired`` (source ``i`` derived from real ``i``, so
    ``s`` stays independent of ``b``). Returns ``floor`` / ``floor_std`` (and
    ``matched`` / ``matched_std``): the mean and standard deviation over the
    splits, per key. The source subsets come from their own generator, so the
    splits depend only on ``seed`` and ``n_real``: every source (paired or not)
    is scored on the same splits as the floor.
    """
    rng = np.random.default_rng(int(seed))
    pick = np.random.default_rng([int(seed), 1])
    half = int(n_real) // 2
    floor: list[dict[str, Any]] = []
    matched: list[dict[str, Any]] = []
    for _ in range(int(num_splits)):
        perm = rng.permutation(int(n_real))
        a, b = np.sort(perm[:half]), np.sort(perm[half : 2 * half])
        floor.append(metric(False, a, b))
        if n_source is not None:
            s = (
                a
                if paired
                else np.sort(
                    pick.choice(int(n_source), min(half, int(n_source)), replace=False)
                )
            )
            matched.append(metric(True, s, b))

    def stats(values: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
        keys = values[0].keys()
        stacked = {
            k: np.asarray([v[k] for v in values], dtype=np.float64) for k in keys
        }
        return {
            "mean": {k: v.mean(axis=0) for k, v in stacked.items()},
            "std": {k: v.std(axis=0) for k, v in stacked.items()},
        }

    out: dict[str, dict[str, Any]] = {}
    if floor:
        s = stats(floor)
        out["floor"], out["floor_std"] = s["mean"], s["std"]
    if matched:
        s = stats(matched)
        out["matched"], out["matched_std"] = s["mean"], s["std"]
    return out


def bootstrap_ci(
    values: Sequence[float],
    n_resamples: int = 2000,
    seed: int = 0,
    level: float = 0.95,
) -> dict[str, float]:
    """Mean over the finite ``values`` (e.g. one per geometry) and its
    percentile bootstrap confidence interval: ``{"mean", "lo", "hi", "n"}``
    (nan bounds with fewer than two values)."""
    v = np.asarray(values, dtype=np.float64).ravel()
    v = v[np.isfinite(v)]
    if v.size == 0:
        return {"mean": float("nan"), "lo": float("nan"), "hi": float("nan"), "n": 0}
    if v.size < 2:
        x = float(v[0])
        return {"mean": x, "lo": float("nan"), "hi": float("nan"), "n": 1}
    rng = np.random.default_rng(int(seed))
    means = v[rng.integers(0, v.size, size=(int(n_resamples), v.size))].mean(axis=1)
    tail = 50.0 * (1.0 - float(level))
    lo, hi = np.percentile(means, [tail, 100.0 - tail])
    return {"mean": float(v.mean()), "lo": float(lo), "hi": float(hi), "n": int(v.size)}


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
