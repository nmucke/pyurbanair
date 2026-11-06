"""Stage 2 of the nested design: a streamed ensemble smoother on saved fields.

``docs/research/multiscale_nested_da.tex``, Section 3. After the ESMDA has
estimated the parameters and the inflow, every saved posterior frame is updated
with the raw sensor values of the window, with no model integration:
:func:`field_analysis` over any frame stream, :func:`field_analysis_netcdf` on
the ensemble NetCDF file ``_BaseESMDA.save_state`` writes in either save mode.

**Within-parameter gain.** The members carry their own posterior parameters, so
a cross-member gain would hold ``Cov(g(theta, x) | y)`` and use ``y`` a second
time on the between-member part. ``P`` projects out ``[1, parameter
anomalies]``, leaving ``nu = J - 1 - p`` degrees of freedom. With ``B`` an
orthonormal basis of ``range(P)``, ``Y`` the centred predictions and
``Y_w = R^(-1/2) Y``::

    W = I + P Y^T (Y P Y^T / nu + R)^-1 D / nu
      = I + B M^-1 B^T Y_w^T D_w,            M = nu I + B^T Y_w^T Y_w B.

``M`` is ``nu x nu`` whatever the number of observations. Columns of ``W`` sum
to 1, because ``B`` is orthogonal to the ones.

**Deterministic transform.** No observation perturbations ``E`` are drawn.
``D = (y - ybar) 1^T - Y_g``, with ``Y_g = Y (I - B B^T)`` the between-member
part of the predictions, moves each member to its own conditional mean, and
the symmetric square root replaces the perturbed-observation anomaly update::

    W = I + B (sqrt(nu) M^(-1/2) - I) B^T + B M^-1 B^T Y_w^T D_w.

Given the sample moments this has exactly the mean of the perturbed form and,
exactly, the covariance that form has in expectation: the within-parameter
anomalies get the Kalman covariance, ``M^-1`` in ensemble coordinates. It is
therefore not an LETKF, whose anomalies keep the between-member part and whose
mean weights are one vector for all members. Being deterministic, neighbouring
cells cannot disagree through a sampling draw.

**Local form.** Observation ``o`` enters cell ``r`` at frame ``t`` with variance
``R_o / rho``, ``rho = GC(|r - r_o|, L_r) GC(|t - t_o|, L_t)``, the Gaspari-Cohn
taper, zero from ``L_r`` and ``L_t`` on. ``rho`` separates, so per frame ``M`` and
``B^T Y_w^T D_w`` are sums over the sensors of small per-sensor matrices, and
each cell costs one matrix product and one ``nu x nu`` eigendecomposition.
Rows with no local observation are never touched. With local ``W`` the affine
constraints all members satisfy are lost unless they hold row by row: a row
equal across members (``u = 0`` in a building) stays exact, divergence does not.
"""

import logging
import math
import pathlib
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from typing import Optional

import jax
import jax.numpy as jnp
import netCDF4
import numpy as np
import xarray
from data_assimilation.augmentation import StateAugmentation
from data_assimilation.filtering.analysis import validate_variances
from data_assimilation.localization.base import gaspari_cohn
from data_assimilation.reduction import _numerical_rank
from jax.experimental import enable_x64

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class FieldAnalysisInfo:
    """What one :func:`field_analysis` call did, for ``run_info.yaml``.

    Attributes:
        num_members: Members in the statistics (``member_mask`` true).
        num_parameters: ``p``, the numerical rank of the parameter anomalies.
        degrees_of_freedom: ``nu = num_members - 1 - p``.
        num_frames: Frames read.
        num_updated_frames: Frames with a local observation (written).
        updated_row_fraction: Share of the frames' rows that were updated.
    """

    num_members: int
    num_parameters: int
    degrees_of_freedom: int
    num_frames: int
    num_updated_frames: int
    updated_row_fraction: float


def within_parameter_basis(params: np.ndarray) -> np.ndarray:
    """Orthonormal basis ``B``, ``(J, nu)``, of the range of ``P``.

    ``params``: ``(N_p, J)`` analysis-space rows of the members in the
    statistics. Constant rows carry nothing and are dropped; the others are
    standardised, so units do not decide the rank, which is the numerical rank
    at the precision of ``params``.
    """
    rows = np.asarray(params)
    num_members = rows.shape[1]
    wide = rows.astype(float)
    anomalies = wide - wide.mean(axis=1, keepdims=True)
    std = anomalies.std(axis=1)
    floor = np.finfo(rows.dtype).eps * np.abs(wide).max(axis=1, initial=0.0)
    varying = std > floor
    columns = [np.full((num_members, 1), num_members**-0.5)]
    if varying.any():
        standard = anomalies[varying] / std[varying, None]
        _, spectrum, right = np.linalg.svd(standard, full_matrices=False)
        rank = _numerical_rank(spectrum.astype(rows.dtype), standard.shape)
        columns.append(right[:rank].T)
    spanned = np.concatenate(columns, axis=1)
    complete, _ = np.linalg.qr(spanned, mode="complete")
    return np.asarray(complete[:, spanned.shape[1] :])


@jax.jit
def _coordinates(
    taper: jnp.ndarray,
    gram: jnp.ndarray,
    cross: jnp.ndarray,
    basis: jnp.ndarray,
) -> jnp.ndarray:
    """``(W - I)`` per taper row, as ``B`` times the returned ``(c, nu, J)``.

    ``taper``: ``(c, S)`` weight of each sensor. ``gram``: ``(S, nu, nu)``,
    ``cross``: ``(S, nu, J)``, the per-sensor sums of ``B^T Y_w^T Y_w B`` and
    ``B^T Y_w^T D_w``.
    """
    nu = basis.shape[1]
    system = jnp.tensordot(taper, gram, axes=1) + nu * jnp.eye(nu, dtype=gram.dtype)
    values, vectors = jnp.linalg.eigh(system)
    root = (vectors * (jnp.sqrt(nu / values) - 1.0)[:, None, :]) @ jnp.swapaxes(
        vectors, 1, 2
    )
    inverse = (vectors / values[:, None, :]) @ jnp.swapaxes(vectors, 1, 2)
    return root @ basis.T + inverse @ jnp.tensordot(taper, cross, axes=1)


@jax.jit
def _local_update(
    rows: jnp.ndarray,
    taper: jnp.ndarray,
    gram: jnp.ndarray,
    cross: jnp.ndarray,
    basis: jnp.ndarray,
) -> jnp.ndarray:
    """One chunk of rows ``(c, J)``, each with its own weights."""
    anomalies = rows - jnp.mean(rows, axis=1, keepdims=True)
    coordinates = _coordinates(taper, gram, cross, basis)
    return rows + jnp.einsum("ck,ckj->cj", anomalies @ basis, coordinates)


class _Analysis:
    """Observation-space quantities of one window, in ensemble coordinates."""

    def __init__(
        self,
        pred_obs: np.ndarray,
        obs: np.ndarray,
        obs_var: np.ndarray,
        params: np.ndarray,
        member_mask: Optional[np.ndarray],
        min_dof: int,
    ) -> None:
        pred_obs = np.asarray(pred_obs, dtype=float)
        obs = np.asarray(obs, dtype=float)
        if pred_obs.ndim != 2 or obs.shape != pred_obs.shape[:1]:
            raise ValueError(
                f"pred_obs must be (N_o, J) and obs (N_o,), got {pred_obs.shape} "
                f"and {obs.shape}."
            )
        total = pred_obs.shape[1]
        members = (
            np.ones(total, dtype=bool)
            if member_mask is None
            else np.asarray(member_mask, dtype=bool)
        )
        params = np.asarray(params)
        if members.shape != (total,) or params.ndim != 2 or params.shape[1] != total:
            raise ValueError(
                f"member_mask must be ({total},) and params (N_p, {total}), got "
                f"{members.shape} and {params.shape}."
            )
        with enable_x64():  # float32 would round the variances
            variances = np.asarray(validate_variances(jnp.asarray(obs_var)))
        if variances.shape != obs.shape:
            raise ValueError(f"obs_var must be {obs.shape}, got {variances.shape}.")
        predictions = pred_obs[:, members]
        if not (np.isfinite(predictions).all() and np.isfinite(obs).all()):
            raise FloatingPointError(
                "Non-finite observations or predictions of unmasked members."
            )

        self.members = members
        self.basis = within_parameter_basis(params[:, members])
        self.num_members = int(members.sum())
        self.dof = self.basis.shape[1]
        self.num_parameters = self.num_members - 1 - self.dof
        if self.dof < min_dof:
            raise ValueError(
                f"The within-parameter gain has nu = {self.dof} degrees of freedom "
                f"(J = {self.num_members}, p = {self.num_parameters}), below "
                f"min_dof = {min_dof}: use more members or estimate fewer "
                "parameters."
            )

        mean = predictions.mean(axis=1, keepdims=True)
        anomalies = predictions - mean
        within = anomalies @ self.basis  # (N_o, nu)
        between = anomalies - within @ self.basis.T
        scale = variances**-0.5
        self.whitened = scale[:, None] * within
        self.innovation = scale[:, None] * ((obs[:, None] - mean) - between)

    def sensor_sums(
        self, local: np.ndarray, weights: np.ndarray, sensors: np.ndarray, count: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """Per-sensor sums of the Gram and cross matrices.

        ``local`` indexes the observations, ``weights`` their temporal tapers
        and ``sensors`` their sensors, among ``count``.
        """
        gram = np.zeros((count, self.dof, self.dof))
        cross = np.zeros((count, self.dof, self.num_members))
        for sensor in np.unique(sensors):
            chosen = local[sensors == sensor]
            tapered = weights[sensors == sensor, None] * self.whitened[chosen]
            gram[sensor] = tapered.T @ self.whitened[chosen]
            cross[sensor] = tapered.T @ self.innovation[chosen]
        return gram, cross

    def global_weights(self) -> np.ndarray:
        """The dense ``W``, ``(J_active, J_active)``, of Eq. (11)."""
        num_obs = self.whitened.shape[0]
        gram, cross = self.sensor_sums(
            np.arange(num_obs), np.ones(num_obs), np.zeros(num_obs, dtype=int), 1
        )
        with enable_x64():
            coordinates = _coordinates(
                jnp.ones((1, 1)),
                jnp.asarray(gram),
                jnp.asarray(cross),
                jnp.asarray(self.basis),
            )
            coordinates = np.asarray(coordinates[0])
        return np.eye(self.num_members) + self.basis @ coordinates


def global_field_weights(
    pred_obs: np.ndarray,
    obs: np.ndarray,
    obs_var: np.ndarray,
    params: np.ndarray,
    member_mask: Optional[np.ndarray] = None,
    min_dof: int = 5,
) -> np.ndarray:
    """The global ``W`` of Eq. (11), over the unmasked members.

    Arguments as in :func:`field_analysis`. ``X_a = X_f W``.
    """
    return _Analysis(
        pred_obs, obs, obs_var, params, member_mask, min_dof
    ).global_weights()


def field_analysis(
    frames: Iterable[tuple[float, np.ndarray]],
    write: Callable[[int, np.ndarray], None],
    *,
    pred_obs: np.ndarray,
    obs: np.ndarray,
    obs_var: np.ndarray,
    obs_time: np.ndarray,
    obs_coords: np.ndarray,
    params: np.ndarray,
    row_coords: Optional[np.ndarray] = None,
    member_mask: Optional[np.ndarray] = None,
    L_r: Optional[float] = None,
    L_t: Optional[float] = None,
    min_dof: int = 5,
    chunk_size: int = 4096,
) -> FieldAnalysisInfo:
    """Update a window's frames one at a time, Eqs. (11) and (12) of the note.

    Args:
        frames: Yields ``(time, values)`` per frame, ``values`` ``(N_rows, J)``
            with the members in the order of ``pred_obs``. One frame is held at
            a time. Floating dtype; the updated frame keeps it.
        write: ``write(frame_index, values)`` for each frame with a local
            observation; other frames are not written.
        pred_obs: ``(N_o, J)`` predicted raw sensor values, one row per
            (frame, component, sensor). Not tke or spectral rows, and not bins.
        obs: ``(N_o,)`` observed values, same rows.
        obs_var: ``(N_o,)`` instrument plus representation variance. No sampling
            error: the realised field is the target.
        obs_time: ``(N_o,)`` observation times, on the clock of ``frames``.
        obs_coords: ``(N_o, 3)`` sensor positions. Equal rows are one sensor.
        params: ``(N_p, J)`` analysis-space parameter rows of the members,
            estimated or not (log or bounded space, one row per time knot).
        row_coords: ``(N_rows, 3)`` positions of the rows; needed with ``L_r``.
        member_mask: ``(J,)``, ``False`` for failed or substituted members: they
            are excluded from every statistic and their columns stay unchanged.
            ``None`` keeps all.
        L_r: Distance at which the spatial taper reaches zero, or ``None`` for
            Eq. (11), one global ``W``, in which case ``L_t`` is not used.
        L_t: Time difference at which the temporal taper reaches zero. Required
            with ``L_r``.
        min_dof: Smallest ``nu`` accepted; below it the call raises.
        chunk_size: Rows per batch of ``nu x nu`` eigendecompositions.

    Returns:
        A :class:`FieldAnalysisInfo`.
    """
    if chunk_size < 1:
        raise ValueError(f"chunk_size must be >= 1, got {chunk_size}.")
    if L_r is not None and (L_t is None or row_coords is None):
        raise ValueError("L_r needs L_t and row_coords.")
    if L_r is not None and (L_r <= 0.0 or L_t is None or L_t <= 0.0):
        raise ValueError(f"L_r and L_t must be positive, got {L_r} and {L_t}.")
    analysis = _Analysis(pred_obs, obs, obs_var, params, member_mask, min_dof)
    obs_time = np.asarray(obs_time, dtype=float)
    obs_coords = np.asarray(obs_coords, dtype=float)
    if obs_time.shape != analysis.whitened.shape[:1] or obs_coords.shape != (
        obs_time.size,
        3,
    ):
        raise ValueError(
            f"obs_time must be ({analysis.whitened.shape[0]},) and obs_coords "
            f"({analysis.whitened.shape[0]}, 3), got {obs_time.shape} and "
            f"{obs_coords.shape}."
        )
    members = analysis.members

    weights = None
    spatial = sensors = sensor_xyz = None
    if L_r is None:
        weights = analysis.global_weights() - np.eye(analysis.num_members)
    else:
        sensor_xyz, sensors = np.unique(obs_coords, axis=0, return_inverse=True)
        sensors = np.asarray(sensors).reshape(-1)
        rows_xyz = np.asarray(row_coords, dtype=float)
        difference = rows_xyz[:, None, :] - sensor_xyz[None, :, :]
        spatial = np.asarray(
            _taper(np.sqrt(np.sum(difference**2, axis=-1)), L_r), dtype=float
        )
        del difference

    num_frames = num_updated = 0
    rows_seen = rows_updated = 0
    for index, (time, values) in enumerate(frames):
        num_frames += 1
        rows_seen += values.shape[0]
        if weights is not None:
            _apply_global(values, weights, members, chunk_size)
            rows_updated += values.shape[0]
            num_updated += 1
            write(index, values)
            continue
        assert spatial is not None and sensors is not None and L_t is not None
        temporal = np.asarray(_taper(np.abs(obs_time - time), L_t), dtype=float)
        local = np.flatnonzero(temporal)
        live = np.flatnonzero(spatial[:, np.unique(sensors[local])].any(axis=1))
        if live.size == 0:
            continue
        gram, cross = analysis.sensor_sums(
            local, temporal[local], sensors[local], spatial.shape[1]
        )
        _apply_local(
            values, live, spatial, gram, cross, analysis.basis, members, chunk_size
        )
        rows_updated += live.size
        num_updated += 1
        write(index, values)

    info = FieldAnalysisInfo(
        num_members=analysis.num_members,
        num_parameters=analysis.num_parameters,
        degrees_of_freedom=analysis.dof,
        num_frames=num_frames,
        num_updated_frames=num_updated,
        updated_row_fraction=rows_updated / max(rows_seen, 1),
    )
    logger.info("Field analysis: %s", info)
    return info


def _taper(distance: np.ndarray, cutoff: float) -> jnp.ndarray:
    with enable_x64():
        return gaspari_cohn(jnp.asarray(distance), cutoff)


def _apply_global(
    values: np.ndarray, weights: np.ndarray, members: np.ndarray, chunk_size: int
) -> None:
    """``values += anomalies (W - I)`` on the unmasked columns, in place.

    Anomalies rather than raw rows, as in ``apply_ensemble_transform``: the
    columns of ``W - I`` sum to zero, and centring keeps the sum from cancelling
    large means.
    """
    for start in range(0, values.shape[0], chunk_size):
        block = values[start : start + chunk_size, members].astype(float)
        anomalies = block - block.mean(axis=1, keepdims=True)
        updated = block + anomalies @ weights
        values[start : start + chunk_size, members] = updated


def _apply_local(
    values: np.ndarray,
    live: np.ndarray,
    spatial: np.ndarray,
    gram: np.ndarray,
    cross: np.ndarray,
    basis: np.ndarray,
    members: np.ndarray,
    chunk_size: int,
) -> None:
    """Update the ``live`` rows of one frame in place, ``chunk_size`` at a time."""
    columns = np.flatnonzero(members)
    with enable_x64():
        gram_j, cross_j, basis_j = (jnp.asarray(a) for a in (gram, cross, basis))
        for start in range(0, live.size, chunk_size):
            rows = live[start : start + chunk_size]
            # Padded to one shape, so one compilation serves every chunk; a zero
            # taper is the identity on a zero row.
            pad = chunk_size - rows.size
            block = np.pad(
                values[np.ix_(rows, columns)].astype(float), ((0, pad), (0, 0))
            )
            taper = np.pad(spatial[rows], ((0, pad), (0, 0)))
            updated = _local_update(
                jnp.asarray(block), jnp.asarray(taper), gram_j, cross_j, basis_j
            )
            values[np.ix_(rows, columns)] = np.asarray(updated)[: rows.size]


def field_analysis_netcdf(
    path: pathlib.Path,
    *,
    variables: Optional[Sequence[str]] = None,
    **kwargs: object,
) -> FieldAnalysisInfo:
    """Run :func:`field_analysis` in place on an ensemble NetCDF state file.

    ``path`` is a ``_BaseESMDA.save_state`` file (dims ``ensemble``, ``time`` and
    the spatial ones), written the same way in both ESMDA save modes, so the
    analysis needs no second path and only one frame is ever in memory. Copy the
    file first to keep the forecast. A failure midway leaves a partly updated
    file.

    Args:
        path: The file, updated in place.
        variables: Fields to update; ``None`` for all with ``ensemble`` and
            ``time`` dims. Rows are these variables in sorted order, as in
            ``StateAugmentation``.
        **kwargs: The keyword arguments of :func:`field_analysis`, except
            ``row_coords``, which comes from the file; the frame times too.
    """
    with xarray.open_dataset(path, decode_times=False) as dataset:
        names = sorted(
            str(name)
            for name, variable in dataset.data_vars.items()
            if {"ensemble", "time"} <= set(variable.dims)
            and (variables is None or name in variables)
        )
        missing = set(variables or ()) - set(names)
        if missing:
            raise ValueError(f"Fields absent from {path}: {sorted(missing)}")
        row_coords = np.asarray(
            StateAugmentation().row_coords(dataset[names].isel(time=0)), dtype=float
        )

    with netCDF4.Dataset(path, "r+") as file:
        fields = [file.variables[name] for name in names]
        for field in fields:
            field.set_auto_maskandscale(False)
        sizes = [
            math.prod(f.shape)
            // (
                f.shape[f.dimensions.index("ensemble")]
                * f.shape[f.dimensions.index("time")]
            )
            for f in fields
        ]
        offsets = np.cumsum([0, *sizes])
        times = np.asarray(file.variables["time"][:], dtype=float)

        def selection(field: netCDF4.Variable, index: int) -> tuple[object, ...]:
            return tuple(
                index if dim == "time" else slice(None) for dim in field.dimensions
            )

        def frames() -> Iterable[tuple[float, np.ndarray]]:
            for index, time in enumerate(times):
                parts = []
                for field in fields:
                    axis = [d for d in field.dimensions if d != "time"].index(
                        "ensemble"
                    )
                    frame = np.moveaxis(field[selection(field, index)], axis, -1)
                    parts.append(frame.reshape(-1, frame.shape[-1]))
                yield float(time), np.concatenate(parts)

        def write(index: int, values: np.ndarray) -> None:
            for i, field in enumerate(fields):
                dims = [d for d in field.dimensions if d != "time"]
                axis = dims.index("ensemble")
                shape = [field.shape[field.dimensions.index(d)] for d in dims]
                moved = shape[:axis] + shape[axis + 1 :] + [shape[axis]]
                part = values[offsets[i] : offsets[i + 1]].reshape(moved)
                field[selection(field, index)] = np.moveaxis(part, -1, axis)

        return field_analysis(frames(), write, row_coords=row_coords, **kwargs)  # type: ignore[arg-type]
