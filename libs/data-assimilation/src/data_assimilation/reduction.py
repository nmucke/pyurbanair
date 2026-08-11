"""Online state-space SVD reductions used by smoothing and filtering.

``OnlineStateReduction`` refits a basis from the supplied snapshots.  Its
historical default is a whitened KL parameterisation for ESMDA.  Filtering can
select orthonormal coefficients instead, so singular values used to choose the
basis are not incorrectly treated as the covariance of a different ensemble.

``StreamingStateReduction`` maintains one exponentially weighted incremental
SVD.  It stores only the retained left singular vectors and singular values;
past snapshot blocks are never retained.

``basis_source`` and ``snapshot_stride`` are ESMDA-only knobs (they select
which snapshots the *smoother* hands to :meth:`OnlineStateReduction.fit`); the
sequential filter always fits the current cycle's final forecast frame.
"""

from __future__ import annotations

import logging
import math
import warnings
from collections.abc import Mapping, Sequence
from typing import Optional

import jax.numpy as jnp
import numpy as np
import xarray
from data_assimilation.augmentation import StateAugmentation

logger = logging.getLogger(__name__)

BASIS_SOURCES = ("initial_condition", "window_snapshots")

# Multiples of the dtype's eps below which a singular value is round-off rather
# than signal on the non-whitened path. See :func:`_numerical_rank`; a small
# ensemble must not tighten the cut below the measured float32 noise floor.
_RANK_TOLERANCE_FLOOR = 64


def _validate_fraction(energy_fraction: float) -> float:
    value = float(energy_fraction)
    if not math.isfinite(value) or not 0.0 < value <= 1.0:
        raise ValueError(
            f"energy_fraction must be finite and in (0, 1], got {energy_fraction}."
        )
    return value


def _orthogonality_error(modes: jnp.ndarray) -> float:
    """Frobenius error ``||U.T U - I||`` of a set of modes."""
    if modes.shape[1] == 0:
        return 0.0
    gram = modes.T @ modes
    return float(jnp.linalg.norm(gram - jnp.eye(gram.shape[0], dtype=gram.dtype)))


def _numerical_rank(
    singular_values: jnp.ndarray,
    matrix_shape: Sequence[int],
    conservative: bool = True,
) -> int:
    """Count singular values above the thin SVD's round-off floor.

    ``conservative=True`` reproduces ``numpy.linalg.matrix_rank``'s
    ``eps * max(shape)`` cut, which is what the whitened ESMDA parameterisation
    needs: ``encode`` divides by the retained singular values there, so a mode
    admitted just above the round-off floor is amplified by ``1/sigma``.

    ``conservative=False`` is for the non-whitened filtering path, which never
    divides by ``sigma``. The ``max`` variant is actively wrong there in
    float32: at ``N_s ~ 1e5`` it discards every mode below ~1 percent of
    ``sigma_max``, quietly turning ``energy_fraction=1.0`` into a truncated
    basis. Measured across ``N_s`` from 1e2 to 2e5 and ``N_e`` from 2 to 50, the
    round-off floor (the null direction centring creates) stays between
    ``0.3`` and ``4`` times eps relative to ``sigma_max`` and barely grows with
    either dimension, while genuine trailing modes sit above
    ``1e-4 * sigma_max``. ``_RANK_TOLERANCE_FLOOR * eps`` therefore sits about
    20x above the noise and ~100x below the smallest real mode; the sample
    dimension only takes over if it exceeds that floor.
    """
    if singular_values.size == 0:
        return 0
    largest = float(singular_values[0])
    if not math.isfinite(largest) or largest == 0.0:
        return 0
    dimension = (
        max(matrix_shape)
        if conservative
        else max(min(matrix_shape), _RANK_TOLERANCE_FLOOR)
    )
    tolerance = largest * np.finfo(singular_values.dtype).eps * dimension
    return int(jnp.sum(singular_values > tolerance))


def _select_rank(
    singular_values: jnp.ndarray,
    matrix_shape: Sequence[int],
    energy_fraction: float,
    max_rank: Optional[int],
    conservative_rank: bool = True,
) -> tuple[int, int, float]:
    """Return retained rank, numerically nonzero rank, and retained energy.

    The retained energy is measured against the **whole** spectrum, so a rank
    lost to the numerical cut above still shows up as retained energy below
    one instead of being normalised away.
    """
    available_rank = _numerical_rank(singular_values, matrix_shape, conservative_rank)
    if available_rank == 0:
        return 0, 0, 0.0

    energy = np.square(np.asarray(singular_values, dtype=np.float64))
    cumulative = np.cumsum(energy) / np.sum(energy)
    rank = int(np.searchsorted(cumulative, energy_fraction - 1e-12)) + 1
    if max_rank is not None:
        rank = min(rank, max_rank)
    rank = min(rank, available_rank)
    return rank, available_rank, float(cumulative[rank - 1])


class OnlineStateReduction:
    """Reduced SVD/KL parameterisation of a state update.

    Args:
        energy_fraction: Smallest retained rank whose squared singular values
            contain this fraction of the spectrum.
        max_rank: Optional hard cap on the retained rank.
        basis_source: ESMDA snapshot source.  The reduction itself treats both
            sources identically; the smoother chooses which array to pass.
        snapshot_stride: Time thinning used by the ESMDA window-snapshot path.
        whiten: Divide encoded coefficients by singular values and undo that
            division when decoding.  ``True`` preserves the existing ESMDA
            behaviour; filtering uses ``False`` for orthonormal coefficients.
        variable_scales: Optional characteristic scale per state variable.
            Scaled coordinates are ``state / scale``. Missing state variables
            use ``1.0``; unknown configured variables are rejected by
            :meth:`resolve_row_scales`.

    ``decode_increment`` maps only an increment.  Consequently a discarded
    projection residual stays attached to its ensemble member, and zero gain
    leaves the full physical state unchanged.
    """

    def __init__(
        self,
        energy_fraction: float = 0.99,
        max_rank: Optional[int] = None,
        basis_source: str = "initial_condition",
        snapshot_stride: int = 1,
        whiten: bool = True,
        variable_scales: Optional[Mapping[str, float]] = None,
    ) -> None:
        self.energy_fraction = _validate_fraction(energy_fraction)
        if basis_source not in BASIS_SOURCES:
            raise ValueError(
                f"basis_source must be one of {BASIS_SOURCES}, got {basis_source!r}."
            )
        if max_rank is not None and max_rank < 1:
            raise ValueError(f"max_rank must be >= 1, got {max_rank}.")
        if snapshot_stride < 1:
            raise ValueError(f"snapshot_stride must be >= 1, got {snapshot_stride}.")

        scales = dict(variable_scales) if variable_scales is not None else None
        if scales is not None:
            for name, value in scales.items():
                if not isinstance(name, str) or not name:
                    raise ValueError("variable_scales keys must be non-empty strings.")
                if not math.isfinite(float(value)) or float(value) <= 0.0:
                    raise ValueError(
                        "variable_scales values must be finite and positive; "
                        f"got {name}={value}."
                    )

        self.max_rank = max_rank
        self.basis_source = basis_source
        self.snapshot_stride = snapshot_stride
        self.whiten = bool(whiten)
        self.variable_scales = scales

        self._mean: Optional[jnp.ndarray] = None  # scaled mean, (N_s, 1)
        self._modes: Optional[jnp.ndarray] = None  # scaled-space modes, (N_s, r)
        self._singular_values: Optional[jnp.ndarray] = None
        # None means unit scaling — the common case, kept out of the per-cycle
        # (N_s, N_e) arithmetic entirely.
        self._row_scales: Optional[jnp.ndarray] = None
        self._fit_coordinates: Optional[jnp.ndarray] = None
        self._available_rank: Optional[int] = None
        self._retained_energy_fraction: Optional[float] = None
        self._resolved_variable_scales: Optional[dict[str, float]] = None
        self._warned_mixed_units = False
        self._basis_updated = False

    @property
    def rank(self) -> int:
        """Rank of the currently fitted basis."""
        self._require_fit()
        assert self._singular_values is not None
        return int(self._singular_values.shape[0])

    @property
    def available_rank(self) -> int:
        """Numerically nonzero rank before scientific truncation."""
        self._require_fit()
        assert self._available_rank is not None
        return self._available_rank

    @property
    def retained_energy_fraction(self) -> float:
        """Fraction of the full squared singular-value spectrum retained."""
        self._require_fit()
        assert self._retained_energy_fraction is not None
        return self._retained_energy_fraction

    @property
    def condition_number(self) -> Optional[float]:
        """Ratio of largest to smallest retained singular value, if nonempty."""
        self._require_fit()
        assert self._singular_values is not None
        if self._singular_values.size == 0:
            return None
        return float(self._singular_values[0] / self._singular_values[-1])

    @property
    def basis_updated(self) -> bool:
        """Whether the most recent :meth:`fit` rebuilt the basis."""
        return self._basis_updated

    @property
    def modes(self) -> jnp.ndarray:
        """Retained left singular vectors ``U_r`` (immutable, not a copy)."""
        self._require_fit()
        assert self._modes is not None
        return self._modes

    @property
    def singular_values(self) -> jnp.ndarray:
        """Retained singular values (immutable, not a copy)."""
        self._require_fit()
        assert self._singular_values is not None
        return self._singular_values

    @property
    def spectrum_max(self) -> Optional[float]:
        """Largest singular value, normalized to compare across cycles.

        A refit spectrum is already comparable; the streaming accumulator grows
        with its weight, so the subclass divides by ``sqrt(w_k)``.
        """
        spectrum = self.singular_values
        return None if spectrum.size == 0 else float(spectrum[0])

    @property
    def fit_coordinates(self) -> Optional[jnp.ndarray]:
        """Fitted anomalies in the FULL (untruncated) basis, shape ``(k, N)``.

        ``anomalies = U_full @ coordinates`` in scaled coordinates, with rows
        ordered like the spectrum, so rows ``[:rank]`` are exactly what the
        truncated basis can represent and rows ``[rank:]`` are what it discards.
        Both come free from the decomposition, which lets callers measure a
        truncation loss without re-running a full-space analysis. ``None`` when
        the last fit did not produce a complete split (see
        :class:`StreamingStateReduction`).
        """
        self._require_fit()
        if self._fit_coordinates is None:
            return None
        return jnp.array(self._fit_coordinates)

    @property
    def resolved_variable_scales(self) -> Optional[dict[str, float]]:
        """Resolved scale for every variable after template validation."""
        if self._resolved_variable_scales is None:
            return None
        return dict(self._resolved_variable_scales)

    @property
    def subspace_drift(self) -> Optional[float]:
        """Largest principal-angle drift; current-cycle refits have no history."""
        return None

    def resolve_row_scales(self, state: xarray.Dataset) -> Optional[jnp.ndarray]:
        """Expand configured per-variable scales into flatten-order row scales.

        Returns ``None`` when no scales are configured, so the unit-scale
        default never touches the ``(N_s, N_e)`` arrays. Row expansion itself is
        :meth:`StateAugmentation.row_scales`, which owns the flattening order.
        If no scales were supplied and complete nonempty ``units`` attributes
        conflict, a warning calls attention to the mixed physical norm without
        guessing a conversion.
        """
        variable_names = sorted(str(name) for name in state.data_vars)
        if self.variable_scales is None:
            if not self._warned_mixed_units:
                units = [
                    str(state[name].attrs.get("units", "")).strip()
                    for name in variable_names
                ]
                if units and all(units) and len(set(units)) > 1:
                    warnings.warn(
                        "State variables have conflicting units but "
                        "variable_scales was omitted; using Euclidean scale 1.0 "
                        "for every variable.",
                        UserWarning,
                        stacklevel=2,
                    )
                    self._warned_mixed_units = True
            self._resolved_variable_scales = {name: 1.0 for name in variable_names}
            return None

        unknown = sorted(set(self.variable_scales) - set(variable_names))
        if unknown:
            raise ValueError(
                "variable_scales contains variables absent from the state: "
                + ", ".join(unknown)
            )
        rows, resolved = StateAugmentation().row_scales(state, self.variable_scales)
        self._resolved_variable_scales = resolved
        return rows

    def fit(
        self,
        snapshots_flat: jnp.ndarray,
        row_scales: Optional[Sequence[float] | jnp.ndarray] = None,
    ) -> None:
        """Fit the truncated basis to ``(state rows, snapshot columns)``."""
        snapshots, scales = self._validated_inputs(snapshots_flat, row_scales)
        n_samples = snapshots.shape[1]
        scaled = self._scale(snapshots, scales)
        mean = jnp.mean(scaled, axis=1, keepdims=True)
        anomalies = (scaled - mean) / jnp.sqrt(n_samples - 1)
        modes, singular_values, vt = jnp.linalg.svd(anomalies, full_matrices=False)
        rank, available_rank, retained_energy = _select_rank(
            singular_values,
            anomalies.shape,
            self.energy_fraction,
            self.max_rank,
            conservative_rank=self.whiten,
        )

        # Commit only after all validation and decompositions succeed.
        self._mean = mean
        self._modes = modes[:, :rank]
        self._singular_values = singular_values[:rank]
        self._fit_coordinates = singular_values[:, None] * vt
        self._row_scales = scales
        self._available_rank = available_rank
        self._retained_energy_fraction = retained_energy
        self._basis_updated = True
        logger.info(
            "SVD state reduction: rank %d/%d nonzero modes (rows %d, snapshots "
            "%d, retained energy %.4f)",
            rank,
            available_rank,
            snapshots.shape[0],
            n_samples,
            retained_energy,
        )

    def encode(self, states_flat: jnp.ndarray) -> jnp.ndarray:
        """Encode states as whitened KL or orthonormal POD coefficients."""
        self._require_fit()
        assert self._modes is not None
        assert self._mean is not None
        states = jnp.asarray(states_flat)
        if states.ndim != 2 or states.shape[0] != self._modes.shape[0]:
            raise ValueError(
                f"states_flat must have shape ({self._modes.shape[0]}, N), got "
                f"{states.shape}."
            )
        coefficients = self._modes.T @ (
            self._scale(states, self._row_scales) - self._mean
        )
        if self.whiten:
            assert self._singular_values is not None
            coefficients = coefficients / self._singular_values[:, None]
        return coefficients

    def decode_increment(self, coefficient_increment: jnp.ndarray) -> jnp.ndarray:
        """Decode an increment and invert row scaling into physical state space."""
        self._require_fit()
        assert self._modes is not None
        increment = jnp.asarray(coefficient_increment)
        if increment.ndim != 2 or increment.shape[0] != self._modes.shape[1]:
            raise ValueError(
                f"coefficient_increment must have shape ({self._modes.shape[1]}, N), "
                f"got {increment.shape}."
            )
        if self.whiten:
            assert self._singular_values is not None
            increment = self._singular_values[:, None] * increment
        decoded = self._modes @ increment
        if self._row_scales is None:
            return decoded
        return self._row_scales[:, None] * decoded

    def projection_residual(self, snapshots_flat: jnp.ndarray) -> float:
        """Relative residual of current centered anomalies in the fitted basis."""
        self._require_fit()
        assert self._modes is not None
        snapshots = jnp.asarray(snapshots_flat)
        if snapshots.ndim != 2 or snapshots.shape[0] != self._modes.shape[0]:
            raise ValueError(
                f"snapshots_flat must have shape ({self._modes.shape[0]}, N), got "
                f"{snapshots.shape}."
            )
        scaled = self._scale(snapshots, self._row_scales)
        anomalies = scaled - jnp.mean(scaled, axis=1, keepdims=True)
        denominator = float(jnp.linalg.norm(anomalies))
        if denominator == 0.0:
            return 0.0
        residual = anomalies - self._modes @ (self._modes.T @ anomalies)
        return float(jnp.linalg.norm(residual) / denominator)

    @staticmethod
    def _scale(states: jnp.ndarray, row_scales: Optional[jnp.ndarray]) -> jnp.ndarray:
        """Divide by the row scales, skipping the work for the unit default."""
        if row_scales is None:
            return states
        return states / row_scales[:, None]

    def _validated_inputs(
        self,
        snapshots_flat: jnp.ndarray,
        row_scales: Optional[Sequence[float] | jnp.ndarray],
    ) -> tuple[jnp.ndarray, Optional[jnp.ndarray]]:
        snapshots = jnp.asarray(snapshots_flat)
        if snapshots.ndim != 2:
            raise ValueError(
                f"snapshots_flat must be a two-dimensional array, got {snapshots.shape}."
            )
        if snapshots.shape[1] < 2:
            raise ValueError(
                f"Need at least 2 snapshots to build a basis, got {snapshots.shape[1]}."
            )
        bad_columns = np.flatnonzero(
            ~np.all(np.isfinite(np.asarray(snapshots)), axis=0)
        ).tolist()
        if bad_columns:
            raise ValueError(
                "State snapshots contain non-finite values in ensemble "
                f"member/sample columns {bad_columns}; the basis was not updated."
            )

        if row_scales is None:
            return snapshots, None
        scales = jnp.asarray(row_scales, dtype=snapshots.dtype)
        if scales.ndim != 1 or scales.shape[0] != snapshots.shape[0]:
            raise ValueError(
                f"row_scales must have shape ({snapshots.shape[0]},), got "
                f"{scales.shape}."
            )
        if not bool(jnp.all(jnp.isfinite(scales) & (scales > 0.0))):
            raise ValueError("row_scales values must be finite and positive.")
        return snapshots, scales

    def _require_fit(self) -> None:
        if self._modes is None:
            raise RuntimeError("fit() must be called before using the reduction.")


class StreamingStateReduction(OnlineStateReduction):
    """Exponentially forgotten incremental POD of forecast-anomaly blocks.

    The represented unnormalized accumulator is
    ``C_k = forgetting_factor * C_{k-1} + B_k B_k.T``. The first accepted block
    is fit directly, and subsequent blocks update a small projected matrix.
    ``update_every_n_cycles`` reuses the last basis on skipped calls; the
    default of one includes every cycle.
    """

    def __init__(
        self,
        energy_fraction: float = 0.99,
        max_rank: Optional[int] = None,
        variable_scales: Optional[Mapping[str, float]] = None,
        forgetting_factor: float = 1.0,
        update_every_n_cycles: int = 1,
        orthogonality_tolerance: float = 1e-5,
        subspace_warning_threshold: Optional[float] = None,
    ) -> None:
        forgetting_factor = float(forgetting_factor)
        if not math.isfinite(forgetting_factor) or not 0.0 < forgetting_factor <= 1.0:
            raise ValueError(
                "forgetting_factor must be finite and in (0, 1], got "
                f"{forgetting_factor}."
            )
        if update_every_n_cycles < 1:
            raise ValueError(
                "update_every_n_cycles must be >= 1, got " f"{update_every_n_cycles}."
            )
        if (
            not math.isfinite(float(orthogonality_tolerance))
            or orthogonality_tolerance <= 0
        ):
            raise ValueError("orthogonality_tolerance must be finite and positive.")
        if subspace_warning_threshold is not None and (
            not math.isfinite(float(subspace_warning_threshold))
            or not 0.0 <= subspace_warning_threshold <= math.pi / 2
        ):
            raise ValueError("subspace_warning_threshold must be in [0, pi/2] radians.")

        super().__init__(
            energy_fraction=energy_fraction,
            max_rank=max_rank,
            whiten=False,
            variable_scales=variable_scales,
        )
        self.forgetting_factor = forgetting_factor
        self.update_every_n_cycles = update_every_n_cycles
        self.orthogonality_tolerance = float(orthogonality_tolerance)
        self.subspace_warning_threshold = subspace_warning_threshold
        self._accumulator_weight = 0.0
        self._fit_calls = 0
        self._subspace_drift: Optional[float] = None
        half_life = self.covariance_half_life
        logger.info(
            "Streaming state reduction: forgetting_factor=%.4g (%s), "
            "update_every_n_cycles=%d",
            forgetting_factor,
            (
                "equal-weight history"
                if half_life is None
                else f"old-block half-life {half_life:.2f} cycles"
            ),
            update_every_n_cycles,
        )

    @property
    def accumulator_weight(self) -> float:
        """Exponentially accumulated weight of accepted anomaly blocks."""
        return self._accumulator_weight

    @property
    def covariance_half_life(self) -> Optional[float]:
        """Old-block covariance half-life in cycles; ``None`` for no forgetting."""
        if self.forgetting_factor == 1.0:
            return None
        return math.log(0.5) / math.log(self.forgetting_factor)

    @property
    def spectrum_max(self) -> Optional[float]:
        spectrum = self.singular_values
        if spectrum.size == 0:
            return None
        return float(spectrum[0]) / math.sqrt(self._accumulator_weight)

    @property
    def subspace_drift(self) -> Optional[float]:
        """Largest principal-angle change from the previous accepted basis.

        Resolves large rotations, not small ones: see :meth:`_principal_angle`
        for the float32 floor below which a value here means "no measurable
        drift" rather than a trustworthy angle.
        """
        return self._subspace_drift

    def fit(
        self,
        snapshots_flat: jnp.ndarray,
        row_scales: Optional[Sequence[float] | jnp.ndarray] = None,
    ) -> None:
        """Validate and incrementally incorporate one current anomaly block."""
        snapshots, scales = self._validated_inputs(snapshots_flat, row_scales)
        call_index = self._fit_calls
        if self._modes is not None:
            unchanged = (
                scales is None
                if self._row_scales is None
                else scales is not None
                and scales.shape == self._row_scales.shape
                and np.array_equal(np.asarray(scales), np.asarray(self._row_scales))
            )
            if not unchanged:
                raise ValueError(
                    "row_scales must remain unchanged across streaming fits."
                )
        scaled = self._scale(snapshots, scales)
        mean = jnp.mean(scaled, axis=1, keepdims=True)

        if self._modes is not None and call_index % self.update_every_n_cycles != 0:
            # Validation still occurs on skipped calls so bad forecasts cannot
            # silently pass through analysis. Re-center encoding on the current
            # forecast while deliberately reusing the configured stale basis.
            # Splitting this block against the stale basis would need the very
            # decomposition the cadence skips, so the coordinate-split
            # diagnostics report ``None`` rather than a stale value.
            self._mean = mean
            self._fit_coordinates = None
            self._subspace_drift = None
            self._basis_updated = False
            self._fit_calls += 1
            return

        block = (scaled - mean) / jnp.sqrt(snapshots.shape[1] - 1)
        old_modes = self._modes

        if old_modes is None:
            modes_all, singular_values_all, vt = jnp.linalg.svd(
                block, full_matrices=False
            )
            coordinates_all = singular_values_all[:, None] * vt
            spectrum_shape = block.shape
            new_weight = 1.0
        else:
            assert self._singular_values is not None
            projected = old_modes.T @ block
            residual = block - old_modes @ projected
            # One classical Gram-Schmidt pass loses orthogonality as the
            # accumulated basis grows; a second projection (CGS2) costs the same
            # O(N_s * rank * N_e) and keeps ||U.T U - I|| at round-off, which is
            # what stops the expensive re-orthogonalization below from firing
            # every cycle (and with it the loss of the coordinate split).
            residual = residual - old_modes @ (old_modes.T @ residual)
            q_raw, r_raw = jnp.linalg.qr(residual, mode="reduced")
            r_modes, r_values, r_vt = jnp.linalg.svd(r_raw, full_matrices=False)
            residual_rank = _numerical_rank(
                r_values, residual.shape, conservative=False
            )
            q_residual = q_raw @ r_modes[:, :residual_rank]
            residual_coordinates = (
                r_values[:residual_rank, None] * r_vt[:residual_rank, :]
            )

            old_rank = old_modes.shape[1]
            top = jnp.concatenate(
                [
                    math.sqrt(self.forgetting_factor) * jnp.diag(self._singular_values),
                    projected,
                ],
                axis=1,
            )
            bottom = jnp.concatenate(
                [
                    jnp.zeros((residual_rank, old_rank), dtype=block.dtype),
                    residual_coordinates,
                ],
                axis=1,
            )
            small = jnp.concatenate([top, bottom], axis=0)
            small_modes, singular_values_all, _ = jnp.linalg.svd(
                small, full_matrices=False
            )
            expanded_modes = jnp.concatenate([old_modes, q_residual], axis=1)
            modes_all = expanded_modes @ small_modes
            # ``expanded_modes`` is orthonormal and contains the whole block, so
            # the block's coordinates in the rotated basis are exact and cost
            # only small-matrix products.
            coordinates_all = small_modes.T @ jnp.concatenate(
                [projected, residual_coordinates], axis=0
            )
            spectrum_shape = (block.shape[0], old_rank + block.shape[1])
            new_weight = self.forgetting_factor * self._accumulator_weight + 1.0

        rank, available_rank, retained_energy = _select_rank(
            singular_values_all,
            spectrum_shape,
            self.energy_fraction,
            self.max_rank,
            conservative_rank=False,  # never whitened
        )
        new_modes = modes_all[:, :rank]
        new_singular_values = singular_values_all[:rank]

        # Incremental roundoff is normally tiny. Reorthogonalize only when the
        # diagnostic crosses the configured tolerance, absorbing the resulting
        # triangular factor into a small SVD so covariance is preserved.
        # ||U.T U - I||_F accumulates over rank**2 entries, so compare against a
        # rank-scaled tolerance rather than a fixed one.
        drift_tolerance = self.orthogonality_tolerance * math.sqrt(max(rank, 1))
        if rank and _orthogonality_error(new_modes) > drift_tolerance:
            q, r = jnp.linalg.qr(new_modes, mode="reduced")
            correction_modes, corrected_values, _ = jnp.linalg.svd(
                r * new_singular_values[None, :], full_matrices=False
            )
            new_modes = q @ correction_modes
            new_singular_values = corrected_values
            # The rotation applies to the retained block only, so the retained
            # and discarded coordinates no longer share one consistent basis.
            # This corrective branch is rare; drop the split rather than
            # publish an inconsistent one.
            coordinates_all = None

        drift = self._principal_angle(old_modes, new_modes)
        if (
            drift is not None
            and self.subspace_warning_threshold is not None
            and drift > self.subspace_warning_threshold
        ):
            logger.warning(
                "Streaming state-reduction subspace drift %.4f rad exceeds "
                "configured threshold %.4f rad.",
                drift,
                self.subspace_warning_threshold,
            )

        # Transactional commit: all candidate quantities are finite and all
        # decompositions completed before existing streaming state is mutated.
        self._mean = mean
        self._modes = new_modes
        self._singular_values = new_singular_values
        self._fit_coordinates = coordinates_all
        self._row_scales = scales
        self._available_rank = available_rank
        self._retained_energy_fraction = retained_energy
        self._accumulator_weight = new_weight
        self._subspace_drift = drift
        self._basis_updated = True
        self._fit_calls += 1

    @staticmethod
    def _principal_angle(
        old_modes: Optional[jnp.ndarray], new_modes: jnp.ndarray
    ) -> Optional[float]:
        if old_modes is None:
            return None
        old_rank, new_rank = old_modes.shape[1], new_modes.shape[1]
        if old_rank == 0 and new_rank == 0:
            return 0.0
        if old_rank == 0 or new_rank == 0:
            return math.pi / 2
        # Principal angles are defined between subspaces of different dimension:
        # there are min(old_rank, new_rank) of them, and the largest measures how
        # far the smaller subspace tilts out of the larger. A growing streaming
        # rank must therefore NOT be reported as a 90-degree rotation.
        cosines = jnp.linalg.svd(old_modes.T @ new_modes, compute_uv=False)
        smallest_cosine = float(
            jnp.clip(jnp.min(cosines[: min(old_rank, new_rank)]), 0.0, 1.0)
        )
        # acos has an infinite derivative at 1, so a cosine perturbed by k*eps
        # reads back as an angle of ~sqrt(2*k*eps): in float32 that is a floor of
        # a few times 1e-3, and it moves with the platform's LAPACK. Small values
        # here mean "no measurable rotation", not a resolved angle. That is
        # enough for subspace_warning_threshold, which watches for large drift.
        return math.acos(smallest_cosine)
