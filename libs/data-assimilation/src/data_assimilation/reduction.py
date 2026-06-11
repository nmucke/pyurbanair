"""Online SVD/KL reduction of the state rows in the augmented ESMDA vector.

Implements the reduced parameterisation of ``docs/reduced_state_da.md`` with
the basis built ONLINE: each ESMDA iteration fits a thin SVD to the current
forecast ensemble's anomalies, so the basis tracks the ensemble as it moves
(the doc's "re-anchoring" caveat is handled by construction).

The smoother encodes the flattened state ensemble into reduced coefficients
``xi = Sigma_r^{-1} Phi_r^T (u - u_bar)``, Kalman-updates ``[xi | params]``,
and decodes the update as an INCREMENT ``u += Phi_r Sigma_r (xi_post -
xi_prior)``. Increment decoding keeps each member's projection residual, so a
zero Kalman update leaves the state untouched for either basis source, and at
full rank with ``basis_source="initial_condition"`` the update is identical to
the full-space update (which is confined to the ensemble span anyway).
"""

from typing import Optional

import jax.numpy as jnp

BASIS_SOURCES = ("initial_condition", "window_snapshots")


class OnlineStateReduction:
    """Reduced SVD/KL parameterisation of the ESMDA state update.

    Args:
        energy_fraction: Retained-energy truncation criterion. The rank ``r``
            is the smallest ``k`` with ``sum_{i<=k} s_i^2 / sum_i s_i^2 >=
            energy_fraction``. ``1.0`` keeps every nonzero mode.
        max_rank: Optional hard cap on ``r`` (applied after the energy
            criterion; always additionally capped by the number of nonzero
            singular values).
        basis_source: Which ensemble snapshots build the basis each iteration.
            ``"initial_condition"`` uses the time=0 ensemble anomalies (rank
            <= N_e - 1; the encoded coefficients are exactly whitened).
            ``"window_snapshots"`` uses every output frame of every member
            (N_e * N_t samples; richer basis, approximately whitened).
        snapshot_stride: Thin the time frames of the ``"window_snapshots"``
            source (every ``snapshot_stride``-th frame). Ignored by the
            ``"initial_condition"`` source.
    """

    def __init__(
        self,
        energy_fraction: float = 0.99,
        max_rank: Optional[int] = None,
        basis_source: str = "initial_condition",
        snapshot_stride: int = 1,
    ) -> None:
        if not 0.0 < energy_fraction <= 1.0:
            raise ValueError(
                f"energy_fraction must be in (0, 1], got {energy_fraction}."
            )
        if basis_source not in BASIS_SOURCES:
            raise ValueError(
                f"basis_source must be one of {BASIS_SOURCES}, got "
                f"{basis_source!r}."
            )
        if max_rank is not None and max_rank < 1:
            raise ValueError(f"max_rank must be >= 1, got {max_rank}.")
        if snapshot_stride < 1:
            raise ValueError(f"snapshot_stride must be >= 1, got {snapshot_stride}.")

        self.energy_fraction = energy_fraction
        self.max_rank = max_rank
        self.basis_source = basis_source
        self.snapshot_stride = snapshot_stride

        self._mean: Optional[jnp.ndarray] = None  # (N_s, 1)
        self._modes: Optional[jnp.ndarray] = None  # Phi_r, (N_s, r)
        self._singular_values: Optional[jnp.ndarray] = None  # Sigma_r diag, (r,)

    @property
    def rank(self) -> int:
        """Rank of the currently fitted basis."""
        if self._singular_values is None:
            raise RuntimeError("fit() must be called before querying the rank.")
        return int(self._singular_values.shape[0])

    def fit(self, snapshots_flat: jnp.ndarray) -> None:
        """Fit the truncated basis to snapshot anomalies.

        Args:
            snapshots_flat: Array of shape (N_s, N_samples) — flattened state
                snapshots as columns.
        """
        n_samples = snapshots_flat.shape[1]
        if n_samples < 2:
            raise ValueError(
                f"Need at least 2 snapshots to build a basis, got {n_samples}."
            )
        self._mean = jnp.mean(snapshots_flat, axis=1, keepdims=True)
        anomalies = (snapshots_flat - self._mean) / jnp.sqrt(n_samples - 1)
        modes, singular_values, _ = jnp.linalg.svd(anomalies, full_matrices=False)

        energy = singular_values**2
        total = jnp.sum(energy)
        # Smallest k whose cumulative energy reaches the retained fraction.
        # Tiny tolerance so energy_fraction=1.0 is not defeated by float
        # round-off in the cumulative sum.
        cumulative = jnp.cumsum(energy) / total
        rank = int(jnp.searchsorted(cumulative, self.energy_fraction - 1e-12)) + 1
        nonzero = int(jnp.sum(singular_values > singular_values[0] * 1e-12))
        rank = min(rank, nonzero)
        if self.max_rank is not None:
            rank = min(rank, self.max_rank)

        self._modes = modes[:, :rank]
        self._singular_values = singular_values[:rank]
        print(
            f"SVD state reduction: rank {rank}/{nonzero} nonzero modes "
            f"(rows {snapshots_flat.shape[0]}, snapshots {n_samples}, "
            f"retained energy {float(cumulative[rank - 1]):.4f})"
        )

    def encode(self, states_flat: jnp.ndarray) -> jnp.ndarray:
        """Project flattened states onto the basis: ``Sigma^{-1} Phi^T (u - u_bar)``.

        Args:
            states_flat: Array of shape (N_s, N_e).

        Returns:
            Coefficients of shape (r, N_e).
        """
        if self._modes is None:
            raise RuntimeError("fit() must be called before encode().")
        return (self._modes.T @ (states_flat - self._mean)) / self._singular_values[
            :, None
        ]

    def decode_increment(self, d_xi: jnp.ndarray) -> jnp.ndarray:
        """Map a coefficient increment back to state space: ``Phi Sigma d_xi``.

        Args:
            d_xi: Coefficient increment of shape (r, N_e).

        Returns:
            State-space increment of shape (N_s, N_e).
        """
        if self._modes is None:
            raise RuntimeError("fit() must be called before decode_increment().")
        return self._modes @ (self._singular_values[:, None] * d_xi)
