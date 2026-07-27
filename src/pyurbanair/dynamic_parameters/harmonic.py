"""Deterministic harmonic time series for prescribed dynamic forcing."""

from __future__ import annotations

import math
from typing import Any, Optional

import jax
import jax.numpy as jnp
import xarray

from .ar2_relaxation import build_knot_times
from .base import ParameterTimeSeries


class HarmonicParameterModel(ParameterTimeSeries):
    """Generate deterministic sine/cosine parameter trajectories.

    ``profiles`` maps each time-varying parameter to a mapping with:

    - ``waveform``: ``"sine"`` or ``"cosine"``;
    - ``offset``: mean value;
    - ``amplitude``: excursion about that mean;
    - ``frequency``: cycles per second (Hz);
    - optional ``phase`` in radians and ``min`` / ``max`` clips.

    Every ensemble member receives the same prescribed profile.  This is useful
    for controlled forward-model comparisons: the only varying factor is the
    solver, not a stochastic parameter draw.  Rollout windows evaluate the
    waveform on a continuous global clock, even though the solvers receive a
    local clock for each window.
    """

    def __init__(
        self,
        profiles: dict[str, dict[str, Any]],
        simulation_time: float,
        seconds_per_knot: float,
        static_parameters: Optional[dict[str, Any]] = None,
        seed: int = 0,
    ) -> None:
        if not profiles:
            raise ValueError("HarmonicParameterModel needs at least one profile.")
        self.profiles = profiles
        self.simulation_time = float(simulation_time)
        self.seconds_per_knot = float(seconds_per_knot)
        self.static_parameters = static_parameters or {}
        self._next_window_start = 0.0
        time_coords = build_knot_times(0.0, simulation_time, seconds_per_knot)

        # ParameterTimeSeries supplies the public interface and time grid. Its
        # external-prior helpers do not apply here, since profiles are fixed.
        super().__init__({}, time_coords, seed)
        self.param_names = list(profiles)
        self._validate_profiles()

    def _validate_profiles(self) -> None:
        for name, profile in self.profiles.items():
            waveform = str(profile.get("waveform", "sine")).lower()
            if waveform not in {"sine", "cosine"}:
                raise ValueError(
                    f"Profile {name!r} has waveform={waveform!r}; use 'sine' "
                    "or 'cosine'."
                )
            if "amplitude" not in profile or "frequency" not in profile:
                raise ValueError(
                    f"Profile {name!r} needs both amplitude and frequency (Hz)."
                )

    def _values(
        self, time_coords: jnp.ndarray, origin: float
    ) -> dict[str, jnp.ndarray]:
        absolute_time = jnp.asarray(time_coords, dtype=float) + origin
        arrays: dict[str, jnp.ndarray] = {}
        for name, profile in self.profiles.items():
            angle = 2.0 * math.pi * float(profile["frequency"]) * absolute_time + float(
                profile.get("phase", 0.0)
            )
            waveform = str(profile.get("waveform", "sine")).lower()
            oscillation = jnp.sin(angle) if waveform == "sine" else jnp.cos(angle)
            values = (
                float(profile.get("offset", 0.0))
                + float(profile["amplitude"]) * oscillation
            )
            if profile.get("min") is not None:
                values = jnp.maximum(values, float(profile["min"]))
            if profile.get("max") is not None:
                values = jnp.minimum(values, float(profile["max"]))
            arrays[name] = values
        return arrays

    def _dataset(
        self,
        time_coords: jnp.ndarray,
        origin: float,
        ensemble_size: int,
        passthrough: Optional[dict[str, xarray.DataArray]] = None,
    ) -> xarray.Dataset:
        arrays = self._values(time_coords, origin)
        data_vars: dict[str, tuple[tuple[str, ...], jnp.ndarray] | xarray.DataArray] = {
            name: (
                ("time", "ensemble"),
                jnp.broadcast_to(values[:, None], (len(time_coords), ensemble_size)),
            )
            for name, values in arrays.items()
        }
        if passthrough:
            data_vars.update(passthrough)
        return xarray.Dataset(
            data_vars=data_vars,
            coords={
                "time": jnp.asarray(time_coords),
                "ensemble": jnp.arange(ensemble_size),
            },
        )

    def _sample_statics(self, ensemble_size: int) -> dict[str, tuple[str, jnp.ndarray]]:
        if not self.static_parameters:
            return {}
        key = jax.random.fold_in(self.rng_key, 0x5A715)
        keys = jax.random.split(key, len(self.static_parameters))
        return {
            name: ("ensemble", dist.sample(subkey, ensemble_size))
            for subkey, (name, dist) in zip(keys, self.static_parameters.items())
        }

    def sample(self, ensemble_size: int) -> xarray.Dataset:
        """Return the prescribed initial-window profiles for every member."""
        self._next_window_start = self.simulation_time
        return self._dataset(
            self.time_coords,
            origin=0.0,
            ensemble_size=ensemble_size,
            passthrough=self._sample_statics(ensemble_size),
        )

    def extrapolate(
        self,
        posterior: xarray.Dataset,
        prediction_times: jnp.ndarray,
        rng_key: jax.Array,
    ) -> xarray.Dataset:
        """Continue the prescribed waveform; preserve posterior static values."""
        del rng_key  # The trajectory is deterministic by design.
        prediction_times = jnp.asarray(prediction_times)
        ensemble_size = posterior.sizes["ensemble"]
        passthrough = {
            name: posterior[name]
            for name in posterior.data_vars
            if "time" not in posterior[name].dims
        }
        result = self._dataset(
            prediction_times,
            origin=self._next_window_start,
            ensemble_size=ensemble_size,
            passthrough=passthrough,
        )
        self._next_window_start += float(prediction_times[-1] - prediction_times[0])
        return result
