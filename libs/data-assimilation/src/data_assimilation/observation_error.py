"""Physical diagonal observation error for labelled, time-resolved products.

Instrument noise is sampled on raw frames. Representation uncertainty affects
the likelihood only. The independent time model permits exact propagation of
both diagonal contributions through a mean aggregation. The explicit ``none``
policy instead assigns the configured variance to each observation product.
"""

from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping

import numpy as np
import xarray as xr
from data_assimilation.observation_operator import AggregateObservations, ObservationBin


def _readonly(values: Any) -> np.ndarray:
    result = np.array(values, dtype=float, copy=True)
    result.setflags(write=False)
    return result


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, (tuple, list)):
        return tuple(_freeze(item) for item in value)
    return value


def _valid_std(value: Any, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite non-negative number.") from exc
    if not np.isfinite(result) or result < 0:
        raise ValueError(f"{name} must be a finite non-negative number.")
    return result


def _standard_deviations(
    setting: float | Mapping[str, Any],
    components: tuple[str, ...],
    sensors: tuple[int, ...],
    heights: np.ndarray | None,
    name: str,
) -> np.ndarray:
    """Resolve defaults, height bands, component and sensor overrides.

    More specific overrides win in that order. Sensor keys are zero-based
    indices, matching ``ObservationOperator`` sensor order.
    """
    if isinstance(setting, Mapping):
        known = {"default", "height_bands", "components", "sensors"}
        unknown = set(setting) - known
        if unknown:
            raise ValueError(f"Unknown {name} keys: {sorted(unknown)}")
        if "default" not in setting:
            raise ValueError(f"{name} requires a scalar 'default' fallback.")
        values = np.full(
            len(components), _valid_std(setting["default"], f"{name} default")
        )
        bands = setting.get("height_bands", ())
        if bands:
            if heights is None:
                raise ValueError(f"{name} height_bands require sensor z metadata.")
            seen_bands: list[tuple[float, float]] = []
            for band in bands:
                lower = float(band["min_z"])
                upper = float(band["max_z"])
                if not np.isfinite([lower, upper]).all() or lower >= upper:
                    raise ValueError(
                        f"{name} height band bounds must be finite and ordered."
                    )
                if any(
                    lower < previous_upper and previous_lower < upper
                    for previous_lower, previous_upper in seen_bands
                ):
                    raise ValueError(f"{name} height bands overlap or are duplicated.")
                seen_bands.append((lower, upper))
                band_std = _valid_std(band["std"], f"{name} height band std")
                for j, sensor in enumerate(sensors):
                    if lower <= heights[sensor] < upper:
                        values[j] = band_std
        component_map = setting.get("components", {})
        if len(component_map) != len(set(component_map)):
            raise ValueError(f"{name} has duplicate component labels.")
        missing = set(component_map) - set(components)
        if missing:
            raise ValueError(f"{name} has unknown component labels: {sorted(missing)}")
        component_values = {
            component: _valid_std(value, f"{name} component {component}")
            for component, value in component_map.items()
        }
        for j, component in enumerate(components):
            if component in component_map:
                values[j] = component_values[component]
        sensor_map = setting.get("sensors", {})
        normalized = {int(k): v for k, v in sensor_map.items()}
        if len(normalized) != len(sensor_map):
            raise ValueError(f"{name} has duplicate sensor labels.")
        missing_sensors = set(normalized) - set(sensors)
        if missing_sensors:
            raise ValueError(
                f"{name} has unknown sensor labels: {sorted(missing_sensors)}"
            )
        sensor_values = {
            sensor: _valid_std(value, f"{name} sensor {sensor}")
            for sensor, value in normalized.items()
        }
        for j, sensor in enumerate(sensors):
            if sensor in normalized:
                values[j] = sensor_values[sensor]
    else:
        values = np.full(len(components), _valid_std(setting, name))
    if not np.all(np.isfinite(values)) or np.any(values < 0):
        raise ValueError(f"{name} standard deviations must be finite and non-negative.")
    return values


@dataclass(frozen=True)
class ResolvedObservationError:
    """Covariance and labels for one actual window of observations."""

    raw_instrument_std: np.ndarray
    raw_instrument_variance: np.ndarray
    raw_representation_variance: np.ndarray
    instrument_variance: np.ndarray
    representation_variance: np.ndarray
    variance: np.ndarray
    raw_times: np.ndarray
    times: np.ndarray
    components: tuple[str, ...]
    sensor_indices: tuple[int, ...]
    frame_ids: tuple[tuple[int, ...], ...]
    weights: tuple[tuple[float, ...], ...]
    provenance: str = "observation_error.v1:diagonal:independent"

    @property
    def covariance_diag(self) -> np.ndarray:
        """Physical variance in time-major, sensor-innermost vector order."""
        return self.variance.reshape(-1)

    @property
    def std(self) -> np.ndarray:
        return np.asarray(np.sqrt(self.covariance_diag))


@dataclass(frozen=True)
class ObservationErrorSpec:
    """Immutable, opt-in diagonal observation-error specification.

    A standard deviation can be scalar or a mapping with ``default`` and
    optional ``height_bands``, ``components`` and ``sensors`` overrides.
    A height band is ``{min_z, max_z, std}``. Bounds use ``[min_z, max_z)``.
    """

    instrument_std: float | Mapping[str, Any]
    representation_std: float | Mapping[str, Any] = 0.0
    representation_time_model: str = "independent"
    aggregation: str = "propagate_mean"

    def __post_init__(self) -> None:
        object.__setattr__(self, "instrument_std", _freeze(self.instrument_std))
        object.__setattr__(self, "representation_std", _freeze(self.representation_std))

    def variance_upper_bound(self) -> float:
        """An upper bound on every physical variance this spec can resolve to.

        Resolved variances need the operator (height bands and sensor
        overrides), but a pre-flight check -- e.g. that a tempered ``beta * R``
        cannot overflow -- needs a number before any solver runs. Each resolved
        variance is ``instrument_std**2 + representation_std**2`` for one of the
        configured stds: ``aggregation="none"`` assigns exactly that to every
        product, and ``propagate_mean`` only shrinks it (the squared bin weights
        sum to <= 1), so the largest configured std of each part bounds it.
        """

        def largest(setting: float | Mapping[str, Any]) -> float:
            if not isinstance(setting, Mapping):
                return float(setting)
            values = [float(setting["default"])]
            values += [float(band["std"]) for band in setting.get("height_bands", ())]
            values += [float(v) for v in setting.get("components", {}).values()]
            values += [float(v) for v in setting.get("sensors", {}).values()]
            return max(values)

        return largest(self.instrument_std) ** 2 + largest(self.representation_std) ** 2

    def resolve(
        self,
        observations: xr.DataArray,
        observation_operator: Any,
        aggregate_observations: AggregateObservations | None = None,
    ) -> ResolvedObservationError:
        """Resolve the physical covariance on the current window's raw times."""
        if self.representation_time_model != "independent":
            raise ValueError(
                "Only independent representation_time_model is supported; "
                "persistent errors need a calibrated temporal covariance."
            )
        if self.aggregation not in ("propagate_mean", "none"):
            raise ValueError("aggregation must be 'propagate_mean' or 'none'.")
        if observations.dims != ("time", "obs"):
            raise ValueError("Observation error requires raw dims ('time', 'obs').")
        if "time" not in observations.coords:
            raise ValueError("Observation error requires physical time coordinates.")
        raw_times = np.asarray(observations["time"].values, dtype=float)
        if not raw_times.size or not np.all(np.isfinite(raw_times)):
            raise ValueError("Observation times must be finite and non-empty.")
        if np.any(np.diff(raw_times) <= 0):
            raise ValueError("Observation times must be strictly increasing.")
        base = getattr(
            observation_operator, "observation_operator", observation_operator
        )
        n_sensors = int(base.num_sensors)
        component_names = tuple(base.obs_states)
        if n_sensors <= 0 or not component_names:
            raise ValueError(
                "Observation sensor and component labels must be non-empty."
            )
        if len(component_names) != len(set(component_names)):
            raise ValueError("Observation components must have unique labels.")
        if observations.sizes["obs"] != n_sensors * len(component_names):
            raise ValueError("Observation count does not match operator labels.")
        if "obs" in observations.coords:
            obs_labels = np.asarray(observations["obs"].values)
            if not np.array_equal(obs_labels, np.arange(observations.sizes["obs"])):
                raise ValueError(
                    "Observation coordinate labels must be unique positional "
                    "indices matching the operator's component/sensor order."
                )
        if not np.all(np.isfinite(observations.values)):
            raise ValueError("Raw observations must be finite in corrected mode.")
        components = tuple(c for c in component_names for _ in range(n_sensors))
        sensors = tuple(i for _ in component_names for i in range(n_sensors))
        # Index-based obs_ids_z is a grid index, not a physical height.
        heights = getattr(base, "obs_z", None)
        if heights is not None:
            heights = np.asarray(heights, dtype=float)
            if heights.size != n_sensors or not np.all(np.isfinite(heights)):
                raise ValueError("Sensor heights must be finite and labelled.")
        instrument = _standard_deviations(
            self.instrument_std, components, sensors, heights, "instrument_std"
        )
        representation = _standard_deviations(
            self.representation_std, components, sensors, heights, "representation_std"
        )
        raw_instrument = np.broadcast_to(instrument, observations.shape)
        raw_instrument_variance = raw_instrument**2
        raw_representation_variance = np.broadcast_to(
            representation**2, observations.shape
        )
        if aggregate_observations is None:
            bins = tuple(
                ObservationBin(float(t), (i,), (1.0,)) for i, t in enumerate(raw_times)
            )
        else:
            if aggregate_observations.mode != "mean":
                raise ValueError(
                    "Corrected observation error supports only mean aggregation; "
                    "median, min and max require calibrated product likelihoods."
                )
            bins = aggregate_observations.bins(
                observations, allow_interval_count_change=True
            )
        if self.aggregation == "none":
            # The configured error applies to each product, irrespective of
            # its frame count. Raw synthetic measurement noise is unchanged.
            instrument_variance = np.broadcast_to(
                instrument**2, (len(bins), len(instrument))
            )
            representation_variance = np.broadcast_to(
                representation**2, instrument_variance.shape
            )
        else:
            instrument_variance = np.stack(
                [
                    np.sum(
                        raw_instrument_variance[list(b.frame_ids)]
                        * np.square(b.weights)[:, None],
                        axis=0,
                    )
                    for b in bins
                ]
            )
            representation_variance = np.stack(
                [
                    np.sum(
                        raw_representation_variance[list(b.frame_ids)]
                        * np.square(b.weights)[:, None],
                        axis=0,
                    )
                    for b in bins
                ]
            )
        variance = instrument_variance + representation_variance
        if not np.all(np.isfinite(variance)) or np.any(variance <= 0):
            raise ValueError("Total observation variances must be finite and positive.")
        return ResolvedObservationError(
            raw_instrument_std=_readonly(raw_instrument),
            raw_instrument_variance=_readonly(raw_instrument_variance),
            raw_representation_variance=_readonly(raw_representation_variance),
            instrument_variance=_readonly(instrument_variance),
            representation_variance=_readonly(representation_variance),
            variance=_readonly(variance),
            raw_times=_readonly(raw_times),
            times=_readonly([b.start_time for b in bins]),
            components=components,
            sensor_indices=sensors,
            frame_ids=tuple(b.frame_ids for b in bins),
            weights=tuple(b.weights for b in bins),
            provenance=f"observation_error.v1:diagonal:independent:{self.aggregation}",
        )
