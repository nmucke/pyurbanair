"""Map video frames to simulation time.

The render timeline is uniform in video frames: frame ``i`` shows sim time
``t_start + i * playback_speed / fps``. ``playback_speed`` is sim seconds per
video second (20 => a 1000 s simulation plays in 50 s). The first
``preroll`` sim seconds before ``t_start`` are used by particle layers to
fill the domain, so the first rendered frame is never empty.
"""

from __future__ import annotations

import dataclasses
from typing import Any, Optional

import numpy as np


@dataclasses.dataclass(frozen=True)
class Timeline:
    fps: float
    playback_speed: float
    t_start: float
    n_frames: int

    @property
    def dt(self) -> float:
        """Sim seconds between consecutive video frames."""
        return self.playback_speed / self.fps

    @property
    def frame_times(self) -> np.ndarray:
        return self.t_start + self.dt * np.arange(self.n_frames)

    @property
    def t_end(self) -> float:
        return float(self.frame_times[-1])

    def to_dict(self) -> dict[str, Any]:
        return {
            "fps": self.fps,
            "playback_speed": self.playback_speed,
            "t_start": self.t_start,
            "t_end": self.t_end,
            "dt": self.dt,
            "n_frames": self.n_frames,
            "frame_times": [round(float(t), 4) for t in self.frame_times],
        }


def make_timeline(
    sim_times: np.ndarray,
    fps: float = 30.0,
    playback_speed: float = 20.0,
    t_start: Optional[float] = None,
    t_end: Optional[float] = None,
    duration: Optional[float] = None,
) -> Timeline:
    """Build a timeline clipped to the stored snapshots.

    ``duration`` (video seconds) wins over ``t_end`` when both are given.
    """
    t0 = (
        float(sim_times[0])
        if t_start is None
        else max(float(t_start), float(sim_times[0]))
    )
    t1 = (
        float(sim_times[-1])
        if t_end is None
        else min(float(t_end), float(sim_times[-1]))
    )
    if duration is not None:
        t1 = min(t0 + duration * playback_speed, float(sim_times[-1]))
    dt = playback_speed / fps
    n = int(np.floor((t1 - t0) / dt + 1e-9)) + 1
    if n < 2:
        raise ValueError(
            f"time window [{t0}, {t1}] too short for fps={fps}, speed={playback_speed}"
        )
    return Timeline(float(fps), float(playback_speed), t0, n)
