"""Shared per-frame dispatch helpers for the volumes/isosurfaces/slices exporters.

Frames are independent, so each of those exporters farms a list of per-file
jobs out over ``spec["workers"]``. This private module factors out the
dispatch logic (:func:`map_frames`) and the frame/time enumeration
(:func:`file_frames`) that used to be copy-pasted across ``volumes.py``,
``isosurfaces.py`` and ``slices.py`` -- both now call into here instead, with
no change in behaviour.

Parallelism (:func:`map_frames`)
---------------------------------
When the input dataset was opened from a file on disk (the normal case --
``fields.ds.encoding["source"]``), workers are OS processes started with the
``forkserver`` multiprocessing context (never ``fork``: this process may
already have loaded CUDA/JAX elsewhere in a longer pipeline, and fork after
that deadlocks -- see repo memory). Each worker process re-opens the state
file once via a pool initializer and reuses it for all frames it handles, so
the (non-trivially-picklable, ``functools.lru_cache``-wrapped) ``FieldSeries``
object itself never needs to cross a process boundary. If the dataset has no
on-disk source (e.g. an in-memory ``xr.Dataset`` built by a test), the same
worker function runs in a thread pool instead -- frames still overlap, just
without process isolation. ``workers <= 1`` (or a single job) skips pooling
entirely and runs serially in this process.

Each caller module keeps its own module-level ``_FIELDS`` global (read by its
``_render_*_file`` worker function) rather than sharing one here, because the
worker functions must stay plain, picklable, module-level callables for the
process-pool path. ``set_fields`` is that module's small setter for its own
global (e.g. ``def _set_fields(f): global _FIELDS; _FIELDS = f``) -- passed
in by reference, so it pickles fine and runs correctly whichever process
calls it.
"""

from __future__ import annotations

import concurrent.futures as cf
import multiprocessing as mp
from typing import Any, Callable, Optional, TypeVar

from .fields import FieldSeries
from .timeline import Timeline

T = TypeVar("T")


# Sentinel-free null handling: a spec value explicitly set to ``None`` (a
# preset's or render.yaml's ``null`` / Hydra ``~``) means "use the default",
# same convention as particles.py -- not "pass None to the int/float/str
# caster at the call site", which raises TypeError.
def opt(spec: dict[str, Any], key: str, default: Any = None) -> Any:
    """``spec.get(key, default)`` that also treats a stored ``None`` as missing."""
    value = spec.get(key, default)
    return default if value is None else value


def file_frames(timeline: Timeline, frame_step: int) -> list[tuple[int, int, float]]:
    """(file_index, video_frame, sim_time) for every exported file."""
    times = timeline.frame_times
    video_frames = list(range(0, timeline.n_frames, frame_step))
    return [(f, vf, float(times[vf])) for f, vf in enumerate(video_frames)]


def _init_pool(
    set_fields: Callable[[FieldSeries], None],
    source: str,
    numba_threads: Optional[int],
) -> None:
    """``ProcessPoolExecutor`` initializer: reopen the dataset in this worker
    process and install it via the caller's own ``set_fields``."""
    from .fields import open_fields

    if numba_threads is not None:
        import numba

        # Each worker may run a parallel numba kernel (slices' LIC); split
        # the cores between them instead of each grabbing every core.
        numba.set_num_threads(
            max(1, min(numba_threads, numba.config.NUMBA_NUM_THREADS))
        )

    set_fields(open_fields(source))


def map_frames(
    fields: FieldSeries,
    jobs: list[dict[str, Any]],
    worker_fn: Callable[[dict[str, Any]], T],
    workers: int,
    set_fields: Callable[[FieldSeries], None],
    numba_threads: Optional[int] = None,
) -> list[T]:
    """Run ``worker_fn`` over ``jobs``, in parallel when it's worth it.

    ``set_fields`` installs ``fields`` (or, in a forkserver worker, a fresh
    ``FieldSeries`` reopened from disk) into the caller module's own
    ``_FIELDS`` global, which ``worker_fn`` reads. ``numba_threads``, when
    given, is the per-worker numba thread count for the process-pool path
    only (slices' LIC kernel); the serial and thread-pool paths run in this
    process and never touch it.
    """
    if workers <= 1 or len(jobs) <= 1:
        set_fields(fields)
        return [worker_fn(j) for j in jobs]

    source = None
    encoding = getattr(fields.ds, "encoding", None)
    if encoding:
        source = encoding.get("source")

    if source:
        ctx = mp.get_context("forkserver")
        with cf.ProcessPoolExecutor(
            max_workers=workers,
            mp_context=ctx,
            initializer=_init_pool,
            initargs=(set_fields, source, numba_threads),
        ) as ex:
            return list(ex.map(worker_fn, jobs))

    set_fields(fields)
    with cf.ThreadPoolExecutor(max_workers=workers) as ex:
        return list(ex.map(worker_fn, jobs))


__all__ = ["opt", "file_frames", "map_frames"]
