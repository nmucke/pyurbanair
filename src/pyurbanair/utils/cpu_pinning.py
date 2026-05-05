"""CPU pinning for ensemble parallel execution.

Spreads ensemble workers across distinct physical cores (deduping SMT
siblings) and across L3-cache domains where possible, so that
concurrent members do not time-share an SMT pair or evict each
other's working set from a shared L3.

Disabled by setting ``PYURBANAIR_DISABLE_CPU_PINNING=1`` in the
environment, e.g. on shared clusters where the resource manager
already controls affinity.
"""

from __future__ import annotations

import logging
import multiprocessing as mp
import os
from typing import Iterable

logger = logging.getLogger(__name__)

_DISABLE_ENV = "PYURBANAIR_DISABLE_CPU_PINNING"


def cpu_pinning_disabled() -> bool:
    """True if the env var disables pinning, or sched_setaffinity is unavailable."""
    if not hasattr(os, "sched_setaffinity") or not hasattr(os, "sched_getaffinity"):
        return True
    raw = os.environ.get(_DISABLE_ENV, "").strip().lower()
    return raw not in ("", "0", "false", "no", "off")


def _parse_cpu_list(s: str) -> list[int]:
    out: list[int] = []
    for piece in s.split(","):
        piece = piece.strip()
        if not piece:
            continue
        if "-" in piece:
            lo_s, hi_s = piece.split("-", 1)
            out.extend(range(int(lo_s), int(hi_s) + 1))
        else:
            out.append(int(piece))
    return out


def _read_topology_list(cpu: int, attr: str) -> list[int]:
    path = f"/sys/devices/system/cpu/cpu{cpu}/{attr}"
    try:
        with open(path) as fh:
            return _parse_cpu_list(fh.read().strip())
    except (FileNotFoundError, OSError, PermissionError):
        return [cpu]


def _list_physical_cores(available: Iterable[int]) -> list[int]:
    """One logical CPU per physical core, deduped via thread_siblings_list."""
    available_set = set(available)
    seen: set[int] = set()
    out: list[int] = []
    for cpu in sorted(available_set):
        siblings = _read_topology_list(cpu, "topology/thread_siblings_list")
        canonical = min(siblings)
        if canonical in seen:
            continue
        seen.add(canonical)
        # Prefer the canonical (lowest) sibling that is in the available set.
        if canonical in available_set:
            out.append(canonical)
        else:
            out.append(cpu)
    return out


def _physical_cores_by_l3(available: Iterable[int]) -> list[list[int]]:
    """Group physical cores by shared L3 cache, ordered by L3 set."""
    cores = _list_physical_cores(available)
    groups: dict[tuple[int, ...], list[int]] = {}
    for cpu in cores:
        l3_set = tuple(_read_topology_list(cpu, "cache/index3/shared_cpu_list"))
        groups.setdefault(l3_set, []).append(cpu)
    return [groups[k] for k in sorted(groups.keys())]


def round_robin_cpu_ids(num_slots: int) -> list[int]:
    """Assign physical CPU ids to ``num_slots`` workers, spreading across L3 groups.

    Groups are filled round-robin: with 4 L3 groups of 4 cores each,
    the first 4 ids come from 4 distinct groups, then the next 4 wrap
    back through the same groups (sharing L3 with one peer), etc.
    """
    if not hasattr(os, "sched_getaffinity"):
        return []
    available = os.sched_getaffinity(0)
    if not available:
        return []
    groups = [list(g) for g in _physical_cores_by_l3(available)]
    out: list[int] = []
    while len(out) < num_slots:
        progress = False
        for grp in groups:
            if grp:
                out.append(grp.pop(0))
                progress = True
                if len(out) == num_slots:
                    return out
        if not progress:
            break
    return out


def build_cpu_queue(num_workers: int, cpus_per_worker: int) -> "mp.Queue":
    """Build a Queue holding one frozenset of CPU ids per worker."""
    ctx = mp.get_context("fork")
    queue: "mp.Queue" = ctx.Queue()
    pool = round_robin_cpu_ids(num_workers * cpus_per_worker)
    if not pool:
        logger.info("CPU pinning skipped: no available CPUs detected.")
        return queue
    if len(pool) < num_workers * cpus_per_worker:
        logger.warning(
            "Asked for %d cpu slots (workers=%d × cpus_per_worker=%d) but only "
            "%d distinct physical cores available; some workers may share cores.",
            num_workers * cpus_per_worker,
            num_workers,
            cpus_per_worker,
            len(pool),
        )
    for i in range(num_workers):
        slice_ = pool[i * cpus_per_worker : (i + 1) * cpus_per_worker]
        if not slice_:
            break
        queue.put(frozenset(slice_))
    return queue


def pin_worker_initializer(cpu_queue: "mp.Queue") -> None:
    """ProcessPoolExecutor initializer: pin this worker to a CPU set.

    Each worker pulls a CPU set off the queue once at startup and
    calls ``os.sched_setaffinity``. Subprocesses (bash, mpiexec,
    uDALES) inherit this affinity, which keeps OpenMPI from spreading
    the rank across cores it doesn't own.
    """
    if cpu_pinning_disabled():
        return
    try:
        cpu_set = cpu_queue.get(timeout=1.0)
    except Exception:
        logger.debug("CPU pinning queue empty; worker pid=%d not pinned.", os.getpid())
        return
    try:
        os.sched_setaffinity(0, set(cpu_set))
        logger.info(
            "Worker pid=%d pinned to cpus=%s",
            os.getpid(),
            sorted(cpu_set),
        )
    except OSError as exc:
        logger.warning("Failed to set affinity for pid=%d: %s", os.getpid(), exc)
