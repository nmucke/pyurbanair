"""Derive PALM's processor topology (npex/npey) from a single NCPU value.

PALM splits the horizontal domain into an ``npex`` x ``npey`` grid of MPI
subdomains. With inflow/outflow lateral BCs pypalm selects the multigrid
pressure solver, which requires *uniform* subdomains, i.e. the number of grid
points must divide evenly by the processor count in each direction.

To let callers pick only NCPU (as in pyudales' ``validate_and_sync_ncpu``), we
use a slab decomposition: ``npex = ncpu``, ``npey = 1``. The undivided y
direction then carries no divisibility constraint, so the only requirement is
that the number of x grid points (``nx + 1``) be divisible by ``ncpu``.
"""

from __future__ import annotations


def _divisors(n: int) -> list[int]:
    return [d for d in range(1, n + 1) if n % d == 0]


def derive_npex_npey(ncpu: int, nx_points: int) -> tuple[int, int]:
    """Return ``(npex, npey)`` for a slab decomposition of ``ncpu`` PEs.

    Mirrors pyudales: all PEs go in x (``npex = ncpu``, ``npey = 1``). For the
    multigrid solver to accept uniform subdomains, ``nx_points`` (PALM's
    ``nx + 1``) must be divisible by ``ncpu``.

    Args:
        ncpu: Total MPI ranks (must equal ``npex * npey``).
        nx_points: Number of grid points in x (PALM ``nx + 1``).

    Raises:
        ValueError: If ``nx_points`` is not divisible by ``ncpu``; the message
            lists the NCPU values that do divide it evenly.
    """
    if ncpu < 1:
        raise ValueError(f"ncpu must be >= 1, got {ncpu}")
    if nx_points % ncpu != 0:
        valid = [d for d in _divisors(nx_points) if d <= nx_points]
        raise ValueError(
            f"ncpu ({ncpu}) must divide the number of x grid points "
            f"(nx+1 = {nx_points}) evenly for the multigrid solver's uniform "
            f"subdomains. Valid NCPU values for this grid: {valid}."
        )
    return ncpu, 1


def multigrid_levels(nx: int, ny: int, nz: int, npex: int = 1) -> int:
    """Number of grid levels PALM's multigrid solver can coarsen to.

    PALM halves the grid until any direction becomes odd, so the usable level
    count is set by how many times 2 divides *every* direction — including the
    per-rank subdomain in x, since coarsening happens on the decomposed grid.
    One level means no coarsening at all: the solver degenerates to a handful
    of Gauss-Seidel sweeps (PALM warns PAC0242) and never properly solves the
    perturbation pressure, so the velocity field stays divergent and the run
    blows up.

    ``nx``/``ny``/``nz`` are POINT counts (pyurbanair's ``domain.*``), not
    PALM's ``nx = points - 1`` namelist convention.
    """

    def _pow2(n: int) -> int:
        levels = 0
        while n > 1 and n % 2 == 0:
            n //= 2
            levels += 1
        return levels

    if npex > 0 and nx % npex == 0:
        nx_sub = nx // npex
    else:
        nx_sub = nx
    return 1 + min(_pow2(nx_sub), _pow2(ny), _pow2(nz))
