"""Flow statistics over ensemble state fields.

Time-mean velocities, resolved second moments (TKE, ``<u'w'>``), block
bootstrap of the truth's own sampling uncertainty, and -- from phase 3 --
Welch frequency spectra and the log-spectral distance.

Everything here streams. Full-ensemble window state files are ``(ensemble,
time, z, y, x)`` and run to gigabytes, so accumulation is member-at-a-time or
z-slab-wise with at most ~2 reader threads (the DRAM-bandwidth plateau on this
hardware); nothing in this module may ``.load()`` a whole window file. The
streaming moment accumulator is the one class the library allows, because it
is genuinely stateful.

Populated in WP0.2 (move) and extended in WP1.4 / phase 3.
"""
