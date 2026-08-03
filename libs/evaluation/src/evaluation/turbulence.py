"""Flow statistics over ensemble state fields.

Time-mean velocities, resolved second moments, block bootstrap, and (phase 3)
Welch spectra with the log-spectral distance.

Everything streams: window state files are ``(ensemble, time, z, y, x)`` and
run to gigabytes, so nothing here may ``.load()`` one whole. The streaming
moment accumulator is the one class the library allows.

Populated in WP0.2 (move), extended in WP1.4 and phase 3.
"""
